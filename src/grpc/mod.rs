pub mod compat;
pub mod inference;

use std::ops::Deref;
use std::sync::Arc;
use std::vec;

use hashbrown::HashMap;

use arc_swap::ArcSwap;
use inference::GrpcInferenceService;
use inference::*;
use ort::value::Shape;
use pajamax::status::{Code, Status};
use smallvec::SmallVec;
use tracing::{debug, info};

use crate::grpc::inference::model_infer_response::InferOutputTensor;
use crate::metrics::with_local_metrics;
use crate::scheduler::ModelProxy;
use crate::tensor::batched_tensor::TensorBytes;
use crate::tensor::supertensor::InferError;
use crate::tracing::ClientTrace;

fn infer_error_to_status(err: &InferError) -> Status {
    let code = match err {
        InferError::RingFull => Code::ResourceExhausted,
        _ => Code::Internal,
    };
    Status {
        code,
        message: err.message().into_owned(),
    }
}

pub struct TritonService {
    pub loaded_models: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>,
}

impl TritonService {
    pub fn new(model_map: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>) -> Self {
        TritonService {
            loaded_models: model_map,
        }
    }
}

unsafe impl Send for ModelInferResponse {}

#[async_trait::async_trait(?Send)]
impl GrpcInferenceService for TritonService {
    async fn server_live(&self, _req: ServerLiveRequest) -> Result<ServerLiveResponse, Status> {
        debug!("is live?");
        Ok(ServerLiveResponse { live: true })
    }

    async fn server_ready(&self, _req: ServerReadyRequest) -> Result<ServerReadyResponse, Status> {
        debug!("is server ready?");
        Ok(ServerReadyResponse { ready: true })
    }

    async fn model_ready(&self, request: ModelReadyRequest) -> Result<ModelReadyResponse, Status> {
        debug!("is model ready?");
        let _ = request.name.clone();
        let is_ready = false;
        Ok(ModelReadyResponse { ready: is_ready })
    }

    async fn server_metadata(
        &self,
        _req: ServerMetadataRequest,
    ) -> Result<ServerMetadataResponse, Status> {
        debug!("server metadata");
        let reply = ServerMetadataResponse {
            name: "inference-server".to_string(),
            version: "1.0.0-demo".to_string(),
            extensions: vec!["classification".to_string(), "model_repository".to_string()],
        };
        Ok(reply)
    }

    async fn model_metadata(
        &self,
        request: ModelMetadataRequest,
    ) -> Result<ModelMetadataResponse, Status> {
        let model_name = &request.name;
        info!(model_name, "Getting model metadata");
        match self.loaded_models.load().get(model_name) {
            Some(proxy) => {
                debug!(model_name, "Got model, responding");
                Ok(ModelMetadataResponse {
                    name: model_name.clone(),
                    versions: vec![String::from("1")],
                    platform: String::from("onnxruntime_onnx"),
                    inputs: proxy.model_metadata.input_meta.clone(),
                    outputs: proxy.model_metadata.output_meta.clone(),
                })
            }
            None => Err(Status {
                code: Code::NotFound,
                message: format!("Model {} not found", model_name),
            }),
        }
    }

    async fn model_infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, Status> {
        let mut trace = ClientTrace::start();
        let start = std::time::Instant::now();
        let model_name = request.model_name.clone();
        let client_batch_size = request
            .inputs
            .first()
            .and_then(|input| input.shape.first())
            .copied();
        let models = self.loaded_models.load();

        let proxy: &ModelProxy = match models.get(&model_name) {
            Some(proxy) => proxy,
            None => {
                with_local_metrics(|m| m.inc_requests_not_found(&model_name));
                return Err(Status {
                    code: Code::NotFound,
                    message: format!("Model {} not found", &model_name),
                });
            }
        };
        trace.record_model_proxy_aquired();

        // Validate input shapes against model config (except dim 0 which is batch)
        for req_input in &request.inputs {
            let model_input = proxy
                .model_config
                .input
                .iter()
                .find(|i| i.name == req_input.name)
                .ok_or_else(|| Status {
                    code: Code::InvalidArgument,
                    message: format!("Unknown input '{}' in request", req_input.name),
                })?;

            // Validate shape dimensions (skip dim 0 which is batch size)
            let req_shape: Vec<i64> = req_input.shape.clone().into_iter().collect();
            let model_shape: Vec<i64> = model_input.dims.clone();

            if req_shape.len() != model_shape.len() {
                return Err(Status {
                    code: Code::InvalidArgument,
                    message: format!(
                        "Input '{}' shape rank mismatch: expected {}, got {}",
                        req_input.name,
                        model_shape.len(),
                        req_shape.len()
                    ),
                });
            }

            for (i, (req_dim, model_dim)) in req_shape.iter().zip(model_shape.iter()).enumerate() {
                if i == 0 {
                    // Dim 0 (batch) is always validated - just check it's positive
                    if *req_dim <= 0 {
                        return Err(Status {
                            code: Code::InvalidArgument,
                            message: format!(
                                "Input '{}' batch size must be positive, got {}",
                                req_input.name, req_dim
                            ),
                        });
                    }
                } else if *req_dim != *model_dim {
                    return Err(Status {
                        code: Code::InvalidArgument,
                        message: format!(
                            "Input '{}' dimension {} mismatch: expected {}, got {}",
                            req_input.name, i, model_dim, req_dim
                        ),
                    });
                }
            }
        }

        let mut ordered_inputs = request.inputs.iter().enumerate().collect::<Vec<_>>();

        ordered_inputs.sort_by_key(|(_, req_input)| {
            proxy
                .model_metadata
                .input_set
                .get(&req_input.name)
                .expect("invariant: input validated against model_metadata.input_set")
                .order
        });
        let mut inputs: SmallVec<[TensorBytes; 6]> = SmallVec::with_capacity(ordered_inputs.len());

        for (i, req_input) in ordered_inputs {
            let dimensions = Shape::new(req_input.shape.clone().into_iter());
            let tensor = TensorBytes {
                data_type: DataType::from_str(&req_input.datatype),
                shape: dimensions,
                data: &request.raw_input_contents[i],
            };

            inputs.push(tensor);
        }

        trace.record_serialization_done();
        let inference_outputs = match proxy.data.infer(&inputs, &mut trace).await {
            Ok(resp) => resp,
            Err(e) => {
                with_local_metrics(|m| match &e {
                    InferError::RingFull => m.inc_requests_resource_exhausted(&model_name),
                    _ => m.inc_requests_error(&model_name),
                });
                return Err(infer_error_to_status(&e));
            }
        };
        let mut raw_output: Vec<Vec<u8>> = Vec::new();

        let output_metadata = match inference_outputs.get_data(&mut raw_output).await {
            Ok(meta) => meta,
            Err(e) => {
                with_local_metrics(|m| match &e {
                    InferError::RingFull => m.inc_requests_resource_exhausted(&model_name),
                    _ => m.inc_requests_error(&model_name),
                });
                return Err(infer_error_to_status(&e));
            }
        };
        let outputs = output_metadata.0
            .iter()
            .zip(proxy.model_metadata.output_meta.iter())
            .map(|((data_type, shape), meta)| InferOutputTensor {
                name: meta.name.clone(),
                datatype: DataType::from(*data_type).as_str_name().to_string(),
                shape: Vec::from(shape.deref()),
                parameters: std::collections::HashMap::new(),
                contents: None,
            })
            .collect();
        trace.record_execution_trace(output_metadata.1);
        trace.record_output_processed();

        with_local_metrics(|m| {
            m.inc_requests_ok(&model_name);
            m.observe_request_duration(&model_name, start.elapsed().as_secs_f64());
            if let Some(batch_size) = client_batch_size {
                m.observe_client_batch_size(&model_name, batch_size as f64);
            }
            trace.record_metrics(model_name.as_str(), m);
        });

        Ok(ModelInferResponse {
            model_name: proxy.model_config.name.clone(),
            model_version: String::from("1"),
            id: request.id.clone(),
            parameters: std::collections::HashMap::with_capacity(0),
            outputs,
            raw_output_contents: raw_output,
        })
    }

    async fn model_config(
        &self,
        request: ModelConfigRequest,
    ) -> Result<ModelConfigResponse, Status> {
        let model_name = &request.name;
        info!(model_name, "Getting model config");
        match self.loaded_models.load().get(model_name) {
            Some(proxy) => Ok(ModelConfigResponse {
                config: Some(proxy.model_config.clone()),
            }),
            None => Err(Status {
                code: Code::NotFound,
                message: format!("Model {} not found", model_name),
            }),
        }
    }

    async fn model_statistics(
        &self,
        _request: ModelStatisticsRequest,
    ) -> Result<ModelStatisticsResponse, Status> {
        debug!("model statistics");
        todo!()
    }

    async fn repository_index(
        &self,
        _request: RepositoryIndexRequest,
    ) -> Result<RepositoryIndexResponse, Status> {
        debug!("repository index");
        Ok(RepositoryIndexResponse { models: vec![] })
    }

    async fn repository_model_load(
        &self,
        _request: RepositoryModelLoadRequest,
    ) -> Result<RepositoryModelLoadResponse, Status> {
        debug!("repository model load");
        Ok(RepositoryModelLoadResponse {})
    }

    async fn repository_model_unload(
        &self,
        _request: RepositoryModelUnloadRequest,
    ) -> Result<RepositoryModelUnloadResponse, Status> {
        debug!("repository model unload");
        Ok(RepositoryModelUnloadResponse {})
    }

    async fn system_shared_memory_status(
        &self,
        _request: SystemSharedMemoryStatusRequest,
    ) -> Result<SystemSharedMemoryStatusResponse, Status> {
        debug!("system shared memory status");
        todo!()
    }

    async fn system_shared_memory_register(
        &self,
        _request: SystemSharedMemoryRegisterRequest,
    ) -> Result<SystemSharedMemoryRegisterResponse, Status> {
        todo!()
    }

    async fn system_shared_memory_unregister(
        &self,
        _request: SystemSharedMemoryUnregisterRequest,
    ) -> Result<SystemSharedMemoryUnregisterResponse, Status> {
        todo!()
    }

    async fn cuda_shared_memory_status(
        &self,
        _request: CudaSharedMemoryStatusRequest,
    ) -> Result<CudaSharedMemoryStatusResponse, Status> {
        todo!()
    }

    async fn cuda_shared_memory_register(
        &self,
        _request: CudaSharedMemoryRegisterRequest,
    ) -> Result<CudaSharedMemoryRegisterResponse, Status> {
        Ok(CudaSharedMemoryRegisterResponse {})
    }

    async fn cuda_shared_memory_unregister(
        &self,
        _request: CudaSharedMemoryUnregisterRequest,
    ) -> Result<CudaSharedMemoryUnregisterResponse, Status> {
        Ok(CudaSharedMemoryUnregisterResponse {})
    }

    async fn trace_setting(
        &self,
        _request: TraceSettingRequest,
    ) -> Result<TraceSettingResponse, Status> {
        todo!()
    }

    async fn log_settings(
        &self,
        _request: LogSettingsRequest,
    ) -> Result<LogSettingsResponse, Status> {
        todo!()
    }
}
