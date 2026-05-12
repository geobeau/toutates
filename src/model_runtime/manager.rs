use hashbrown::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use smallvec::SmallVec;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::grpc::inference::{
    model_metadata_response::TensorMetadata, DataType, ModelConfig, ModelInput, ModelOutput,
};
use crate::tensor::supertensor::SessionValues;
use crate::metrics::MetricsRegistry;
use crate::model_repository::config::{AllocatorKind, ModelRepositoryConfig};
use crate::scheduler::{ModelInputMetadata, ModelMetadata, ModelProxy};
use crate::tensor::supertensor::SuperTensorBuffer;

use tracing::info;

use super::session_starter::SessionStartRequest;

pub struct LoadModelRequest {
    pub model_name: String,
    pub version: u64,
    pub model_path: PathBuf,
    pub config: ModelRepositoryConfig,
    pub reply: oneshot::Sender<Result<(), LoadError>>,
}

#[derive(Debug)]
pub enum LoadError {
    SessionBuild(String),
    AllocatorCreate(String),
    SuperTensorBuild,
    DispatchFailed(String),
}

pub struct ModelRuntimeManager {
    receiver: mpsc::Receiver<LoadModelRequest>,
    starters: Vec<mpsc::Sender<SessionStartRequest>>,
    loaded_models: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>,
    metrics: Arc<MetricsRegistry>,
    round_robin: AtomicUsize,
    custom_op_libraries: Vec<PathBuf>,
}

impl ModelRuntimeManager {
    pub fn new(
        receiver: mpsc::Receiver<LoadModelRequest>,
        starters: Vec<mpsc::Sender<SessionStartRequest>>,
        loaded_models: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>,
        metrics: Arc<MetricsRegistry>,
        custom_op_libraries: Vec<PathBuf>,
    ) -> Self {
        Self {
            receiver,
            starters,
            loaded_models,
            metrics,
            round_robin: AtomicUsize::new(0),
            custom_op_libraries,
        }
    }

    pub async fn run(mut self) {
        while let Some(req) = self.receiver.recv().await {
            let result = self
                .load_model(req.model_name, req.model_path, &req.config)
                .await;
            let _ = req.reply.send(result);
        }
    }

    fn spawn_supertensor_metrics_reporter(
        metrics: Arc<MetricsRegistry>,
        model_name: String,
        model_proxy: Arc<ModelProxy>,
    ) {
        compio::runtime::spawn(async move {
            let executors_in_use = metrics
                .inference_executors_in_use
                .with_label_values(&[model_name.as_str()]);
            let ring_tail_index = metrics
                .inference_ring_tail_index
                .with_label_values(&[model_name.as_str()]);
            let ring_in_use_index = metrics
                .inference_ring_in_use_index
                .with_label_values(&[model_name.as_str()]);
            let ring_head_index = metrics
                .inference_ring_head_index
                .with_label_values(&[model_name.as_str()]);
            let to_i64 = |value: usize| i64::try_from(value).unwrap_or(i64::MAX);

            loop {
                let snapshot = model_proxy.data.metrics_snapshot();
                executors_in_use.set(to_i64(snapshot.executors_in_use));
                ring_tail_index.set(to_i64(snapshot.tail_index));
                ring_in_use_index.set(to_i64(snapshot.in_use_index));
                ring_head_index.set(to_i64(snapshot.head_index));

                compio::time::sleep(Duration::from_secs(1)).await;
            }
        })
        .detach();
    }

    async fn load_model(
        &self,
        model_name: String,
        model_path: PathBuf,
        config: &ModelRepositoryConfig,
    ) -> Result<(), LoadError> {
        info!(%model_name, ?model_path, "Loading model");
        let num_executors = config.num_executors;
        let batch_size = config.batch_size;
        let capacity = config.capacity;

        // Build N sessions
        let mut sessions = Vec::with_capacity(num_executors);
        for _ in 0..num_executors {
            let mut builder = Session::builder()
                .map_err(|e| LoadError::SessionBuild(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| LoadError::SessionBuild(e.to_string()))?;

            if let Some(profiling) = &config.profiling {
                builder = builder
                    .with_profiling(&profiling.file_prefix)
                    .map_err(|e| LoadError::SessionBuild(e.to_string()))?;
            }

            for lib_path in &self.custom_op_libraries {
                info!(?lib_path, "Registering custom op library");
                builder = builder
                    .with_operator_library(lib_path)
                    .map_err(|e| LoadError::SessionBuild(e.to_string()))?;
            }

            let session = builder
                .commit_from_file(&model_path)
                .map_err(|e| LoadError::SessionBuild(e.to_string()))?;
            sessions.push(session);
        }

        // Build SuperTensorBuffer and metadata from the first session
        let first_session = &sessions[0];

        let alloc_device = match config.allocator {
            AllocatorKind::Cpu => AllocationDevice::CPU,
            AllocatorKind::CudaPinned => AllocationDevice::CUDA_PINNED,
        };
        let memory_type = match config.allocator {
            AllocatorKind::Cpu => MemoryType::CPUInput,
            AllocatorKind::CudaPinned => MemoryType::CPUInput,
        };

        let allocator = Allocator::new(
            first_session,
            MemoryInfo::new(alloc_device, 0, AllocatorType::Device, memory_type)
                .map_err(|e| LoadError::AllocatorCreate(e.to_string()))?,
        )
        .map_err(|e| LoadError::AllocatorCreate(e.to_string()))?;

        let inputs_ref: Vec<_> = first_session.inputs().iter().collect();
        let super_tensor_buffer = SuperTensorBuffer::new_with_overrides(
            capacity,
            batch_size,
            &inputs_ref,
            &allocator,
            &config.input_shapes,
        )
        .map_err(|_| LoadError::SuperTensorBuild)?;

        let inputs = first_session
            .inputs()
            .iter()
            .map(|input| {
                let tensor_type: &DataType = &input.dtype().tensor_type().unwrap().into();

                // Get the base shape from the model
                let mut tensor_shape: Vec<i64> = input
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();

                // Apply shape override if specified for this input
                if let Some(override_shape) = config.input_shapes.get(input.name()) {
                    // Merge override shape with model shape
                    // Override shape has same rank as model shape
                    // Dim 0 is -1 (batch) in both, will be replaced at request time
                    for (i, override_dim) in override_shape.iter().enumerate() {
                        if i < tensor_shape.len() {
                            tensor_shape[i] = *override_dim;
                        }
                    }
                }

                ModelInput {
                    name: input.name().to_string(),
                    data_type: (*tensor_type).into(),
                    format: 0,
                    dims: tensor_shape,
                    reshape: None,
                    is_shape_tensor: false,
                    allow_ragged_batch: false,
                    optional: false,
                    is_non_linear_format_io: false,
                }
            })
            .collect();

        let outputs = first_session
            .outputs()
            .iter()
            .map(|output| {
                let tensor_type: &DataType = &output.dtype().tensor_type().unwrap().into();

                // Get the base shape from the model
                let mut tensor_shape: Vec<i64> = output
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();

                // Apply shape override if specified for this output
                if let Some(override_shape) = config.output_shapes.get(output.name()) {
                    // Merge override shape with model shape
                    for (i, override_dim) in override_shape.iter().enumerate() {
                        if i < tensor_shape.len() {
                            tensor_shape[i] = *override_dim;
                        }
                    }
                }

                ModelOutput {
                    name: output.name().to_string(),
                    data_type: (*tensor_type).into(),
                    dims: tensor_shape,
                    reshape: None,
                    is_shape_tensor: false,
                    is_non_linear_format_io: false,
                    label_filename: String::from(""),
                }
            })
            .collect();

        let mut input_set = HashMap::new();
        let inputs_metadata = first_session
            .inputs()
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let tensor_type: &DataType = &input.dtype().tensor_type().unwrap().into();

                // Get the base shape from the model
                let mut tensor_shape: Vec<i64> = input
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();

                // Apply shape override if specified for this input
                if let Some(override_shape) = config.input_shapes.get(input.name()) {
                    // Merge override shape with model shape
                    for (i, override_dim) in override_shape.iter().enumerate() {
                        if i < tensor_shape.len() {
                            tensor_shape[i] = *override_dim;
                        }
                    }
                }

                let mut input_shape = tensor_shape.clone();
                input_shape[0] = 1;
                let input_metadata = ModelInputMetadata {
                    shape: ort::value::Shape::from(input_shape),
                    order: i,
                };
                input_set.insert(input.name().to_string(), input_metadata);
                TensorMetadata {
                    name: input.name().to_string(),
                    datatype: tensor_type.to_metadata_string(),
                    shape: tensor_shape,
                }
            })
            .collect();

        let outputs_metadata = first_session
            .outputs()
            .iter()
            .map(|output| {
                let tensor_type: &DataType = &output.dtype().tensor_type().unwrap().into();

                // Get the base shape from the model
                let mut tensor_shape: Vec<i64> = output
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();

                // Apply shape override if specified for this output
                if let Some(override_shape) = config.output_shapes.get(output.name()) {
                    // Merge override shape with model shape
                    for (i, override_dim) in override_shape.iter().enumerate() {
                        if i < tensor_shape.len() {
                            tensor_shape[i] = *override_dim;
                        }
                    }
                }

                TensorMetadata {
                    name: output.name().to_string(),
                    datatype: tensor_type.to_metadata_string(),
                    shape: tensor_shape,
                }
            })
            .collect();

        let model_metadata = ModelMetadata {
            input_meta: inputs_metadata,
            output_meta: outputs_metadata,
            input_set,
        };

        let model_config = ModelConfig {
            name: model_name.clone(),
            platform: String::from("onnxruntime_onnx"),
            backend: String::from("onnxruntime"),
            runtime: String::from("onnxruntime"),
            version_policy: None,
            max_batch_size: batch_size as i32,
            input: inputs,
            output: outputs,
            batch_input: vec![],
            batch_output: vec![],
            optimization: None,
            instance_group: vec![],
            default_model_filename: String::from("todo"),
            cc_model_filenames: std::collections::HashMap::new(),
            metric_tags: std::collections::HashMap::new(),
            parameters: std::collections::HashMap::new(),
            model_warmup: vec![],
            model_operations: None,
            model_transaction_policy: None,
            model_repository_agents: None,
            response_cache: None,
            model_metrics: None,
            scheduling_choice: None,
        };

        let model_proxy = Arc::new(ModelProxy {
            data: super_tensor_buffer,
            model_config,
            model_metadata,
        });

        // Warmup each session before making the model available
        for (i, session) in sessions.iter_mut().enumerate() {
            info!(%model_name, executor = i, "warming up session");
            model_proxy
                .data
                .warmup(async |inputs| {
                    let run_options = RunOptions::new().unwrap();
                    let session_outputs = session
                        .run_async(inputs, &run_options)
                        .unwrap()
                        .await
                        .unwrap();
                    let mut values: SmallVec<[ort::value::Value; 4]> =
                        SmallVec::with_capacity(session_outputs.len());
                    session_outputs.into_iter().for_each(|(_, value)| {
                        values.push(value);
                    });
                    SessionValues { values }
                })
                .await;
        }
        info!(%model_name, "all sessions warmed up");

        let metrics_model_name = model_name.clone();

        // Register in the shared model map
        let current = self.loaded_models.load();
        let mut new_map = (**current).clone();
        new_map.insert(model_name.clone(), model_proxy.clone());
        self.loaded_models.store(Arc::new(new_map));
        self.metrics.loaded_models.inc();
        self.metrics
            .inference_configured_batch_size
            .with_label_values(&[&metrics_model_name])
            .set(batch_size as i64);
        Self::spawn_supertensor_metrics_reporter(
            self.metrics.clone(),
            metrics_model_name,
            model_proxy.clone(),
        );

        // Dispatch sessions round-robin to executor cores
        let stop_profiling_after = config
            .profiling
            .as_ref()
            .and_then(|p| p.stop_after_batches);
        for (i, session) in sessions.into_iter().enumerate() {
            let idx = self.round_robin.fetch_add(1, Ordering::Relaxed) % self.starters.len();
            self.starters[idx]
                .send(SessionStartRequest {
                    executor_id: format!("{}-executor-{i}", &model_name),
                    session,
                    model_proxy: model_proxy.clone(),
                    stop_profiling_after,
                })
                .await
                .map_err(|e| LoadError::DispatchFailed(e.to_string()))?;
        }

        Ok(())
    }
}
