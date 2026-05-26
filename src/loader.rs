use std::sync::Arc;

use ort::session::{RunOptions, Session};
use smallvec::SmallVec;

use tracing::info;

use crate::{
    metrics::with_local_metrics,
    model_runtime::gpu_binding::GpuBoundIo,
    scheduler::ModelProxy,
    tensor::supertensor::{InferError, SessionValues},
};

pub struct OnnxExecutor {
    pub id: String,
    pub session: Session,
    pub model: Arc<ModelProxy>,
    pub stop_profiling_after: Option<u64>,
    pub gpu_io: Option<GpuBoundIo>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        info!(id = %self.id, "executor started");
        let model_name = self.model.model_config.name.clone();
        let mut i: u64 = 0;
        loop {
            let batch_items = self
                .model
                .data
                .execute_on_batch(self.id.clone(), async |inputs| {
                    let start = std::time::Instant::now();
                    let session_values = match self.gpu_io.as_mut() {
                        Some(gpu_io) => run_gpu_bound(&mut self.session, gpu_io, inputs, &model_name)?,
                        None => run_host(&mut self.session, inputs, &model_name)?,
                    };
                    with_local_metrics(|m| {
                        m.observe_model_execution(&model_name, start.elapsed().as_secs_f64());
                    });
                    Ok(session_values)
                })
                .await;
            with_local_metrics(|m| {
                m.observe_batch_items(&model_name, batch_items as f64);
            });
            i += 1;
            if Some(i) == self.stop_profiling_after {
                self.session.end_profiling().unwrap();
            }
        }
    }
}

pub fn run_host(
    session: &mut Session,
    inputs: &[ort::session::SessionInputValue<'_>],
    model_name: &str,
) -> Result<SessionValues, InferError> {
    let run_options: RunOptions = RunOptions::new()
        .map_err(|e| InferError::SessionRun(Arc::from(e.to_string())))?;
    let run_start = std::time::Instant::now();
    let session_outputs = session
        .run_with_options(inputs, &run_options)
        .map_err(|e| InferError::SessionRun(Arc::from(e.to_string())))?;
    let run_elapsed = run_start.elapsed().as_secs_f64();
    with_local_metrics(|m| m.observe_model_session_run(model_name, run_elapsed));
    let mut values: SmallVec<[ort::value::Value; 4]> = SmallVec::with_capacity(session_outputs.len());
    session_outputs.into_iter().for_each(|(_, value)| {
        values.push(value);
    });
    Ok(SessionValues { values })
}

pub fn run_gpu_bound(
    session: &mut Session,
    gpu_io: &mut GpuBoundIo,
    inputs: &[ort::session::SessionInputValue<'_>],
    model_name: &str,
) -> Result<SessionValues, InferError> {
    // 1. Explicit H2D: pinned-host supertensor view -> bound GPU input tensors.
    //    Async: no per-copy stream sync. run_binding below sees the queued
    //    copies on the shared CUDA stream before it dispatches kernels.
    let h2d_start = std::time::Instant::now();
    for (cpu_view, gpu_input) in inputs.iter().zip(gpu_io.gpu_inputs.iter_mut()) {
        cpu_view
            .copy_into_async(gpu_input)
            .map_err(|e| InferError::SessionRun(Arc::from(e.to_string())))?;
    }
    let h2d_elapsed = h2d_start.elapsed().as_secs_f64();

    // 2. Run. ORT writes outputs in-place into the GPU tensors we bound at
    //    construction. The returned SessionOutputs are fresh DynValue handles
    //    over the same persistent GPU memory.
    let run_start = std::time::Instant::now();
    let session_outputs = session
        .run_binding(&gpu_io.binding)
        .map_err(|e| InferError::SessionRun(Arc::from(e.to_string())))?;
    let run_elapsed = run_start.elapsed().as_secs_f64();

    // 3. Explicit D2H: allocate a fresh CPU tensor per output, then async-copy
    //    into it. First N-1 copies are queued without sync; the last copy uses
    //    the sync variant so its tail stream-sync drains all prior async D2H
    //    copies on the same stream — single sync per batch instead of N.
    let d2h_start = std::time::Instant::now();
    let outputs: SmallVec<[_; 4]> = session_outputs.into_iter().collect();
    let n_outputs = outputs.len();
    let mut values: SmallVec<[ort::value::Value; 4]> = SmallVec::with_capacity(n_outputs);
    for (i, ((_, gpu_value), (dtype, shape))) in outputs
        .into_iter()
        .zip(gpu_io.output_dtypes_shapes.iter())
        .enumerate()
    {
        let mut cpu_t = ort::value::DynTensor::new(&gpu_io.cpu_alloc, *dtype, shape.clone())
            .map_err(|e| InferError::SessionRun(Arc::from(e.to_string())))?
            .into_dyn();
        let is_last = i + 1 == n_outputs;
        if is_last {
            gpu_value.copy_into(&mut cpu_t)
        } else {
            gpu_value.copy_into_async(&mut cpu_t)
        }
        .map_err(|e| InferError::SessionRun(Arc::from(e.to_string())))?;
        values.push(cpu_t);
    }
    let d2h_elapsed = d2h_start.elapsed().as_secs_f64();

    with_local_metrics(|m| {
        m.observe_model_session_run(model_name, run_elapsed);
        m.observe_model_h2d_copy(model_name, h2d_elapsed);
        m.observe_model_d2h_copy(model_name, d2h_elapsed);
    });

    Ok(SessionValues { values })
}
