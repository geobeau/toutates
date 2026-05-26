use std::sync::Arc;
use std::time::Duration;

use ort::session::Session;

use crate::loader::OnnxExecutor;
use crate::metrics::{flush_local_metrics, init_local_metrics, MetricsRegistry};
use crate::model_runtime::gpu_binding::{GpuBoundIo};
use crate::scheduler::ModelProxy;

use tracing::info;

pub fn spawn_session_thread(
    executor_id: String,
    session: Session,
    model_proxy: Arc<ModelProxy>,
    stop_profiling_after: Option<u64>,
    pin_cpu: Option<core_affinity::CoreId>,
    metrics: Arc<MetricsRegistry>,
    gpu_io: Option<GpuBoundIo>,
) {
    let thread_name = executor_id.clone();
    std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            // A panic on a session thread leaves its ring un-drained; abort the
            // whole process to match the pre-refactor handle.join() semantics.
            std::panic::set_hook(Box::new(|info| {
                eprintln!("session thread panic: {info}");
                std::process::abort();
            }));

            if let Some(core) = pin_cpu {
                if !core_affinity::set_for_current(core) {
                    panic!("Failed to pin {executor_id} to CPU {}", core.id);
                }
                info!("Pinned {executor_id} to CPU {}", core.id);
            }

            let rt = compio::runtime::RuntimeBuilder::new()
                .build()
                .expect("failed to build session compio runtime");
            rt.block_on(async move {
                init_local_metrics(metrics);
                compio::runtime::spawn(async {
                    loop {
                        compio::time::sleep(Duration::from_secs(1)).await;
                        flush_local_metrics();
                    }
                })
                .detach();

                let mut executor = OnnxExecutor {
                    id: executor_id,
                    session,
                    model: model_proxy,
                    stop_profiling_after,
                    gpu_io,
                };
                executor.run().await;
            });
        })
        .expect("failed to spawn session thread");
}
