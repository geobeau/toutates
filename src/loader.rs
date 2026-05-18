use std::sync::Arc;

use ort::session::{RunOptions, Session};
use smallvec::SmallVec;

use tracing::{info, warn};

use crate::{
    metrics::with_local_metrics, scheduler::ModelProxy, tensor::supertensor::SessionValues,
};

pub struct OnnxExecutor {
    pub id: String,
    pub session: Session,
    pub model: Arc<ModelProxy>,
    pub stop_profiling_after: Option<u64>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        info!(id = %self.id, "executor started");
        let model_name = self.model.model_config.name.clone();
        let mut i: u64 = 0;
        loop {
            // println!("trying to execute another batch");
            let batch_items = self.model
                .data
                .execute_on_batch(self.id.clone(), async |inputs| {
                    let start = std::time::Instant::now();
                    let run_options: RunOptions = RunOptions::new().unwrap();
                    let fut = self.session.run_async(inputs, &run_options).unwrap();
                    let mut fut = std::pin::pin!(fut);
                    let session_outputs = tokio::select! {
                        biased;
                        result = fut.as_mut() => result.unwrap(),
                        _ = compio::time::sleep(std::time::Duration::from_millis(10)) => {
                            warn!(
                                executor = %self.id,
                                model = %model_name,
                                "ort::run_async exceeded 10ms — likely missed-wakeup hang"
                            );
                            fut.await.unwrap()
                        }
                    };
                    with_local_metrics(|m| {
                        m.observe_model_execution(&model_name, start.elapsed().as_secs_f64());
                    });
                    let mut values: smallvec::SmallVec<[ort::value::Value; 4]> =
                        SmallVec::with_capacity(session_outputs.len());
                    session_outputs.into_iter().for_each(|(_, value)| {
                        values.push(value);
                    });
                    SessionValues { values }
                })
                .await;
            with_local_metrics(|m| {
                m.observe_batch_items(&model_name, batch_items as f64);
            });
            i += 1;
            if Some(i) == self.stop_profiling_after {
                self.session.end_profiling().unwrap();
            }
            // println!("executed batch")
        }
    }
}
