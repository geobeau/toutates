use std::sync::Arc;

use ort::session::Session;
use tokio::sync::mpsc;

use crate::loader::OnnxExecutor;
use crate::scheduler::ModelProxy;

pub struct SessionStartRequest {
    pub executor_id: String,
    pub session: Session,
    pub model_proxy: Arc<ModelProxy>,
    pub stop_profiling_after: Option<u64>,
}

pub struct SessionStarter {
    receiver: mpsc::Receiver<SessionStartRequest>,
}

impl SessionStarter {
    pub fn new(receiver: mpsc::Receiver<SessionStartRequest>) -> Self {
        Self { receiver }
    }

    pub async fn run(mut self) {
        while let Some(req) = self.receiver.recv().await {
            compio::runtime::spawn(async move {
                let mut executor = OnnxExecutor {
                    id: req.executor_id,
                    session: req.session,
                    model: req.model_proxy,
                    stop_profiling_after: req.stop_profiling_after,
                };
                executor.run().await;
            })
            .detach();
        }
    }
}
