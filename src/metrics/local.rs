use std::cell::RefCell;
use hashbrown::HashMap;
use std::sync::Arc;

use prometheus::local::{LocalHistogram, LocalIntCounter};

use super::MetricsRegistry;

thread_local! {
    static LOCAL_METRICS: RefCell<Option<LocalMetrics>> = const { RefCell::new(None) };
}

pub fn init_local_metrics(registry: Arc<MetricsRegistry>) {
    LOCAL_METRICS.with(|m| {
        *m.borrow_mut() = Some(LocalMetrics::new(registry));
    });
}

pub fn with_local_metrics<F, R>(f: F) -> R
where
    F: FnOnce(&LocalMetrics) -> R,
{
    LOCAL_METRICS.with(|m| {
        let borrow = m.borrow();
        f(borrow.as_ref().expect("LocalMetrics not initialized"))
    })
}

pub fn flush_local_metrics() {
    LOCAL_METRICS.with(|m| {
        if let Some(metrics) = m.borrow().as_ref() {
            metrics.flush();
        }
    });
}

struct LocalModelMetrics {
    requests_ok: LocalIntCounter,
    requests_not_found: LocalIntCounter,
    requests_resource_exhausted: LocalIntCounter,
    requests_error: LocalIntCounter,
    request_duration: LocalHistogram,
    batch_items: LocalHistogram,
    client_batch_size: LocalHistogram,
    model_execution: LocalHistogram,
    model_proxy_aquired: LocalHistogram,
    serialization_done: LocalHistogram,
    inference_in_queue: LocalHistogram,
    inference_exec_start: LocalHistogram,
    inference_exec_end: LocalHistogram,
    output_processed: LocalHistogram,
}

pub struct LocalMetrics {
    registry: Arc<MetricsRegistry>,
    per_model: RefCell<HashMap<String, LocalModelMetrics>>,
}

impl LocalMetrics {
    fn new(registry: Arc<MetricsRegistry>) -> Self {
        Self {
            registry,
            per_model: RefCell::new(HashMap::new()),
        }
    }

    fn ensure_model(&self, model: &str) {
        let mut map = self.per_model.borrow_mut();
        if !map.contains_key(model) {
            let r = &self.registry;
            map.insert(
                model.to_string(),
                LocalModelMetrics {
                    requests_ok: r
                        .inference_requests_total
                        .with_label_values(&[model, "ok"])
                        .local(),
                    requests_not_found: r
                        .inference_requests_total
                        .with_label_values(&[model, "not_found"])
                        .local(),
                    requests_resource_exhausted: r
                        .inference_requests_total
                        .with_label_values(&[model, "resource_exhausted"])
                        .local(),
                    requests_error: r
                        .inference_requests_total
                        .with_label_values(&[model, "error"])
                        .local(),
                    request_duration: r
                        .inference_request_duration_seconds
                        .with_label_values(&[model])
                        .local(),
                    batch_items: r
                        .inference_batch_items
                        .with_label_values(&[model])
                        .local(),
                    client_batch_size: r
                        .inference_client_batch_size
                        .with_label_values(&[model])
                        .local(),
                    model_execution: r
                        .inference_model_execution_seconds
                        .with_label_values(&[model])
                        .local(),
                    model_proxy_aquired: r
                        .inference_requests_model_proxy_aquired
                        .with_label_values(&[model])
                        .local(),
                    serialization_done: r
                        .inference_requests_serialization_done
                        .with_label_values(&[model])
                        .local(),
                    inference_in_queue: r
                        .inference_requests_inference_in_queue
                        .with_label_values(&[model])
                        .local(),
                    inference_exec_start: r
                        .inference_requests_inference_exec_start
                        .with_label_values(&[model])
                        .local(),
                    inference_exec_end: r
                        .inference_requests_inference_exec_end
                        .with_label_values(&[model])
                        .local(),
                    output_processed: r
                        .inference_requests_output_processed
                        .with_label_values(&[model])
                        .local(),
                },
            );
        }
    }

    pub fn inc_requests_ok(&self, model: &str) {
        self.ensure_model(model);
        self.per_model.borrow()[model].requests_ok.inc();
    }

    pub fn inc_requests_not_found(&self, model: &str) {
        self.ensure_model(model);
        self.per_model.borrow()[model].requests_not_found.inc();
    }

    pub fn inc_requests_resource_exhausted(&self, model: &str) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .requests_resource_exhausted
            .inc();
    }

    pub fn inc_requests_error(&self, model: &str) {
        self.ensure_model(model);
        self.per_model.borrow()[model].requests_error.inc();
    }

    pub fn observe_request_duration(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .request_duration
            .observe(duration);
    }

    pub fn observe_model_proxy_aquired(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .model_proxy_aquired
            .observe(duration);
    }

    pub fn observe_batch_items(&self, model: &str, count: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model].batch_items.observe(count);
    }

    pub fn observe_client_batch_size(&self, model: &str, count: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model].client_batch_size.observe(count);
    }

    pub fn observe_model_execution(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .model_execution
            .observe(duration);
    }

    pub fn observe_serialization_done(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .serialization_done
            .observe(duration);
    }

    pub fn observe_inference_in_queue(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .inference_in_queue
            .observe(duration);
    }

    pub fn observe_inference_exec_start(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .inference_exec_start
            .observe(duration);
    }

    pub fn observe_inference_exec_end(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .inference_exec_end
            .observe(duration);
    }

    pub fn observe_output_processed(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .output_processed
            .observe(duration);
    }

    pub fn flush(&self) {
        for (_, m) in self.per_model.borrow().iter() {
            m.requests_ok.flush();
            m.requests_not_found.flush();
            m.requests_resource_exhausted.flush();
            m.requests_error.flush();
            m.request_duration.flush();
            m.batch_items.flush();
            m.client_batch_size.flush();
            m.model_execution.flush();
            m.model_proxy_aquired.flush();
            m.serialization_done.flush();
            m.inference_in_queue.flush();
            m.inference_exec_start.flush();
            m.inference_exec_end.flush();
            m.output_processed.flush();
        }
    }
}

impl Drop for LocalMetrics {
    fn drop(&mut self) {
        self.flush();
    }
}
