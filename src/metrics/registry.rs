use prometheus::{
    exponential_buckets, Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry, TextEncoder,
};

pub struct MetricsRegistry {
    registry: Registry,

    pub inference_requests_total: IntCounterVec,
    pub inference_batches_total: IntCounterVec,
    pub inference_batch_items: HistogramVec,
    pub inference_client_batch_size: HistogramVec,

    pub inference_request_duration_seconds: HistogramVec,
    pub inference_batch_duration_seconds: HistogramVec,
    pub inference_model_execution_seconds: HistogramVec,
    pub inference_model_session_run_seconds: HistogramVec,
    pub inference_model_h2d_copy_seconds: HistogramVec,
    pub inference_model_d2h_copy_seconds: HistogramVec,

    pub inference_requests_model_proxy_aquired: HistogramVec,
    pub inference_requests_serialization_done: HistogramVec,
    pub inference_requests_inference_in_queue: HistogramVec,
    pub inference_requests_inference_exec_start: HistogramVec,
    pub inference_requests_inference_exec_end: HistogramVec,
    pub inference_requests_output_processed: HistogramVec,

    pub inference_inflight: IntGaugeVec,
    pub inference_capacity: IntGaugeVec,
    pub inference_executors_in_use: IntGaugeVec,
    pub inference_configured_batch_size: IntGaugeVec,
    pub loaded_models: IntGauge,
}

fn make_histogram_opts(name: &str, help: &str) -> HistogramOpts {
    HistogramOpts::new(name, help).buckets(exponential_buckets(0.0001, 2.0, 18).unwrap())
}

impl MetricsRegistry {
    pub fn new() -> Self {
        let registry = Registry::new();

        let inference_requests_total = IntCounterVec::new(
            Opts::new(
                "inference_requests_total",
                "Total number of inference requests",
            ),
            &["model", "status"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_total.clone()))
            .unwrap();

        let inference_batches_total = IntCounterVec::new(
            Opts::new(
                "inference_batches_total",
                "Total number of inference batches",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_batches_total.clone()))
            .unwrap();

        let inference_batch_items = HistogramVec::new(
            HistogramOpts::new(
                "inference_batch_items",
                "Number of items in a batch before execution",
            )
            .buckets(exponential_buckets(1.0, 2.0, 12).unwrap()),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_batch_items.clone()))
            .unwrap();

        let inference_client_batch_size = HistogramVec::new(
            HistogramOpts::new(
                "inference_client_batch_size",
                "Batch size requested by clients (input tensor dim 0)",
            )
            .buckets(exponential_buckets(1.0, 2.0, 12).unwrap()),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_client_batch_size.clone()))
            .unwrap();

        let inference_request_duration_seconds = HistogramVec::new(
            make_histogram_opts(
                "inference_request_duration_seconds",
                "Duration of inference requests in seconds",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_request_duration_seconds.clone()))
            .unwrap();

        let inference_batch_duration_seconds = HistogramVec::new(
            make_histogram_opts(
                "inference_batch_duration_seconds",
                "Duration of inference batches in seconds",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_batch_duration_seconds.clone()))
            .unwrap();

        let inference_model_execution_seconds = HistogramVec::new(
            make_histogram_opts(
                "inference_model_execution_seconds",
                "Duration of model execution (ORT session run) in seconds",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_model_execution_seconds.clone()))
            .unwrap();

        let inference_model_session_run_seconds = HistogramVec::new(
            make_histogram_opts(
                "inference_model_session_run_seconds",
                "Duration of the ORT session run call only (excludes H2D/D2H copies) in seconds",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_model_session_run_seconds.clone()))
            .unwrap();

        let inference_model_h2d_copy_seconds = HistogramVec::new(
            make_histogram_opts(
                "inference_model_h2d_copy_seconds",
                "Duration of host-to-device input copy for GPU-bound executors in seconds",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_model_h2d_copy_seconds.clone()))
            .unwrap();

        let inference_model_d2h_copy_seconds = HistogramVec::new(
            make_histogram_opts(
                "inference_model_d2h_copy_seconds",
                "Duration of device-to-host output copy for GPU-bound executors in seconds",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_model_d2h_copy_seconds.clone()))
            .unwrap();

        let inference_requests_model_proxy_aquired = HistogramVec::new(
            make_histogram_opts(
                "inference_request_model_proxy_aquired_seconds",
                "STEP 1: How long it took to fetch the reference the model",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_model_proxy_aquired.clone()))
            .unwrap();

        let inference_requests_serialization_done = HistogramVec::new(
            make_histogram_opts(
                "inference_serialization_done_seconds",
                "STEP 2: How long it took to deserialize from GRPC and serialize to the inner batch structure",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_serialization_done.clone()))
            .unwrap();

        let inference_requests_inference_in_queue = HistogramVec::new(
            make_histogram_opts(
                "inference_inference_in_queue_seconds",
                "STEP 3: How long it took to acquire a slot within the inner batch structure and copy data",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_inference_in_queue.clone()))
            .unwrap();

        let inference_requests_inference_exec_start = HistogramVec::new(
            make_histogram_opts(
                "inference_exec_start_seconds",
                "STEP 4: How long request was queued",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_inference_exec_start.clone()))
            .unwrap();

        let inference_requests_inference_exec_end = HistogramVec::new(
            make_histogram_opts(
                "inference_exec_end_seconds",
                "STEP 5: How long request was executed",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_inference_exec_end.clone()))
            .unwrap();

        let inference_requests_output_processed = HistogramVec::new(
            make_histogram_opts(
                "inference_output_processed_seconds",
                "STEP 6: how long after execution it was processed by the response thread",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_requests_output_processed.clone()))
            .unwrap();

        let inference_inflight = IntGaugeVec::new(
            Opts::new(
                "inference_inflight",
                "Total slots in the pipeline (queued + executing)",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_inflight.clone()))
            .unwrap();

        let inference_capacity = IntGaugeVec::new(
            Opts::new(
                "inference_capacity",
                "Total ring buffer capacity (batch_size * num_batches)",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_capacity.clone()))
            .unwrap();

        let inference_executors_in_use = IntGaugeVec::new(
            Opts::new(
                "inference_executors_in_use",
                "Number of executors currently running a batch",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_executors_in_use.clone()))
            .unwrap();

        let inference_configured_batch_size = IntGaugeVec::new(
            Opts::new(
                "inference_configured_batch_size",
                "Configured batch size per model",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_configured_batch_size.clone()))
            .unwrap();

        let loaded_models =
            IntGauge::new("loaded_models", "Number of currently loaded models").unwrap();
        registry.register(Box::new(loaded_models.clone())).unwrap();

        Self {
            registry,
            inference_requests_total,
            inference_batches_total,
            inference_batch_items,
            inference_client_batch_size,
            inference_request_duration_seconds,
            inference_batch_duration_seconds,
            inference_model_execution_seconds,
            inference_model_session_run_seconds,
            inference_model_h2d_copy_seconds,
            inference_model_d2h_copy_seconds,
            inference_requests_model_proxy_aquired,
            inference_requests_serialization_done,
            inference_requests_inference_in_queue,
            inference_requests_inference_exec_start,
            inference_requests_inference_exec_end,
            inference_requests_output_processed,
            inference_inflight,
            inference_capacity,
            inference_executors_in_use,
            inference_configured_batch_size,
            loaded_models,
        }
    }

    pub fn encode(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buf = Vec::new();
        encoder
            .encode(&metric_families, &mut buf)
            .expect("failed to encode metrics");
        String::from_utf8(buf).expect("metrics output is not valid UTF-8")
    }
}
