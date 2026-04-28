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

    pub inference_requests_model_proxy_aquired: HistogramVec,
    pub inference_requests_serialization_done: HistogramVec,
    pub inference_requests_inference_in_queue: HistogramVec,
    pub inference_requests_inference_exec_start: HistogramVec,
    pub inference_requests_inference_exec_end: HistogramVec,
    pub inference_requests_output_processed: HistogramVec,

    pub inference_queue_depth: IntGauge,
    pub inference_executors_in_use: IntGaugeVec,
    pub inference_ring_tail_index: IntGaugeVec,
    pub inference_ring_in_use_index: IntGaugeVec,
    pub inference_ring_head_index: IntGaugeVec,
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

        let inference_queue_depth = IntGauge::new(
            "inference_queue_depth",
            "Current depth of the inference queue",
        )
        .unwrap();
        registry
            .register(Box::new(inference_queue_depth.clone()))
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

        let inference_ring_tail_index = IntGaugeVec::new(
            Opts::new(
                "inference_ring_tail_index",
                "Current ring buffer tail index",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_ring_tail_index.clone()))
            .unwrap();

        let inference_ring_in_use_index = IntGaugeVec::new(
            Opts::new(
                "inference_ring_in_use_index",
                "Current ring buffer in-use index",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_ring_in_use_index.clone()))
            .unwrap();

        let inference_ring_head_index = IntGaugeVec::new(
            Opts::new(
                "inference_ring_head_index",
                "Current ring buffer head index",
            ),
            &["model"],
        )
        .unwrap();
        registry
            .register(Box::new(inference_ring_head_index.clone()))
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
            inference_requests_model_proxy_aquired,
            inference_requests_serialization_done,
            inference_requests_inference_in_queue,
            inference_requests_inference_exec_start,
            inference_requests_inference_exec_end,
            inference_requests_output_processed,
            inference_queue_depth,
            inference_executors_in_use,
            inference_ring_tail_index,
            inference_ring_in_use_index,
            inference_ring_head_index,
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
