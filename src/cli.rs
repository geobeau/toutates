use std::path::PathBuf;

use clap::Parser;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionProviderKind {
    Cpu,
    Cuda,
    TensorRT,
}

impl std::str::FromStr for ExecutionProviderKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" | "trt" => Ok(Self::TensorRT),
            other => Err(format!(
                "unknown execution provider '{other}'; expected one of: cpu, cuda, tensorrt (or trt)"
            )),
        }
    }
}

impl std::fmt::Display for ExecutionProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::TensorRT => write!(f, "tensorrt"),
        }
    }
}

#[derive(Parser)]
#[command(name = "toutates")]
pub struct Args {
    // ── Thread pool ──────────────────────────────────────────
    /// Number of compio worker threads for gRPC processing.
    /// Defaults to the number of available CPUs.
    #[arg(long, default_value_t = default_num_cores())]
    pub processing_cores: usize,

    /// Number of compio worker threads for model executors.
    /// Defaults to 1.
    #[arg(long, default_value_t = 1)]
    pub executor_cores: usize,

    /// ORT intra-op thread pool size (parallelism within a single operator).
    #[arg(long, default_value_t = 5)]
    pub ort_intra_threads: usize,

    /// ORT inter-op thread pool size (parallelism between independent operators).
    #[arg(long, default_value_t = 5)]
    pub ort_inter_threads: usize,

    // ── Model repository (local or S3, mutually exclusive) ───
    /// Path to a local model repository directory.
    /// Mutually exclusive with --s3-bucket.
    #[arg(long, conflicts_with_all = ["s3_bucket", "s3_endpoint"])]
    pub model_repository: Option<PathBuf>,

    // ── S3 model repository ──────────────────────────────────
    /// S3-compatible endpoint URL (e.g. "https://s3.my-infra.com").
    #[arg(long, requires = "s3_bucket")]
    pub s3_endpoint: Option<String>,

    /// S3 bucket containing the model repository.
    #[arg(long, requires = "s3_endpoint")]
    pub s3_bucket: Option<String>,

    /// Key prefix inside the bucket (e.g. "models/prod").
    #[arg(long, default_value = "")]
    pub s3_prefix: String,

    /// AWS region for the S3 bucket.
    #[arg(long, default_value = "us-east-1")]
    pub s3_region: String,

    /// Local directory to cache downloaded S3 models.
    #[arg(long, default_value = "/tmp/model_cache")]
    pub model_cache_dir: PathBuf,

    /// Comma-separated list of model names to load at startup.
    /// If omitted, all discovered models are loaded.
    #[arg(long, value_delimiter = ',')]
    pub load_models: Option<Vec<String>>,

    // ── Execution providers ────────────────────────────────────
    /// Comma-separated list of execution providers in priority order.
    /// Supported: cpu, cuda, tensorrt (or trt).
    /// Defaults to "tensorrt".
    #[arg(long, value_delimiter = ',', default_values_t = vec![ExecutionProviderKind::TensorRT])]
    pub execution_providers: Vec<ExecutionProviderKind>,

    /// Paths to custom operator shared libraries (.so) to register.
    #[arg(long, value_delimiter = ',')]
    pub custom_op_libraries: Option<Vec<PathBuf>>,

    /// Comma-separated list of op types to exclude from TensorRT execution.
    /// These ops will fall back to CPU instead.
    /// Example: --trt-op-types-to-exclude LabelEncoder,Imputer,Cast
    #[arg(long, value_delimiter = ',')]
    pub trt_op_types_to_exclude: Option<Vec<String>>,

    /// Minimum number of nodes in a subgraph for TensorRT partitioning.
    #[arg(long)]
    pub trt_min_subgraph_size: Option<usize>,

    // ── gRPC / HTTP/2 server ─────────────────────────────────
    /// gRPC listen address.
    #[arg(long, default_value = "0.0.0.0:8001")]
    pub grpc_addr: String,

    /// Maximum concurrent HTTP/2 connections.
    #[arg(long, default_value_t = 100_000)]
    pub max_concurrent_connections: usize,

    /// Maximum concurrent HTTP/2 streams per connection.
    #[arg(long, default_value_t = 100_000)]
    pub max_concurrent_streams: usize,

    /// Maximum HTTP/2 frame size in bytes (max 16777215).
    #[arg(long, default_value_t = 32 * 1024)]
    pub max_frame_size: usize,

    /// Pin each compio runtime thread to a CPU from the process cpuset in round-robin order.
    #[arg(long, default_value_t = false)]
    pub cpu_pinning: bool,

    /// Enable io_uring SQPOLL for gRPC processing cores. Each runtime gets its
    /// own SQPOLL kernel thread pinned to a dedicated vCPU. Runtimes and
    /// SQPOLL kthreads are paired within the same NUMA node but on different
    /// physical cores (no SMT contention between userspace and kthread).
    /// Requires --cpu-pinning.
    #[arg(long, default_value_t = false)]
    pub sqpoll_enabled: bool,

    /// SQPOLL kernel-thread idle timeout, in milliseconds.
    #[arg(long, default_value_t = 100)]
    pub sqpoll_idle_ms: u32,

    /// io_uring SQ/CQ ring capacity (entries) per runtime. The rings are
    /// kernel-mapped and charged against RLIMIT_MEMLOCK on recent kernels —
    /// drop this if you hit ENOMEM during runtime build with many workers.
    #[arg(long, default_value_t = 8096)]
    pub io_uring_capacity: u32,

    /// io_uring registered buffer pool: number of buffers per runtime.
    /// Rounded up to next power of two.
    #[arg(long, default_value_t = 4096)]
    pub buffer_pool_size: u16,

    /// io_uring registered buffer pool: size of each buffer in bytes.
    /// Each read fills at most one buffer; sets the per-recv ceiling.
    #[arg(long, default_value_t = 8 * 1024)]
    pub buffer_pool_buffer_len: usize,
}

pub enum ModelSource {
    Local(PathBuf),
    S3 {
        endpoint: String,
        bucket: String,
        prefix: String,
        region: String,
        cache_dir: PathBuf,
    },
}

impl Args {
    pub fn model_source(&self) -> ModelSource {
        if let Some(ref path) = self.model_repository {
            ModelSource::Local(path.clone())
        } else if let (Some(ref endpoint), Some(ref bucket)) = (&self.s3_endpoint, &self.s3_bucket)
        {
            ModelSource::S3 {
                endpoint: endpoint.clone(),
                bucket: bucket.clone(),
                prefix: self.s3_prefix.clone(),
                region: self.s3_region.clone(),
                cache_dir: self.model_cache_dir.clone(),
            }
        } else {
            eprintln!(
                "error: either --model-repository or --s3-bucket + --s3-endpoint is required"
            );
            std::process::exit(1);
        }
    }
}

fn default_num_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}


