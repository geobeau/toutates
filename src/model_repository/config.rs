use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Supertensor,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocatorKind {
    Cpu,
    CudaPinned,
    Cuda,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfilingConfig {
    #[serde(default = "default_profile_prefix")]
    pub file_prefix: String,
    pub stop_after_batches: Option<u64>,
}

fn default_profile_prefix() -> String {
    String::from("ort_profile")
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRepositoryConfig {
    pub backend: Backend,
    pub batch_size: usize,
    pub capacity: usize,
    pub num_executors: usize,
    pub allocator: AllocatorKind,
    #[serde(default)]
    pub input_shapes: HashMap<String, Vec<i64>>,
    #[serde(default)]
    pub output_shapes: HashMap<String, Vec<i64>>,
    #[serde(default)]
    pub profiling: Option<ProfilingConfig>,
}
