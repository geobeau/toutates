use std::sync::Arc;

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::{DynTensor, DynValue, Shape, TensorElementType};
use smallvec::SmallVec;

use crate::tensor::supertensor::InferError;

pub struct GpuBoundIoSpec {
    pub inputs: SmallVec<[(String, TensorElementType, Shape); 6]>,
    pub outputs: SmallVec<[(String, TensorElementType, Shape); 4]>,
}

pub struct GpuBoundIo {
    pub binding: IoBinding,
    /// Persistent GPU-resident input tensors. Bound once at construction;
    /// `copy_into` rewrites their underlying GPU memory each iteration.
    pub gpu_inputs: SmallVec<[DynValue; 6]>,
    /// CPU allocator used for per-run output landing tensors. Fresh CPU
    /// tensors are allocated each iteration so the SessionValues we hand to
    /// downstream consumers don't race with the next iteration overwriting
    /// the same buffer.
    pub cpu_alloc: Allocator,
    /// dtype + shape for each model output, in output order. Used to
    /// allocate the CPU landing tensors each run.
    pub output_dtypes_shapes: SmallVec<[(TensorElementType, Shape); 4]>,
}

impl GpuBoundIo {
    pub fn new(session: &Session, spec: &GpuBoundIoSpec) -> Result<Self, InferError> {
        let gpu_alloc = Allocator::new(
            session,
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(map_err)?,
        )
        .map_err(map_err)?;

        let cpu_alloc = Allocator::new(
            session,
            MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )
            .map_err(map_err)?,
        )
        .map_err(map_err)?;

        let mut binding = session.create_binding().map_err(map_err)?;

        let mut gpu_inputs: SmallVec<[DynValue; 6]> = SmallVec::new();
        for (name, dtype, shape) in spec.inputs.iter() {
            let t = DynTensor::new(&gpu_alloc, *dtype, shape.clone())
                .map_err(map_err)?
                .into_dyn();
            binding.bind_input(name.as_str(), &t).map_err(map_err)?;
            gpu_inputs.push(t);
        }

        let mut output_dtypes_shapes: SmallVec<[(TensorElementType, Shape); 4]> = SmallVec::new();
        for (name, dtype, shape) in spec.outputs.iter() {
            // Allocate the persistent GPU output buffer and hand ownership to
            // the binding. ORT writes into this same buffer on each run.
            let gpu_t = DynTensor::new(&gpu_alloc, *dtype, shape.clone())
                .map_err(map_err)?
                .into_dyn();
            binding
                .bind_output(name.as_str(), gpu_t)
                .map_err(map_err)?;
            output_dtypes_shapes.push((*dtype, shape.clone()));
        }

        Ok(GpuBoundIo {
            binding,
            gpu_inputs,
            cpu_alloc,
            output_dtypes_shapes,
        })
    }
}

fn map_err<E: std::fmt::Display>(e: E) -> InferError {
    InferError::SessionRun(Arc::from(e.to_string()))
}
