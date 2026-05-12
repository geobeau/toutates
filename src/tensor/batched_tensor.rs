pub struct TensorBytes<'a> {
    pub data_type: DataType,
    pub shape: Shape,
    pub data: &'a [u8],
}

impl<'a> TensorBytes<'a> {
    pub fn slice_dim0(&self, start: usize, end: usize) -> &[u8] {
        assert!(!self.shape.is_empty());

        let dim0_size = self.shape[0] as usize;
        // println!("{start} < {end} < {dim0_size} ({:?})", self.shape);
        assert!(start < end);

        // Calculate the number of bytes each index in dimension 0 occupies
        let bytes_per_index = self.data.len() / dim0_size;

        let byte_start = start * bytes_per_index;
        let byte_end = end * bytes_per_index;

        return &self.data[byte_start..byte_end];
    }
}

pub struct BatchableTensor {
    pub inner_tensor: DynTensor,
    pub data_type: TensorElementType,
    pub single_tensor_shape: Shape,
}

pub struct BatchedTensor {
    pub inner_tensor: DynValue,
    pub data_type: TensorElementType,
    tensor_shape: Shape,
}

macro_rules! copy_tensor_slice_from_bytes {
    ($ty:ty, $inner:expr, $single_tensor_shape:expr, $tensor_to_append:expr, $slot:expr) => {{
        let tensor_size = $single_tensor_shape.num_elements();
        let (_, data) = $inner.try_extract_tensor_mut::<$ty>().unwrap();
        let offset = $slot * tensor_size;
        let src: &[u8] = $tensor_to_append;
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut data[offset..offset + tensor_size]);
        dst.copy_from_slice(src);
    }};
}

macro_rules! copy_tensor_slice {
    ($ty:ty, $inner:expr, $tensor_to_append:expr, $slot:expr) => {{
        let (new_shape, new_data) = $tensor_to_append.try_extract_tensor::<$ty>().unwrap();
        let tensor_size = new_shape.num_elements();
        let (_, data) = $inner.try_extract_tensor_mut::<$ty>().unwrap();
        data[$slot * tensor_size..($slot + 1) * tensor_size].copy_from_slice(new_data);
    }};
}

macro_rules! pop_tensor_slice {
    ($ty:ty, $inner:expr, $new_tensor:expr, $offset:expr) => {{
        let (new_shape, data) = $new_tensor.try_extract_tensor_mut::<$ty>().unwrap();
        let (_, inner_data) = $inner.try_extract_tensor::<$ty>().unwrap();
        let tensor_size = new_shape.num_elements();
        data.copy_from_slice(&inner_data[$offset * tensor_size..($offset + 1) * tensor_size]);
    }};
}

macro_rules! as_byte_slice {
    ($ty:ty, $inner:expr, $start:expr, $end:expr) => {{
        let (_, inner_data) = $inner.try_extract_tensor::<$ty>().unwrap();
        bytemuck::cast_slice(&inner_data[$start..$end])
    }};
}

// pub struct BatchableInputs {
//     pub inputs: HashMap<String, BatchableTensor>,
// }

// impl BatchableInputs {
//     pub fn new(inputs: &[Input], batch_size: usize) -> BatchableInputs {
//         let mut inner_inputs = HashMap::with_capacity(inputs.len());
//         inputs.iter().for_each(|input| {
//             let shape = input.input_type.tensor_shape().unwrap();
//             let data_type = input.input_type.tensor_type().unwrap();
//             inner_inputs.insert(
//                 input.name.clone(),
//                 BatchableTensor::new(data_type, shape, batch_size),
//             );
//         });

//         BatchableInputs {
//             inputs: inner_inputs,
//         }
//     }

//     pub fn copy_at(&mut self, slot: usize, inputs: HashMap<String, DynTensor>) {
//         inputs.iter().for_each(|(name, tensor)| {
//             self.inputs.get_mut(name).unwrap().copy_at(slot, tensor);
//         });
//     }

//     pub fn session_inputs_view(&self) -> HashMap<String, ValueRef<'_, DynTensorValueType>> {
//         let mut all_inputs = HashMap::new();

//         self.inputs.iter().for_each(|(name, batch_tensor)| {
//             all_inputs.insert(name.clone(), batch_tensor.inner_tensor.view());
//         });
//         all_inputs
//     }
// }

pub struct BatchedOutputs {
    pub outputs: HashMap<String, BatchedTensor>,
}

impl BatchedOutputs {
    pub fn new(outputs: SessionOutputs) -> BatchedOutputs {
        let mut inner_ouputs = HashMap::with_capacity(outputs.len());
        outputs.into_iter().for_each(|(name, dyn_value)| {
            let batched_tensor = BatchedTensor::from(dyn_value);
            inner_ouputs.insert(String::from(name), batched_tensor);
        });

        BatchedOutputs {
            outputs: inner_ouputs,
        }
    }

    pub fn pop_outputs(&self, slot: usize) -> HashMap<String, DynTensor> {
        let mut outputs = HashMap::new();

        self.outputs.iter().for_each(|(name, batch_tensor)| {
            outputs.insert(name.clone(), batch_tensor.pop_at(slot));
        });
        outputs
    }
}

pub fn value_as_byte_slice(value: &Value, start: usize, end: usize) -> &[u8] {
    match value.data_type() {
        TensorElementType::Float32 => {
            as_byte_slice!(f32, value, start, end)
        }
        TensorElementType::Uint8 => {
            as_byte_slice!(u8, value, start, end)
        }
        TensorElementType::Int8 => {
            as_byte_slice!(i8, value, start, end)
        }
        TensorElementType::Uint16 => {
            as_byte_slice!(u16, value, start, end)
        }
        TensorElementType::Int16 => {
            as_byte_slice!(i16, value, start, end)
        }
        TensorElementType::Int32 => {
            as_byte_slice!(i32, value, start, end)
        }
        TensorElementType::Int64 => {
            as_byte_slice!(i64, value, start, end)
        }
        TensorElementType::Bool => {
            as_byte_slice!(bool, value, start, end)
        }
        TensorElementType::Float64 => {
            as_byte_slice!(f64, value, start, end)
        }
        TensorElementType::Uint32 => {
            as_byte_slice!(u32, value, start, end)
        }
        TensorElementType::Uint64 => {
            as_byte_slice!(u64, value, start, end)
        }

        // Unsupported or special handling types
        TensorElementType::Float16 => todo!(),
        TensorElementType::Bfloat16 => todo!(),
        TensorElementType::Complex64 => todo!(),
        TensorElementType::Complex128 => todo!(),
        TensorElementType::Float8E4M3FN => todo!(),
        TensorElementType::Float8E4M3FNUZ => todo!(),
        TensorElementType::Float8E5M2 => todo!(),
        TensorElementType::Float8E5M2FNUZ => todo!(),
        TensorElementType::Uint4 => todo!(),
        TensorElementType::Int4 => todo!(),
        TensorElementType::String => todo!(),
        TensorElementType::Undefined => todo!(),
        _ => todo!(),
    }
}

impl BatchedTensor {
    fn from(dyn_value: DynValue) -> BatchedTensor {
        let mut shape = dyn_value.shape().clone();
        // the shape of each sub tensor
        shape[0] = 1;
        BatchedTensor {
            data_type: *dyn_value.data_type(),
            tensor_shape: shape,
            inner_tensor: dyn_value,
        }
    }

    pub fn as_byte_slice(&self, start: usize, end: usize) -> &[u8] {
        match self.data_type {
            TensorElementType::Float32 => {
                as_byte_slice!(f32, self.inner_tensor, start, end)
            }
            TensorElementType::Uint8 => {
                as_byte_slice!(u8, self.inner_tensor, start, end)
            }
            TensorElementType::Int8 => {
                as_byte_slice!(i8, self.inner_tensor, start, end)
            }
            TensorElementType::Uint16 => {
                as_byte_slice!(u16, self.inner_tensor, start, end)
            }
            TensorElementType::Int16 => {
                as_byte_slice!(i16, self.inner_tensor, start, end)
            }
            TensorElementType::Int32 => {
                as_byte_slice!(i32, self.inner_tensor, start, end)
            }
            TensorElementType::Int64 => {
                as_byte_slice!(i64, self.inner_tensor, start, end)
            }
            TensorElementType::Bool => {
                as_byte_slice!(bool, self.inner_tensor, start, end)
            }
            TensorElementType::Float64 => {
                as_byte_slice!(f64, self.inner_tensor, start, end)
            }
            TensorElementType::Uint32 => {
                as_byte_slice!(u32, self.inner_tensor, start, end)
            }
            TensorElementType::Uint64 => {
                as_byte_slice!(u64, self.inner_tensor, start, end)
            }

            // Unsupported or special handling types
            TensorElementType::Float16 => todo!(),
            TensorElementType::Bfloat16 => todo!(),
            TensorElementType::Complex64 => todo!(),
            TensorElementType::Complex128 => todo!(),
            TensorElementType::Float8E4M3FN => todo!(),
            TensorElementType::Float8E4M3FNUZ => todo!(),
            TensorElementType::Float8E5M2 => todo!(),
            TensorElementType::Float8E5M2FNUZ => todo!(),
            TensorElementType::Uint4 => todo!(),
            TensorElementType::Int4 => todo!(),
            TensorElementType::String => todo!(),
            TensorElementType::Undefined => todo!(),
        _ => todo!(),
        }
    }

    pub fn pop_at(&self, slot: usize) -> DynTensor {
        let mut tensor = DynTensor::new(
            &Allocator::default(),
            self.data_type,
            self.tensor_shape.clone(),
        )
        .unwrap();
        match self.data_type {
            TensorElementType::Float32 => {
                pop_tensor_slice!(f32, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint8 => {
                pop_tensor_slice!(u8, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int8 => {
                pop_tensor_slice!(i8, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint16 => {
                pop_tensor_slice!(u16, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int16 => {
                pop_tensor_slice!(i16, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int32 => {
                pop_tensor_slice!(i32, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int64 => {
                pop_tensor_slice!(i64, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Bool => {
                pop_tensor_slice!(bool, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Float64 => {
                pop_tensor_slice!(f64, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint32 => {
                pop_tensor_slice!(u32, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint64 => {
                pop_tensor_slice!(u64, self.inner_tensor, tensor, slot)
            }

            // Unsupported or special handling types
            TensorElementType::Float16 => todo!(),
            TensorElementType::Bfloat16 => todo!(),
            TensorElementType::Complex64 => todo!(),
            TensorElementType::Complex128 => todo!(),
            TensorElementType::Float8E4M3FN => todo!(),
            TensorElementType::Float8E4M3FNUZ => todo!(),
            TensorElementType::Float8E5M2 => todo!(),
            TensorElementType::Float8E5M2FNUZ => todo!(),
            TensorElementType::Uint4 => todo!(),
            TensorElementType::Int4 => todo!(),
            TensorElementType::String => todo!(),
            TensorElementType::Undefined => todo!(),
        _ => todo!(),
        };
        tensor
    }
}

impl BatchableTensor {
    pub fn new(
        data_type: TensorElementType,
        shape: &Shape,
        batch_size: usize,
        allocator: &Allocator,
    ) -> BatchableTensor {
        assert_eq!(shape[0], -1, "Shape is not dynamic");
        let mut batch_shape = shape.clone();
        let mut single_tensor_shape = shape.clone();
        batch_shape[0] = batch_size as i64;
        let all_shape_static = batch_shape.iter().all(|dim| *dim >= 0);
        assert!(all_shape_static, "Batch shape contains a dimension that is dynamic and not the batching dimension {batch_shape:?}");
        println!("Shape {:?}", batch_shape);
        single_tensor_shape[0] = 1;
        BatchableTensor {
            inner_tensor: DynTensor::new(allocator, data_type, batch_shape).unwrap(),
            data_type,
            single_tensor_shape: single_tensor_shape,
        }
    }

    pub fn copy_at_from_bytes(&mut self, slot: usize, tensor: &[u8]) {
        match self.data_type {
            TensorElementType::Float32 => {
                copy_tensor_slice_from_bytes!(
                    f32,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Uint8 => {
                copy_tensor_slice_from_bytes!(
                    u8,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Int8 => {
                copy_tensor_slice_from_bytes!(
                    i8,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Uint16 => {
                copy_tensor_slice_from_bytes!(
                    u16,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Int16 => {
                copy_tensor_slice_from_bytes!(
                    i16,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Int32 => {
                copy_tensor_slice_from_bytes!(
                    i32,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Int64 => {
                copy_tensor_slice_from_bytes!(
                    i64,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Bool => {
                todo!()
            }
            TensorElementType::Float64 => {
                copy_tensor_slice_from_bytes!(
                    f64,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Uint32 => {
                copy_tensor_slice_from_bytes!(
                    u32,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }
            TensorElementType::Uint64 => {
                copy_tensor_slice_from_bytes!(
                    u64,
                    self.inner_tensor,
                    self.single_tensor_shape,
                    tensor,
                    slot
                )
            }

            // Unsupported or special handling types
            TensorElementType::Float16 => todo!(),
            TensorElementType::Bfloat16 => todo!(),
            TensorElementType::Complex64 => todo!(),
            TensorElementType::Complex128 => todo!(),
            TensorElementType::Float8E4M3FN => todo!(),
            TensorElementType::Float8E4M3FNUZ => todo!(),
            TensorElementType::Float8E5M2 => todo!(),
            TensorElementType::Float8E5M2FNUZ => todo!(),
            TensorElementType::Uint4 => todo!(),
            TensorElementType::Int4 => todo!(),
            TensorElementType::String => todo!(),
            TensorElementType::Undefined => todo!(),
        _ => todo!(),
        };
    }

    pub fn copy_at_from_dyntensor(&mut self, slot: usize, tensor: &DynTensor) {
        match self.data_type {
            TensorElementType::Float32 => {
                copy_tensor_slice!(f32, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint8 => {
                copy_tensor_slice!(u8, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int8 => {
                copy_tensor_slice!(i8, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint16 => {
                copy_tensor_slice!(u16, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int16 => {
                copy_tensor_slice!(i16, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int32 => {
                copy_tensor_slice!(i32, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Int64 => {
                copy_tensor_slice!(i64, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Bool => {
                copy_tensor_slice!(bool, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Float64 => {
                copy_tensor_slice!(f64, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint32 => {
                copy_tensor_slice!(u32, self.inner_tensor, tensor, slot)
            }
            TensorElementType::Uint64 => {
                copy_tensor_slice!(u64, self.inner_tensor, tensor, slot)
            }

            // Unsupported or special handling types
            TensorElementType::Float16 => todo!(),
            TensorElementType::Bfloat16 => todo!(),
            TensorElementType::Complex64 => todo!(),
            TensorElementType::Complex128 => todo!(),
            TensorElementType::Float8E4M3FN => todo!(),
            TensorElementType::Float8E4M3FNUZ => todo!(),
            TensorElementType::Float8E5M2 => todo!(),
            TensorElementType::Float8E5M2FNUZ => todo!(),
            TensorElementType::Uint4 => todo!(),
            TensorElementType::Int4 => todo!(),
            TensorElementType::String => todo!(),
            TensorElementType::Undefined => todo!(),
        _ => todo!(),
        };
    }
}

use hashbrown::HashMap;

use ort::{
    memory::Allocator,
    session::SessionOutputs,
    value::{DynTensor, DynValue, Shape, Tensor, TensorElementType, Value},
};
use tracing::info;

use crate::grpc::inference::DataType;

impl From<TensorElementType> for DataType {
    fn from(val: TensorElementType) -> Self {
        match val {
            TensorElementType::Float32 => DataType::TypeFp32,
            TensorElementType::Uint8 => DataType::TypeUint8,
            TensorElementType::Int8 => DataType::TypeInt8,
            TensorElementType::Uint16 => DataType::TypeUint16,
            TensorElementType::Int16 => DataType::TypeInt16,
            TensorElementType::Int32 => DataType::TypeInt32,
            TensorElementType::Int64 => DataType::TypeInt64,
            TensorElementType::String => DataType::TypeString,
            TensorElementType::Bool => DataType::TypeBool,
            TensorElementType::Float16 => DataType::TypeFp16,
            TensorElementType::Float64 => DataType::TypeFp64,
            TensorElementType::Uint32 => DataType::TypeUint32,
            TensorElementType::Uint64 => DataType::TypeUint64,
            TensorElementType::Bfloat16 => DataType::TypeBf16,
            TensorElementType::Complex64 => DataType::TypeInvalid,
            TensorElementType::Complex128 => DataType::TypeInvalid,
            TensorElementType::Float8E4M3FN => DataType::TypeInvalid,
            TensorElementType::Float8E4M3FNUZ => DataType::TypeInvalid,
            TensorElementType::Float8E5M2 => DataType::TypeInvalid,
            TensorElementType::Float8E5M2FNUZ => DataType::TypeInvalid,
            TensorElementType::Uint4 => DataType::TypeInvalid,
            TensorElementType::Int4 => DataType::TypeInvalid,
            TensorElementType::Undefined => DataType::TypeInvalid,
            _ => DataType::TypeInvalid,
        }
    }
}

impl DataType {
    pub fn to_metadata_string(self) -> String {
        match self {
            Self::TypeInvalid => String::from("INVALID"),
            Self::TypeBool => String::from("BOOL"),
            Self::TypeUint8 => String::from("UINT8"),
            Self::TypeUint16 => String::from("UINT16"),
            Self::TypeUint32 => String::from("UINT32"),
            Self::TypeUint64 => String::from("UINT64"),
            Self::TypeInt8 => String::from("INT8"),
            Self::TypeInt16 => String::from("INT16"),
            Self::TypeInt32 => String::from("INT32"),
            Self::TypeInt64 => String::from("INT64"),
            Self::TypeFp16 => String::from("FP16"),
            Self::TypeFp32 => String::from("FP32"),
            Self::TypeFp64 => String::from("FP64"),
            Self::TypeString => String::from("STRING"),
            Self::TypeBf16 => String::from("BF16"),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "INVALID" => Self::TypeInvalid,
            "BOOL" => Self::TypeBool,
            "INT8" => Self::TypeInt8,
            "UINT8" => Self::TypeUint8,
            "INT16" => Self::TypeInt16,
            "UINT16" => Self::TypeUint16,
            "INT32" => Self::TypeInt32,
            "UINT32" => Self::TypeUint32,
            "INT64" => Self::TypeInt64,
            "UINT64" => Self::TypeUint64,
            "FP16" => Self::TypeFp16,
            "BF16" => Self::TypeBf16,
            "FP32" => Self::TypeFp32,
            "FP64" => Self::TypeFp64,
            "STRING" => Self::TypeString,
            _ => Self::TypeInvalid,
        }
    }
}

pub fn dyntensor_from_bytes(data_type: DataType, shape: &[usize], bytes: &[u8]) -> DynTensor {
    match data_type {
        DataType::TypeInvalid => todo!(),
        DataType::TypeBool => {
            let data: Vec<bool> = bytes.iter().map(|d| *d != 0).collect();
            Tensor::from_array((shape, data)).unwrap().upcast()
        }
        DataType::TypeUint8 => Tensor::from_array((shape, bytes.to_vec()))
            .unwrap()
            .upcast(),
        DataType::TypeUint16 => {
            let data: &[u16] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeUint32 => {
            let data: &[u32] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeUint64 => {
            let data: &[u64] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeInt8 => {
            let data: &[i8] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeInt16 => {
            let data: &[i16] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeInt32 => {
            let data: &[i32] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeInt64 => {
            let data: &[i64] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeFp16 => todo!(),
        DataType::TypeFp32 => {
            let data: &[f32] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeFp64 => {
            let data: &[f64] = bytemuck::cast_slice(bytes);
            Tensor::from_array((shape, data.to_vec())).unwrap().upcast()
        }
        DataType::TypeString => todo!(),
        DataType::TypeBf16 => todo!(),
    }
}
