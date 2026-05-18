use std::{
    cell::{OnceCell, RefCell, UnsafeCell},
    cmp::min,
    collections::HashMap,
    sync::{
        Arc, atomic::{AtomicUsize, Ordering, fence}
    },
    time::{Duration, Instant},
};

use compio::runtime::time::sleep_until;
use ort::{
    memory::Allocator,
    session::SessionInputValue,
    value::{Outlet, Shape, TensorElementType, Value},
};
use smallvec::SmallVec;
use tokio::{sync::Notify};
use tracing::info;

use crate::{
    tensor::batched_tensor::{value_as_byte_slice, BatchableTensor, TensorBytes},
    tracing::ClientTrace,
};

const HALF_RANGE: usize = usize::MAX / 2;

struct ExecutorsInUseGuard<'a> {
    executors_in_use: &'a AtomicUsize,
}

impl<'a> ExecutorsInUseGuard<'a> {
    fn new(executors_in_use: &'a AtomicUsize) -> Self {
        executors_in_use.fetch_add(1, Ordering::AcqRel);
        Self { executors_in_use }
    }
}

impl Drop for ExecutorsInUseGuard<'_> {
    fn drop(&mut self) {
        self.executors_in_use.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct SuperTensorMetricsSnapshot {
    pub tail_index: usize,
    pub in_use_index: usize,
    pub head_index: usize,
    pub executors_in_use: usize,
}

/// Hold the timers to track where time is lost
pub struct ExecutionTrace {
    pub queue_start: OnceCell<Instant>,
    pub exec_start: OnceCell<Instant>,
    pub exec_end: OnceCell<Instant>,
}

struct BatchState {
    deadline: OnceCell<Instant>,
    output: Arc<OnceCell<Result<SessionValues, usize>>>,
    executor_notifier: Arc<Notify>,
    response_ready_notifier: Arc<Notify>,
    execution_trace: Arc<ExecutionTrace>
}

impl BatchState {
    fn new() -> Self {
        Self {
            deadline: OnceCell::new(),
            output: Arc::new(OnceCell::new()),
            executor_notifier: Arc::new(Notify::new()),
            response_ready_notifier: Arc::new(Notify::new()),
            execution_trace: Arc::new(ExecutionTrace { queue_start: OnceCell::new(), exec_start: OnceCell::new(), exec_end: OnceCell::new() })
        }
    }
}

struct DataTracker {
    // Dirty buffer should not be reused
    dirty: AtomicUsize,
    written_slots: AtomicUsize,
    state: UnsafeCell<BatchState>,
}

impl DataTracker {
    /// # Safety
    /// The caller is responsible for ensuring no race conditions on the borrowed state.
    unsafe fn unsafe_borrow(&self) -> &BatchState {
        &*self.state.get()
    }

    /// # Safety
    /// The caller is responsible for ensuring no race conditions on the borrowed state.
    unsafe fn unsafe_borrow_mut(&self) -> &mut BatchState {
        &mut *self.state.get()
    }
}

pub struct SessionValues {
    pub values: SmallVec<[Value; 4]>,
}

pub struct InferenceResponse {
    // 6 is arbitrary
    outputs: SmallVec<[WriteReservation; 6]>,
}

impl InferenceResponse {
    pub fn new() -> InferenceResponse {
        InferenceResponse {
            outputs: SmallVec::new(),
        }
    }

    pub fn push(&mut self, reservation: WriteReservation) {
        self.outputs.push(reservation);
    }

    pub async fn get_data(self, data: &mut Vec<Vec<u8>>) -> (Vec<(TensorElementType, Shape)>, Arc<ExecutionTrace>) {
        let mut maybe_metadatas: Option<(Vec<(TensorElementType, Shape)>, Arc<ExecutionTrace>)> = None;
        for output in self.outputs.into_iter() {
            let batch_metadatas = output.get_result(data).await;
            match &mut maybe_metadatas {
                Some(metadatas) => {
                    batch_metadatas
                    .0
                        .iter()
                        .enumerate()
                        .for_each(|(i, (_, shape))| {
                            // Add the shape of all subbatches to the response shape
                            metadatas.0[i].1[0] += shape[0]
                        });
                }
                None => maybe_metadatas = Some(batch_metadatas),
            };
        }
        maybe_metadatas.unwrap()
    }
}

// Ensure accounting when tasks are canceled
pub struct WriteReservation {
    output: Arc<OnceCell<Result<SessionValues, usize>>>,
    trace: Arc<ExecutionTrace>,
    response_ready_notifier: Arc<Notify>,
    start: usize,
    end: usize,
}

impl WriteReservation {
    fn new(tracker: &DataTracker, start: usize, end: usize) -> WriteReservation {
        let state = unsafe { tracker.unsafe_borrow() };
        let output = state.output.clone();
        let trace = state.execution_trace.clone();
        let response_ready_notifier = state.response_ready_notifier.clone();
        tracker
            .written_slots
            .fetch_add(end - start, Ordering::Release);
        WriteReservation {
            start,
            end,
            output,
            trace,
            response_ready_notifier,
        }
    }

    async fn get_result(self, data: &mut Vec<Vec<u8>>) -> (Vec<(TensorElementType, Shape)>, Arc<ExecutionTrace>) {
        loop {
            let notified = self.response_ready_notifier.notified();
            if let Some(result) = self.output.get() {
                let mut metadatas = Vec::new();
                match result {
                    Ok(output) => {
                        if data.is_empty() {
                            output.values.iter().for_each(|_| data.push(Vec::new()));
                        }
                        output
                            .values
                            .iter()
                            .enumerate()
                            .for_each(|(i, batch_tensor)| {
                                data[i].extend_from_slice(value_as_byte_slice(
                                    batch_tensor,
                                    self.start,
                                    self.end,
                                ));
                                metadatas.push((
                                    *batch_tensor.data_type(),
                                    batch_tensor.shape().clone(),
                                ));
                            });
                    }
                    Err(_) => todo!(),
                }
                return (metadatas, self.trace.clone());
            }
            notified.await;
        }
    }
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Useful abstraction to get the correct position within the buffer
/// Maybe it should be a macro?
#[derive(Clone)]
pub struct RingBufferIndex<'a> {
    /// Absolute index on the ringbuffer
    index: usize,
    atomic_ref: &'a PaddedAtomic,
}

impl<'a> RingBufferIndex<'a> {
    fn new(index: usize, atomic_ref: &PaddedAtomic) -> RingBufferIndex<'_> {
        RingBufferIndex { index, atomic_ref }
    }
    /// Get the absolute index, return the raw index, useful for operating on the ring
    /// itself
    fn as_absolute_index(&self) -> usize {
        self.index
    }

    /// Return the batch_slot 0 of the next batch in absolute index
    fn as_absolute_batch_higher_bound(&self) -> usize {
        self.as_absolute_batch_lower_bound()
            .wrapping_add(self.atomic_ref.batch_size)
    }
    /// Return the batch_slot 0 of the current batch in absolute index
    fn as_absolute_batch_lower_bound(&self) -> usize {
        self.index.wrapping_sub(self.as_batch_slot_id())
    }

    /// Get the batch id: the idx of the batch that is concerned by this index
    fn as_batch_id(&self) -> usize {
        // println!("{}: {} / {} -> {}", self.index , self.index & self.ring_buffer.ring_mask, self.ring_buffer.batch_size, (self.index & self.ring_buffer.ring_mask) / self.ring_buffer.batch_size);
        (self.index & self.atomic_ref.ring_mask) / self.atomic_ref.batch_size
    }

    /// Get the batch slot id: the idx of the slot relative to the batch id
    fn as_batch_slot_id(&self) -> usize {
        self.index & self.atomic_ref.batch_mask
    }

    pub fn wrapping_sub(&self, other: &RingBufferIndex) -> usize {
        self.index.wrapping_sub(other.index)
    }

    pub fn wrapping_add(&self, rhs: usize) -> usize {
        self.index.wrapping_add(rhs)
    }
}

unsafe impl Send for SuperTensorBuffer {}
unsafe impl Sync for SuperTensorBuffer {}

/// To avoid false sharing, padding the atomic with the worst case cache line
/// This also helps operating safely around the heads/tail
#[repr(align(128))]
struct PaddedAtomic {
    value: AtomicUsize,
    // Mask for the full ring: batch_size * capacity
    ring_mask: usize,
    // Mask to get position within the batch
    batch_mask: usize,
    batch_size: usize,
}

impl PaddedAtomic {
    pub fn load(&self, order: Ordering) -> RingBufferIndex<'_> {
        RingBufferIndex::new(self.value.load(order), self)
    }

    pub fn compare_exchange_weak(&self, current: usize, new: usize) -> Result<usize, usize> {
        self.value
            .compare_exchange_weak(current, new, Ordering::SeqCst, Ordering::Relaxed)
    }
}

pub struct SuperTensorBuffer {
    // 6 is taken from STACK_SESSION_INPUTS of ORT
    input_tensors: Vec<SmallVec<[UnsafeCell<BatchableTensor>; 6]>>,
    trackers: Vec<DataTracker>,
    batch_size: usize,
    capacity: usize,
    head: PaddedAtomic,
    executor_head: PaddedAtomic,
    tail: PaddedAtomic,
    executors_in_use: AtomicUsize,
    // The mask is used for efficient modulo arithmetic to wrap around the ring buffer.
    // The Problem:
    // When you have a ring buffer with n buffers, you need to convert a continuously incrementing index (0, 1, 2, 3, 4, 5...) into a buffer position (0, 1, 2, 3, 0, 1, 2, 3...).
    // Normally you'd use: buffer_idx = head % buffer_count
    // The Optimization:
    // If buffer_count is a power of 2 (e.g., 4, 8, 16), you can replace the slow modulo operation with a fast bitwise AND:
    // If buffer_count = 4 (which is 2^2)
    // mask = 4 - 1 = 3 = 0b0011

    // head = 0  → 0 & 0b0011 = 0
    // head = 1  → 1 & 0b0011 = 1
    // head = 2  → 2 & 0b0011 = 2
    // head = 3  → 3 & 0b0011 = 3
    // head = 4  → 4 & 0b0011 = 0  // Wraps around!
    // head = 5  → 5 & 0b0011 = 1
    // head = 6  → 6 & 0b0011 = 2
    // Mask for the full ring: batch_size * capacity
    ring_mask: usize,
    // Mask to get position within the batch
    batch_mask: usize,
    executor_full_notifier: Notify,
    infer_full_notifier: Notify,
}

impl SuperTensorBuffer {
    pub fn new(
        capacity: usize,
        batch_size: usize,
        inputs: &Vec<&Outlet>,
        allocator: &Allocator,
    ) -> Result<SuperTensorBuffer, ()> {
        Self::new_with_overrides(capacity, batch_size, inputs, allocator, &HashMap::new())
    }

    pub fn new_with_overrides(
        capacity: usize,
        batch_size: usize,
        inputs: &Vec<&Outlet>,
        allocator: &Allocator,
        shape_overrides: &HashMap<String, Vec<i64>>,
    ) -> Result<SuperTensorBuffer, ()> {
        {
            if !is_power_of_two(capacity) {
                panic!("Capacity is not power of 2: {capacity}")
            }
            if !is_power_of_two(batch_size) {
                panic!("Batch size is not power of 2: {capacity}")
            }
            let mut input_tensors = Vec::with_capacity(capacity);

            for _ in 0..capacity {
                let mut batched_input_tensors = SmallVec::with_capacity(capacity);
                inputs.iter().for_each(|input| {
                    let (ty, shape) = match &input.dtype() {
                        ort::value::ValueType::Tensor {
                            ty,
                            shape,
                            dimension_symbols: _,
                        } => (ty, shape),
                        ort::value::ValueType::Sequence(_value_type) => todo!(),
                        ort::value::ValueType::Map { key: _, value: _ } => todo!(),
                        ort::value::ValueType::Optional(_value_type) => todo!(),
                    };

                    // Apply shape override if specified for this input
                    let effective_shape =
                        if let Some(override_shape) = shape_overrides.get(input.name()) {
                            // Create a new shape with overridden dimensions (except dim 0 which is batch)
                            let mut new_shape = shape.clone();
                            for (i, override_dim) in override_shape.iter().enumerate() {
                                if i > 0 && i < new_shape.len() {
                                    new_shape[i] = *override_dim;
                                }
                            }
                            new_shape
                        } else {
                            shape.clone()
                        };

                    batched_input_tensors.push(UnsafeCell::from(BatchableTensor::new(
                        *ty,
                        &effective_shape,
                        batch_size,
                        allocator,
                    )));
                });
                input_tensors.push(batched_input_tensors);
            }

            let mut trackers = Vec::with_capacity(capacity);
            for _i in 0..capacity {
                trackers.push(DataTracker {
                    dirty: AtomicUsize::new(0),
                    written_slots: AtomicUsize::new(0),
                    state: UnsafeCell::new(BatchState::new()),
                });
            }
            let ring_mask = (capacity * batch_size) - 1;
            let batch_mask = batch_size - 1;

            Ok(SuperTensorBuffer {
                input_tensors,
                trackers,
                batch_size,
                head: PaddedAtomic {
                    value: AtomicUsize::new(0),
                    ring_mask,
                    batch_mask,
                    batch_size,
                },
                executor_head: PaddedAtomic {
                    value: AtomicUsize::new(0),
                    ring_mask,
                    batch_mask,
                    batch_size,
                },
                tail: PaddedAtomic {
                    value: AtomicUsize::new(0),
                    ring_mask,
                    batch_mask,
                    batch_size,
                },
                executors_in_use: AtomicUsize::new(0),
                capacity,
                ring_mask,
                batch_mask,
                executor_full_notifier: Notify::new(),
                infer_full_notifier: Notify::new(),
            })
        }
    }

    pub async fn infer(
        &self,
        data: &[TensorBytes<'_>],
        trace: &mut ClientTrace,
    ) -> Result<InferenceResponse, usize> {
        loop {
            let mut current_head = self.head.load(Ordering::Relaxed);
            let current_tail = self.tail.load(Ordering::Acquire);

            // Check if the ring is full
            // TODO: make check batch compatible
            if current_head.wrapping_sub(&current_tail) >= self.capacity * self.batch_size {
                // println!("Buffer full, yielding");
                self.infer_full_notifier.notified().await;
                continue;
            }
            // TODO: validate before that:
            // - data and shape is not empty
            // - all inputs have the same batch size
            let mut client_batch_size = data[0].shape[0] as usize;

            match self.head.compare_exchange_weak(
                current_head.as_absolute_index(),
                current_head.wrapping_add(client_batch_size),
            ) {
                Ok(_) => {
                    let mut input_start = 0;
                    let mut response = InferenceResponse::new();
                    let batch_id = current_head.as_absolute_index();

                    loop {
                        let capacity_in_current_batch = current_head
                            .as_absolute_batch_higher_bound()
                            .wrapping_sub(current_head.as_absolute_index());

                        let to_write = min(client_batch_size, capacity_in_current_batch);

                        let input_end = input_start + to_write;

                        trace.record_inference_in_queue();
                        // println!(
                        //     "{batch_id} inserting {to_write} tensors into {} ({input_start} -> {input_end}) ({to_write} = min({client_batch_size},{capacity_in_current_batch}) head {} hb:{}",
                        //     current_head.as_batch_id(),
                        //     current_head.as_absolute_index(),
                        //     current_head.as_absolute_batch_higher_bound(),
                        // );
                        let write_reservation = self.insert_tensors_at(
                            current_head.clone(),
                            input_start,
                            input_end,
                            data,
                        );
                        response.push(write_reservation);
                        input_start = input_end;
                        client_batch_size -= to_write;

                        if client_batch_size == 0 {
                            break;
                        }

                        current_head = RingBufferIndex {
                            index: current_head.as_absolute_batch_higher_bound(),
                            atomic_ref: &self.executor_head,
                        };
                    }
                    // TODO: process all the writes from different buffers
                    return Ok(response);
                }
                Err(_) => {
                    // Another producer won the race, retry the check
                    continue;
                }
            }
        }
    }

    fn insert_tensors_at(
        &self,
        idx: RingBufferIndex,
        data_start: usize,
        data_end: usize,
        data: &[TensorBytes<'_>],
    ) -> WriteReservation {
        // batch_slot is the index of the slot within a batch
        // Used to track if it's the first or last reservation on the batch
        // println!(
        //     "Inserting in {} at -> {}",
        //     idx.as_batch_id(),
        //     idx.as_batch_slot_id()
        // );
        let batch_slot = idx.as_batch_slot_id();
        let batch_id = idx.as_batch_id();
        self.input_tensors[batch_id]
            .iter()
            .enumerate()
            .for_each(|(i, tensors)| {
                unsafe {
                    // Isolation of portions of the vector is guaranteed by reserved_slots atomics
                    let slice = data[i].slice_dim0(data_start, data_end);
                    let batch_tensor = (&mut *tensors.get());
                    // println!("Slicing of len {} -> {} between {data_start} and {data_end}", slice.len(), batch_tensor.inner_tensor.shape().num_elements() );
                    batch_tensor.copy_at_from_bytes(batch_slot, slice)
                }
            });
        if batch_slot == 0 {
            let tracker = self.trackers.get(idx.as_batch_id()).unwrap();
            let now = Instant::now();
            let deadline = now + Duration::from_millis(2);
            tracker.dirty.store(1, Ordering::Relaxed);
            // println!("{}> writing deadline", idx.as_batch_id());
            let state_borrow = unsafe { tracker.unsafe_borrow() };
            state_borrow.deadline.set(deadline).unwrap();
            state_borrow.execution_trace.queue_start.set(now).unwrap();
        }
        fence(Ordering::Release);

        let tracker = self.trackers.get(idx.as_batch_id()).unwrap();
        let end = batch_slot + (data_end - data_start);
        let reservation = WriteReservation::new(tracker, batch_slot, end);
        if batch_slot == 0 || end == self.batch_size {
            unsafe { tracker.unsafe_borrow() }.executor_notifier.notify_one();
        }
        reservation
    }

    pub async fn execute_on_batch<F>(&self, _id: String, f: F) -> usize
    where
        F: AsyncFnOnce(&[SessionInputValue]) -> SessionValues,
    {
        let mut current_executor_idx; // Defined as RingBufferIndex
        loop {
            current_executor_idx = self.executor_head.load(Ordering::Acquire);
            let current_tail = self.tail.load(Ordering::Acquire);

            // Check if the ring is full: (executor_head - tail) >= total_capacity
            if current_executor_idx.wrapping_sub(&current_tail) >= self.capacity * self.batch_size {
                self.executor_full_notifier.notified().await;
                continue;
            }

            match self.executor_head.compare_exchange_weak(
                current_executor_idx.as_absolute_index(),
                current_executor_idx
                    .as_absolute_index()
                    .wrapping_add(self.batch_size),
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        let tracker = self
            .trackers
            .get(current_executor_idx.as_batch_id())
            .unwrap();
        let executor_notifier = unsafe { tracker.unsafe_borrow() }.executor_notifier.clone();
        executor_notifier.notified().await;

        let notified_full = executor_notifier.notified();
        let head = self.head.load(Ordering::Acquire);

        // Wrapping safe: head < current_executor_idx.as_absolute_batch_higher_bound()
        // Logic: Distance from head to higher_bound is > 0 and < HALF_RANGE
        let dist_to_higher = current_executor_idx
            .as_absolute_batch_higher_bound()
            .wrapping_sub(head.as_absolute_index());
        if dist_to_higher > 0 && dist_to_higher < HALF_RANGE {
            let maybe_deadline = unsafe { tracker.unsafe_borrow() }.deadline.get().copied();
            if let Some(deadline) = maybe_deadline {
                // let _now = Instant::now();
                tokio::select! {
                    _ = sleep_until(deadline) => {
                        // println!("awaken by sleep after {:?}", now.elapsed())
                    },
                    _ = notified_full => {},
                }
            }
        }

        self.seal_current_batch(tracker, &current_executor_idx);
        unsafe { tracker.unsafe_borrow() }.execution_trace.exec_start.set(Instant::now()).unwrap();
        let batch_items = tracker.written_slots.load(Ordering::Acquire);
        let _executors_in_use_guard = ExecutorsInUseGuard::new(&self.executors_in_use);
        self.execute_current_batch(f, tracker, &current_executor_idx)
            .await;

        self.reset_batch(tracker);
        self.move_tail_to_next_non_dirty_buffer();
        batch_items
    }

    pub async fn warmup<F>(&self, f: F)
    where
        F: AsyncFnOnce(&[SessionInputValue]) -> SessionValues,
    {
        let batch_start = self.head.value.fetch_add(self.batch_size, Ordering::AcqRel);
        self.executor_head
            .value
            .fetch_add(self.batch_size, Ordering::AcqRel);

        let idx = RingBufferIndex::new(batch_start, &self.head);
        let tracker = &self.trackers[idx.as_batch_id()];
        tracker.dirty.store(1, Ordering::Release);

        let _guard = ExecutorsInUseGuard::new(&self.executors_in_use);
        let input = self.get_data_view(&idx);
        let _ = f(&input).await;

        self.reset_batch(tracker);
        self.move_tail_to_next_non_dirty_buffer();
    }

    pub fn metrics_snapshot(&self) -> SuperTensorMetricsSnapshot {
        SuperTensorMetricsSnapshot {
            tail_index: self.tail.load(Ordering::Acquire).as_absolute_index(),
            in_use_index: self
                .executor_head
                .load(Ordering::Acquire)
                .as_absolute_index(),
            head_index: self.head.load(Ordering::Acquire).as_absolute_index(),
            executors_in_use: self.executors_in_use.load(Ordering::Acquire),
        }
    }

    async fn execute_current_batch<F>(
        &self,
        f: F,
        tracker: &DataTracker,
        current_executor_idx: &RingBufferIndex<'_>,
    ) where
        F: AsyncFnOnce(&[SessionInputValue]) -> SessionValues,
    {
        let input = self.get_data_view(current_executor_idx);
        let result = f(&input).await;
        // Put the results in the arc output, to be dispatched to consumers
        let state = unsafe { tracker.unsafe_borrow() };
        state.execution_trace.exec_end.set(Instant::now()).unwrap();
        let cell_status = state.output.set(Ok(result));
        if cell_status.is_err() {
            panic!("Failed to set cell for the output of inference")
        }
        let notifier = state.response_ready_notifier.clone();
        notifier.notify_waiters();
    }

    fn reset_batch(&self, tracker: &DataTracker) {
        *unsafe { tracker.unsafe_borrow_mut() } = BatchState::new();
        tracker.written_slots.store(0, Ordering::Relaxed);
        tracker.dirty.store(2, Ordering::Relaxed);
    }
    fn move_tail_to_next_non_dirty_buffer(&self) {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let head = self.head.load(Ordering::Acquire);

            // Wrapping safe: tail.higher_bound >= head
            if tail
                .as_absolute_batch_higher_bound()
                .wrapping_sub(head.as_absolute_index())
                < HALF_RANGE
            {
                break;
            }

            let dirty_state = &self.trackers[tail.as_batch_id()].dirty;
            let dirty = dirty_state.load(Ordering::Acquire);
            if dirty == 2
                && dirty_state
                    .compare_exchange_weak(dirty, 0, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                && self
                    .tail
                    .compare_exchange_weak(
                        tail.as_absolute_index(),
                        tail.as_absolute_batch_higher_bound(),
                    )
                    .is_ok()
            {
                self.executor_full_notifier.notify_waiters();
                self.infer_full_notifier.notify_waiters();
                continue;
            }
            break;
        }
    }
    pub fn get_data_view(
        &self,
        current_executor_idx: &RingBufferIndex,
    ) -> SmallVec<[SessionInputValue<'_>; 6]> {
        unsafe {
            let mut all_inputs = SmallVec::new();
            let batch_id = current_executor_idx.as_batch_id();

            self.input_tensors[batch_id]
                .iter()
                .for_each(|batch_tensors| {
                    all_inputs.push((*batch_tensors.get()).inner_tensor.view().into());
                });
            all_inputs
        }
    }

    fn seal_current_batch(&self, tracker: &DataTracker, current_executor_idx: &RingBufferIndex) {
        let mut remaining_open_slots;
        loop {
            remaining_open_slots = 0;
            let head = self.head.load(Ordering::Acquire);

            // head must be ahead of or equal to the lower bound of the batch we are sealing
            assert!(
                head.as_absolute_index()
                    .wrapping_sub(current_executor_idx.as_absolute_batch_lower_bound())
                    < HALF_RANGE,
                "Batch is being sealed but not even written yet"
            );

            // Check if head is still within this batch: head < higher_bound
            let dist_to_higher = current_executor_idx
                .as_absolute_batch_higher_bound()
                .wrapping_sub(head.as_absolute_index());
            if dist_to_higher > 0 && dist_to_higher < HALF_RANGE {
                remaining_open_slots = dist_to_higher;
                if self
                    .head
                    .compare_exchange_weak(
                        head.as_absolute_index(),
                        head.as_absolute_index().wrapping_add(remaining_open_slots),
                    )
                    .is_ok()
                {
                    break;
                }
                continue;
            }
            break;
        }

        loop {
            let written_slots = tracker.written_slots.load(Ordering::Acquire);
            // println!("{}> remaining_open_slots open_slots:{remaining_open_slots}, written:{written_slots}; expected_written: {} - {}", current_executor_idx.as_batch_id(), self.batch_size , remaining_open_slots);
            let expected_written_slots = self.batch_size - remaining_open_slots;
            if written_slots == expected_written_slots {
                return;
            } else if written_slots < expected_written_slots {
                // Some writes are in progress, spinning until they are visible
                continue;
            } else {
                panic!(
                    "{}> written_slots ({}) is higher than expected_written_slots({})",
                    current_executor_idx.as_batch_id(),
                    written_slots,
                    expected_written_slots
                )
            }
        }
    }
}
