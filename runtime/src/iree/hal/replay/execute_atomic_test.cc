// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "execute_operation.h"
#include "execute_state.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_hal_replay_object_id_t kDeviceId = 1;
static constexpr iree_hal_replay_object_id_t kBufferId = 2;
static constexpr iree_hal_replay_object_id_t kCommandBufferId = 3;
static constexpr iree_hal_replay_object_id_t kWaitSemaphoreId = 4;
static constexpr iree_hal_replay_object_id_t kSignalSemaphoreId = 5;

typedef enum AtomicInvocationKind {
  kAtomicInvocationNone = 0,
  kAtomicInvocationWait,
  kAtomicInvocationStore,
  kAtomicInvocationRmw,
} AtomicInvocationKind;

typedef struct CapturedSemaphoreTimepoint {
  // Semaphore passed to the backend.
  iree_hal_semaphore_t* semaphore;
  // Timeline value passed to the backend.
  uint64_t value;
} CapturedSemaphoreTimepoint;

typedef struct AtomicInvocation {
  // Kind of atomic operation invoked.
  AtomicInvocationKind kind;
  // Source stages passed to a command buffer operation.
  iree_hal_execution_stage_t source_stage_mask;
  // Target stages passed to a command buffer operation.
  iree_hal_execution_stage_t target_stage_mask;
  // Queue affinity passed to a queue operation.
  iree_hal_queue_affinity_t queue_affinity;
  // Number of wait semaphore timepoints passed to a queue operation.
  iree_host_size_t wait_semaphore_count;
  // First wait semaphore timepoint passed to a queue operation.
  CapturedSemaphoreTimepoint wait_timepoint;
  // Number of signal semaphore timepoints passed to a queue operation.
  iree_host_size_t signal_semaphore_count;
  // First signal semaphore timepoint passed to a queue operation.
  CapturedSemaphoreTimepoint signal_timepoint;
  // Atomic target reference passed to the backend.
  iree_hal_buffer_ref_t target_ref;
  // Atomic wait parameters passed to the backend.
  iree_hal_atomic_wait_params_t wait_params;
  // Atomic store parameters passed to the backend.
  iree_hal_atomic_store_params_t store_params;
  // Atomic read-modify-write parameters passed to the backend.
  iree_hal_atomic_rmw_params_t rmw_params;
} AtomicInvocation;

typedef struct CapturingCommandBuffer {
  // Base command buffer dispatched through the public HAL API.
  iree_hal_command_buffer_t base;
  // Most recent atomic backend invocation.
  AtomicInvocation invocation;
  // Total number of atomic backend invocations.
  iree_host_size_t invocation_count;
} CapturingCommandBuffer;

typedef struct CapturingDevice {
  // Base resource dispatched through the public HAL API.
  iree_hal_resource_t resource;
  // Borrowed concrete device providing queue enumeration and synchronization.
  iree_hal_device_t* task_device;
  // Most recent atomic backend invocation.
  AtomicInvocation invocation;
  // Total number of atomic backend invocations.
  iree_host_size_t invocation_count;
} CapturingDevice;

static CapturingCommandBuffer* CastCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  return reinterpret_cast<CapturingCommandBuffer*>(base_command_buffer);
}

static CapturingDevice* CastDevice(iree_hal_device_t* base_device) {
  return reinterpret_cast<CapturingDevice*>(base_device);
}

static void CapturingCommandBufferDestroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
}

static iree_status_t CapturingCommandBufferBegin(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferEnd(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  return iree_ok_status();
}

static AtomicInvocation* BeginCommandBufferInvocation(
    iree_hal_command_buffer_t* base_command_buffer, AtomicInvocationKind kind,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref) {
  CapturingCommandBuffer* command_buffer =
      CastCommandBuffer(base_command_buffer);
  AtomicInvocation* invocation = &command_buffer->invocation;
  memset(invocation, 0, sizeof(*invocation));
  invocation->kind = kind;
  invocation->source_stage_mask = source_stage_mask;
  invocation->target_stage_mask = target_stage_mask;
  invocation->target_ref = target_ref;
  ++command_buffer->invocation_count;
  return invocation;
}

static iree_status_t CapturingCommandBufferAtomicWait(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  AtomicInvocation* invocation = BeginCommandBufferInvocation(
      base_command_buffer, kAtomicInvocationWait, source_stage_mask,
      target_stage_mask, target_ref);
  invocation->wait_params = params;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferAtomicStore(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  AtomicInvocation* invocation = BeginCommandBufferInvocation(
      base_command_buffer, kAtomicInvocationStore, source_stage_mask,
      target_stage_mask, target_ref);
  invocation->store_params = params;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferAtomicRmw(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  AtomicInvocation* invocation = BeginCommandBufferInvocation(
      base_command_buffer, kAtomicInvocationRmw, source_stage_mask,
      target_stage_mask, target_ref);
  invocation->rmw_params = params;
  return iree_ok_status();
}

static void CapturingDeviceDestroy(iree_hal_device_t* base_device) {
  (void)base_device;
}

static const iree_hal_device_spec_t* CapturingDeviceSpec(
    iree_hal_device_t* base_device) {
  return iree_hal_device_spec(CastDevice(base_device)->task_device);
}

static const iree_hal_queue_family_t* CapturingDeviceQueueFamily(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal) {
  return iree_hal_device_queue_family(CastDevice(base_device)->task_device,
                                      family_ordinal);
}

static iree_hal_queue_t* CapturingDeviceQueue(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal) {
  return iree_hal_device_queue(CastDevice(base_device)->task_device,
                               family_ordinal, queue_ordinal);
}

static AtomicInvocation* BeginDeviceInvocation(
    iree_hal_device_t* base_device, AtomicInvocationKind kind,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_width_t width) {
  CapturingDevice* device = CastDevice(base_device);
  AtomicInvocation* invocation = &device->invocation;
  memset(invocation, 0, sizeof(*invocation));
  invocation->kind = kind;
  invocation->queue_affinity = queue_affinity;
  invocation->wait_semaphore_count = wait_semaphore_list.count;
  if (wait_semaphore_list.count != 0) {
    invocation->wait_timepoint.semaphore = wait_semaphore_list.semaphores[0];
    invocation->wait_timepoint.value = wait_semaphore_list.payload_values[0];
  }
  invocation->signal_semaphore_count = signal_semaphore_list.count;
  if (signal_semaphore_list.count != 0) {
    invocation->signal_timepoint.semaphore =
        signal_semaphore_list.semaphores[0];
    invocation->signal_timepoint.value =
        signal_semaphore_list.payload_values[0];
  }
  invocation->target_ref = iree_hal_make_buffer_ref(
      target_buffer, target_offset, iree_hal_atomic_width_byte_count(width));
  ++device->invocation_count;
  return invocation;
}

static iree_status_t CapturingDeviceAtomicWait(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  AtomicInvocation* invocation = BeginDeviceInvocation(
      base_device, kAtomicInvocationWait, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, target_buffer, target_offset, params.width);
  invocation->wait_params = params;
  return iree_hal_semaphore_list_signal(signal_semaphore_list,
                                        /*frontier=*/nullptr);
}

static iree_status_t CapturingDeviceAtomicStore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  AtomicInvocation* invocation = BeginDeviceInvocation(
      base_device, kAtomicInvocationStore, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, target_buffer, target_offset, params.width);
  invocation->store_params = params;
  return iree_hal_semaphore_list_signal(signal_semaphore_list,
                                        /*frontier=*/nullptr);
}

static iree_status_t CapturingDeviceAtomicRmw(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  AtomicInvocation* invocation = BeginDeviceInvocation(
      base_device, kAtomicInvocationRmw, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, target_buffer, target_offset, params.width);
  invocation->rmw_params = params;
  return iree_hal_semaphore_list_signal(signal_semaphore_list,
                                        /*frontier=*/nullptr);
}

static iree_hal_device_t* CreateTaskDevice() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

class OperationRecord {
 public:
  template <typename Payload>
  void Reset(
      iree_hal_replay_operation_code_t operation_code,
      iree_hal_replay_payload_type_t payload_type,
      iree_hal_replay_object_id_t object_id, const Payload& payload,
      std::initializer_list<iree_hal_replay_semaphore_timepoint_payload_t>
          timepoints = {}) {
    payload_storage_.assign(
        sizeof(payload) +
            timepoints.size() *
                sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
        0);
    memcpy(payload_storage_.data(), &payload, sizeof(payload));
    iree_host_size_t offset = sizeof(payload);
    for (const auto& timepoint : timepoints) {
      memcpy(payload_storage_.data() + offset, &timepoint, sizeof(timepoint));
      offset += sizeof(timepoint);
    }
    memset(&record_, 0, sizeof(record_));
    record_.header.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
    record_.header.operation_code = operation_code;
    record_.header.payload_type = payload_type;
    record_.header.object_id = object_id;
    record_.header.status_code = IREE_STATUS_OK;
    RefreshPayloadSpan();
  }

  void AppendPayloadByte(uint8_t value) {
    payload_storage_.push_back(value);
    RefreshPayloadSpan();
  }

  const iree_hal_replay_file_record_t* get() const { return &record_; }

 private:
  void RefreshPayloadSpan() {
    record_.payload = iree_make_const_byte_span(payload_storage_.data(),
                                                payload_storage_.size());
  }

  // Owned bytes referenced by |record_|.
  std::vector<uint8_t> payload_storage_;
  // Replay record projected over |payload_storage_|.
  iree_hal_replay_file_record_t record_ = {};
};

class ReplayAtomicExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    task_device_ = CreateTaskDevice();

    const iree_hal_buffer_params_t buffer_params = {
        /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
        /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    };
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(task_device_), buffer_params, 64,
        &target_buffer_));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        task_device_,
        /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(0),
        /*initial_value=*/7, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &wait_semaphore_));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        task_device_,
        /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(0),
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &signal_semaphore_));

    const iree_host_size_t validation_state_size =
        iree_hal_command_buffer_validation_state_size(
            /*mode=*/0, /*binding_capacity=*/1);
    if (validation_state_size != 0) {
      IREE_ASSERT_OK(iree_allocator_malloc(
          iree_allocator_system(), validation_state_size, &validation_state_));
      memset(validation_state_, 0, validation_state_size);
    }
    iree_hal_queue_t* task_queue = iree_hal_device_queue(
        task_device_, /*family_ordinal=*/0, /*queue_ordinal=*/0);
    ASSERT_NE(nullptr, task_queue);
    command_buffer_vtable_.destroy = CapturingCommandBufferDestroy;
    command_buffer_vtable_.begin = CapturingCommandBufferBegin;
    command_buffer_vtable_.end = CapturingCommandBufferEnd;
    command_buffer_vtable_.atomic_wait = CapturingCommandBufferAtomicWait;
    command_buffer_vtable_.atomic_store = CapturingCommandBufferAtomicStore;
    command_buffer_vtable_.atomic_rmw = CapturingCommandBufferAtomicRmw;
    iree_hal_command_buffer_initialize(
        iree_hal_device_allocator(task_device_),
        iree_hal_queue_family(task_queue), /*mode=*/0,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC,
        /*binding_capacity=*/1, validation_state_, &command_buffer_vtable_,
        &command_buffer_.base);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer_.base));

    device_vtable_.destroy = CapturingDeviceDestroy;
    device_vtable_.device_spec = CapturingDeviceSpec;
    device_vtable_.queue_family = CapturingDeviceQueueFamily;
    device_vtable_.queue = CapturingDeviceQueue;
    device_vtable_.queue_atomic_wait = CapturingDeviceAtomicWait;
    device_vtable_.queue_atomic_store = CapturingDeviceAtomicStore;
    device_vtable_.queue_atomic_rmw = CapturingDeviceAtomicRmw;
    device_.task_device = task_device_;
    iree_hal_resource_initialize(&device_vtable_, &device_.resource);
    execute_options_ = iree_hal_replay_execute_options_default();
    IREE_ASSERT_OK(iree_hal_replay_executor_initialize(
        &executor_, iree_const_byte_span_empty(), /*object_capacity=*/8,
        /*device_group=*/nullptr, &execute_options_, iree_allocator_system()));
    iree_hal_replay_object_entry_t device_entry = {};
    device_entry.value.device = reinterpret_cast<iree_hal_device_t*>(&device_);
    StoreObject(kDeviceId, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE, device_entry);
    iree_hal_replay_object_entry_t buffer_entry = {};
    buffer_entry.value.buffer = target_buffer_;
    StoreObject(kBufferId, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, buffer_entry);
    iree_hal_replay_object_entry_t command_buffer_entry = {};
    command_buffer_entry.value.command_buffer = &command_buffer_.base;
    StoreObject(kCommandBufferId, IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
                command_buffer_entry);
    iree_hal_replay_object_entry_t wait_semaphore_entry = {};
    wait_semaphore_entry.value.semaphore = wait_semaphore_;
    StoreObject(kWaitSemaphoreId, IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE,
                wait_semaphore_entry);
    iree_hal_replay_object_entry_t signal_semaphore_entry = {};
    signal_semaphore_entry.value.semaphore = signal_semaphore_;
    StoreObject(kSignalSemaphoreId, IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE,
                signal_semaphore_entry);
  }

  void TearDown() override {
    iree_hal_replay_executor_deinitialize(&executor_);
    iree_allocator_free(iree_allocator_system(), validation_state_);
    iree_hal_device_release(task_device_);
  }

  void StoreObject(iree_hal_replay_object_id_t object_id,
                   iree_hal_replay_object_type_t object_type,
                   iree_hal_replay_object_entry_t entry) {
    IREE_ASSERT_OK(iree_hal_replay_executor_store(&executor_, object_id,
                                                  object_type, entry));
  }

  iree_status_t Replay(const OperationRecord& record) {
    return iree_hal_replay_executor_replay_operation(&executor_, record.get());
  }

  static iree_hal_replay_buffer_ref_payload_t DirectTarget(uint64_t offset,
                                                           uint64_t length) {
    return {
        /*.buffer_id=*/kBufferId,
        /*.offset=*/offset,
        /*.length=*/length,
        /*.buffer_slot=*/0,
        /*.reserved0=*/0,
    };
  }

  static iree_hal_replay_semaphore_timepoint_payload_t WaitTimepoint() {
    return {/*.semaphore_id=*/kWaitSemaphoreId, /*.value=*/7};
  }

  static iree_hal_replay_semaphore_timepoint_payload_t SignalTimepoint(
      uint64_t value) {
    return {/*.semaphore_id=*/kSignalSemaphoreId, /*.value=*/value};
  }

  // Task-driver device owning the concrete test buffer and semaphores.
  iree_hal_device_t* task_device_ = nullptr;
  // Borrowed target buffer transferred to |executor_|.
  iree_hal_buffer_t* target_buffer_ = nullptr;
  // Borrowed wait semaphore transferred to |executor_|.
  iree_hal_semaphore_t* wait_semaphore_ = nullptr;
  // Borrowed signal semaphore transferred to |executor_|.
  iree_hal_semaphore_t* signal_semaphore_ = nullptr;
  // Validation storage backing |command_buffer_|.
  void* validation_state_ = nullptr;
  // Vtable implementing the command buffer methods under test.
  iree_hal_command_buffer_vtable_t command_buffer_vtable_ = {};
  // Backend command buffer capturing decoded operations.
  CapturingCommandBuffer command_buffer_ = {};
  // Vtable implementing the device methods under test.
  iree_hal_device_vtable_t device_vtable_ = {};
  // Backend device capturing decoded queue operations.
  CapturingDevice device_ = {};
  // Options retained by |executor_|.
  iree_hal_replay_execute_options_t execute_options_ = {};
  // Executor under test.
  iree_hal_replay_executor_t executor_ = {};
};

TEST_F(ReplayAtomicExecutionTest, ReplaysCommandBufferOperations) {
  OperationRecord record;

  iree_hal_replay_command_buffer_atomic_wait_payload_t wait_payload = {};
  wait_payload.target_ref = DirectTarget(/*offset=*/8, /*length=*/8);
  wait_payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  wait_payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_ATOMIC;
  wait_payload.params.value = UINT64_C(0x1020304050607080);
  wait_payload.params.mask = UINT64_C(0xFFEEDDCCBBAA9988);
  wait_payload.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  wait_payload.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  wait_payload.params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_WAIT,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT,
               kCommandBufferId, wait_payload);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(1u, command_buffer_.invocation_count);
  EXPECT_EQ(kAtomicInvocationWait, command_buffer_.invocation.kind);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_DISPATCH,
            command_buffer_.invocation.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_ATOMIC,
            command_buffer_.invocation.target_stage_mask);
  EXPECT_EQ(target_buffer_, command_buffer_.invocation.target_ref.buffer);
  EXPECT_EQ(8u, command_buffer_.invocation.target_ref.offset);
  EXPECT_EQ(8u, command_buffer_.invocation.target_ref.length);
  EXPECT_EQ(wait_payload.params.value,
            command_buffer_.invocation.wait_params.value);
  EXPECT_EQ(wait_payload.params.mask,
            command_buffer_.invocation.wait_params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN,
            command_buffer_.invocation.wait_params.flags);
  EXPECT_EQ(wait_payload.params.width,
            command_buffer_.invocation.wait_params.width);
  EXPECT_EQ(wait_payload.params.condition,
            command_buffer_.invocation.wait_params.condition);

  iree_hal_replay_command_buffer_atomic_store_payload_t store_payload = {};
  store_payload.target_ref = {
      /*.buffer_id=*/IREE_HAL_REPLAY_OBJECT_ID_NONE,
      /*.offset=*/12,
      /*.length=*/4,
      /*.buffer_slot=*/0,
      /*.reserved0=*/0,
  };
  store_payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER;
  store_payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;
  store_payload.params.value = UINT64_C(0xAABBCCDD);
  store_payload.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  store_payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
               kCommandBufferId, store_payload);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(2u, command_buffer_.invocation_count);
  EXPECT_EQ(kAtomicInvocationStore, command_buffer_.invocation.kind);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_TRANSFER,
            command_buffer_.invocation.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            command_buffer_.invocation.target_stage_mask);
  EXPECT_EQ(nullptr, command_buffer_.invocation.target_ref.buffer);
  EXPECT_EQ(0u, command_buffer_.invocation.target_ref.buffer_slot);
  EXPECT_EQ(12u, command_buffer_.invocation.target_ref.offset);
  EXPECT_EQ(4u, command_buffer_.invocation.target_ref.length);
  EXPECT_EQ(store_payload.params.value,
            command_buffer_.invocation.store_params.value);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN,
            command_buffer_.invocation.store_params.flags);
  EXPECT_EQ(store_payload.params.width,
            command_buffer_.invocation.store_params.width);

  iree_hal_replay_command_buffer_atomic_rmw_payload_t rmw_payload = {};
  rmw_payload.target_ref = DirectTarget(/*offset=*/32, /*length=*/4);
  rmw_payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST;
  rmw_payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS;
  rmw_payload.params.operand = UINT64_C(0x11223344);
  rmw_payload.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  rmw_payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  rmw_payload.params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_XOR;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_RMW,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW,
               kCommandBufferId, rmw_payload);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(3u, command_buffer_.invocation_count);
  EXPECT_EQ(kAtomicInvocationRmw, command_buffer_.invocation.kind);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_HOST,
            command_buffer_.invocation.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
            command_buffer_.invocation.target_stage_mask);
  EXPECT_EQ(target_buffer_, command_buffer_.invocation.target_ref.buffer);
  EXPECT_EQ(32u, command_buffer_.invocation.target_ref.offset);
  EXPECT_EQ(rmw_payload.params.operand,
            command_buffer_.invocation.rmw_params.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN,
            command_buffer_.invocation.rmw_params.flags);
  EXPECT_EQ(rmw_payload.params.width,
            command_buffer_.invocation.rmw_params.width);
  EXPECT_EQ(rmw_payload.params.operation,
            command_buffer_.invocation.rmw_params.operation);
}

TEST_F(ReplayAtomicExecutionTest, ReplaysQueueOperations) {
  OperationRecord record;

  iree_hal_replay_device_queue_atomic_wait_payload_t wait_payload = {};
  wait_payload.target_ref = DirectTarget(/*offset=*/8, /*length=*/8);
  wait_payload.queue_affinity = 1;
  wait_payload.wait_semaphore_count = 1;
  wait_payload.signal_semaphore_count = 1;
  wait_payload.params.value = UINT64_C(0x1020304050607080);
  wait_payload.params.mask = UINT64_C(0xFFEEDDCCBBAA9988);
  wait_payload.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  wait_payload.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  wait_payload.params.condition =
      IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT, kDeviceId,
               wait_payload, {WaitTimepoint(), SignalTimepoint(9)});
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(1u, device_.invocation_count);
  EXPECT_EQ(kAtomicInvocationWait, device_.invocation.kind);
  EXPECT_EQ(1u, device_.invocation.queue_affinity);
  EXPECT_EQ(1u, device_.invocation.wait_semaphore_count);
  EXPECT_EQ(wait_semaphore_, device_.invocation.wait_timepoint.semaphore);
  EXPECT_EQ(7u, device_.invocation.wait_timepoint.value);
  EXPECT_EQ(1u, device_.invocation.signal_semaphore_count);
  EXPECT_EQ(signal_semaphore_, device_.invocation.signal_timepoint.semaphore);
  EXPECT_EQ(9u, device_.invocation.signal_timepoint.value);
  EXPECT_EQ(target_buffer_, device_.invocation.target_ref.buffer);
  EXPECT_EQ(8u, device_.invocation.target_ref.offset);
  EXPECT_EQ(wait_payload.params.value, device_.invocation.wait_params.value);
  EXPECT_EQ(wait_payload.params.mask, device_.invocation.wait_params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, device_.invocation.wait_params.flags);
  EXPECT_EQ(wait_payload.params.width, device_.invocation.wait_params.width);
  EXPECT_EQ(wait_payload.params.condition,
            device_.invocation.wait_params.condition);

  iree_hal_replay_device_queue_atomic_store_payload_t store_payload = {};
  store_payload.target_ref = DirectTarget(/*offset=*/16, /*length=*/4);
  store_payload.queue_affinity = 1;
  store_payload.wait_semaphore_count = 1;
  store_payload.signal_semaphore_count = 1;
  store_payload.params.value = UINT64_C(0xAABBCCDD);
  store_payload.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  store_payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
               kDeviceId, store_payload,
               {WaitTimepoint(), SignalTimepoint(10)});
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(2u, device_.invocation_count);
  EXPECT_EQ(kAtomicInvocationStore, device_.invocation.kind);
  EXPECT_EQ(16u, device_.invocation.target_ref.offset);
  EXPECT_EQ(4u, device_.invocation.target_ref.length);
  EXPECT_EQ(10u, device_.invocation.signal_timepoint.value);
  EXPECT_EQ(store_payload.params.value, device_.invocation.store_params.value);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, device_.invocation.store_params.flags);
  EXPECT_EQ(store_payload.params.width, device_.invocation.store_params.width);

  iree_hal_replay_device_queue_atomic_rmw_payload_t rmw_payload = {};
  rmw_payload.target_ref = DirectTarget(/*offset=*/24, /*length=*/8);
  rmw_payload.queue_affinity = 1;
  rmw_payload.wait_semaphore_count = 1;
  rmw_payload.signal_semaphore_count = 1;
  rmw_payload.params.operand = UINT64_C(0x1122334455667788);
  rmw_payload.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  rmw_payload.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  rmw_payload.params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_RMW,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW, kDeviceId,
               rmw_payload, {WaitTimepoint(), SignalTimepoint(11)});
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(3u, device_.invocation_count);
  EXPECT_EQ(kAtomicInvocationRmw, device_.invocation.kind);
  EXPECT_EQ(24u, device_.invocation.target_ref.offset);
  EXPECT_EQ(8u, device_.invocation.target_ref.length);
  EXPECT_EQ(11u, device_.invocation.signal_timepoint.value);
  EXPECT_EQ(rmw_payload.params.operand, device_.invocation.rmw_params.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, device_.invocation.rmw_params.flags);
  EXPECT_EQ(rmw_payload.params.width, device_.invocation.rmw_params.width);
  EXPECT_EQ(rmw_payload.params.operation,
            device_.invocation.rmw_params.operation);
}

TEST_F(ReplayAtomicExecutionTest, RejectsMalformedRecords) {
  OperationRecord record;

  iree_hal_replay_command_buffer_atomic_store_payload_t command_payload = {};
  command_payload.target_ref = DirectTarget(/*offset=*/0, /*length=*/4);
  command_payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE;
  command_payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;
  command_payload.params.value = 1;
  command_payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
               kCommandBufferId, command_payload);
  record.AppendPayloadByte(0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));

  command_payload.params.flags = 1u << 31;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
               kCommandBufferId, command_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Replay(record));

  command_payload.params.flags = IREE_HAL_ATOMIC_FLAG_NONE;
  command_payload.params.reserved0[0] = 1;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
               kCommandBufferId, command_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Replay(record));
  EXPECT_EQ(0u, command_buffer_.invocation_count);

  iree_hal_replay_device_queue_atomic_store_payload_t queue_payload = {};
  queue_payload.target_ref = DirectTarget(/*offset=*/0, /*length=*/4);
  queue_payload.queue_affinity = 1;
  queue_payload.wait_semaphore_count = 1;
  queue_payload.signal_semaphore_count = 1;
  queue_payload.params.value = 1;
  queue_payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;

  queue_payload.target_ref.reserved0 = 1;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
               kDeviceId, queue_payload, {WaitTimepoint(), SignalTimepoint(9)});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));

  queue_payload.target_ref.reserved0 = 0;
  queue_payload.target_ref.buffer_slot = 1;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
               kDeviceId, queue_payload, {WaitTimepoint(), SignalTimepoint(9)});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));

  queue_payload.target_ref.buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  queue_payload.target_ref.buffer_slot = 0;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
               kDeviceId, queue_payload, {WaitTimepoint(), SignalTimepoint(9)});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));

  queue_payload.target_ref = DirectTarget(/*offset=*/0, /*length=*/8);
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
               kDeviceId, queue_payload, {WaitTimepoint(), SignalTimepoint(9)});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));

  queue_payload.target_ref = DirectTarget(/*offset=*/0, /*length=*/4);
  queue_payload.wait_semaphore_count = 2;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
               kDeviceId, queue_payload, {WaitTimepoint(), SignalTimepoint(9)});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));
  EXPECT_EQ(0u, device_.invocation_count);
}

}  // namespace
