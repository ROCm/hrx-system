// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/driver/task/registration/driver_module.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

static iree_hal_device_group_t* CreateDeviceGroup(
    iree_hal_device_t* const* devices, iree_host_size_t device_count) {
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    IREE_CHECK_OK(
        iree_hal_device_group_builder_add_device(&builder, devices[i]));
  }

  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  return group;
}

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  iree_hal_device_t* device = CreateTaskDevice();
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group =
      CreateDeviceGroup(devices, IREE_ARRAYSIZE(devices));
  iree_hal_device_release(device);
  return group;
}

static iree_const_byte_span_t GetCapturedFileContents(
    const std::vector<uint8_t>& storage) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_LE(file_header.file_length, storage.size());
  return iree_make_const_byte_span(storage.data(),
                                   (iree_host_size_t)file_header.file_length);
}

static void AppendReplayRecord(
    iree_hal_replay_file_writer_t* writer,
    const iree_hal_replay_file_record_metadata_t& metadata,
    std::initializer_list<iree_const_byte_span_t> payload_iovecs) {
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, payload_iovecs.size(), payload_iovecs.begin(),
      /*out_payload_range=*/nullptr));
}

static void AppendDeviceObjectRecord(iree_hal_replay_file_writer_t* writer,
                                     uint64_t sequence_ordinal,
                                     iree_hal_replay_object_id_t device_id) {
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = device_id,
      .related_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_NONE,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata, {});
}

static void AppendQueueObjectRecord(iree_hal_replay_file_writer_t* writer,
                                    uint64_t sequence_ordinal,
                                    iree_hal_replay_object_id_t device_id,
                                    iree_hal_replay_object_id_t queue_id,
                                    uint32_t queue_ordinal) {
  const iree_hal_replay_provisioned_queue_object_payload_t payload = {
      .family_ordinal = 0,
      .queue_ordinal = queue_ordinal,
      .reserved0 = 0,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = queue_id,
      .related_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_PROVISIONED_QUEUE_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_NONE,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendSemaphoreCreateRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t semaphore_id) {
  const iree_hal_replay_semaphore_object_payload_t payload = {
      .queue_family_affinity = iree_hal_make_queue_family_affinity(0),
      .initial_value = 0,
      .flags = IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      .reserved0 = 0,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = device_id,
      .related_object_id = semaphore_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendQueueBarrierRecord(iree_hal_replay_file_writer_t* writer,
                                     uint64_t sequence_ordinal,
                                     iree_hal_replay_object_id_t device_id,
                                     iree_hal_replay_object_id_t queue_id,
                                     iree_hal_replay_object_id_t semaphore_id,
                                     bool is_wait) {
  const iree_hal_replay_queue_barrier_payload_t payload = {
      .flags = IREE_HAL_QUEUE_BARRIER_FLAG_NONE,
      .wait_semaphore_count = is_wait ? 1u : 0u,
      .signal_semaphore_count = is_wait ? 0u : 1u,
      .reserved0 = 0,
  };
  const iree_hal_replay_semaphore_timepoint_payload_t timepoint = {
      .semaphore_id = semaphore_id,
      .value = 1,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = queue_id,
      .related_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, metadata,
      {iree_make_const_byte_span(&payload, sizeof(payload)),
       iree_make_const_byte_span(&timepoint, sizeof(timepoint))});
}

static void AppendQueueAtomicWaitRecord(iree_hal_replay_file_writer_t* writer,
                                        uint64_t sequence_ordinal,
                                        iree_hal_replay_object_id_t device_id,
                                        iree_hal_replay_object_id_t queue_id) {
  iree_hal_replay_queue_atomic_wait_payload_t payload = {};
  payload.target_ref.buffer_id = queue_id + 10;
  payload.target_ref.length = 4;
  payload.params.value = 1;
  payload.params.mask = UINT32_MAX;
  payload.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  payload.params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL;
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = queue_id,
      .related_object_id = payload.target_ref.buffer_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_WAIT,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendImmediateQueueTransferRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t queue_id) {
  const iree_hal_replay_queue_transfer_payload_t payload = {
      .wait_semaphore_count = 0,
      .signal_semaphore_count = 0,
      .operation_count = 1,
      .data_length = 0,
  };
  iree_hal_replay_queue_transfer_operation_payload_t operation = {
      .type = IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  operation.source_ref.buffer_id = queue_id + 11;
  operation.source_ref.length = 4;
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = queue_id,
      .related_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TRANSFER,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(
      writer, metadata,
      {iree_make_const_byte_span(&payload, sizeof(payload)),
       iree_make_const_byte_span(&operation, sizeof(operation))});
}

static void AppendDispatchRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t target_id,
    iree_hal_replay_object_id_t executable_id,
    iree_hal_replay_operation_code_t operation_code) {
  iree_hal_replay_dispatch_payload_t payload = {
      .executable_id = executable_id,
  };
  payload.workgroup_size[0] = 1;
  payload.workgroup_size[1] = 1;
  payload.workgroup_size[2] = 1;
  payload.workgroup_count[0] = 1;
  payload.workgroup_count[1] = 1;
  payload.workgroup_count[2] = 1;
  const bool is_queue_dispatch =
      operation_code == IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_DISPATCH;
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = target_id,
      .related_object_id =
          is_queue_dispatch ? executable_id : IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      .object_type = static_cast<iree_hal_replay_object_type_t>(
          is_queue_dispatch ? IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE
                            : IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER),
      .operation_code = operation_code,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendCommandBufferCreateRecords(
    iree_hal_replay_file_writer_t* writer, uint64_t* sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t command_buffer_id) {
  const iree_hal_replay_queue_family_command_buffer_object_payload_t payload = {
      .mode = IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      .command_categories = IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      .queue_family_ordinal = 0,
      .reserved0 = 0,
      .binding_capacity = 0,
  };
  const iree_hal_replay_file_record_metadata_t operation_metadata = {
      .sequence_ordinal = (*sequence_ordinal)++,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = device_id,
      .related_object_id = command_buffer_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type =
          IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_FAMILY_COMMAND_BUFFER_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      .operation_code =
          IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, operation_metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
  const iree_hal_replay_file_record_metadata_t object_metadata = {
      .sequence_ordinal = (*sequence_ordinal)++,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = command_buffer_id,
      .related_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type =
          IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_FAMILY_COMMAND_BUFFER_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_NONE,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, object_metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendQueueExecuteRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id, iree_hal_replay_object_id_t queue_id,
    iree_hal_replay_object_id_t command_buffer_id) {
  const iree_hal_replay_queue_execute_payload_t payload = {
      .command_buffer_id = command_buffer_id,
      .flags = IREE_HAL_QUEUE_EXECUTE_FLAG_NONE,
      .wait_semaphore_count = 0,
      .signal_semaphore_count = 0,
      .binding_count = 0,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = device_id,
      .object_id = queue_id,
      .related_object_id = command_buffer_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_EXECUTE,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_EXECUTE,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendVirtualBufferDetachRecord(
    iree_hal_replay_file_writer_t* writer, uint64_t sequence_ordinal,
    iree_hal_replay_object_id_t device_id,
    iree_status_code_t status_code = IREE_STATUS_OK) {
  const iree_hal_replay_virtual_buffer_detach_payload_t payload = {};
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .device_id = device_id,
      .object_id = 21,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_DETACH,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_DETACH,
      .status_code = static_cast<uint32_t>(status_code),
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload))});
}

static void AppendScopeBeginRecord(iree_hal_replay_file_writer_t* writer,
                                   uint64_t sequence_ordinal) {
  const iree_hal_replay_scope_payload_t payload = {
      .name_length = 4,
      .flags = IREE_HAL_REPLAY_SCOPE_FLAG_NONE,
      .reserved0 = 0,
      .reserved1 = 0,
  };
  const iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .thread_id = 0,
      .device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .related_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_NONE,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN,
      .status_code = IREE_STATUS_OK,
  };
  AppendReplayRecord(writer, metadata,
                     {iree_make_const_byte_span(&payload, sizeof(payload)),
                      iree_make_const_byte_span("fail", 4)});
}

typedef enum QueueDependencyReplayShape {
  kForwardDependencyAcrossMemoryBoundary,
  kForwardDependencyAcrossFailedDetach,
  kSatisfiedDependencyBeforeMemoryBoundary,
  kDeviceMemoryDependencyAcrossMemoryBoundary,
  kImmediateCompletionBehindDeviceMemoryWait,
  kCallbackErrorBeforeProducer,
} QueueDependencyReplayShape;

static std::vector<uint8_t> MakeQueueDependencyReplay(
    QueueDependencyReplayShape shape) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kConsumerQueueId = 2;
  constexpr iree_hal_replay_object_id_t kProducerQueueId = 3;
  constexpr iree_hal_replay_object_id_t kSemaphoreId = 4;
  std::vector<uint8_t> storage(32768, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  uint64_t sequence_ordinal = 0;
  AppendDeviceObjectRecord(writer, sequence_ordinal++, kDeviceId);
  AppendQueueObjectRecord(writer, sequence_ordinal++, kDeviceId,
                          kConsumerQueueId, /*queue_ordinal=*/0);
  if (shape != kCallbackErrorBeforeProducer &&
      shape != kDeviceMemoryDependencyAcrossMemoryBoundary &&
      shape != kImmediateCompletionBehindDeviceMemoryWait) {
    AppendQueueObjectRecord(writer, sequence_ordinal++, kDeviceId,
                            kProducerQueueId, /*queue_ordinal=*/1);
  }
  AppendSemaphoreCreateRecord(writer, sequence_ordinal++, kDeviceId,
                              kSemaphoreId);

  if (shape == kSatisfiedDependencyBeforeMemoryBoundary) {
    AppendQueueBarrierRecord(writer, sequence_ordinal++, kDeviceId,
                             kProducerQueueId, kSemaphoreId,
                             /*is_wait=*/false);
  }
  if (shape == kDeviceMemoryDependencyAcrossMemoryBoundary ||
      shape == kImmediateCompletionBehindDeviceMemoryWait) {
    AppendQueueAtomicWaitRecord(writer, sequence_ordinal++, kDeviceId,
                                kConsumerQueueId);
  } else {
    AppendQueueBarrierRecord(writer, sequence_ordinal++, kDeviceId,
                             kConsumerQueueId, kSemaphoreId,
                             /*is_wait=*/true);
  }
  if (shape == kImmediateCompletionBehindDeviceMemoryWait) {
    AppendImmediateQueueTransferRecord(writer, sequence_ordinal++, kDeviceId,
                                       kConsumerQueueId);
  }
  if (shape == kCallbackErrorBeforeProducer) {
    AppendScopeBeginRecord(writer, sequence_ordinal++);
    AppendQueueObjectRecord(writer, sequence_ordinal++, kDeviceId,
                            kProducerQueueId, /*queue_ordinal=*/1);
  } else if (shape == kForwardDependencyAcrossFailedDetach) {
    AppendVirtualBufferDetachRecord(writer, sequence_ordinal++, kDeviceId,
                                    IREE_STATUS_INTERNAL);
  } else {
    AppendVirtualBufferDetachRecord(writer, sequence_ordinal++, kDeviceId);
  }
  if (shape == kForwardDependencyAcrossMemoryBoundary ||
      shape == kForwardDependencyAcrossFailedDetach ||
      shape == kCallbackErrorBeforeProducer) {
    AppendQueueBarrierRecord(writer, sequence_ordinal++, kDeviceId,
                             kProducerQueueId, kSemaphoreId,
                             /*is_wait=*/false);
  }

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

typedef enum OpaqueDispatchReplayShape {
  kDirectDispatchBeforeMemoryBoundary,
  kCommandBufferDispatchAtEndOfFile,
  kBoundedCommandBufferAtEndOfFile,
} OpaqueDispatchReplayShape;

static std::vector<uint8_t> MakeOpaqueDispatchReplay(
    OpaqueDispatchReplayShape shape) {
  constexpr iree_hal_replay_object_id_t kDeviceId = 1;
  constexpr iree_hal_replay_object_id_t kQueueId = 2;
  constexpr iree_hal_replay_object_id_t kCommandBufferId = 3;
  constexpr iree_hal_replay_object_id_t kExecutableId = 4;
  std::vector<uint8_t> storage(32768, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  uint64_t sequence_ordinal = 0;
  AppendDeviceObjectRecord(writer, sequence_ordinal++, kDeviceId);
  AppendQueueObjectRecord(writer, sequence_ordinal++, kDeviceId, kQueueId,
                          /*queue_ordinal=*/0);
  if (shape == kDirectDispatchBeforeMemoryBoundary) {
    AppendDispatchRecord(writer, sequence_ordinal++, kDeviceId, kQueueId,
                         kExecutableId,
                         IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_DISPATCH);
    AppendVirtualBufferDetachRecord(writer, sequence_ordinal++, kDeviceId);
  } else {
    AppendCommandBufferCreateRecords(writer, &sequence_ordinal, kDeviceId,
                                     kCommandBufferId);
    if (shape == kCommandBufferDispatchAtEndOfFile) {
      AppendDispatchRecord(
          writer, sequence_ordinal++, kDeviceId, kCommandBufferId,
          kExecutableId,
          IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH);
    }
    AppendQueueExecuteRecord(writer, sequence_ordinal++, kDeviceId, kQueueId,
                             kCommandBufferId);
  }

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static iree_status_t FailReplayScopeEvent(
    void* user_data, const iree_hal_replay_scope_event_t* event) {
  (void)event;
  iree_host_size_t* invocation_count = (iree_host_size_t*)user_data;
  ++*invocation_count;
  return iree_make_status(IREE_STATUS_ABORTED, "injected replay scope failure");
}

static IREE_ALLOCATOR_INLINE_STORAGE(g_callback_failure_host_storage,
                                     64 * 1024);
static iree_hal_device_group_t* g_callback_failure_device_group = nullptr;

TEST(ReplayExecuteTest,
     RejectsForwardQueueDependencyAcrossMemoryBoundaryBeforeExecution) {
  std::vector<uint8_t> storage =
      MakeQueueDependencyReplay(kForwardDependencyAcrossMemoryBoundary);
  iree_hal_replay_plan_t* plan = nullptr;

  iree::Status status =
      iree::internal::ConsumeForTest(iree_hal_replay_plan_allocate(
          GetCapturedFileContents(storage), iree_allocator_system(), &plan));

  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "replay completion boundary at sequence 5 cannot wait for queue "
          "operation at sequence 4 with an unresolved forward or "
          "device-memory dependency"));
  EXPECT_EQ(plan, nullptr);
}

TEST(ReplayExecuteTest, FailedDetachDoesNotIntroduceCompletionBoundary) {
  std::vector<uint8_t> storage =
      MakeQueueDependencyReplay(kForwardDependencyAcrossFailedDetach);
  iree_hal_replay_plan_t* plan = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_plan_allocate(GetCapturedFileContents(storage),
                                               iree_allocator_system(), &plan));
  iree_hal_replay_plan_free(plan);
}

TEST(ReplayExecuteTest, AllowsSatisfiedQueueDependencyBeforeMemoryBoundary) {
  std::vector<uint8_t> storage =
      MakeQueueDependencyReplay(kSatisfiedDependencyBeforeMemoryBoundary);
  iree_hal_replay_plan_t* plan = nullptr;

  IREE_ASSERT_OK(iree_hal_replay_plan_allocate(GetCapturedFileContents(storage),
                                               iree_allocator_system(), &plan));

  iree_hal_replay_plan_free(plan);
}

TEST(ReplayExecuteTest,
     RejectsDeviceMemoryQueueDependencyAcrossMemoryBoundaryBeforeExecution) {
  std::vector<uint8_t> storage =
      MakeQueueDependencyReplay(kDeviceMemoryDependencyAcrossMemoryBoundary);
  iree_hal_replay_plan_t* plan = nullptr;

  iree::Status status =
      iree::internal::ConsumeForTest(iree_hal_replay_plan_allocate(
          GetCapturedFileContents(storage), iree_allocator_system(), &plan));

  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "replay completion boundary at sequence 4 cannot wait for queue "
          "operation at sequence 3 with an unresolved forward or "
          "device-memory dependency"));
  EXPECT_EQ(plan, nullptr);
}

TEST(ReplayExecuteTest, RejectsOpaqueDirectQueueDispatchBeforeMemoryBoundary) {
  std::vector<uint8_t> storage =
      MakeOpaqueDispatchReplay(kDirectDispatchBeforeMemoryBoundary);
  iree_hal_replay_plan_t* plan = nullptr;

  iree::Status status =
      iree::internal::ConsumeForTest(iree_hal_replay_plan_allocate(
          GetCapturedFileContents(storage), iree_allocator_system(), &plan));

  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "replay completion boundary at sequence 3 cannot wait for queue "
          "operation at sequence 2 with an unresolved forward or "
          "device-memory dependency"));
  EXPECT_EQ(plan, nullptr);
}

TEST(ReplayExecuteTest, RejectsOpaqueCommandBufferDispatchAtEndOfFile) {
  std::vector<uint8_t> storage =
      MakeOpaqueDispatchReplay(kCommandBufferDispatchAtEndOfFile);
  iree_hal_replay_plan_t* plan = nullptr;

  iree::Status status =
      iree::internal::ConsumeForTest(iree_hal_replay_plan_allocate(
          GetCapturedFileContents(storage), iree_allocator_system(), &plan));

  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "replay end-of-file cannot wait for queue operation at sequence 5 "
          "with an unresolved forward or device-memory dependency"));
  EXPECT_EQ(plan, nullptr);
}

TEST(ReplayExecuteTest, AllowsBoundedCommandBufferAtEndOfFile) {
  std::vector<uint8_t> storage =
      MakeOpaqueDispatchReplay(kBoundedCommandBufferAtEndOfFile);
  iree_hal_replay_plan_t* plan = nullptr;

  IREE_ASSERT_OK(iree_hal_replay_plan_allocate(GetCapturedFileContents(storage),
                                               iree_allocator_system(), &plan));

  iree_hal_replay_plan_free(plan);
}

TEST(ReplayExecuteTest,
     RejectsImmediateOperationBehindBlockedSameQueuePredecessor) {
  std::vector<uint8_t> storage =
      MakeQueueDependencyReplay(kImmediateCompletionBehindDeviceMemoryWait);
  iree_hal_replay_plan_t* plan = nullptr;

  iree::Status status =
      iree::internal::ConsumeForTest(iree_hal_replay_plan_allocate(
          GetCapturedFileContents(storage), iree_allocator_system(), &plan));

  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "replay queue operation at sequence 4 requires immediate completion "
          "but has an unresolved forward or device-memory dependency"));
  EXPECT_EQ(plan, nullptr);
}

TEST(ReplayExecuteTest, ReturnsCallbackErrorWithPendingQueueCompletion) {
  std::vector<uint8_t> storage =
      MakeQueueDependencyReplay(kCallbackErrorBeforeProducer);
  iree_host_size_t invocation_count = 0;
  iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  options.scope_event_callback.fn = FailReplayScopeEvent;
  options.scope_event_callback.user_data = &invocation_count;
  ASSERT_EQ(g_callback_failure_device_group, nullptr);
  g_callback_failure_device_group = CreateTaskDeviceGroup();
  iree_allocator_t host_allocator =
      iree_allocator_inline_arena(&g_callback_failure_host_storage.header);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                   g_callback_failure_device_group, &options,
                                   host_allocator));

  EXPECT_EQ(invocation_count, 1u);
  // The executor retains its dependencies on this error path so accepted work
  // remains valid for process lifetime. Keep the caller's device-group owner
  // and fixed host storage alive for the same lifetime.
  // This rooting is only LeakSanitizer hygiene for the deliberate containment;
  // it is not a production substitute for durable ownership and eventual
  // cleanup of accepted work.
}

}  // namespace
