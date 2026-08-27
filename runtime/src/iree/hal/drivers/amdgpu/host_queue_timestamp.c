// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/timestamp.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

static iree_hal_amdgpu_pm4_ib_slot_t* iree_hal_amdgpu_host_queue_pm4_ib_slot(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id) {
  return &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
}

void iree_hal_amdgpu_host_queue_commit_timestamp_start(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control, uint64_t* start_tick) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      iree_hal_amdgpu_host_queue_pm4_ib_slot(queue, packet_id);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_start(
      &packet->pm4_ib, pm4_ib_slot, packet_control,
      queue->pm4_timestamp_strategy, start_tick, &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_timestamp_end(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint64_t* end_tick) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      iree_hal_amdgpu_host_queue_pm4_ib_slot(queue, packet_id);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_end(
      &packet->pm4_ib, pm4_ib_slot, packet_control,
      queue->pm4_timestamp_strategy, completion_signal, end_tick, &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_timestamp_range(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint64_t* start_tick,
    uint64_t* end_tick) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      iree_hal_amdgpu_host_queue_pm4_ib_slot(queue, packet_id);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_range(
      &packet->pm4_ib, pm4_ib_slot, packet_control,
      queue->pm4_timestamp_strategy, completion_signal, start_tick, end_tick,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

static_assert(sizeof(iree_hal_amdgpu_queue_timestamp_capture_args_t) <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "timestamp capture kernargs must fit in one kernarg ring block");

// Validates a timestamp-capture target and resolves the 8-byte-aligned target
// device pointer the GPU clock tick is written to.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_timestamp_target(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    uint8_t** out_target_device_ptr) {
  *out_target_device_ptr = NULL;
  if (IREE_UNLIKELY(!target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target buffer must be non-null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      target_buffer, target_offset, sizeof(uint64_t)));

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;
  if (IREE_UNLIKELY(
          !iree_host_ptr_has_alignment(target_device_ptr, sizeof(uint64_t)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "timestamp target must be 8-byte aligned");
  }

  *out_target_device_ptr = target_device_ptr;
  return iree_ok_status();
}

bool iree_hal_amdgpu_host_queue_can_use_pm4_timestamp(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->pm4_ib_slots && iree_hal_amdgpu_pm4_copy_timestamp_control(
                                    queue->pm4_timestamp_strategy) != 0;
}

// Captures the tick from the command processor's GPU clock counter with a PM4
// COPY_DATA packet.
static iree_status_t iree_hal_amdgpu_host_queue_submit_pm4_timestamp(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, uint8_t* target_device_ptr,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  iree_hal_amdgpu_host_queue_pm4_ib_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
      queue, resolution, signal_semaphore_list,
      IREE_ARRAYSIZE(operation_resources), iree_hsa_signal_null(),
      /*profile_queue_event_info=*/NULL, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
          &submission.pm4_ib_builder, queue->pm4_timestamp_strategy,
          target_device_ptr);
  if (IREE_UNLIKELY(!did_emit)) {
    iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue, &submission);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 timestamp payload does not fit IB slot");
  }
  submission.minimum_release_scope =
      iree_hal_amdgpu_host_queue_buffer_release_scope(target_buffer);
  uint64_t submission_epoch = 0;
  return iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), /*profile_queue_event_info=*/NULL,
      submission_flags, &submission, &submission_epoch);
}

// Prepares a timestamp capture dispatch packet and kernargs in stack-local
// storage without touching queue rings. All user-input validation must happen
// before this so the caller can avoid reserving AQL slots before the packet
// shape is known.
static void iree_hal_amdgpu_host_queue_prepare_timestamp_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, uint8_t* target_device_ptr,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_queue_timestamp_capture_args_t* out_kernargs) {
  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_queue_timestamp_capture_args_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  iree_hal_amdgpu_device_timestamp_emplace_queue_capture(
      &queue->transfer_context->kernels
           ->iree_hal_amdgpu_device_timestamp_capture_queue_tick,
      (iree_amdgpu_device_tick_t*)target_device_ptr, &dispatch_packet,
      &kernargs);
  dispatch_packet.kernarg_address = NULL;

  *out_dispatch_packet = dispatch_packet;
  *out_kernargs = kernargs;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_timestamp(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_TIMESTAMP_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported timestamp flags: 0x%" PRIx64, flags);
  }

  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_timestamp_target(
      target_buffer, target_offset, &target_device_ptr));

  if (iree_hal_amdgpu_host_queue_can_use_pm4_timestamp(queue)) {
    return iree_hal_amdgpu_host_queue_submit_pm4_timestamp(
        queue, resolution, signal_semaphore_list, target_buffer,
        target_device_ptr, submission_flags, out_ready);
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_queue_timestamp_capture_args_t kernargs;
  iree_hal_amdgpu_host_queue_prepare_timestamp_dispatch(
      queue, target_device_ptr, &dispatch_packet, &kernargs);

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), IREE_HSA_FENCE_SCOPE_NONE,
      iree_hal_amdgpu_host_queue_buffer_release_scope(target_buffer),
      /*profile_queue_event_info=*/NULL, submission_flags, out_ready,
      /*out_submission_id=*/NULL);
}
