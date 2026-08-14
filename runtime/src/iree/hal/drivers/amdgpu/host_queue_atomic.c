// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_atomic.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/barrier.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/atomic.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending_operation.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/util/pm4_atomic.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"

typedef union iree_hal_amdgpu_host_queue_atomic_kernargs_t {
  iree_hal_amdgpu_device_atomic_wait_kernargs_t wait;
  iree_hal_amdgpu_device_atomic_store_kernargs_t store;
  iree_hal_amdgpu_device_atomic_rmw_kernargs_t rmw;
} iree_hal_amdgpu_host_queue_atomic_kernargs_t;

static_assert(sizeof(iree_hal_amdgpu_host_queue_atomic_kernargs_t) <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "atomic kernargs must fit in one kernarg ring block");

static iree_hal_atomic_width_t iree_hal_amdgpu_host_queue_atomic_width(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      return operation->params.wait.width;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      return operation->params.store.width;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW:
      return operation->params.rmw.width;
    default:
      IREE_ASSERT_UNREACHABLE("atomic operation kind must be validated");
      return IREE_HAL_ATOMIC_WIDTH_32;
  }
}

static iree_hal_atomic_flags_t iree_hal_amdgpu_host_queue_atomic_flags(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      return operation->params.wait.flags;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      return operation->params.store.flags;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW:
      return operation->params.rmw.flags;
    default:
      IREE_ASSERT_UNREACHABLE("atomic operation kind must be validated");
      return IREE_HAL_ATOMIC_FLAG_NONE;
  }
}

static iree_hal_profile_queue_event_type_t
iree_hal_amdgpu_host_queue_atomic_profile_event_type(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_WAIT;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_STORE;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ATOMIC_RMW;
    default:
      IREE_ASSERT_UNREACHABLE("atomic operation kind must be validated");
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
  }
}

static iree_hal_amdgpu_host_queue_profile_event_info_t
iree_hal_amdgpu_host_queue_atomic_profile_event_info(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  iree_hal_amdgpu_host_queue_profile_event_info_t info = {
      .type = iree_hal_amdgpu_host_queue_atomic_profile_event_type(operation),
      .payload_length = iree_hal_atomic_width_byte_count(
          iree_hal_amdgpu_host_queue_atomic_width(operation)),
      .operation_count = 1,
  };
  return info;
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_resolve_target(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    uint8_t** out_target_device_ptr) {
  *out_target_device_ptr = NULL;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(operation->target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "atomic target must be backed by a materialized AMDGPU allocation");
  }
  target_device_ptr += iree_hal_buffer_byte_offset(operation->target_buffer) +
                       operation->target_offset;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_atomic_memory_validate_target(
      iree_hal_amdgpu_buffer_atomic_memory_cells(allocated_buffer),
      target_device_ptr, iree_hal_amdgpu_host_queue_atomic_width(operation),
      iree_hal_amdgpu_host_queue_atomic_flags(operation)));
  *out_target_device_ptr = target_device_ptr;
  return iree_ok_status();
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_atomic_minimum_acquire_scope(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  if (operation->kind == IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  return iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
      IREE_HAL_EXECUTION_STAGE_ATOMIC,
      iree_hal_amdgpu_host_queue_atomic_flags(operation),
      IREE_HAL_ATOMIC_FLAG_RELEASE);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_atomic_minimum_release_scope(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  iree_hsa_fence_scope_t release_scope = IREE_HSA_FENCE_SCOPE_NONE;
  if (operation->kind != IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE) {
    release_scope = iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
        IREE_HAL_EXECUTION_STAGE_ATOMIC,
        iree_hal_amdgpu_host_queue_atomic_flags(operation),
        IREE_HAL_ATOMIC_FLAG_ACQUIRE);
  }
  if (operation->kind != IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT) {
    release_scope = iree_hal_amdgpu_host_queue_max_fence_scope(
        release_scope, iree_hal_amdgpu_host_queue_buffer_release_scope(
                           operation->target_buffer));
  }
  return release_scope;
}

static bool iree_hal_amdgpu_host_queue_atomic_can_use_native_pm4(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  if (!queue->pm4_ib_slots) return false;
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      return iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_wait(
          queue->vendor_packet_capabilities);
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      return iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_store(
          queue->vendor_packet_capabilities);
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW:
      return false;
    default:
      IREE_ASSERT_UNREACHABLE("atomic operation kind must be validated");
      return false;
  }
}

static bool iree_hal_amdgpu_host_queue_atomic_emit_native_pm4(
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    uint8_t* target_device_ptr, iree_hal_amdgpu_pm4_ib_builder_t* builder) {
  const iree_hal_atomic_width_t width =
      iree_hal_amdgpu_host_queue_atomic_width(operation);
  uint32_t dword_count = 0;
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      dword_count = iree_hal_amdgpu_pm4_atomic_wait_dword_count(width);
      break;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      dword_count = IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT;
      break;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW:
    default:
      IREE_ASSERT_UNREACHABLE("native PM4 atomic must be wait or store");
      return false;
  }

  uint32_t* dwords =
      iree_hal_amdgpu_pm4_ib_builder_append_dwords(builder, dword_count);
  if (!dwords) return false;
  uint32_t emitted_dword_count = 0;
  bool did_emit = false;
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      did_emit = iree_hal_amdgpu_pm4_atomic_wait_emit(
          width, operation->params.wait.condition,
          (uint64_t)(uintptr_t)target_device_ptr, operation->params.wait.value,
          operation->params.wait.mask, dword_count, dwords,
          &emitted_dword_count);
      break;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      did_emit = iree_hal_amdgpu_pm4_atomic_store_emit(
          width, (uint64_t)(uintptr_t)target_device_ptr,
          operation->params.store.value, dword_count, dwords,
          &emitted_dword_count);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("native PM4 atomic must be wait or store");
  }
  return did_emit && emitted_dword_count == dword_count;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_atomic_pm4(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    uint8_t* target_device_ptr,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)operation->target_buffer,
  };
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_atomic_profile_event_info(operation);
  iree_hal_amdgpu_host_queue_pm4_ib_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
      queue, resolution, signal_semaphore_list,
      IREE_ARRAYSIZE(operation_resources), iree_hsa_signal_null(),
      &profile_event_info, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  if (IREE_UNLIKELY(!iree_hal_amdgpu_host_queue_atomic_emit_native_pm4(
          operation, target_device_ptr, &submission.pm4_ib_builder))) {
    iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue, &submission);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "native PM4 atomic payload does not fit IB slot");
  }
  submission.minimum_acquire_scope =
      iree_hal_amdgpu_host_queue_atomic_minimum_acquire_scope(operation);
  submission.minimum_release_scope =
      iree_hal_amdgpu_host_queue_atomic_minimum_release_scope(operation);
  uint64_t submission_id = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), &profile_event_info,
      submission_flags, &submission, &submission_id));
  profile_event_info.submission_id = submission_id;
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, resolution, signal_semaphore_list, &profile_event_info);
  return iree_ok_status();
}

static void iree_hal_amdgpu_host_queue_prepare_atomic_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    uint8_t* target_device_ptr,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_host_queue_atomic_kernargs_t* out_kernargs,
    iree_host_size_t* out_kernarg_length) {
  memset(out_dispatch_packet, 0, sizeof(*out_dispatch_packet));
  memset(out_kernargs, 0, sizeof(*out_kernargs));
  switch (operation->kind) {
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT:
      iree_hal_amdgpu_device_atomic_wait_emplace(
          queue->transfer_context->kernels, out_dispatch_packet,
          target_device_ptr, operation->params.wait, out_kernargs);
      *out_kernarg_length = IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_SIZE;
      break;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE:
      iree_hal_amdgpu_device_atomic_store_emplace(
          queue->transfer_context->kernels, out_dispatch_packet,
          target_device_ptr, operation->params.store, out_kernargs);
      *out_kernarg_length = IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_SIZE;
      break;
    case IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW:
      iree_hal_amdgpu_device_atomic_rmw_emplace(
          queue->transfer_context->kernels, out_dispatch_packet,
          target_device_ptr, operation->params.rmw, out_kernargs);
      *out_kernarg_length = IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_SIZE;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("atomic operation kind must be validated");
      break;
  }
  out_dispatch_packet->kernarg_address = NULL;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_atomic_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    uint8_t* target_device_ptr,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_host_queue_atomic_kernargs_t kernargs;
  iree_host_size_t kernarg_length = 0;
  iree_hal_amdgpu_host_queue_prepare_atomic_dispatch(
      queue, operation, target_device_ptr, &dispatch_packet, &kernargs,
      &kernarg_length);

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)operation->target_buffer,
  };
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_atomic_profile_event_info(operation);
  uint64_t submission_id = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      kernarg_length, operation_resources, IREE_ARRAYSIZE(operation_resources),
      iree_hal_amdgpu_host_queue_atomic_minimum_acquire_scope(operation),
      iree_hal_amdgpu_host_queue_atomic_minimum_release_scope(operation),
      &profile_event_info, submission_flags, out_ready, &submission_id));
  if (*out_ready) {
    profile_event_info.submission_id = submission_id;
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        queue, resolution, signal_semaphore_list, &profile_event_info);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_atomic(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_atomic_resolve_target(
      operation, &target_device_ptr));
  if (iree_hal_amdgpu_host_queue_atomic_can_use_native_pm4(queue, operation)) {
    return iree_hal_amdgpu_host_queue_submit_atomic_pm4(
        queue, resolution, signal_semaphore_list, operation, target_device_ptr,
        submission_flags, out_ready);
  }
  return iree_hal_amdgpu_host_queue_submit_atomic_dispatch(
      queue, resolution, signal_semaphore_list, operation, target_device_ptr,
      submission_flags, out_ready);
}

iree_status_t iree_hal_amdgpu_host_queue_defer_atomic(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_ATOMIC, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(
      op, (iree_hal_resource_t*)operation->target_buffer);
  op->atomic = *operation;
  *out_op = op;
  return iree_ok_status();
}
