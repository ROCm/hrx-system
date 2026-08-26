// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/device/tsan.h"
#include "iree/hal/drivers/amdgpu/feedback_state.h"
#include "iree/hal/drivers/amdgpu/host_queue_atomic.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_scratch.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/host_queue_file.h"
#include "iree/hal/drivers/amdgpu/host_queue_host_call.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/tsan_state.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/hal/utils/resource_set.h"

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;

static iree_status_t iree_hal_amdgpu_host_queue_submit_tsan_state_initialize(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_tsan_queue_initialize_args_t* initialize_args) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  iree_hal_amdgpu_host_queue_kernel_submission_t submission;
  bool ready = false;
  uint64_t submission_epoch = 0;
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
      queue, &resolution, iree_hal_semaphore_list_empty(),
      /*operation_resource_count=*/0, /*payload_packet_count=*/1,
      (uint32_t)iree_host_size_ceil_div(
          sizeof(iree_hal_amdgpu_tsan_queue_initialize_args_t),
          sizeof(iree_hal_amdgpu_kernarg_block_t)),
      &ready, &submission);
  if (iree_status_is_ok(status) && !ready) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue had insufficient capacity for TSAN state initialization");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
        &queue->aql_ring, submission.first_packet_id);
    iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
        &queue->transfer_context->kernels
             ->iree_hal_amdgpu_device_tsan_initialize_queue_state,
        initialize_args, queue->transfer_context->max_workgroup_count,
        &packet->dispatch, submission.kernargs.blocks->data);
    packet->dispatch.completion_signal =
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring);
    const uint16_t setup = packet->dispatch.setup;
    const iree_hsa_fence_scope_t acquire_scope =
        iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
            IREE_HSA_FENCE_SCOPE_AGENT);
    const uint16_t header = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_packet_control_barrier(acquire_scope,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));

    iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, &resolution,
                                                             &submission);
    submission_epoch = iree_hal_amdgpu_host_queue_finish_kernel_submission(
        queue, &resolution, iree_hal_semaphore_list_empty(),
        /*operation_resources=*/NULL, /*operation_resource_count=*/0,
        /*inout_resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &submission);
    iree_hal_amdgpu_host_queue_publish_submission_kernargs(queue, &submission);
    iree_hal_amdgpu_notification_ring_publish_epoch(&queue->notification_ring,
                                                    submission_epoch);
    iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
    iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                      submission.first_packet_id);
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_wait_for_setup_epoch(queue,
                                                             submission_epoch);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent,
    hsa_amd_memory_pool_t pm4_ib_pool, uint32_t aql_queue_capacity,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, aql_queue_capacity);
  iree_host_size_t pm4_ib_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &pm4_ib_size,
              IREE_STRUCT_FIELD(aql_queue_capacity,
                                iree_hal_amdgpu_pm4_ib_slot_t, NULL)));
  if (IREE_UNLIKELY(!pm4_ib_pool.handle)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "PM4 IB memory pool is required"));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, pm4_ib_size);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slots = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(libhsa), pm4_ib_pool, pm4_ib_size,
              HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, (void**)&pm4_ib_slots));
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(libhsa), /*num_agents=*/1, &gpu_agent, /*flags=*/NULL,
      pm4_ib_slots);
  if (iree_status_is_ok(status)) {
    memset(pm4_ib_slots, 0, pm4_ib_size);
    out_queue->pm4_ib_slots = pm4_ib_slots;
  } else {
    status = iree_status_join(status, iree_hsa_amd_memory_pool_free(
                                          IREE_LIBHSA(libhsa), pm4_ib_slots));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_initialize_tsan_state(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_tsan_memory_policy_t* memory_policy,
    iree_host_size_t queue_ordinal, iree_host_size_t physical_queue_ordinal,
    iree_device_size_t workgroup_shadow_stride,
    iree_device_size_t dispatch_shadow_stride, uint32_t workgroup_capacity,
    uint32_t shadow_entry_size, uint32_t memory_granule_shift,
    uint32_t shadow_slot_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, shadow_slot_count);

  if (IREE_UNLIKELY(queue->tsan.allocation_base)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "AMDGPU host queue TSAN state is already "
                             "initialized"));
  }
  if (IREE_UNLIKELY(queue_ordinal > UINT32_MAX ||
                    queue->device_ordinal > UINT32_MAX ||
                    physical_queue_ordinal > UINT32_MAX)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU TSAN queue ordinals exceed device ABI "
                             "limits"));
  }
  if (IREE_UNLIKELY(queue->aql_ring.mask >= UINT32_MAX)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU TSAN AQL ring mask %u"
                             " exceeds payload slot derivation ABI limits",
                             queue->aql_ring.mask));
  }

  iree_device_size_t shadow_size = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul(
          dispatch_shadow_stride, shadow_slot_count, &shadow_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "AMDGPU TSAN queue shadow size overflow: "
                "dispatch_shadow_stride=%" PRIu64 ", shadow_slot_count=%u",
                (uint64_t)dispatch_shadow_stride, shadow_slot_count));
  }
  if (IREE_UNLIKELY(shadow_size == 0 ||
                    (iree_device_size_t)(iree_host_size_t)shadow_size !=
                        shadow_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "AMDGPU TSAN queue shadow allocation size is "
                             "invalid: %" PRIu64,
                             (uint64_t)shadow_size));
  }

  iree_host_size_t queue_state_offset = 0;
  iree_host_size_t shadow_offset = 0;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &allocation_size,
              IREE_STRUCT_FIELD(1, iree_hal_amdgpu_tsan_queue_state_t,
                                &queue_state_offset),
              IREE_STRUCT_FIELD_ALIGNED((iree_host_size_t)shadow_size, uint8_t,
                                        64, &shadow_offset)));

  void* allocation_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_memory_pool_allocate(
              IREE_LIBHSA(queue->libhsa), memory_policy->memory_pool,
              allocation_size, HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
              &allocation_base));
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(queue->libhsa), memory_policy->access_agent_count,
      memory_policy->access_agents, /*flags=*/NULL, allocation_base);
  if (iree_status_is_ok(status)) {
    uint8_t* storage = (uint8_t*)allocation_base;
    iree_hal_amdgpu_tsan_queue_state_t* queue_state =
        (iree_hal_amdgpu_tsan_queue_state_t*)(storage + queue_state_offset);
    void* shadow_base = storage + shadow_offset;
    const iree_hal_amdgpu_tsan_queue_state_t host_state = {
        .record_length = sizeof(iree_hal_amdgpu_tsan_queue_state_t),
        .abi_version = IREE_HAL_AMDGPU_TSAN_QUEUE_STATE_ABI_VERSION_0,
        .flags = IREE_HAL_AMDGPU_TSAN_QUEUE_STATE_FLAG_NONE,
        .queue_ordinal = (uint32_t)queue_ordinal,
        .physical_device_ordinal = (uint32_t)queue->device_ordinal,
        .physical_queue_ordinal = (uint32_t)physical_queue_ordinal,
        .reserved0 = 0,
        .shadow_slot_count = shadow_slot_count,
        .aql_ring_base = (uint64_t)(uintptr_t)queue->aql_ring.base,
        .aql_ring_mask = queue->aql_ring.mask,
        .reserved1 = 0,
        .shadow_base = (uint64_t)(uintptr_t)shadow_base,
        .shadow_size = shadow_size,
        .dispatch_shadow_stride = dispatch_shadow_stride,
        .workgroup_shadow_stride = workgroup_shadow_stride,
        .workgroup_capacity = workgroup_capacity,
        .shadow_entry_size = shadow_entry_size,
        .memory_granule_shift = memory_granule_shift,
        .reserved2 = 0,
        .reserved3 = 0,
    };
    const iree_hal_amdgpu_tsan_queue_initialize_args_t initialize_args = {
        .queue_state = queue_state,
        .shadow_base = shadow_base,
        .shadow_size = shadow_size,
        .queue_state_template_value = host_state,
    };
    status = iree_hal_amdgpu_host_queue_submit_tsan_state_initialize(
        queue, &initialize_args);
    if (iree_status_is_ok(status)) {
      queue->tsan.allocation_base = allocation_base;
      queue->tsan.allocation_size = allocation_size;
      queue->tsan.host_state = host_state;
      queue->tsan.queue_state = queue_state;
      queue->tsan.shadow_base = shadow_base;
      queue->tsan.shadow_size = shadow_size;
    } else {
      status = iree_status_join(
          status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(queue->libhsa),
                                                allocation_base));
    }
  } else {
    status = iree_status_join(
        status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(queue->libhsa),
                                              allocation_base));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize_tsan_state(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->tsan.allocation_base) return;
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_memory_pool_free_raw(
      queue->libhsa, queue->tsan.allocation_base));
  memset(&queue->tsan, 0, sizeof(queue->tsan));
}

static void iree_hal_amdgpu_host_queue_retire_reclaim_entry(
    iree_hal_amdgpu_reclaim_entry_t* entry,
    iree_hal_amdgpu_reclaim_retire_flags_t flags, void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  if (iree_any_bit_set(flags, IREE_HAL_AMDGPU_RECLAIM_RETIRE_FLAG_FAILED)) {
    return;
  }
  iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation = {
      .first_event_position = entry->profile_event_first_position,
      .event_count = entry->profile_event_count,
  };
  iree_hal_amdgpu_host_queue_retire_profile_dispatch_events(queue, reservation);
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      queue_device_reservation = {
          .first_event_position = entry->queue_device_event_first_position,
          .event_count = entry->queue_device_event_count,
      };
  iree_hal_amdgpu_host_queue_retire_profile_queue_device_events(
      queue, queue_device_reservation);
}

static void iree_hal_amdgpu_host_queue_reclaim_retired(
    iree_hal_amdgpu_reclaim_entry_t* entry, uint64_t epoch,
    iree_hal_amdgpu_reclaim_retire_flags_t flags, void* user_data) {
  (void)epoch;
  iree_hal_amdgpu_host_queue_retire_reclaim_entry(entry, flags, user_data);
}

static void iree_hal_amdgpu_host_queue_reclaim_retired_with_feedback(
    iree_hal_amdgpu_reclaim_entry_t* entry, uint64_t epoch,
    iree_hal_amdgpu_reclaim_retire_flags_t flags, void* user_data) {
  (void)epoch;
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;
  iree_hal_amdgpu_feedback_state_drain_physical_device(queue->feedback_state,
                                                       queue->device_ordinal);
  iree_hal_amdgpu_host_queue_retire_reclaim_entry(entry, flags, queue);
}

static iree_hal_amdgpu_reclaim_retire_fn_t
iree_hal_amdgpu_host_queue_reclaim_retire_fn(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->feedback_state
             ? iree_hal_amdgpu_host_queue_reclaim_retired_with_feedback
             : iree_hal_amdgpu_host_queue_reclaim_retired;
}

static void iree_hal_amdgpu_host_queue_reclaim_queue_owned_positions(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_reclaim_positions_t reclaim_positions) {
  if (reclaim_positions.kernarg_write_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(
        &queue->kernarg_ring, reclaim_positions.kernarg_write_position);
  }
  if (reclaim_positions.queue_upload_write_position > 0) {
    IREE_ASSERT(queue->queue_upload_ring.base,
                "queue upload bytes retired without an initialized upload "
                "ring");
    iree_hal_amdgpu_queue_upload_ring_reclaim(
        &queue->queue_upload_ring,
        reclaim_positions.queue_upload_write_position);
  }
}

//===----------------------------------------------------------------------===//
// Initialization / deinitialization
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_post_drain_action_t* action,
    iree_hal_amdgpu_host_queue_post_drain_fn_t fn, void* user_data) {
  action->next = NULL;
  action->fn = fn;
  action->user_data = user_data;

  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  if (queue->post_drain.tail) {
    queue->post_drain.tail->next = action;
  } else {
    queue->post_drain.head = action;
  }
  queue->post_drain.tail = action;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
}

static void iree_hal_amdgpu_host_queue_run_post_drain_actions(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  iree_hal_amdgpu_host_queue_post_drain_action_t* action =
      queue->post_drain.head;
  queue->post_drain.head = NULL;
  queue->post_drain.tail = NULL;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);

  while (action) {
    iree_hal_amdgpu_host_queue_post_drain_action_t* next_action = action->next;
    action->next = NULL;
    action->fn(action->user_data);
    action = next_action;
  }
}

void iree_hal_amdgpu_host_queue_query_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_scope_t* out_scope) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_scope);

  *out_scope = (iree_hal_amdgpu_queue_scope_t){
      .queue_affinity = queue->queue_affinity,
      .queue_ordinal = queue->queue_ordinal,
      .physical_device_ordinal = queue->device_ordinal,
      .physical_queue_ordinal = queue->physical_queue_ordinal,
      .aql_ring_base = (uint64_t)(uintptr_t)queue->aql_ring.base,
      .aql_ring_mask = queue->aql_ring.mask,
      .tsan =
          {
              .queue_state_base = (uint64_t)(uintptr_t)queue->tsan.queue_state,
              .shadow_base = queue->tsan.host_state.shadow_base,
              .shadow_size = queue->tsan.host_state.shadow_size,
              .dispatch_shadow_stride =
                  queue->tsan.host_state.dispatch_shadow_stride,
              .workgroup_shadow_stride =
                  queue->tsan.host_state.workgroup_shadow_stride,
              .workgroup_capacity = queue->tsan.host_state.workgroup_capacity,
              .shadow_entry_size = queue->tsan.host_state.shadow_entry_size,
              .memory_granule_shift =
                  queue->tsan.host_state.memory_granule_shift,
              .shadow_slot_count = queue->tsan.host_state.shadow_slot_count,
          },
  };
}

// Drains completed notification entries and reclaims kernarg space. If the GPU
// queue has faulted (error_status is set), fails all pending entries instead of
// draining normally. Caller must hold completion_drain_mutex.
static iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions_locked(
    iree_hal_amdgpu_host_queue_t* queue) {
  // Check for GPU queue error (set by the HSA error callback on another
  // thread). If the queue has faulted, no further epochs will advance;
  // fail all pending entries so waiters get the actual GPU error instead
  // of hanging or timing out.
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  const uint64_t previous_epoch = (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_relaxed);
  iree_hal_amdgpu_reclaim_positions_t reclaim_positions = {0};
  iree_host_size_t count = 0;
  if (IREE_UNLIKELY(error)) {
    count = iree_hal_amdgpu_notification_ring_fail_all_reclaim_positions(
        &queue->notification_ring, error,
        iree_hal_amdgpu_host_queue_reclaim_retire_fn(queue), queue,
        &reclaim_positions);
    iree_hal_amdgpu_host_queue_clear_profile_events(queue);
    iree_async_frontier_tracker_fail_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(iree_status_code(error)));
  } else {
    count = iree_hal_amdgpu_notification_ring_drain_reclaim_positions(
        &queue->notification_ring,
        /*fallback_frontier=*/NULL,
        iree_hal_amdgpu_host_queue_reclaim_retire_fn(queue), queue,
        &reclaim_positions);
    const uint64_t current_epoch =
        (uint64_t)iree_atomic_load(&queue->notification_ring.epoch.last_drained,
                                   iree_memory_order_acquire);
    if (current_epoch > previous_epoch) {
      iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                          current_epoch);
    }
  }
  iree_hal_amdgpu_host_queue_reclaim_queue_owned_positions(queue,
                                                           reclaim_positions);
  return count;
}

iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  iree_host_size_t count =
      iree_hal_amdgpu_host_queue_drain_completions_locked(queue);
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
  return count;
}

iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions_for_waiter(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.completion_drain_mutex);
  iree_host_size_t count =
      iree_hal_amdgpu_host_queue_drain_completions_locked(queue);
  iree_slim_mutex_unlock(&queue->locks.completion_drain_mutex);
  return count;
}

// Returns the queue's recorded failure without transferring ownership. The
// queue owns the status until it is taken by deinitialization.
static iree_status_t iree_hal_amdgpu_host_queue_error_status(
    iree_hal_amdgpu_host_queue_t* queue) {
  return (iree_status_t)iree_atomic_load(&queue->error_status,
                                         iree_memory_order_acquire);
}

static bool iree_hal_amdgpu_host_queue_has_error(
    iree_hal_amdgpu_host_queue_t* queue) {
  return !iree_status_is_ok(iree_hal_amdgpu_host_queue_error_status(queue));
}

static iree_status_t iree_hal_amdgpu_host_queue_clone_error_status(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_status_t error = iree_hal_amdgpu_host_queue_error_status(queue);
  return iree_status_is_ok(error) ? iree_ok_status() : iree_status_clone(error);
}

static bool iree_hal_amdgpu_host_queue_store_error(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t error) {
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &queue->error_status, &expected, (intptr_t)error,
          iree_memory_order_release, iree_memory_order_acquire)) {
    return true;
  }
  iree_status_free(error);
  return false;
}

static void iree_hal_amdgpu_host_queue_request_completion_thread_stop(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (queue->completion.stop_signal.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(queue->libhsa),
                                    queue->completion.stop_signal, 1);
  }
}

// Permanently closes the queue to further submission. Idempotent, and the only
// writer of |is_shutting_down|.
//
// Every notification epoch is published under submission_mutex, so taking the
// mutex here waits out any publisher already inside the critical section and
// keeps any later one out. A drain that runs after this returns therefore sees
// a published epoch that can no longer advance.
static void iree_hal_amdgpu_host_queue_close_submission(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
}

void iree_hal_amdgpu_host_queue_record_failure(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(queue);
  if (iree_hal_amdgpu_host_queue_store_error(queue, status)) {
    iree_hal_amdgpu_host_queue_request_completion_thread_stop(queue);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_wait_for_setup_epoch(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t epoch) {
  IREE_ASSERT_ARGUMENT(queue);
  if (epoch == 0) return iree_ok_status();
  if (!queue->hardware_queue || !queue->notification_ring.epoch.signal.handle) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU host queue cannot wait for epoch %" PRIu64
                            " without an active hardware queue",
                            epoch);
  }
  if (IREE_UNLIKELY(epoch > (uint64_t)IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU host queue epoch %" PRIu64
                            " exceeds representable HSA signal range",
                            epoch);
  }

  hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  hsa_signal_t stop_signal = queue->completion.stop_signal;
  const hsa_signal_value_t target_signal_value =
      (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - epoch);
  const hsa_signal_value_t compare_value = target_signal_value + 1;

  if (stop_signal.handle) {
    enum {
      IREE_HAL_AMDGPU_EPOCH_WAIT_EPOCH_SIGNAL = 0,
      IREE_HAL_AMDGPU_EPOCH_WAIT_STOP_SIGNAL = 1,
      IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT = 2,
    };
    hsa_signal_t signals[IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT] = {
        epoch_signal,
        stop_signal,
    };
    hsa_signal_condition_t conditions[IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT] =
        {
            HSA_SIGNAL_CONDITION_LT,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT] = {
        compare_value,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(queue->libhsa), IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT,
        signals, conditions, values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);
    if (IREE_UNLIKELY(signal_index >=
                      IREE_HAL_AMDGPU_EPOCH_WAIT_SIGNAL_COUNT)) {
      iree_status_t error = iree_make_status(
          IREE_STATUS_INTERNAL,
          "hsa_amd_signal_wait_any returned invalid signal index %u while "
          "waiting for AMDGPU host queue epoch %" PRIu64,
          signal_index, epoch);
      iree_hal_amdgpu_host_queue_record_failure(queue,
                                                iree_status_clone(error));
      return error;
    } else if (signal_index == IREE_HAL_AMDGPU_EPOCH_WAIT_STOP_SIGNAL) {
      iree_status_t error =
          iree_hal_amdgpu_host_queue_clone_error_status(queue);
      if (!iree_status_is_ok(error)) return error;
      return iree_make_status(IREE_STATUS_CANCELLED,
                              "AMDGPU host queue stopped while waiting for "
                              "epoch %" PRIu64,
                              epoch);
    }
  } else {
    (void)iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(queue->libhsa), epoch_signal, HSA_SIGNAL_CONDITION_LT,
        compare_value, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_clone_error_status(queue));
  iree_hal_amdgpu_host_queue_drain_completions_for_waiter(queue);
  return iree_hal_amdgpu_host_queue_clone_error_status(queue);
}

static hsa_signal_value_t iree_hal_amdgpu_host_queue_last_drained_signal_value(
    iree_hal_amdgpu_host_queue_t* queue) {
  const uint64_t last_drained_epoch = (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_acquire);
  return (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                              last_drained_epoch);
}

static void iree_hal_amdgpu_host_queue_wait_idle_before_teardown(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->hardware_queue || !queue->notification_ring.epoch.signal.handle) {
    return;
  }
  const uint64_t submitted_epoch =
      queue->notification_ring.epoch.next_submission;
  if (submitted_epoch == 0 || iree_hal_amdgpu_host_queue_has_error(queue)) {
    return;
  }

  hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  hsa_signal_t stop_signal = queue->completion.stop_signal;
  const hsa_signal_value_t idle_epoch_signal_value =
      (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                           submitted_epoch);
  const hsa_signal_value_t idle_compare_value = idle_epoch_signal_value + 1;

  // Submission is already shut out. Wait only for hardware to publish the final
  // submitted epoch so no late CP write can race with HSA queue/signal
  // teardown; the completion thread still owns notification-ring draining.
  if (stop_signal.handle) {
    enum {
      IREE_HAL_AMDGPU_IDLE_WAIT_EPOCH_SIGNAL = 0,
      IREE_HAL_AMDGPU_IDLE_WAIT_STOP_SIGNAL = 1,
      IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT = 2,
    };
    hsa_signal_t signals[IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT] = {
        epoch_signal,
        stop_signal,
    };
    hsa_signal_condition_t conditions[IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT] =
        {
            HSA_SIGNAL_CONDITION_LT,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT] = {
        idle_compare_value,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(queue->libhsa), IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT,
        signals, conditions, values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);
    if (IREE_UNLIKELY(signal_index >= IREE_HAL_AMDGPU_IDLE_WAIT_SIGNAL_COUNT)) {
      iree_status_t error = iree_make_status(
          IREE_STATUS_INTERNAL,
          "hsa_amd_signal_wait_any returned invalid signal index %u during "
          "AMDGPU host queue teardown",
          signal_index);
      iree_hal_amdgpu_host_queue_record_failure(queue, error);
    }
  } else {
    (void)iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(queue->libhsa), epoch_signal, HSA_SIGNAL_CONDITION_LT,
        idle_compare_value, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  }
}

// Completion thread entry point. Blocks in HSA until either the queue epoch
// signal changes or teardown/error signals the stop signal. Completion wakeups
// drain normally; stop/error wakeups perform one final drain/fail before exit.
static int iree_hal_amdgpu_host_queue_completion_thread_main(void* entry_arg) {
  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_host_queue_completion_thread_start");
    IREE_TRACE_ZONE_END(z0);
  }
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)entry_arg;

  enum {
    IREE_HAL_AMDGPU_COMPLETION_WAIT_EPOCH_SIGNAL = 0,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_STOP_SIGNAL = 1,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT = 2,
  };

  hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  hsa_signal_t stop_signal = queue->completion.stop_signal;
  hsa_signal_value_t last_epoch_value =
      iree_hal_amdgpu_host_queue_last_drained_signal_value(queue);

  bool keep_running = true;
  while (keep_running) {
    hsa_signal_t signals[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
        epoch_signal,
        stop_signal,
    };
    hsa_signal_condition_t
        conditions[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
            HSA_SIGNAL_CONDITION_NE,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
        last_epoch_value,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(queue->libhsa),
        IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT, signals, conditions,
        values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);

    {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z0, "iree_hal_amdgpu_host_queue_completion_thread_pump");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_index);

      if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_EPOCH_SIGNAL) {
        iree_hal_amdgpu_host_queue_drain_completions(queue);
        // Arm the next wait from the epoch we actually drained, not from a raw
        // HSA signal load. A GPU completion can race with the drain and update
        // the signal after drain() sampled it; observing that newer value here
        // would mark an undrained epoch as already seen and could sleep forever
        // with a user semaphore still pending.
        last_epoch_value =
            iree_hal_amdgpu_host_queue_last_drained_signal_value(queue);
      }

      // An out-of-range index means every signal handle passed was null or
      // invalid, so nothing can ever wake this wait again. Fail the queue and
      // take the terminal path below rather than spinning on a dead wait.
      if (IREE_UNLIKELY(signal_index >=
                        IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT)) {
        iree_hal_amdgpu_host_queue_record_failure(
            queue,
            iree_make_status(
                IREE_STATUS_INTERNAL,
                "hsa_amd_signal_wait_any returned invalid signal index %u",
                signal_index));
      }

      if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_STOP_SIGNAL ||
          iree_hal_amdgpu_host_queue_has_error(queue)) {
        // Close admission before the final drain so no epoch can be published
        // behind it. On the teardown path admission is already closed and this
        // is a no-op; on the failure path it is what makes the drain final.
        iree_hal_amdgpu_host_queue_close_submission(queue);
        iree_hal_amdgpu_host_queue_drain_completions(queue);
        if (iree_hal_amdgpu_host_queue_has_error(queue)) {
          // Capacity-parked pending operations are retried by post-drain
          // callbacks and the drain's own pass can enqueue more. Flush them
          // under closed admission so they observe cancellation and run their
          // normal failure path instead of being destroyed under the callback.
          iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
          iree_hal_amdgpu_host_queue_cancel_pending(
              queue, iree_hal_amdgpu_host_queue_error_status(queue));
        }
        keep_running = false;
      }

      IREE_TRACE_ZONE_END(z0);
    }
  }

  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_host_queue_completion_thread_exit");
    IREE_TRACE_ZONE_END(z0);
  }
  return 0;
}

// HSA queue error callback. Called by the HSA runtime (on an internal thread)
// when the queue encounters an unrecoverable error (page fault, invalid AQL
// packet, ECC error). Stores the error atomically on the queue so the
// completion drain path can fail pending semaphores with the actual GPU error.
static void iree_hal_amdgpu_host_queue_error_callback(hsa_status_t status,
                                                      hsa_queue_t* source,
                                                      void* data) {
  iree_hal_amdgpu_host_queue_t* queue = (iree_hal_amdgpu_host_queue_t*)data;

  // Convert the HSA error to an IREE status with diagnostic information.
  iree_status_t error = iree_status_from_hsa_status(
      __FILE__, __LINE__, status, "hsa_queue_error_callback",
      "GPU queue encountered an unrecoverable error");

  // First-error-wins: store the error with release semantics so the status
  // payload (heap-allocated string, backtrace) is visible to any thread that
  // loads with acquire. If another error already won the race, free ours.
  iree_hal_amdgpu_host_queue_record_failure(queue, error);
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_hal_device_t* logical_device,
    void* hostcall_buffer, iree_async_proactor_t* proactor,
    hsa_agent_t gpu_agent,
    const iree_hal_amdgpu_kernarg_ring_memory_t* kernarg_memory,
    hsa_amd_memory_pool_t pm4_ib_pool,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t queue_ordinal,
    iree_host_size_t physical_queue_ordinal,
    iree_thread_affinity_t completion_thread_affinity,
    iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_table,
    iree_hal_amdgpu_feedback_state_t* feedback_state,
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_host_queue_profiling_memory_t profiling_memory,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context,
    const iree_hal_pool_set_t* default_pool_set, iree_hal_pool_t* default_pool,
    iree_hal_amdgpu_transient_buffer_pool_t* transient_buffer_pool,
    iree_hal_amdgpu_staging_pool_t* staging_pool,
    iree_host_size_t device_ordinal, uint32_t aql_queue_capacity,
    uint32_t notification_capacity, uint32_t kernarg_capacity_in_blocks,
    uint32_t upload_capacity, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(kernarg_memory);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(epoch_table);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(transfer_context);
  IREE_ASSERT_ARGUMENT(default_pool_set);
  IREE_ASSERT_ARGUMENT(default_pool);
  IREE_ASSERT_ARGUMENT(transient_buffer_pool);
  IREE_ASSERT_ARGUMENT(out_queue);

  if (!iree_host_size_is_power_of_two(aql_queue_capacity) ||
      !iree_host_size_is_power_of_two(notification_capacity) ||
      !iree_host_size_is_power_of_two(kernarg_capacity_in_blocks) ||
      (upload_capacity != 0 &&
       !iree_host_size_is_power_of_two(upload_capacity))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "all enabled capacities must be powers of two");
  }
  if (kernarg_capacity_in_blocks / 2u < aql_queue_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ring capacity must be at least 2x the AQL ring capacity "
        "to cover one tail-padding gap at wrap (got kernarg_blocks=%u, "
        "aql_packets=%u)",
        kernarg_capacity_in_blocks, aql_queue_capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  out_queue->libhsa = libhsa;
  out_queue->logical_device = logical_device;
  out_queue->hostcall_buffer = hostcall_buffer;
  out_queue->proactor = proactor;
  out_queue->frontier_tracker = frontier_tracker;
  out_queue->host_allocator = host_allocator;

  // Submission pipeline state.
  iree_slim_mutex_initialize(&out_queue->locks.submission_mutex);
  iree_slim_mutex_initialize(&out_queue->locks.completion_drain_mutex);
  iree_slim_mutex_initialize(&out_queue->locks.post_drain_mutex);
  iree_slim_mutex_initialize(&out_queue->profiling.event_mutex);
  out_queue->profiling.memory = profiling_memory;
  out_queue->axis = axis;
  out_queue->wait_barrier_strategy = wait_barrier_strategy;
  out_queue->vendor_packet_capabilities = vendor_packet_capabilities;
  out_queue->pm4_timestamp_strategy = pm4_timestamp_strategy;
  out_queue->queue_affinity = queue_affinity;
  out_queue->queue_ordinal = queue_ordinal;
  out_queue->physical_queue_ordinal = physical_queue_ordinal;
  out_queue->last_signal.semaphore = NULL;
  out_queue->last_signal.epoch = 0;
  out_queue->feedback_state =
      iree_hal_amdgpu_feedback_state_is_enabled(feedback_state) ? feedback_state
                                                                : NULL;
  out_queue->block_pool = block_pool;
  out_queue->can_publish_frontier = true;
  out_queue->transfer_context = transfer_context;
  out_queue->default_pool_set = default_pool_set;
  out_queue->default_pool = default_pool;
  out_queue->transient_buffer_pool = transient_buffer_pool;
  out_queue->staging_pool = staging_pool;
  out_queue->device_ordinal = device_ordinal;
  out_queue->pending_head = NULL;
  iree_async_frontier_initialize(iree_hal_amdgpu_host_queue_frontier(out_queue),
                                 /*entry_count=*/0);

  // The optional tracker semaphore is an iree_async_semaphore_t bridge for
  // CPU-side wait integration. The queue's GPU-visible HSA epoch signal is
  // created by the notification ring below and registered in the epoch table.
  iree_status_t status = iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL);

  // Create the host-only stop signal before the hardware queue so the HSA error
  // callback always has a valid signal to wake if queue creation races with an
  // asynchronous fault.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_queue->completion.stop_signal);
  }

  // Create the HSA hardware AQL queue.
  //
  // HSA_QUEUE_TYPE_MULTI is required (not just an optimization). Once command
  // buffers start performing device-side enqueue, the CP itself becomes a
  // concurrent producer alongside the host submission path, so the queue must
  // permit multiple concurrent producers. The host-side reserve already uses
  // an atomic fetch_add on the write index, which is well-defined only on
  // MULTI queues.
  hsa_queue_t* hardware_queue = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_queue_create(
        IREE_LIBHSA(libhsa), gpu_agent, aql_queue_capacity,
        HSA_QUEUE_TYPE_MULTI, iree_hal_amdgpu_host_queue_error_callback,
        /*data=*/out_queue,
        /*private_segment_size=*/UINT32_MAX,
        /*group_segment_size=*/UINT32_MAX, &hardware_queue);
  }

  // Initialize the AQL ring from the hardware queue.
  if (iree_status_is_ok(status)) {
    out_queue->hardware_queue = hardware_queue;
    iree_hal_amdgpu_aql_ring_initialize(
        libhsa, (iree_amd_queue_t*)hardware_queue, &out_queue->aql_ring);
  }

  // Initialize the kernarg ring from the selected HSA memory pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_kernarg_ring_initialize(libhsa, kernarg_memory,
                                                     kernarg_capacity_in_blocks,
                                                     &out_queue->kernarg_ring);
  }

  // Initialize the optional queue-control upload ring from the same
  // host-visible memory policy as queue-owned kernargs. A zero capacity keeps
  // future device-side fixup storage opt-in and avoids charging every queue for
  // an unused allocation.
  if (iree_status_is_ok(status) && upload_capacity != 0) {
    const iree_hal_amdgpu_queue_upload_ring_memory_t upload_memory = {
        .memory_pool = kernarg_memory->memory_pool,
        .access_agents = kernarg_memory->access_agents,
        .access_agent_count = kernarg_memory->access_agent_count,
        .publication = kernarg_memory->publication,
    };
    status = iree_hal_amdgpu_queue_upload_ring_initialize(
        libhsa, &upload_memory, upload_capacity, &out_queue->queue_upload_ring);
  }

  // Initialize the optional PM4 IB slot buffer. Capability-driven allocation
  // keeps dynamic PM4 storage available on CDNA queues that use BARRIER_VALUE
  // for waits but still support AQL PM4-IB snippets for other features. The
  // buffer is indexed by AQL packet id and inherits AQL ring
  // backpressure/reuse; there is no separate PM4 producer or reclaim position.
  if (iree_status_is_ok(status) &&
      (vendor_packet_capabilities &
       IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB)) {
    status = iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
        libhsa, gpu_agent, pm4_ib_pool, aql_queue_capacity, out_queue);
  }

  // Initialize the notification ring (creates epoch signal + entry buffer).
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_notification_ring_initialize(
        libhsa, block_pool, notification_capacity, host_allocator,
        &out_queue->notification_ring);
  }

  // Register this queue's epoch signal in the shared table for cross-queue
  // barrier emission lookups. Must happen after notification ring init (which
  // creates the epoch signal) and before any submissions.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_epoch_signal_table_register(
        epoch_table, iree_async_axis_queue_index(axis),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &out_queue->notification_ring));
    out_queue->epoch_table = epoch_table;
  }

  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t thread_params;
    memset(&thread_params, 0, sizeof(thread_params));
    char thread_name[32] = {0};
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name), "amdgpu-d%uq%u-c",
             (unsigned)iree_async_axis_device_index(axis),
             (unsigned)iree_async_axis_queue_index(axis));
    thread_params.name = iree_make_cstring_view(thread_name);
    thread_params.initial_affinity = completion_thread_affinity;
    status = iree_thread_create(
        iree_hal_amdgpu_host_queue_completion_thread_main, out_queue,
        thread_params, host_allocator, &out_queue->completion.thread);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_deinitialize(out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_host_queue_close_submission(queue);

  iree_hal_amdgpu_host_queue_wait_idle_before_teardown(queue);

  if (queue->completion.thread) {
    iree_hal_amdgpu_host_queue_request_completion_thread_stop(queue);
    // There is only one owner for the thread, so this also joins the thread.
    iree_thread_release(queue->completion.thread);
    queue->completion.thread = NULL;
  }

  // Destroy the hardware queue before the remaining host-side resources so the
  // HSA runtime cannot race a late error callback against signal teardown.
  if (queue->hardware_queue) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_queue_destroy_raw(queue->libhsa, queue->hardware_queue));
    queue->hardware_queue = NULL;
  }

  // Capacity-parked pending ops are retried by post-drain callbacks. Flush
  // those callbacks under shutdown first so they observe cancellation and own
  // their normal failure path instead of being destroyed out from under the
  // callback storage.
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);

  // Cancel all pending (deferred) operations. Their signal semaphores are
  // failed so downstream waiters don't hang. A queue that failed has already
  // cancelled these on its completion thread with the recorded failure, so
  // anything still linked here was deferred during a clean teardown.
  if (queue->pending_head) {
    iree_status_t cancellation_status =
        iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
    iree_hal_amdgpu_host_queue_cancel_pending(queue, cancellation_status);
    iree_status_free(cancellation_status);
  }

  // Process any remaining notification entries before destroying resources.
  // If the GPU faulted, fail all pending entries so waiters get the actual
  // error. Otherwise drain normally (entries completed but not yet processed).
  //
  // Emptying the slot in the same step that takes ownership of the failure is
  // what makes the free below safe: a non-zero slot names a live status, and
  // every reader that dereferences it - including the post-drain pass this
  // function runs after the free - only clones from it.
  iree_status_t error = (iree_status_t)iree_atomic_exchange(
      &queue->error_status, 0, iree_memory_order_acq_rel);
  iree_hal_amdgpu_reclaim_positions_t reclaim_positions = {0};
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_notification_ring_fail_all_reclaim_positions(
        &queue->notification_ring, error,
        iree_hal_amdgpu_host_queue_reclaim_retire_fn(queue), queue,
        &reclaim_positions);
    iree_hal_amdgpu_host_queue_clear_profile_events(queue);
    iree_status_free(error);
  } else {
    iree_hal_amdgpu_notification_ring_drain_reclaim_positions(
        &queue->notification_ring,
        /*fallback_frontier=*/NULL,
        iree_hal_amdgpu_host_queue_reclaim_retire_fn(queue), queue,
        &reclaim_positions);
  }
  iree_hal_amdgpu_host_queue_reclaim_queue_owned_positions(queue,
                                                           reclaim_positions);
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);

  // Deregister from the epoch signal table before destroying the notification
  // ring (which owns the epoch signal). Guarded by epoch_table != NULL to
  // handle partial initialization (init failed before registration).
  if (queue->epoch_table) {
    iree_hal_amdgpu_epoch_signal_table_deregister(
        queue->epoch_table, iree_async_axis_queue_index(queue->axis));
    queue->epoch_table = NULL;
  }

  if (queue->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    queue->frontier_tracker = NULL;
    queue->axis = 0;
  }

  iree_hal_amdgpu_notification_ring_deinitialize(&queue->notification_ring);

  if (queue->queue_upload_ring.base) {
    iree_hal_amdgpu_queue_upload_ring_deinitialize(queue->libhsa,
                                                   &queue->queue_upload_ring);
  }

  iree_hal_amdgpu_host_queue_deinitialize_tsan_state(queue);

  iree_hal_amdgpu_kernarg_ring_deinitialize(queue->libhsa,
                                            &queue->kernarg_ring);

  if (queue->pm4_ib_slots) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(queue->libhsa, queue->pm4_ib_slots));
    queue->pm4_ib_slots = NULL;
  }

  iree_hal_amdgpu_host_queue_deallocate_profiling_completion_signals(queue);
  iree_hal_amdgpu_host_queue_deallocate_profile_events(queue);

  if (queue->command_buffer_scratch) {
    iree_allocator_free(queue->host_allocator, queue->command_buffer_scratch);
    queue->command_buffer_scratch = NULL;
  }

  if (queue->completion.stop_signal.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
        queue->libhsa, queue->completion.stop_signal));
    queue->completion.stop_signal.handle = 0;
  }

  iree_slim_mutex_deinitialize(&queue->locks.post_drain_mutex);
  iree_slim_mutex_deinitialize(&queue->locks.completion_drain_mutex);
  iree_slim_mutex_deinitialize(&queue->profiling.event_mutex);
  iree_slim_mutex_deinitialize(&queue->locks.submission_mutex);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(
    iree_hal_amdgpu_host_queue_t* queue, bool enabled) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, enabled ? 1 : 0);

  if (enabled) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_host_queue_ensure_profile_event_storage(queue));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_host_queue_ensure_profiling_completion_signals(queue));
    iree_hal_amdgpu_host_queue_clear_profile_events(queue);
  }

  iree_status_t status = iree_hsa_amd_profiling_set_profiler_enabled(
      IREE_LIBHSA(queue->libhsa), queue->hardware_queue, enabled ? 1 : 0);
  if (iree_status_is_ok(status)) {
    queue->profiling.hsa_queue_timestamps_enabled = enabled ? 1 : 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_host_queue_op_submission_t {
  // Queue whose submission_mutex is held between begin/end.
  iree_hal_amdgpu_host_queue_t* queue;

  // Wait resolution computed while holding submission_mutex.
  iree_hal_amdgpu_wait_resolution_t resolution;

  // Deferred operation captured while holding submission_mutex, if any.
  iree_hal_amdgpu_pending_op_t* deferred_op;

  // Number of input waits. Capacity retries only need post-drain resubmission
  // when no semantic waits are available to naturally re-enter the queue.
  iree_host_size_t wait_semaphore_count;

  // Whether the direct submit helper found enough queue capacity.
  bool ready;

  // Whether |deferred_op| should retry after notification-ring drain.
  bool wait_for_capacity;
} iree_hal_amdgpu_host_queue_op_submission_t;

// Begins one direct/deferred queue operation attempt. The caller must pair this
// with iree_hal_amdgpu_host_queue_op_submission_end exactly once.
static inline void iree_hal_amdgpu_host_queue_op_submission_begin(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_host_queue_op_submission_t* out_submission) {
  out_submission->queue = queue;
  out_submission->deferred_op = NULL;
  out_submission->wait_semaphore_count = wait_semaphore_list.count;
  out_submission->ready = true;
  out_submission->wait_for_capacity = false;

  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &out_submission->resolution);
}

// Marks a captured pending op as retrying after notification-ring drain because
// direct submission ran out of queue capacity.
static inline void iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(
    iree_hal_amdgpu_host_queue_op_submission_t* submission) {
  submission->wait_for_capacity = submission->wait_semaphore_count == 0;
}

// Ends one direct/deferred queue operation attempt by releasing
// submission_mutex and starting any captured pending op outside the lock.
static inline iree_status_t iree_hal_amdgpu_host_queue_op_submission_end(
    iree_hal_amdgpu_host_queue_op_submission_t* submission,
    iree_status_t status) {
  iree_slim_mutex_unlock(&submission->queue->locks.submission_mutex);

  if (iree_status_is_ok(status) && submission->deferred_op) {
    status = iree_hal_amdgpu_pending_op_start(submission->deferred_op,
                                              submission->wait_for_capacity);
  }
  return status;
}

typedef enum iree_hal_amdgpu_host_queue_buffer_state_e {
  IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY = 0,
  IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_UNSTAGED = 1,
  IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED = 2,
} iree_hal_amdgpu_host_queue_buffer_state_t;

// Classifies a queue_alloca wrapper for queue submission. The caller must hold
// submission_mutex so staging and decommit cannot race the query. Unstaged live
// wrappers must wait on their alloca signal through the host pending path;
// deallocated wrappers are terminal and cannot be submitted again.
static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_buffer_state(iree_hal_buffer_t* buffer) {
  if (!buffer) return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_amdgpu_transient_buffer_isa(allocated_buffer)) {
    return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  }
  if (iree_hal_amdgpu_transient_buffer_backing_buffer(allocated_buffer)) {
    return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  }
  if (iree_hal_amdgpu_transient_buffer_is_deallocated(allocated_buffer)) {
    return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED;
  }
  return IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_UNSTAGED;
}

static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_merge_buffer_states(
    iree_hal_amdgpu_host_queue_buffer_state_t lhs,
    iree_hal_amdgpu_host_queue_buffer_state_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_binding_table_state(
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_amdgpu_host_queue_buffer_state_t state =
      IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  if (!binding_table.bindings) return state;
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    state = iree_hal_amdgpu_host_queue_merge_buffer_states(
        state, iree_hal_amdgpu_host_queue_buffer_state(
                   binding_table.bindings[i].buffer));
    if (state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED) break;
  }
  return state;
}

static iree_hal_amdgpu_host_queue_buffer_state_t
iree_hal_amdgpu_host_queue_binding_refs_state(
    iree_hal_buffer_ref_list_t bindings) {
  iree_hal_amdgpu_host_queue_buffer_state_t state =
      IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_READY;
  if (!bindings.values) return state;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    state = iree_hal_amdgpu_host_queue_merge_buffer_states(
        state,
        iree_hal_amdgpu_host_queue_buffer_state(bindings.values[i].buffer));
    if (state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED) break;
  }
  return state;
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_buffer_state(
    iree_hal_amdgpu_host_queue_buffer_state_t state,
    iree_hal_semaphore_list_t wait_semaphore_list, bool* out_needs_staging) {
  *out_needs_staging =
      state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_UNSTAGED;
  if (state == IREE_HAL_AMDGPU_HOST_QUEUE_BUFFER_STATE_DEALLOCATED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue operation references a deallocated queue_alloca buffer");
  }
  if (*out_needs_staging && wait_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue operation references an unstaged queue_alloca buffer without "
        "waiting on its allocation signal");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_signal_empty_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);

  if (iree_status_is_ok(status)) {
    // Signal outside submission_mutex: semaphore signaling dispatches satisfied
    // timepoints, and those callbacks may submit additional queue work.
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_execute_flags(flags));

  if (!command_buffer && wait_semaphore_list.count == 0) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    }
    return iree_hal_amdgpu_host_queue_signal_empty_barrier(
        queue, signal_semaphore_list);
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      iree_hal_amdgpu_host_queue_binding_table_state(binding_table),
      wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_execute(
        queue, &wait_semaphore_list, &signal_semaphore_list, command_buffer,
        binding_table, flags, &submission.deferred_op);
  } else if (iree_status_is_ok(status) && !command_buffer) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    } else {
      uint64_t submission_id = 0;
      iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
          .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
          .operation_count = 0,
      };
      status = iree_hal_amdgpu_host_queue_try_submit_barrier(
          queue, &submission.resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){0},
          /*operation_resources=*/NULL,
          /*operation_resource_count=*/0, &profile_event_info,
          iree_hal_amdgpu_host_queue_post_commit_callback_null(),
          /*resource_set=*/NULL,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
          &submission.ready, &submission_id);
      if (iree_status_is_ok(status) && submission.ready) {
        profile_event_info.submission_id = submission_id;
        iree_hal_amdgpu_host_queue_record_profile_queue_event(
            queue, &submission.resolution, signal_semaphore_list,
            &profile_event_info);
      }
      if (iree_status_is_ok(status) && !submission.ready) {
        status = iree_hal_amdgpu_host_queue_defer_execute(
            queue, &wait_semaphore_list, &signal_semaphore_list,
            /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
            flags, &submission.deferred_op);
        iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(
            &submission);
      }
    }
  } else if (iree_status_is_ok(status)) {
    iree_hal_resource_set_t* binding_resource_set = NULL;
    status = iree_hal_amdgpu_host_queue_submit_command_buffer(
        queue, &submission.resolution, signal_semaphore_list, command_buffer,
        binding_table, flags, &binding_resource_set, &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      iree_hal_resource_set_free(binding_resource_set);
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list, command_buffer,
          binding_table, flags, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    } else if (!iree_status_is_ok(status)) {
      iree_hal_resource_set_free(binding_resource_set);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_enqueue_atomic(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      iree_hal_amdgpu_host_queue_buffer_state(operation->target_buffer),
      wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_atomic(
        queue, &wait_semaphore_list, &signal_semaphore_list, operation,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_atomic(
        queue, &submission.resolution, signal_semaphore_list, operation,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_atomic(
          queue, &wait_semaphore_list, &signal_semaphore_list, operation,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_wait(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  const iree_hal_amdgpu_host_queue_atomic_operation_t operation = {
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .kind = IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT,
      .params.wait = params,
  };
  return iree_hal_amdgpu_host_queue_enqueue_atomic(
      base_queue, wait_semaphore_list, signal_semaphore_list, &operation);
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_store(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  const iree_hal_amdgpu_host_queue_atomic_operation_t operation = {
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .kind = IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE,
      .params.store = params,
  };
  return iree_hal_amdgpu_host_queue_enqueue_atomic(
      base_queue, wait_semaphore_list, signal_semaphore_list, &operation);
}

static iree_status_t iree_hal_amdgpu_host_queue_atomic_rmw(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  const iree_hal_amdgpu_host_queue_atomic_operation_t operation = {
      .target_buffer = target_buffer,
      .target_offset = target_offset,
      .kind = IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW,
      .params.rmw = params,
  };
  return iree_hal_amdgpu_host_queue_enqueue_atomic(
      base_queue, wait_semaphore_list, signal_semaphore_list, &operation);
}

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_pool_t* allocation_pool = NULL;
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_alloca_wrapper(
      queue, pool, &params, allocation_size, flags, &allocation_pool, &buffer));
  // Always ask the pool to surface waitable death-frontier candidates so the
  // queue can distinguish true pool pressure from a dependency the caller did
  // not authorize. The HAL alloca flag is checked before consuming any
  // OK_NEEDS_WAIT reservation. Disallow growth while submission_mutex is held;
  // growable pools report that as a cold retry instead of calling into their
  // slab provider on the serialized queue path.
  const iree_hal_pool_reserve_flags_t reserve_flags =
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER |
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH;

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
  if (submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_alloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, allocation_pool,
        params, allocation_size, flags, reserve_flags, buffer,
        &submission.deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_alloca(
        queue, &submission.resolution, signal_semaphore_list, allocation_pool,
        params, allocation_size, flags, reserve_flags, buffer,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        /*pending_op=*/NULL, &memory_wait_op, &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready && !memory_wait_op) {
      status = iree_hal_amdgpu_host_queue_defer_alloca(
          queue, &wait_semaphore_list, &signal_semaphore_list, allocation_pool,
          params, allocation_size, flags, reserve_flags, buffer,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  status = iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
  if (iree_status_is_ok(status) && memory_wait_op) {
    iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  if (IREE_UNLIKELY(
          iree_any_bit_set(flags, ~(IREE_HAL_DEALLOCA_FLAG_NONE |
                                    IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported dealloca flags: 0x%" PRIx64, flags);
  }

  // iree_hal_device_queue_dealloca() applies PREFER_ORIGIN before vtable
  // dispatch by rewriting the device and queue affinity from the buffer's
  // allocation placement. Transient wrappers created by queue_alloca carry this
  // queue's one-bit affinity in that placement, so this host-queue path can use
  // |base_queue| directly.
  if (!iree_hal_amdgpu_transient_buffer_isa(buffer)) {
    return iree_hal_amdgpu_host_queue_execute(
        base_queue, wait_semaphore_list, signal_semaphore_list,
        /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  if (IREE_UNLIKELY(!iree_hal_amdgpu_transient_buffer_begin_dealloca(buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has already been queued for deallocation");
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  iree_status_t status = iree_ok_status();
  if (submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_dealloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, buffer,
        &submission.deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_dealloca(
        queue, &submission.resolution, signal_semaphore_list, buffer,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_dealloca(
          queue, &wait_semaphore_list, &signal_semaphore_list, buffer,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  status = iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_abort_dealloca(buffer);
  }
  return status;
}

// Queue fill entry point. Resolves waits under submission_mutex and captures a
// pending operation when waits, an unstaged target or submission capacity
// require deferral.
static iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      iree_hal_amdgpu_host_queue_buffer_state(target_buffer),
      wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_fill(
        queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
        target_offset, length, pattern_bits, pattern_length, flags,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_fill(
        queue, &submission.resolution, signal_semaphore_list, target_buffer,
        target_offset, length, pattern_bits, pattern_length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_fill(
          queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
          target_offset, length, pattern_bits, pattern_length, flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

// Queue timestamp entry point. Resolves waits under submission_mutex and
// captures a pending operation when waits, target staging, or submission
// capacity require deferral.
static iree_status_t iree_hal_amdgpu_host_queue_timestamp(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  // Reject unsupported flags synchronously at enqueue, before the
  // immediate/deferred split, so the error is deterministic regardless of
  // wait-list state (mirrors the up-front flag validation in queue_execute).
  if (IREE_UNLIKELY(flags != IREE_HAL_TIMESTAMP_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported timestamp flags: 0x%" PRIx64, flags);
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      iree_hal_amdgpu_host_queue_buffer_state(target_buffer),
      wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_timestamp(
        queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
        target_offset, flags, &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_timestamp(
        queue, &submission.resolution, signal_semaphore_list, target_buffer,
        target_offset, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_timestamp(
          queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
          target_offset, flags, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

iree_status_t iree_hal_amdgpu_host_queue_copy_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type) {
  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  iree_hal_amdgpu_host_queue_buffer_state_t buffer_state =
      iree_hal_amdgpu_host_queue_buffer_state(source_buffer);
  buffer_state = iree_hal_amdgpu_host_queue_merge_buffer_states(
      buffer_state, iree_hal_amdgpu_host_queue_buffer_state(target_buffer));
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      buffer_state, wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_copy(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        profile_event_type, &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_copy(
        queue, &submission.resolution, signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        profile_event_type,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_copy(
          queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
          source_offset, target_buffer, target_offset, length, flags,
          profile_event_type, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

// Queue copy entry point. The shared copy path is also used by file read/write
// staging so all copy-shaped operations use the same wait/backpressure path.
static iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_hal_amdgpu_host_queue_copy_buffer(
      (iree_hal_amdgpu_host_queue_t*)base_queue, wait_semaphore_list,
      signal_semaphore_list, source_buffer, source_offset, target_buffer,
      target_offset, length, flags, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY);
}

// Queue update entry point. Immediate updates copy into queue-owned kernarg
// memory; deferred updates copy into the pending-op arena.
static iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      iree_hal_amdgpu_host_queue_buffer_state(target_buffer),
      wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    status = iree_hal_amdgpu_host_queue_defer_update(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        &submission.deferred_op);
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_update(
        queue, &submission.resolution, signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_update(
          queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
          source_offset, target_buffer, target_offset, length, flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static bool iree_hal_amdgpu_host_queue_is_noop_dispatch(
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  return !iree_hal_dispatch_uses_indirect_parameters(flags) &&
         (config.workgroup_count[0] | config.workgroup_count[1] |
          config.workgroup_count[2]) == 0;
}

// Queue dispatch entry point. Empty direct dispatches route through the barrier
// path so they still signal semaphores and profile as dispatch submissions.
static iree_status_t iree_hal_amdgpu_host_queue_dispatch(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;
  const bool is_noop_dispatch =
      iree_hal_amdgpu_host_queue_is_noop_dispatch(config, flags);

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  iree_hal_amdgpu_host_queue_buffer_state_t buffer_state =
      iree_hal_amdgpu_host_queue_binding_refs_state(bindings);
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    buffer_state = iree_hal_amdgpu_host_queue_merge_buffer_states(
        buffer_state, iree_hal_amdgpu_host_queue_buffer_state(
                          config.workgroup_count_ref.buffer));
  }
  bool needs_staging = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_validate_buffer_state(
      buffer_state, wait_semaphore_list, &needs_staging);
  if (iree_status_is_ok(status) &&
      (submission.resolution.needs_deferral || needs_staging)) {
    if (is_noop_dispatch) {
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list,
          /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_EXECUTE_FLAG_NONE, &submission.deferred_op);
    } else {
      status = iree_hal_amdgpu_host_queue_defer_dispatch(
          queue, &wait_semaphore_list, &signal_semaphore_list, executable,
          export_ordinal, config, constants, bindings, flags,
          &submission.deferred_op);
    }
  } else if (iree_status_is_ok(status) && is_noop_dispatch) {
    uint64_t submission_id = 0;
    iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
        .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH,
        .operation_count = 0,
    };
    status = iree_hal_amdgpu_host_queue_try_submit_barrier(
        queue, &submission.resolution, signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){0},
        /*operation_resources=*/NULL,
        /*operation_resource_count=*/0, &profile_event_info,
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready, &submission_id);
    if (iree_status_is_ok(status) && submission.ready) {
      profile_event_info.submission_id = submission_id;
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, &submission.resolution, signal_semaphore_list,
          &profile_event_info);
    }
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list,
          /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_EXECUTE_FLAG_NONE, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_submit_dispatch(
        queue, &submission.resolution, signal_semaphore_list, executable,
        export_ordinal, config, constants, bindings, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_dispatch(
          queue, &wait_semaphore_list, &signal_semaphore_list, executable,
          export_ordinal, config, constants, bindings, flags,
          &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_read(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_hal_amdgpu_host_queue_read_file(
      base_queue, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_hal_amdgpu_host_queue_write_file(
      base_queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags);
}

iree_status_t iree_hal_amdgpu_host_queue_enqueue_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count) {
  if (IREE_UNLIKELY(!action.fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host action callback must be non-null");
  }
  if (IREE_UNLIKELY(operation_resource_count > 0 && !operation_resources)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host action resources must be non-null");
  }

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  // Host actions execute on CPU threads and must observe device-produced
  // host-visible memory even when a semaphore edge itself is device-local.
  submission.resolution.inline_acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          submission.resolution.inline_acquire_scope,
          IREE_HSA_FENCE_SCOPE_SYSTEM);
  submission.resolution.barrier_acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          submission.resolution.barrier_acquire_scope,
          IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_status_t status = iree_ok_status();
  if (submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_host_action(
        queue, &wait_semaphore_list, action, operation_resources,
        operation_resource_count, &submission.deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_try_submit_barrier(
        queue, &submission.resolution, iree_hal_semaphore_list_empty(), action,
        operation_resources, operation_resource_count,
        /*profile_event_info=*/NULL,
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        &submission.ready, /*out_submission_id=*/NULL);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_host_action(
          queue, &wait_semaphore_list, action, operation_resources,
          operation_resource_count, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_host_call(call, args, flags));

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_amdgpu_host_queue_op_submission_t submission;
  iree_hal_amdgpu_host_queue_op_submission_begin(queue, wait_semaphore_list,
                                                 &submission);
  iree_status_t status = iree_ok_status();
  if (submission.resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_host_call(
        queue, &wait_semaphore_list, &signal_semaphore_list, call, args, flags,
        &submission.deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_host_call(
        queue, &submission.resolution, signal_semaphore_list, call, args, flags,
        &submission.ready);
    if (iree_status_is_ok(status) && !submission.ready) {
      status = iree_hal_amdgpu_host_queue_defer_host_call(
          queue, &wait_semaphore_list, &signal_semaphore_list, call, args,
          flags, &submission.deferred_op);
      iree_hal_amdgpu_host_queue_op_submission_defer_for_capacity(&submission);
    }
  }
  return iree_hal_amdgpu_host_queue_op_submission_end(&submission, status);
}

static iree_status_t iree_hal_amdgpu_host_queue_flush(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Virtual queue vtable
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_deinitialize_vtable(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  iree_hal_amdgpu_host_queue_deinitialize(
      (iree_hal_amdgpu_host_queue_t*)base_queue);
}

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable = {
        .deinitialize = iree_hal_amdgpu_host_queue_deinitialize_vtable,
        .trim = iree_hal_amdgpu_host_queue_trim,
        .alloca = iree_hal_amdgpu_host_queue_alloca,
        .dealloca = iree_hal_amdgpu_host_queue_dealloca,
        .fill = iree_hal_amdgpu_host_queue_fill,
        .update = iree_hal_amdgpu_host_queue_update,
        .copy = iree_hal_amdgpu_host_queue_copy,
        .read = iree_hal_amdgpu_host_queue_read,
        .write = iree_hal_amdgpu_host_queue_write,
        .host_call = iree_hal_amdgpu_host_queue_host_call,
        .dispatch = iree_hal_amdgpu_host_queue_dispatch,
        .execute = iree_hal_amdgpu_host_queue_execute,
        .atomic_wait = iree_hal_amdgpu_host_queue_atomic_wait,
        .atomic_store = iree_hal_amdgpu_host_queue_atomic_store,
        .atomic_rmw = iree_hal_amdgpu_host_queue_atomic_rmw,
        .timestamp = iree_hal_amdgpu_host_queue_timestamp,
        .flush = iree_hal_amdgpu_host_queue_flush,
};
