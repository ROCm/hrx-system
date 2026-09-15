// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/kernel_queue.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/wait.h"
#include "libamdf/src/xdna/context.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/umd/kernel_queue.h"

typedef uint32_t amdf_xdna_kernel_queue_occupancy_t;
enum amdf_xdna_kernel_queue_occupancy_e {
  AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE = 0,
  AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RESERVING = 1,
  AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_PENDING = 2,
  AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RETIRING = 3,
};

enum {
  AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_BITS = 2,
};

#define AMDF_XDNA_KERNEL_QUEUE_MAXIMUM_SUBMISSION \
  (UINT64_MAX >> AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_BITS)

typedef struct amdf_xdna_kernel_queue_t {
  // Generic kernel-queue state shared by every engine implementation.
  amdf_kernel_queue_t base;
  // Exact scheduling context borrowed by this queue.
  amdf_xdna_context_t* context;
  // Exact native kernel-mediated queue lease.
  amdf_xdna_umd_kernel_queue_t* umd;
  // Submission sequence and ownership state of the single pending slot.
  amdf_atomic_uint64_t slot_state;
  // Instruction backing borrowed until checked native retirement.
  amdf_memory_t* pending_memory;
  // Native progress value covering the pending submission.
  amdf_atomic_uint64_t pending_native_submission;
} amdf_xdna_kernel_queue_t;

_Static_assert(offsetof(amdf_xdna_kernel_queue_t, base) == 0,
               "XDNA kernel queue base must be the first field");

static uint64_t amdf_xdna_kernel_queue_make_slot_state(
    uint64_t submission, amdf_xdna_kernel_queue_occupancy_t occupancy) {
  return (submission << AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_BITS) | occupancy;
}

static uint64_t amdf_xdna_kernel_queue_slot_submission(uint64_t slot_state) {
  return slot_state >> AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_BITS;
}

static amdf_xdna_kernel_queue_occupancy_t amdf_xdna_kernel_queue_slot_occupancy(
    uint64_t slot_state) {
  return (
      amdf_xdna_kernel_queue_occupancy_t)(slot_state &
                                          ((UINT64_C(1)
                                            << AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_BITS) -
                                           1));
}

// Samples native progress and claims completed software retirement exactly
// once. A true result proves that `submission` has completely retired,
// including release of its instruction-memory borrow.
static bool amdf_xdna_kernel_queue_try_retire(amdf_xdna_kernel_queue_t* queue,
                                              uint64_t submission) {
  uint64_t slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
  const uint64_t slot_submission =
      amdf_xdna_kernel_queue_slot_submission(slot_state);
  const amdf_xdna_kernel_queue_occupancy_t occupancy =
      amdf_xdna_kernel_queue_slot_occupancy(slot_state);
  if (submission < slot_submission ||
      (submission == slot_submission &&
       (occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE ||
        occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RESERVING))) {
    return true;
  }
  if (submission != slot_submission ||
      occupancy != AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_PENDING) {
    return false;
  }

  const uint64_t pending_native_submission =
      amdf_atomic_uint64_load_acquire(&queue->pending_native_submission);
  if (amdf_xdna_umd_kernel_queue_query_progress(queue->umd) <
      pending_native_submission) {
    return false;
  }

  const uint64_t retiring_state = amdf_xdna_kernel_queue_make_slot_state(
      submission, AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RETIRING);
  if (!amdf_atomic_uint64_compare_exchange_acq_rel(
          &queue->slot_state, &slot_state, retiring_state)) {
    const uint64_t observed_submission =
        amdf_xdna_kernel_queue_slot_submission(slot_state);
    const amdf_xdna_kernel_queue_occupancy_t observed_occupancy =
        amdf_xdna_kernel_queue_slot_occupancy(slot_state);
    return submission < observed_submission ||
           (submission == observed_submission &&
            (observed_occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE ||
             observed_occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RESERVING));
  }

  amdf_xdna_umd_kernel_queue_retire_command(queue->umd);
  amdf_memory_unregister_child(queue->pending_memory);
  queue->pending_memory = NULL;
  amdf_atomic_uint64_store_release(
      &queue->slot_state,
      amdf_xdna_kernel_queue_make_slot_state(
          submission, AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE));
  return true;
}

static amdf_status_t amdf_xdna_kernel_queue_query_status(
    amdf_kernel_queue_t* base_queue, amdf_kernel_queue_status_t* out_status) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  uint64_t slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
  uint64_t slot_submission = amdf_xdna_kernel_queue_slot_submission(slot_state);
  if (amdf_xdna_kernel_queue_slot_occupancy(slot_state) ==
      AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_PENDING) {
    amdf_xdna_kernel_queue_try_retire(queue, slot_submission);
    slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
    slot_submission = amdf_xdna_kernel_queue_slot_submission(slot_state);
  }
  const amdf_xdna_kernel_queue_occupancy_t occupancy =
      amdf_xdna_kernel_queue_slot_occupancy(slot_state);
  out_status->retired_submission =
      occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE ||
              occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RESERVING
          ? slot_submission
          : slot_submission - 1;
  out_status->terminal_status =
      amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
  out_status->state = amdf_status_is_ok(out_status->terminal_status)
                          ? AMDF_QUEUE_STATE_ACTIVE
                          : AMDF_QUEUE_STATE_DEVICE_LOST;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_xdna_kernel_queue_wait(
    amdf_kernel_queue_t* base_queue, uint64_t submission,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  uint64_t slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
  if (submission > amdf_xdna_kernel_queue_slot_submission(slot_state)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_wait_deadline_t deadline;
  amdf_status_t status = amdf_wait_deadline_initialize(
      timeout_nanoseconds, poll_duration_nanoseconds, &deadline);
  if (!amdf_status_is_ok(status)) return status;
  // Even an expired deadline permits the first nonblocking native refresh.
  bool native_polled = false;
  for (;;) {
    const bool retired = amdf_xdna_kernel_queue_try_retire(queue, submission);
    const amdf_status_t terminal_status =
        amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
    if (!amdf_status_is_ok(terminal_status) || retired) return terminal_status;

    slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
    const uint64_t slot_submission =
        amdf_xdna_kernel_queue_slot_submission(slot_state);
    const amdf_xdna_kernel_queue_occupancy_t occupancy =
        amdf_xdna_kernel_queue_slot_occupancy(slot_state);
    if (submission < slot_submission ||
        (submission == slot_submission &&
         (occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE ||
          occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RESERVING))) {
      return amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
    }
    amdf_wait_budget_t remaining;
    status = amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(status)) return status;
    if (remaining.timeout == 0 &&
        (native_polled ||
         occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RETIRING)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (occupancy == AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RETIRING) {
      amdf_platform_wait_yield();
      continue;
    }
    if (occupancy != AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_PENDING ||
        submission != slot_submission) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    const uint64_t pending_native_submission =
        amdf_atomic_uint64_load_acquire(&queue->pending_native_submission);
    if (slot_state != amdf_atomic_uint64_load_acquire(&queue->slot_state)) {
      continue;
    }
    native_polled = true;
    status = amdf_xdna_umd_kernel_queue_wait(
        queue->umd, pending_native_submission, &deadline);
    // A wait error is observable even if another thread established progress.
    // Still release the borrow when independently confirmed retirement permits
    // it.
    amdf_xdna_kernel_queue_try_retire(queue, submission);
    if (!amdf_status_is_ok(status)) return status;
  }
}

static amdf_status_t amdf_xdna_kernel_queue_destroy_native(
    amdf_kernel_queue_t* base_queue) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  uint64_t slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
  const uint64_t slot_submission =
      amdf_xdna_kernel_queue_slot_submission(slot_state);
  if (amdf_xdna_kernel_queue_slot_occupancy(slot_state) ==
      AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_PENDING) {
    amdf_xdna_kernel_queue_try_retire(queue, slot_submission);
    slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
  }
  if (amdf_xdna_kernel_queue_slot_occupancy(slot_state) !=
      AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_xdna_umd_kernel_queue_destroy(queue->umd);
  if (amdf_status_is_ok(status)) {
    queue->umd = NULL;
    amdf_xdna_context_unregister_child(queue->context);
    queue->context = NULL;
  }
  return status;
}

static const amdf_kernel_queue_vtable_t amdf_xdna_kernel_queue_vtable = {
    .query_status = amdf_xdna_kernel_queue_query_status,
    .wait = amdf_xdna_kernel_queue_wait,
    .destroy_native = amdf_xdna_kernel_queue_destroy_native,
};

amdf_status_t AMDF_CALL amdf_xdna_kernel_queue_create(
    amdf_xdna_context_t* context,
    const amdf_xdna_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue) {
  if (out_queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_device_t* device = amdf_xdna_context_get_device(context);
  amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      (uint32_t)sizeof(amdf_xdna_kernel_queue_create_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  amdf_queue_family_info_t family_info = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(family_info),
  };
  status = amdf_endpoint_query_queue_family_info(
      device->endpoint, create_info->queue_family_ordinal, &family_info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (family_info.command_type != AMDF_QUEUE_COMMAND_TYPE_XDNA ||
      (family_info.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) ==
          0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const amdf_allocator_t host_allocator = amdf_device_host_allocator(device);
  amdf_xdna_kernel_queue_t* queue = NULL;
  status = amdf_calloc(host_allocator, sizeof(*queue),
                       amdf_alignof(amdf_xdna_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) return status;
  const amdf_xdna_context_info_t* context_info =
      amdf_xdna_context_get_info(context);
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
      .device_id = context_info->device_id,
      .reset_epoch = context_info->reset_epoch,
      .queue_family_ordinal = family_info.ordinal,
      .command_type = family_info.command_type,
      .maximum_pending_submission_count = 1,
      .maximum_command_count = 1,
  };
  amdf_atomic_uint64_initialize(&queue->slot_state,
                                amdf_xdna_kernel_queue_make_slot_state(
                                    0, AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE));
  amdf_atomic_uint64_initialize(&queue->pending_native_submission, 0);
  status = amdf_xdna_context_register_child(context);
  const bool context_registered = amdf_status_is_ok(status);
  if (amdf_status_is_ok(status)) {
    status = amdf_kernel_queue_initialize(
        &queue->base, &amdf_xdna_kernel_queue_vtable, device, &info);
  }
  if (amdf_status_is_ok(status)) {
    queue->context = context;
    status = amdf_xdna_umd_kernel_queue_create(
        amdf_xdna_context_get_umd(context), &queue->umd);
  }
  if (amdf_status_is_ok(status)) {
    *out_queue = &queue->base;
  } else {
    if (queue->base.device != NULL) {
      amdf_kernel_queue_deinitialize(&queue->base);
    }
    if (context_registered) {
      amdf_xdna_context_unregister_child(context);
    }
    amdf_free(host_allocator, queue);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_xdna_kernel_queue_submit(
    amdf_kernel_queue_t* base_queue,
    const amdf_xdna_kernel_queue_submission_info_t* submission_info,
    uint64_t* out_submission) {
  if (base_queue == NULL || out_submission == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (base_queue->info.command_type != AMDF_QUEUE_COMMAND_TYPE_XDNA) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_structure_validate_input(
      submission_info, AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
      (uint32_t)sizeof(amdf_xdna_kernel_queue_submission_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (submission_info->reserved != 0 || submission_info->command_count != 1 ||
      submission_info->commands == NULL ||
      submission_info->commands[0].memory == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  const amdf_xdna_kernel_command_t* command = &submission_info->commands[0];
  const amdf_xdna_endpoint_info_t* endpoint_info =
      amdf_xdna_device_get_profile(base_queue->device)->info;
  amdf_memory_t* memory = command->memory;
  if (command->reserved != 0 ||
      memory->scope != amdf_xdna_context_get_memory_scope(queue->context)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (command->access_ordinal >= memory->info.access_count ||
      command->byte_length == 0 ||
      command->byte_length > endpoint_info->instruction.maximum_byte_length ||
      command->byte_length %
              endpoint_info->instruction.byte_length_granularity !=
          0 ||
      command->byte_offset > memory->info.byte_length ||
      command->byte_length > memory->info.byte_length - command->byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_access_state_t* access =
      &memory->accesses[command->access_ordinal];
  if ((access->info.access & AMDF_MEMORY_ACCESS_EXECUTE) == 0 ||
      (access->info.address_kinds &
       (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE)) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (base_queue->info.reset_epoch !=
          amdf_xdna_device_query_reset_epoch(base_queue->device) ||
      access->info.reset_epoch != base_queue->info.reset_epoch) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  const uint64_t instruction_address =
      access->addresses[AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE] +
      command->byte_offset;
  if (instruction_address % endpoint_info->instruction.address_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  uint64_t slot_state = amdf_atomic_uint64_load_acquire(&queue->slot_state);
  const uint64_t last_submitted =
      amdf_xdna_kernel_queue_slot_submission(slot_state);
  if (last_submitted == AMDF_XDNA_KERNEL_QUEUE_MAXIMUM_SUBMISSION) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  if (amdf_xdna_kernel_queue_slot_occupancy(slot_state) !=
      AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const uint64_t reserving_state = amdf_xdna_kernel_queue_make_slot_state(
      last_submitted, AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_RESERVING);
  if (!amdf_atomic_uint64_compare_exchange_acq_rel(
          &queue->slot_state, &slot_state, reserving_state)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }

  status = amdf_memory_register_child(memory);
  const bool memory_borrowed = amdf_status_is_ok(status);
  uint64_t native_submission = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_xdna_umd_kernel_queue_submit(queue->umd, instruction_address,
                                               (uint32_t)command->byte_length,
                                               &native_submission);
  }
  if (!amdf_status_is_ok(status)) {
    if (memory_borrowed) amdf_memory_unregister_child(memory);
    amdf_atomic_uint64_store_release(
        &queue->slot_state,
        amdf_xdna_kernel_queue_make_slot_state(
            last_submitted, AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_FREE));
    return status;
  }

  const uint64_t submission = last_submitted + 1;
  queue->pending_memory = memory;
  amdf_atomic_uint64_store_release(&queue->pending_native_submission,
                                   native_submission);
  amdf_atomic_uint64_store_release(
      &queue->slot_state,
      amdf_xdna_kernel_queue_make_slot_state(
          submission, AMDF_XDNA_KERNEL_QUEUE_OCCUPANCY_PENDING));
  *out_submission = submission;
  return AMDF_STATUS_OK;
}
