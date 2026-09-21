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

typedef struct amdf_xdna_kernel_queue_t {
  // Generic kernel-queue state shared by every engine implementation.
  amdf_kernel_queue_t base;
  // Exact scheduling context borrowed by this queue.
  amdf_xdna_context_t* context;
  // Native queue and its preallocated packet/result slots.
  amdf_xdna_umd_kernel_queue_t* umd;
  // Nonzero only while one host caller publishes a native submission.
  amdf_atomic_uint32_t publishing;
  // Nonzero while one observer consumes completed command results.
  amdf_atomic_uint32_t retiring;
  // Greatest publicly accepted queue-local point.
  amdf_atomic_uint64_t submitted;
  // Greatest accepted point whose native completion and result were checked.
  amdf_atomic_uint64_t retired;
  // Native identities for occupied slots. Atomic because an old waiter may
  // read a slot concurrently with reuse, then recheck retirement.
  amdf_atomic_uint64_t native_submissions[];
} amdf_xdna_kernel_queue_t;

_Static_assert(offsetof(amdf_xdna_kernel_queue_t, base) == 0,
               "XDNA kernel queue base must be the first field");

// Consumes a completed prefix exactly once before publishing reusable credits.
// The accepted snapshot excludes native work not yet published by submit.
static uint64_t amdf_xdna_kernel_queue_refresh_retirement(
    amdf_xdna_kernel_queue_t* queue) {
  uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  const uint64_t submitted = amdf_atomic_uint64_load_acquire(&queue->submitted);
  if (retired >= submitted) {
    return retired;
  }
  uint32_t expected = 0;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(&queue->retiring, &expected,
                                                   1)) {
    return amdf_atomic_uint64_load_acquire(&queue->retired);
  }
  retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  const uint64_t progress =
      amdf_xdna_umd_kernel_queue_query_progress(queue->umd);
  while (retired < submitted) {
    const uint32_t slot =
        retired % queue->base.info.maximum_pending_submission_count;
    if (amdf_atomic_uint64_load_acquire(&queue->native_submissions[slot]) >
        progress) {
      break;
    }
    amdf_xdna_umd_kernel_queue_retire_command(queue->umd, slot);
    ++retired;
  }
  amdf_atomic_uint64_store_release(&queue->retired, retired);
  amdf_atomic_uint32_store_release(&queue->retiring, 0);
  return retired;
}

static amdf_status_t amdf_xdna_kernel_queue_query_status(
    amdf_kernel_queue_t* base_queue, amdf_kernel_queue_status_t* out_status) {
  const amdf_xdna_kernel_queue_t* queue =
      (const amdf_xdna_kernel_queue_t*)base_queue;
  out_status->retired_submission =
      amdf_atomic_uint64_load_acquire(&queue->retired);
  out_status->terminal_status =
      amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
  out_status->state = amdf_status_is_ok(out_status->terminal_status)
                          ? AMDF_QUEUE_STATE_ACTIVE
                          : AMDF_QUEUE_STATE_DEVICE_LOST;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_xdna_kernel_queue_refresh_status(
    amdf_kernel_queue_t* base_queue, amdf_kernel_queue_status_t* out_status) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  if (amdf_atomic_uint64_load_acquire(&queue->retired) <
      amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    const amdf_status_t status =
        amdf_xdna_umd_kernel_queue_refresh_progress(queue->umd);
    // Independent known progress remains usable even if observation fails.
    amdf_xdna_kernel_queue_refresh_retirement(queue);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  return amdf_xdna_kernel_queue_query_status(base_queue, out_status);
}

static amdf_status_t amdf_xdna_kernel_queue_request_notification(
    amdf_kernel_queue_t* base_queue, uint64_t submission,
    const amdf_native_event_t* event) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  uint64_t native_submission = 0;
  if (amdf_atomic_uint64_load_acquire(&queue->retired) < submission) {
    const uint32_t slot =
        (submission - 1) % base_queue->info.maximum_pending_submission_count;
    native_submission =
        amdf_atomic_uint64_load_acquire(&queue->native_submissions[slot]);
    // A slot's new native identity is published after its old point retires.
    // An old-point request must never become a wait for that replacement.
    if (amdf_atomic_uint64_load_acquire(&queue->retired) >= submission) {
      native_submission = 0;
    }
  }
  return amdf_xdna_umd_kernel_queue_request_notification(
      queue->umd, native_submission, event);
}

static amdf_status_t amdf_xdna_kernel_queue_wait(
    amdf_kernel_queue_t* base_queue, uint64_t submission,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_wait_deadline_t deadline;
  amdf_status_t status = amdf_wait_deadline_initialize(
      timeout_nanoseconds, poll_duration_nanoseconds, &deadline);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  bool native_polled = false;
  for (;;) {
    const uint64_t retired = amdf_xdna_kernel_queue_refresh_retirement(queue);
    const amdf_status_t terminal_status =
        amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
    if (retired >= submission) {
      return terminal_status;
    }
    if (!amdf_status_is_ok(terminal_status)) {
      // Failure is not completion. A nonblocking refresh can still establish
      // independent retirement of accepted commands after an execution error.
      status = amdf_xdna_umd_kernel_queue_refresh_progress(queue->umd);
      amdf_xdna_kernel_queue_refresh_retirement(queue);
      return amdf_status_is_ok(status) ? terminal_status : status;
    }
    amdf_wait_budget_t remaining;
    status = amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (native_polled && remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    const uint32_t slot =
        (submission - 1) % base_queue->info.maximum_pending_submission_count;
    const uint64_t native_submission =
        amdf_atomic_uint64_load_acquire(&queue->native_submissions[slot]);
    // Reuse release-publishes the new native identity after retirement. An old
    // waiter that reads that identity must not wait for the new command.
    if (amdf_atomic_uint64_load_acquire(&queue->retired) >= submission) {
      return amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
    }
    if (amdf_xdna_umd_kernel_queue_query_progress(queue->umd) >=
        native_submission) {
      native_polled = true;
      amdf_platform_wait_yield();
      continue;
    }
    native_polled = true;
    status = amdf_xdna_umd_kernel_queue_wait(queue->umd, native_submission,
                                             &deadline);
    // Native wait errors remain observable independently of confirmed progress.
    amdf_xdna_kernel_queue_refresh_retirement(queue);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
}

static amdf_status_t amdf_xdna_kernel_queue_destroy_native(
    amdf_kernel_queue_t* base_queue) {
  amdf_xdna_kernel_queue_t* queue = (amdf_xdna_kernel_queue_t*)base_queue;
  if (amdf_xdna_kernel_queue_refresh_retirement(queue) <
      amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_xdna_umd_kernel_queue_destroy(queue->umd);
  queue->umd = NULL;
  amdf_xdna_context_unregister_child(queue->context);
  queue->context = NULL;
  return status;
}

static const amdf_kernel_queue_vtable_t amdf_xdna_kernel_queue_vtable = {
    .query_status = amdf_xdna_kernel_queue_query_status,
    .refresh_status = amdf_xdna_kernel_queue_refresh_status,
    .request_notification = amdf_xdna_kernel_queue_request_notification,
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
  const uint32_t capacity =
      create_info->maximum_pending_submission_count != 0
          ? create_info->maximum_pending_submission_count
          : AMDF_KERNEL_QUEUE_DEFAULT_PENDING_SUBMISSION_COUNT;
  const size_t slot_count = capacity;
  if (slot_count >
      (SIZE_MAX - sizeof(*queue)) / sizeof(queue->native_submissions[0])) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_calloc(
      host_allocator,
      sizeof(*queue) + slot_count * sizeof(queue->native_submissions[0]),
      amdf_alignof(amdf_xdna_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_xdna_context_info_t* context_info =
      amdf_xdna_context_get_info(context);
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
      .device_id = context_info->device_id,
      .reset_epoch = context_info->reset_epoch,
      .queue_family_ordinal = family_info.ordinal,
      .command_type = family_info.command_type,
      .maximum_pending_submission_count = capacity,
      .maximum_command_count = 1,
  };
  amdf_atomic_uint32_initialize(&queue->publishing, 0);
  amdf_atomic_uint32_initialize(&queue->retiring, 0);
  amdf_atomic_uint64_initialize(&queue->submitted, 0);
  amdf_atomic_uint64_initialize(&queue->retired, 0);
  for (uint32_t i = 0; i < capacity; ++i) {
    amdf_atomic_uint64_initialize(&queue->native_submissions[i], 0);
  }
  status = amdf_xdna_context_register_child(context);
  const bool context_registered = amdf_status_is_ok(status);
  if (amdf_status_is_ok(status)) {
    status = amdf_kernel_queue_initialize(
        &queue->base, &amdf_xdna_kernel_queue_vtable, device, &info);
  }
  if (amdf_status_is_ok(status)) {
    queue->context = context;
    status = amdf_xdna_umd_kernel_queue_create(
        amdf_xdna_context_get_umd(context), capacity, &queue->umd);
  }
  if (amdf_status_is_ok(status)) {
    queue->base.info.notification_types =
        amdf_xdna_umd_kernel_queue_query_notification_types(queue->umd);
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
  const amdf_xdna_device_info_t* device_info =
      amdf_xdna_device_get_profile(base_queue->device)->info;
  amdf_memory_t* memory = command->memory;
  if (command->reserved != 0 ||
      memory->scope != amdf_xdna_context_get_memory_scope(queue->context)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (command->access_ordinal >= memory->info.access_count ||
      command->byte_length == 0 ||
      command->byte_length > device_info->instruction.maximum_byte_length ||
      command->byte_length % device_info->instruction.byte_length_granularity !=
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
  if (instruction_address % device_info->instruction.address_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  uint32_t expected = 0;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(&queue->publishing,
                                                   &expected, 1)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const uint64_t last_submitted =
      amdf_atomic_uint64_load_acquire(&queue->submitted);
  uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  if (last_submitted - retired >=
      base_queue->info.maximum_pending_submission_count) {
    status = amdf_xdna_umd_kernel_queue_refresh_progress(queue->umd);
    retired = amdf_xdna_kernel_queue_refresh_retirement(queue);
    if (amdf_status_is_ok(status)) {
      status = amdf_xdna_umd_kernel_queue_query_terminal_status(queue->umd);
    }
    if (amdf_status_is_ok(status) &&
        last_submitted - retired >=
            base_queue->info.maximum_pending_submission_count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
  }
  if (amdf_status_is_ok(status) && last_submitted == UINT64_MAX) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  if (amdf_status_is_ok(status)) {
    const uint32_t slot =
        last_submitted % base_queue->info.maximum_pending_submission_count;
    uint64_t native_submission = 0;
    status = amdf_xdna_umd_kernel_queue_submit(
        queue->umd, slot, instruction_address, (uint32_t)command->byte_length,
        &native_submission);
    if (amdf_status_is_ok(status)) {
      const uint64_t submission = last_submitted + 1;
      amdf_atomic_uint64_store_release(&queue->native_submissions[slot],
                                       native_submission);
      amdf_atomic_uint64_store_release(&queue->submitted, submission);
      *out_submission = submission;
    }
  }
  amdf_atomic_uint32_store_release(&queue->publishing, 0);
  return status;
}
