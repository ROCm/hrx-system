// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/kernel_queue.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/kernel_queue.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/wait.h"

typedef struct amdf_gpu_kernel_queue_t {
  // Generic kernel-queue state shared by every engine implementation.
  amdf_kernel_queue_t base;
  // Exact native kernel-mediated GPU queue lease.
  amdf_gpu_umd_kernel_queue_t* umd;
  // Nonzero only while one host caller publishes a native submission.
  amdf_atomic_uint32_t publishing;
  // Greatest successfully accepted native progress point.
  amdf_atomic_uint64_t submitted;
  // Greatest accepted point with independently confirmed native completion.
  amdf_atomic_uint64_t retired;
} amdf_gpu_kernel_queue_t;

_Static_assert(offsetof(amdf_gpu_kernel_queue_t, base) == 0,
               "GPU kernel queue base must be the first field");

// Refreshes the completed prefix without waiting. Native execution can finish
// before submit publishes its accepted point; cap completion at that snapshot.
static uint64_t amdf_gpu_kernel_queue_refresh_retirement(
    amdf_gpu_kernel_queue_t* queue) {
  const uint64_t submitted = amdf_atomic_uint64_load_acquire(&queue->submitted);
  uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  if (retired >= submitted) {
    return retired;
  }
  const uint64_t progress =
      amdf_gpu_umd_kernel_queue_query_progress(queue->umd);
  const uint64_t completed = progress < submitted ? progress : submitted;
  while (retired < completed && !amdf_atomic_uint64_compare_exchange_acq_rel(
                                    &queue->retired, &retired, completed)) {
  }
  return retired < completed ? completed : retired;
}

static amdf_status_t amdf_gpu_kernel_queue_query_status(
    amdf_kernel_queue_t* base_queue, amdf_kernel_queue_status_t* out_status) {
  const amdf_gpu_kernel_queue_t* queue =
      (const amdf_gpu_kernel_queue_t*)base_queue;
  out_status->retired_submission =
      amdf_atomic_uint64_load_acquire(&queue->retired);
  out_status->terminal_status =
      amdf_gpu_umd_kernel_queue_query_terminal_status(queue->umd);
  out_status->state = amdf_status_is_ok(out_status->terminal_status)
                          ? AMDF_QUEUE_STATE_ACTIVE
                          : AMDF_QUEUE_STATE_DEVICE_LOST;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kernel_queue_refresh_status(
    amdf_kernel_queue_t* base_queue, amdf_kernel_queue_status_t* out_status) {
  amdf_gpu_kernel_queue_refresh_retirement(
      (amdf_gpu_kernel_queue_t*)base_queue);
  return amdf_gpu_kernel_queue_query_status(base_queue, out_status);
}

static amdf_status_t amdf_gpu_kernel_queue_request_notification(
    amdf_kernel_queue_t* base_queue, uint64_t submission,
    const amdf_native_event_t* event) {
  (void)base_queue;
  (void)submission;
  (void)event;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_gpu_kernel_queue_wait(
    amdf_kernel_queue_t* base_queue, uint64_t submission,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds) {
  amdf_gpu_kernel_queue_t* queue = (amdf_gpu_kernel_queue_t*)base_queue;
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_wait_deadline_t deadline;
  amdf_status_t status = amdf_wait_deadline_initialize(
      timeout_nanoseconds, poll_duration_nanoseconds, &deadline);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const uint64_t retired = amdf_gpu_kernel_queue_refresh_retirement(queue);
  status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue->umd);
  if (!amdf_status_is_ok(status) || retired >= submission) {
    return status;
  }
  status = amdf_gpu_umd_kernel_queue_wait(queue->umd, submission, &deadline);
  // A native error is observable even when independent progress proves that
  // accepted commands have completed. Neither fact substitutes for the other.
  amdf_gpu_kernel_queue_refresh_retirement(queue);
  return amdf_status_is_ok(status)
             ? amdf_gpu_umd_kernel_queue_query_terminal_status(queue->umd)
             : status;
}

static amdf_status_t amdf_gpu_kernel_queue_destroy_native(
    amdf_kernel_queue_t* base_queue) {
  amdf_gpu_kernel_queue_t* queue = (amdf_gpu_kernel_queue_t*)base_queue;
  if (amdf_gpu_kernel_queue_refresh_retirement(queue) <
      amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_gpu_umd_kernel_queue_destroy(queue->umd);
  queue->umd = NULL;
  return status;
}

static const amdf_kernel_queue_vtable_t amdf_gpu_kernel_queue_vtable = {
    .query_status = amdf_gpu_kernel_queue_query_status,
    .refresh_status = amdf_gpu_kernel_queue_refresh_status,
    .request_notification = amdf_gpu_kernel_queue_request_notification,
    .wait = amdf_gpu_kernel_queue_wait,
    .destroy_native = amdf_gpu_kernel_queue_destroy_native,
};

amdf_status_t AMDF_CALL amdf_gpu_kernel_queue_create(
    amdf_device_t* device,
    const amdf_gpu_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue) {
  if (out_queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!amdf_device_is_engine(device, AMDF_ENGINE_KIND_GPU)) {
    return device == NULL
               ? amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT)
               : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      (uint32_t)sizeof(amdf_gpu_kernel_queue_create_info_t));
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
  if ((family_info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
       family_info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) ||
      (family_info.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) ==
          0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const amdf_allocator_t host_allocator = amdf_device_host_allocator(device);
  amdf_gpu_kernel_queue_t* queue = NULL;
  status = amdf_calloc(host_allocator, sizeof(*queue),
                       amdf_alignof(amdf_gpu_kernel_queue_t), (void**)&queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_gpu_device_info_t* device_info = amdf_gpu_device_get_info(device);
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
      .device_id = device_info->id,
      .reset_epoch = device_info->reset_epoch,
      .queue_family_ordinal = family_info.ordinal,
      .command_type = family_info.command_type,
      .maximum_pending_submission_count =
          create_info->maximum_pending_submission_count != 0
              ? create_info->maximum_pending_submission_count
              : AMDF_KERNEL_QUEUE_DEFAULT_PENDING_SUBMISSION_COUNT,
      .maximum_command_count = 1,
  };
  amdf_atomic_uint32_initialize(&queue->publishing, 0);
  amdf_atomic_uint64_initialize(&queue->submitted, 0);
  amdf_atomic_uint64_initialize(&queue->retired, 0);
  status = amdf_kernel_queue_initialize(
      &queue->base, &amdf_gpu_kernel_queue_vtable, device, &info);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_umd_kernel_queue_create(
        amdf_gpu_device_get_umd(device), family_info.command_type, &queue->umd);
  }
  if (amdf_status_is_ok(status)) {
    *out_queue = &queue->base;
  } else {
    if (queue->base.device != NULL) {
      amdf_kernel_queue_deinitialize(&queue->base);
    }
    amdf_free(host_allocator, queue);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_gpu_kernel_queue_submit(
    amdf_kernel_queue_t* base_queue,
    const amdf_gpu_kernel_queue_submission_info_t* submission_info,
    uint64_t* out_submission) {
  if (base_queue == NULL || out_submission == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (base_queue->info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
      base_queue->info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_structure_validate_input(
      submission_info, AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      (uint32_t)sizeof(amdf_gpu_kernel_queue_submission_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (submission_info->reserved != 0 || submission_info->command_count != 1 ||
      submission_info->commands == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  const amdf_gpu_kernel_command_t* command = &submission_info->commands[0];
  amdf_memory_t* memory = command->memory;
  if (memory == NULL || command->reserved != 0 || command->byte_length == 0 ||
      (command->byte_offset & 3) != 0 || (command->byte_length & 3) != 0 ||
      command->byte_offset > memory->info.byte_length ||
      command->byte_length > memory->info.byte_length - command->byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (command->access_ordinal >= memory->info.access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_access_state_t* access =
      &memory->accesses[command->access_ordinal];
  if (access->device != base_queue->device) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((access->info.address_kinds & (UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU)) ==
          0 ||
      (access->info.access & AMDF_MEMORY_ACCESS_EXECUTE) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (access->info.reset_epoch != base_queue->info.reset_epoch ||
      base_queue->info.reset_epoch !=
          amdf_gpu_device_query_reset_epoch(base_queue->device)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (access->addresses[AMDF_MEMORY_ADDRESS_GPU] >
      UINT64_MAX - command->byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  amdf_gpu_kernel_queue_t* queue = (amdf_gpu_kernel_queue_t*)base_queue;
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
    retired = amdf_gpu_kernel_queue_refresh_retirement(queue);
    status = amdf_gpu_umd_kernel_queue_query_terminal_status(queue->umd);
    if (amdf_status_is_ok(status) &&
        last_submitted - retired >=
            base_queue->info.maximum_pending_submission_count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
  }
  // UINT64_MAX is reserved for native monitored-fence reset notification.
  if (amdf_status_is_ok(status) && last_submitted == UINT64_MAX - 1) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t submission = last_submitted + 1;
    status = amdf_gpu_umd_kernel_queue_submit(
        queue->umd,
        access->addresses[AMDF_MEMORY_ADDRESS_GPU] + command->byte_offset,
        command->byte_length, submission);
    if (amdf_status_is_ok(status)) {
      amdf_atomic_uint64_store_release(&queue->submitted, submission);
      *out_submission = submission;
    }
  }
  amdf_atomic_uint32_store_release(&queue->publishing, 0);
  return status;
}
