// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/wait.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"

enum {
  AMDF_WINDOWS_WAIT_SIGNALED = 0,
  AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_ALLOCATION_SIZE = 8192,
};

typedef enum amdf_windows_xdna_kernel_execution_preparation_phase_e {
  AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_INERT = 0,
  AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_COMMAND_STORAGE_READY,
  AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_PACKET_STORAGE_READY,
  AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_READY,
} amdf_windows_xdna_kernel_execution_preparation_phase_t;

struct amdf_windows_xdna_kernel_execution_t {
  // Platform device borrowed until destruction succeeds.
  amdf_xdna_umd_device_t* device;
  // KMT context borrowed from the owning public context.
  D3DKMT_HANDLE context;
  // Native prefix preceding the copied command, resolved during admission.
  uint32_t submission_header_byte_length;
  // First observed command or bootstrap failure, scoped to this context.
  amdf_atomic_uint64_t terminal_status;
  // Serializes native preparation, private binding, and queue leasing.
  SRWLOCK state_lock;
  // Serializes use of the one reusable asynchronous wait event.
  SRWLOCK wait_lock;
  // Native hardware queue owned by this execution path.
  D3DKMT_HANDLE hardware_queue;
  // Monitored fence signaled as hardware-queue commands retire.
  D3DKMT_HANDLE progress_fence;
  // CPU-readable monotonic hardware-queue progress value.
  const volatile uint64_t* progress_fence_pointer;
  // GPU-visible progress address retained for later queue interoperability.
  uint64_t progress_fence_device_address;
  // Manual-reset event reused by externally serialized native waits.
  HANDLE wait_event;
  // Native submission registered to signal `wait_event`, or zero when idle.
  uint64_t wait_event_submission;
  // Shared completion and context-initialization allocation.
  amdf_windows_xdna_private_allocation_t command_allocation;
  // Native transport packet reused under the context's exclusive queue lease.
  amdf_windows_xdna_private_allocation_t packet_allocation;
  // One after the context's singular native instruction binding is claimed.
  uint32_t instruction_binding_claimed;
  // One while a public queue exclusively leases the hardware queue.
  uint32_t queue_lease_count;
  // Completed storage or bootstrap preparation phase.
  uint32_t preparation_phase;
  // Native setup submission accepted but not yet observed retired.
  uint64_t preparation_submission;
  // Greatest progress value assigned to a native submission.
  uint64_t last_native_submission;
};

uint64_t amdf_windows_xdna_kernel_execution_query_progress(
    const amdf_windows_xdna_kernel_execution_t* execution) {
  return execution->progress_fence_pointer == NULL
             ? 0
             : *execution->progress_fence_pointer;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_submit_native(
    amdf_windows_xdna_kernel_execution_t* execution, uint64_t command_address,
    uint32_t command_byte_length,
    const amdf_windows_xdna_legacy_submission_t* submission,
    uint64_t* out_native_submission) {
  const amdf_status_t terminal_status =
      amdf_windows_xdna_kernel_execution_query_terminal_status(execution);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  if (execution->last_native_submission == UINT64_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  const uint64_t native_submission = execution->last_native_submission + 1;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {0};
  submit.hHwQueue = execution->hardware_queue;
  submit.HwQueueProgressFenceId = native_submission;
  submit.CommandBuffer = command_address;
  submit.CommandLength = command_byte_length;
  submit.PrivateDriverDataSize = submission->byte_length;
  submit.pPrivateDriverData = (void*)submission->bytes;
  MemoryBarrier();
  const amdf_status_t status = amdf_kmt_make_status(
      execution->device->kmt->submit_command_to_hardware_queue(&submit));
  if (amdf_status_is_ok(status)) {
    execution->last_native_submission = native_submission;
    *out_native_submission = native_submission;
  } else {
    return amdf_kmt_device_status_observe_error(
        &execution->device->status, execution->device->kmt,
        execution->device->device, status);
  }
  return status;
}

amdf_status_t amdf_windows_xdna_kernel_execution_query_terminal_status(
    const amdf_windows_xdna_kernel_execution_t* execution) {
  const amdf_status_t status =
      amdf_atomic_uint64_load_acquire(&execution->terminal_status);
  return amdf_status_is_ok(status)
             ? amdf_kmt_device_status_query(&execution->device->status)
             : status;
}

static void amdf_windows_xdna_kernel_execution_record_failure(
    amdf_windows_xdna_kernel_execution_t* execution, amdf_status_t failure) {
  uint64_t expected = AMDF_STATUS_OK;
  amdf_atomic_uint64_compare_exchange_acq_rel(&execution->terminal_status,
                                              &expected, failure);
}

static amdf_status_t amdf_windows_xdna_kernel_execution_wait_synchronous(
    amdf_windows_xdna_kernel_execution_t* execution, uint64_t submission) {
  if (amdf_windows_xdna_kernel_execution_query_progress(execution) >=
      submission) {
    MemoryBarrier();
    return AMDF_STATUS_OK;
  }
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {0};
  wait.hDevice = execution->device->device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = &execution->progress_fence;
  wait.FenceValueArray = &submission;
  const amdf_status_t status =
      amdf_kmt_make_status(execution->device->kmt->wait_from_cpu(&wait));
  if (!amdf_status_is_ok(status)) {
    return amdf_kmt_device_status_observe_error(
        &execution->device->status, execution->device->kmt,
        execution->device->device, status);
  }
  MemoryBarrier();
  return amdf_windows_xdna_kernel_execution_query_progress(execution) >=
                 submission
             ? AMDF_STATUS_OK
             : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
}

static amdf_status_t amdf_windows_xdna_kernel_execution_submit_preparation(
    amdf_windows_xdna_kernel_execution_t* execution, uint64_t command_address,
    uint32_t command_byte_length,
    const amdf_windows_xdna_legacy_submission_t* submission) {
  amdf_status_t status = AMDF_STATUS_OK;
  if (execution->preparation_submission == 0) {
    status = amdf_windows_xdna_kernel_execution_submit_native(
        execution, command_address, command_byte_length, submission,
        &execution->preparation_submission);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_kernel_execution_wait_synchronous(
        execution, execution->preparation_submission);
  }
  if (amdf_status_is_ok(status)) {
    execution->preparation_submission = 0;
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_prepare_wait_state(
    amdf_windows_xdna_kernel_execution_t* execution) {
  if (execution->wait_event == NULL) {
    execution->wait_event = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (execution->wait_event == NULL) {
      return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_realize_queue(
    amdf_windows_xdna_kernel_execution_t* execution) {
  amdf_status_t status =
      amdf_windows_xdna_kernel_execution_prepare_wait_state(execution);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (execution->hardware_queue != 0) {
    return execution->progress_fence != 0 &&
                   execution->progress_fence_pointer != NULL &&
                   execution->progress_fence_device_address != 0
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  D3DKMT_CREATEHWQUEUE create = {0};
  create.hHwContext = execution->context;
  status = amdf_kmt_make_status(
      execution->device->kmt->create_hardware_queue(&create));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  execution->hardware_queue = create.hHwQueue;
  execution->progress_fence = create.hHwQueueProgressFence;
  execution->progress_fence_pointer =
      (const volatile uint64_t*)create.HwQueueProgressFenceCPUVirtualAddress;
  execution->progress_fence_device_address =
      create.HwQueueProgressFenceGPUVirtualAddress;
  if (execution->hardware_queue == 0 || execution->progress_fence == 0 ||
      execution->progress_fence_pointer == NULL ||
      execution->progress_fence_device_address == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  execution->last_native_submission =
      amdf_windows_xdna_kernel_execution_query_progress(execution);
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_realize_command_storage(
    amdf_windows_xdna_kernel_execution_t* execution) {
  amdf_status_t status = amdf_windows_xdna_private_allocation_realize(
      &execution->command_allocation);
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_private_allocation_lock(
        &execution->command_allocation);
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_publish_aperture(
    amdf_windows_xdna_kernel_execution_t* execution,
    const amdf_windows_xdna_private_allocation_t* allocation) {
  amdf_windows_xdna_legacy_submission_t submission;
  amdf_windows_xdna_legacy_submission_build_aperture(
      execution->submission_header_byte_length, allocation, &submission);
  return amdf_windows_xdna_kernel_execution_submit_preparation(
      execution, allocation->device_address,
      (uint32_t)allocation->descriptor.allocation_byte_length, &submission);
}

static amdf_status_t amdf_windows_xdna_kernel_execution_publish_pdi(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_windows_xdna_private_allocation_t* allocation) {
  const amdf_xdna_bootstrap_t* bootstrap =
      execution->device->profile->bootstrap;
  memset(allocation->host_pointer, 0,
         (size_t)AMDF_WINDOWS_XDNA_PRIVATE_BOOTSTRAP_SIZE);
  memcpy(allocation->host_pointer, bootstrap->pdi_bytes,
         bootstrap->pdi_byte_length);
  return amdf_windows_xdna_private_allocation_publish(
      allocation, 0, AMDF_WINDOWS_XDNA_PRIVATE_BOOTSTRAP_SIZE);
}

static amdf_status_t amdf_windows_xdna_kernel_execution_activate_context(
    amdf_windows_xdna_kernel_execution_t* execution,
    uint64_t firmware_address) {
  if (execution->preparation_submission == 0) {
    memset(execution->command_allocation.host_pointer, 0,
           (size_t)
               execution->command_allocation.descriptor.allocation_byte_length);
    uint64_t* command_words =
        (uint64_t*)execution->command_allocation.host_pointer;
    command_words[0] = 1;
    command_words[1] = firmware_address;
  }
  amdf_windows_xdna_legacy_submission_t submission;
  amdf_windows_xdna_legacy_submission_build_context_initialize(
      execution->submission_header_byte_length, &execution->command_allocation,
      &submission);
  amdf_status_t status = amdf_windows_xdna_kernel_execution_submit_preparation(
      execution, 0, 0, &submission);
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_legacy_submission_query_initialize_result(
        &execution->command_allocation);
    if (!amdf_status_is_ok(status)) {
      amdf_windows_xdna_kernel_execution_record_failure(execution, status);
    }
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_realize_packet_storage(
    amdf_windows_xdna_kernel_execution_t* execution) {
  amdf_status_t status = amdf_windows_xdna_private_allocation_realize(
      &execution->packet_allocation);
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_private_allocation_lock(
        &execution->packet_allocation);
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_kernel_execution_prepare_through(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_windows_xdna_kernel_execution_preparation_phase_t target_phase) {
  if (!amdf_kmt_api_supports_xdna_kernel_execution(execution->device->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  AcquireSRWLockExclusive(&execution->state_lock);
  amdf_status_t status =
      amdf_windows_xdna_kernel_execution_query_terminal_status(execution);
  while (amdf_status_is_ok(status) &&
         execution->preparation_phase < target_phase) {
    switch (execution->preparation_phase) {
      case AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_INERT:
        status = amdf_windows_xdna_kernel_execution_realize_command_storage(
            execution);
        break;
      case AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_COMMAND_STORAGE_READY:
        status = amdf_windows_xdna_kernel_execution_realize_packet_storage(
            execution);
        break;
      case AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_PACKET_STORAGE_READY:
        status = amdf_windows_xdna_kernel_execution_realize_queue(execution);
        break;
      case AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_READY:
      default:
        status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
        break;
    }
    if (amdf_status_is_ok(status)) {
      ++execution->preparation_phase;
    }
  }
  ReleaseSRWLockExclusive(&execution->state_lock);
  return status;
}

amdf_status_t amdf_windows_xdna_kernel_execution_prepare_memory(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_windows_xdna_private_allocation_t* allocation) {
  amdf_status_t status = amdf_windows_xdna_kernel_execution_prepare_through(
      execution, AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_READY);
  if (!amdf_status_is_ok(status)) return status;
  AcquireSRWLockExclusive(&execution->state_lock);
  if (execution->instruction_binding_claimed != 0) {
    ReleaseSRWLockExclusive(&execution->state_lock);
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  execution->instruction_binding_claimed = 1;
  status = amdf_windows_xdna_private_allocation_realize(allocation);
  const uint64_t maximum_address =
      (UINT64_C(1) << execution->device->profile->dma.address_bit_count) - 1;
  if (amdf_status_is_ok(status) &&
      (allocation->firmware_address == 0 ||
       allocation->firmware_address > maximum_address ||
       allocation->descriptor.allocation_byte_length - 1 >
           maximum_address - allocation->firmware_address ||
       allocation->firmware_address % execution->device->profile->info
                                          ->instruction.address_alignment !=
           0)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_private_allocation_lock(allocation);
  }
  if (amdf_status_is_ok(status) &&
      (uintptr_t)allocation->host_pointer % 4096 != 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_kernel_execution_publish_aperture(execution,
                                                                 allocation);
  }
  if (amdf_status_is_ok(status)) {
    status =
        amdf_windows_xdna_kernel_execution_publish_pdi(execution, allocation);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_kernel_execution_activate_context(
        execution, allocation->firmware_address);
  }
  if (amdf_status_is_ok(status)) {
    amdf_windows_xdna_legacy_submission_t submission;
    amdf_windows_xdna_legacy_submission_build_accounting(
        execution->submission_header_byte_length, allocation,
        allocation->descriptor.allocation_byte_length, &submission);
    status = amdf_windows_xdna_kernel_execution_submit_preparation(
        execution, 0, 0, &submission);
  }
  if (!amdf_status_is_ok(status)) {
    amdf_windows_xdna_kernel_execution_record_failure(execution, status);
  }
  ReleaseSRWLockExclusive(&execution->state_lock);
  return status;
}

amdf_status_t amdf_windows_xdna_kernel_execution_release_memory(
    amdf_windows_xdna_kernel_execution_t* execution) {
  AcquireSRWLockExclusive(&execution->state_lock);
  amdf_status_t status = AMDF_STATUS_OK;
  if (execution->preparation_submission != 0) {
    status = amdf_windows_xdna_kernel_execution_wait_synchronous(
        execution, execution->preparation_submission);
    if (amdf_status_is_ok(status)) execution->preparation_submission = 0;
  }
  ReleaseSRWLockExclusive(&execution->state_lock);
  return status;
}

static void amdf_windows_xdna_kernel_execution_initialize_storage(
    amdf_windows_xdna_kernel_execution_t* execution) {
  const amdf_windows_xdna_private_allocation_descriptor_t command_descriptor = {
      .requested_byte_length = 4096,
      .allocation_byte_length = 4096,
      .type = 0x332B,
      .policy = 0,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_SHARED_RESOURCE,
  };
  amdf_windows_xdna_private_allocation_initialize(
      execution->device, &command_descriptor, &execution->command_allocation);

  const amdf_windows_xdna_private_allocation_descriptor_t packet_descriptor = {
      .requested_byte_length = 4096 + execution->submission_header_byte_length,
      .allocation_byte_length =
          AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_ALLOCATION_SIZE,
      .type = 0x3328,
      .policy = 2,
      .xcl_flags = 0x80000000u,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS,
  };
  amdf_windows_xdna_private_allocation_initialize(
      execution->device, &packet_descriptor, &execution->packet_allocation);
}

amdf_status_t amdf_windows_xdna_kernel_execution_create(
    amdf_xdna_umd_context_t* context,
    amdf_windows_xdna_kernel_execution_t** out_execution) {
  if (context == NULL || out_execution == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_xdna_umd_device_t* device = context->device;
  amdf_windows_xdna_kernel_execution_t* execution = NULL;
  amdf_status_t status = amdf_calloc(
      device->host_allocator, sizeof(*execution),
      amdf_alignof(amdf_windows_xdna_kernel_execution_t), (void**)&execution);
  if (!amdf_status_is_ok(status)) return status;
  execution->device = device;
  execution->context = context->handle;
  execution->submission_header_byte_length =
      context->native_abi->submission_header_byte_length;
  amdf_atomic_uint64_initialize(&execution->terminal_status, AMDF_STATUS_OK);
  InitializeSRWLock(&execution->state_lock);
  InitializeSRWLock(&execution->wait_lock);
  amdf_windows_xdna_kernel_execution_initialize_storage(execution);
  *out_execution = execution;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_windows_xdna_kernel_execution_prepare_context_destroy(
    amdf_windows_xdna_kernel_execution_t* execution) {
  if (execution == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  AcquireSRWLockExclusive(&execution->state_lock);
  if (execution->queue_lease_count != 0) {
    ReleaseSRWLockExclusive(&execution->state_lock);
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  if (execution->preparation_submission != 0 &&
      amdf_windows_xdna_kernel_execution_query_progress(execution) <
          execution->preparation_submission) {
    ReleaseSRWLockExclusive(&execution->state_lock);
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  execution->preparation_submission = 0;
  if (execution->wait_event_submission != 0) {
    const DWORD wait_result = WaitForSingleObject(execution->wait_event, 0);
    if (wait_result == WAIT_TIMEOUT) {
      ReleaseSRWLockExclusive(&execution->state_lock);
      return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
    if (wait_result != AMDF_WINDOWS_WAIT_SIGNALED) {
      const amdf_status_t status =
          wait_result == WAIT_FAILED
              ? amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError())
              : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      ReleaseSRWLockExclusive(&execution->state_lock);
      return status;
    }
    execution->wait_event_submission = 0;
  }

  amdf_status_t status = AMDF_STATUS_OK;
  if (execution->hardware_queue != 0) {
    D3DKMT_DESTROYHWQUEUE destroy = {0};
    destroy.hHwQueue = execution->hardware_queue;
    status = amdf_kmt_make_status(
        execution->device->kmt->destroy_hardware_queue(&destroy));
    if (amdf_status_is_ok(status)) {
      execution->hardware_queue = 0;
      execution->progress_fence = 0;
      execution->progress_fence_pointer = NULL;
      execution->progress_fence_device_address = 0;
    }
  }
  ReleaseSRWLockExclusive(&execution->state_lock);
  return status;
}

amdf_status_t amdf_windows_xdna_kernel_execution_destroy(
    amdf_windows_xdna_kernel_execution_t* execution) {
  if (execution == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  AcquireSRWLockExclusive(&execution->state_lock);
  amdf_assert(execution->queue_lease_count == 0);
  amdf_assert(execution->preparation_submission == 0);
  amdf_assert(execution->wait_event_submission == 0);
  amdf_assert(execution->hardware_queue == 0);

  amdf_status_t status = AMDF_STATUS_OK;
  if (execution->packet_allocation.device != NULL) {
    status = amdf_windows_xdna_private_allocation_destroy(
        &execution->packet_allocation);
  }
  if (amdf_status_is_ok(status) &&
      execution->command_allocation.device != NULL) {
    status = amdf_windows_xdna_private_allocation_destroy(
        &execution->command_allocation);
  }
  if (amdf_status_is_ok(status) && execution->wait_event != NULL) {
    if (!CloseHandle(execution->wait_event)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    } else {
      execution->wait_event = NULL;
    }
  }
  ReleaseSRWLockExclusive(&execution->state_lock);
  if (amdf_status_is_ok(status)) {
    amdf_free(execution->device->host_allocator, execution);
  }
  return status;
}

amdf_status_t amdf_windows_xdna_kernel_execution_acquire_queue(
    amdf_windows_xdna_kernel_execution_t* execution) {
  if (execution == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_windows_xdna_kernel_execution_prepare_through(
      execution, AMDF_WINDOWS_XDNA_KERNEL_EXECUTION_PHASE_READY);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  AcquireSRWLockExclusive(&execution->state_lock);
  if (execution->queue_lease_count != 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  } else {
    execution->queue_lease_count = 1;
  }
  ReleaseSRWLockExclusive(&execution->state_lock);
  return status;
}

void amdf_windows_xdna_kernel_execution_release_queue(
    amdf_windows_xdna_kernel_execution_t* execution) {
  AcquireSRWLockExclusive(&execution->state_lock);
  amdf_assert(execution->queue_lease_count == 1 &&
              "releasing an unowned XDNA hardware queue");
  execution->queue_lease_count = 0;
  ReleaseSRWLockExclusive(&execution->state_lock);
}

amdf_status_t amdf_windows_xdna_kernel_execution_submit(
    amdf_windows_xdna_kernel_execution_t* execution,
    uint64_t instruction_address, uint32_t instruction_byte_length,
    uint64_t* out_native_submission) {
  amdf_xdna_transaction_interpreter_packet_t* packet =
      execution->packet_allocation.host_pointer;
  amdf_xdna_transaction_interpreter_packet_build(
      instruction_address, instruction_byte_length, packet);
  amdf_windows_xdna_legacy_submission_t submission;
  amdf_windows_xdna_legacy_submission_build_execute(
      execution->submission_header_byte_length, &execution->packet_allocation,
      &execution->command_allocation, packet, &submission);
  const amdf_status_t status = amdf_windows_xdna_private_allocation_publish(
      &execution->packet_allocation, 0, sizeof(*packet));
  if (!amdf_status_is_ok(status)) return status;
  uint64_t* command_words = execution->command_allocation.host_pointer;
  command_words[0] = 1;
  command_words[1] = 0;
  return amdf_windows_xdna_kernel_execution_submit_native(
      execution, execution->packet_allocation.device_address,
      (uint32_t)execution->packet_allocation.descriptor.requested_byte_length,
      &submission, out_native_submission);
}

void amdf_windows_xdna_kernel_execution_retire_command(
    amdf_windows_xdna_kernel_execution_t* execution) {
  const amdf_status_t status =
      amdf_windows_xdna_legacy_submission_query_execute_result(
          &execution->command_allocation);
  if (!amdf_status_is_ok(status)) {
    amdf_windows_xdna_kernel_execution_record_failure(execution, status);
  }
}

amdf_status_t amdf_windows_xdna_kernel_execution_wait(
    amdf_windows_xdna_kernel_execution_t* execution, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline) {
  const amdf_status_t terminal_status =
      amdf_windows_xdna_kernel_execution_query_terminal_status(execution);
  if (!amdf_status_is_ok(terminal_status)) return terminal_status;
  amdf_wait_budget_t remaining;
  while (amdf_windows_xdna_kernel_execution_query_progress(execution) <
         native_submission) {
    const amdf_status_t status =
        amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) return status;
    if (remaining.poll == 0) break;
    YieldProcessor();
  }
  // A finite waiter never blocks acquiring the reusable event behind an
  // infinite native wait. Contention consumes the same original deadline.
  for (;;) {
    if (amdf_windows_xdna_kernel_execution_query_progress(execution) >=
        native_submission) {
      MemoryBarrier();
      return AMDF_STATUS_OK;
    }
    const amdf_status_t status =
        amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) return status;
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (TryAcquireSRWLockExclusive(&execution->wait_lock)) break;
    amdf_platform_wait_yield();
  }
  amdf_status_t status = AMDF_STATUS_OK;
  while (amdf_status_is_ok(status) &&
         amdf_windows_xdna_kernel_execution_query_progress(execution) <
             native_submission) {
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) break;
    if (remaining.timeout == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
      break;
    }
    if (execution->wait_event_submission == 0) {
      if (!ResetEvent(execution->wait_event)) {
        status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
        break;
      }
      D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {0};
      wait.hDevice = execution->device->device;
      wait.ObjectCount = 1;
      wait.ObjectHandleArray = &execution->progress_fence;
      wait.FenceValueArray = &native_submission;
      wait.hAsyncEvent = execution->wait_event;
      status =
          amdf_kmt_make_status(execution->device->kmt->wait_from_cpu(&wait));
      if (!amdf_status_is_ok(status)) {
        status = amdf_kmt_device_status_observe_error(
            &execution->device->status, execution->device->kmt,
            execution->device->device, status);
        break;
      }
      execution->wait_event_submission = native_submission;
    }
    // Native event registration may itself consume time, so sample again.
    status = amdf_wait_deadline_query_remaining(deadline, &remaining);
    if (!amdf_status_is_ok(status)) break;
    DWORD wait_milliseconds = INFINITE;
    if (remaining.timeout != AMDF_TIMEOUT_INFINITE) {
      const uint64_t milliseconds =
          remaining.timeout / UINT64_C(1000000) +
          (remaining.timeout % UINT64_C(1000000) != 0);
      wait_milliseconds =
          milliseconds >= MAXDWORD ? MAXDWORD - 1 : (DWORD)milliseconds;
    }
    const DWORD wait_result =
        WaitForSingleObject(execution->wait_event, wait_milliseconds);
    if (wait_result == WAIT_TIMEOUT) continue;
    if (wait_result != AMDF_WINDOWS_WAIT_SIGNALED) {
      status = wait_result == WAIT_FAILED
                   ? amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError())
                   : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      break;
    }
    execution->wait_event_submission = 0;
  }
  ReleaseSRWLockExclusive(&execution->wait_lock);
  if (amdf_status_is_ok(status)) MemoryBarrier();
  return status;
}
