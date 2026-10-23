// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/queue.h"

#include <ntstatus.h>

#include "libamdf/src/media/d3d12/command.h"
#include "libamdf/src/media/d3d12/object.h"
#include "libamdf/src/platform/native_event.h"
#include "libamdf/src/platform/wait.h"
#include "libamdf/src/wait.h"

namespace amdf::media::d3d12 {
namespace {

struct Dependency {
  // Consumer-native binding retained by the core dependency owner.
  amdf_kernel_queue_native_dependency_t base;
  // Allocator copied for terminal release.
  amdf_allocator_t host_allocator;
  // Cold import of the producer's KMT-origin progress fence.
  ComPtr<ID3D12Fence> fence;
};

void Latch(amdf_atomic_uint64_t* destination, amdf_status_t status) {
  uint64_t expected = AMDF_STATUS_OK;
  amdf_atomic_uint64_compare_exchange_acq_rel(destination, &expected, status);
}

void QueryStatus(amdf_kernel_queue_t* base,
                 amdf_kernel_queue_status_t* out_status) {
  auto* queue = reinterpret_cast<Queue*>(base);
  out_status->retired_submission =
      amdf_atomic_uint64_load_acquire(&queue->retired);
  const auto* device = reinterpret_cast<const Device*>(base->device);
  const amdf_status_t removed =
      amdf_atomic_uint64_load_acquire(&device->removed_status);
  out_status->terminal_status =
      amdf_status_is_ok(removed)
          ? amdf_atomic_uint64_load_acquire(&queue->terminal_status)
          : removed;
  out_status->state = !amdf_status_is_ok(removed) ? AMDF_QUEUE_STATE_DEVICE_LOST
                      : amdf_status_is_ok(out_status->terminal_status)
                          ? AMDF_QUEUE_STATE_ACTIVE
                          : AMDF_QUEUE_STATE_FAILED;
}

amdf_status_t RefreshStatus(amdf_kernel_queue_t* base,
                            amdf_kernel_queue_status_t* out_status) {
  RefreshRetirement(reinterpret_cast<Queue*>(base));
  QueryStatus(base, out_status);
  return AMDF_STATUS_OK;
}

amdf_status_t RequestNotification(amdf_kernel_queue_t* base,
                                  uint64_t submission,
                                  const amdf_native_event_t* event) {
  auto* queue = reinterpret_cast<Queue*>(base);
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  RefreshRetirement(queue);
  if (submission <= amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_platform_native_event_signal(event);
  }
  const amdf_status_t status = QueueTerminalStatus(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return Hresult(queue->fence->SetEventOnCompletion(
      submission, event->payload.native_handle));
}

amdf_status_t Wait(amdf_kernel_queue_t* base, uint64_t submission,
                   uint64_t timeout_nanoseconds,
                   uint64_t poll_duration_nanoseconds) {
  auto* queue = reinterpret_cast<Queue*>(base);
  if (submission > amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_wait_deadline_t deadline;
  amdf_status_t status = amdf_wait_deadline_initialize(
      timeout_nanoseconds, poll_duration_nanoseconds, &deadline);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_wait_budget_t remaining;
  for (;;) {
    RefreshRetirement(queue);
    if (submission <= amdf_atomic_uint64_load_acquire(&queue->retired)) {
      return AMDF_STATUS_OK;
    }
    status = QueueTerminalStatus(queue);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    status = amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (remaining.timeout == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
    }
    if (remaining.poll != 0) {
      YieldProcessor();
    } else if (TryAcquireSRWLockExclusive(&queue->wait_lock)) {
      break;
    } else {
      // Contention consumes the same budget as native waiting; a finite waiter
      // cannot block behind an infinite waiter while acquiring this event.
      amdf_platform_wait_yield();
    }
  }
  bool complete = false;
  while (amdf_status_is_ok(status) && !complete) {
    RefreshRetirement(queue);
    complete = submission <= amdf_atomic_uint64_load_acquire(&queue->retired);
    if (complete) {
      break;
    }
    status = QueueTerminalStatus(queue);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    status = amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    if (remaining.timeout == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
      break;
    }
    // An earlier timed-out registration remains valid. An earlier requested
    // point gets its own registration on the same event; every wake rechecks
    // progress, so stale notifications cannot establish retirement.
    if (queue->wait_event_submission == 0 ||
        submission < queue->wait_event_submission) {
      if (!ResetEvent(queue->wait_event)) {
        status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
        break;
      }
      status = Hresult(
          queue->fence->SetEventOnCompletion(submission, queue->wait_event));
      if (!amdf_status_is_ok(status)) {
        break;
      }
      queue->wait_event_submission = submission;
    }
    status = amdf_wait_deadline_query_remaining(&deadline, &remaining);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    DWORD milliseconds = INFINITE;
    if (remaining.timeout != AMDF_TIMEOUT_INFINITE) {
      const uint64_t rounded = remaining.timeout / UINT64_C(1000000) +
                               (remaining.timeout % UINT64_C(1000000) != 0);
      milliseconds =
          rounded >= MAXDWORD ? MAXDWORD - 1 : static_cast<DWORD>(rounded);
    }
    const DWORD result = WaitForSingleObject(queue->wait_event, milliseconds);
    if (result == WAIT_OBJECT_0) {
      queue->wait_event_submission = 0;
    } else if (result != WAIT_TIMEOUT) {
      status = result == WAIT_FAILED
                   ? amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError())
                   : amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  ReleaseSRWLockExclusive(&queue->wait_lock);
  return status;
}

amdf_status_t ReleaseNative(Queue* queue, Device* device) {
  queue->native.Reset();
  queue->fence.Reset();
  amdf_status_t status = AMDF_STATUS_OK;
  if (queue->shared_progress.object != 0) {
    status = amdf_windows_monitored_fence_deinitialize(device->kmt,
                                                       &queue->shared_progress);
    queue->shared_progress = {};
  }
  if (queue->wait_event != nullptr) {
    if (!CloseHandle(queue->wait_event) && amdf_status_is_ok(status)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    queue->wait_event = nullptr;
  }
  return status;
}

amdf_status_t DestroyNative(amdf_kernel_queue_t* base) {
  auto* queue = reinterpret_cast<Queue*>(base);
  RefreshRetirement(queue);
  if (amdf_atomic_uint64_load_acquire(&queue->retired) !=
      amdf_atomic_uint64_load_acquire(&queue->submitted)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  return ReleaseNative(queue, reinterpret_cast<Device*>(base->device));
}

uint64_t QueryLastSubmission(const amdf_kernel_queue_t* base) {
  return amdf_atomic_uint64_load_acquire(
      &reinterpret_cast<const Queue*>(base)->submitted);
}

const amdf_kernel_queue_native_progress_t* QueryNativeProgress(
    const amdf_kernel_queue_t* base) {
  return &reinterpret_cast<const Queue*>(base)->shared_progress.progress.base;
}

amdf_status_t DestroyDependency(amdf_kernel_queue_native_dependency_t* base) {
  auto* dependency = reinterpret_cast<Dependency*>(base);
  Delete(dependency->host_allocator, dependency);
  return AMDF_STATUS_OK;
}

amdf_status_t CreateDependency(
    amdf_kernel_queue_t* consumer,
    const amdf_kernel_queue_native_progress_t* progress,
    amdf_kernel_queue_native_dependency_t** out_native) {
  if (progress->protocol !=
      AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_WINDOWS_MONITORED_FENCE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  auto* device = reinterpret_cast<Device*>(consumer->device);
  const auto* source =
      reinterpret_cast<const amdf_windows_queue_progress_t*>(progress);
  if (source->adapter_luid != device->adapter_luid) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  Dependency* dependency = nullptr;
  amdf_status_t status = Allocate(consumer->host_allocator, &dependency);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  dependency->host_allocator = consumer->host_allocator;
  dependency->base.destroy = DestroyDependency;
  dependency->base.wait_operation_count = 1;
  status = Hresult(device->native->OpenSharedHandle(
      source->shared_handle, IID_PPV_ARGS(&dependency->fence)));
  if (!amdf_status_is_ok(status)) {
    Delete(consumer->host_allocator, dependency);
    return status;
  }
  *out_native = &dependency->base;
  return AMDF_STATUS_OK;
}

const amdf_kernel_queue_vtable_t kQueueVtable = {
    QueryStatus,     RefreshStatus, RequestNotification, Wait,
    DestroyNative,   nullptr,       QueryLastSubmission, QueryNativeProgress,
    CreateDependency};

}  // namespace

amdf_status_t QueueTerminalStatus(const Queue* queue) {
  const auto* device = reinterpret_cast<const Device*>(queue->base.device);
  const amdf_status_t removed =
      amdf_atomic_uint64_load_acquire(&device->removed_status);
  return amdf_status_is_ok(removed)
             ? amdf_atomic_uint64_load_acquire(&queue->terminal_status)
             : removed;
}

void RefreshRetirement(Queue* queue) {
  auto* device = reinterpret_cast<Device*>(queue->base.device);
  if (!amdf_status_is_ok(
          amdf_atomic_uint64_load_acquire(&device->removed_status))) {
    return;
  }
  const uint64_t submitted = amdf_atomic_uint64_load_acquire(&queue->submitted);
  uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  const uint64_t completed = queue->fence->GetCompletedValue();
  if (completed == UINT64_MAX || completed > retired) {
    const HRESULT removed = device->native->GetDeviceRemovedReason();
    if (FAILED(removed) || completed == UINT64_MAX) {
      Latch(&device->removed_status,
            Hresult(FAILED(removed) ? removed : DXGI_ERROR_DEVICE_REMOVED));
      return;
    }
    // A signal can execute before its host publication completes. Only the
    // published prefix is visible to other callers and eligible for retirement.
    const uint64_t checked = completed < submitted ? completed : submitted;
    while (retired < checked && !amdf_atomic_uint64_compare_exchange_acq_rel(
                                    &queue->retired, &retired, checked)) {
    }
  }
}

amdf_status_t AMDF_CALL
QueueCreate(amdf_device_t* base,
            const amdf_media_kernel_queue_create_info_t* create_info,
            amdf_kernel_queue_t** out_queue) {
  auto* device = reinterpret_cast<Device*>(base);
  if ((create_info->required_capabilities &
       ~device->family.kernel_queue_capabilities) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  Queue* queue = nullptr;
  amdf_status_t status = Allocate(base->host_allocator, &queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  InitializeSRWLock(&queue->wait_lock);
  amdf_atomic_uint32_initialize(&queue->publishing, 0);
  amdf_atomic_uint64_initialize(&queue->submitted, 0);
  amdf_atomic_uint64_initialize(&queue->retired, 0);
  amdf_atomic_uint64_initialize(&queue->terminal_status, AMDF_STATUS_OK);
  D3D12_COMMAND_QUEUE_DESC description = {
      .Type = D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE,
  };
  status = Hresult(device->native->CreateCommandQueue(
      &description, IID_PPV_ARGS(&queue->native)));
  if (amdf_status_is_ok(status)) {
    if ((create_info->required_capabilities &
         AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION) != 0) {
      status = amdf_windows_monitored_fence_initialize(
          device->kmt, device->kmt_device, device->adapter_luid,
          &queue->shared_progress);
      if (amdf_status_is_ok(status)) {
        // This replaces the ordinary progress signal, so interoperability adds
        // no producer signal beyond the queue's existing retirement operation.
        queue->shared_progress.progress.base.signal_operation_count = 0;
        status = Hresult(device->native->OpenSharedHandle(
            queue->shared_progress.progress.shared_handle,
            IID_PPV_ARGS(&queue->fence)));
      }
    } else {
      status = Hresult(device->native->CreateFence(
          0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&queue->fence)));
    }
  }
  if (amdf_status_is_ok(status)) {
    queue->wait_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (queue->wait_event == nullptr) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  if (amdf_status_is_ok(status)) {
    amdf_device_info_t identity = {};
    base->vtable->query_info(base, &identity);
    amdf_kernel_queue_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
        .structure_size = sizeof(info),
        .device_id = identity.id,
        .reset_epoch = identity.reset_epoch,
        .queue_family_ordinal = create_info->queue_family_ordinal,
        .command_type = AMDF_QUEUE_COMMAND_TYPE_MEDIA,
        .maximum_pending_submission_count =
            create_info->maximum_pending_submission_count,
        .maximum_command_count = 1,
        .notification_types = AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT,
        .capabilities = create_info->required_capabilities,
        .maximum_wait_count = create_info->maximum_wait_count != 0
                                  ? create_info->maximum_wait_count
                                  : UINT32_MAX,
    };
    status =
        amdf_kernel_queue_initialize(&queue->base, &kQueueVtable, base, &info);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t cleanup = ReleaseNative(queue, device);
    Delete(base->host_allocator, queue);
    return amdf_status_is_ok(cleanup) ? status : cleanup;
  }
  *out_queue = &queue->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
QueueSubmit(amdf_kernel_queue_t* base,
            const amdf_media_kernel_queue_submission_info_t* submission_info,
            uint64_t* accepted_submission) {
  auto* queue = reinterpret_cast<Queue*>(base);
  auto* command = reinterpret_cast<Command*>(submission_info->command);
  uint32_t expected = 0;
  if (!amdf_atomic_uint32_compare_exchange_acq_rel(&queue->publishing,
                                                   &expected, 1)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_status_t status = QueueTerminalStatus(queue);
  const uint64_t submitted = amdf_atomic_uint64_load_acquire(&queue->submitted);
  const uint64_t retired = amdf_atomic_uint64_load_acquire(&queue->retired);
  if (amdf_status_is_ok(status)) {
    if (!command->prepared || command->submission != 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
    } else if (submitted == UINT64_MAX - 1) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    } else if (submitted - retired >=
               base->info.maximum_pending_submission_count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
  }
  uint32_t accepted_wait_count = 0;
  bool executed = false;
  while (amdf_status_is_ok(status) &&
         accepted_wait_count < submission_info->wait_count) {
    const auto& wait = submission_info->waits[accepted_wait_count];
    const auto* dependency =
        reinterpret_cast<const Dependency*>(wait.dependency->native);
    status = Hresult(
        queue->native->Wait(dependency->fence.Get(), wait.producer_submission));
    if (amdf_status_is_ok(status)) {
      ++accepted_wait_count;
    }
  }
  if (amdf_status_is_ok(status)) {
    ID3D12CommandList* native = command->native.Get();
    queue->native->ExecuteCommandLists(1, &native);
    executed = true;
    command->executed = true;
  }
  if (accepted_wait_count != 0 || executed) {
    const uint64_t point = submitted + 1;
    // Even a wait-only prefix must retain its bindings. A later failure cannot
    // erase native acceptance or turn it into permission to reuse resources.
    const amdf_status_t signal =
        Hresult(queue->native->Signal(queue->fence.Get(), point));
    if (amdf_status_is_ok(status)) {
      status = signal;
    }
    command->submission = point;
    amdf_kernel_queue_dependency_record_waits(accepted_wait_count,
                                              submission_info->waits, point);
    if (!amdf_status_is_ok(status)) {
      Latch(&queue->terminal_status, status);
    }
    amdf_atomic_uint64_store_release(&queue->submitted, point);
    *accepted_submission = point;
  }
  amdf_atomic_uint32_store_release(&queue->publishing, 0);
  return status;
}

}  // namespace amdf::media::d3d12
