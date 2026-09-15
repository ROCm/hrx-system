// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/kernel_queue.h"

#include <cstddef>
#include <cstring>
#include <limits>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#define WIN32_NO_STATUS
#include <windows.h>
#include <winternl.h>
#undef WIN32_NO_STATUS

#include <d3dkmthk.h>
#include <ntstatus.h>

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter_state.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_allocator.h"
#include "wkmi.h"

struct amdf_wkmi_bridge_gpu_kernel_queue_t {
  // Parsed adapter state borrowed for the lifetime of the queue.
  amdf_wkmi_bridge_gpu_adapter_t* adapter = nullptr;
  // Native execution context owning the hardware queue.
  D3DKMT_HANDLE context = 0;
  // Native kernel-mediated hardware queue.
  D3DKMT_HANDLE handle = 0;
  // Fixed WKMI record reused by externally serialized submissions.
  amdf::wkmi_bridge::HostBuffer submission_private_data;
  // True after this queue enters its adapter's live-child count.
  bool counted_live = false;
};

namespace amdf::wkmi_bridge {
namespace {

constexpr uint32_t kCommandBufferAlignment = 4;
constexpr uint64_t kMaximumCommandBufferByteLength =
    std::numeric_limits<uint32_t>::max() & ~uint64_t{3};

amdf_wkmi_bridge_result_t MakeNativeFailure(NTSTATUS status,
                                            uint32_t* out_native_status) {
  *out_native_status = static_cast<uint32_t>(status);
  return AMDF_WKMI_BRIDGE_RESULT_NATIVE_FAILURE;
}

NTSTATUS DestroyNativeQueue(amdf_wkmi_bridge_gpu_kernel_queue_t* queue) {
  if (queue->handle != 0) {
    D3DKMT_DESTROYHWQUEUE destroy_queue = {};
    destroy_queue.hHwQueue = queue->handle;
    const NTSTATUS status = D3DKMTDestroyHwQueue(&destroy_queue);
    if (status != STATUS_SUCCESS) {
      return status;
    }
    queue->handle = 0;
  }
  if (queue->context != 0) {
    D3DKMT_DESTROYCONTEXT destroy_context = {};
    destroy_context.hContext = queue->context;
    const NTSTATUS status = D3DKMTDestroyContext(&destroy_context);
    if (status != STATUS_SUCCESS) {
      return status;
    }
    queue->context = 0;
  }
  return STATUS_SUCCESS;
}

}  // namespace

bool SelectGpuHardwareQueueScheduler(
    Wkmi::DeviceInfo& device_info,
    amdf_wkmi_bridge_gpu_queue_command_type_t command_type,
    uint32_t* out_scheduler) {
  if (command_type == AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4) {
    const uint32_t scheduler = device_info.compute_schedid;
    if (Wkmi::EngineOrdinal(scheduler, &device_info) < 0 ||
        !Wkmi::GetHwsEnabled(scheduler, &device_info)) {
      return false;
    }
    *out_scheduler = scheduler;
    return true;
  }
  if (command_type == AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA) {
    for (int scheduler : device_info.sdma_schedid) {
      if (scheduler >= 0 && Wkmi::EngineOrdinal(scheduler, &device_info) >= 0 &&
          Wkmi::GetHwsEnabled(scheduler, &device_info)) {
        *out_scheduler = static_cast<uint32_t>(scheduler);
        return true;
      }
    }
  }
  return false;
}

namespace {

amdf_wkmi_bridge_result_t CreateNativeQueue(
    amdf_wkmi_bridge_gpu_adapter_t* adapter, uint32_t device_handle,
    amdf_wkmi_bridge_gpu_queue_command_type_t command_type,
    amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
    amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  Wkmi::DeviceInfo& device_info = adapter->device_info;
  uint32_t scheduler = 0;
  if (!SelectGpuHardwareQueueScheduler(device_info, command_type, &scheduler)) {
    return AMDF_WKMI_BRIDGE_RESULT_UNSUPPORTED;
  }
  const int engine_ordinal = Wkmi::EngineOrdinal(scheduler, &device_info);

  const int context_private_size = Wkmi::GetContextPrivDataSize();
  const int queue_private_size = Wkmi::GetHwQueuePrivDataSize();
  const int submission_private_size = Wkmi::GetSubmitPrivDataSize();
  if (context_private_size <= 0 || queue_private_size <= 0 ||
      submission_private_size <= 0) {
    return AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH;
  }

  HostObject<amdf_wkmi_bridge_gpu_kernel_queue_t> queue(
      adapter->host_allocator);
  if (!queue.Allocate()) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  HostBuffer context_private;
  HostBuffer queue_private;
  if (!context_private.Allocate(adapter->host_allocator,
                                static_cast<size_t>(context_private_size)) ||
      !queue_private.Allocate(adapter->host_allocator,
                              static_cast<size_t>(queue_private_size)) ||
      !queue->submission_private_data.Allocate(
          adapter->host_allocator,
          static_cast<size_t>(submission_private_size))) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  queue->adapter = adapter;

  Wkmi::FillinContextPrivData(context_private.data(),
                              device_info.state_shadowing_by_cpfw, scheduler,
                              0);
  D3DKMT_CREATECONTEXTVIRTUAL create_context = {};
  create_context.hDevice = static_cast<D3DKMT_HANDLE>(device_handle);
  create_context.EngineAffinity = 1;
  create_context.NodeOrdinal = static_cast<uint32_t>(engine_ordinal);
  create_context.Flags.HwQueueSupported = 1;
  create_context.pPrivateDriverData = context_private.data();
  create_context.PrivateDriverDataSize =
      static_cast<uint32_t>(context_private.byte_length());
  create_context.ClientHint = D3DKMT_CLIENTHINT_OPENCL;
  NTSTATUS native_status = D3DKMTCreateContextVirtual(&create_context);
  if (native_status != STATUS_SUCCESS) {
    return MakeNativeFailure(native_status, out_native_status);
  }
  queue->context = create_context.hContext;
  if (queue->context == 0) {
    native_status = STATUS_INVALID_HANDLE;
  }

  D3DKMT_CREATEHWQUEUE create_queue = {};
  if (native_status == STATUS_SUCCESS) {
    Wkmi::FillinHwQueuePrivData(queue_private.data(),
                                device_info.state_shadowing_by_cpfw,
                                Wkmi::kNormal, false);
    create_queue.hHwContext = queue->context;
    // Retain the operating-system watchdog. Disabling it requires an explicit
    // queue contract rather than an implicit device-profile side effect.
    create_queue.Flags.DisableGpuTimeout = 0;
    create_queue.pPrivateDriverData = queue_private.data();
    create_queue.PrivateDriverDataSize =
        static_cast<uint32_t>(queue_private.byte_length());
    native_status = D3DKMTCreateHwQueue(&create_queue);
  }
  if (native_status == STATUS_SUCCESS) {
    queue->handle = create_queue.hHwQueue;
    if (queue->handle == 0 || create_queue.hHwQueueProgressFence == 0 ||
        create_queue.HwQueueProgressFenceCPUVirtualAddress == nullptr ||
        create_queue.HwQueueProgressFenceGPUVirtualAddress == 0) {
      native_status = STATUS_INVALID_HANDLE;
    }
  }
  if (native_status != STATUS_SUCCESS) {
    const NTSTATUS rollback_status = DestroyNativeQueue(queue.get());
    if (rollback_status != STATUS_SUCCESS) {
      // No command has been submitted. Native handles do not borrow the
      // unpublished queue or its unused submission-private host storage.
      return MakeNativeFailure(rollback_status, out_native_status);
    }
    return MakeNativeFailure(native_status, out_native_status);
  }

  amdf_wkmi_bridge_gpu_kernel_queue_info_t info = {};
  info.structure_size = sizeof(info);
  info.command_buffer_alignment = kCommandBufferAlignment;
  info.maximum_command_buffer_byte_length = kMaximumCommandBufferByteLength;
  info.progress_fence_handle = create_queue.hHwQueueProgressFence;
  info.progress_fence_pointer = static_cast<const volatile uint64_t*>(
      create_queue.HwQueueProgressFenceCPUVirtualAddress);
  info.progress_fence_device_address =
      create_queue.HwQueueProgressFenceGPUVirtualAddress;

  AcquireSRWLockExclusive(&adapter->state_lock);
  ++adapter->live_queue_count;
  queue->counted_live = true;
  ReleaseSRWLockExclusive(&adapter->state_lock);
  *out_info = info;
  *out_queue = queue.release();
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

}  // namespace

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuKernelQueueCreate(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_wkmi_bridge_gpu_kernel_queue_t** out_queue,
    amdf_wkmi_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || create_info == nullptr ||
      create_info->structure_size < sizeof(*create_info) ||
      create_info->device_handle == 0 ||
      (create_info->command_type !=
           AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4 &&
       create_info->command_type !=
           AMDF_WKMI_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA) ||
      create_info->reserved != 0 || out_queue == nullptr ||
      out_info == nullptr || out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  try {
    return CreateNativeQueue(adapter, create_info->device_handle,
                             create_info->command_type, out_queue, out_info,
                             out_native_status);
  } catch (const std::bad_alloc&) {
    return AMDF_WKMI_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  } catch (...) {
    return AMDF_WKMI_BRIDGE_RESULT_INTERNAL;
  }
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL GpuKernelQueueSubmit(
    amdf_wkmi_bridge_gpu_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t progress_value,
    uint32_t* out_native_status) noexcept {
  if (queue == nullptr || queue->handle == 0 || out_native_status == nullptr ||
      command_buffer_address == 0 ||
      (command_buffer_address & (kCommandBufferAlignment - 1)) != 0 ||
      command_buffer_byte_length == 0 ||
      (command_buffer_byte_length & (kCommandBufferAlignment - 1)) != 0 ||
      command_buffer_byte_length > kMaximumCommandBufferByteLength ||
      progress_value == 0) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  std::memset(queue->submission_private_data.data(), 0,
              queue->submission_private_data.byte_length());
  Wkmi::FillinSubmitPrivData(queue->submission_private_data.data(),
                             queue->handle, command_buffer_address,
                             command_buffer_byte_length, true);

  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = queue->handle;
  submit.HwQueueProgressFenceId = progress_value;
  submit.CommandBuffer = command_buffer_address;
  submit.CommandLength = static_cast<uint32_t>(command_buffer_byte_length);
  submit.pPrivateDriverData = queue->submission_private_data.data();
  submit.PrivateDriverDataSize =
      static_cast<uint32_t>(queue->submission_private_data.byte_length());
  const NTSTATUS native_status = D3DKMTSubmitCommandToHwQueue(&submit);
  return native_status == STATUS_SUCCESS
             ? AMDF_WKMI_BRIDGE_RESULT_SUCCESS
             : MakeNativeFailure(native_status, out_native_status);
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
GpuKernelQueueDestroy(amdf_wkmi_bridge_gpu_kernel_queue_t* queue,
                      uint32_t* out_native_status) noexcept {
  if (queue == nullptr || out_native_status == nullptr) {
    return AMDF_WKMI_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  AcquireSRWLockExclusive(&queue->adapter->state_lock);
  if (!queue->counted_live || queue->adapter->live_queue_count == 0) {
    ReleaseSRWLockExclusive(&queue->adapter->state_lock);
    return AMDF_WKMI_BRIDGE_RESULT_INTERNAL;
  }
  const NTSTATUS native_status = DestroyNativeQueue(queue);
  if (native_status == STATUS_SUCCESS) {
    --queue->adapter->live_queue_count;
    queue->counted_live = false;
  }
  ReleaseSRWLockExclusive(&queue->adapter->state_lock);
  if (native_status != STATUS_SUCCESS) {
    return MakeNativeFailure(native_status, out_native_status);
  }
  const amdf_allocator_t host_allocator = queue->adapter->host_allocator;
  DestroyHostObject(host_allocator, queue);
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

}  // namespace amdf::wkmi_bridge
