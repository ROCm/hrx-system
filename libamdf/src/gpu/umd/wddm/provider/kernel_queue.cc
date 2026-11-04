// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/provider/kernel_queue.h"

#include <array>
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

#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/provider_allocator.h"
#include "libamdf/src/gpu/umd/wddm/provider/adapter_state.h"

struct amdf_dxg_provider_gpu_kernel_queue_t {
  // Parsed adapter state borrowed for the lifetime of the queue.
  amdf_dxg_provider_gpu_adapter_t* adapter = nullptr;
  // Native execution context owning all accepted commands.
  D3DKMT_HANDLE context = 0;
  // Hardware queue, or zero for a software-scheduled context.
  D3DKMT_HANDLE handle = 0;
  // Separately owned monitored fence for SWS; HWS owns its fence implicitly.
  D3DKMT_HANDLE progress_fence = 0;
};

namespace amdf::wddm_provider {

using amdf::dxg_provider::DestroyHostObject;
using amdf::dxg_provider::HostObject;
namespace {

constexpr uint32_t kCommandBufferAlignment = 4;
constexpr uint64_t kMaximumCommandBufferByteLength =
    std::numeric_limits<uint32_t>::max() & ~uint64_t{3};

amdf_dxg_provider_result_t MakeNativeFailure(NTSTATUS status,
                                             uint32_t* out_native_status) {
  *out_native_status = static_cast<uint32_t>(status);
  return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
}

NTSTATUS DestroyNativeQueue(amdf_dxg_provider_gpu_kernel_queue_t* queue) {
  NTSTATUS first_status = STATUS_SUCCESS;
  if (queue->handle != 0) {
    D3DKMT_DESTROYHWQUEUE destroy_queue = {};
    destroy_queue.hHwQueue = queue->handle;
    const NTSTATUS status =
        queue->adapter->kmt->destroy_hardware_queue(&destroy_queue);
    if (first_status == STATUS_SUCCESS && status != STATUS_SUCCESS) {
      first_status = status;
    }
    queue->handle = 0;
  }
  if (queue->context != 0) {
    D3DKMT_DESTROYCONTEXT destroy_context = {};
    destroy_context.hContext = queue->context;
    const NTSTATUS status =
        queue->adapter->kmt->destroy_context(&destroy_context);
    if (first_status == STATUS_SUCCESS && status != STATUS_SUCCESS) {
      first_status = status;
    }
    queue->context = 0;
  }
  if (queue->progress_fence != 0) {
    D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy_fence = {};
    destroy_fence.hSyncObject = queue->progress_fence;
    const NTSTATUS status =
        queue->adapter->kmt->destroy_synchronization_object(&destroy_fence);
    if (first_status == STATUS_SUCCESS && status != STATUS_SUCCESS) {
      first_status = status;
    }
    queue->progress_fence = 0;
  }
  return first_status;
}

amdf_dxg_provider_result_t CreateNativeQueue(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint32_t device_handle,
    amdf_dxg_provider_gpu_queue_command_type_t command_type,
    amdf_dxg_provider_gpu_kernel_queue_t** out_queue,
    amdf_dxg_provider_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  const amdf_dxg_private_queue_profile_t* profile =
      SelectGpuKernelQueueProfile(adapter, command_type);
  if (profile == nullptr || !profile->is_supported) {
    return AMDF_DXG_PROVIDER_RESULT_UNSUPPORTED;
  }

  HostObject<amdf_dxg_provider_gpu_kernel_queue_t> queue(
      adapter->host_allocator);
  if (!queue.Allocate()) {
    return AMDF_DXG_PROVIDER_RESULT_RESOURCE_EXHAUSTED;
  }
  queue->adapter = adapter;

  std::array<uint8_t, AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH>
      context_private;
  amdf_dxg_private_initialize_context_record(
      adapter->private_info.supports_firmware_state_shadowing,
      profile->scheduler_id, 0, context_private.data());
  D3DKMT_CREATECONTEXTVIRTUAL create_context = {};
  create_context.hDevice = static_cast<D3DKMT_HANDLE>(device_handle);
  create_context.EngineAffinity = 1;
  create_context.NodeOrdinal = profile->node_ordinal;
  create_context.Flags.HwQueueSupported = profile->uses_hardware_scheduling;
  create_context.pPrivateDriverData = context_private.data();
  create_context.PrivateDriverDataSize =
      static_cast<uint32_t>(context_private.size());
  create_context.ClientHint = D3DKMT_CLIENTHINT_OPENCL;
  NTSTATUS native_status =
      adapter->kmt->create_context_virtual(&create_context);
  if (native_status != STATUS_SUCCESS) {
    return MakeNativeFailure(native_status, out_native_status);
  }
  queue->context = create_context.hContext;
  if (queue->context == 0) {
    native_status = STATUS_INVALID_HANDLE;
  }

  D3DKMT_CREATEHWQUEUE create_queue = {};
  D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create_fence = {};
  std::array<uint8_t, AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH>
      queue_private;
  if (native_status == STATUS_SUCCESS && profile->uses_hardware_scheduling) {
    amdf_dxg_private_initialize_hardware_queue_record(
        adapter->private_info.supports_firmware_state_shadowing,
        AMDF_DXG_PRIVATE_QUEUE_PRIORITY_NORMAL, 0, queue_private.data());
    create_queue.hHwContext = queue->context;
    // Retain the operating-system watchdog. Disabling it requires an explicit
    // queue contract rather than an implicit device-profile side effect.
    create_queue.Flags.DisableGpuTimeout = 0;
    create_queue.pPrivateDriverData = queue_private.data();
    create_queue.PrivateDriverDataSize =
        static_cast<uint32_t>(queue_private.size());
    native_status = adapter->kmt->create_hardware_queue(&create_queue);
  }
  if (native_status == STATUS_SUCCESS && profile->uses_hardware_scheduling) {
    queue->handle = create_queue.hHwQueue;
    if (queue->handle == 0 || create_queue.hHwQueueProgressFence == 0 ||
        create_queue.HwQueueProgressFenceCPUVirtualAddress == nullptr ||
        create_queue.HwQueueProgressFenceGPUVirtualAddress == 0) {
      native_status = STATUS_INVALID_HANDLE;
    }
  }
  if (native_status == STATUS_SUCCESS && !profile->uses_hardware_scheduling) {
    create_fence.hDevice = static_cast<D3DKMT_HANDLE>(device_handle);
    create_fence.Info.Type = D3DDDI_MONITORED_FENCE;
    create_fence.Info.MonitoredFence.EngineAffinity = 1;
    native_status = adapter->kmt->create_synchronization_object(&create_fence);
    if (native_status == STATUS_SUCCESS) {
      queue->progress_fence = create_fence.hSyncObject;
      if (queue->progress_fence == 0 ||
          create_fence.Info.MonitoredFence.FenceValueCPUVirtualAddress ==
              nullptr ||
          create_fence.Info.MonitoredFence.FenceValueGPUVirtualAddress == 0) {
        native_status = STATUS_INVALID_HANDLE;
      }
    }
  }
  if (native_status != STATUS_SUCCESS) {
    const NTSTATUS rollback_status = DestroyNativeQueue(queue.get());
    if (rollback_status != STATUS_SUCCESS) {
      // No command has been submitted. Native handles do not borrow the
      // unpublished host queue object.
      return MakeNativeFailure(rollback_status, out_native_status);
    }
    return MakeNativeFailure(native_status, out_native_status);
  }

  amdf_dxg_provider_gpu_kernel_queue_info_t info = {};
  info.command_buffer_alignment = kCommandBufferAlignment;
  info.maximum_command_buffer_byte_length = kMaximumCommandBufferByteLength;
  info.progress_fence_handle = profile->uses_hardware_scheduling
                                   ? create_queue.hHwQueueProgressFence
                                   : create_fence.hSyncObject;
  info.context_handle = queue->context;
  info.queue_handle = queue->handle;
  info.progress_fence_pointer = static_cast<const volatile uint64_t*>(
      profile->uses_hardware_scheduling
          ? create_queue.HwQueueProgressFenceCPUVirtualAddress
          : create_fence.Info.MonitoredFence.FenceValueCPUVirtualAddress);
  info.progress_fence_device_address =
      profile->uses_hardware_scheduling
          ? create_queue.HwQueueProgressFenceGPUVirtualAddress
          : create_fence.Info.MonitoredFence.FenceValueGPUVirtualAddress;

  adapter->live_queue_count.fetch_add(1, std::memory_order_release);
  *out_info = info;
  *out_queue = queue.release();
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

}  // namespace

amdf_dxg_provider_result_t GpuKernelQueueCreate(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_provider_gpu_kernel_queue_t** out_queue,
    amdf_dxg_provider_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || create_info == nullptr ||
      create_info->device_handle == 0 ||
      (create_info->command_type !=
           AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4 &&
       create_info->command_type !=
           AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA) ||
      out_queue == nullptr || out_info == nullptr ||
      out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  return CreateNativeQueue(adapter, create_info->device_handle,
                           create_info->command_type, out_queue, out_info,
                           out_native_status);
}

amdf_dxg_provider_result_t GpuKernelQueueSubmit(
    amdf_dxg_provider_gpu_kernel_queue_t* queue,
    uint64_t command_buffer_address, uint64_t command_buffer_byte_length,
    uint64_t progress_value, uint32_t* out_accepted,
    uint32_t* out_native_status) noexcept {
  if (queue == nullptr || queue->context == 0 || out_accepted == nullptr ||
      out_native_status == nullptr || command_buffer_address == 0 ||
      (command_buffer_address & (kCommandBufferAlignment - 1)) != 0 ||
      command_buffer_byte_length == 0 ||
      (command_buffer_byte_length & (kCommandBufferAlignment - 1)) != 0 ||
      command_buffer_byte_length > kMaximumCommandBufferByteLength ||
      progress_value == 0) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_accepted = 0;
  *out_native_status = 0;
  std::array<uint8_t, AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH>
      submission_private_data;
  amdf_dxg_private_initialize_submission_record(
      queue->handle, command_buffer_address,
      static_cast<uint32_t>(command_buffer_byte_length), queue->handle != 0,
      submission_private_data.data());

  if (queue->handle == 0) {
    D3DKMT_SUBMITCOMMAND submit = {};
    submit.Commands = command_buffer_address;
    submit.CommandLength = static_cast<uint32_t>(command_buffer_byte_length);
    submit.BroadcastContextCount = 1;
    submit.BroadcastContext[0] = queue->context;
    submit.pPrivateDriverData = submission_private_data.data();
    submit.PrivateDriverDataSize =
        static_cast<uint32_t>(submission_private_data.size());
    NTSTATUS native_status = queue->adapter->kmt->submit_command(&submit);
    if (native_status != STATUS_SUCCESS) {
      return MakeNativeFailure(native_status, out_native_status);
    }
    *out_accepted = 1;
    D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU signal = {};
    signal.hContext = queue->context;
    signal.ObjectCount = 1;
    signal.ObjectHandleArray = &queue->progress_fence;
    signal.MonitoredFenceValueArray = &progress_value;
    native_status = queue->adapter->kmt->signal_from_gpu(&signal);
    return native_status == STATUS_SUCCESS
               ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
               : MakeNativeFailure(native_status, out_native_status);
  }

  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = queue->handle;
  submit.HwQueueProgressFenceId = progress_value;
  submit.CommandBuffer = command_buffer_address;
  submit.CommandLength = static_cast<uint32_t>(command_buffer_byte_length);
  submit.pPrivateDriverData = submission_private_data.data();
  submit.PrivateDriverDataSize =
      static_cast<uint32_t>(submission_private_data.size());
  const NTSTATUS native_status =
      queue->adapter->kmt->submit_command_to_hardware_queue(&submit);
  *out_accepted = native_status == STATUS_SUCCESS;
  return native_status == STATUS_SUCCESS
             ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
             : MakeNativeFailure(native_status, out_native_status);
}

amdf_dxg_provider_result_t GpuKernelQueueDestroy(
    amdf_dxg_provider_gpu_kernel_queue_t* queue,
    uint32_t* out_native_status) noexcept {
  if (queue == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_PROVIDER_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const NTSTATUS native_status = DestroyNativeQueue(queue);
  queue->adapter->live_queue_count.fetch_sub(1, std::memory_order_acq_rel);
  // Native handles do not borrow this object. Failed removal leaks native
  // residue, not a retry owner.
  const amdf_allocator_t host_allocator = queue->adapter->host_allocator;
  DestroyHostObject(host_allocator, queue);
  return native_status == STATUS_SUCCESS
             ? AMDF_DXG_PROVIDER_RESULT_SUCCESS
             : MakeNativeFailure(native_status, out_native_status);
}

}  // namespace amdf::wddm_provider
