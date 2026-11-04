// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/kernel_queue.h"

#include <sys/ioctl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "libamdf/src/gpu/umd/dxg/bridge_allocator.h"
#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "shared/include/d3dkmt_types.h"

struct amdf_dxg_bridge_gpu_kernel_queue_t {
  // Parsed adapter state borrowed through queue destruction.
  amdf_dxg_bridge_gpu_adapter_t* adapter = nullptr;
  // Native execution context owning all accepted commands.
  D3DKMT_HANDLE context = 0;
  // Hardware queue, or zero for a software-scheduled context.
  D3DKMT_HANDLE handle = 0;
  // Separately owned monitored fence for software scheduling.
  D3DKMT_HANDLE progress_fence = 0;
};

namespace amdf::rocdxg_bridge {
namespace {

using amdf::dxg_bridge::DestroyHostObject;
using amdf::dxg_bridge::HostObject;

constexpr uint32_t kCreateContextOperation = 0x04;
constexpr uint32_t kDestroyContextOperation = 0x05;
constexpr uint32_t kSubmitCommandOperation = 0x0F;
constexpr uint32_t kCreateSynchronizationObjectOperation = 0x10;
constexpr uint32_t kCreateHardwareQueueOperation = 0x18;
constexpr uint32_t kDestroyHardwareQueueOperation = 0x1B;
constexpr uint32_t kDestroySynchronizationObjectOperation = 0x1D;
constexpr uint32_t kSignalSynchronizationObjectFromGpuOperation = 0x32;
constexpr uint32_t kSubmitCommandToHardwareQueueOperation = 0x34;

constexpr uint32_t kCommandBufferAlignment = 4;
constexpr uint64_t kMaximumCommandBufferByteLength =
    std::numeric_limits<uint32_t>::max() & ~uint64_t{3};

template <uint32_t Operation, typename Arguments>
NTSTATUS InvokeNative(const amdf_dxg_bridge_gpu_adapter_t* adapter,
                      Arguments* arguments) {
  return static_cast<NTSTATUS>(adapter->native_transport.invoke(
      adapter->native_transport.context, _IOWR('G', Operation, Arguments),
      arguments));
}

amdf_dxg_bridge_result_t MakeNativeFailure(NTSTATUS status,
                                           uint32_t* out_native_status) {
  *out_native_status = static_cast<uint32_t>(status);
  return AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE;
}

void RetainFirstFailure(NTSTATUS candidate, NTSTATUS* inout_status) {
  if (*inout_status == STATUS_SUCCESS && candidate != STATUS_SUCCESS) {
    *inout_status = candidate;
  }
}

NTSTATUS DestroyNativeQueue(amdf_dxg_bridge_gpu_kernel_queue_t* queue) {
  NTSTATUS first_status = STATUS_SUCCESS;
  if (queue->handle != 0) {
    D3DKMT_DESTROYHWQUEUE destroy = {.hHwQueue = queue->handle};
    RetainFirstFailure(
        InvokeNative<kDestroyHardwareQueueOperation>(queue->adapter, &destroy),
        &first_status);
    queue->handle = 0;
  }
  if (queue->context != 0) {
    D3DKMT_DESTROYCONTEXT destroy = {.hContext = queue->context};
    RetainFirstFailure(
        InvokeNative<kDestroyContextOperation>(queue->adapter, &destroy),
        &first_status);
    queue->context = 0;
  }
  if (queue->progress_fence != 0) {
    D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {
        .hSyncObject = queue->progress_fence,
    };
    RetainFirstFailure(InvokeNative<kDestroySynchronizationObjectOperation>(
                           queue->adapter, &destroy),
                       &first_status);
    queue->progress_fence = 0;
  }
  return first_status;
}

amdf_dxg_bridge_result_t CreateNativeQueue(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_kernel_queue_create_info_t& create_info,
    amdf_dxg_bridge_gpu_kernel_queue_t** out_queue,
    amdf_dxg_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  const amdf_dxg_private_queue_profile_t* profile =
      SelectGpuKernelQueueProfile(adapter, create_info.command_type);
  if (profile == nullptr || !profile->is_supported) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }

  HostObject<amdf_dxg_bridge_gpu_kernel_queue_t> queue(adapter->host_allocator);
  if (!queue.Allocate()) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  queue->adapter = adapter;

  std::array<uint8_t, AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH>
      context_private;
  amdf_dxg_private_initialize_context_record(
      adapter->private_info.supports_firmware_state_shadowing,
      profile->scheduler_id, 0, context_private.data());
  D3DKMT_CREATECONTEXTVIRTUAL create_context = {
      .hDevice = static_cast<D3DKMT_HANDLE>(create_info.device_handle),
      .NodeOrdinal = profile->node_ordinal,
      .EngineAffinity = 1,
      .Flags = {.HwQueueSupported = profile->uses_hardware_scheduling},
      .pPrivateDriverData = context_private.data(),
      .PrivateDriverDataSize = static_cast<uint32_t>(context_private.size()),
      .ClientHint = D3DKMT_CLIENTHINT_OPENCL,
  };
  NTSTATUS native_status =
      InvokeNative<kCreateContextOperation>(adapter, &create_context);
  if (native_status == STATUS_SUCCESS) {
    queue->context = create_context.hContext;
    if (queue->context == 0) {
      native_status = STATUS_INVALID_HANDLE;
    }
  }

  D3DKMT_CREATEHWQUEUE create_queue = {};
  D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create_fence = {};
  std::array<uint8_t, AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH>
      queue_private;
  if (native_status == STATUS_SUCCESS && profile->uses_hardware_scheduling) {
    // This queue publishes complete immutable PM4 or SDMA command buffers;
    // both use the operating-system PM4 hardware-queue representation.
    amdf_dxg_private_initialize_hardware_queue_record(
        adapter->private_info.supports_firmware_state_shadowing,
        AMDF_DXG_PRIVATE_QUEUE_PRIORITY_NORMAL, 0, queue_private.data());
    create_queue.hHwContext = queue->context;
    create_queue.Flags.DisableGpuTimeout = 0;
    create_queue.pPrivateDriverData = queue_private.data();
    create_queue.PrivateDriverDataSize =
        static_cast<uint32_t>(queue_private.size());
    native_status =
        InvokeNative<kCreateHardwareQueueOperation>(adapter, &create_queue);
    if (native_status == STATUS_SUCCESS) {
      queue->handle = create_queue.hHwQueue;
      if (queue->handle == 0 || create_queue.hHwQueueProgressFence == 0 ||
          create_queue.HwQueueProgressFenceCPUVirtualAddress == nullptr ||
          create_queue.HwQueueProgressFenceGPUVirtualAddress == 0) {
        native_status = STATUS_INVALID_HANDLE;
      }
    }
  }
  if (native_status == STATUS_SUCCESS && !profile->uses_hardware_scheduling) {
    create_fence.hDevice =
        static_cast<D3DKMT_HANDLE>(create_info.device_handle);
    create_fence.Info.Type = D3DDDI_MONITORED_FENCE;
    create_fence.Info.MonitoredFence.EngineAffinity = 1;
    native_status = InvokeNative<kCreateSynchronizationObjectOperation>(
        adapter, &create_fence);
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
    return MakeNativeFailure(
        rollback_status != STATUS_SUCCESS ? rollback_status : native_status,
        out_native_status);
  }

  const amdf_dxg_bridge_gpu_kernel_queue_info_t info = {
      .structure_size = sizeof(amdf_dxg_bridge_gpu_kernel_queue_info_t),
      .command_buffer_alignment = kCommandBufferAlignment,
      .maximum_command_buffer_byte_length = kMaximumCommandBufferByteLength,
      .progress_fence_handle = profile->uses_hardware_scheduling
                                   ? create_queue.hHwQueueProgressFence
                                   : create_fence.hSyncObject,
      .reserved = 0,
      .progress_fence_pointer = static_cast<const volatile uint64_t*>(
          profile->uses_hardware_scheduling
              ? create_queue.HwQueueProgressFenceCPUVirtualAddress
              : create_fence.Info.MonitoredFence.FenceValueCPUVirtualAddress),
      .progress_fence_device_address =
          profile->uses_hardware_scheduling
              ? create_queue.HwQueueProgressFenceGPUVirtualAddress
              : create_fence.Info.MonitoredFence.FenceValueGPUVirtualAddress,
      .context_handle = queue->context,
      .queue_handle = queue->handle,
  };
  adapter->live_queue_count.fetch_add(1, std::memory_order_release);
  *out_info = info;
  *out_queue = queue.release();
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

}  // namespace

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuKernelQueueCreate(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_bridge_gpu_kernel_queue_t** out_queue,
    amdf_dxg_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) noexcept {
  if (adapter == nullptr || create_info == nullptr ||
      create_info->structure_size < sizeof(*create_info) ||
      create_info->device_handle == 0 ||
      (create_info->command_type !=
           AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4 &&
       create_info->command_type !=
           AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA) ||
      create_info->reserved != 0 || out_queue == nullptr ||
      out_info == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  return CreateNativeQueue(adapter, *create_info, out_queue, out_info,
                           out_native_status);
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL GpuKernelQueueSubmit(
    amdf_dxg_bridge_gpu_kernel_queue_t* queue, uint64_t command_buffer_address,
    uint64_t command_buffer_byte_length, uint64_t progress_value,
    uint32_t* out_accepted, uint32_t* out_native_status) noexcept {
  if (queue == nullptr || queue->context == 0 || out_accepted == nullptr ||
      out_native_status == nullptr || command_buffer_address == 0 ||
      (command_buffer_address & (kCommandBufferAlignment - 1)) != 0 ||
      command_buffer_byte_length == 0 ||
      (command_buffer_byte_length & (kCommandBufferAlignment - 1)) != 0 ||
      command_buffer_byte_length > kMaximumCommandBufferByteLength ||
      progress_value == 0) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
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
    D3DKMT_SUBMITCOMMAND submit = {
        .Commands = command_buffer_address,
        .CommandLength = static_cast<uint32_t>(command_buffer_byte_length),
        .BroadcastContextCount = 1,
    };
    submit.BroadcastContext[0] = queue->context;
    submit.pPrivateDriverData = submission_private_data.data();
    submit.PrivateDriverDataSize =
        static_cast<uint32_t>(submission_private_data.size());
    NTSTATUS native_status =
        InvokeNative<kSubmitCommandOperation>(queue->adapter, &submit);
    if (native_status != STATUS_SUCCESS) {
      return MakeNativeFailure(native_status, out_native_status);
    }
    *out_accepted = 1;
    D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU signal = {
        .hContext = queue->context,
        .ObjectCount = 1,
        .ObjectHandleArray = &queue->progress_fence,
        .MonitoredFenceValueArray = &progress_value,
    };
    native_status = InvokeNative<kSignalSynchronizationObjectFromGpuOperation>(
        queue->adapter, &signal);
    return native_status == STATUS_SUCCESS
               ? AMDF_DXG_BRIDGE_RESULT_SUCCESS
               : MakeNativeFailure(native_status, out_native_status);
  }

  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {
      .hHwQueue = queue->handle,
      .HwQueueProgressFenceId = progress_value,
      .CommandBuffer = command_buffer_address,
      .CommandLength = static_cast<uint32_t>(command_buffer_byte_length),
      .PrivateDriverDataSize =
          static_cast<uint32_t>(submission_private_data.size()),
      .pPrivateDriverData = submission_private_data.data(),
  };
  const NTSTATUS native_status =
      InvokeNative<kSubmitCommandToHardwareQueueOperation>(queue->adapter,
                                                           &submit);
  *out_accepted = native_status == STATUS_SUCCESS;
  return native_status == STATUS_SUCCESS
             ? AMDF_DXG_BRIDGE_RESULT_SUCCESS
             : MakeNativeFailure(native_status, out_native_status);
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
GpuKernelQueueDestroy(amdf_dxg_bridge_gpu_kernel_queue_t* queue,
                      uint32_t* out_native_status) noexcept {
  if (queue == nullptr || out_native_status == nullptr) {
    return AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT;
  }
  *out_native_status = 0;
  const NTSTATUS native_status = DestroyNativeQueue(queue);
  amdf_dxg_bridge_gpu_adapter_t* adapter = queue->adapter;
  adapter->live_queue_count.fetch_sub(1, std::memory_order_acq_rel);
  const amdf_allocator_t host_allocator = adapter->host_allocator;
  DestroyHostObject(host_allocator, queue);
  return native_status == STATUS_SUCCESS
             ? AMDF_DXG_BRIDGE_RESULT_SUCCESS
             : MakeNativeFailure(native_status, out_native_status);
}

}  // namespace amdf::rocdxg_bridge
