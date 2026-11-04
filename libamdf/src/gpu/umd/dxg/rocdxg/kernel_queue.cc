// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/kernel_queue.h"

#include <sys/ioctl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "libamdf/src/gpu/umd/dxg/bridge_allocator.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "shared/include/d3dkmt_types.h"
#include "wkmi.h"

struct amdf_dxg_bridge_gpu_kernel_queue_t {
  // Parsed adapter state borrowed through queue destruction.
  amdf_dxg_bridge_gpu_adapter_t* adapter = nullptr;
  // Native execution context owning all accepted commands.
  D3DKMT_HANDLE context = 0;
  // Hardware queue, or zero for a software-scheduled context.
  D3DKMT_HANDLE handle = 0;
  // Separately owned monitored fence for software scheduling.
  D3DKMT_HANDLE progress_fence = 0;
  // Fixed WKMI record reused by externally serialized submissions.
  amdf::dxg_bridge::HostBuffer submission_private_data;
};

namespace amdf::rocdxg_bridge {
namespace {

using amdf::dxg_bridge::DestroyHostObject;
using amdf::dxg_bridge::HostBuffer;
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

bool TryMakeQueueProfile(const thunk_proxy::DeviceContext& device,
                         uint32_t scheduler_id, bool require_hws,
                         amdf_rocdxg_kernel_queue_profile_t* out_profile) {
  if (scheduler_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const int scheduler = static_cast<int>(scheduler_id);
  const int node_ordinal = device.EngineOrdinal(scheduler);
  if (node_ordinal < 0) {
    return false;
  }
  const bool uses_hardware_scheduling = device.IsHwsEnabled(scheduler);
  if (require_hws && !uses_hardware_scheduling) {
    return false;
  }
  *out_profile = {
      .scheduler_id = scheduler_id,
      .node_ordinal = static_cast<uint32_t>(node_ordinal),
      .is_supported = true,
      .uses_hardware_scheduling = uses_hardware_scheduling,
  };
  return true;
}

amdf_dxg_bridge_result_t CreateNativeQueue(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_kernel_queue_create_info_t& create_info,
    amdf_dxg_bridge_gpu_kernel_queue_t** out_queue,
    amdf_dxg_bridge_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  const amdf_rocdxg_kernel_queue_profile_t* profile =
      SelectGpuKernelQueueProfile(adapter, create_info.command_type);
  if (profile == nullptr || !profile->is_supported) {
    return AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  }

  const int context_private_size = Wkmi::GetContextPrivDataSize();
  const int queue_private_size = Wkmi::GetHwQueuePrivDataSize();
  const int submission_private_size = Wkmi::GetSubmitPrivDataSize();
  if (context_private_size <= 0 ||
      (profile->uses_hardware_scheduling && queue_private_size <= 0) ||
      submission_private_size <= 0) {
    return AMDF_DXG_BRIDGE_RESULT_VERSION_MISMATCH;
  }

  HostObject<amdf_dxg_bridge_gpu_kernel_queue_t> queue(adapter->host_allocator);
  if (!queue.Allocate()) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  HostBuffer context_private;
  HostBuffer queue_private;
  if (!context_private.Allocate(adapter->host_allocator,
                                static_cast<size_t>(context_private_size)) ||
      (profile->uses_hardware_scheduling &&
       !queue_private.Allocate(adapter->host_allocator,
                               static_cast<size_t>(queue_private_size))) ||
      !queue->submission_private_data.Allocate(
          adapter->host_allocator,
          static_cast<size_t>(submission_private_size))) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  }
  queue->adapter = adapter;

  Wkmi::FillinContextPrivData(context_private.data(),
                              adapter->state_shadowing_by_cpfw,
                              profile->scheduler_id, 0, false);
  D3DKMT_CREATECONTEXTVIRTUAL create_context = {
      .hDevice = static_cast<D3DKMT_HANDLE>(create_info.device_handle),
      .NodeOrdinal = profile->node_ordinal,
      .EngineAffinity = 1,
      .Flags = {.HwQueueSupported = profile->uses_hardware_scheduling},
      .pPrivateDriverData = context_private.data(),
      .PrivateDriverDataSize =
          static_cast<uint32_t>(context_private.byte_length()),
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
  if (native_status == STATUS_SUCCESS && profile->uses_hardware_scheduling) {
    // This queue publishes complete immutable PM4 or SDMA command buffers.
    // WKMI's SDMA kind instead describes a persistent user-mode ring with a
    // doorbell; the operating-system queue representation remains PM4.
    Wkmi::FillinHwQueuePrivData(queue_private.data(),
                                adapter->state_shadowing_by_cpfw, Wkmi::kNormal,
                                Wkmi::kHwQueuePm4);
    create_queue.hHwContext = queue->context;
    create_queue.Flags.DisableGpuTimeout = 0;
    create_queue.pPrivateDriverData = queue_private.data();
    create_queue.PrivateDriverDataSize =
        static_cast<uint32_t>(queue_private.byte_length());
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

void InitializeGpuKernelQueueProfiles(
    amdf_dxg_bridge_gpu_adapter_t* adapter) noexcept {
  const thunk_proxy::DeviceContext& device = *adapter->device;
  adapter->state_shadowing_by_cpfw = device.SupportStateShadowingByCpFw();

  TryMakeQueueProfile(device, device.ComputeEngine(), true,
                      &adapter->pm4_queue_profile);

  for (uint32_t i = 0; i < device.NumSdmaEngines(); ++i) {
    if (TryMakeQueueProfile(device, device.SdmaEngine(i), true,
                            &adapter->sdma_queue_profile)) {
      return;
    }
  }
  // A nonzero user-queue size means the driver selected persistent user-mode
  // SDMA rings, which this immutable command-stream API does not represent.
  if (device.UserQueueSize() != 0) {
    return;
  }
  for (uint32_t i = 0; i < device.NumSdmaEngines(); ++i) {
    if (TryMakeQueueProfile(device, device.SdmaEngine(i), false,
                            &adapter->sdma_queue_profile)) {
      return;
    }
  }
}

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
  try {
    return CreateNativeQueue(adapter, *create_info, out_queue, out_info,
                             out_native_status);
  } catch (const std::bad_alloc&) {
    return AMDF_DXG_BRIDGE_RESULT_RESOURCE_EXHAUSTED;
  } catch (...) {
    return AMDF_DXG_BRIDGE_RESULT_INTERNAL;
  }
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
  std::memset(queue->submission_private_data.data(), 0,
              queue->submission_private_data.byte_length());
  Wkmi::FillinSubmitPrivData(queue->submission_private_data.data(),
                             queue->handle, command_buffer_address,
                             command_buffer_byte_length, queue->handle != 0);

  if (queue->handle == 0) {
    D3DKMT_SUBMITCOMMAND submit = {
        .Commands = command_buffer_address,
        .CommandLength = static_cast<uint32_t>(command_buffer_byte_length),
        .BroadcastContextCount = 1,
    };
    submit.BroadcastContext[0] = queue->context;
    submit.pPrivateDriverData = queue->submission_private_data.data();
    submit.PrivateDriverDataSize =
        static_cast<uint32_t>(queue->submission_private_data.byte_length());
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
          static_cast<uint32_t>(queue->submission_private_data.byte_length()),
      .pPrivateDriverData = queue->submission_private_data.data(),
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
