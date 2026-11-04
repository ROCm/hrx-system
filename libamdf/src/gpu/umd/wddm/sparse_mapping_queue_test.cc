// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/sparse_mapping_queue_test.h"

#include <array>
#include <memory>

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

#include "amdf/gpu.h"
#include "libamdf/src/gpu/umd/dxg/private_records.h"

struct amdf_sparse_mapping_test_queue_t {
  // Borrowed device owning the native context and synchronization object.
  D3DKMT_HANDLE device = 0;
  // Software-scheduled context retained across every mapping generation.
  D3DKMT_HANDLE context = 0;
  // Explicit monitored fence ordered after each submitted command stream.
  D3DKMT_HANDLE fence = 0;
  // Native-owned progress view, valid until the fence is destroyed.
  const volatile uint64_t* progress = nullptr;
};

namespace {

uint32_t QueryPrivateAdapter(void*, uint32_t adapter_handle, void* buffer,
                             uint32_t buffer_byte_length) {
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = static_cast<D3DKMT_HANDLE>(adapter_handle);
  query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
  query.pPrivateDriverData = buffer;
  query.PrivateDriverDataSize = buffer_byte_length;
  return static_cast<uint32_t>(D3DKMTQueryAdapterInfo(&query));
}

}  // namespace

int32_t amdf_sparse_mapping_test_queue_select(
    uint32_t adapter, amdf_sparse_mapping_test_scheduler_t* out_scheduler) {
  std::array<uint8_t, AMDF_DXG_PRIVATE_SINGLE_QUERY_SCRATCH_BYTE_LENGTH>
      scratch;
  const amdf_dxg_private_adapter_query_t query = {
      .adapter_handle = adapter,
      .physical_adapter_index = 0,
      .query_context = nullptr,
      .query = QueryPrivateAdapter,
      .scratch = scratch.data(),
  };
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = 0;
  const amdf_dxg_private_result_t result =
      amdf_dxg_private_query_single_adapter(&query, &info, &native_status);
  if (result == AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED) {
    return STATUS_NOT_SUPPORTED;
  }
  if (result == AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE) {
    return static_cast<int32_t>(native_status);
  }
  if (result != AMDF_DXG_PRIVATE_RESULT_SUCCESS) {
    return STATUS_INVALID_PARAMETER;
  }

  amdf_dxg_private_queue_profile_t pm4 = {};
  amdf_dxg_private_queue_profile_t sdma = {};
  amdf_dxg_private_initialize_queue_profiles(&info, &pm4, &sdma);
  if (!sdma.is_supported || sdma.uses_hardware_scheduling) {
    return STATUS_NOT_SUPPORTED;
  }

  amdf_sparse_mapping_test_scheduler_t scheduler = {
      .scheduler = sdma.scheduler_id,
      .node_ordinal = sdma.node_ordinal,
      .supports_firmware_state_shadowing =
          info.supports_firmware_state_shadowing,
  };
  if (info.gfx_ip_major >= 12) {
    scheduler.format_features |= AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM;
  } else if (info.gfx_ip_major >= 10) {
    scheduler.format_features |= AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE;
  }
  if (info.gfx_ip_major == 12 && info.gfx_ip_minor >= 5) {
    scheduler.format_features |= AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE;
  }
  *out_scheduler = scheduler;
  return STATUS_SUCCESS;
}

int32_t amdf_sparse_mapping_test_queue_create(
    uint32_t device, uint32_t physical_adapter_index,
    const amdf_sparse_mapping_test_scheduler_t* scheduler,
    amdf_sparse_mapping_test_queue_t** out_queue) {
  auto queue = std::make_unique<amdf_sparse_mapping_test_queue_t>();
  queue->device = device;
  std::array<uint8_t, AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH> context_data;
  amdf_dxg_private_initialize_context_record(
      scheduler->supports_firmware_state_shadowing, scheduler->scheduler, 0,
      context_data.data());
  D3DKMT_CREATECONTEXTVIRTUAL context = {};
  context.hDevice = device;
  context.NodeOrdinal = scheduler->node_ordinal;
  context.EngineAffinity = 1u << physical_adapter_index;
  context.ClientHint = D3DKMT_CLIENTHINT_OPENCL;
  context.pPrivateDriverData = context_data.data();
  context.PrivateDriverDataSize = static_cast<uint32_t>(context_data.size());
  NTSTATUS status = D3DKMTCreateContextVirtual(&context);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  queue->context = context.hContext;
  D3DKMT_CREATESYNCHRONIZATIONOBJECT2 fence = {};
  fence.hDevice = device;
  fence.Info.Type = D3DDDI_MONITORED_FENCE;
  fence.Info.MonitoredFence.EngineAffinity = 1u << physical_adapter_index;
  status = D3DKMTCreateSynchronizationObject2(&fence);
  if (status != STATUS_SUCCESS) {
    D3DKMT_DESTROYCONTEXT destroy = {queue->context};
    const NTSTATUS cleanup_status = D3DKMTDestroyContext(&destroy);
    return cleanup_status == STATUS_SUCCESS ? status : cleanup_status;
  }
  queue->fence = fence.hSyncObject;
  queue->progress = static_cast<const volatile uint64_t*>(
      fence.Info.MonitoredFence.FenceValueCPUVirtualAddress);
  *out_queue = queue.release();
  return STATUS_SUCCESS;
}

int32_t amdf_sparse_mapping_test_queue_execute(
    amdf_sparse_mapping_test_queue_t* queue, uint64_t address,
    uint32_t byte_length, uint64_t progress) {
  std::array<uint8_t, AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH>
      submission;
  amdf_dxg_private_initialize_submission_record(0, address, byte_length, false,
                                                submission.data());
  D3DKMT_SUBMITCOMMAND submit = {};
  submit.Commands = address;
  submit.CommandLength = byte_length;
  submit.BroadcastContextCount = 1;
  submit.BroadcastContext[0] = queue->context;
  submit.pPrivateDriverData = submission.data();
  submit.PrivateDriverDataSize = static_cast<uint32_t>(submission.size());
  NTSTATUS status = D3DKMTSubmitCommand(&submit);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU signal = {};
  signal.hContext = queue->context;
  signal.ObjectCount = 1;
  signal.ObjectHandleArray = &queue->fence;
  signal.MonitoredFenceValueArray = &progress;
  status = D3DKMTSignalSynchronizationObjectFromGpu(&signal);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {};
  wait.hDevice = queue->device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = &queue->fence;
  wait.FenceValueArray = &progress;
  status = D3DKMTWaitForSynchronizationObjectFromCpu(&wait);
  if (status != STATUS_SUCCESS) {
    return status;
  }
  MemoryBarrier();
  return *queue->progress == progress ? STATUS_SUCCESS : STATUS_DEVICE_REMOVED;
}

int32_t amdf_sparse_mapping_test_queue_destroy(
    amdf_sparse_mapping_test_queue_t* queue) {
  D3DKMT_DESTROYCONTEXT context = {queue->context};
  NTSTATUS first_status = D3DKMTDestroyContext(&context);
  D3DKMT_DESTROYSYNCHRONIZATIONOBJECT fence = {queue->fence};
  const NTSTATUS fence_status = D3DKMTDestroySynchronizationObject(&fence);
  if (first_status == STATUS_SUCCESS) {
    first_status = fence_status;
  }
  delete queue;
  return first_status;
}
