// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/mapping_context.h"

#include "libamdf/src/allocator.h"

#define AMDF_DXG_STATUS_NOT_SUPPORTED UINT32_C(0xC00000BB)
#define AMDF_DXG_STATUS_NOT_IMPLEMENTED UINT32_C(0xC0000002)

static void amdf_gpu_dxg_write_mapping_operation(
    void* context, uint32_t ordinal,
    const amdf_gpu_dxg_mapping_operation_t* source) {
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations = context;
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operation = &operations[ordinal];
  if (source->allocation_handle == 0) {
    operation->OperationType = D3DDDI_UPDATEGPUVIRTUALADDRESS_UNMAP;
    operation->Unmap.BaseAddress = source->target_address;
    operation->Unmap.SizeInBytes = source->byte_length;
    operation->Unmap.Protection.NoAccess = 1;
  } else {
    operation->OperationType = D3DDDI_UPDATEGPUVIRTUALADDRESS_MAP_PROTECT;
    operation->MapProtect.BaseAddress = source->target_address;
    operation->MapProtect.SizeInBytes = source->byte_length;
    operation->MapProtect.hAllocation = source->allocation_handle;
    operation->MapProtect.AllocationOffsetInBytes = source->source_byte_offset;
    operation->MapProtect.Protection.Write =
        (source->access & AMDF_MEMORY_ACCESS_WRITE) != 0;
    operation->MapProtect.Protection.Execute =
        (source->access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
  }
}

amdf_status_t amdf_gpu_dxg_mapping_operations_create(
    const amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t update_count,
    const amdf_gpu_umd_mapping_update_t* updates, uint64_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION** out_operations) {
  if (operation_count > UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations = NULL;
  const amdf_status_t status = amdf_calloc_array(
      amdf_gpu_dxg_memory_reservation_host_allocator(reservation),
      operation_count, sizeof(*operations),
      amdf_alignof(D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION),
      (void**)&operations);
  if (amdf_status_is_ok(status)) {
    amdf_gpu_dxg_memory_reservation_visit_mapping_operations(
        reservation, update_count, updates,
        amdf_gpu_dxg_write_mapping_operation, operations);
    *out_operations = operations;
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_mapping_operations_submit(
    const amdf_dxg_kmt_t* kmt, uint32_t device_handle, uint32_t context_handle,
    uint32_t fence_handle, uint64_t predecessor_value, uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations) {
  D3DKMT_UPDATEGPUVIRTUALADDRESS update = {
      .hDevice = device_handle,
      .hContext = context_handle,
      .hFenceObject = fence_handle,
      .NumOperations = operation_count,
      .Operations = operations,
      .FenceValue = predecessor_value,
  };
  return AMDF_DXG_KMT_CALL(kmt, update_gpu_virtual_address, &update);
}

static amdf_status_t amdf_gpu_dxg_mapping_context_prepare(
    amdf_gpu_dxg_mapping_context_t* context, const amdf_dxg_kmt_t* kmt,
    uint32_t device_handle, uint32_t physical_adapter_index) {
  // The explicit engine mask selects the same physical page table as the
  // device's paging queue on a linked adapter.
  const uint32_t affinity = UINT32_C(1) << physical_adapter_index;
  if (context->handle == 0) {
    D3DKMT_CREATECONTEXTVIRTUAL create = {
        .hDevice = device_handle,
        .EngineAffinity = affinity,
        .Flags = {.SynchronizationOnly = 1},
        .ClientHint = D3DKMT_CLIENTHINT_OPENCL,
    };
    const amdf_status_t status =
        AMDF_DXG_KMT_CALL(kmt, create_context_virtual, &create);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    context->handle = create.hContext;
    if (context->handle == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create = {.hDevice = device_handle};
  create.Info.Type = D3DDDI_MONITORED_FENCE;
  create.Info.MonitoredFence.EngineAffinity = affinity;
  const amdf_status_t status =
      AMDF_DXG_KMT_CALL(kmt, create_synchronization_object, &create);
  if (amdf_status_is_ok(status)) {
    if (create.hSyncObject == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    context->latest.fence = create.hSyncObject;
    context->latest.progress =
        create.Info.MonitoredFence.FenceValueCPUVirtualAddress;
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_mapping_context_submit(
    amdf_gpu_dxg_mapping_context_t* context, const amdf_dxg_kmt_t* kmt,
    uint32_t device_handle, uint32_t physical_adapter_index,
    uint32_t operation_count,
    D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION* operations,
    bool* mutation_started, amdf_gpu_dxg_mapping_point_t* out_point) {
  if (context->latest.fence == 0) {
    if (context->unsupported) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    const amdf_status_t status = amdf_gpu_dxg_mapping_context_prepare(
        context, kmt, device_handle, physical_adapter_index);
    if (status == amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                                   AMDF_DXG_STATUS_NOT_SUPPORTED) ||
        status == amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                                   AMDF_DXG_STATUS_NOT_IMPLEMENTED)) {
      context->unsupported = true;
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  // An all-ones monitored fence denotes device removal.
  if (context->latest.value == UINT64_MAX - 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  *mutation_started = true;
  const amdf_status_t status = amdf_gpu_dxg_mapping_operations_submit(
      kmt, device_handle, context->handle, context->latest.fence,
      context->latest.value, operation_count, operations);
  if (amdf_status_is_ok(status)) {
    ++context->latest.value;
    *out_point = context->latest;
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_mapping_context_deinitialize(
    amdf_gpu_dxg_mapping_context_t* context, const amdf_dxg_kmt_t* kmt,
    const amdf_gpu_dxg_memory_device_t* device) {
  if (context->latest.value != 0) {
    const amdf_status_t status = device->native_api->wait_for_paging(
        device->native_context, device->device_handle, context->latest.fence,
        context->latest.progress, context->latest.value);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  context->latest.value = 0;
  if (context->handle != 0) {
    D3DKMT_DESTROYCONTEXT destroy = {.hContext = context->handle};
    const amdf_status_t status =
        AMDF_DXG_KMT_CALL(kmt, destroy_context, &destroy);
    if (amdf_dxg_kmt_destroy_consumed(status)) {
      context->handle = 0;
    }
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (context->latest.fence != 0) {
    D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {
        .hSyncObject = context->latest.fence,
    };
    const amdf_status_t status =
        AMDF_DXG_KMT_CALL(kmt, destroy_synchronization_object, &destroy);
    if (amdf_dxg_kmt_destroy_consumed(status)) {
      context->latest = (amdf_gpu_dxg_mapping_point_t){0};
    }
    return status;
  }
  return AMDF_STATUS_OK;
}
