// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dxg/endpoint.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/pci_product_name.h"
#include "libamdf/src/platform/dxg/endpoint_identity.h"

static uint64_t amdf_linux_dxg_luid_encode(LUID luid) {
  return ((uint64_t)(uint32_t)luid.HighPart << 32) | luid.LowPart;
}

static LUID amdf_linux_dxg_luid_decode(uint64_t value) {
  return (LUID){
      .LowPart = (DWORD)value,
      .HighPart = (LONG)(uint32_t)(value >> 32),
  };
}

static amdf_status_t amdf_linux_dxg_query_adapter_info(
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE adapter,
    KMTQUERYADAPTERINFOTYPE type, void* data, uint32_t data_size) {
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = adapter,
      .Type = type,
      .pPrivateDriverData = data,
      .PrivateDriverDataSize = data_size,
  };
  return amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_query_adapter_info(kmt, &query));
}

static amdf_status_t amdf_linux_dxg_query_physical_adapter_count(
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE adapter,
    uint32_t* out_count) {
  D3DKMT_PHYSICAL_ADAPTER_COUNT physical_adapter_count = {0};
  const amdf_status_t status = amdf_linux_dxg_query_adapter_info(
      kmt, adapter, KMTQAITYPE_PHYSICALADAPTERCOUNT, &physical_adapter_count,
      sizeof(physical_adapter_count));
  if (amdf_status_is_ok(status)) {
    *out_count = physical_adapter_count.Count;
  }
  return status;
}

static amdf_status_t amdf_linux_dxg_query_endpoint_info(
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE adapter, LUID adapter_luid,
    uint32_t physical_adapter_index, amdf_endpoint_info_t* out_info) {
  D3DKMT_QUERY_DEVICE_IDS device_ids = {
      .PhysicalAdapterIndex = physical_adapter_index,
  };
  amdf_status_t status = amdf_linux_dxg_query_adapter_info(
      kmt, adapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &device_ids,
      sizeof(device_ids));

  D3DKMT_ADAPTERTYPE adapter_type = {0};
  if (amdf_status_is_ok(status)) {
    status =
        amdf_linux_dxg_query_adapter_info(kmt, adapter, KMTQAITYPE_ADAPTERTYPE,
                                          &adapter_type, sizeof(adapter_type));
  }

  amdf_endpoint_info_t info = {0};
  if (amdf_status_is_ok(status)) {
    const uint64_t encoded_luid = amdf_linux_dxg_luid_encode(adapter_luid);
    const amdf_dxg_device_ids_t native_device_ids = {
        .vendor_id = device_ids.DeviceIds.VendorID,
        .device_id = device_ids.DeviceIds.DeviceID,
        .subsystem_vendor_id = device_ids.DeviceIds.SubVendorID,
        .subsystem_device_id = device_ids.DeviceIds.SubSystemID,
        .revision_id = device_ids.DeviceIds.RevisionID,
        .bus_type = device_ids.DeviceIds.BusType,
    };
    info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    info.id = amdf_dxg_endpoint_id_encode(encoded_luid, physical_adapter_index,
                                          &native_device_ids);
    info.native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER;
    info.native_identity.value.windows_adapter.luid = encoded_luid;
    info.native_identity.value.windows_adapter.physical_adapter_index =
        physical_adapter_index;
    if (adapter_type.DisplaySupported) {
      info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_DISPLAY_SUPPORTED;
    }
    if (adapter_type.RenderSupported) {
      info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED;
    }
    if (adapter_type.ComputeOnly) {
      info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY;
    }
    if (adapter_type.SoftwareDevice) {
      info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE;
    }
    info.pci.vendor_id = device_ids.DeviceIds.VendorID;
    info.pci.device_id = device_ids.DeviceIds.DeviceID;
    info.pci.subsystem_vendor_id = device_ids.DeviceIds.SubVendorID;
    info.pci.subsystem_device_id = device_ids.DeviceIds.SubSystemID;
    info.pci.revision_id = device_ids.DeviceIds.RevisionID;
    if (info.pci.vendor_id == 0x1002u) {
      info.engine_kind = AMDF_ENGINE_KIND_GPU;
    } else if (info.pci.vendor_id == 0x1022u && adapter_type.ComputeOnly) {
      info.engine_kind = AMDF_ENGINE_KIND_XDNA;
    }
    const char* product_name = info.engine_kind == AMDF_ENGINE_KIND_GPU
                                   ? amdf_pci_product_name(&info.pci)
                                   : NULL;
    if (product_name) {
      memcpy(info.name, product_name, strlen(product_name) + 1);
    } else {
      const int name_length =
          snprintf(info.name, sizeof(info.name),
                   "AMD %s %04" PRIx32 ":%04" PRIx32 ":%02" PRIx32
                   " (DXG %016" PRIx64 "/%" PRIu32 ")",
                   info.engine_kind == AMDF_ENGINE_KIND_XDNA ? "XDNA" : "GPU",
                   info.pci.vendor_id, info.pci.device_id, info.pci.revision_id,
                   encoded_luid, physical_adapter_index);
      if (name_length < 0 || (size_t)name_length >= sizeof(info.name)) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_info = info;
  }
  return status;
}

static amdf_status_t amdf_linux_dxg_close_adapters(
    const amdf_linux_dxg_kmt_t* kmt, uint32_t adapter_count,
    const D3DKMT_ADAPTERINFO* adapters) {
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < adapter_count; ++i) {
    if (adapters[i].hAdapter == 0) {
      continue;
    }
    D3DKMT_CLOSEADAPTER close_adapter = {
        .hAdapter = adapters[i].hAdapter,
    };
    const amdf_status_t close_status = amdf_linux_dxg_kmt_make_status(
        amdf_linux_dxg_kmt_close_adapter(kmt, &close_adapter));
    if (amdf_status_is_ok(status)) {
      status = close_status;
    }
  }
  return status;
}

static void amdf_linux_dxg_copy_endpoint_summary(
    const amdf_endpoint_info_t* info, amdf_endpoint_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->id = info->id;
  out_summary->engine_kind = info->engine_kind;
  out_summary->type_flags = info->type_flags;
  memcpy(out_summary->name, info->name, sizeof(out_summary->name));
}

amdf_status_t amdf_linux_dxg_endpoint_enumerate(
    amdf_platform_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count) {
  if ((size_t)capacity > SIZE_MAX / sizeof(*summaries)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  amdf_endpoint_summary_t* staged_summaries = NULL;
  if (capacity != 0) {
    const amdf_status_t allocation_status = amdf_calloc_array(
        instance->host_allocator, capacity, sizeof(*staged_summaries),
        amdf_alignof(amdf_endpoint_summary_t), (void**)&staged_summaries);
    if (!amdf_status_is_ok(allocation_status)) {
      return allocation_status;
    }
  }

  D3DKMT_ENUMADAPTERS2 enumeration = {0};
  amdf_status_t status = amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_enumerate_adapters(&instance->kmt, &enumeration));
  const uint32_t adapter_capacity = enumeration.NumAdapters;
  D3DKMT_ADAPTERINFO* adapters = NULL;
  if (amdf_status_is_ok(status) && adapter_capacity != 0) {
    status = amdf_calloc_array(
        instance->host_allocator, adapter_capacity, sizeof(*adapters),
        amdf_alignof(D3DKMT_ADAPTERINFO), (void**)&adapters);
  }
  if (amdf_status_is_ok(status) && adapter_capacity != 0) {
    enumeration.pAdapters = adapters;
    status = amdf_linux_dxg_kmt_make_status(
        amdf_linux_dxg_kmt_enumerate_adapters(&instance->kmt, &enumeration));
    if (amdf_status_is_ok(status) &&
        enumeration.NumAdapters > adapter_capacity) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  uint32_t endpoint_count = 0;
  const uint32_t enumerated_adapter_count =
      adapters == NULL                             ? 0
      : enumeration.NumAdapters < adapter_capacity ? enumeration.NumAdapters
                                                   : adapter_capacity;
  for (uint32_t adapter_index = 0;
       amdf_status_is_ok(status) && adapter_index < enumerated_adapter_count;
       ++adapter_index) {
    uint32_t physical_adapter_count = 0;
    status = amdf_linux_dxg_query_physical_adapter_count(
        &instance->kmt, adapters[adapter_index].hAdapter,
        &physical_adapter_count);
    if (amdf_status_is_ok(status) && physical_adapter_count == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    for (uint32_t physical_adapter_index = 0;
         amdf_status_is_ok(status) &&
         physical_adapter_index < physical_adapter_count;
         ++physical_adapter_index) {
      amdf_endpoint_info_t info = {0};
      status = amdf_linux_dxg_query_endpoint_info(
          &instance->kmt, adapters[adapter_index].hAdapter,
          adapters[adapter_index].AdapterLuid, physical_adapter_index, &info);
      if (!amdf_status_is_ok(status) ||
          info.engine_kind != AMDF_ENGINE_KIND_GPU) {
        continue;
      }
      if (endpoint_count == UINT32_MAX) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
        break;
      }
      if (endpoint_count < capacity) {
        amdf_linux_dxg_copy_endpoint_summary(&info,
                                             &staged_summaries[endpoint_count]);
      }
      ++endpoint_count;
    }
  }

  const amdf_status_t close_status = amdf_linux_dxg_close_adapters(
      &instance->kmt, enumerated_adapter_count, adapters);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  amdf_free(instance->host_allocator, adapters);
  if (amdf_status_is_ok(status)) {
    if (capacity != 0) {
      const uint32_t copied_count =
          endpoint_count < capacity ? endpoint_count : capacity;
      memcpy(summaries, staged_summaries,
             (size_t)copied_count * sizeof(*summaries));
    }
    *out_count = endpoint_count;
    if (capacity != 0 && endpoint_count > capacity) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    }
  }
  amdf_free(instance->host_allocator, staged_summaries);
  return status;
}

static amdf_status_t amdf_linux_dxg_close_endpoint_adapter(
    amdf_platform_endpoint_t* endpoint) {
  if (endpoint->adapter == 0) {
    return AMDF_STATUS_OK;
  }
  D3DKMT_CLOSEADAPTER close_adapter = {
      .hAdapter = endpoint->adapter,
  };
  const amdf_status_t status =
      amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_close_adapter(
          &endpoint->instance->kmt, &close_adapter));
  if (amdf_status_is_ok(status)) {
    endpoint->adapter = 0;
  }
  return status;
}

amdf_status_t amdf_linux_dxg_endpoint_open(
    amdf_platform_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_platform_endpoint_t** out_endpoint, amdf_endpoint_info_t* out_info) {
  amdf_platform_endpoint_t* endpoint = NULL;
  amdf_status_t status =
      amdf_calloc(instance->host_allocator, sizeof(*endpoint),
                  amdf_alignof(amdf_platform_endpoint_t), (void**)&endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  endpoint->instance = instance;
  endpoint->host_numa_node = UINT32_MAX;
  endpoint->pci_address.type = AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS;
  endpoint->pci_address.structure_size = sizeof(endpoint->pci_address);

  uint64_t encoded_luid = 0;
  amdf_dxg_endpoint_id_decode(id, &encoded_luid,
                              &endpoint->physical_adapter_index);
  const LUID adapter_luid = amdf_linux_dxg_luid_decode(encoded_luid);
  D3DKMT_OPENADAPTERFROMLUID open_adapter = {
      .AdapterLuid = adapter_luid,
  };
  status = amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_open_adapter_from_luid(&instance->kmt, &open_adapter));
  endpoint->adapter = open_adapter.hAdapter;
  if (amdf_status_is_ok(status) && endpoint->adapter == 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  uint32_t physical_adapter_count = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_dxg_query_physical_adapter_count(
        &instance->kmt, endpoint->adapter, &physical_adapter_count);
  }
  if (amdf_status_is_ok(status) &&
      endpoint->physical_adapter_index >= physical_adapter_count) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_dxg_query_endpoint_info(
        &instance->kmt, endpoint->adapter, adapter_luid,
        endpoint->physical_adapter_index, &endpoint->info);
  }
  if (amdf_status_is_ok(status) &&
      (endpoint->info.engine_kind != AMDF_ENGINE_KIND_GPU ||
       !amdf_endpoint_id_is_equal(id, &endpoint->info.id))) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }

  if (amdf_status_is_ok(status)) {
    *out_info = endpoint->info;
    *out_endpoint = endpoint;
  } else {
    const amdf_status_t close_status =
        amdf_linux_dxg_close_endpoint_adapter(endpoint);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
    amdf_free(instance->host_allocator, endpoint);
  }
  return status;
}

amdf_status_t amdf_linux_dxg_endpoint_close(
    amdf_platform_endpoint_t* endpoint) {
  const amdf_status_t status = amdf_linux_dxg_close_endpoint_adapter(endpoint);
  if (amdf_status_is_ok(status)) {
    amdf_free(endpoint->instance->host_allocator, endpoint);
  }
  return status;
}
