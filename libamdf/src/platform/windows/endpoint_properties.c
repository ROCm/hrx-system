// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/endpoint_properties.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "libamdf/src/pci.h"

#define AMDF_WINDOWS_DRIVER_DESCRIPTION_CAPACITY                \
  (sizeof(((D3DKMT_DRIVER_DESCRIPTION*)0)->DriverDescription) / \
   sizeof(((D3DKMT_DRIVER_DESCRIPTION*)0)->DriverDescription[0]))
#define AMDF_WINDOWS_UTF8_DESCRIPTION_CAPACITY \
  (AMDF_WINDOWS_DRIVER_DESCRIPTION_CAPACITY * 3u + 1u)

_Static_assert(AMDF_WINDOWS_DRIVER_DESCRIPTION_CAPACITY <= UINT32_MAX,
               "driver description length must fit in uint32_t");

static uint64_t amdf_windows_luid_encode(LUID luid) {
  return ((uint64_t)(uint32_t)luid.HighPart << 32) | luid.LowPart;
}

static LUID amdf_windows_luid_decode(uint64_t value) {
  LUID luid;
  luid.LowPart = (DWORD)value;
  luid.HighPart = (LONG)(uint32_t)(value >> 32);
  return luid;
}

static uint32_t amdf_windows_hash_word(uint32_t hash, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    hash ^= value & 0xFFu;
    hash *= 16777619u;
    value >>= 8;
  }
  return hash;
}

static uint32_t amdf_windows_hash_device_ids(
    const D3DKMT_DEVICE_IDS* device_ids) {
  uint32_t hash = 2166136261u;
  hash = amdf_windows_hash_word(hash, device_ids->VendorID);
  hash = amdf_windows_hash_word(hash, device_ids->DeviceID);
  hash = amdf_windows_hash_word(hash, device_ids->SubVendorID);
  hash = amdf_windows_hash_word(hash, device_ids->SubSystemID);
  hash = amdf_windows_hash_word(hash, device_ids->RevisionID);
  hash = amdf_windows_hash_word(hash, device_ids->BusType);
  return hash;
}

static amdf_endpoint_id_t amdf_windows_endpoint_id_encode(
    LUID adapter_luid, uint32_t physical_adapter_index,
    const D3DKMT_DEVICE_IDS* device_ids) {
  amdf_endpoint_id_t id;
  id.words[0] = amdf_windows_luid_encode(adapter_luid);
  id.words[1] = ((uint64_t)amdf_windows_hash_device_ids(device_ids) << 32) |
                physical_adapter_index;
  return id;
}

void amdf_windows_endpoint_id_decode(const amdf_endpoint_id_t* id,
                                     LUID* out_adapter_luid,
                                     uint32_t* out_physical_adapter_index) {
  *out_adapter_luid = amdf_windows_luid_decode(id->words[0]);
  *out_physical_adapter_index = (uint32_t)id->words[1];
}

static amdf_status_t amdf_windows_copy_endpoint_name(
    const D3DKMT_DRIVER_DESCRIPTION* description,
    char out_name[AMDF_ENDPOINT_NAME_CAPACITY]) {
  uint32_t wide_length = 0;
  while (wide_length < AMDF_WINDOWS_DRIVER_DESCRIPTION_CAPACITY &&
         description->DriverDescription[wide_length] != L'\0') {
    ++wide_length;
  }
  if (wide_length == 0) {
    out_name[0] = '\0';
    return AMDF_STATUS_OK;
  }

  char utf8_name[AMDF_WINDOWS_UTF8_DESCRIPTION_CAPACITY];
  const int utf8_length = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, description->DriverDescription,
      (int)wide_length, utf8_name, (int)(sizeof(utf8_name) - 1), NULL, NULL);
  if (utf8_length == 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  utf8_name[utf8_length] = '\0';

  uint32_t copy_length = (uint32_t)utf8_length;
  if (copy_length >= AMDF_ENDPOINT_NAME_CAPACITY) {
    copy_length = AMDF_ENDPOINT_NAME_CAPACITY - 1;
    while (copy_length != 0 &&
           ((uint8_t)utf8_name[copy_length] & 0xC0u) == 0x80u) {
      --copy_length;
    }
  }
  memcpy(out_name, utf8_name, copy_length);
  out_name[copy_length] = '\0';
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_windows_query_physical_adapter_count(
    const amdf_kmt_api_t* api, D3DKMT_HANDLE adapter, uint32_t* out_count) {
  D3DKMT_PHYSICAL_ADAPTER_COUNT physical_adapter_count = {0};
  const amdf_status_t status = amdf_kmt_query_adapter_info(
      api, adapter, KMTQAITYPE_PHYSICALADAPTERCOUNT, &physical_adapter_count,
      sizeof(physical_adapter_count));
  if (amdf_status_is_ok(status)) {
    *out_count = physical_adapter_count.Count;
  }
  return status;
}

amdf_status_t amdf_windows_query_endpoint_info(const amdf_kmt_api_t* api,
                                               D3DKMT_HANDLE adapter,
                                               LUID adapter_luid,
                                               uint32_t physical_adapter_index,
                                               amdf_endpoint_info_t* out_info) {
  D3DKMT_QUERY_DEVICE_IDS device_ids = {0};
  device_ids.PhysicalAdapterIndex = physical_adapter_index;
  amdf_status_t status = amdf_kmt_query_adapter_info(
      api, adapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &device_ids,
      sizeof(device_ids));

  D3DKMT_ADAPTERTYPE adapter_type = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_query_adapter_info(api, adapter, KMTQAITYPE_ADAPTERTYPE,
                                         &adapter_type, sizeof(adapter_type));
  }

  D3DKMT_DRIVER_DESCRIPTION description = {0};
  if (amdf_status_is_ok(status)) {
    status =
        amdf_kmt_query_adapter_info(api, adapter, KMTQAITYPE_DRIVER_DESCRIPTION,
                                    &description, sizeof(description));
  }

  amdf_endpoint_info_t endpoint_info = {0};
  if (amdf_status_is_ok(status)) {
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    endpoint_info.id = amdf_windows_endpoint_id_encode(
        adapter_luid, physical_adapter_index, &device_ids.DeviceIds);
    if (adapter_type.DisplaySupported) {
      endpoint_info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_DISPLAY_SUPPORTED;
    }
    if (adapter_type.RenderSupported) {
      endpoint_info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED;
    }
    if (adapter_type.ComputeOnly) {
      endpoint_info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY;
    }
    if (adapter_type.SoftwareDevice) {
      endpoint_info.type_flags |= AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE;
    }
    endpoint_info.pci.vendor_id = device_ids.DeviceIds.VendorID;
    endpoint_info.pci.device_id = device_ids.DeviceIds.DeviceID;
    endpoint_info.pci.subsystem_vendor_id = device_ids.DeviceIds.SubVendorID;
    endpoint_info.pci.subsystem_device_id = device_ids.DeviceIds.SubSystemID;
    endpoint_info.pci.revision_id = device_ids.DeviceIds.RevisionID;
    if (endpoint_info.pci.vendor_id == 0x1002u) {
      endpoint_info.engine_kind = AMDF_ENGINE_KIND_GPU;
    } else if (endpoint_info.pci.vendor_id == 0x1022u &&
               adapter_type.ComputeOnly) {
      endpoint_info.engine_kind = AMDF_ENGINE_KIND_XDNA;
    }
    status = amdf_windows_copy_endpoint_name(&description, endpoint_info.name);
  }
  if (amdf_status_is_ok(status)) {
    *out_info = endpoint_info;
  }
  return status;
}
