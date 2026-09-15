// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/endpoint_snapshot.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/pci.h"
#include "libamdf/src/platform/windows/endpoint_properties.h"

// Every returned handle gets one close attempt, including partial enumeration
// output. Native failure has no completion protocol or host-storage borrow.
static amdf_status_t amdf_windows_endpoint_snapshot_close(
    const amdf_kmt_api_t* api, uint32_t adapter_count,
    const D3DKMT_ADAPTERINFO* adapters) {
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < adapter_count; ++i) {
    if (adapters[i].hAdapter == 0) {
      continue;
    }
    D3DKMT_CLOSEADAPTER close_adapter = {0};
    close_adapter.hAdapter = adapters[i].hAdapter;
    const amdf_status_t close_status =
        amdf_kmt_make_status(api->close_adapter(&close_adapter));
    if (amdf_status_is_ok(status)) {
      status = close_status;
    }
  }
  return status;
}

static void amdf_windows_copy_endpoint_summary(
    const amdf_endpoint_info_t* info, amdf_endpoint_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->id = info->id;
  out_summary->engine_kind = info->engine_kind;
  out_summary->type_flags = info->type_flags;
  memcpy(out_summary->name, info->name, sizeof(out_summary->name));
}

amdf_status_t amdf_windows_endpoint_snapshot_enumerate(
    const amdf_kmt_api_t* api, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count,
    amdf_allocator_t host_allocator) {
  if ((size_t)capacity > SIZE_MAX / sizeof(*summaries)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  amdf_endpoint_summary_t* staged_summaries = NULL;
  if (capacity != 0) {
    const amdf_status_t allocation_status = amdf_calloc_array(
        host_allocator, (size_t)capacity, sizeof(*staged_summaries),
        amdf_alignof(amdf_endpoint_summary_t), (void**)&staged_summaries);
    if (!amdf_status_is_ok(allocation_status)) return allocation_status;
  }

  // Include compute/display-only application adapters. GPU-P partition
  // adapters remain hidden on the host; they are intended for guest use.
  D3DKMT_ENUMADAPTERS3 enumeration = {0};
  enumeration.Filter.IncludeComputeOnly = 1;
  enumeration.Filter.IncludeDisplayOnly = 1;
  amdf_status_t status =
      amdf_kmt_make_status(api->enumerate_adapters(&enumeration));
  if (!amdf_status_is_ok(status)) {
    amdf_free(host_allocator, staged_summaries);
    return status;
  }
  if (enumeration.NumAdapters == 0) {
    amdf_free(host_allocator, staged_summaries);
    *out_count = 0;
    return AMDF_STATUS_OK;
  }

  const uint32_t adapter_capacity = enumeration.NumAdapters;
  if ((size_t)adapter_capacity > SIZE_MAX / sizeof(D3DKMT_ADAPTERINFO)) {
    amdf_free(host_allocator, staged_summaries);
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  D3DKMT_ADAPTERINFO* adapters = NULL;
  const amdf_status_t allocation_status =
      amdf_calloc_array(host_allocator, adapter_capacity, sizeof(*adapters),
                        amdf_alignof(D3DKMT_ADAPTERINFO), (void**)&adapters);
  if (!amdf_status_is_ok(allocation_status)) {
    amdf_free(host_allocator, staged_summaries);
    return allocation_status;
  }
  enumeration.NumAdapters = adapter_capacity;
  enumeration.pAdapters = adapters;
  status = amdf_kmt_make_status(api->enumerate_adapters(&enumeration));
  if (amdf_status_is_ok(status) && enumeration.NumAdapters > adapter_capacity) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  uint32_t endpoint_count = 0;
  for (uint32_t adapter_index = 0;
       amdf_status_is_ok(status) && adapter_index < enumeration.NumAdapters;
       ++adapter_index) {
    uint32_t physical_adapter_count = 0;
    status = amdf_windows_query_physical_adapter_count(
        api, adapters[adapter_index].hAdapter, &physical_adapter_count);
    if (amdf_status_is_ok(status) && physical_adapter_count == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    for (uint32_t physical_adapter_index = 0;
         amdf_status_is_ok(status) &&
         physical_adapter_index < physical_adapter_count;
         ++physical_adapter_index) {
      amdf_endpoint_info_t info = {0};
      status = amdf_windows_query_endpoint_info(
          api, adapters[adapter_index].hAdapter,
          adapters[adapter_index].AdapterLuid, physical_adapter_index, &info);
      if (!amdf_status_is_ok(status) || !amdf_pci_is_amd(&info.pci)) {
        continue;
      }
      if (endpoint_count == UINT32_MAX) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
        break;
      }
      if (endpoint_count < capacity) {
        amdf_windows_copy_endpoint_summary(&info,
                                           &staged_summaries[endpoint_count]);
      }
      ++endpoint_count;
    }
  }

  const amdf_status_t close_status =
      amdf_windows_endpoint_snapshot_close(api, adapter_capacity, adapters);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  amdf_free(host_allocator, adapters);
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
  amdf_free(host_allocator, staged_summaries);
  return status;
}
