// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/endpoint.h"

#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/pci.h"
#include "libamdf/src/platform/windows/endpoint_properties.h"
#include "libamdf/src/platform/windows/instance.h"

static amdf_status_t amdf_windows_close_endpoint_adapter(
    amdf_platform_endpoint_t* endpoint) {
  if (endpoint->adapter == 0) return AMDF_STATUS_OK;
  D3DKMT_CLOSEADAPTER close_adapter = {0};
  close_adapter.hAdapter = endpoint->adapter;
  const amdf_status_t status = amdf_kmt_make_status(
      endpoint->instance->kmt.close_adapter(&close_adapter));
  if (amdf_status_is_ok(status)) endpoint->adapter = 0;
  return status;
}

amdf_status_t amdf_platform_endpoint_open(
    amdf_platform_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_platform_endpoint_t** out_endpoint, amdf_endpoint_info_t* out_info) {
  if (!amdf_kmt_api_supports_endpoint_discovery(&instance->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_platform_endpoint_t* endpoint = NULL;
  amdf_status_t status =
      amdf_calloc(instance->host_allocator, sizeof(*endpoint),
                  amdf_alignof(amdf_platform_endpoint_t), (void**)&endpoint);
  if (!amdf_status_is_ok(status)) return status;
  endpoint->instance = instance;

  LUID adapter_luid;
  uint32_t physical_adapter_index = 0;
  amdf_windows_endpoint_id_decode(id, &adapter_luid, &physical_adapter_index);
  endpoint->physical_adapter_index = physical_adapter_index;
  endpoint->id = *id;
  D3DKMT_OPENADAPTERFROMLUID open_adapter = {0};
  open_adapter.AdapterLuid = adapter_luid;
  status =
      amdf_kmt_make_status(instance->kmt.open_adapter_from_luid(&open_adapter));
  endpoint->adapter = open_adapter.hAdapter;
  if (amdf_status_is_ok(status) && endpoint->adapter == 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  uint32_t physical_adapter_count = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_query_physical_adapter_count(
        &instance->kmt, endpoint->adapter, &physical_adapter_count);
  }
  if (amdf_status_is_ok(status) &&
      physical_adapter_index >= physical_adapter_count) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }

  amdf_endpoint_info_t endpoint_info = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_query_endpoint_info(
        &instance->kmt, endpoint->adapter, adapter_luid, physical_adapter_index,
        &endpoint_info);
  }
  if (amdf_status_is_ok(status) &&
      (!amdf_pci_is_amd(&endpoint_info.pci) ||
       !amdf_endpoint_id_is_equal(id, &endpoint_info.id))) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }

  if (amdf_status_is_ok(status)) {
    *out_info = endpoint_info;
    *out_endpoint = endpoint;
  } else {
    const amdf_status_t close_status =
        amdf_windows_close_endpoint_adapter(endpoint);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
    // Query handles have no accepted work borrowing this unpublished object.
    amdf_free(instance->host_allocator, endpoint);
  }
  return status;
}

amdf_queue_publication_modes_t
amdf_platform_endpoint_query_queue_publication_modes(
    const amdf_platform_endpoint_t* endpoint,
    amdf_queue_command_type_t command_type) {
  if ((command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 ||
       command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) &&
      amdf_kmt_api_supports_gpu_kernel_execution(&endpoint->instance->kmt)) {
    return AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  }
  if (command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
      amdf_kmt_api_supports_xdna_kernel_execution(&endpoint->instance->kmt)) {
    return AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  }
  return 0;
}

amdf_status_t amdf_platform_endpoint_close(amdf_platform_endpoint_t* endpoint) {
  const amdf_status_t status = amdf_windows_close_endpoint_adapter(endpoint);
  if (amdf_status_is_ok(status)) {
    amdf_free(endpoint->instance->host_allocator, endpoint);
  }
  return status;
}
