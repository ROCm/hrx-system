// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/amdf.h"

#include <stddef.h>

#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/host_mapping.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/provider/extension.h"
#include "libamdf/src/user_queue.h"

static const amdf_api_t amdf_api_v1 = {
    .structure_size = sizeof(amdf_api_t),
    .abi_version = AMDF_ABI_VERSION_1,
    .instance_create = amdf_instance_create,
    .instance_destroy = amdf_instance_destroy,
    .endpoint_enumerate = amdf_endpoint_enumerate,
    .endpoint_open = amdf_provider_endpoint_open,
    .endpoint_query_info = amdf_endpoint_query_info,
    .endpoint_close = amdf_endpoint_close,
    .query_extension = amdf_extension_query,
    .endpoint_query_queue_family_info = amdf_endpoint_query_queue_family_info,
    .device_destroy = amdf_device_destroy,
    .instance_enumerate_memory_scopes = amdf_instance_enumerate_memory_scopes,
    .endpoint_enumerate_memory_scopes = amdf_endpoint_enumerate_memory_scopes,
    .device_enumerate_memory_scopes = amdf_device_enumerate_memory_scopes,
    .memory_scope_query_info = amdf_memory_scope_query_info,
    .memory_scope_query_profile = amdf_memory_scope_query_profile,
    .memory_scope_query_device_profile = amdf_memory_scope_query_device_profile,
    .memory_create = amdf_memory_create,
    .memory_import = amdf_memory_import,
    .memory_query_info = amdf_memory_query_info,
    .memory_query_access_info = amdf_memory_query_access_info,
    .memory_export = amdf_memory_export,
    .external_memory_release = amdf_external_memory_release,
    .memory_query_pair_info = amdf_memory_query_pair_info,
    .memory_map = amdf_memory_map,
    .host_mapping_query_info = amdf_host_mapping_query_info,
    .host_mapping_cache_control = amdf_host_mapping_cache_control,
    .host_mapping_destroy = amdf_host_mapping_destroy,
    .memory_destroy = amdf_memory_destroy,
    .kernel_queue_query_info = amdf_kernel_queue_query_info,
    .kernel_queue_query_status = amdf_kernel_queue_query_status,
    .kernel_queue_wait = amdf_kernel_queue_wait,
    .kernel_queue_destroy = amdf_kernel_queue_destroy,
    .user_queue_query_info = amdf_user_queue_query_info,
    .user_queue_map = amdf_user_queue_map,
    .user_queue_mapping_query_info = amdf_user_queue_mapping_query_info,
    .user_queue_mapping_destroy = amdf_user_queue_mapping_destroy,
    .user_queue_query_status = amdf_user_queue_query_status,
    .user_queue_wait_consumed = amdf_user_queue_wait_consumed,
    .user_queue_destroy = amdf_user_queue_destroy,
    .memory_query_address = amdf_memory_query_address,
};

amdf_status_t AMDF_CALL amdf_query_api(amdf_abi_version_t minimum_version,
                                       amdf_abi_version_t maximum_version,
                                       const amdf_api_t** out_api) {
  if (out_api == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (minimum_version > maximum_version) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (minimum_version > AMDF_ABI_VERSION_1 ||
      maximum_version < AMDF_ABI_VERSION_1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  *out_api = &amdf_api_v1;
  return AMDF_STATUS_OK;
}
