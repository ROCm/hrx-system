// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/memory.h"

#include <stdlib.h>

void amdf_info_query_memory_pools(const amdf_api_t* api,
                                  amdf_instance_t* instance,
                                  amdf_device_t* device,
                                  amdf_info_output_t* output,
                                  amdf_info_result_t* result) {
  const char* operation = device ? "device_enumerate_memory_scopes"
                                 : "instance_enumerate_memory_scopes";
  const amdf_info_query_requirement_t requirement =
      device ? AMDF_INFO_QUERY_REQUIRED : AMDF_INFO_QUERY_OPTIONAL;
  uint32_t count = 0;
  amdf_status_t status =
      device ? api->device_enumerate_memory_scopes(device, 0, NULL, &count)
             : api->instance_enumerate_memory_scopes(instance, 0, NULL, &count);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    status = AMDF_STATUS_OK;
  }
  amdf_memory_scope_t** scopes = NULL;
  if (amdf_status_is_ok(status) && count) {
    scopes = (amdf_memory_scope_t**)calloc(count, sizeof(*scopes));
    if (scopes) {
      status = device ? api->device_enumerate_memory_scopes(device, count,
                                                            scopes, &count)
                      : api->instance_enumerate_memory_scopes(instance, count,
                                                              scopes, &count);
    } else {
      status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
  }
  if (amdf_info_report_query(output, result, "memory_pools", operation, status,
                             requirement)) {
    amdf_info_report_pools_begin(output, device != NULL);
    for (uint32_t i = 0; !output->failed && i < count; ++i) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      const amdf_status_t query_status =
          api->memory_scope_query_info(scopes[i], &info);
      if (amdf_info_report_query(output, result, NULL,
                                 "memory_scope_query_info", query_status,
                                 requirement)) {
        amdf_info_report_pool(output, i, &info);
      }
    }
    amdf_info_report_pools_end(output);
  }
  free(scopes);
}
