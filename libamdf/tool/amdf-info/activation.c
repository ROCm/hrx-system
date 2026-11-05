// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/activation.h"

static amdf_status_t amdf_info_device_create(
    const amdf_info_library_t* library, const amdf_info_endpoint_t* endpoint,
    amdf_device_t** out_device) {
  if (endpoint->info.engine_kind == AMDF_ENGINE_KIND_GPU) {
    const amdf_gpu_device_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    return library->gpu->device_create(endpoint->handle, &create_info,
                                       out_device);
  }
  const amdf_xdna_device_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create_info),
  };
  return library->xdna->device_create(endpoint->handle, &create_info,
                                      out_device);
}

static void amdf_info_device_query(const amdf_info_library_t* library,
                                   const amdf_info_endpoint_t* endpoint,
                                   amdf_device_t* device,
                                   amdf_info_output_t* output,
                                   amdf_info_result_t* result) {
  if (endpoint->info.engine_kind == AMDF_ENGINE_KIND_GPU) {
    amdf_gpu_device_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
        .structure_size = sizeof(info),
    };
    const amdf_status_t status = library->gpu->device_query_info(device, &info);
    if (amdf_info_report_query(output, result, "device",
                               "gpu.device_query_info", status,
                               AMDF_INFO_QUERY_REQUIRED)) {
      amdf_info_report_gpu_device(output, &info);
    }
  } else {
    amdf_xdna_device_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
        .structure_size = sizeof(info),
    };
    const amdf_status_t status =
        library->xdna->device_query_info(device, &info);
    if (amdf_info_report_query(output, result, "device",
                               "xdna.device_query_info", status,
                               AMDF_INFO_QUERY_REQUIRED)) {
      amdf_info_report_xdna_device(output, &info);
    }
  }
}

void amdf_info_activate(const amdf_info_library_t* library,
                        amdf_info_endpoint_t* endpoint,
                        amdf_info_output_t* output,
                        amdf_info_result_t* result) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "activation", '{');
  }
  amdf_device_t* device = NULL;
  const amdf_status_t create_status =
      amdf_info_device_create(library, endpoint, &device);
  if (amdf_info_report_query(output, result, "create", "device_create",
                             create_status, AMDF_INFO_QUERY_REQUIRED)) {
    amdf_info_device_query(library, endpoint, device, output, result);
    if (output->format == AMDF_INFO_FORMAT_JSON) {
      amdf_info_json_begin(output, "queue_placements", '[');
    }
    for (uint32_t i = 0;
         !output->failed && i < endpoint->info.queue_family_count; ++i) {
      amdf_queue_family_placement_info_t placement = {
          .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_PLACEMENT_INFO,
          .structure_size = sizeof(placement),
      };
      const amdf_status_t status =
          library->api->device_query_queue_family_placement_info(device, i,
                                                                 &placement);
      if (amdf_info_report_query(output, result, NULL,
                                 "device_query_queue_family_placement_info",
                                 status, AMDF_INFO_QUERY_REQUIRED)) {
        amdf_info_report_placement(output, &placement);
      }
    }
    if (output->format == AMDF_INFO_FORMAT_JSON) {
      amdf_info_json_end(output, ']');
    }
    const amdf_status_t destroy_status = library->api->device_destroy(device);
    endpoint->retain_on_exit =
        !amdf_info_report_query(output, result, "destroy", "device_destroy",
                                destroy_status, AMDF_INFO_QUERY_REQUIRED);
    if (output->format == AMDF_INFO_FORMAT_JSON) {
      amdf_info_json_bool(output, "device_destroyed",
                          !endpoint->retain_on_exit);
    }
  }
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, '}');
  }
}
