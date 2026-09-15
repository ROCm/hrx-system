// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "amdf/xdna.h"

static void amdf_example_report_status(const char* operation,
                                       amdf_status_t status) {
  fprintf(stderr, "%s failed: domain=%" PRIu32 " code=%" PRIu32 "\n", operation,
          amdf_status_domain(status), amdf_status_code(status));
}

static const char* amdf_example_engine_name(amdf_engine_kind_t engine_kind) {
  switch (engine_kind) {
    case AMDF_ENGINE_KIND_GPU:
      return "gpu";
    case AMDF_ENGINE_KIND_XDNA:
      return "xdna";
    default:
      return "unknown";
  }
}

static const char* amdf_example_xdna_architecture_name(
    amdf_xdna_architecture_t architecture) {
  switch (architecture) {
    case AMDF_XDNA_ARCHITECTURE_AIE2:
      return "AIE2";
    case AMDF_XDNA_ARCHITECTURE_AIE2P:
      return "AIE2P";
    case AMDF_XDNA_ARCHITECTURE_AIE4:
      return "AIE4";
    default:
      return "unknown";
  }
}

static const char* amdf_example_queue_command_name(
    amdf_queue_command_type_t command_type) {
  switch (command_type) {
    case AMDF_QUEUE_COMMAND_TYPE_GPU_PM4:
      return "gpu-pm4";
    case AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA:
      return "gpu-sdma";
    case AMDF_QUEUE_COMMAND_TYPE_GPU_AQL:
      return "gpu-aql";
    case AMDF_QUEUE_COMMAND_TYPE_XDNA:
      return "xdna";
    default:
      return "unknown";
  }
}

static const char* amdf_example_queue_publication_name(
    amdf_queue_publication_modes_t publication_modes) {
  switch (publication_modes) {
    case AMDF_QUEUE_PUBLICATION_MODE_USER:
      return "user";
    case AMDF_QUEUE_PUBLICATION_MODE_KERNEL:
      return "kernel";
    case AMDF_QUEUE_PUBLICATION_MODE_USER | AMDF_QUEUE_PUBLICATION_MODE_KERNEL:
      return "user|kernel";
    default:
      return "unknown";
  }
}

int main(void) {
  const amdf_api_t* api = NULL;
  amdf_status_t status =
      amdf_query_api(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api);
  if (!amdf_status_is_ok(status)) {
    amdf_example_report_status("amdf_query_api", status);
    return 1;
  }

  const void* extension_api = NULL;
  status =
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                           AMDF_GPU_EXTENSION_VERSION_LATEST, &extension_api);
  const amdf_gpu_api_t* gpu_api = NULL;
  if (amdf_status_is_ok(status)) {
    gpu_api = (const amdf_gpu_api_t*)extension_api;
  } else if (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API &&
             amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    status = AMDF_STATUS_OK;
  } else {
    amdf_example_report_status("query_extension(GPU)", status);
    return 1;
  }

  extension_api = NULL;
  status =
      api->query_extension(AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                           AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension_api);
  const amdf_xdna_api_t* xdna_api = NULL;
  if (amdf_status_is_ok(status)) {
    xdna_api = (const amdf_xdna_api_t*)extension_api;
  } else if (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API &&
             amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    status = AMDF_STATUS_OK;
  } else {
    amdf_example_report_status("query_extension(XDNA)", status);
    return 1;
  }

  const amdf_instance_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .structure_size = sizeof(create_info),
  };
  amdf_instance_t* instance = NULL;
  status = api->instance_create(&create_info, &instance);
  if (!amdf_status_is_ok(status)) {
    amdf_example_report_status("instance_create", status);
    return 1;
  }

  uint32_t endpoint_count = 0;
  status = api->endpoint_enumerate(instance, 0, NULL, &endpoint_count);
  if (!amdf_status_is_ok(status)) {
    amdf_example_report_status("endpoint_enumerate", status);
  }
  if (amdf_status_is_ok(status)) {
    printf("Found %" PRIu32 " AMD endpoint(s).\n", endpoint_count);
  }

  amdf_endpoint_summary_t* summaries = NULL;
  if (amdf_status_is_ok(status) && endpoint_count != 0) {
    summaries =
        (amdf_endpoint_summary_t*)calloc(endpoint_count, sizeof(*summaries));
    if (summaries == NULL) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      amdf_example_report_status("endpoint summary allocation", status);
    }
  }
  if (amdf_status_is_ok(status) && endpoint_count != 0) {
    status = api->endpoint_enumerate(instance, endpoint_count, summaries,
                                     &endpoint_count);
    if (!amdf_status_is_ok(status)) {
      amdf_example_report_status("endpoint_enumerate", status);
    }
  }

  for (uint32_t i = 0; amdf_status_is_ok(status) && i < endpoint_count; ++i) {
    amdf_endpoint_t* endpoint = NULL;
    status = api->endpoint_open(instance, &summaries[i].id, &endpoint);
    if (!amdf_status_is_ok(status)) {
      amdf_example_report_status("endpoint_open", status);
      break;
    }

    amdf_endpoint_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
        .structure_size = sizeof(info),
    };
    status = api->endpoint_query_info(endpoint, &info);
    if (amdf_status_is_ok(status)) {
      printf("[%" PRIu32 "] %s pci=%04" PRIx32 ":%04" PRIx32
             " revision=%02" PRIx32 " engine=%s flags=0x%08" PRIx32
             " queue-families=%" PRIu32 "\n",
             i, info.name, info.pci.vendor_id, info.pci.device_id,
             info.pci.revision_id, amdf_example_engine_name(info.engine_kind),
             info.type_flags, info.queue_family_count);
      for (uint32_t family_ordinal = 0;
           amdf_status_is_ok(status) &&
           family_ordinal < info.queue_family_count;
           ++family_ordinal) {
        amdf_queue_family_info_t family_info = {
            .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
            .structure_size = sizeof(family_info),
        };
        status = api->endpoint_query_queue_family_info(endpoint, family_ordinal,
                                                       &family_info);
        if (amdf_status_is_ok(status)) {
          printf("    queue-family[%" PRIu32 "] command=%s publication=%s\n",
                 family_info.ordinal,
                 amdf_example_queue_command_name(family_info.command_type),
                 amdf_example_queue_publication_name(
                     family_info.publication_modes));
        } else {
          amdf_example_report_status("endpoint_query_queue_family_info",
                                     status);
        }
      }
      if (amdf_status_is_ok(status) && gpu_api != NULL &&
          info.engine_kind == AMDF_ENGINE_KIND_GPU) {
        amdf_gpu_endpoint_info_t gpu_info = {
            .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
            .structure_size = sizeof(gpu_info),
        };
        status = gpu_api->endpoint_query_info(endpoint, &gpu_info);
        if (amdf_status_is_ok(status)) {
          printf("    gfx=%" PRIu32 ".%" PRIu32 ".%" PRIu32
                 " asic-revision=%" PRIu32 " wave=%" PRIu32
                 " compute-units=%" PRIu32 " waves/cu=%" PRIu32
                 " scratch-waves/cu=%" PRIu32 " lds=%" PRIu64 "\n",
                 gpu_info.gfx_ip.major, gpu_info.gfx_ip.minor,
                 gpu_info.gfx_ip.stepping, gpu_info.asic_revision,
                 gpu_info.compute.wavefront_size,
                 gpu_info.compute.compute_unit_count,
                 gpu_info.compute.maximum_wave_count_per_compute_unit,
                 gpu_info.compute.maximum_scratch_wave_count_per_compute_unit,
                 gpu_info.compute.local_data_share_byte_length);
          printf("    xcc=%" PRIu32 " shader-engines/xcc=%" PRIu32 "\n",
                 gpu_info.topology.xcc_count,
                 gpu_info.topology.shader_engine_count_per_xcc);
        } else {
          amdf_example_report_status("GPU endpoint_query_info", status);
        }
      }
      if (amdf_status_is_ok(status) && xdna_api != NULL &&
          info.engine_kind == AMDF_ENGINE_KIND_XDNA) {
        amdf_xdna_endpoint_info_t xdna_info = {
            .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
            .structure_size = sizeof(xdna_info),
        };
        status = xdna_api->endpoint_query_info(endpoint, &xdna_info);
        if (amdf_status_is_ok(status)) {
          printf("    target=%s architecture=%s column-origin=%" PRIu32
                 " array=%" PRIu32 "x%" PRIu32 " column-stride=%" PRIu64 "\n",
                 xdna_info.target_id,
                 amdf_example_xdna_architecture_name(xdna_info.architecture),
                 xdna_info.array.column_origin, xdna_info.array.column_count,
                 xdna_info.array.row_count, xdna_info.array.column_stride);
        } else {
          amdf_example_report_status("XDNA endpoint_query_info", status);
        }
      }
    } else {
      amdf_example_report_status("endpoint_query_info", status);
    }

    const amdf_status_t close_status = api->endpoint_close(endpoint);
    if (!amdf_status_is_ok(close_status)) {
      amdf_example_report_status("endpoint_close", close_status);
      status = close_status;
    }
  }

  free(summaries);
  const amdf_status_t destroy_status = api->instance_destroy(instance);
  if (!amdf_status_is_ok(destroy_status)) {
    amdf_example_report_status("instance_destroy", destroy_status);
    status = destroy_status;
  }
  return amdf_status_is_ok(status) ? 0 : 1;
}
