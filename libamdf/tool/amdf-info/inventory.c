// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/inventory.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void amdf_info_endpoint_inspect(const amdf_info_library_t* library,
                                amdf_instance_t* instance,
                                amdf_info_endpoint_t* endpoint) {
  endpoint->status = library->api->endpoint_open(
      instance, &endpoint->summary.id, &endpoint->handle);
  if (!amdf_status_is_ok(endpoint->status)) {
    return;
  }
  endpoint->stage = AMDF_INFO_ENDPOINT_OPEN;
  endpoint->info = (amdf_endpoint_info_t){
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint->info),
  };
  endpoint->status =
      library->api->endpoint_query_info(endpoint->handle, &endpoint->info);
  if (!amdf_status_is_ok(endpoint->status)) {
    return;
  }
  endpoint->stage = AMDF_INFO_ENDPOINT_INFO;
  switch (endpoint->info.engine_kind) {
    case AMDF_ENGINE_KIND_GPU:
      endpoint->target.gpu = (amdf_gpu_endpoint_info_t){
          .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
          .structure_size = sizeof(endpoint->target.gpu),
      };
      endpoint->status = library->gpu
                             ? library->gpu->endpoint_query_info(
                                   endpoint->handle, &endpoint->target.gpu)
                             : library->gpu_status;
      break;
    case AMDF_ENGINE_KIND_XDNA:
      endpoint->target.xdna = (amdf_xdna_endpoint_info_t){
          .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
          .structure_size = sizeof(endpoint->target.xdna),
      };
      endpoint->status = library->xdna
                             ? library->xdna->endpoint_query_info(
                                   endpoint->handle, &endpoint->target.xdna)
                             : library->xdna_status;
      break;
    default:
      endpoint->status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      break;
  }
  if (amdf_status_is_ok(endpoint->status)) {
    endpoint->stage = AMDF_INFO_ENDPOINT_TARGET;
  }
}

const char* amdf_info_endpoint_scheme(const amdf_info_endpoint_t* endpoint) {
  const amdf_engine_kind_t engine = endpoint->stage >= AMDF_INFO_ENDPOINT_INFO
                                        ? endpoint->info.engine_kind
                                        : endpoint->summary.engine_kind;
  if (engine == AMDF_ENGINE_KIND_XDNA) {
    return "xdna";
  }
  if (engine != AMDF_ENGINE_KIND_GPU) {
    return "unknown";
  }
  if (endpoint->stage == AMDF_INFO_ENDPOINT_TARGET) {
    switch (amdf_gpu_endpoint_architecture(&endpoint->target.gpu)) {
      case AMDF_GPU_ARCHITECTURE_RDNA:
        return "rdna";
      case AMDF_GPU_ARCHITECTURE_CDNA:
        return "cdna";
      default:
        break;
    }
  }
  return "amdgpu";
}

void amdf_info_endpoint_target(const amdf_info_endpoint_t* endpoint,
                               char out_target[AMDF_XDNA_TARGET_ID_CAPACITY]) {
  out_target[0] = 0;
  if (endpoint->stage != AMDF_INFO_ENDPOINT_TARGET) {
    return;
  }
  if (endpoint->info.engine_kind == AMDF_ENGINE_KIND_GPU) {
    amdf_info_gpu_target(&endpoint->target.gpu, out_target);
  } else if (endpoint->info.engine_kind == AMDF_ENGINE_KIND_XDNA) {
    memcpy(out_target, endpoint->target.xdna.target_id,
           AMDF_XDNA_TARGET_ID_CAPACITY);
  }
}

void amdf_info_gpu_target(const amdf_gpu_endpoint_info_t* gpu,
                          char out_target[AMDF_XDNA_TARGET_ID_CAPACITY]) {
  snprintf(out_target, AMDF_XDNA_TARGET_ID_CAPACITY,
           "gfx%" PRIu32 "%" PRIu32 "%" PRIx32, gpu->gfx_ip.major,
           gpu->gfx_ip.minor, gpu->gfx_ip.stepping);
}

int amdf_info_endpoint_compare(const void* lhs, const void* rhs) {
  const amdf_endpoint_summary_t* a =
      &((const amdf_info_endpoint_t*)lhs)->summary;
  const amdf_endpoint_summary_t* b =
      &((const amdf_info_endpoint_t*)rhs)->summary;
  if (a->engine_kind != b->engine_kind) {
    return a->engine_kind < b->engine_kind ? -1 : 1;
  }
  for (size_t i = 0; i < 2; ++i) {
    if (a->id.words[i] != b->id.words[i]) {
      return a->id.words[i] < b->id.words[i] ? -1 : 1;
    }
  }
  return 0;
}

static uint32_t amdf_info_take_ordinal(uint32_t* ordinal) {
  return *ordinal == UINT32_MAX ? UINT32_MAX : (*ordinal)++;
}

amdf_status_t amdf_info_inventory_read(const amdf_info_library_t* library,
                                       amdf_instance_t* instance,
                                       uint32_t* out_count,
                                       amdf_info_endpoint_t** out_endpoints) {
  uint32_t count = 0;
  amdf_status_t status =
      library->api->endpoint_enumerate(instance, 0, NULL, &count);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_endpoint_summary_t* summaries = NULL;
  amdf_info_endpoint_t* endpoints = NULL;
  if (count) {
    summaries = (amdf_endpoint_summary_t*)calloc(count, sizeof(*summaries));
    endpoints = (amdf_info_endpoint_t*)calloc(count, sizeof(*endpoints));
    if (!summaries || !endpoints) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    } else {
      // Growth returns BUFFER_TOO_SMALL and fails this snapshot. No implicit
      // rediscovery can change an ordinal or extend an activation request.
      status =
          library->api->endpoint_enumerate(instance, count, summaries, &count);
    }
  }
  if (amdf_status_is_ok(status)) {
    uint32_t gpu_count = 0, rdna_count = 0, cdna_count = 0, xdna_count = 0;
    for (uint32_t i = 0; i < count; ++i) {
      endpoints[i].summary = summaries[i];
      endpoints[i].enumeration_ordinal = i;
      amdf_info_endpoint_inspect(library, instance, &endpoints[i]);
      const char* scheme = amdf_info_endpoint_scheme(&endpoints[i]);
      if (strcmp(scheme, "xdna") == 0) {
        endpoints[i].family_ordinal = amdf_info_take_ordinal(&xdna_count);
      } else if (strcmp(scheme, "unknown") == 0) {
        endpoints[i].family_ordinal = UINT32_MAX;
        const amdf_engine_kind_t engine =
            endpoints[i].stage >= AMDF_INFO_ENDPOINT_INFO
                ? endpoints[i].info.engine_kind
                : endpoints[i].summary.engine_kind;
        if (engine == AMDF_ENGINE_KIND_UNKNOWN) {
          gpu_count = rdna_count = cdna_count = xdna_count = UINT32_MAX;
        }
      } else {
        endpoints[i].family_ordinal = amdf_info_take_ordinal(&gpu_count);
        if (strcmp(scheme, "rdna") == 0) {
          endpoints[i].family_ordinal = amdf_info_take_ordinal(&rdna_count);
        } else if (strcmp(scheme, "cdna") == 0) {
          endpoints[i].family_ordinal = amdf_info_take_ordinal(&cdna_count);
        } else if (endpoints[i].stage != AMDF_INFO_ENDPOINT_TARGET) {
          // This GPU may belong to either architecture-specific view. The
          // broad GPU ordinal remains known, but later family ordinals do not.
          rdna_count = cdna_count = UINT32_MAX;
        }
      }
    }
    *out_count = count;
    *out_endpoints = endpoints;
  } else {
    free(endpoints);
  }
  free(summaries);
  return status;
}
