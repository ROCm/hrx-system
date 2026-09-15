// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/provider/extension.h"

#include <stddef.h>

#include "libamdf/src/endpoint.h"

#if defined(AMDF_HAVE_GPU)
#include "libamdf/src/gpu/extension.h"
#endif  // AMDF_HAVE_GPU

#if defined(AMDF_HAVE_XDNA)
#include "libamdf/src/xdna/endpoint_profile.h"
#include "libamdf/src/xdna/extension.h"
#endif  // AMDF_HAVE_XDNA

amdf_status_t AMDF_CALL amdf_provider_endpoint_open(
    amdf_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_endpoint_t** out_endpoint) {
  const amdf_status_t status = amdf_endpoint_open(instance, id, out_endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  amdf_extension_initialize_endpoint(*out_endpoint);

  amdf_queue_family_info_t queue_families[AMDF_ENDPOINT_QUEUE_FAMILY_CAPACITY] =
      {0};
  const uint32_t queue_family_count =
      amdf_extension_query_endpoint_queue_families(
          *out_endpoint, AMDF_ENDPOINT_QUEUE_FAMILY_CAPACITY, queue_families);
  amdf_endpoint_set_queue_families(*out_endpoint, queue_family_count,
                                   queue_families);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_extension_query(amdf_extension_id_t extension_id,
                                             uint32_t minimum_version,
                                             uint32_t maximum_version,
                                             const void** out_extension_api) {
  if (out_extension_api == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (minimum_version > maximum_version) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  switch (extension_id) {
#if defined(AMDF_HAVE_GPU)
    case AMDF_EXTENSION_GPU:
      return amdf_gpu_extension_query(minimum_version, maximum_version,
                                      out_extension_api);
#endif  // AMDF_HAVE_GPU
#if defined(AMDF_HAVE_XDNA)
    case AMDF_EXTENSION_XDNA:
      return amdf_xdna_extension_query(minimum_version, maximum_version,
                                       out_extension_api);
#endif  // AMDF_HAVE_XDNA
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
}

void amdf_extension_initialize_endpoint(amdf_endpoint_t* endpoint) {
  const amdf_endpoint_info_t* endpoint_info =
      amdf_endpoint_get_cached_info(endpoint);
  switch (endpoint_info->engine_kind) {
#if defined(AMDF_HAVE_GPU)
    case AMDF_ENGINE_KIND_GPU:
      amdf_gpu_extension_initialize_endpoint(endpoint);
      break;
#endif  // AMDF_HAVE_GPU
#if defined(AMDF_HAVE_XDNA)
    case AMDF_ENGINE_KIND_XDNA:
      amdf_xdna_extension_initialize_endpoint(endpoint);
      break;
#endif  // AMDF_HAVE_XDNA
    default:
      break;
  }
}

uint32_t amdf_extension_query_endpoint_queue_families(
    amdf_endpoint_t* endpoint, uint32_t capacity,
    amdf_queue_family_info_t* out_families) {
  const amdf_endpoint_info_t* endpoint_info =
      amdf_endpoint_get_cached_info(endpoint);
  switch (endpoint_info->engine_kind) {
#if defined(AMDF_HAVE_GPU)
    case AMDF_ENGINE_KIND_GPU: {
      const void* untyped_profile = NULL;
      if (!amdf_status_is_ok(amdf_endpoint_query_engine_profile(
              endpoint, AMDF_ENGINE_KIND_GPU, &untyped_profile))) {
        return 0;
      }
      return amdf_gpu_extension_query_endpoint_queue_families(
          (const amdf_gpu_endpoint_profile_t*)untyped_profile, capacity,
          out_families);
    }
#endif  // AMDF_HAVE_GPU
#if defined(AMDF_HAVE_XDNA)
    case AMDF_ENGINE_KIND_XDNA:
      return amdf_xdna_extension_query_endpoint_queue_families(
          amdf_xdna_endpoint_profile_select(endpoint_info),
          amdf_endpoint_get_platform(endpoint), capacity, out_families);
#endif  // AMDF_HAVE_XDNA
    default:
      return 0;
  }
}
