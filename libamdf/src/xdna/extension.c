// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/extension.h"

#include <stddef.h>

#include "amdf/xdna.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/xdna/context.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/endpoint_profile.h"
#include "libamdf/src/xdna/kernel_queue.h"
#include "libamdf/src/xdna/umd/device.h"
#include "libamdf/src/xdna/umd/memory_profile.h"

static amdf_status_t amdf_xdna_endpoint_query_memory_profile(
    amdf_endpoint_t* endpoint, uint32_t profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  const amdf_xdna_endpoint_profile_t* target =
      amdf_xdna_endpoint_profile_select(
          amdf_endpoint_get_cached_info(endpoint));
  if (target == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return amdf_xdna_umd_query_endpoint_memory_profile(
      amdf_endpoint_get_platform(endpoint), target, profile_ordinal,
      out_profile);
}

void amdf_xdna_extension_initialize_endpoint(amdf_endpoint_t* endpoint) {
  amdf_endpoint_set_memory_profile_query(
      endpoint, amdf_xdna_endpoint_query_memory_profile);
}

static amdf_status_t AMDF_CALL amdf_xdna_endpoint_query_info(
    amdf_endpoint_t* endpoint, amdf_xdna_endpoint_info_t* out_info) {
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
      (uint32_t)sizeof(amdf_xdna_endpoint_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const amdf_xdna_endpoint_profile_t* profile =
      amdf_xdna_endpoint_profile_select(
          amdf_endpoint_get_cached_info(endpoint));
  if (profile == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = *amdf_xdna_endpoint_profile_get_info(profile);
  const amdf_xdna_umd_context_capabilities_t context_capabilities =
      amdf_xdna_umd_query_context_capabilities(profile);
  out_info->context.scheduling_modes &= context_capabilities.scheduling_modes;
  out_info->context.placement_modes = context_capabilities.placement_modes;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

static const amdf_xdna_api_t amdf_xdna_api_v1 = {
    .structure_size = sizeof(amdf_xdna_api_t),
    .extension_version = AMDF_XDNA_EXTENSION_VERSION_1,
    .endpoint_query_info = amdf_xdna_endpoint_query_info,
    .device_create = amdf_xdna_device_create,
    .device_query_info = amdf_xdna_device_query_info,
    .kernel_queue_create = amdf_xdna_kernel_queue_create,
    .kernel_queue_submit = amdf_xdna_kernel_queue_submit,
    .context_create = amdf_xdna_context_create,
    .context_query_info = amdf_xdna_context_query_info,
    .context_query_placement_info = amdf_xdna_context_query_placement_info,
    .context_enumerate_memory_scopes =
        amdf_xdna_context_enumerate_memory_scopes,
    .context_destroy = amdf_xdna_context_destroy,
};

uint32_t amdf_xdna_extension_query_endpoint_queue_families(
    const amdf_xdna_endpoint_profile_t* profile,
    const amdf_platform_endpoint_t* platform_endpoint, uint32_t capacity,
    amdf_queue_family_info_t* out_families) {
  if (profile == NULL || profile->info->instruction.maximum_byte_length == 0 ||
      capacity == 0) {
    return 0;
  }
  const amdf_queue_publication_modes_t publication_modes =
      amdf_platform_endpoint_query_queue_publication_modes(
          platform_endpoint, AMDF_QUEUE_COMMAND_TYPE_XDNA);
  if (publication_modes == 0) {
    return 0;
  }
  out_families[0].type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
  out_families[0].structure_size = sizeof(out_families[0]);
  out_families[0].ordinal = 0;
  out_families[0].command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA;
  out_families[0].publication_modes = publication_modes;
  out_families[0].format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1;
  out_families[0].roles = AMDF_QUEUE_ROLE_COMPUTE;
  return 1;
}

amdf_status_t amdf_xdna_extension_query(uint32_t minimum_version,
                                        uint32_t maximum_version,
                                        const void** out_extension_api) {
  if (minimum_version > AMDF_XDNA_EXTENSION_VERSION_1 ||
      maximum_version < AMDF_XDNA_EXTENSION_VERSION_1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  *out_extension_api = &amdf_xdna_api_v1;
  return AMDF_STATUS_OK;
}
