// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/extension.h"

#include <stddef.h>

#include "amdf/gpu.h"
#include "libamdf/src/gpu/clock.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/event.h"
#include "libamdf/src/gpu/kernel_queue.h"
#include "libamdf/src/gpu/umd/endpoint_profile.h"
#include "libamdf/src/gpu/user_queue.h"
#include "libamdf/src/structure.h"

static amdf_status_t AMDF_CALL amdf_gpu_endpoint_query_info(
    amdf_endpoint_t* endpoint, amdf_gpu_endpoint_info_t* out_info) {
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t validation_status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      (uint32_t)sizeof(amdf_gpu_endpoint_info_t));
  if (!amdf_status_is_ok(validation_status)) {
    return validation_status;
  }

  const void* untyped_profile = NULL;
  const amdf_status_t profile_status = amdf_endpoint_query_engine_profile(
      endpoint, AMDF_ENGINE_KIND_GPU, &untyped_profile);
  if (!amdf_status_is_ok(profile_status)) {
    return profile_status;
  }
  const amdf_gpu_endpoint_profile_t* profile =
      (const amdf_gpu_endpoint_profile_t*)untyped_profile;

  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = *amdf_gpu_endpoint_profile_get_info(profile);
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

static const amdf_gpu_api_t amdf_gpu_api_v1 = {
    .structure_size = offsetof(amdf_gpu_api_t, device_query_queue_clock_info),
    .extension_version = AMDF_GPU_EXTENSION_VERSION_1,
    .endpoint_query_info = amdf_gpu_endpoint_query_info,
    .device_create = amdf_gpu_device_create,
    .device_query_info = amdf_gpu_device_query_info,
    .kernel_queue_create = amdf_gpu_kernel_queue_create,
    .kernel_queue_submit = amdf_gpu_kernel_queue_submit,
    .user_queue_create = amdf_gpu_user_queue_create,
};

static const amdf_gpu_api_t amdf_gpu_api_v2 = {
    .structure_size = offsetof(amdf_gpu_api_t, event_create),
    .extension_version = AMDF_GPU_EXTENSION_VERSION_2,
    .endpoint_query_info = amdf_gpu_endpoint_query_info,
    .device_create = amdf_gpu_device_create,
    .device_query_info = amdf_gpu_device_query_info,
    .kernel_queue_create = amdf_gpu_kernel_queue_create,
    .kernel_queue_submit = amdf_gpu_kernel_queue_submit,
    .user_queue_create = amdf_gpu_user_queue_create,
    .device_query_queue_clock_info = amdf_gpu_device_query_queue_clock_info,
    .device_query_clock_sample = amdf_gpu_device_query_clock_sample,
};

static const amdf_gpu_api_t amdf_gpu_api_v3 = {
    .structure_size = offsetof(amdf_gpu_api_t, user_queue_query_scratch),
    .extension_version = AMDF_GPU_EXTENSION_VERSION_3,
    .endpoint_query_info = amdf_gpu_endpoint_query_info,
    .device_create = amdf_gpu_device_create,
    .device_query_info = amdf_gpu_device_query_info,
    .kernel_queue_create = amdf_gpu_kernel_queue_create,
    .kernel_queue_submit = amdf_gpu_kernel_queue_submit,
    .user_queue_create = amdf_gpu_user_queue_create,
    .device_query_queue_clock_info = amdf_gpu_device_query_queue_clock_info,
    .device_query_clock_sample = amdf_gpu_device_query_clock_sample,
    .event_create = amdf_gpu_event_create,
    .event_query_info = amdf_gpu_event_query_info,
    .event_signal = amdf_gpu_event_signal,
    .event_wait = amdf_gpu_event_wait,
    .event_destroy = amdf_gpu_event_destroy,
};

static const amdf_gpu_api_t amdf_gpu_api_v4 = {
    .structure_size = sizeof(amdf_gpu_api_t),
    .extension_version = AMDF_GPU_EXTENSION_VERSION_4,
    .endpoint_query_info = amdf_gpu_endpoint_query_info,
    .device_create = amdf_gpu_device_create,
    .device_query_info = amdf_gpu_device_query_info,
    .kernel_queue_create = amdf_gpu_kernel_queue_create,
    .kernel_queue_submit = amdf_gpu_kernel_queue_submit,
    .user_queue_create = amdf_gpu_user_queue_create,
    .device_query_queue_clock_info = amdf_gpu_device_query_queue_clock_info,
    .device_query_clock_sample = amdf_gpu_device_query_clock_sample,
    .event_create = amdf_gpu_event_create,
    .event_query_info = amdf_gpu_event_query_info,
    .event_signal = amdf_gpu_event_signal,
    .event_wait = amdf_gpu_event_wait,
    .event_destroy = amdf_gpu_event_destroy,
    .user_queue_query_scratch = amdf_gpu_user_queue_query_scratch,
    .user_queue_resume_scratch = amdf_gpu_user_queue_resume_scratch,
};

void amdf_gpu_extension_initialize_endpoint(amdf_endpoint_t* endpoint) {
  const amdf_endpoint_info_t* endpoint_info =
      amdf_endpoint_get_cached_info(endpoint);
  if (endpoint_info->engine_kind != AMDF_ENGINE_KIND_GPU) {
    return;
  }

  amdf_gpu_endpoint_profile_t* profile = NULL;
  const amdf_status_t status = amdf_gpu_umd_create_endpoint_profile(
      amdf_endpoint_get_platform(endpoint),
      amdf_endpoint_host_allocator(endpoint), &profile);
  if (!amdf_status_is_ok(status)) {
    amdf_endpoint_store_engine_profile_error(endpoint, status);
  } else {
    amdf_endpoint_store_engine_profile(endpoint, profile);
  }
}

uint32_t amdf_gpu_extension_query_endpoint_queue_families(
    const amdf_gpu_endpoint_profile_t* profile, uint32_t capacity,
    amdf_queue_family_info_t* out_families) {
  if (profile == NULL || capacity == 0) {
    return 0;
  }
  const uint32_t family_count = profile->queue_family_count < capacity
                                    ? profile->queue_family_count
                                    : capacity;
  for (uint32_t i = 0; i < family_count; ++i) {
    out_families[i] = profile->queue_families[i];
  }
  return family_count;
}

amdf_status_t amdf_gpu_extension_query(uint32_t minimum_version,
                                       uint32_t maximum_version,
                                       const void** out_extension_api) {
  if (minimum_version <= AMDF_GPU_EXTENSION_VERSION_4 &&
      maximum_version >= AMDF_GPU_EXTENSION_VERSION_4) {
    *out_extension_api = &amdf_gpu_api_v4;
    return AMDF_STATUS_OK;
  }
  if (minimum_version <= AMDF_GPU_EXTENSION_VERSION_3 &&
      maximum_version >= AMDF_GPU_EXTENSION_VERSION_3) {
    *out_extension_api = &amdf_gpu_api_v3;
    return AMDF_STATUS_OK;
  }
  if (minimum_version <= AMDF_GPU_EXTENSION_VERSION_2 &&
      maximum_version >= AMDF_GPU_EXTENSION_VERSION_2) {
    *out_extension_api = &amdf_gpu_api_v2;
    return AMDF_STATUS_OK;
  }
  if (minimum_version <= AMDF_GPU_EXTENSION_VERSION_1 &&
      maximum_version >= AMDF_GPU_EXTENSION_VERSION_1) {
    *out_extension_api = &amdf_gpu_api_v1;
    return AMDF_STATUS_OK;
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
}
