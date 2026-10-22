// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/clock.h"

#include "libamdf/src/device.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/clock.h"

static amdf_status_t amdf_gpu_clock_query_family(
    amdf_device_t* device, uint32_t queue_family_ordinal,
    amdf_queue_command_type_t* out_command_type) {
  if (device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!amdf_device_is_extension(device, AMDF_EXTENSION_GPU)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_queue_family_info_t family = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(family),
  };
  const amdf_status_t status = amdf_device_query_queue_family_info(
      device, queue_family_ordinal, &family);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *out_command_type = family.command_type;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_gpu_device_query_queue_clock_info(
    amdf_device_t* device, uint32_t queue_family_ordinal,
    amdf_observation_clock_info_t* out_info) {
  if (out_info == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_queue_command_type_t command_type;
  const amdf_status_t status =
      amdf_gpu_clock_query_family(device, queue_family_ordinal, &command_type);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_gpu_umd_query_queue_clock_info(amdf_gpu_device_get_umd(device),
                                             command_type, out_info);
}

amdf_status_t AMDF_CALL amdf_gpu_device_query_clock_sample(
    amdf_device_t* device, uint32_t queue_family_ordinal,
    amdf_observation_clock_sample_t* out_sample) {
  if (out_sample == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_queue_command_type_t command_type;
  const amdf_status_t status =
      amdf_gpu_clock_query_family(device, queue_family_ordinal, &command_type);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_gpu_umd_query_clock_sample(amdf_gpu_device_get_umd(device),
                                         command_type, out_sample);
}
