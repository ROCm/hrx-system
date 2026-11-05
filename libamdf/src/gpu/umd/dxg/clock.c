// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/clock.h"

#include "libamdf/src/platform/observation.h"

static amdf_dxg_provider_gpu_queue_command_type_t
amdf_gpu_dxg_clock_command_type(amdf_queue_command_type_t command_type) {
  return command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
             ? AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4
             : AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA;
}

amdf_status_t amdf_gpu_dxg_query_queue_clock_info(
    const amdf_gpu_dxg_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index, amdf_queue_command_type_t command_type,
    amdf_observation_clock_info_t* out_info) {
  amdf_dxg_provider_gpu_clock_sample_t sample = {0};
  const amdf_status_t status = amdf_gpu_dxg_adapter_query_clock_sample(
      adapter, adapter_handle, physical_adapter_index,
      amdf_gpu_dxg_clock_command_type(command_type), &sample);
  if (amdf_status_is_ok(status)) {
    *out_info =
        (amdf_observation_clock_info_t){.frequency_hz = sample.frequency_hz};
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_query_clock_sample(
    const amdf_gpu_dxg_adapter_t* adapter, uint32_t adapter_handle,
    uint32_t physical_adapter_index, amdf_queue_command_type_t command_type,
    amdf_observation_clock_sample_t* out_sample) {
  amdf_observation_clock_sample_t sample = {0};
  amdf_status_t status =
      amdf_platform_observation_query_host_clock(&sample.host.before);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_dxg_provider_gpu_clock_sample_t native = {0};
  status = amdf_gpu_dxg_adapter_query_clock_sample(
      adapter, adapter_handle, physical_adapter_index,
      amdf_gpu_dxg_clock_command_type(command_type), &native);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_platform_observation_query_host_clock(&sample.host.after);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  sample.clock.frequency_hz = native.frequency_hz;
  sample.timestamp = native.timestamp;
  sample.host.timestamp = native.host_timestamp;
  *out_sample = sample;
  return AMDF_STATUS_OK;
}
