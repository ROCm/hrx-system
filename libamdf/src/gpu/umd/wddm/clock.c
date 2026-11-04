// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/clock.h"

#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/platform/observation.h"

amdf_status_t amdf_gpu_umd_query_queue_clock_info(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_observation_clock_info_t* out_info) {
  const amdf_dxg_bridge_gpu_queue_command_type_t native_type =
      command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
          ? AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4
          : AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA;
  amdf_dxg_bridge_gpu_clock_sample_t sample = {0};
  const amdf_status_t status = amdf_gpu_dxg_adapter_query_clock_sample(
      &device->bridge_adapter, device->adapter, device->physical_adapter_index,
      native_type, &sample);
  if (amdf_status_is_ok(status)) {
    *out_info =
        (amdf_observation_clock_info_t){.frequency_hz = sample.frequency_hz};
  }
  return status;
}

amdf_status_t amdf_gpu_umd_query_clock_sample(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_observation_clock_sample_t* out_sample) {
  const amdf_dxg_bridge_gpu_queue_command_type_t native_type =
      command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
          ? AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4
          : AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA;
  amdf_observation_clock_sample_t sample = {0};
  amdf_status_t status =
      amdf_platform_observation_query_host_clock(&sample.host.before);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_dxg_bridge_gpu_clock_sample_t native = {0};
  status = amdf_gpu_dxg_adapter_query_clock_sample(
      &device->bridge_adapter, device->adapter, device->physical_adapter_index,
      native_type, &native);
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
