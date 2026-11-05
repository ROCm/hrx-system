// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/clock.h"

#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"

amdf_status_t amdf_gpu_linux_dxg_query_queue_clock_info(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_observation_clock_info_t* out_info) {
  return amdf_gpu_dxg_query_queue_clock_info(
      &device->logical_device.provider_adapter, device->adapter,
      device->physical_adapter_index, command_type, out_info);
}

amdf_status_t amdf_gpu_linux_dxg_query_clock_sample(
    amdf_gpu_umd_device_t* device, amdf_queue_command_type_t command_type,
    amdf_observation_clock_sample_t* out_sample) {
  return amdf_gpu_dxg_query_clock_sample(
      &device->logical_device.provider_adapter, device->adapter,
      device->physical_adapter_index, command_type, out_sample);
}
