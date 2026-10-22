// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/clock.h"

#include "libamdf/src/device.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/umd/clock.h"

amdf_status_t AMDF_CALL amdf_xdna_device_query_timer_clock_info(
    amdf_device_t* device, amdf_observation_clock_info_t* out_info) {
  if (device == NULL || out_info == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (!amdf_device_is_extension(device, AMDF_EXTENSION_XDNA)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return amdf_xdna_umd_query_timer_clock_info(amdf_xdna_device_get_umd(device),
                                              out_info);
}
