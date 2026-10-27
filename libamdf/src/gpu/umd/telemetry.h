// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_TELEMETRY_H_
#define AMDF_SRC_GPU_UMD_TELEMETRY_H_

#include "amdf/telemetry.h"
#include "libamdf/src/gpu/umd/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Implements the public cold enumeration protocol for an already-live device.
// The endpoint supplies native sensor identity; no discovery work is retained.
amdf_status_t amdf_gpu_umd_enumerate_telemetry_metrics(
    amdf_gpu_umd_device_t* device, amdf_platform_endpoint_t* endpoint,
    uint32_t capacity, amdf_telemetry_metric_info_t* metrics,
    uint32_t* out_count);

// Reads exactly one native source into transactional caller outputs. Common
// code validates pointers/counts; the provider validates its opaque IDs.
amdf_status_t amdf_gpu_umd_read_telemetry(
    amdf_gpu_umd_device_t* device, amdf_platform_endpoint_t* endpoint,
    amdf_telemetry_source_id_t source_id, uint32_t metric_count,
    const amdf_telemetry_metric_id_t* metric_ids,
    amdf_telemetry_value_t* values, amdf_telemetry_sample_t* out_sample);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_TELEMETRY_H_
