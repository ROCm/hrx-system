// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/telemetry.h"

#include "libamdf/src/device.h"

static amdf_status_t AMDF_CALL amdf_telemetry_device_enumerate_metrics(
    amdf_device_t* device, uint32_t capacity,
    amdf_telemetry_metric_info_t* metrics, uint32_t* out_count) {
  if (device == NULL || out_count == NULL ||
      (capacity != 0 && metrics == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (device->vtable->enumerate_telemetry_metrics == NULL) {
    *out_count = 0;
    return AMDF_STATUS_OK;
  }
  return device->vtable->enumerate_telemetry_metrics(device, capacity, metrics,
                                                     out_count);
}

static amdf_status_t AMDF_CALL amdf_telemetry_device_read(
    amdf_device_t* device, amdf_telemetry_source_id_t source_id,
    uint32_t metric_count, const amdf_telemetry_metric_id_t* metric_ids,
    amdf_telemetry_value_t* values, amdf_telemetry_sample_t* out_sample) {
  if (device == NULL || metric_count == 0 || metric_ids == NULL ||
      values == NULL || out_sample == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (device->vtable->read_telemetry == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  return device->vtable->read_telemetry(device, source_id, metric_count,
                                        metric_ids, values, out_sample);
}

static const amdf_telemetry_api_t amdf_telemetry_api_v1 = {
    .structure_size = sizeof(amdf_telemetry_api_t),
    .extension_version = AMDF_TELEMETRY_EXTENSION_VERSION_1,
    .device_enumerate_metrics = amdf_telemetry_device_enumerate_metrics,
    .device_read = amdf_telemetry_device_read,
};

amdf_status_t amdf_telemetry_extension_query(uint32_t minimum_version,
                                             uint32_t maximum_version,
                                             const void** out_api) {
  if (minimum_version > AMDF_TELEMETRY_EXTENSION_VERSION_1 ||
      maximum_version < AMDF_TELEMETRY_EXTENSION_VERSION_1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  *out_api = &amdf_telemetry_api_v1;
  return AMDF_STATUS_OK;
}
