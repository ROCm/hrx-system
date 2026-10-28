// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/telemetry.h"

#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/platform/observation.h"

#define AMDF_GPU_WDDM_PMLOG_SOURCE UINT64_C(0x0200000000000001)

typedef struct amdf_gpu_wddm_sensor_t {
  // Native CWDDEPM/PMLog sensor ID, independent of enumeration order.
  uint32_t id;
  // Stable SDK sensor name used as native provenance.
  const char* name;
  // Portable measurement meaning.
  amdf_telemetry_metric_kind_t kind;
  // Physical region measured by the sensor.
  amdf_telemetry_domain_kind_t domain;
  // Physical unit after rational scaling.
  amdf_observation_unit_t unit;
  // Exact conversion numerator.
  uint32_t numerator;
  // Exact conversion denominator.
  uint32_t denominator;
} amdf_gpu_wddm_sensor_t;

// Units are specified by the public ADL PMLog sensor definitions. WKMI
// transports these IDs and values without conversion. Sensors with no
// established measurement/scale are not promoted from a suggestive name.
static const amdf_gpu_wddm_sensor_t amdf_gpu_wddm_sensors[] = {
    {1, "PMLOG_CLK_GFXCLK", AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_UNIT_HERTZ, 1000000, 1},
    {2, "PMLOG_CLK_MEMCLK", AMDF_TELEMETRY_METRIC_CLOCK_MEMORY,
     AMDF_TELEMETRY_DOMAIN_MEMORY, AMDF_OBSERVATION_UNIT_HERTZ, 1000000, 1},
    {8, "PMLOG_TEMPERATURE_EDGE", AMDF_TELEMETRY_METRIC_TEMPERATURE_EDGE,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_UNIT_CELSIUS, 1, 1},
    {19, "PMLOG_INFO_ACTIVITY_GFX", AMDF_TELEMETRY_METRIC_ACTIVITY_COMPUTE,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_UNIT_RATIO, 1, 100},
    {20, "PMLOG_INFO_ACTIVITY_MEM", AMDF_TELEMETRY_METRIC_ACTIVITY_MEMORY,
     AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER, AMDF_OBSERVATION_UNIT_RATIO, 1,
     100},
    {23, "PMLOG_ASIC_POWER", AMDF_TELEMETRY_METRIC_POWER,
     AMDF_TELEMETRY_DOMAIN_PACKAGE, AMDF_OBSERVATION_UNIT_WATTS, 1, 1},
    {27, "PMLOG_TEMPERATURE_HOTSPOT", AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_UNIT_CELSIUS, 1, 1},
};

#define AMDF_GPU_WDDM_SENSOR_COUNT \
  (sizeof(amdf_gpu_wddm_sensors) / sizeof(amdf_gpu_wddm_sensors[0]))

static bool amdf_gpu_wddm_sensor_known(amdf_telemetry_metric_id_t id) {
  for (uint32_t i = 0; i < AMDF_GPU_WDDM_SENSOR_COUNT; ++i) {
    if (amdf_gpu_wddm_sensors[i].id == id) {
      return true;
    }
  }
  return false;
}

static amdf_telemetry_metric_info_t amdf_gpu_wddm_sensor_describe(
    const amdf_gpu_wddm_sensor_t* sensor) {
  return (amdf_telemetry_metric_info_t){
      .source_id = AMDF_GPU_WDDM_PMLOG_SOURCE,
      .source_name = "PMLog",
      .source_flags = AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_MAY_BLOCK |
                      AMDF_TELEMETRY_SOURCE_FLAG_MAY_WAKE_DEVICE |
                      AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_SNAPSHOT,
      .domain = sensor->domain,
      .metric_id = sensor->id,
      .instance_scope = AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE,
      .kind = sensor->kind,
      .format = {.kind = AMDF_OBSERVATION_VALUE_KIND_I64,
                 .unit = sensor->unit,
                 .scale_numerator = sensor->numerator,
                 .scale_denominator = sensor->denominator,
                 .semantic = AMDF_OBSERVATION_VALUE_SEMANTIC_UNSPECIFIED},
      .name = sensor->name,
      .origin = {.interface_name = "windows.amdgpu.pmlog",
                 .schema = "CWDDEPM_SENSOR_TYPE",
                 .field = sensor->name,
                 .field_index = UINT32_MAX},
  };
}

amdf_status_t amdf_gpu_umd_enumerate_telemetry_metrics(
    amdf_gpu_umd_device_t* device, amdf_platform_endpoint_t* endpoint,
    uint32_t capacity, amdf_telemetry_metric_info_t* metrics,
    uint32_t* out_count) {
  (void)endpoint;
  amdf_wkmi_bridge_gpu_telemetry_t snapshot;
  const amdf_status_t status = device->wkmi_adapter.api->gpu_telemetry_read(
      device->adapter, device->device, device->physical_adapter_index,
      &snapshot);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    *out_count = 0;
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t count = 0;
  for (uint32_t i = 0; i < AMDF_GPU_WDDM_SENSOR_COUNT; ++i) {
    const amdf_gpu_wddm_sensor_t* sensor = &amdf_gpu_wddm_sensors[i];
    if (snapshot.supported[sensor->id] != 0) {
      if (count < capacity) {
        metrics[count] = amdf_gpu_wddm_sensor_describe(sensor);
      }
      ++count;
    }
  }
  *out_count = count;
  return capacity != 0 && capacity < count
             ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
             : AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_read_telemetry(
    amdf_gpu_umd_device_t* device, amdf_platform_endpoint_t* endpoint,
    amdf_telemetry_source_id_t source_id, uint32_t metric_count,
    const amdf_telemetry_metric_id_t* metric_ids,
    amdf_telemetry_value_t* values, amdf_telemetry_sample_t* out_sample) {
  (void)endpoint;
  if (source_id != AMDF_GPU_WDDM_PMLOG_SOURCE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  for (uint32_t i = 0; i < metric_count; ++i) {
    if (!amdf_gpu_wddm_sensor_known(metric_ids[i])) {
      return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
    }
  }
  amdf_telemetry_sample_t sample = {0};
  amdf_status_t status =
      amdf_platform_observation_query_host_clock(&sample.host_before);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_wkmi_bridge_gpu_telemetry_t snapshot;
  status = device->wkmi_adapter.api->gpu_telemetry_read(
      device->adapter, device->device, device->physical_adapter_index,
      &snapshot);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_platform_observation_query_host_clock(&sample.host_after);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (uint32_t i = 0; i < metric_count; ++i) {
    const uint32_t sensor = (uint32_t)metric_ids[i];
    // ADL's PMLog value is a signed int; WKMI transports those same 32 bits
    // in uint32_t storage. Its synthesized unavailable value accompanies a
    // cleared support flag. Testing the bits alone would lose a valid -1 C.
    const bool available = snapshot.supported[sensor] != 0;
    values[i] = (amdf_telemetry_value_t){
        .status = available ? AMDF_STATUS_OK
                            : amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND),
        .value = {.i64 = available ? (int32_t)snapshot.values[sensor] : 0},
    };
  }
  *out_sample = sample;
  return AMDF_STATUS_OK;
}
