// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBAMDF_SRC_GPU_UMD_KFD_METRICS_H_
#define LIBAMDF_SRC_GPU_UMD_KFD_METRICS_H_

#include "amdf/telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

// gpu_metrics is a sysfs record, bounded by the x86 kernel's PAGE_SIZE - 1.
// One read into this storage captures the whole native result without seeking
// or requesting the header and payload from different firmware observations.
#define AMDF_GPU_KFD_METRICS_CAPACITY 4096u

// Private source namespace shared with the provider's hwmon/snapshot dispatch.
#define AMDF_GPU_KFD_METRICS_SOURCE_BASE UINT64_C(0x0200000000000000)

// Indexed native fields, including timestamp/filter metadata. The decoder owns
// field identities; consumers select public metric IDs instead of these slots.
#define AMDF_GPU_KFD_METRICS_FIELD_COUNT 42u

typedef struct amdf_gpu_kfd_metrics_field_t {
  // Byte offset within the validated, borrowed native record.
  uint16_t offset;
  // Number of native elements; zero means this field is absent.
  uint16_t count;
  // Unsigned native element width in bytes.
  uint8_t width;
  // Native array shape, including arrays with only one reported element.
  uint8_t is_array;
} amdf_gpu_kfd_metrics_field_t;

// Validated native snapshot. The caller keeps data alive and unmodified until
// all description/value operations finish. No allocation or handle is owned.
typedef struct amdf_gpu_kfd_metrics_t {
  // Borrowed bytes whose complete extent and field locations were validated.
  const uint8_t* data;
  // Native producer/schema identity used by public selected-source reads.
  amdf_telemetry_source_id_t source_id;
  // Decoder-owned producer profile selected by the native MP1 IP version.
  uint32_t profile;
  // Native format/content revision, encoded as format << 8 | content.
  uint16_t version;
  // Immutable native schema name.
  const char* schema;
  // Validated field locations, indexed once at the external byte boundary.
  amdf_gpu_kfd_metrics_field_t fields[AMDF_GPU_KFD_METRICS_FIELD_COUNT];
} amdf_gpu_kfd_metrics_t;

// Whether this native MP1 version has an established snapshot interpretation.
bool amdf_gpu_kfd_metrics_supports_producer(uint32_t mp1_version);

// Validates a complete native record and its producer interpretation before
// publishing out_metrics. Unsupported producer/schema returns UNSUPPORTED;
// malformed known records return EPROTO. All failures preserve the output.
amdf_status_t amdf_gpu_kfd_metrics_decode(uint32_t mp1_version, uint32_t length,
                                          const uint8_t* data,
                                          amdf_gpu_kfd_metrics_t* out_metrics);

// Checks caller source/field identities before native access. Tagged array
// extents are checked again against the returned record, not guessed here.
amdf_status_t amdf_gpu_kfd_metrics_validate_selection(
    amdf_telemetry_source_id_t source_id, uint32_t count,
    const amdf_telemetry_metric_id_t* metric_ids);

// Writes up to capacity descriptors and returns the total supported count.
// The validated snapshot is trusted; no native access or allocation occurs.
uint32_t amdf_gpu_kfd_metrics_describe(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t capacity,
    amdf_telemetry_metric_info_t* out_metrics);

// Validates the selected fields against this record before publishing values
// and its native timestamps. The caller owns the enclosing host bracket in
// sample; this operation only assigns driver_time and source_time on success.
amdf_status_t amdf_gpu_kfd_metrics_read(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t count,
    const amdf_telemetry_metric_id_t* metric_ids,
    amdf_telemetry_value_t* values, amdf_telemetry_sample_t* sample);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBAMDF_SRC_GPU_UMD_KFD_METRICS_H_
