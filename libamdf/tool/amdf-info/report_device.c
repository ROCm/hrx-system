// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>

#include "libamdf/tool/amdf-info/report.h"

static const amdf_info_name_t amdf_info_gpu_features[] = {
    {AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION, "host_registration"},
    {AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION, "device_recreation"},
    {AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY, "local_memory"},
    {AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY,
     "host_visible_local_memory"},
    {AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS, "native_events"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_xdna_placements[] = {
    {AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY, "fixed_full_array"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_xdna_scheduling[] = {
    {AMDF_XDNA_SCHEDULING_MODE_EXCLUSIVE, "exclusive"},
    {AMDF_XDNA_SCHEDULING_MODE_SPATIAL, "spatial"},
    {AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED, "time_sliced"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_xdna_encodings[] = {
    {AMDF_XDNA_BINARY_FORMAT_UNKNOWN, "unknown"},
    {AMDF_XDNA_BINARY_FORMAT_TRANSACTION, "transaction"},
    {0, NULL},
};

static void amdf_info_report_device_identity(amdf_info_output_t* output,
                                             const amdf_device_id_t* id,
                                             uint64_t reset_epoch) {
  char identity[33];
  snprintf(identity, sizeof(identity), "%016" PRIx64 "%016" PRIx64,
           id->words[0], id->words[1]);
  amdf_info_json_string(output, "id", identity);
  amdf_info_json_u64(output, "reset_epoch", reset_epoch);
}

void amdf_info_report_gpu_device(amdf_info_output_t* output,
                                 const amdf_gpu_device_info_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "device", '{');
    amdf_info_report_device_identity(output, &info->id, info->reset_epoch);
    amdf_info_json_flags(output, "features", info->features,
                         amdf_info_gpu_features);
    amdf_info_json_end(output, '}');
  } else if (output->details) {
    amdf_info_output_print(output, "  Live GPU    ");
    amdf_info_output_flags(output, info->features, amdf_info_gpu_features);
    amdf_info_output_print(output, "\n  Reset epoch %" PRIu64 "\n",
                           info->reset_epoch);
  }
}

void amdf_info_report_xdna_device(amdf_info_output_t* output,
                                  const amdf_xdna_device_info_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "device", '{');
    amdf_info_report_device_identity(output, &info->id, info->reset_epoch);
    amdf_info_json_flags(output, "placement_modes", info->placement_modes,
                         amdf_info_xdna_placements);
    amdf_info_json_begin(output, "array", '{');
    amdf_info_json_u32(output, "column_origin", info->array.column_origin);
    amdf_info_json_u32(output, "column_count", info->array.column_count);
    amdf_info_json_u32(output, "row_count", info->array.row_count);
    amdf_info_json_u64(output, "column_stride", info->array.column_stride);
    amdf_info_json_end(output, '}');
    amdf_info_json_begin(output, "context", '{');
    amdf_info_json_flags(output, "scheduling_modes",
                         info->context.scheduling_modes,
                         amdf_info_xdna_scheduling);
    amdf_info_json_u32(output, "minimum_column_count",
                       info->context.minimum_column_count);
    amdf_info_json_u32(output, "maximum_column_count",
                       info->context.maximum_column_count);
    amdf_info_json_u32(output, "column_count_granularity",
                       info->context.column_count_granularity);
    amdf_info_json_end(output, '}');
    amdf_info_json_begin(output, "instruction", '{');
    amdf_info_json_u64(output, "maximum_byte_length",
                       info->instruction.maximum_byte_length);
    amdf_info_json_u32(output, "address_alignment",
                       info->instruction.address_alignment);
    amdf_info_json_u32(output, "byte_length_granularity",
                       info->instruction.byte_length_granularity);
    amdf_info_json_enum(output, "format", info->instruction.format.format,
                        amdf_info_xdna_encodings);
    amdf_info_json_u32(output, "format_version",
                       info->instruction.format.version);
    amdf_info_json_end(output, '}');
    amdf_info_json_end(output, '}');
  } else {
    amdf_info_output_print(output,
                           "  Live array  %" PRIu32 " columns from %" PRIu32
                           ", %" PRIu32 " rows\n",
                           info->array.column_count, info->array.column_origin,
                           info->array.row_count);
    if (output->details) {
      amdf_info_output_print(output, "  Placement   ");
      amdf_info_output_flags(output, info->placement_modes,
                             amdf_info_xdna_placements);
      amdf_info_output_print(output, "\n  Scheduling  ");
      amdf_info_output_flags(output, info->context.scheduling_modes,
                             amdf_info_xdna_scheduling);
      amdf_info_output_print(
          output,
          "\n  Context     %" PRIu32 "..%" PRIu32 " columns, step %" PRIu32
          "\n"
          "  Addressing  %" PRIu64
          " bytes per column\n"
          "  Instruction %" PRIu64 " bytes maximum, alignment %" PRIu32
          ", length multiple %" PRIu32
          "\n"
          "  Reset epoch %" PRIu64 "\n",
          info->context.minimum_column_count,
          info->context.maximum_column_count,
          info->context.column_count_granularity, info->array.column_stride,
          info->instruction.maximum_byte_length,
          info->instruction.address_alignment,
          info->instruction.byte_length_granularity, info->reset_epoch);
      const char* format = amdf_info_name(info->instruction.format.format,
                                          amdf_info_xdna_encodings);
      amdf_info_output_print(
          output, "  Encoding    %s (%" PRIu32 "), version %" PRIu32 "\n",
          format ? format : "unknown", info->instruction.format.format,
          info->instruction.format.version);
    }
  }
}
