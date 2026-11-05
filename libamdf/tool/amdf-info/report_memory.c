// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "libamdf/tool/amdf-info/report.h"

static const amdf_info_name_t amdf_info_scope_kinds[] = {
    {AMDF_MEMORY_SCOPE_KIND_SYSTEM, "system"},
    {AMDF_MEMORY_SCOPE_KIND_LOCAL, "local"},
    {AMDF_MEMORY_SCOPE_KIND_PRIVATE, "private"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_scope_flags[] = {
    {AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED, "system_backed"},
    {AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, "available_bytes"},
    {0, NULL},
};

void amdf_info_report_pools_begin(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "memory_pools", '[');
  } else {
    amdf_info_output_print(output, "\n");
    amdf_info_output_heading(output, "Memory pools");
    amdf_info_output_print(output, "\n");
  }
}

void amdf_info_report_pool(amdf_info_output_t* output, uint32_t ordinal,
                           const amdf_memory_scope_info_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, NULL, '{');
    amdf_info_json_u32(output, "ordinal", ordinal);
    amdf_info_json_enum(output, "kind", info->kind, amdf_info_scope_kinds);
    amdf_info_json_u64(output, "capacity_bytes", info->capacity_byte_length);
    amdf_info_json_u32(output, "memory_profile_count",
                       info->memory_profile_count);
    amdf_info_json_flags(output, "flags", info->flags, amdf_info_scope_flags);
    amdf_info_json_end(output, '}');
  } else {
    const char* kind = amdf_info_name(info->kind, amdf_info_scope_kinds);
    amdf_info_output_print(output, "  %s:%" PRIu32 "    ",
                           kind ? kind : "unknown", ordinal);
    if (info->capacity_byte_length) {
      amdf_info_output_print(output, "%.1f GiB",
                             (double)info->capacity_byte_length / 1073741824.0);
    } else {
      amdf_info_output_print(output, "capacity unavailable");
    }
    if (info->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      amdf_info_output_print(output, " system RAM");
    }
    amdf_info_output_print(output, "\n");
    if (output->details) {
      amdf_info_output_print(
          output, "              %" PRIu64 " bytes; %" PRIu32 " profiles; ",
          info->capacity_byte_length, info->memory_profile_count);
      amdf_info_output_flags(output, info->flags, amdf_info_scope_flags);
      amdf_info_output_print(output, "\n");
    }
  }
}

void amdf_info_report_pools_end(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, ']');
  }
}
