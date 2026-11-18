// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>

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

void amdf_info_report_pools_begin(amdf_info_output_t* output,
                                  bool device_pools) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "memory_pools", '[');
  } else if (!device_pools) {
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
    char identity[33];
    snprintf(identity, sizeof(identity), "%016" PRIx64 "%016" PRIx64,
             info->physical_endpoint_id.words[0],
             info->physical_endpoint_id.words[1]);
    amdf_info_json_string(
        output, "physical_endpoint_id",
        info->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL ? identity : NULL);
    amdf_info_json_u64(output, "capacity_bytes", info->capacity_byte_length);
    amdf_info_json_u32(output, "memory_profile_count",
                       info->memory_profile_count);
    amdf_info_json_flags(output, "flags", info->flags, amdf_info_scope_flags);
    amdf_info_json_end(output, '}');
  } else {
    const char* kind = amdf_info_name(info->kind, amdf_info_scope_kinds);
    amdf_info_output_print(
        output, "  %s%s:%" PRIu32 "    ",
        info->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM ? "" : "Memory      ",
        kind ? kind : "unknown", ordinal);
    if (info->capacity_byte_length) {
      amdf_info_output_print(output, "%.1f GiB",
                             (double)info->capacity_byte_length / 1073741824.0);
    } else {
      amdf_info_output_print(output, "capacity unavailable");
    }
    if (info->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      amdf_info_output_print(output, " system RAM");
    } else if (info->flags & AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED) {
      amdf_info_output_print(output, " shared system RAM");
    } else if (info->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL) {
      amdf_info_output_print(output, " device-local memory");
    }
    amdf_info_output_print(output, "\n");
    if (output->details) {
      amdf_info_output_print(
          output, "              %" PRIu64 " bytes; %" PRIu32 " profiles; ",
          info->capacity_byte_length, info->memory_profile_count);
      amdf_info_output_flags(output, info->flags, amdf_info_scope_flags);
      amdf_info_output_print(output, "\n");
      if (info->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL) {
        amdf_info_output_print(output,
                               "              Storage endpoint %016" PRIx64
                               "%016" PRIx64 "\n",
                               info->physical_endpoint_id.words[0],
                               info->physical_endpoint_id.words[1]);
      }
    }
  }
}

void amdf_info_report_pools_end(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, ']');
  }
}
