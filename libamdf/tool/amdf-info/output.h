// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_OUTPUT_H_
#define AMDF_TOOL_INFO_OUTPUT_H_

#include "libamdf/tool/amdf-info/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Streaming output state. The bounded JSON stack follows the compiled schema;
// it is not a DOM, public nesting API, or retained machine model.
typedef struct amdf_info_output_t {
  // Borrowed stream, flushed by the invocation owner before returning success.
  FILE* file;
  // Selected output representation.
  amdf_info_format_t format;
  // Whether text should expose the detailed records.
  bool details;
  // Whether text headings use ANSI emphasis.
  bool color;
  // Sticky write failure; later output becomes inert.
  bool failed;
  // Current JSON indentation/nesting depth, bounded by the report schema.
  uint32_t depth;
  // Whether each enclosing JSON container has emitted its first member.
  bool empty[12];
} amdf_info_output_t;

typedef struct amdf_info_name_t {
  // Numeric enum value or individual flag bit.
  uint64_t value;
  // Stable symbolic name, with NULL terminating a name table.
  const char* name;
} amdf_info_name_t;

// The platform determines terminal support once; serializers need no OS APIs.
void amdf_info_output_initialize(FILE* file, const amdf_info_options_t* options,
                                 bool terminal_color,
                                 amdf_info_output_t* out_output);

// Writes formatted, tool-owned text. Native/user strings use text/string below.
void amdf_info_output_print(amdf_info_output_t* output, const char* format,
                            ...);

// Writes a string while escaping terminal control characters. UTF-8 is
// retained.
void amdf_info_output_text(amdf_info_output_t* output, const char* value);

// Writes a heading, with optional emphasis and no platform-specific behavior.
void amdf_info_output_heading(amdf_info_output_t* output, const char* value);

// JSON keys are NULL for array entries. All functions stream directly to FILE.
void amdf_info_json_begin(amdf_info_output_t* output, const char* key,
                          char delimiter);
void amdf_info_json_end(amdf_info_output_t* output, char delimiter);
void amdf_info_json_string(amdf_info_output_t* output, const char* key,
                           const char* value);
void amdf_info_json_u32(amdf_info_output_t* output, const char* key,
                        uint32_t value);
void amdf_info_json_u64(amdf_info_output_t* output, const char* key,
                        uint64_t value);
void amdf_info_json_hex(amdf_info_output_t* output, const char* key,
                        uint64_t value);
void amdf_info_json_bool(amdf_info_output_t* output, const char* key,
                         bool value);
void amdf_info_json_enum(amdf_info_output_t* output, const char* key,
                         uint32_t value, const amdf_info_name_t* names);
void amdf_info_json_flags(amdf_info_output_t* output, const char* key,
                          uint64_t value, const amdf_info_name_t* names);

// Returns NULL for an unknown value rather than inventing a symbolic identity.
const char* amdf_info_name(uint64_t value, const amdf_info_name_t* names);

// Prints symbolic bits with any remaining unknown bits retained in hexadecimal.
void amdf_info_output_flags(amdf_info_output_t* output, uint64_t value,
                            const amdf_info_name_t* names);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_OUTPUT_H_
