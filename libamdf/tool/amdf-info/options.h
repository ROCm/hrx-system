// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_OPTIONS_H_
#define AMDF_TOOL_INFO_OPTIONS_H_

#include <stdbool.h>
#include <stdio.h>

#include "amdf/base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum amdf_info_format_e {
  AMDF_INFO_FORMAT_TEXT,
  AMDF_INFO_FORMAT_JSON,
} amdf_info_format_t;

typedef enum amdf_info_color_e {
  AMDF_INFO_COLOR_AUTO,
  AMDF_INFO_COLOR_ALWAYS,
  AMDF_INFO_COLOR_NEVER,
} amdf_info_color_t;

// Borrowed command-line values remain valid through the invocation.
typedef struct amdf_info_options_t {
  // Output representation; JSON always contains the detailed queried records.
  amdf_info_format_t format;
  // Terminal emphasis policy for text output.
  amdf_info_color_t color;
  // Whether exactly one matching endpoint is required.
  bool select;
  // Whether to expand text capability records.
  bool details;
  // Whether to check explicit device creation, queries, and teardown.
  bool activate;
  // Whether to print usage without loading or querying the provider.
  bool help;
  // Optional Loom-style URI or native device path.
  const char* device;
  // Optional exact compiler target identifier.
  const char* target;
  // Optional broad engine filter; UNKNOWN means no filter.
  amdf_engine_kind_t engine;
} amdf_info_options_t;

// Returns a static diagnostic on invalid arguments, or NULL on success.
// No provider operation, allocation, or native path resolution occurs here.
const char* amdf_info_options_parse(int argument_count,
                                    const char* const* arguments,
                                    amdf_info_options_t* out_options);

// Prints the command's public syntax and operation boundaries.
void amdf_info_options_print_usage(FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_OPTIONS_H_
