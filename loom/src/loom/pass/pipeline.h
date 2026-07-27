// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shallow textual pass-list parsing.
//
// This parser is the compatibility boundary for tool flags such as
// `--pass=canonicalize{max-iterations=4}` and loom-check `RUN: pass ...`
// directives. It does not execute passes. Tooling translates the parsed entries
// into synthetic pass.pipeline IR, then relies on the normal verifier,
// compiler, and interpreter path.

#ifndef LOOM_PASS_PIPELINE_H_
#define LOOM_PASS_PIPELINE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// A parsed entry from a shallow textual pass list.
typedef struct loom_pass_pipeline_entry_spec_t {
  // Pass name before any option dictionary.
  iree_string_view_t name;
  // Text inside the optional option dictionary, without surrounding braces.
  iree_string_view_t options;
} loom_pass_pipeline_entry_spec_t;

// Callback invoked for each parsed pass option assignment.
typedef iree_status_t (*loom_pass_option_parse_fn_t)(void* user_data,
                                                     iree_string_view_t name,
                                                     iree_string_view_t value);

typedef struct loom_pass_option_parse_callback_t {
  // Callback invoked for each parsed option assignment.
  loom_pass_option_parse_fn_t fn;
  // Opaque caller data passed to |fn|.
  void* user_data;
} loom_pass_option_parse_callback_t;

// Consumes the next entry from a comma-separated pass list. Commas inside one
// level of `{...}` are treated as option separators, not pass separators.
iree_status_t loom_pass_pipeline_consume_entry(
    iree_string_view_t* pipeline, loom_pass_pipeline_entry_spec_t* out_entry,
    bool* out_has_entry);

// Parses comma-separated `name=value` options and invokes |parse| for each
// assignment. Malformed or empty assignments are rejected.
iree_status_t loom_pass_options_parse(iree_string_view_t pass_name,
                                      iree_string_view_t options,
                                      loom_pass_option_parse_callback_t parse);

// Parses a uint32 option value for descriptor create callbacks and tooling.
iree_status_t loom_pass_option_parse_uint32(iree_string_view_t pass_name,
                                            iree_string_view_t option_name,
                                            iree_string_view_t option_value,
                                            uint32_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_PIPELINE_H_
