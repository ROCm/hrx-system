// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Diagnostic materialization and annotation matching for .loom-test tooling.

#ifndef LOOM_TESTING_TEST_DIAGNOSTIC_H_
#define LOOM_TESTING_TEST_DIAGNOSTIC_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/testing/test_file.h"

#ifdef __cplusplus
extern "C" {
#endif

// One diagnostic copied into arena-owned storage for matching and reporting.
typedef struct loom_test_diagnostic_t {
  // Diagnostic severity to match against ERROR/WARNING/REMARK annotations.
  loom_diagnostic_severity_t severity;

  // Structured error domain to match against DOMAIN/CODE annotations.
  loom_error_domain_t domain;

  // Structured error code to match against DOMAIN/CODE annotations.
  uint16_t code;

  // Generated error definition carrying parameter names.
  const loom_error_def_t* error;

  // One-based source line where the diagnostic was emitted, or 0 if unknown.
  uint32_t origin_line;

  // Rendered diagnostic message text, arena-allocated.
  iree_string_view_t message;

  // Rendered parameter values in the error schema's order, arena-allocated.
  iree_string_view_t* param_values;

  // Number of populated entries in param_values.
  iree_host_size_t param_value_count;

  // Full source-rendered diagnostic text, arena-allocated.
  iree_string_view_t formatted_diagnostic;

  // True once this diagnostic has matched one expected annotation.
  bool matched;
} loom_test_diagnostic_t;

// Context used to render diagnostic messages and typed parameter values.
typedef struct loom_test_diagnostic_format_options_t {
  // Module owning types referenced by diagnostic parameters. May be NULL while
  // collecting parser diagnostics before a module exists.
  const loom_module_t* module;

  // Text printer options used for type-bearing diagnostic parameters.
  loom_text_print_options_t text_print_options;
} loom_test_diagnostic_format_options_t;

// Copies and renders |diagnostic| into arena-owned |out_diagnostic| storage.
iree_status_t loom_test_diagnostic_materialize(
    const loom_diagnostic_t* diagnostic,
    const loom_test_diagnostic_format_options_t* options,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    loom_test_diagnostic_t* out_diagnostic);

// Returns true when all constraints in |annotation| match |diagnostic|.
bool loom_test_diagnostic_matches_annotation(
    const loom_test_diagnostic_t* diagnostic,
    const loom_test_annotation_t* annotation);

// Finds a maximum one-to-one matching between diagnostics and annotations.
// Every diagnostic has matched reset and then set when a match is found.
// |out_annotation_to_diagnostic| receives an arena-owned array with one entry
// per annotation; unmatched entries contain IREE_HOST_SIZE_MAX. An empty
// annotation set returns NULL.
iree_status_t loom_test_diagnostics_match_annotations(
    loom_test_diagnostic_t* diagnostics, iree_host_size_t diagnostic_count,
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, iree_arena_allocator_t* arena,
    iree_host_size_t** out_annotation_to_diagnostic);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TESTING_TEST_DIAGNOSTIC_H_
