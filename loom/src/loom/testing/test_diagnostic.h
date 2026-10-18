// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rendered diagnostic records and annotation matching for .loom-test tooling.

#ifndef LOOM_TESTING_TEST_DIAGNOSTIC_H_
#define LOOM_TESTING_TEST_DIAGNOSTIC_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/error_defs.h"
#include "loom/testing/test_file.h"

#ifdef __cplusplus
extern "C" {
#endif

// One named rendered diagnostic parameter copied for annotation matching.
typedef struct loom_test_diagnostic_param_t {
  // Stable parameter name from the diagnostic definition.
  iree_string_view_t name;
  // Canonically rendered parameter value.
  iree_string_view_t value;
} loom_test_diagnostic_param_t;

// One diagnostic copied into arena-owned storage for matching and reporting.
typedef struct loom_test_diagnostic_t {
  // Diagnostic severity to match against ERROR/WARNING/REMARK annotations.
  loom_diagnostic_severity_t severity;

  // Structured error domain to match against DOMAIN/CODE annotations.
  loom_error_domain_t domain;

  // Structured error code to match against DOMAIN/CODE annotations.
  uint16_t code;

  // Original diagnostic source, independent of its rendered message.
  struct {
    // Source identity copied into the diagnostic arena.
    iree_string_view_t filename;
    // One-based source line, or 0 if unknown.
    uint32_t line;
  } origin;

  // Rendered diagnostic message text, arena-allocated.
  iree_string_view_t message;

  // Named rendered parameters, arena-allocated.
  loom_test_diagnostic_param_t* params;

  // Number of populated entries in params.
  iree_host_size_t param_count;

  // Full source-rendered diagnostic text, arena-allocated.
  iree_string_view_t formatted_diagnostic;

  // True once this diagnostic has matched one expected annotation.
  bool matched;
} loom_test_diagnostic_t;

// Returns true when |diagnostic| belongs to |filename| and all constraints in
// |annotation| match. A same-numbered line in another source cannot match.
bool loom_test_diagnostic_matches_annotation(
    const loom_test_diagnostic_t* diagnostic,
    const loom_test_annotation_t* annotation, iree_string_view_t filename);

// Finds a maximum one-to-one matching between diagnostics and annotations.
// Every diagnostic has matched reset and then set when a match is found.
// |out_annotation_to_diagnostic| receives an arena-owned array with one entry
// per annotation; unmatched entries contain IREE_HOST_SIZE_MAX. An empty
// annotation set returns NULL.
iree_status_t loom_test_diagnostics_match_annotations(
    loom_test_diagnostic_t* diagnostics, iree_host_size_t diagnostic_count,
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, iree_string_view_t filename,
    iree_arena_allocator_t* arena,
    iree_host_size_t** out_annotation_to_diagnostic);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TESTING_TEST_DIAGNOSTIC_H_
