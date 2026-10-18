// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native compiler diagnostic rendering for annotation matching.

#ifndef LOOM_TESTING_TEST_DIAGNOSTIC_NATIVE_H_
#define LOOM_TESTING_TEST_DIAGNOSTIC_NATIVE_H_

#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/testing/test_diagnostic.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TESTING_TEST_DIAGNOSTIC_NATIVE_H_
