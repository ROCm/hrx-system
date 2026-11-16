// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_INVOCATION_INPUTS_H_
#define LOOM_TRANSFORMS_SYMBOL_INVOCATION_INPUTS_H_

#include "loom/analysis/symbol_references.h"
#include "loom/pass/types.h"
#include "loom/target/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_invocation_input_read_t loom_invocation_input_read_t;

// Local requirements collected during the preparatory reference traversal.
// No storage is allocated until an admitted helper actually reads an input.
typedef struct loom_invocation_inputs_t {
  // Immutable source observed by the reference producer.
  loom_module_t* module;
  // Compile-invocation scratch, released with the preparation pass.
  iree_arena_allocator_t* arena;
  // Selected function versions; NULL for standalone source preparation.
  const loom_target_function_version_snapshot_t* versions;
  // Stable linked records borrowing query operations from the source snapshot.
  loom_invocation_input_read_t* first_read;
  // Append position preserving source discovery order.
  loom_invocation_input_read_t** next_read;
} loom_invocation_inputs_t;

// Observes admitted ordinary-helper queries. Definitions retained only for
// deferred template applications remain unchanged until selected and versioned.
loom_symbol_reference_observer_t loom_invocation_inputs_observer(
    loom_invocation_inputs_t* inputs);

// Makes helper inputs explicit before worker outlining or native call policy.
// Resolves the affected reverse call closure from the retained reference table,
// diagnoses unavailable/fixed boundaries before mutation, then supplies values
// at each exact invocation entry and appends ordinary helper/call arguments.
// Any mutation invalidates references; query-free snapshots remain reusable.
iree_status_t loom_invocation_inputs_apply(
    loom_invocation_inputs_t* inputs, loom_pass_t* pass,
    const loom_symbol_reference_table_t* references, bool* out_changed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INVOCATION_INPUTS_H_
