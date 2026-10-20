// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Operation-local legality policy shared by IR motion transforms.

#ifndef LOOM_ANALYSIS_MOTION_POLICY_H_
#define LOOM_ANALYSIS_MOTION_POLICY_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |traits| permit effect-free relocation. A root terminator
// cannot move, while nested terminators move with their owning region.
bool loom_motion_traits_are_effect_free_relocatable(loom_trait_flags_t traits,
                                                    bool is_root_op);

// Returns true when |traits| permit execution on additional control paths.
// This classifies effects and identity; callers separately establish
// unconditional or fact-dependent execution safety.
bool loom_motion_traits_are_speculation_compatible(loom_trait_flags_t traits,
                                                   bool is_root_op);

// Returns true when a nested region carries effects, convergence, observable
// execution, or a retained compiler hint.
bool loom_motion_op_has_retained_regions(const loom_op_t* op);

// Returns true if |op| has no live uses, no retained compiler hints, and no
// semantic effects that must keep it in the IR.
bool loom_motion_op_can_erase(const loom_module_t* module, const loom_op_t* op);

// Returns true if |op| may be moved while preserving its original dynamic
// execution predicate. Pending source expansion and retained region effects
// keep the operation in its source context.
bool loom_motion_op_can_relocate_effect_free(const loom_module_t* module,
                                             const loom_op_t* op);

// Returns true if |op| may be rebuilt independently without widening its
// dynamic execution predicate. This rejects identity-bearing and callable
// operations but does not require SAFE_TO_SPECULATE.
bool loom_motion_op_can_rematerialize_effect_free(const loom_module_t* module,
                                                  const loom_op_t* op);

// Returns true when |op| has the effect, identity, ownership, and
// source-context properties required to execute on additional control paths.
// The caller must separately prove unconditional or fact-dependent execution
// safety. Accepting already-loaded |traits| keeps transform scans to one trait
// query per op.
bool loom_motion_op_is_speculation_compatible_with_traits(
    const loom_module_t* module, const loom_op_t* op,
    loom_trait_flags_t traits);

// Returns true if |op| may unconditionally execute on additional control
// paths under its current effective |traits|. Accepting the already-loaded
// traits keeps hot transform scans to one cached trait query per operation.
// Fact-dependent speculation requires a separate fact-aware proof.
bool loom_motion_op_can_speculate_with_traits(const loom_module_t* module,
                                              const loom_op_t* op,
                                              loom_trait_flags_t traits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_MOTION_POLICY_H_
