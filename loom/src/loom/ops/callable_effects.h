// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared effect semantics for operations that reference callable symbols.

#ifndef LOOM_OPS_CALLABLE_EFFECTS_H_
#define LOOM_OPS_CALLABLE_EFFECTS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |function| has pure callable semantics. Definitions are
// classified from their cached body effects while declarations use their
// explicit effect contract.
bool loom_callable_effects_is_pure(loom_func_like_t function);

// Returns true when |function| carries an explicit deterministic read-only
// contract. A pure function does not carry that contract.
bool loom_callable_effects_has_readonly_contract(loom_func_like_t function);

// Returns the strongest callable effect contract established by |callee| in
// |module|. Definitions may infer purity from their cached body summaries;
// deterministic read-only behavior requires an explicit contract. Unresolved
// symbols and effectful callees return zero.
loom_callable_effect_contract_t loom_callable_effects_callee_contract(
    const loom_module_t* module, loom_symbol_ref_t callee);

// Returns true when |provided| satisfies the effects promised by |required|.
// An unspecified requirement accepts any provider. Pure providers satisfy
// both pure and read-only requirements; read-only providers do not satisfy a
// pure requirement.
bool loom_callable_effects_contract_satisfies(
    loom_callable_effect_contract_t provided,
    loom_callable_effect_contract_t required);

// Returns the strongest callable contract satisfied by |traits|. Pure is
// returned when no runtime effects are present, read-only when deterministic
// reads are the only effects, and unspecified for all other effect classes.
// Local register storage relations preserve values across the callable
// boundary; their distinct owners remain an operation-level optimization
// constraint.
loom_callable_effect_contract_t loom_callable_effects_classify_traits(
    loom_trait_flags_t traits);

// Returns the strongest callable contract satisfied by every operation nested
// in |body|. This is a cold verification query and walks the body once.
loom_callable_effect_contract_t loom_callable_effects_classify_body(
    const loom_module_t* module, const loom_region_t* body);

// Returns the exact contract established by |function|. Definitions are
// classified from their body and then limited by any explicit contract;
// declarations return their explicit contract. This is a cold verification
// query and may walk a definition body once.
loom_callable_effect_contract_t loom_callable_effects_classify_function(
    const loom_module_t* module, loom_func_like_t function);

// Returns true when every operation nested in |body| satisfies the
// deterministic read-only callable contract. This is a cold verification
// query and walks the body once; pass hot paths consume explicit contracts.
bool loom_callable_effects_body_is_readonly(const loom_module_t* module,
                                            const loom_region_t* body);

// Returns whether a callable may access storage. Definitions use their cached
// body summary; declarations require an explicit pure contract. A false result
// does not imply purity: clocks, counters, and fences remain effectful.
bool loom_callable_effects_may_access_memory(loom_func_like_t function);

// Returns the retained upper bound on memory spaces invalidated by a callable.
// Definitions use their cached body summary; pure and read-only declarations
// exclude writes. Unknown effects and fences that can import external writes
// remain conservative. This does not weaken the callable's ordinary write,
// ordering, or observability semantics.
loom_memory_space_mask_t loom_callable_effects_written_memory_spaces(
    loom_func_like_t function);

// Returns the effective traits for a callable application with |contract|.
// Its callable boundary remains independent of the selected effects.
loom_trait_flags_t loom_callable_effects_traits(
    loom_callable_effect_contract_t contract);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_CALLABLE_EFFECTS_H_
