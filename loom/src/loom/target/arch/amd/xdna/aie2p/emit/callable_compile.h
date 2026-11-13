// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Detached native callable code and its exact register ABI.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CALLABLE_COMPILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CALLABLE_COMPILE_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_callable_t {
  // Arena-owned argument/result carriers and exact physical-register clobbers.
  loom_low_call_contract_t call_contract;
  // Maximum stack bytes live during this function, including nested callees.
  uint64_t stack_byte_length;
  // Detached code, relocations, and storage facts for that register contract.
  loom_aie2p_leaf_contribution_t contribution;
} loom_aie2p_callable_t;

// Compiles one admitted module-internal helper into a detached native result.
// Source admission has established its supported scalar/pointer signature and
// absence of resource imports and authored mutable storage. The retained
// direct callee contracts are complete before this function is compiled.
// |options| supplies target/source facts, ABI register reservations, and
// optional reporting. This boundary supplies argument/result locations and
// caller-owned stack semantics; allocation repair may mutate |module| during
// this call.
//
// Every pointer in |out_callable| belongs to |arena| or immutable target
// tables. The module and source analysis need survive only through compilation;
// native callers and assembly consume the result without retaining that
// workspace. Optional reports retain their own documented source-lifetime
// obligations. Structured rejection leaves |out_compiled| false and no result.
// Statuses name infrastructure failures, as for ordinary leaf compilation.
iree_status_t loom_aie2p_callable_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    loom_symbol_ref_array_t direct_callees, iree_arena_allocator_t* arena,
    bool* out_compiled, loom_aie2p_callable_t* out_callable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CALLABLE_COMPILE_H_
