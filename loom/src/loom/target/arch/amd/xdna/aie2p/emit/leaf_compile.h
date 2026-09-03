// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Complete detached compilation of one AIE2P core Low function.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_COMPILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_COMPILE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_leaf_compile_options_t {
  // Descriptor registry used to resolve the core representation contract.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional invocation-refined target facts for this function version.
  const loom_target_facts_t* function_target_facts;
  // Diagnostic emitter receiving scheduling and allocation failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Optional compile report receiving exact Low planning evidence.
  loom_target_compile_report_t* compile_report;
} loom_aie2p_leaf_compile_options_t;

// Compiles one verified amd.xdna.aie2p.core Low function into an arena-owned
// detached native contribution and exact realization facts.
iree_status_t loom_aie2p_leaf_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    iree_arena_allocator_t* arena,
    loom_aie2p_leaf_contribution_t* out_contribution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_COMPILE_H_
