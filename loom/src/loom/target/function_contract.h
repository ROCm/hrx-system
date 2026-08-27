// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-aware function contract materialization.
//
// Generic func symbol facts carry only function-local structure: target symbol,
// ABI/export syntax, and signature/import facts. This target layer resolves the
// referenced target-record facts and overlays the func-owned contract onto the
// durable target bundle for lowering, packaging, and execution.

#ifndef LOOM_TARGET_FUNCTION_CONTRACT_H_
#define LOOM_TARGET_FUNCTION_CONTRACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |effective_bundle| and |requirement_bundle| use the same
// codegen representation, artifact format, and target-low descriptor contract.
//
// Device compatibility checks and specialization reporting use this coarse
// contract-shape relation only after target identity has been resolved. The
// comparison intentionally ignores function-owned ABI/export facts and
// target-specific feature and limit facts.
bool loom_target_function_contract_bundles_compatible(
    const loom_target_bundle_t* effective_bundle,
    const loom_target_bundle_t* requirement_bundle);

// Resolves |func_facts|'s target record and materializes the function target
// bundle selected by the func-like symbol.
//
// The target snapshot and config come from target-record facts. The export
// plan starts from the target-record defaults and is then overlaid with
// function-owned ABI/export attrs. Kernel-specific launch metadata is applied
// by kernel-aware callers after they derive it from kernel IR.
// |out_target_facts| optionally receives the immutable projection borrowed
// from |fact_table|.
// |out_bundle_storage| owns the copied payload fields and its embedded bundle
// points at those copies. Returns status only for infrastructure failures.
// Invalid user IR emits a structured diagnostic, sets |out_valid| to false,
// and returns OK.
iree_status_t loom_target_function_contract_resolve(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    const loom_target_facts_t** out_target_facts,
    loom_target_bundle_storage_t* out_bundle_storage);

// Materializes the function target bundle by overlaying |func_facts|'s
// function-owned ABI/export attrs onto |base_bundle|.
//
// |target_name| is used only for diagnostics. Returns status only for
// infrastructure failures; invalid user IR emits diagnostics and sets
// |out_valid| false.
iree_status_t loom_target_function_contract_resolve_from_bundle(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_string_view_t target_name, const loom_target_bundle_t* base_bundle,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage);

// Resolves |func_facts|'s authored target and function-local contract into one
// immutable function target facts allocated from |arena|.
//
// Returns status only for infrastructure failures. Invalid user IR emits a
// structured diagnostic, sets |out_valid| to false, and returns OK.
iree_status_t loom_target_function_contract_resolve_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_target_facts_t** out_facts);

// Applies |func_facts|'s function-local contract to |base_facts| and returns
// immutable function target facts allocated from |arena|.
//
// |target_name| is used only for diagnostics. Returns status only for
// infrastructure failures; invalid user IR emits diagnostics and sets
// |out_valid| false.
iree_status_t loom_target_function_contract_refine_facts(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_string_view_t target_name, const loom_target_facts_t* base_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_target_facts_t** out_facts);

// Applies a module-internal callable's function-local contract to |base_facts|
// and returns immutable function target facts allocated from
// |arena|.
//
// Module-internal callables inherit the target snapshot and configuration but
// have no artifact ABI or export contract by default. An explicitly authored
// function ABI remains authoritative. |target_name| is used only for
// diagnostics. Returns status only for infrastructure failures; invalid user
// IR emits diagnostics and sets |out_valid| false.
iree_status_t loom_target_function_contract_refine_internal_facts(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_string_view_t target_name, const loom_target_facts_t* base_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_target_facts_t** out_facts);

// Applies a fixed workgroup size to an already resolved HAL-kernel export plan.
//
// Kernel dialect code calls this after it has derived launch metadata from
// kernel-owned IR. Ordinary function contract resolution intentionally never
// discovers or stores workgroup sizes.
iree_status_t loom_target_function_contract_apply_hal_workgroup_size(
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    const loom_target_workgroup_size_t* required_workgroup_size,
    iree_diagnostic_emitter_t diagnostic_emitter,
    loom_target_bundle_storage_t* bundle_storage, bool* out_valid);

// Applies a fixed workgroup size to |base_facts| and returns one immutable
// function target facts allocated from |arena|.
//
// Kernel dialect code calls this after deriving launch metadata from
// kernel-owned IR. Invalid user IR emits diagnostics and sets |out_valid|
// false.
iree_status_t loom_target_function_contract_refine_hal_workgroup_size(
    const loom_func_symbol_facts_t* func_facts, iree_string_view_t target_name,
    const loom_target_workgroup_size_t* required_workgroup_size,
    const loom_target_facts_t* base_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_target_facts_t** out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FUNCTION_CONTRACT_H_
