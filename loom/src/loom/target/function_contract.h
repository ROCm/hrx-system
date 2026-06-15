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
// selected target bundle for lowering, packaging, and execution.

#ifndef LOOM_TARGET_FUNCTION_CONTRACT_H_
#define LOOM_TARGET_FUNCTION_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |selected_bundle| can refine |module_bundle| without
// changing the target family, artifact format, or target-low descriptor
// contract selected by the module IR.
//
// Runtime/device target selection uses this to apply concrete limits and
// feature bits after a source module has selected a compatible target record.
// The comparison intentionally ignores ABI/export facts and target feature
// bits: ABI/export selection belongs to the authored function/target contract,
// while feature bits are the primary facts a selected device bundle is expected
// to refine.
bool loom_target_function_contract_bundles_compatible(
    const loom_target_bundle_t* module_bundle,
    const loom_target_bundle_t* selected_bundle);

// Refines |bundle_storage|'s target snapshot and config with a compatible
// runtime-selected bundle while preserving the already resolved function-local
// export plan.
//
// Callers must first validate compatibility with
// loom_target_function_contract_bundles_compatible. This helper only performs
// the structural overlay used by source selection and low target binding.
void loom_target_function_contract_apply_compatible_selection(
    const loom_target_bundle_t* selected_bundle,
    loom_target_bundle_storage_t* bundle_storage);

// Resolves |func_facts|'s target record and materializes the effective target
// bundle selected by the func-like symbol.
//
// The target snapshot and config come from target-record facts. The export
// plan starts from the target-record defaults and is then overlaid with
// function-owned ABI/export attrs. Kernel-specific launch metadata is applied
// by kernel-aware callers after they derive it from kernel IR.
// |out_bundle_storage| owns the copied payload fields and its embedded bundle
// points at those copies. Returns status only for infrastructure failures.
// Invalid user IR emits a structured diagnostic, sets |out_valid| to false,
// and returns OK.
iree_status_t loom_target_function_contract_resolve(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage);

// Materializes the effective target bundle by overlaying |func_facts|'s
// function-owned ABI/export attrs onto |base_bundle|.
//
// This is the contract resolver used when a compile front door has already
// selected an effective target bundle, such as a HAL-device-derived runtime
// target. |target_name| is used only for diagnostics. Returns status only for
// infrastructure failures; invalid user IR emits diagnostics and sets
// |out_valid| false.
iree_status_t loom_target_function_contract_resolve_from_bundle(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_string_view_t target_name, const loom_target_bundle_t* base_bundle,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_target_bundle_storage_t* out_bundle_storage);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FUNCTION_CONTRACT_H_
