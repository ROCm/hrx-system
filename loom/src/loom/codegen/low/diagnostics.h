// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared helpers for target-low backend diagnostics.

#ifndef LOOM_CODEGEN_LOW_DIAGNOSTICS_H_
#define LOOM_CODEGEN_LOW_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns |value| unless it is empty, otherwise returns |placeholder|.
iree_string_view_t loom_low_diagnostic_string_or_placeholder(
    iree_string_view_t value, iree_string_view_t placeholder);

// Returns the module-local symbol name for |symbol_ref|, or "<unnamed>".
iree_string_view_t loom_low_diagnostic_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref);

// Returns the resolved target record name selected by |target|, or "<empty>".
iree_string_view_t loom_low_diagnostic_target_key(
    const loom_low_resolved_target_t* target);

// Returns the resolved export plan name, or "<empty>" when |target| has no
// target-symbol witness and therefore no selected export plan.
iree_string_view_t loom_low_diagnostic_export_name(
    const loom_low_resolved_target_t* target);

// Returns the resolved config name, or "<empty>" when |target| has no
// target-symbol witness and therefore no selected config.
iree_string_view_t loom_low_diagnostic_config_key(
    const loom_low_resolved_target_t* target);

// Returns the low function symbol name for |function_op|, or "<unnamed>".
iree_string_view_t loom_low_diagnostic_function_name(
    const loom_module_t* module, const loom_op_t* function_op);

// Emits LOOM_ERR_TARGET_026 for a target-dependent use of |function_op|.
iree_status_t loom_low_diagnostic_emit_missing_target(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter);

// Checks |workgroup_storage_bytes| against the resolved target limit. Emits
// LOOM_ERR_TARGET_051 and sets |out_valid| false when the nonzero limit is
// exceeded. |out_valid| may be NULL when the caller counts emitted errors.
iree_status_t loom_low_diagnostic_validate_workgroup_storage_limit(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_resolved_target_t* target, uint64_t workgroup_storage_bytes,
    iree_diagnostic_emitter_t emitter, bool* out_valid);

// Returns the dotted operation mnemonic for |op|, or "<unknown>".
iree_string_view_t loom_low_diagnostic_operation_name(
    const loom_module_t* module, const loom_op_t* op);

// Returns the SSA value name for |value_id|, or a diagnostic placeholder.
iree_string_view_t loom_low_diagnostic_value_name(const loom_module_t* module,
                                                  loom_value_id_t value_id);

// Returns a descriptor-local display name for |value_class|, or "<unknown>".
iree_string_view_t loom_low_diagnostic_value_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_liveness_value_class_t value_class);

// Returns a display name for |block|, or "<anonymous>".
iree_string_view_t loom_low_diagnostic_block_name(const loom_module_t* module,
                                                  const loom_block_t* block);

// Returns the defining op for |value_id|, or |fallback_op| for block args and
// malformed value references.
const loom_op_t* loom_low_diagnostic_value_origin_op(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* fallback_op);

// Returns the operation mnemonic that produced |value_id|. Block arguments
// report "<block-argument>" because their anchor op is only diagnostic context.
iree_string_view_t loom_low_diagnostic_value_origin_operation_name(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* fallback_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_DIAGNOSTICS_H_
