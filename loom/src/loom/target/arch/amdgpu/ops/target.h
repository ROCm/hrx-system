// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target op interpretation.

#ifndef LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/resolved_target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the canonical AMDGPU target name selected by |target_op|, or empty.
iree_string_view_t loom_amdgpu_target_record_target_name(
    const loom_op_t* target_op);

// Returns the AMDGPU target row selected by |target_op|, or NULL.
const loom_amdgpu_target_info_t* loom_amdgpu_target_record_target(
    const loom_op_t* target_op);

// Returns the AMDGPU processor name selected by |target_op|, or empty.
iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_op_t* target_op);

// Returns the AMDGPU processor row selected by |target_op|, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_op_t* target_op);

// Resolves the structured AMDGPU identity carried by |target_op|.
//
// Unspecified target-ID features resolve from generated processor defaults.
// Callers consume this same identity shape as live target profiles.
void loom_amdgpu_target_record_resolve_identity(
    const loom_op_t* target_op, loom_amdgpu_target_identity_t* out_identity);

// Resolves the compiler-semantic AMDGPU properties carried by |target_op|.
//
// |common| is the indexed target bundle for the record and may contain
// authored target-neutral overrides.
void loom_amdgpu_target_record_resolve_properties(
    const loom_op_t* target_op, const loom_target_bundle_t* common,
    loom_amdgpu_target_properties_t* out_properties);

// Materializes an exact resolved AMDGPU target as an amdgpu.target definition.
iree_status_t loom_amdgpu_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location);

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_
