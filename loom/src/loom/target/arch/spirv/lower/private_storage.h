// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function storage behind logical positions. Only the final access forms a
// SPIR-V Function pointer; source control flow carries ordinary offset IDs.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_PRIVATE_STORAGE_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_PRIVATE_STORAGE_H_

#include "loom/codegen/low/lower/representation_observer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Four cell representations, indexed by log2(byte width). These identities are
// distinct from coordinate representations and shared with generated rules.
#define LOOM_SPIRV_PRIVATE_REPRESENTATION_BASE 16

bool loom_spirv_value_is_private(const loom_module_t* module,
                                 const loom_value_fact_table_t* facts,
                                 loom_value_id_t value);
bool loom_spirv_value_is_null(const loom_value_fact_table_t* facts,
                              loom_value_id_t value);

// Records only private reference relations. Null claims its use without
// joining independent objects; mixed storage domains receive a diagnostic.
void loom_spirv_private_observe_relation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder);
void loom_spirv_private_observe_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder);
void loom_spirv_private_observe_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t operand, loom_low_lower_representation_recorder_t* recorder);

// Selects and emits an allocation's retained position. Physical storage is
// emitted once at entry, separately from the allocation's control-flow site.
iree_status_t loom_spirv_select_private_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);
bool loom_spirv_private_alloca_plan_isa(loom_low_lower_plan_t plan);
iree_status_t loom_spirv_emit_private_alloca(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan);
iree_status_t loom_spirv_emit_private_storage(
    loom_low_lower_context_t* context);

// Generated memory recipes obtain the typed array and its canonical origin
// through these materializers. Preparation retains the component and source
// address; emission never recovers aliases or revisits source analysis.
iree_status_t loom_spirv_can_materialize_private_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize);
iree_status_t loom_spirv_prepare_private_storage(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan);
loom_type_t loom_spirv_private_storage_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* plan);
loom_type_t loom_spirv_private_origin_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* plan);
iree_status_t loom_spirv_materialize_private_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan, loom_value_id_t* out_value);
iree_status_t loom_spirv_materialize_private_origin(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan, loom_value_id_t* out_value);

// Null has no storage component and can be adapted at each receiving boundary.
iree_status_t loom_spirv_emit_private_null(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t* out_value);
iree_status_t loom_spirv_can_materialize_private_position(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize);
iree_status_t loom_spirv_prepare_private_position(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan);
loom_type_t loom_spirv_private_position_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* plan);
iree_status_t loom_spirv_materialize_private_position(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan, loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_PRIVATE_STORAGE_H_
