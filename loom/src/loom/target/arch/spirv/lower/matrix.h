// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V lowering bridge for generic matrix contracts.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_MATRIX_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_MATRIX_H_

#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/arch/spirv/value_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a target-independent matrix component type to the exact SPIR-V scalar
// type used by cooperative-matrix declarations and packet descriptors.
bool loom_spirv_matrix_component_type(
    loom_contract_numeric_type_t numeric_type,
    loom_spirv_scalar_type_t* out_scalar_type);

// Returns the SPIR-V scope selected for the current subgroup-fragment policy.
// Source value typing and descriptor selection share this policy boundary so
// their independently constructed matrix types cannot drift.
loom_spirv_scope_t loom_spirv_matrix_scope(void);

// Projects the producer-owned fragment interpretation for one source value.
bool loom_spirv_matrix_fragment(const loom_value_fact_table_t* fact_table,
                                loom_value_id_t source_value_id,
                                loom_vector_fragment_fact_t* out_fragment);

// Maps that interpretation to a native component, shape, scope, and use. Both
// source contract selection and Low value mapping consume this projection.
bool loom_spirv_matrix_fragment_type(const loom_module_t* module,
                                     const loom_value_fact_table_t* fact_table,
                                     loom_value_id_t source_value_id,
                                     loom_vector_fragment_fact_t fragment,
                                     loom_spirv_value_type_t* out_type);

// Rejects physical payload interpretations that cannot preserve an opaque
// cooperative matrix. Uniform constructors continue through descriptor rules.
iree_status_t loom_spirv_query_cooperative_fragment(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Retains the shared matrix-contract diagnostic for unsupported authored input.
iree_status_t loom_spirv_matrix_contract_query_reject(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, iree_string_view_t matrix_constraint,
    uint32_t source_rejection_bits, uint32_t target_rejection_bits,
    uint32_t missing_feature_bits,
    loom_target_contract_query_result_t* out_result);

// Supplies SPIR-V options for shared descriptor-matrix source adapters.
iree_status_t loom_spirv_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

// Projects generic matrix contracts to SPIR-V cooperative matrix properties.
iree_status_t loom_spirv_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_MATRIX_H_
