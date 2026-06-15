// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_SCF_OPS_H_
#define LOOM_OPS_SCF_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_SCF_FOR = LOOM_OP_KIND(LOOM_DIALECT_SCF, 0),
  LOOM_OP_SCF_IF = LOOM_OP_KIND(LOOM_DIALECT_SCF, 1),
  LOOM_OP_SCF_SWITCH = LOOM_OP_KIND(LOOM_DIALECT_SCF, 2),
  LOOM_OP_SCF_YIELD = LOOM_OP_KIND(LOOM_DIALECT_SCF, 3),
  LOOM_OP_SCF_SELECT = LOOM_OP_KIND(LOOM_DIALECT_SCF, 4),
  LOOM_OP_SCF_LOOKUP = LOOM_OP_KIND(LOOM_DIALECT_SCF, 5),
  LOOM_OP_SCF_CONDITION = LOOM_OP_KIND(LOOM_DIALECT_SCF, 6),
  LOOM_OP_SCF_WHILE = LOOM_OP_KIND(LOOM_DIALECT_SCF, 7),
  LOOM_OP_SCF_COUNT_ = 8,
};

// Local scf.for unroll policy.
typedef enum loom_scf_for_unroll_policy_e {
  LOOM_SCF_FOR_UNROLL_POLICY_UNROLL = 1,
  LOOM_SCF_FOR_UNROLL_POLICY_COUNT_ = 2,
} loom_scf_for_unroll_policy_t;

// LOOM_OP_SCF_FOR: Bounded counted loop with optional loop-carried state.
// scf.for %iv = [%c0 to %n step %c1] {
//   scf.yield
// }
LOOM_DEFINE_ISA(loom_scf_for_isa, LOOM_OP_SCF_FOR)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_scf_for_lower_bound, 0)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_scf_for_upper_bound, 1)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_scf_for_step, 2)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_scf_for_iter_args, 3)
LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND(loom_scf_for_unroll_factor, 4)
LOOM_DEFINE_VARIADIC_RESULTS(loom_scf_for_results, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_scf_for_unroll_policy, 0, loom_scf_for_unroll_policy_t)
LOOM_DEFINE_REGION(loom_scf_for_body, 0)
enum loom_scf_for_build_flag_bits_e {
  LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR = 1u << 0,
  LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY = 1u << 1,
};
typedef uint32_t loom_scf_for_build_flags_t;
iree_status_t loom_scf_for_build(
    loom_builder_t* builder,
    loom_scf_for_build_flags_t build_flags,
    loom_may_consume loom_value_id_t lower_bound,
    loom_may_consume loom_value_id_t upper_bound,
    loom_may_consume loom_value_id_t step,
    loom_may_consume const loom_value_id_t* iter_args,
    iree_host_size_t iter_args_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional loom_may_consume loom_value_id_t unroll_factor,
    loom_optional uint8_t unroll_policy,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scf_for_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scf_for_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SCF_IF: Conditional execution with optional else region for resultless conditionals.
// scf.if %cond {
//   scf.yield
// }
LOOM_DEFINE_ISA(loom_scf_if_isa, LOOM_OP_SCF_IF)
LOOM_DEFINE_OPERAND(loom_scf_if_condition, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_scf_if_results, 0)
LOOM_DEFINE_REGION(loom_scf_if_then_region, 0)
LOOM_DEFINE_OPTIONAL_REGION(loom_scf_if_else_region, 1)
enum loom_scf_if_build_flag_bits_e {
  LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION = 1u << 0,
};
typedef uint32_t loom_scf_if_build_flags_t;
iree_status_t loom_scf_if_build(
    loom_builder_t* builder,
    loom_scf_if_build_flags_t build_flags,
    loom_may_consume loom_value_id_t condition,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scf_if_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scf_region_branch_type_transfer(
    loom_type_transfer_context_t* context,
    const loom_module_t* module, loom_op_t* op);
iree_status_t loom_scf_if_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SCF_SWITCH: Multi-way branch over an index selector. Case keys are sorted unique i64 literals. The default region is mandatory and is selected when the selector does not equal any explicit case key. Every region must terminate with scf.yield matching the switch result tuple.
// scf.switch %selector {
//   case 0 {
//     scf.yield
//   }
//   default {
//     scf.yield
//   }
// }
LOOM_DEFINE_ISA(loom_scf_switch_isa, LOOM_OP_SCF_SWITCH)
LOOM_DEFINE_OPERAND(loom_scf_switch_selector, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_scf_switch_results, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_scf_switch_case_keys, 0)
LOOM_DEFINE_REGION(loom_scf_switch_default_region, 0)
LOOM_DEFINE_VARIADIC_REGIONS(loom_scf_switch_case_regions, 1)
iree_status_t loom_scf_switch_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t selector,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    const int64_t* case_keys,
    iree_host_size_t case_keys_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scf_switch_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scf_switch_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SCF_YIELD: Region terminator forwarding values to the parent scf op.
// scf.yield
LOOM_DEFINE_ISA(loom_scf_yield_isa, LOOM_OP_SCF_YIELD)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_scf_yield_values, 0)
iree_status_t loom_scf_yield_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_SCF_SELECT: Select between two same-typed SSA values using a scalar i1 condition. This is whole-value selection: if the selected values are vectors or tiles, the entire aggregate is chosen as one value. Lanewise vector masking remains vector.select.
// %r = scf.select %cond, %a, %b : f32
LOOM_DEFINE_ISA(loom_scf_select_isa, LOOM_OP_SCF_SELECT)
LOOM_DEFINE_OPERAND(loom_scf_select_condition, 0)
LOOM_DEFINE_OPERAND(loom_scf_select_true_value, 1)
LOOM_DEFINE_OPERAND(loom_scf_select_false_value, 2)
LOOM_DEFINE_RESULT(loom_scf_select_result, 0)
iree_status_t loom_scf_select_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t condition,
    loom_may_consume loom_value_id_t true_value,
    loom_may_consume loom_value_id_t false_value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scf_select_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scf_select_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCF_LOOKUP: Total table lookup over already-computed SSA values. The selector is an index value. The table gives sorted unique explicit case rows. default(...) gives the total fallback row. The result count is the row width, and every payload value in a column must match that column's result type.
// %ordinal, %wgx = scf.lookup %variant {
//   0 = (%gemm0, %x0),
//   1 = (%gemm1, %x1)
// } default(%fallback, %xf) : index, index
LOOM_DEFINE_ISA(loom_scf_lookup_isa, LOOM_OP_SCF_LOOKUP)
LOOM_DEFINE_OPERAND(loom_scf_lookup_selector, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_scf_lookup_values, 1)
LOOM_DEFINE_VARIADIC_RESULTS(loom_scf_lookup_results, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_scf_lookup_case_keys, 0)
iree_status_t loom_scf_lookup_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t selector,
    const int64_t* case_keys,
    iree_host_size_t case_keys_count,
    loom_may_consume const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scf_lookup_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scf_lookup_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_scf_lookup_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SCF_CONDITION: Terminates the before region of scf.while with a scalar i1 continuation condition and the values forwarded to the after region.
// scf.condition %keep_going : i1
LOOM_DEFINE_ISA(loom_scf_condition_isa, LOOM_OP_SCF_CONDITION)
LOOM_DEFINE_OPERAND(loom_scf_condition_condition, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_scf_condition_forwarded, 1)
iree_status_t loom_scf_condition_build(
    loom_builder_t* builder,
    loom_value_id_t condition,
    const loom_value_id_t* forwarded,
    iree_host_size_t forwarded_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_SCF_WHILE: Unbounded loop with explicit before and after regions. The before region terminates with scf.condition, and the after region terminates with scf.yield.
// scf.while {
//   scf.condition %cond : i1
// } do {
//   scf.yield
// }
LOOM_DEFINE_ISA(loom_scf_while_isa, LOOM_OP_SCF_WHILE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_scf_while_iter_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_scf_while_results, 0)
LOOM_DEFINE_REGION(loom_scf_while_before, 0)
LOOM_DEFINE_REGION(loom_scf_while_after, 1)
iree_status_t loom_scf_while_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* iter_args,
    iree_host_size_t iter_args_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scf_while_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the scf dialect.
const loom_op_vtable_t* const* loom_scf_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the scf dialect.
const loom_op_semantics_t* loom_scf_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a scf op kind, or empty metadata.
loom_op_semantics_t loom_scf_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_SCF_OPS_H_
