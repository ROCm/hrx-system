// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_TEMPLATE_OPS_H_
#define LOOM_OPS_TEMPLATE_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_TEMPLATE_DECL = LOOM_OP_KIND(LOOM_DIALECT_TEMPLATE, 0),
  LOOM_OP_TEMPLATE_DEF = LOOM_OP_KIND(LOOM_DIALECT_TEMPLATE, 1),
  LOOM_OP_TEMPLATE_UKERNEL = LOOM_OP_KIND(LOOM_DIALECT_TEMPLATE, 2),
  LOOM_OP_TEMPLATE_APPLY = LOOM_OP_KIND(LOOM_DIALECT_TEMPLATE, 3),
  LOOM_OP_TEMPLATE_CALL = LOOM_OP_KIND(LOOM_DIALECT_TEMPLATE, 4),
  LOOM_OP_TEMPLATE_RETURN = LOOM_OP_KIND(LOOM_DIALECT_TEMPLATE, 5),
  LOOM_OP_TEMPLATE_COUNT_ = 6,
};

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_template_visibility_e {
  LOOM_TEMPLATE_VISIBILITY_PUBLIC = 1,
  LOOM_TEMPLATE_VISIBILITY_COUNT_ = 2,
} loom_template_visibility_t;

// Function calling convention. Absent (0) means host.
typedef enum loom_template_cc_e {
  LOOM_TEMPLATE_CC_HOST = 1,
  LOOM_TEMPLATE_CC_DEVICE = 2,
  LOOM_TEMPLATE_CC_INITIALIZER = 3,
  LOOM_TEMPLATE_CC_DEINITIALIZER = 4,
  LOOM_TEMPLATE_CC_COUNT_ = 5,
} loom_template_cc_t;

// Function purity. Absent (0) means unspecified (conservative).
typedef enum loom_template_purity_e {
  LOOM_TEMPLATE_PURITY_PURE = 1,
  LOOM_TEMPLATE_PURITY_COUNT_ = 2,
} loom_template_purity_t;

// Execution temperature hint. Absent (0) means unspecified.
typedef enum loom_template_temperature_e {
  LOOM_TEMPLATE_TEMPERATURE_HOT = 1,
  LOOM_TEMPLATE_TEMPERATURE_COLD = 2,
  LOOM_TEMPLATE_TEMPERATURE_COUNT_ = 3,
} loom_template_temperature_t;

// Private symbol retention policy. Absent (0) permits ordinary DCE.
typedef enum loom_template_retain_e {
  LOOM_TEMPLATE_RETAIN_RETAIN = 1,
  LOOM_TEMPLATE_RETAIN_COUNT_ = 2,
} loom_template_retain_t;

// LOOM_OP_TEMPLATE_DECL: Abstract compile-time callable-family declaration. The declaration owns the stable signature and coarse applicability contract.
// template.decl public @vector_transform(%value: vector<32xf32>) -> (vector<32xf32>)
LOOM_DEFINE_ISA(loom_template_decl_isa, LOOM_OP_TEMPLATE_DECL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_template_decl_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_template_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_decl_family, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_decl_visibility, 1, loom_template_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_decl_cc, 2, loom_template_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_decl_purity, 3, loom_template_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_decl_temperature, 4, loom_template_temperature_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_template_decl_predicates, 5)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_decl_target, 6)
LOOM_DEFINE_ATTR_PARAMETERIZED_ARRAY(loom_template_decl_requires, 7)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_decl_retain, 8, loom_template_retain_t)
enum loom_template_decl_build_flag_bits_e {
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_CC = 1u << 2,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_PURITY = 1u << 3,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 4,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_TARGET = 1u << 5,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_REQUIRES = 1u << 6,
  LOOM_TEMPLATE_DECL_BUILD_FLAG_HAS_PREDICATES = 1u << 7,
};
typedef uint32_t loom_template_decl_build_flags_t;
iree_status_t loom_template_decl_build(
    loom_builder_t* builder,
    loom_template_decl_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional loom_symbol_ref_t target,
    loom_optional loom_parameterized_attr_array_t requires_,
    loom_symbol_ref_t family,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_template_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEMPLATE_DEF: Constraint-matched bodyful implementation of a template family.
// template.def<@vector_transform> device priority(20) @vector_transform_fast(%value: vector<32xf32>) -> (vector<32xf32>) {
//   template.return %value : vector<32xf32>
// }
LOOM_DEFINE_ISA(loom_template_def_isa, LOOM_OP_TEMPLATE_DEF)
LOOM_DEFINE_VARIADIC_RESULTS(loom_template_def_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_def_family, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_def_implementation, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_def_visibility, 2, loom_template_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_def_cc, 3, loom_template_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_def_purity, 4, loom_template_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_def_temperature, 5, loom_template_temperature_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_template_def_predicates, 6)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_def_target, 7)
LOOM_DEFINE_ATTR_PARAMETERIZED_ARRAY(loom_template_def_requires, 8)
LOOM_DEFINE_ATTR_I64(loom_template_def_priority, 9)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_def_retain, 10, loom_template_retain_t)
LOOM_DEFINE_REGION(loom_template_def_body, 0)
enum loom_template_def_build_flag_bits_e {
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_CC = 1u << 2,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_PURITY = 1u << 3,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_TEMPERATURE = 1u << 4,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_TARGET = 1u << 5,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_PRIORITY = 1u << 6,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_REQUIRES = 1u << 7,
  LOOM_TEMPLATE_DEF_BUILD_FLAG_HAS_PREDICATES = 1u << 8,
};
typedef uint32_t loom_template_def_build_flags_t;
iree_status_t loom_template_def_build(
    loom_builder_t* builder,
    loom_template_def_build_flags_t build_flags,
    loom_symbol_ref_t family,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional loom_symbol_ref_t target,
    loom_optional loom_parameterized_attr_array_t requires_,
    loom_optional int64_t priority,
    loom_symbol_ref_t implementation,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_template_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEMPLATE_UKERNEL: Constraint-matched opaque implementation of a template family.
// template.ukernel<@vector_transform> device priority(10) @vector_transform_asm(%value: vector<32xf32>) -> (vector<32xf32>)
LOOM_DEFINE_ISA(loom_template_ukernel_isa, LOOM_OP_TEMPLATE_UKERNEL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_template_ukernel_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_template_ukernel_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_ukernel_family, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_ukernel_implementation, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_ukernel_visibility, 2, loom_template_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_ukernel_cc, 3, loom_template_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_ukernel_purity, 4, loom_template_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_ukernel_temperature, 5, loom_template_temperature_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_template_ukernel_predicates, 6)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_ukernel_target, 7)
LOOM_DEFINE_ATTR_PARAMETERIZED_ARRAY(loom_template_ukernel_requires, 8)
LOOM_DEFINE_ATTR_I64(loom_template_ukernel_priority, 9)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_ukernel_retain, 10, loom_template_retain_t)
enum loom_template_ukernel_build_flag_bits_e {
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_CC = 1u << 2,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_PURITY = 1u << 3,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 4,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_TARGET = 1u << 5,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_PRIORITY = 1u << 6,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_REQUIRES = 1u << 7,
  LOOM_TEMPLATE_UKERNEL_BUILD_FLAG_HAS_PREDICATES = 1u << 8,
};
typedef uint32_t loom_template_ukernel_build_flags_t;
iree_status_t loom_template_ukernel_build(
    loom_builder_t* builder,
    loom_template_ukernel_build_flags_t build_flags,
    loom_symbol_ref_t family,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional loom_symbol_ref_t target,
    loom_optional loom_parameterized_attr_array_t requires_,
    loom_optional int64_t priority,
    loom_symbol_ref_t implementation,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_template_ukernel_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEMPLATE_APPLY: Compile-time family application. A specializing link selects one applicable implementation before executable lowering. In a command program this is source composition whose selected implementation is inlined before portable command preparation.
// %result = template.apply<@vector_transform>(%value) : (vector<32xf32>) -> (vector<32xf32>)
LOOM_DEFINE_ISA(loom_template_apply_isa, LOOM_OP_TEMPLATE_APPLY)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_template_apply_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_template_apply_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_apply_family, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_apply_purity, 1, loom_template_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_apply_temperature, 2, loom_template_temperature_t)
enum loom_template_apply_build_flag_bits_e {
  LOOM_TEMPLATE_APPLY_BUILD_FLAG_HAS_PURITY = 1u << 0,
  LOOM_TEMPLATE_APPLY_BUILD_FLAG_HAS_TEMPERATURE = 1u << 1,
};
typedef uint32_t loom_template_apply_build_flags_t;
iree_status_t loom_template_apply_build(
    loom_builder_t* builder,
    loom_template_apply_build_flags_t build_flags,
    loom_symbol_ref_t family,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_template_apply_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
loom_trait_flags_t loom_template_apply_effective_traits(const loom_op_t* op);
iree_status_t loom_template_apply_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEMPLATE_CALL: Exact compile-time implementation call. This bypasses candidate ranking but still checks the implementation contract. In a command program the implementation is inlined before portable command preparation.
// %result = template.call @vector_transform_fast(%value) : (vector<32xf32>) -> (vector<32xf32>)
LOOM_DEFINE_ISA(loom_template_call_isa, LOOM_OP_TEMPLATE_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_template_call_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_template_call_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_template_call_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_call_purity, 1, loom_template_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_template_call_temperature, 2, loom_template_temperature_t)
enum loom_template_call_build_flag_bits_e {
  LOOM_TEMPLATE_CALL_BUILD_FLAG_HAS_PURITY = 1u << 0,
  LOOM_TEMPLATE_CALL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 1,
};
typedef uint32_t loom_template_call_build_flags_t;
iree_status_t loom_template_call_build(
    loom_builder_t* builder,
    loom_template_call_build_flags_t build_flags,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_template_call_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
loom_trait_flags_t loom_template_call_effective_traits(const loom_op_t* op);
iree_status_t loom_template_call_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TEMPLATE_RETURN: Return values from a template implementation body.
// template.return
LOOM_DEFINE_ISA(loom_template_return_isa, LOOM_OP_TEMPLATE_RETURN)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_template_return_operands, 0)
iree_status_t loom_template_return_build(
    loom_builder_t* builder,
    const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the template dialect.
const loom_op_vtable_t* const* loom_template_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the template dialect.
const loom_op_semantics_t* loom_template_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a template op kind, or empty metadata.
loom_op_semantics_t loom_template_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TEMPLATE_OPS_H_
