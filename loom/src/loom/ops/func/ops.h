// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_FUNC_OPS_H_
#define LOOM_OPS_FUNC_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_FUNC_DEF = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 0),
  LOOM_OP_FUNC_DECL = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 1),
  LOOM_OP_FUNC_TEMPLATE = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 2),
  LOOM_OP_FUNC_UKERNEL = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 3),
  LOOM_OP_FUNC_CALL = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 4),
  LOOM_OP_FUNC_APPLY = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 5),
  LOOM_OP_FUNC_RETURN = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 6),
  LOOM_OP_FUNC_COUNT_ = 7,
};

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_func_visibility_e {
  LOOM_FUNC_VISIBILITY_PUBLIC = 1,
  LOOM_FUNC_VISIBILITY_COUNT_ = 2,
} loom_func_visibility_t;

// Function calling convention. Absent (0) means host.
typedef enum loom_func_cc_e {
  LOOM_FUNC_CC_HOST = 1,
  LOOM_FUNC_CC_DEVICE = 2,
  LOOM_FUNC_CC_INITIALIZER = 3,
  LOOM_FUNC_CC_DEINITIALIZER = 4,
  LOOM_FUNC_CC_COUNT_ = 5,
} loom_func_cc_t;

// Function purity. Absent (0) means unspecified (conservative).
typedef enum loom_func_purity_e {
  LOOM_FUNC_PURITY_PURE = 1,
  LOOM_FUNC_PURITY_COUNT_ = 2,
} loom_func_purity_t;

// Execution temperature hint. Absent (0) means unspecified.
typedef enum loom_func_temperature_e {
  LOOM_FUNC_TEMPERATURE_HOT = 1,
  LOOM_FUNC_TEMPERATURE_COLD = 2,
  LOOM_FUNC_TEMPERATURE_COUNT_ = 3,
} loom_func_temperature_t;

// Author inline policy. Absent (0) leaves the edge to the current pass.
typedef enum loom_func_inline_policy_e {
  LOOM_FUNC_INLINE_POLICY_INLINE = 1,
  LOOM_FUNC_INLINE_POLICY_NOINLINE = 2,
  LOOM_FUNC_INLINE_POLICY_COUNT_ = 3,
} loom_func_inline_policy_t;

// Private symbol retention policy. Absent (0) permits ordinary DCE.
typedef enum loom_func_retain_e {
  LOOM_FUNC_RETAIN_RETAIN = 1,
  LOOM_FUNC_RETAIN_COUNT_ = 2,
} loom_func_retain_t;

// LOOM_OP_FUNC_DEF: Function definition. Callable by name via func.call.
// func.def @negate(%input: f32) -> (f32) {
//   func.return %input : f32
// }
LOOM_DEFINE_ISA(loom_func_def_isa, LOOM_OP_FUNC_DEF)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_def_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_def_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_visibility, 1, loom_func_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_cc, 2, loom_func_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_purity, 3, loom_func_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_temperature, 4, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_inline_policy, 5, loom_func_inline_policy_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_func_def_predicates, 6)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_def_target, 7)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_abi, 8, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_func_def_abi_attrs, 9)
LOOM_DEFINE_ATTR_STRING(loom_func_def_export_symbol, 10)
LOOM_DEFINE_ATTR_DICT(loom_func_def_export_attrs, 11)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_retain, 12, loom_func_retain_t)
LOOM_DEFINE_REGION(loom_func_def_body, 0)
enum loom_func_def_build_flag_bits_e {
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC = 1u << 2,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY = 1u << 3,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_TEMPERATURE = 1u << 4,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 5,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET = 1u << 6,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_ABI = 1u << 7,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 8,
};
typedef uint32_t loom_func_def_build_flags_t;
iree_status_t loom_func_def_build(
    loom_builder_t* builder,
    loom_func_def_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_optional loom_symbol_ref_t target,
    loom_optional uint8_t abi,
    loom_optional loom_named_attr_slice_t abi_attrs,
    loom_optional loom_string_id_t export_symbol,
    loom_optional loom_named_attr_slice_t export_attrs,
    loom_symbol_ref_t callee,
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
iree_status_t loom_func_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_FUNC_DECL: External function declaration. Callable by name via func.call.
// func.decl @extern_matmul(%a: tensor<[%M]xf32>, %b: tensor<[%K]xf32>) -> (tensor<[%M]xf32>)
LOOM_DEFINE_ISA(loom_func_decl_isa, LOOM_OP_FUNC_DECL)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_decl_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_visibility, 1, loom_func_visibility_t)
LOOM_DEFINE_ATTR_STRING(loom_func_decl_import_module, 2)
LOOM_DEFINE_ATTR_STRING(loom_func_decl_import_symbol, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_cc, 4, loom_func_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_purity, 5, loom_func_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_temperature, 6, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_inline_policy, 7, loom_func_inline_policy_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_decl_target, 8)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_abi, 9, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_func_decl_abi_attrs, 10)
LOOM_DEFINE_ATTR_STRING(loom_func_decl_export_symbol, 11)
LOOM_DEFINE_ATTR_DICT(loom_func_decl_export_attrs, 12)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_func_decl_predicates, 13)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_retain, 14, loom_func_retain_t)
enum loom_func_decl_build_flag_bits_e {
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE = 1u << 2,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_SYMBOL = 1u << 3,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_CC = 1u << 4,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_PURITY = 1u << 5,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 6,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 7,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_TARGET = 1u << 8,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_ABI = 1u << 9,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 10,
};
typedef uint32_t loom_func_decl_build_flags_t;
iree_status_t loom_func_decl_build(
    loom_builder_t* builder,
    loom_func_decl_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional loom_string_id_t import_module,
    loom_optional loom_string_id_t import_symbol,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_optional loom_symbol_ref_t target,
    loom_optional uint8_t abi,
    loom_optional loom_named_attr_slice_t abi_attrs,
    loom_optional loom_string_id_t export_symbol,
    loom_optional loom_named_attr_slice_t export_attrs,
    loom_symbol_ref_t callee,
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
iree_status_t loom_func_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_FUNC_TEMPLATE: Constraint-matched visible implementation of an abstract op.
// func.template<tile.contract> device @vnni_q8(%w: tensor<[%M]xi8>, %x: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)] {
//   func.return %x : tensor<[%K]xf32>
// }
LOOM_DEFINE_ISA(loom_func_template_isa, LOOM_OP_FUNC_TEMPLATE)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_template_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_func_template_implements, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_template_callee, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_template_visibility, 2, loom_func_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_template_cc, 3, loom_func_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_template_purity, 4, loom_func_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_template_temperature, 5, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_template_inline_policy, 6, loom_func_inline_policy_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_func_template_predicates, 7)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_template_target, 8)
LOOM_DEFINE_ATTR_I64(loom_func_template_priority, 9)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_template_retain, 10, loom_func_retain_t)
LOOM_DEFINE_REGION(loom_func_template_body, 0)
enum loom_func_template_build_flag_bits_e {
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_CC = 1u << 2,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_PURITY = 1u << 3,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_TEMPERATURE = 1u << 4,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 5,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_TARGET = 1u << 6,
  LOOM_FUNC_TEMPLATE_BUILD_FLAG_HAS_PRIORITY = 1u << 7,
};
typedef uint32_t loom_func_template_build_flags_t;
iree_status_t loom_func_template_build(
    loom_builder_t* builder,
    loom_func_template_build_flags_t build_flags,
    loom_string_id_t implements,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_optional loom_symbol_ref_t target,
    loom_optional int64_t priority,
    loom_symbol_ref_t callee,
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

// LOOM_OP_FUNC_UKERNEL: Constraint-matched opaque implementation of an abstract op.
// func.ukernel<tile.contract> device @vnni_q8_asm(%w: tensor<[%M]xi8>, %x: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)]
LOOM_DEFINE_ISA(loom_func_ukernel_isa, LOOM_OP_FUNC_UKERNEL)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_ukernel_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_func_ukernel_implements, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_ukernel_callee, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_ukernel_visibility, 2, loom_func_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_ukernel_cc, 3, loom_func_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_ukernel_purity, 4, loom_func_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_ukernel_temperature, 5, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_ukernel_inline_policy, 6, loom_func_inline_policy_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_func_ukernel_predicates, 7)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_ukernel_target, 8)
LOOM_DEFINE_ATTR_I64(loom_func_ukernel_priority, 9)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_ukernel_retain, 10, loom_func_retain_t)
enum loom_func_ukernel_build_flag_bits_e {
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_CC = 1u << 2,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_PURITY = 1u << 3,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 4,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 5,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_TARGET = 1u << 6,
  LOOM_FUNC_UKERNEL_BUILD_FLAG_HAS_PRIORITY = 1u << 7,
};
typedef uint32_t loom_func_ukernel_build_flags_t;
iree_status_t loom_func_ukernel_build(
    loom_builder_t* builder,
    loom_func_ukernel_build_flags_t build_flags,
    loom_string_id_t implements,
    loom_optional uint8_t visibility,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_optional loom_symbol_ref_t target,
    loom_optional int64_t priority,
    loom_symbol_ref_t callee,
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

// LOOM_OP_FUNC_CALL: Function-like symbol call. Runtime calls target func.def/func.decl; required-inline exact template calls are consumed before executable lowering.
// %r = func.call @add(%a, %b) : (f32, f32) -> (f32)
LOOM_DEFINE_ISA(loom_func_call_isa, LOOM_OP_FUNC_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_func_call_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_call_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_call_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_call_purity, 1, loom_func_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_call_temperature, 2, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_call_inline_policy, 3, loom_func_inline_policy_t)
enum loom_func_call_build_flag_bits_e {
  LOOM_FUNC_CALL_BUILD_FLAG_HAS_PURITY = 1u << 0,
  LOOM_FUNC_CALL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 1,
  LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 2,
};
typedef uint32_t loom_func_call_build_flags_t;
iree_status_t loom_func_call_build(
    loom_builder_t* builder,
    loom_func_call_build_flags_t build_flags,
    loom_optional uint8_t purity,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_func_call_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
loom_trait_flags_t loom_func_call_effective_traits(const loom_op_t* op);

// LOOM_OP_FUNC_APPLY: Compile-time implementation demand. Contract key must be selected before executable lowering.
// %r = func.apply<qwen.q4.matmul>(%w, %x) : (tensor<16x32xi8>, tensor<32xf32>) -> (tensor<16xf32>)
LOOM_DEFINE_ISA(loom_func_apply_isa, LOOM_OP_FUNC_APPLY)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_func_apply_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_apply_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_func_apply_contract, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_apply_purity, 1, loom_func_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_apply_temperature, 2, loom_func_temperature_t)
enum loom_func_apply_build_flag_bits_e {
  LOOM_FUNC_APPLY_BUILD_FLAG_HAS_PURITY = 1u << 0,
  LOOM_FUNC_APPLY_BUILD_FLAG_HAS_TEMPERATURE = 1u << 1,
};
typedef uint32_t loom_func_apply_build_flags_t;
iree_status_t loom_func_apply_build(
    loom_builder_t* builder,
    loom_func_apply_build_flags_t build_flags,
    loom_string_id_t contract,
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
iree_status_t loom_func_apply_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
loom_trait_flags_t loom_func_apply_effective_traits(const loom_op_t* op);

// LOOM_OP_FUNC_RETURN: Return values from function body. Types must match enclosing function's result types.
// func.return
LOOM_DEFINE_ISA(loom_func_return_isa, LOOM_OP_FUNC_RETURN)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_func_return_operands, 0)
iree_status_t loom_func_return_build(
    loom_builder_t* builder,
    const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the func dialect.
const loom_op_vtable_t* const* loom_func_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the func dialect.
const loom_op_semantics_t* loom_func_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a func op kind, or empty metadata.
loom_op_semantics_t loom_func_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_FUNC_OPS_H_
