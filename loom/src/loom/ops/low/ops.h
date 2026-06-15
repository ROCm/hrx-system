// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_LOW_OPS_H_
#define LOOM_OPS_LOW_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_LOW_FUNC_DEF = LOOM_OP_KIND(LOOM_DIALECT_LOW, 0),
  LOOM_OP_LOW_KERNEL_DEF = LOOM_OP_KIND(LOOM_DIALECT_LOW, 1),
  LOOM_OP_LOW_FUNC_DECL = LOOM_OP_KIND(LOOM_DIALECT_LOW, 2),
  LOOM_OP_LOW_RETURN = LOOM_OP_KIND(LOOM_DIALECT_LOW, 3),
  LOOM_OP_LOW_FUNC_CALL = LOOM_OP_KIND(LOOM_DIALECT_LOW, 4),
  LOOM_OP_LOW_OP = LOOM_OP_KIND(LOOM_DIALECT_LOW, 5),
  LOOM_OP_LOW_CONST = LOOM_OP_KIND(LOOM_DIALECT_LOW, 6),
  LOOM_OP_LOW_COPY = LOOM_OP_KIND(LOOM_DIALECT_LOW, 7),
  LOOM_OP_LOW_SLICE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 8),
  LOOM_OP_LOW_CONCAT = LOOM_OP_KIND(LOOM_DIALECT_LOW, 9),
  LOOM_OP_LOW_INVOKE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 10),
  LOOM_OP_LOW_STORAGE_RESERVE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 11),
  LOOM_OP_LOW_STORAGE_VIEW = LOOM_OP_KIND(LOOM_DIALECT_LOW, 12),
  LOOM_OP_LOW_SPILL = LOOM_OP_KIND(LOOM_DIALECT_LOW, 13),
  LOOM_OP_LOW_RELOAD = LOOM_OP_KIND(LOOM_DIALECT_LOW, 14),
  LOOM_OP_LOW_STORAGE_ADDRESS = LOOM_OP_KIND(LOOM_DIALECT_LOW, 15),
  LOOM_OP_LOW_BR = LOOM_OP_KIND(LOOM_DIALECT_LOW, 16),
  LOOM_OP_LOW_COND_BR = LOOM_OP_KIND(LOOM_DIALECT_LOW, 17),
  LOOM_OP_LOW_RESOURCE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 18),
  LOOM_OP_LOW_LIVE_IN = LOOM_OP_KIND(LOOM_DIALECT_LOW, 19),
  LOOM_OP_LOW_SCF_YIELD = LOOM_OP_KIND(LOOM_DIALECT_LOW, 20),
  LOOM_OP_LOW_SCF_IF = LOOM_OP_KIND(LOOM_DIALECT_LOW, 21),
  LOOM_OP_LOW_SCF_FOR = LOOM_OP_KIND(LOOM_DIALECT_LOW, 22),
  LOOM_OP_LOW_COUNT_ = 23,
};

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_low_visibility_e {
  LOOM_LOW_VISIBILITY_PUBLIC = 1,
  LOOM_LOW_VISIBILITY_COUNT_ = 2,
} loom_low_visibility_t;

// Function calling convention. Absent (0) means host.
typedef enum loom_low_cc_e {
  LOOM_LOW_CC_HOST = 1,
  LOOM_LOW_CC_DEVICE = 2,
  LOOM_LOW_CC_INITIALIZER = 3,
  LOOM_LOW_CC_DEINITIALIZER = 4,
  LOOM_LOW_CC_COUNT_ = 5,
} loom_low_cc_t;

// Function purity. Absent (0) means unspecified (conservative).
typedef enum loom_low_purity_e {
  LOOM_LOW_PURITY_PURE = 1,
  LOOM_LOW_PURITY_COUNT_ = 2,
} loom_low_purity_t;

// Register allocation exactness mode for a low function. Absent means virtual.
typedef enum loom_low_allocation_e {
  LOOM_LOW_ALLOCATION_VIRTUAL = 1,
  LOOM_LOW_ALLOCATION_ASSIGNED = 2,
  LOOM_LOW_ALLOCATION_FIXED = 3,
  LOOM_LOW_ALLOCATION_COUNT_ = 4,
} loom_low_allocation_t;

// Instruction scheduling exactness mode for a low function. Absent means free.
typedef enum loom_low_schedule_e {
  LOOM_LOW_SCHEDULE_FREE = 1,
  LOOM_LOW_SCHEDULE_CONSTRAINED = 2,
  LOOM_LOW_SCHEDULE_LOCKED = 3,
  LOOM_LOW_SCHEDULE_COUNT_ = 4,
} loom_low_schedule_t;

// External code source kind for an imported low function declaration.
typedef enum loom_low_func_decl_import_kind_e {
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM = 1,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE = 2,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_ROCASM = 3,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_OBJECT = 4,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_COUNT_ = 5,
} loom_low_func_decl_import_kind_t;

// Target-provided ABI resource imported into a low function body.
typedef enum loom_low_resource_import_kind_e {
  LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER = 1,
  LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE = 2,
  LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT = 3,
  LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING = 4,
  LOOM_LOW_RESOURCE_IMPORT_KIND_COUNT_ = 5,
} loom_low_resource_import_kind_t;

// Local low.scf.for unroll policy.
typedef enum loom_low_scf_for_unroll_policy_e {
  LOOM_LOW_SCF_FOR_UNROLL_POLICY_UNROLL = 1,
  LOOM_LOW_SCF_FOR_UNROLL_POLICY_COUNT_ = 2,
} loom_low_scf_for_unroll_policy_t;

// LOOM_OP_LOW_FUNC_DEF: Target-bound low function definition with register-typed signature values.
// low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {
//   %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>
//   low.return %sum : reg<amdgpu.vgpr x1>
// }
LOOM_DEFINE_ISA(loom_low_func_def_isa, LOOM_OP_LOW_FUNC_DEF)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_func_def_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_def_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_def_target, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_abi, 2, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_low_func_def_abi_attrs, 3)
LOOM_DEFINE_ATTR_DICT(loom_low_func_def_abi_layout, 4)
LOOM_DEFINE_ATTR_STRING(loom_low_func_def_export_symbol, 5)
LOOM_DEFINE_ATTR_DICT(loom_low_func_def_export_attrs, 6)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_visibility, 7, loom_low_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_cc, 8, loom_low_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_purity, 9, loom_low_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_allocation, 10, loom_low_allocation_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_schedule, 11, loom_low_schedule_t)
LOOM_DEFINE_REGION(loom_low_func_def_body, 0)
enum loom_low_func_def_build_flag_bits_e {
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_CC = 1u << 1,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PURITY = 1u << 2,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ALLOCATION = 1u << 3,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_SCHEDULE = 1u << 4,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI = 1u << 5,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 6,
};
typedef uint32_t loom_low_func_def_build_flags_t;
iree_status_t loom_low_func_def_build(
    loom_builder_t* builder,
    loom_low_func_def_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t allocation,
    loom_optional uint8_t schedule,
    loom_symbol_ref_t target,
    loom_optional uint8_t abi,
    loom_optional loom_named_attr_slice_t abi_attrs,
    loom_optional loom_named_attr_slice_t abi_layout,
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
iree_status_t loom_low_func_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_KERNEL_DEF: Target-bound low kernel entry with register-typed launch ABI values. Helper calls stay in low.func.def; kernel launch/export contracts live on this entry op.
// low.kernel.def target(@gfx1100) export("matmul") workgroup_size(16, 4, 1) @matmul(%lhs: reg<amdgpu.sgpr x4>, %rhs: reg<amdgpu.sgpr x4>, %out: reg<amdgpu.sgpr x4>) {
//   low.return
// }
LOOM_DEFINE_ISA(loom_low_kernel_def_isa, LOOM_OP_LOW_KERNEL_DEF)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_kernel_def_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_kernel_def_target, 1)
LOOM_DEFINE_ATTR_DICT(loom_low_kernel_def_abi_layout, 2)
LOOM_DEFINE_ATTR_STRING(loom_low_kernel_def_export_symbol, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_kernel_def_export_linkage, 4, loom_target_linkage_t)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_workgroup_size_x, 5)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_workgroup_size_y, 6)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_workgroup_size_z, 7)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_kernel_def_allocation, 8, loom_low_allocation_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_kernel_def_schedule, 9, loom_low_schedule_t)
LOOM_DEFINE_REGION(loom_low_kernel_def_body, 0)
enum loom_low_kernel_def_build_flag_bits_e {
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_ALLOCATION = 1u << 0,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_SCHEDULE = 1u << 1,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 2,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_LINKAGE = 1u << 3,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_X = 1u << 4,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Y = 1u << 5,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Z = 1u << 6,
};
typedef uint32_t loom_low_kernel_def_build_flags_t;
iree_status_t loom_low_kernel_def_build(
    loom_builder_t* builder,
    loom_low_kernel_def_build_flags_t build_flags,
    loom_optional uint8_t allocation,
    loom_optional uint8_t schedule,
    loom_symbol_ref_t target,
    loom_optional loom_named_attr_slice_t abi_layout,
    loom_optional loom_string_id_t export_symbol,
    loom_optional uint8_t export_linkage,
    loom_optional int64_t workgroup_size_x,
    loom_optional int64_t workgroup_size_y,
    loom_optional int64_t workgroup_size_z,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_kernel_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_FUNC_DECL: Target-bound low function declaration with register-typed signature values.
// low.func.decl target(@gfx1100) @extern_add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)
LOOM_DEFINE_ISA(loom_low_func_decl_isa, LOOM_OP_LOW_FUNC_DECL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_func_decl_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_func_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_decl_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_decl_target, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_abi, 2, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_low_func_decl_abi_attrs, 3)
LOOM_DEFINE_ATTR_DICT(loom_low_func_decl_abi_layout, 4)
LOOM_DEFINE_ATTR_STRING(loom_low_func_decl_export_symbol, 5)
LOOM_DEFINE_ATTR_DICT(loom_low_func_decl_export_attrs, 6)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_visibility, 7, loom_low_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_cc, 8, loom_low_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_purity, 9, loom_low_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_allocation, 10, loom_low_allocation_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_schedule, 11, loom_low_schedule_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_import_kind, 13, loom_low_func_decl_import_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_low_func_decl_code_symbol, 14)
enum loom_low_func_decl_build_flag_bits_e {
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CC = 1u << 1,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_PURITY = 1u << 2,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ALLOCATION = 1u << 3,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_SCHEDULE = 1u << 4,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_KIND = 1u << 5,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CODE_SYMBOL = 1u << 6,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI = 1u << 7,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 8,
};
typedef uint32_t loom_low_func_decl_build_flags_t;
iree_status_t loom_low_func_decl_build(
    loom_builder_t* builder,
    loom_low_func_decl_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t allocation,
    loom_optional uint8_t schedule,
    loom_optional uint8_t import_kind,
    loom_optional loom_string_id_t code_symbol,
    loom_symbol_ref_t target,
    loom_optional uint8_t abi,
    loom_optional loom_named_attr_slice_t abi_attrs,
    loom_optional loom_named_attr_slice_t abi_layout,
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
iree_status_t loom_low_func_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_RETURN: Return register values from a low function.
// low.return
LOOM_DEFINE_ISA(loom_low_return_isa, LOOM_OP_LOW_RETURN)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_return_values, 0)
iree_status_t loom_low_return_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_LOW_FUNC_CALL: Direct call from one low function body to another same-target low function.
// %result = low.func.call @extern_add(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)
LOOM_DEFINE_ISA(loom_low_func_call_isa, LOOM_OP_LOW_FUNC_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_func_call_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_func_call_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_call_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_call_purity, 1, loom_low_purity_t)
enum loom_low_func_call_build_flag_bits_e {
  LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY = 1u << 0,
};
typedef uint32_t loom_low_func_call_build_flags_t;
iree_status_t loom_low_func_call_build(
    loom_builder_t* builder,
    loom_low_func_call_build_flags_t build_flags,
    loom_optional uint8_t purity,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
loom_trait_flags_t loom_low_func_call_effective_traits(const loom_op_t* op);
iree_status_t loom_low_func_call_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_OP: Descriptor-backed target instruction over virtual registers.
// %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>
LOOM_DEFINE_ISA(loom_low_op_isa, LOOM_OP_LOW_OP)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_op_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_op_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_low_op_opcode, 0)
LOOM_DEFINE_ATTR_I64(loom_low_op_descriptor_ordinal, 1)
LOOM_DEFINE_ATTR_DICT(loom_low_op_attrs, 2)
iree_status_t loom_low_op_build(
    loom_builder_t* builder,
    loom_string_id_t opcode,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    loom_optional loom_named_attr_slice_t attrs,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_CONST: Descriptor-backed constant or immediate materialization into a register.
// %c0 = low.const<amdgpu.s_mov_b32> {imm = 0} : reg<amdgpu.sgpr x1>
LOOM_DEFINE_ISA(loom_low_const_isa, LOOM_OP_LOW_CONST)
LOOM_DEFINE_RESULT(loom_low_const_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_low_const_opcode, 0)
LOOM_DEFINE_ATTR_I64(loom_low_const_descriptor_ordinal, 1)
LOOM_DEFINE_ATTR_DICT(loom_low_const_attrs, 2)
iree_status_t loom_low_const_build(
    loom_builder_t* builder,
    loom_string_id_t opcode,
    loom_optional loom_named_attr_slice_t attrs,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_const_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_low_const_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_COPY: Explicit virtual-register copy used by lowering and allocation. Each copy produces a fresh virtual-register identity.
// %copy = low.copy %value : reg<amdgpu.vgpr x1> -> reg<amdgpu.vgpr x1>
LOOM_DEFINE_ISA(loom_low_copy_isa, LOOM_OP_LOW_COPY)
LOOM_DEFINE_OPERAND(loom_low_copy_source, 0)
LOOM_DEFINE_RESULT(loom_low_copy_result, 0)
iree_status_t loom_low_copy_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_copy_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_low_copy_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SLICE: Project a contiguous subrange from a register-range value.
// %lane = low.slice %quad[2] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr>
LOOM_DEFINE_ISA(loom_low_slice_isa, LOOM_OP_LOW_SLICE)
LOOM_DEFINE_OPERAND(loom_low_slice_source, 0)
LOOM_DEFINE_RESULT(loom_low_slice_result, 0)
LOOM_DEFINE_ATTR_I64(loom_low_slice_offset, 0)
iree_status_t loom_low_slice_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_slice_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_low_slice_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_low_slice_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_CONCAT: Compose one fresh register-range identity from ordered register subranges.
// %pair = low.concat(%lo, %hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
LOOM_DEFINE_ISA(loom_low_concat_isa, LOOM_OP_LOW_CONCAT)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_concat_sources, 0)
LOOM_DEFINE_RESULT(loom_low_concat_result, 0)
iree_status_t loom_low_concat_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* sources,
    iree_host_size_t sources_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_concat_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_LOW_INVOKE: Invoke an explicitly selected translated low function from non-low IR.
// %result = low.invoke @extern_add(%lhs, %rhs) : (i32, i32) -> (i32)
LOOM_DEFINE_ISA(loom_low_invoke_isa, LOOM_OP_LOW_INVOKE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_invoke_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_invoke_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_invoke_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_invoke_purity, 1, loom_low_purity_t)
enum loom_low_invoke_build_flag_bits_e {
  LOOM_LOW_INVOKE_BUILD_FLAG_HAS_PURITY = 1u << 0,
};
typedef uint32_t loom_low_invoke_build_flags_t;
iree_status_t loom_low_invoke_build(
    loom_builder_t* builder,
    loom_low_invoke_build_flags_t build_flags,
    loom_optional uint8_t purity,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
loom_trait_flags_t loom_low_invoke_effective_traits(const loom_op_t* op);
iree_status_t loom_low_invoke_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_STORAGE_RESERVE: Reserve target-low function-local storage and preserve its segment footprint.
// %slot = low.storage.reserve {byte_alignment = 4, byte_length = 16} : low.storage<private>
LOOM_DEFINE_ISA(loom_low_storage_reserve_isa, LOOM_OP_LOW_STORAGE_RESERVE)
LOOM_DEFINE_RESULT(loom_low_storage_reserve_storage, 0)
LOOM_DEFINE_ATTR_I64(loom_low_storage_reserve_byte_length, 0)
LOOM_DEFINE_ATTR_I64(loom_low_storage_reserve_byte_alignment, 1)
iree_status_t loom_low_storage_reserve_build(
    loom_builder_t* builder,
    int64_t byte_length,
    int64_t byte_alignment,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_storage_reserve_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_low_storage_reserve_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_STORAGE_VIEW: Project a byte subspan from function-local storage.
// %tile = low.storage.view %scratch {offset = 128, byte_length = 64} : low.storage<workgroup> -> low.storage<workgroup>
LOOM_DEFINE_ISA(loom_low_storage_view_isa, LOOM_OP_LOW_STORAGE_VIEW)
LOOM_DEFINE_OPERAND(loom_low_storage_view_source, 0)
LOOM_DEFINE_RESULT(loom_low_storage_view_result, 0)
LOOM_DEFINE_ATTR_I64(loom_low_storage_view_offset, 0)
LOOM_DEFINE_ATTR_I64(loom_low_storage_view_byte_length, 1)
iree_status_t loom_low_storage_view_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    int64_t byte_length,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_storage_view_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_low_storage_view_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SPILL: Explicit spill store from a register value into low storage.
// low.spill %value, %slot : reg<amdgpu.vgpr x4>, low.storage<private>
LOOM_DEFINE_ISA(loom_low_spill_isa, LOOM_OP_LOW_SPILL)
LOOM_DEFINE_OPERAND(loom_low_spill_value, 0)
LOOM_DEFINE_OPERAND(loom_low_spill_storage, 1)
LOOM_DEFINE_ATTR_I64(loom_low_spill_offset, 0)
iree_status_t loom_low_spill_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t storage,
    int64_t offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_spill_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_RELOAD: Explicit reload from low storage into a register value.
// %reload = low.reload %slot : low.storage<private> -> reg<amdgpu.vgpr x4>
LOOM_DEFINE_ISA(loom_low_reload_isa, LOOM_OP_LOW_RELOAD)
LOOM_DEFINE_OPERAND(loom_low_reload_storage, 0)
LOOM_DEFINE_RESULT(loom_low_reload_result, 0)
LOOM_DEFINE_ATTR_I64(loom_low_reload_offset, 0)
iree_status_t loom_low_reload_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t storage,
    int64_t offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_reload_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_STORAGE_ADDRESS: Materialize a target address for function-local storage.
// %addr = low.storage.address %slot : low.storage<workgroup> -> reg<amdgpu.vgpr>
LOOM_DEFINE_ISA(loom_low_storage_address_isa, LOOM_OP_LOW_STORAGE_ADDRESS)
LOOM_DEFINE_OPERAND(loom_low_storage_address_storage, 0)
LOOM_DEFINE_RESULT(loom_low_storage_address_result, 0)
LOOM_DEFINE_ATTR_I64(loom_low_storage_address_offset, 0)
iree_status_t loom_low_storage_address_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t storage,
    int64_t offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_storage_address_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_BR: Unconditional branch to a low successor block, forwarding register values.
// low.br ^done
LOOM_DEFINE_ISA(loom_low_br_isa, LOOM_OP_LOW_BR)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_br_args, 0)
LOOM_DEFINE_SUCCESSOR(loom_low_br_dest, 0)
iree_status_t loom_low_br_build(
    loom_builder_t* builder,
    loom_block_t* dest,
    const loom_value_id_t* args,
    iree_host_size_t args_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_br_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_COND_BR: Conditional branch to one of two low successor blocks based on a register predicate.
// low.cond_br %condition, ^then, ^else : reg<vm.i32>
LOOM_DEFINE_ISA(loom_low_cond_br_isa, LOOM_OP_LOW_COND_BR)
LOOM_DEFINE_OPERAND(loom_low_cond_br_condition, 0)
LOOM_DEFINE_SUCCESSOR(loom_low_cond_br_true_dest, 0)
LOOM_DEFINE_SUCCESSOR(loom_low_cond_br_false_dest, 1)
iree_status_t loom_low_cond_br_build(
    loom_builder_t* builder,
    loom_value_id_t condition,
    loom_block_t* true_dest,
    loom_block_t* false_dest,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_cond_br_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_RESOURCE: Import a function-local target resource into a low register value.
// %state = low.resource<vm_state> {index = 0, source_type = i64} : reg<vm.i64>
LOOM_DEFINE_ISA(loom_low_resource_isa, LOOM_OP_LOW_RESOURCE)
LOOM_DEFINE_OPTIONAL_OPERAND(loom_low_resource_extent_value, 0)
LOOM_DEFINE_RESULT(loom_low_resource_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_resource_import_kind, 0, loom_low_resource_import_kind_t)
LOOM_DEFINE_ATTR_I64(loom_low_resource_index, 1)
LOOM_DEFINE_ATTR_TYPE(loom_low_resource_source_type, 2)
LOOM_DEFINE_ATTR_I64(loom_low_resource_extent, 3)
LOOM_DEFINE_ATTR_I64(loom_low_resource_cache_swizzle_stride, 4)
enum loom_low_resource_build_flag_bits_e {
  LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT_VALUE = 1u << 0,
  LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT = 1u << 1,
  LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_CACHE_SWIZZLE_STRIDE = 1u << 2,
};
typedef uint32_t loom_low_resource_build_flags_t;
iree_status_t loom_low_resource_build(
    loom_builder_t* builder,
    loom_low_resource_build_flags_t build_flags,
    loom_low_resource_import_kind_t import_kind,
    loom_optional loom_may_consume loom_value_id_t extent_value,
    int64_t index,
    uint32_t source_type,
    loom_optional int64_t extent,
    loom_optional int64_t cache_swizzle_stride,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_resource_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_LIVE_IN: Import a target-provided ABI live-in register value at low-function entry.
// %kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : reg<amdgpu.sgpr x2>
LOOM_DEFINE_ISA(loom_low_live_in_isa, LOOM_OP_LOW_LIVE_IN)
LOOM_DEFINE_RESULT(loom_low_live_in_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_low_live_in_source, 0)
LOOM_DEFINE_ATTR_I64(loom_low_live_in_source_id, 1)
LOOM_DEFINE_ATTR_DICT(loom_low_live_in_attrs, 2)
iree_status_t loom_low_live_in_build(
    loom_builder_t* builder,
    loom_string_id_t source,
    loom_optional loom_named_attr_slice_t attrs,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_live_in_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SCF_YIELD: Forward register values from a low structured-control region.
// low.scf.yield
LOOM_DEFINE_ISA(loom_low_scf_yield_isa, LOOM_OP_LOW_SCF_YIELD)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_scf_yield_values, 0)
iree_status_t loom_low_scf_yield_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_LOW_SCF_IF: Conditional execution over target-low register values.
// low.scf.if %cond {
//   low.scf.yield
// }
LOOM_DEFINE_ISA(loom_low_scf_if_isa, LOOM_OP_LOW_SCF_IF)
LOOM_DEFINE_OPERAND(loom_low_scf_if_condition, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_scf_if_results, 0)
LOOM_DEFINE_REGION(loom_low_scf_if_then_region, 0)
LOOM_DEFINE_OPTIONAL_REGION(loom_low_scf_if_else_region, 1)
enum loom_low_scf_if_build_flag_bits_e {
  LOOM_LOW_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION = 1u << 0,
};
typedef uint32_t loom_low_scf_if_build_flags_t;
iree_status_t loom_low_scf_if_build(
    loom_builder_t* builder,
    loom_low_scf_if_build_flags_t build_flags,
    loom_may_consume loom_value_id_t condition,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_scf_if_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SCF_FOR: Bounded counted target-low loop with optional loop-carried register state.
// low.scf.for [%lo to %hi step %step] do(%iv: reg<amdgpu.sgpr x1>) {
//   low.scf.yield
// }
LOOM_DEFINE_ISA(loom_low_scf_for_isa, LOOM_OP_LOW_SCF_FOR)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_low_scf_for_lower_bound, 0)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_low_scf_for_upper_bound, 1)
LOOM_DEFINE_SEGMENTED_OPERAND(loom_low_scf_for_step, 2)
LOOM_DEFINE_SEGMENTED_OPERANDS(loom_low_scf_for_iter_args, 3)
LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND(loom_low_scf_for_unroll_factor, 4)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_scf_for_results, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_scf_for_unroll_policy, 0, loom_low_scf_for_unroll_policy_t)
LOOM_DEFINE_REGION(loom_low_scf_for_body, 0)
enum loom_low_scf_for_build_flag_bits_e {
  LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR = 1u << 0,
  LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY = 1u << 1,
};
typedef uint32_t loom_low_scf_for_build_flags_t;
iree_status_t loom_low_scf_for_build(
    loom_builder_t* builder,
    loom_low_scf_for_build_flags_t build_flags,
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
iree_status_t loom_low_scf_for_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the low dialect.
const loom_op_vtable_t* const* loom_low_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the low dialect.
const loom_op_semantics_t* loom_low_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a low op kind, or empty metadata.
loom_op_semantics_t loom_low_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_LOW_OPS_H_
