// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_LLVMIR_OPS_H_
#define LOOM_OPS_LLVMIR_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_LLVMIR_TARGET = LOOM_OP_KIND(LOOM_DIALECT_LLVMIR, 0),
  LOOM_OP_LLVMIR_INLINE_ASM = LOOM_OP_KIND(LOOM_DIALECT_LLVMIR, 1),
  LOOM_OP_LLVMIR_INTRINSIC = LOOM_OP_KIND(LOOM_DIALECT_LLVMIR, 2),
  LOOM_OP_LLVMIR_COUNT_ = 3,
};

// LLVM inline asm call flags.
#define LOOM_LLVMIR_ASMFLAGS_SIDEEFFECT ((uint8_t)1)
#define LOOM_LLVMIR_ASMFLAGS_ALIGNSTACK ((uint8_t)2)
#define LOOM_LLVMIR_ASMFLAGS_INTELDIALECT ((uint8_t)4)

// LLVMIR target row selected by llvmir.target.
typedef enum loom_llvmir_target_kind_e {
  LOOM_LLVMIR_TARGET_KIND_OBJECT = 1,
  LOOM_LLVMIR_TARGET_KIND_COUNT_ = 2,
} loom_llvmir_target_kind_t;

// LOOM_OP_LLVMIR_TARGET: LLVMIR target-family record. The selector chooses an LLVMIR bundle row while optional LLVM-specific attributes own the triple, data layout, CPU, and feature-string vocabulary.
// llvmir.target<object> @llvm_host {triple = "x86_64-unknown-linux-gnu"}
LOOM_DEFINE_ISA(loom_llvmir_target_isa, LOOM_OP_LLVMIR_TARGET)
LOOM_DEFINE_ATTR_SYMBOL(loom_llvmir_target_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_llvmir_target_kind, 1, loom_llvmir_target_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_llvmir_target_codegen_format, 2, loom_target_codegen_format_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_llvmir_target_artifact_format, 3, loom_target_artifact_format_t)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_default_pointer_bitwidth, 4)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_index_bitwidth, 5)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_offset_bitwidth, 6)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_workgroup_size_x, 7)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_workgroup_size_y, 8)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_workgroup_size_z, 9)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_flat_workgroup_size, 10)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_subgroup_size, 11)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_grid_size_x, 12)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_grid_size_y, 13)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_grid_size_z, 14)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_flat_grid_size, 15)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_workgroup_count_x, 16)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_workgroup_count_y, 17)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_max_workgroup_count_z, 18)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_generic, 19)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_global, 20)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_workgroup, 21)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_constant, 22)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_private, 23)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_host, 24)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_memory_space_descriptor, 25)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_llvmir_target_abi, 26, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_target_export_symbol, 27)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_llvmir_target_linkage, 28, loom_target_linkage_t)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_hal_buffer_resource_flags, 29)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_target_contract_set_key, 30)
LOOM_DEFINE_ATTR_I64(loom_llvmir_target_contract_feature_bits, 31)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_target_triple, 32)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_target_data_layout, 33)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_target_cpu, 34)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_target_features, 35)
enum loom_llvmir_target_build_flag_bits_e {
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT = UINT64_C(1) << 0,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT = UINT64_C(1) << 1,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH = UINT64_C(1) << 2,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH = UINT64_C(1) << 3,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH = UINT64_C(1) << 4,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X = UINT64_C(1) << 5,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y = UINT64_C(1) << 6,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z = UINT64_C(1) << 7,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE = UINT64_C(1) << 8,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE = UINT64_C(1) << 9,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X = UINT64_C(1) << 10,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y = UINT64_C(1) << 11,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z = UINT64_C(1) << 12,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE = UINT64_C(1) << 13,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X = UINT64_C(1) << 14,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y = UINT64_C(1) << 15,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z = UINT64_C(1) << 16,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC = UINT64_C(1) << 17,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL = UINT64_C(1) << 18,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP = UINT64_C(1) << 19,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT = UINT64_C(1) << 20,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE = UINT64_C(1) << 21,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST = UINT64_C(1) << 22,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR = UINT64_C(1) << 23,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_ABI = UINT64_C(1) << 24,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL = UINT64_C(1) << 25,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_LINKAGE = UINT64_C(1) << 26,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_HAL_BUFFER_RESOURCE_FLAGS = UINT64_C(1) << 27,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY = UINT64_C(1) << 28,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS = UINT64_C(1) << 29,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_DATA_LAYOUT = UINT64_C(1) << 30,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_CPU = UINT64_C(1) << 31,
  LOOM_LLVMIR_TARGET_BUILD_FLAG_HAS_FEATURES = UINT64_C(1) << 32,
};
typedef uint64_t loom_llvmir_target_build_flags_t;
iree_status_t loom_llvmir_target_build(
    loom_builder_t* builder,
    loom_llvmir_target_build_flags_t build_flags,
    loom_llvmir_target_kind_t kind,
    loom_symbol_ref_t symbol,
    loom_optional uint8_t codegen_format,
    loom_optional uint8_t artifact_format,
    loom_optional int64_t default_pointer_bitwidth,
    loom_optional int64_t index_bitwidth,
    loom_optional int64_t offset_bitwidth,
    loom_optional int64_t max_workgroup_size_x,
    loom_optional int64_t max_workgroup_size_y,
    loom_optional int64_t max_workgroup_size_z,
    loom_optional int64_t max_flat_workgroup_size,
    loom_optional int64_t subgroup_size,
    loom_optional int64_t max_grid_size_x,
    loom_optional int64_t max_grid_size_y,
    loom_optional int64_t max_grid_size_z,
    loom_optional int64_t max_flat_grid_size,
    loom_optional int64_t max_workgroup_count_x,
    loom_optional int64_t max_workgroup_count_y,
    loom_optional int64_t max_workgroup_count_z,
    loom_optional int64_t memory_space_generic,
    loom_optional int64_t memory_space_global,
    loom_optional int64_t memory_space_workgroup,
    loom_optional int64_t memory_space_constant,
    loom_optional int64_t memory_space_private,
    loom_optional int64_t memory_space_host,
    loom_optional int64_t memory_space_descriptor,
    loom_optional uint8_t abi,
    loom_optional loom_string_id_t export_symbol,
    loom_optional uint8_t linkage,
    loom_optional int64_t hal_buffer_resource_flags,
    loom_optional loom_string_id_t contract_set_key,
    loom_optional int64_t contract_feature_bits,
    loom_string_id_t triple,
    loom_optional loom_string_id_t data_layout,
    loom_optional loom_string_id_t cpu,
    loom_optional loom_string_id_t features,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LLVMIR_INLINE_ASM: Structured LLVM inline assembly call. The asm template and constraint strings use LLVM inline asm syntax; operands/results remain ordinary typed Loom SSA values.
// %sum = llvmir.inline_asm<sideeffect> "addl $2, $0", "=r,r,r"(%lhs, %rhs) : (i32, i32) -> i32
LOOM_DEFINE_ISA(loom_llvmir_inline_asm_isa, LOOM_OP_LLVMIR_INLINE_ASM)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_llvmir_inline_asm_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_llvmir_inline_asm_results, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_llvmir_inline_asm_flags)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_inline_asm_asm_template, 0)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_inline_asm_constraints, 1)
iree_status_t loom_llvmir_inline_asm_build(
    loom_builder_t* builder,
    uint8_t instance_flags,
    loom_string_id_t asm_template,
    loom_string_id_t constraints,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_LLVMIR_INTRINSIC: Structured call to a supported LLVM intrinsic. The intrinsic spelling is a string so target-family providers can recognize their own intrinsics without extending a central enum.
// %ticks = llvmir.intrinsic<llvm.x86.rdtsc> () : () -> i64
LOOM_DEFINE_ISA(loom_llvmir_intrinsic_isa, LOOM_OP_LLVMIR_INTRINSIC)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_llvmir_intrinsic_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_llvmir_intrinsic_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_llvmir_intrinsic_kind, 0)
iree_status_t loom_llvmir_intrinsic_build(
    loom_builder_t* builder,
    loom_string_id_t kind,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// Returns the vtable array for the llvmir dialect.
const loom_op_vtable_t* const* loom_llvmir_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the llvmir dialect.
const loom_op_semantics_t* loom_llvmir_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a llvmir op kind, or empty metadata.
loom_op_semantics_t loom_llvmir_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_LLVMIR_OPS_H_
