// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_WASM_OPS_H_
#define LOOM_OPS_WASM_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_WASM_TARGET = LOOM_OP_KIND(LOOM_DIALECT_WASM, 0),
  LOOM_OP_WASM_COUNT_ = 1,
};

// Wasm target row selected by wasm.target.
typedef enum loom_wasm_target_kind_e {
  LOOM_WASM_TARGET_KIND_SIMD128 = 1,
  LOOM_WASM_TARGET_KIND_COUNT_ = 2,
} loom_wasm_target_kind_t;

// LOOM_OP_WASM_TARGET: Wasm target-family record. The selector chooses an authored Wasm row; optional attrs structurally override common target fields.
// wasm.target<simd128> @wasm
LOOM_DEFINE_ISA(loom_wasm_target_isa, LOOM_OP_WASM_TARGET)
LOOM_DEFINE_ATTR_SYMBOL(loom_wasm_target_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_wasm_target_kind, 1, loom_wasm_target_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_wasm_target_codegen_format, 2, loom_target_codegen_format_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_wasm_target_artifact_format, 3, loom_target_artifact_format_t)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_default_pointer_bitwidth, 4)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_index_bitwidth, 5)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_offset_bitwidth, 6)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_workgroup_size_x, 7)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_workgroup_size_y, 8)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_workgroup_size_z, 9)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_flat_workgroup_size, 10)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_subgroup_size, 11)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_grid_size_x, 12)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_grid_size_y, 13)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_grid_size_z, 14)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_flat_grid_size, 15)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_workgroup_count_x, 16)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_workgroup_count_y, 17)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_max_workgroup_count_z, 18)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_generic, 19)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_global, 20)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_workgroup, 21)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_constant, 22)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_private, 23)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_host, 24)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_memory_space_descriptor, 25)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_wasm_target_abi, 26, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_wasm_target_export_symbol, 27)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_wasm_target_linkage, 28, loom_target_linkage_t)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_hal_buffer_resource_flags, 29)
LOOM_DEFINE_ATTR_STRING(loom_wasm_target_contract_set_key, 30)
LOOM_DEFINE_ATTR_I64(loom_wasm_target_contract_feature_bits, 31)
enum loom_wasm_target_build_flag_bits_e {
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT = 1u << 0,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT = 1u << 1,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH = 1u << 2,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH = 1u << 3,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH = 1u << 4,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X = 1u << 5,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y = 1u << 6,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z = 1u << 7,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE = 1u << 8,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE = 1u << 9,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X = 1u << 10,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y = 1u << 11,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z = 1u << 12,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE = 1u << 13,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X = 1u << 14,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y = 1u << 15,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z = 1u << 16,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC = 1u << 17,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL = 1u << 18,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP = 1u << 19,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT = 1u << 20,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE = 1u << 21,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST = 1u << 22,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR = 1u << 23,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_ABI = 1u << 24,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 25,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_LINKAGE = 1u << 26,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_HAL_BUFFER_RESOURCE_FLAGS = 1u << 27,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY = 1u << 28,
  LOOM_WASM_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS = 1u << 29,
};
typedef uint32_t loom_wasm_target_build_flags_t;
iree_status_t loom_wasm_target_build(
    loom_builder_t* builder,
    loom_wasm_target_build_flags_t build_flags,
    loom_wasm_target_kind_t kind,
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
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the wasm dialect.
const loom_op_vtable_t* const* loom_wasm_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the wasm dialect.
const loom_op_semantics_t* loom_wasm_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a wasm op kind, or empty metadata.
loom_op_semantics_t loom_wasm_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_WASM_OPS_H_
