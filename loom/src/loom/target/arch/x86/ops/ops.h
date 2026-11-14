// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_X86_OPS_H_
#define LOOM_OPS_X86_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/arch/x86/records/target_records.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_X86_TARGET = LOOM_OP_KIND(LOOM_DIALECT_X86, 0),
  LOOM_OP_X86_COUNT_ = 1,
};

// x86 target row selected by x86.target.
typedef enum loom_x86_target_kind_e {
  LOOM_X86_TARGET_KIND_AVX512 = 1,
  LOOM_X86_TARGET_KIND_PACKED_DOT = 2,
  LOOM_X86_TARGET_KIND_AVX512_PACKED_DOT = 3,
  LOOM_X86_TARGET_KIND_SCALAR = 4,
  LOOM_X86_TARGET_KIND_AVX2 = 5,
  LOOM_X86_TARGET_KIND_COUNT_ = 6,
} loom_x86_target_kind_t;

// Optional x86 instruction-set feature.
typedef enum loom_x86_target_features_e {
  LOOM_X86_TARGET_FEATURES_AVX512VNNI = 0,
  LOOM_X86_TARGET_FEATURES_AVX512VL = 1,
  LOOM_X86_TARGET_FEATURES_AVXVNNI = 2,
  LOOM_X86_TARGET_FEATURES_AVXVNNIINT8 = 3,
  LOOM_X86_TARGET_FEATURES_AVXVNNIINT16 = 4,
  LOOM_X86_TARGET_FEATURES_AVX10_2 = 5,
  LOOM_X86_TARGET_FEATURES_AVX512BF16 = 6,
  LOOM_X86_TARGET_FEATURES_AVX512FP16 = 7,
  LOOM_X86_TARGET_FEATURES_AVXNECONVERT = 8,
  LOOM_X86_TARGET_FEATURES_F16C = 9,
  LOOM_X86_TARGET_FEATURES_BMI2 = 10,
  LOOM_X86_TARGET_FEATURES_BMI = 11,
  LOOM_X86_TARGET_FEATURES_POPCNT = 12,
  LOOM_X86_TARGET_FEATURES_LZCNT = 13,
  LOOM_X86_TARGET_FEATURES_AVX512VBMI = 14,
  LOOM_X86_TARGET_FEATURES_COUNT_ = 15,
} loom_x86_target_features_t;

// LOOM_OP_X86_TARGET: x86 target-family record. The selector chooses an authored CPU row; optional attrs structurally override common target fields.
// x86.target<scalar> @host
LOOM_DEFINE_ISA(loom_x86_target_isa, LOOM_OP_X86_TARGET)
LOOM_DEFINE_ATTR_SYMBOL(loom_x86_target_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_x86_target_kind, 1, loom_x86_target_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_x86_target_codegen_format, 2, loom_target_codegen_format_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_x86_target_artifact_format, 3, loom_target_artifact_format_t)
LOOM_DEFINE_ATTR_I64(loom_x86_target_default_pointer_bitwidth, 4)
LOOM_DEFINE_ATTR_I64(loom_x86_target_index_bitwidth, 5)
LOOM_DEFINE_ATTR_I64(loom_x86_target_offset_bitwidth, 6)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_size_x, 7)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_size_y, 8)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_size_z, 9)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_flat_workgroup_size, 10)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_storage_bytes, 11)
LOOM_DEFINE_ATTR_I64(loom_x86_target_subgroup_size, 12)
LOOM_DEFINE_ATTR_I64(loom_x86_target_execution_unit_count, 13)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_grid_size_x, 14)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_grid_size_y, 15)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_grid_size_z, 16)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_flat_grid_size, 17)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_count_x, 18)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_count_y, 19)
LOOM_DEFINE_ATTR_I64(loom_x86_target_max_workgroup_count_z, 20)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_generic, 21)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_global, 22)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_workgroup, 23)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_constant, 24)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_private, 25)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_host, 26)
LOOM_DEFINE_ATTR_I64(loom_x86_target_memory_space_descriptor, 27)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_x86_target_abi, 28, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_x86_target_export_symbol, 29)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_x86_target_linkage, 30, loom_target_linkage_t)
LOOM_DEFINE_ATTR_STRING(loom_x86_target_contract_set_key, 31)
LOOM_DEFINE_ATTR_I64(loom_x86_target_contract_feature_bits, 32)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_x86_target_tune, 33, loom_x86_target_tune_t)
LOOM_DEFINE_ATTR_SIGNED_ENUM_SET(loom_x86_target_features, 34)
// Build flag values use macros because C enums cannot portably represent
// values wider than int.
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT (UINT64_C(1) << 0)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT (UINT64_C(1) << 1)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH \
  (UINT64_C(1) << 2)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH (UINT64_C(1) << 3)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH (UINT64_C(1) << 4)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X (UINT64_C(1) << 5)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y (UINT64_C(1) << 6)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z (UINT64_C(1) << 7)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE \
  (UINT64_C(1) << 8)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_STORAGE_BYTES \
  (UINT64_C(1) << 9)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE (UINT64_C(1) << 10)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_EXECUTION_UNIT_COUNT (UINT64_C(1) << 11)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X (UINT64_C(1) << 12)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y (UINT64_C(1) << 13)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z (UINT64_C(1) << 14)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE (UINT64_C(1) << 15)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X (UINT64_C(1) << 16)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y (UINT64_C(1) << 17)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z (UINT64_C(1) << 18)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC (UINT64_C(1) << 19)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL (UINT64_C(1) << 20)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP \
  (UINT64_C(1) << 21)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT (UINT64_C(1) << 22)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE (UINT64_C(1) << 23)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST (UINT64_C(1) << 24)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR \
  (UINT64_C(1) << 25)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_ABI (UINT64_C(1) << 26)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL (UINT64_C(1) << 27)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_LINKAGE (UINT64_C(1) << 28)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY (UINT64_C(1) << 29)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS (UINT64_C(1) << 30)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_TUNE (UINT64_C(1) << 31)
#define LOOM_X86_TARGET_BUILD_FLAG_HAS_FEATURES (UINT64_C(1) << 32)
typedef uint64_t loom_x86_target_build_flags_t;
iree_status_t loom_x86_target_build(
    loom_builder_t* builder,
    loom_x86_target_build_flags_t build_flags,
    loom_x86_target_kind_t kind,
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
    loom_optional int64_t max_workgroup_storage_bytes,
    loom_optional int64_t subgroup_size,
    loom_optional int64_t execution_unit_count,
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
    loom_optional loom_string_id_t contract_set_key,
    loom_optional int64_t contract_feature_bits,
    loom_optional uint8_t tune,
    loom_optional loom_signed_enum_set_t features,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_x86_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the x86 dialect.
const loom_op_vtable_t* const* loom_x86_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the x86 dialect.
const loom_op_semantics_t* loom_x86_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a x86 op kind, or empty metadata.
loom_op_semantics_t loom_x86_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/target/arch/x86/ops/ops.inc"

#endif  // LOOM_OPS_X86_OPS_H_
