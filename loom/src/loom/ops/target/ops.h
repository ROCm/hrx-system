// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_TARGET_OPS_H_
#define LOOM_OPS_TARGET_OPS_H_

#include "loom/ir/parameterized_attr.h"
#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

enum {
  LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TARGET, 0),
  LOOM_PARAMETERIZED_ATTR_TARGET_COUNT_ = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_condition_descriptor_t loom_target_subgroup_size_condition;

// Requires the active function-version facts to establish this subgroup size.
static inline bool loom_target_subgroup_size_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE;
}
enum { LOOM_TARGET_SUBGROUP_SIZE_ATTR_SIZE_PARAMETER_INDEX = 0 };
static inline int64_t loom_target_subgroup_size_attr_size(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_TARGET_SUBGROUP_SIZE_ATTR_SIZE_PARAMETER_INDEX]);
}
iree_status_t loom_target_subgroup_size_attr_make(
    loom_module_t* module,
    int64_t size,
    loom_attribute_t* out_attr);

enum {
  LOOM_OP_TARGET_GENERIC = LOOM_OP_KIND(LOOM_DIALECT_TARGET, 0),
  LOOM_OP_TARGET_DECL = LOOM_OP_KIND(LOOM_DIALECT_TARGET, 1),
  LOOM_OP_TARGET_SUBGROUP_SIZE = LOOM_OP_KIND(LOOM_DIALECT_TARGET, 2),
  LOOM_OP_TARGET_COUNT_ = 3,
};

// Generic target-family row selected by target.generic.
typedef enum loom_target_generic_kind_e {
  LOOM_TARGET_GENERIC_KIND_REFERENCE = 1,
  LOOM_TARGET_GENERIC_KIND_COUNT_ = 2,
} loom_target_generic_kind_t;

// LOOM_OP_TARGET_GENERIC: Generic target-family record for target-independent or host-neutral compilation. The typed selector chooses a generated row; optional attrs structurally override only the authored fields.
// target.generic<reference> @oracle
LOOM_DEFINE_ISA(loom_target_generic_isa, LOOM_OP_TARGET_GENERIC)
LOOM_DEFINE_ATTR_SYMBOL(loom_target_generic_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_target_generic_kind, 1, loom_target_generic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_target_generic_codegen_format, 2, loom_target_codegen_format_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_target_generic_artifact_format, 3, loom_target_artifact_format_t)
LOOM_DEFINE_ATTR_I64(loom_target_generic_default_pointer_bitwidth, 4)
LOOM_DEFINE_ATTR_I64(loom_target_generic_index_bitwidth, 5)
LOOM_DEFINE_ATTR_I64(loom_target_generic_offset_bitwidth, 6)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_size_x, 7)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_size_y, 8)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_size_z, 9)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_flat_workgroup_size, 10)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_storage_bytes, 11)
LOOM_DEFINE_ATTR_I64(loom_target_generic_subgroup_size, 12)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_grid_size_x, 13)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_grid_size_y, 14)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_grid_size_z, 15)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_flat_grid_size, 16)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_count_x, 17)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_count_y, 18)
LOOM_DEFINE_ATTR_I64(loom_target_generic_max_workgroup_count_z, 19)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_generic, 20)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_global, 21)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_workgroup, 22)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_constant, 23)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_private, 24)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_host, 25)
LOOM_DEFINE_ATTR_I64(loom_target_generic_memory_space_descriptor, 26)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_target_generic_abi, 27, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_target_generic_export_symbol, 28)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_target_generic_linkage, 29, loom_target_linkage_t)
LOOM_DEFINE_ATTR_STRING(loom_target_generic_contract_set_key, 30)
LOOM_DEFINE_ATTR_I64(loom_target_generic_contract_feature_bits, 31)
enum loom_target_generic_build_flag_bits_e {
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CODEGEN_FORMAT = 1u << 0,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_ARTIFACT_FORMAT = 1u << 1,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH = 1u << 2,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_INDEX_BITWIDTH = 1u << 3,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_OFFSET_BITWIDTH = 1u << 4,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X = 1u << 5,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y = 1u << 6,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z = 1u << 7,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE = 1u << 8,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_STORAGE_BYTES = 1u << 9,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_SUBGROUP_SIZE = 1u << 10,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_GRID_SIZE_X = 1u << 11,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y = 1u << 12,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z = 1u << 13,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE = 1u << 14,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X = 1u << 15,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y = 1u << 16,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z = 1u << 17,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC = 1u << 18,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL = 1u << 19,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP = 1u << 20,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT = 1u << 21,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE = 1u << 22,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_HOST = 1u << 23,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR = 1u << 24,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_ABI = 1u << 25,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 26,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_LINKAGE = 1u << 27,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CONTRACT_SET_KEY = 1u << 28,
  LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS = 1u << 29,
};
typedef uint32_t loom_target_generic_build_flags_t;
iree_status_t loom_target_generic_build(
    loom_builder_t* builder,
    loom_target_generic_build_flags_t build_flags,
    loom_target_generic_kind_t kind,
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
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_TARGET_DECL: Declares a target-record contract whose definition may be provided by linking. The declaration remains valid in partially specialized IR and a linked definition must satisfy its contract.
// target.decl @external_target
LOOM_DEFINE_ISA(loom_target_decl_isa, LOOM_OP_TARGET_DECL)
LOOM_DEFINE_ATTR_SYMBOL(loom_target_decl_symbol, 0)
iree_status_t loom_target_decl_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_TARGET_SUBGROUP_SIZE: Read the selected subgroup size of the current function version.
// %size = target.subgroup.size : index
LOOM_DEFINE_ISA(loom_target_subgroup_size_isa, LOOM_OP_TARGET_SUBGROUP_SIZE)
LOOM_DEFINE_RESULT(loom_target_subgroup_size_result, 0)
iree_status_t loom_target_subgroup_size_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_subgroup_size_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Returns the vtable array for the target dialect.
const loom_op_vtable_t* const* loom_target_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the target dialect.
const loom_op_semantics_t* loom_target_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a target op kind, or empty metadata.
loom_op_semantics_t loom_target_op_semantics(
    loom_op_kind_t kind);

// Returns parameterized attribute descriptors for the target dialect.
const loom_parameterized_attr_descriptor_t* loom_target_dialect_parameterized_attrs(
    iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TARGET_OPS_H_
