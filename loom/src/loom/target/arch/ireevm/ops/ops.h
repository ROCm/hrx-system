// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_IREEVM_OPS_H_
#define LOOM_OPS_IREEVM_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_IREEVM_TARGET = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 0),
  LOOM_OP_IREEVM_IMPORT_DECL = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 1),
  LOOM_OP_IREEVM_REF_RETAIN = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 2),
  LOOM_OP_IREEVM_REF_RELEASE = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 3),
  LOOM_OP_IREEVM_REF_DISCARD = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 4),
  LOOM_OP_IREEVM_BUFFER_LENGTH = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 5),
  LOOM_OP_IREEVM_BUFFER_LOAD_I8_U = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 6),
  LOOM_OP_IREEVM_BUFFER_LOAD_I8_S = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 7),
  LOOM_OP_IREEVM_BUFFER_LOAD_I16_U = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 8),
  LOOM_OP_IREEVM_BUFFER_LOAD_I16_S = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 9),
  LOOM_OP_IREEVM_BUFFER_LOAD_I32 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 10),
  LOOM_OP_IREEVM_BUFFER_LOAD_I64 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 11),
  LOOM_OP_IREEVM_BUFFER_LOAD_F32 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 12),
  LOOM_OP_IREEVM_BUFFER_LOAD_F64 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 13),
  LOOM_OP_IREEVM_BUFFER_STORE_I8 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 14),
  LOOM_OP_IREEVM_BUFFER_STORE_I16 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 15),
  LOOM_OP_IREEVM_BUFFER_STORE_I32 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 16),
  LOOM_OP_IREEVM_BUFFER_STORE_I64 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 17),
  LOOM_OP_IREEVM_BUFFER_STORE_F32 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 18),
  LOOM_OP_IREEVM_BUFFER_STORE_F64 = LOOM_OP_KIND(LOOM_DIALECT_IREEVM, 19),
  LOOM_OP_IREEVM_COUNT_ = 20,
};

// IREE VM target row selected by ireevm.target.
typedef enum loom_ireevm_target_kind_e {
  LOOM_IREEVM_TARGET_KIND_CORE = 1,
  LOOM_IREEVM_TARGET_KIND_COUNT_ = 2,
} loom_ireevm_target_kind_t;

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_ireevm_import_decl_visibility_e {
  LOOM_IREEVM_IMPORT_DECL_VISIBILITY_PUBLIC = 1,
  LOOM_IREEVM_IMPORT_DECL_VISIBILITY_COUNT_ = 2,
} loom_ireevm_import_decl_visibility_t;

// Function calling convention. Absent (0) means host.
typedef enum loom_ireevm_import_decl_cc_e {
  LOOM_IREEVM_IMPORT_DECL_CC_HOST = 1,
  LOOM_IREEVM_IMPORT_DECL_CC_DEVICE = 2,
  LOOM_IREEVM_IMPORT_DECL_CC_INITIALIZER = 3,
  LOOM_IREEVM_IMPORT_DECL_CC_DEINITIALIZER = 4,
  LOOM_IREEVM_IMPORT_DECL_CC_COUNT_ = 5,
} loom_ireevm_import_decl_cc_t;

// Function purity. Absent (0) means unspecified (conservative).
typedef enum loom_ireevm_import_decl_purity_e {
  LOOM_IREEVM_IMPORT_DECL_PURITY_PURE = 1,
  LOOM_IREEVM_IMPORT_DECL_PURITY_COUNT_ = 2,
} loom_ireevm_import_decl_purity_t;

// LOOM_OP_IREEVM_TARGET: IREE VM target-family record. The selector chooses a VM emission row; optional attrs structurally override authored common target fields.
// ireevm.target<core> @vm
LOOM_DEFINE_ISA(loom_ireevm_target_isa, LOOM_OP_IREEVM_TARGET)
LOOM_DEFINE_ATTR_SYMBOL(loom_ireevm_target_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_target_kind, 1, loom_ireevm_target_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_target_codegen_format, 2, loom_target_codegen_format_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_target_artifact_format, 3, loom_target_artifact_format_t)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_default_pointer_bitwidth, 4)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_index_bitwidth, 5)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_offset_bitwidth, 6)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_workgroup_size_x, 7)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_workgroup_size_y, 8)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_workgroup_size_z, 9)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_flat_workgroup_size, 10)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_subgroup_size, 11)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_grid_size_x, 12)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_grid_size_y, 13)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_grid_size_z, 14)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_flat_grid_size, 15)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_workgroup_count_x, 16)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_workgroup_count_y, 17)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_max_workgroup_count_z, 18)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_generic, 19)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_global, 20)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_workgroup, 21)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_constant, 22)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_private, 23)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_host, 24)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_memory_space_descriptor, 25)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_target_abi, 26, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_ireevm_target_export_symbol, 27)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_target_linkage, 28, loom_target_linkage_t)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_hal_buffer_resource_flags, 29)
LOOM_DEFINE_ATTR_STRING(loom_ireevm_target_contract_set_key, 30)
LOOM_DEFINE_ATTR_I64(loom_ireevm_target_contract_feature_bits, 31)
enum loom_ireevm_target_build_flag_bits_e {
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_CODEGEN_FORMAT = 1u << 0,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_ARTIFACT_FORMAT = 1u << 1,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_DEFAULT_POINTER_BITWIDTH = 1u << 2,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_INDEX_BITWIDTH = 1u << 3,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_OFFSET_BITWIDTH = 1u << 4,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X = 1u << 5,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y = 1u << 6,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z = 1u << 7,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE = 1u << 8,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE = 1u << 9,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X = 1u << 10,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y = 1u << 11,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z = 1u << 12,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE = 1u << 13,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X = 1u << 14,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y = 1u << 15,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z = 1u << 16,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GENERIC = 1u << 17,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_GLOBAL = 1u << 18,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_WORKGROUP = 1u << 19,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_CONSTANT = 1u << 20,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_PRIVATE = 1u << 21,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_HOST = 1u << 22,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_MEMORY_SPACE_DESCRIPTOR = 1u << 23,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_ABI = 1u << 24,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 25,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_LINKAGE = 1u << 26,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_HAL_BUFFER_RESOURCE_FLAGS = 1u << 27,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_CONTRACT_SET_KEY = 1u << 28,
  LOOM_IREEVM_TARGET_BUILD_FLAG_HAS_CONTRACT_FEATURE_BITS = 1u << 29,
};
typedef uint32_t loom_ireevm_target_build_flags_t;
iree_status_t loom_ireevm_target_build(
    loom_builder_t* builder,
    loom_ireevm_target_build_flags_t build_flags,
    loom_ireevm_target_kind_t kind,
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

// LOOM_OP_IREEVM_IMPORT_DECL: IREE VM imported function declaration. Callable by name via func.call and lowered to a VM module import during target materialization.
// ireevm.import.decl target(@vm) symbol("hal.buffer.length") @hal_buffer_length(%buffer: ireevm.ref<ireevm.list<i32>>) -> (i64)
LOOM_DEFINE_ISA(loom_ireevm_import_decl_isa, LOOM_OP_IREEVM_IMPORT_DECL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_ireevm_import_decl_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_ireevm_import_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_ireevm_import_decl_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_import_decl_visibility, 1, loom_ireevm_import_decl_visibility_t)
LOOM_DEFINE_ATTR_SYMBOL(loom_ireevm_import_decl_target, 2)
LOOM_DEFINE_ATTR_STRING(loom_ireevm_import_decl_import_symbol, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_import_decl_cc, 4, loom_ireevm_import_decl_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_ireevm_import_decl_purity, 5, loom_ireevm_import_decl_purity_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_ireevm_import_decl_predicates, 6)
enum loom_ireevm_import_decl_build_flag_bits_e {
  LOOM_IREEVM_IMPORT_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_IREEVM_IMPORT_DECL_BUILD_FLAG_HAS_CC = 1u << 1,
  LOOM_IREEVM_IMPORT_DECL_BUILD_FLAG_HAS_PURITY = 1u << 2,
};
typedef uint32_t loom_ireevm_import_decl_build_flags_t;
iree_status_t loom_ireevm_import_decl_build(
    loom_builder_t* builder,
    loom_ireevm_import_decl_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_symbol_ref_t target,
    loom_string_id_t import_symbol,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
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
iree_status_t loom_ireevm_import_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_REF_RETAIN: Retain an additional owned IREE VM reference.
// %owned = ireevm.ref.retain %resource : ireevm.ref<ireevm.list<i32>> -> ireevm.ref<ireevm.list<i32>>
LOOM_DEFINE_ISA(loom_ireevm_ref_retain_isa, LOOM_OP_IREEVM_REF_RETAIN)
LOOM_DEFINE_OPERAND(loom_ireevm_ref_retain_resource, 0)
LOOM_DEFINE_RESULT(loom_ireevm_ref_retain_result, 0)
iree_status_t loom_ireevm_ref_retain_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t resource,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_IREEVM_REF_RELEASE: Release an owned IREE VM reference.
// ireevm.ref.release %resource : ireevm.ref<ireevm.list<i32>>
LOOM_DEFINE_ISA(loom_ireevm_ref_release_isa, LOOM_OP_IREEVM_REF_RELEASE)
LOOM_DEFINE_OPERAND(loom_ireevm_ref_release_resource, 0)
iree_status_t loom_ireevm_ref_release_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_IREEVM_REF_DISCARD: Discard compiler ownership of an IREE VM reference without releasing it.
// ireevm.ref.discard %resource : ireevm.ref<ireevm.list<i32>>
LOOM_DEFINE_ISA(loom_ireevm_ref_discard_isa, LOOM_OP_IREEVM_REF_DISCARD)
LOOM_DEFINE_OPERAND(loom_ireevm_ref_discard_resource, 0)
iree_status_t loom_ireevm_ref_discard_build(
    loom_builder_t* builder,
    loom_value_id_t resource,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_IREEVM_BUFFER_LENGTH: Return the byte length of an IREE VM host byte buffer.
// %length = ireevm.buffer.length %buffer : ireevm.ref<ireevm.buffer> -> i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_length_isa, LOOM_OP_IREEVM_BUFFER_LENGTH)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_length_buffer, 0)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_length_result, 0)
iree_status_t loom_ireevm_buffer_length_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_I8_U: Load an unsigned i8 value as i32 from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.i8.u %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> i32
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_i8_u_isa, LOOM_OP_IREEVM_BUFFER_LOAD_I8_U)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i8_u_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i8_u_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_i8_u_result, 0)
iree_status_t loom_ireevm_buffer_load_i8_u_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_I8_S: Load a sign-extended i8 value as i32 from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.i8.s %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> i32
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_i8_s_isa, LOOM_OP_IREEVM_BUFFER_LOAD_I8_S)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i8_s_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i8_s_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_i8_s_result, 0)
iree_status_t loom_ireevm_buffer_load_i8_s_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_I16_U: Load an unsigned i16 value as i32 from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.i16.u %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> i32
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_i16_u_isa, LOOM_OP_IREEVM_BUFFER_LOAD_I16_U)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i16_u_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i16_u_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_i16_u_result, 0)
iree_status_t loom_ireevm_buffer_load_i16_u_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_I16_S: Load a sign-extended i16 value as i32 from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.i16.s %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> i32
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_i16_s_isa, LOOM_OP_IREEVM_BUFFER_LOAD_I16_S)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i16_s_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i16_s_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_i16_s_result, 0)
iree_status_t loom_ireevm_buffer_load_i16_s_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_I32: Load an i32 value from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.i32 %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> i32
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_i32_isa, LOOM_OP_IREEVM_BUFFER_LOAD_I32)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i32_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i32_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_i32_result, 0)
iree_status_t loom_ireevm_buffer_load_i32_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_I64: Load an i64 value from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.i64 %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_i64_isa, LOOM_OP_IREEVM_BUFFER_LOAD_I64)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i64_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_i64_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_i64_result, 0)
iree_status_t loom_ireevm_buffer_load_i64_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_F32: Load an f32 value from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.f32 %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> f32
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_f32_isa, LOOM_OP_IREEVM_BUFFER_LOAD_F32)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_f32_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_f32_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_f32_result, 0)
iree_status_t loom_ireevm_buffer_load_f32_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_LOAD_F64: Load an f64 value from an IREE VM host byte buffer.
// %value = ireevm.buffer.load.f64 %buffer[%offset] : ireevm.ref<ireevm.buffer>, i64 -> f64
LOOM_DEFINE_ISA(loom_ireevm_buffer_load_f64_isa, LOOM_OP_IREEVM_BUFFER_LOAD_F64)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_f64_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_load_f64_element_offset, 1)
LOOM_DEFINE_RESULT(loom_ireevm_buffer_load_f64_result, 0)
iree_status_t loom_ireevm_buffer_load_f64_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t buffer,
    loom_may_consume loom_value_id_t element_offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_STORE_I8: Store the low 8 bits of an i32 value into an IREE VM host byte buffer.
// ireevm.buffer.store.i8 %value, %buffer[%offset] : i32, ireevm.ref<ireevm.buffer>, i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_store_i8_isa, LOOM_OP_IREEVM_BUFFER_STORE_I8)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i8_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i8_element_offset, 1)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i8_value, 2)
iree_status_t loom_ireevm_buffer_store_i8_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t buffer,
    loom_value_id_t element_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_STORE_I16: Store the low 16 bits of an i32 value into an IREE VM host byte buffer.
// ireevm.buffer.store.i16 %value, %buffer[%offset] : i32, ireevm.ref<ireevm.buffer>, i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_store_i16_isa, LOOM_OP_IREEVM_BUFFER_STORE_I16)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i16_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i16_element_offset, 1)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i16_value, 2)
iree_status_t loom_ireevm_buffer_store_i16_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t buffer,
    loom_value_id_t element_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_STORE_I32: Store an i32 value into an IREE VM host byte buffer.
// ireevm.buffer.store.i32 %value, %buffer[%offset] : i32, ireevm.ref<ireevm.buffer>, i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_store_i32_isa, LOOM_OP_IREEVM_BUFFER_STORE_I32)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i32_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i32_element_offset, 1)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i32_value, 2)
iree_status_t loom_ireevm_buffer_store_i32_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t buffer,
    loom_value_id_t element_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_STORE_I64: Store an i64 value into an IREE VM host byte buffer.
// ireevm.buffer.store.i64 %value, %buffer[%offset] : i64, ireevm.ref<ireevm.buffer>, i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_store_i64_isa, LOOM_OP_IREEVM_BUFFER_STORE_I64)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i64_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i64_element_offset, 1)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_i64_value, 2)
iree_status_t loom_ireevm_buffer_store_i64_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t buffer,
    loom_value_id_t element_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_STORE_F32: Store an f32 value into an IREE VM host byte buffer.
// ireevm.buffer.store.f32 %value, %buffer[%offset] : f32, ireevm.ref<ireevm.buffer>, i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_store_f32_isa, LOOM_OP_IREEVM_BUFFER_STORE_F32)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_f32_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_f32_element_offset, 1)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_f32_value, 2)
iree_status_t loom_ireevm_buffer_store_f32_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t buffer,
    loom_value_id_t element_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_IREEVM_BUFFER_STORE_F64: Store an f64 value into an IREE VM host byte buffer.
// ireevm.buffer.store.f64 %value, %buffer[%offset] : f64, ireevm.ref<ireevm.buffer>, i64
LOOM_DEFINE_ISA(loom_ireevm_buffer_store_f64_isa, LOOM_OP_IREEVM_BUFFER_STORE_F64)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_f64_buffer, 0)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_f64_element_offset, 1)
LOOM_DEFINE_OPERAND(loom_ireevm_buffer_store_f64_value, 2)
iree_status_t loom_ireevm_buffer_store_f64_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_value_id_t buffer,
    loom_value_id_t element_offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_ireevm_buffer_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the ireevm dialect.
const loom_op_vtable_t* const* loom_ireevm_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the ireevm dialect.
const loom_op_semantics_t* loom_ireevm_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a ireevm op kind, or empty metadata.
loom_op_semantics_t loom_ireevm_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_IREEVM_OPS_H_
