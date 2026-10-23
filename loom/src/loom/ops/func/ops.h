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

#include "loom/ir/parameterized_attr.h"
#include "loom/ops/op_defs.h"
#include "loom/ir/ir.h"
#include "loom/ir/location.h"
#include "loom/target/types.h"

enum {
  LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_UNKNOWN = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 0),
  LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_FIELD = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 1),
  LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_FILE = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 2),
  LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_FUSED = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 3),
  LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_OPAQUE = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 4),
  LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_TAGGED = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 5),
  LOOM_PARAMETERIZED_ATTR_FUNC_ABI_C_FIELD = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 6),
  LOOM_PARAMETERIZED_ATTR_FUNC_ABI_C_TYPE = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 7),
  LOOM_PARAMETERIZED_ATTR_FUNC_ABI_C_SIGNATURE = LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_FUNC, 8),
  LOOM_PARAMETERIZED_ATTR_FUNC_COUNT_ = 9,
};

// Semantic source object kind in a retained C ABI signature.
typedef enum loom_func_abi_c_type_kind_e {
  LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN = 1,
  LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER = 2,
  LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER = 3,
  LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING = 4,
  LOOM_FUNC_ABI_C_TYPE_KIND_POINTER = 5,
  LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR = 6,
  LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY = 7,
  LOOM_FUNC_ABI_C_TYPE_KIND_RECORD = 8,
  LOOM_FUNC_ABI_C_TYPE_KIND_COUNT_ = 9,
} loom_func_abi_c_type_kind_t;

// Floating representation in a retained C ABI source object.
typedef enum loom_func_abi_c_type_format_e {
  LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E4M3FN = 1,
  LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E5M2 = 2,
  LOOM_FUNC_ABI_C_TYPE_FORMAT_F16 = 3,
  LOOM_FUNC_ABI_C_TYPE_FORMAT_BF16 = 4,
  LOOM_FUNC_ABI_C_TYPE_FORMAT_F32 = 5,
  LOOM_FUNC_ABI_C_TYPE_FORMAT_F64 = 6,
  LOOM_FUNC_ABI_C_TYPE_FORMAT_COUNT_ = 7,
} loom_func_abi_c_type_format_t;

#ifdef __cplusplus
extern "C" {
#endif

// An explicitly unknown captured location.
static inline bool loom_func_location_unknown_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_UNKNOWN;
}
iree_status_t loom_func_location_unknown_attr_make(
    loom_module_t* module,
    loom_attribute_t* out_attr);

// A captured source field range in its file node's coordinate space.
static inline bool loom_func_location_field_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_FIELD;
}
enum { LOOM_FUNC_LOCATION_FIELD_ATTR_KIND_PARAMETER_INDEX = 0 };
static inline loom_location_field_kind_t loom_func_location_field_attr_kind(loom_attribute_t attr) {
  return (loom_location_field_kind_t)loom_attr_as_enum(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FIELD_ATTR_KIND_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FIELD_ATTR_INDEX_PARAMETER_INDEX = 1 };
static inline int64_t loom_func_location_field_attr_index(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FIELD_ATTR_INDEX_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FIELD_ATTR_RANGE_PARAMETER_INDEX = 2 };
static inline loom_i64_array_t loom_func_location_field_attr_range(loom_attribute_t attr) {
  return loom_attr_as_i64_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FIELD_ATTR_RANGE_PARAMETER_INDEX]);
}
iree_status_t loom_func_location_field_attr_make(
    loom_module_t* module,
    loom_location_field_kind_t kind,
    int64_t index,
    loom_i64_array_t range,
    loom_attribute_t* out_attr);

// Captured file range and optional original source text beginning at the start line.
enum loom_func_location_file_attr_build_flag_bits_e {
  LOOM_FUNC_LOCATION_FILE_ATTR_BUILD_FLAG_HAS_SYNTHETIC = 1u << 0,
  LOOM_FUNC_LOCATION_FILE_ATTR_BUILD_FLAG_HAS_FIELDS = 1u << 1,
  LOOM_FUNC_LOCATION_FILE_ATTR_BUILD_FLAG_HAS_TEXT = 1u << 2,
};
typedef uint32_t loom_func_location_file_attr_build_flags_t;
static inline bool loom_func_location_file_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_FILE;
}
enum { LOOM_FUNC_LOCATION_FILE_ATTR_SOURCE_PARAMETER_INDEX = 0 };
static inline loom_string_id_t loom_func_location_file_attr_source(loom_attribute_t attr) {
  return loom_attr_as_string_id(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_SOURCE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FILE_ATTR_RANGE_PARAMETER_INDEX = 1 };
static inline loom_i64_array_t loom_func_location_file_attr_range(loom_attribute_t attr) {
  return loom_attr_as_i64_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_RANGE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FILE_ATTR_SYNTHETIC_PARAMETER_INDEX = 2 };
static inline bool loom_func_location_file_attr_has_synthetic(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
static inline bool loom_func_location_file_attr_synthetic(loom_attribute_t attr) {
  return loom_attr_as_bool(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FILE_ATTR_FIELDS_PARAMETER_INDEX = 3 };
static inline bool loom_func_location_file_attr_has_fields(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_FIELDS_PARAMETER_INDEX]);
}
static inline loom_parameterized_attr_array_t loom_func_location_file_attr_fields(loom_attribute_t attr) {
  return loom_attr_as_parameterized_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_FIELDS_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FILE_ATTR_TEXT_PARAMETER_INDEX = 4 };
static inline bool loom_func_location_file_attr_has_text(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_TEXT_PARAMETER_INDEX]);
}
static inline iree_const_byte_span_t loom_func_location_file_attr_text(loom_attribute_t attr) {
  return loom_attr_as_bytes(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FILE_ATTR_TEXT_PARAMETER_INDEX]);
}
iree_status_t loom_func_location_file_attr_make(
    loom_module_t* module,
    loom_func_location_file_attr_build_flags_t build_flags,
    loom_string_id_t source,
    loom_i64_array_t range,
    bool synthetic,
    loom_parameterized_attr_array_t fields,
    iree_const_byte_span_t text,
    loom_attribute_t* out_attr);

// Provenance derived from several captured locations.
enum loom_func_location_fused_attr_build_flag_bits_e {
  LOOM_FUNC_LOCATION_FUSED_ATTR_BUILD_FLAG_HAS_SYNTHETIC = 1u << 0,
};
typedef uint32_t loom_func_location_fused_attr_build_flags_t;
static inline bool loom_func_location_fused_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_FUSED;
}
enum { LOOM_FUNC_LOCATION_FUSED_ATTR_CHILDREN_PARAMETER_INDEX = 0 };
static inline loom_i64_array_t loom_func_location_fused_attr_children(loom_attribute_t attr) {
  return loom_attr_as_i64_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FUSED_ATTR_CHILDREN_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_FUSED_ATTR_SYNTHETIC_PARAMETER_INDEX = 1 };
static inline bool loom_func_location_fused_attr_has_synthetic(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FUSED_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
static inline bool loom_func_location_fused_attr_synthetic(loom_attribute_t attr) {
  return loom_attr_as_bool(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_FUSED_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
iree_status_t loom_func_location_fused_attr_make(
    loom_module_t* module,
    loom_func_location_fused_attr_build_flags_t build_flags,
    loom_i64_array_t children,
    bool synthetic,
    loom_attribute_t* out_attr);

// Captured external source identity and uninterpreted payload.
enum loom_func_location_opaque_attr_build_flag_bits_e {
  LOOM_FUNC_LOCATION_OPAQUE_ATTR_BUILD_FLAG_HAS_SYNTHETIC = 1u << 0,
};
typedef uint32_t loom_func_location_opaque_attr_build_flags_t;
static inline bool loom_func_location_opaque_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_OPAQUE;
}
enum { LOOM_FUNC_LOCATION_OPAQUE_ATTR_SOURCE_PARAMETER_INDEX = 0 };
static inline loom_string_id_t loom_func_location_opaque_attr_source(loom_attribute_t attr) {
  return loom_attr_as_string_id(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_OPAQUE_ATTR_SOURCE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_OPAQUE_ATTR_DATA_PARAMETER_INDEX = 1 };
static inline iree_const_byte_span_t loom_func_location_opaque_attr_data(loom_attribute_t attr) {
  return loom_attr_as_bytes(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_OPAQUE_ATTR_DATA_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_OPAQUE_ATTR_SYNTHETIC_PARAMETER_INDEX = 2 };
static inline bool loom_func_location_opaque_attr_has_synthetic(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_OPAQUE_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
static inline bool loom_func_location_opaque_attr_synthetic(loom_attribute_t attr) {
  return loom_attr_as_bool(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_OPAQUE_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
iree_status_t loom_func_location_opaque_attr_make(
    loom_module_t* module,
    loom_func_location_opaque_attr_build_flags_t build_flags,
    loom_string_id_t source,
    iree_const_byte_span_t data,
    bool synthetic,
    loom_attribute_t* out_attr);

// Tagged provenance with an optional captured child.
enum loom_func_location_tagged_attr_build_flag_bits_e {
  LOOM_FUNC_LOCATION_TAGGED_ATTR_BUILD_FLAG_HAS_CHILD = 1u << 0,
  LOOM_FUNC_LOCATION_TAGGED_ATTR_BUILD_FLAG_HAS_SYNTHETIC = 1u << 1,
};
typedef uint32_t loom_func_location_tagged_attr_build_flags_t;
static inline bool loom_func_location_tagged_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_LOCATION_TAGGED;
}
enum { LOOM_FUNC_LOCATION_TAGGED_ATTR_TAG_PARAMETER_INDEX = 0 };
static inline int64_t loom_func_location_tagged_attr_tag(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_TAGGED_ATTR_TAG_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_TAGGED_ATTR_DATA_PARAMETER_INDEX = 1 };
static inline iree_const_byte_span_t loom_func_location_tagged_attr_data(loom_attribute_t attr) {
  return loom_attr_as_bytes(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_TAGGED_ATTR_DATA_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_TAGGED_ATTR_CHILD_PARAMETER_INDEX = 2 };
static inline bool loom_func_location_tagged_attr_has_child(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_TAGGED_ATTR_CHILD_PARAMETER_INDEX]);
}
static inline int64_t loom_func_location_tagged_attr_child(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_TAGGED_ATTR_CHILD_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_LOCATION_TAGGED_ATTR_SYNTHETIC_PARAMETER_INDEX = 3 };
static inline bool loom_func_location_tagged_attr_has_synthetic(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_TAGGED_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
static inline bool loom_func_location_tagged_attr_synthetic(loom_attribute_t attr) {
  return loom_attr_as_bool(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_LOCATION_TAGGED_ATTR_SYNTHETIC_PARAMETER_INDEX]);
}
iree_status_t loom_func_location_tagged_attr_make(
    loom_module_t* module,
    loom_func_location_tagged_attr_build_flags_t build_flags,
    int64_t tag,
    iree_const_byte_span_t data,
    int64_t child,
    bool synthetic,
    loom_attribute_t* out_attr);

// One named field in a retained C ABI record layout.
static inline bool loom_func_abi_c_field_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_ABI_C_FIELD;
}
enum { LOOM_FUNC_ABI_C_FIELD_ATTR_NAME_PARAMETER_INDEX = 0 };
static inline loom_string_id_t loom_func_abi_c_field_attr_name(loom_attribute_t attr) {
  return loom_attr_as_string_id(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_FIELD_ATTR_NAME_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_FIELD_ATTR_TYPE_PARAMETER_INDEX = 1 };
static inline int64_t loom_func_abi_c_field_attr_type(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_FIELD_ATTR_TYPE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_FIELD_ATTR_BYTE_OFFSET_PARAMETER_INDEX = 2 };
static inline int64_t loom_func_abi_c_field_attr_byte_offset(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_FIELD_ATTR_BYTE_OFFSET_PARAMETER_INDEX]);
}
iree_status_t loom_func_abi_c_field_attr_make(
    loom_module_t* module,
    loom_string_id_t name,
    int64_t type,
    int64_t byte_offset,
    loom_attribute_t* out_attr);

// One postorder source object type node. Composite references name only preceding nodes.
enum loom_func_abi_c_type_attr_build_flag_bits_e {
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_BITWIDTH = 1u << 0,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_FORMAT = 1u << 1,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ADDRESS_SPACE = 1u << 2,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE = 1u << 3,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT = 1u << 4,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_STRIDE = 1u << 5,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_NAME = 1u << 6,
  LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_FIELDS = 1u << 7,
};
typedef uint32_t loom_func_abi_c_type_attr_build_flags_t;
static inline bool loom_func_abi_c_type_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_ABI_C_TYPE;
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_KIND_PARAMETER_INDEX = 0 };
static inline loom_func_abi_c_type_kind_t loom_func_abi_c_type_attr_kind(loom_attribute_t attr) {
  return (loom_func_abi_c_type_kind_t)loom_attr_as_enum(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_KIND_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_BYTE_LENGTH_PARAMETER_INDEX = 1 };
static inline int64_t loom_func_abi_c_type_attr_byte_length(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_BYTE_LENGTH_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_BYTE_ALIGNMENT_PARAMETER_INDEX = 2 };
static inline int64_t loom_func_abi_c_type_attr_byte_alignment(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_BYTE_ALIGNMENT_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_COMPONENT_COUNT_PARAMETER_INDEX = 3 };
static inline int64_t loom_func_abi_c_type_attr_component_count(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_COMPONENT_COUNT_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_BITWIDTH_PARAMETER_INDEX = 4 };
static inline bool loom_func_abi_c_type_attr_has_bitwidth(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_BITWIDTH_PARAMETER_INDEX]);
}
static inline int64_t loom_func_abi_c_type_attr_bitwidth(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_BITWIDTH_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_FORMAT_PARAMETER_INDEX = 5 };
static inline bool loom_func_abi_c_type_attr_has_format(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_FORMAT_PARAMETER_INDEX]);
}
static inline loom_func_abi_c_type_format_t loom_func_abi_c_type_attr_format(loom_attribute_t attr) {
  return (loom_func_abi_c_type_format_t)loom_attr_as_enum(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_FORMAT_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_ADDRESS_SPACE_PARAMETER_INDEX = 6 };
static inline bool loom_func_abi_c_type_attr_has_address_space(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ADDRESS_SPACE_PARAMETER_INDEX]);
}
static inline int64_t loom_func_abi_c_type_attr_address_space(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ADDRESS_SPACE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_TYPE_PARAMETER_INDEX = 7 };
static inline bool loom_func_abi_c_type_attr_has_element_type(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_TYPE_PARAMETER_INDEX]);
}
static inline int64_t loom_func_abi_c_type_attr_element_type(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_TYPE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_COUNT_PARAMETER_INDEX = 8 };
static inline bool loom_func_abi_c_type_attr_has_element_count(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_COUNT_PARAMETER_INDEX]);
}
static inline int64_t loom_func_abi_c_type_attr_element_count(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_COUNT_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_STRIDE_PARAMETER_INDEX = 9 };
static inline bool loom_func_abi_c_type_attr_has_element_stride(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_STRIDE_PARAMETER_INDEX]);
}
static inline int64_t loom_func_abi_c_type_attr_element_stride(loom_attribute_t attr) {
  return loom_attr_as_i64(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_ELEMENT_STRIDE_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_NAME_PARAMETER_INDEX = 10 };
static inline bool loom_func_abi_c_type_attr_has_name(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_NAME_PARAMETER_INDEX]);
}
static inline loom_string_id_t loom_func_abi_c_type_attr_name(loom_attribute_t attr) {
  return loom_attr_as_string_id(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_NAME_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_TYPE_ATTR_FIELDS_PARAMETER_INDEX = 11 };
static inline bool loom_func_abi_c_type_attr_has_fields(loom_attribute_t attr) {
  return !loom_attr_is_absent(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_FIELDS_PARAMETER_INDEX]);
}
static inline loom_parameterized_attr_array_t loom_func_abi_c_type_attr_fields(loom_attribute_t attr) {
  return loom_attr_as_parameterized_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_TYPE_ATTR_FIELDS_PARAMETER_INDEX]);
}
iree_status_t loom_func_abi_c_type_attr_make(
    loom_module_t* module,
    loom_func_abi_c_type_attr_build_flags_t build_flags,
    loom_func_abi_c_type_kind_t kind,
    int64_t byte_length,
    int64_t byte_alignment,
    int64_t component_count,
    int64_t bitwidth,
    loom_func_abi_c_type_format_t format,
    int64_t address_space,
    int64_t element_type,
    int64_t element_count,
    int64_t element_stride,
    loom_string_id_t name,
    loom_parameterized_attr_array_t fields,
    loom_attribute_t* out_attr);

// C-compatible source signature retained independently of its flattened Loom value projection and selected output ABI.
static inline bool loom_func_abi_c_signature_attr_isa(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_PARAMETERIZED && loom_attr_as_parameterized_kind(attr) == LOOM_PARAMETERIZED_ATTR_FUNC_ABI_C_SIGNATURE;
}
enum { LOOM_FUNC_ABI_C_SIGNATURE_ATTR_ARGUMENTS_PARAMETER_INDEX = 0 };
static inline loom_i64_array_t loom_func_abi_c_signature_attr_arguments(loom_attribute_t attr) {
  return loom_attr_as_i64_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_SIGNATURE_ATTR_ARGUMENTS_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_SIGNATURE_ATTR_RESULTS_PARAMETER_INDEX = 1 };
static inline loom_i64_array_t loom_func_abi_c_signature_attr_results(loom_attribute_t attr) {
  return loom_attr_as_i64_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_SIGNATURE_ATTR_RESULTS_PARAMETER_INDEX]);
}
enum { LOOM_FUNC_ABI_C_SIGNATURE_ATTR_TYPES_PARAMETER_INDEX = 2 };
static inline loom_parameterized_attr_array_t loom_func_abi_c_signature_attr_types(loom_attribute_t attr) {
  return loom_attr_as_parameterized_array(loom_attr_as_parameterized_slots(attr)[LOOM_FUNC_ABI_C_SIGNATURE_ATTR_TYPES_PARAMETER_INDEX]);
}
iree_status_t loom_func_abi_c_signature_attr_make(
    loom_module_t* module,
    loom_i64_array_t arguments,
    loom_i64_array_t results,
    loom_parameterized_attr_array_t types,
    loom_attribute_t* out_attr);

enum {
  LOOM_OP_FUNC_DEF = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 0),
  LOOM_OP_FUNC_DECL = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 1),
  LOOM_OP_FUNC_CALL = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 2),
  LOOM_OP_FUNC_RETURN = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 3),
  LOOM_OP_FUNC_LOCATION = LOOM_OP_KIND(LOOM_DIALECT_FUNC, 4),
  LOOM_OP_FUNC_COUNT_ = 5,
};

// Symbol visibility. Absent (0) means private (module-internal).
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

// Execution temperature hint. Absent (0) means unspecified.
typedef enum loom_func_temperature_e {
  LOOM_FUNC_TEMPERATURE_HOT = 1,
  LOOM_FUNC_TEMPERATURE_COLD = 2,
  LOOM_FUNC_TEMPERATURE_COUNT_ = 3,
} loom_func_temperature_t;

// Private symbol retention policy. Absent (0) permits ordinary DCE.
typedef enum loom_func_retain_e {
  LOOM_FUNC_RETAIN_RETAIN = 1,
  LOOM_FUNC_RETAIN_COUNT_ = 2,
} loom_func_retain_t;

// LOOM_OP_FUNC_DEF: Function definition. Callable by name via func.call. Function arguments retain their initial values throughout the body, including in dependent result types and predicates. The entry block has no internal predecessors; CFG loops carry changing values on separate header blocks.
// func.def @negate(%input: f32) -> (f32) {
//   func.return %input : f32
// }
LOOM_DEFINE_ISA(loom_func_def_isa, LOOM_OP_FUNC_DEF)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_def_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_def_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_visibility, 1, loom_func_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_linkage, 2, loom_symbol_linkage_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_cc, 3, loom_func_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_effect_contract, 4, loom_callable_effect_contract_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_temperature, 5, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_inline_policy, 6, loom_inline_policy_t)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_func_def_predicates, 7)
LOOM_DEFINE_ATTR_PARAMETERIZED(loom_func_def_source_signature, 8)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_def_target, 9)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_abi, 10, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_func_def_abi_attrs, 11)
LOOM_DEFINE_ATTR_STRING(loom_func_def_export_symbol, 12)
LOOM_DEFINE_ATTR_DICT(loom_func_def_export_attrs, 13)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_def_retain, 14, loom_func_retain_t)
LOOM_DEFINE_REGION(loom_func_def_body, 0)
enum loom_func_def_build_flag_bits_e {
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_LINKAGE = 1u << 1,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_RETAIN = 1u << 2,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC = 1u << 3,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_EFFECT_CONTRACT = 1u << 4,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_TEMPERATURE = 1u << 5,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 6,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET = 1u << 7,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_ABI = 1u << 8,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 9,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_ABI_ATTRS = 1u << 10,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_ATTRS = 1u << 11,
  LOOM_FUNC_DEF_BUILD_FLAG_HAS_PREDICATES = 1u << 12,
};
typedef uint32_t loom_func_def_build_flags_t;
iree_status_t loom_func_def_build(
    loom_builder_t* builder,
    loom_func_def_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t linkage,
    loom_optional uint8_t retain,
    loom_optional uint8_t cc,
    loom_optional uint8_t effect_contract,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_optional loom_symbol_ref_t target,
    loom_optional loom_attribute_t source_signature,
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
LOOM_DEFINE_VARIADIC_OPERANDS(loom_func_decl_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_decl_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_visibility, 1, loom_func_visibility_t)
LOOM_DEFINE_ATTR_STRING(loom_func_decl_import_module, 2)
LOOM_DEFINE_ATTR_STRING(loom_func_decl_import_symbol, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_cc, 4, loom_func_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_effect_contract, 5, loom_callable_effect_contract_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_temperature, 6, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_inline_policy, 7, loom_inline_policy_t)
LOOM_DEFINE_ATTR_PARAMETERIZED(loom_func_decl_source_signature, 8)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_decl_target, 9)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_abi, 10, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_func_decl_abi_attrs, 11)
LOOM_DEFINE_ATTR_STRING(loom_func_decl_export_symbol, 12)
LOOM_DEFINE_ATTR_DICT(loom_func_decl_export_attrs, 13)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_func_decl_predicates, 14)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_decl_retain, 15, loom_func_retain_t)
enum loom_func_decl_build_flag_bits_e {
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_RETAIN = 1u << 1,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_MODULE = 1u << 2,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_SYMBOL = 1u << 3,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_CC = 1u << 4,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_EFFECT_CONTRACT = 1u << 5,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 6,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 7,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_TARGET = 1u << 8,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_ABI = 1u << 9,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 10,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_ABI_ATTRS = 1u << 11,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_ATTRS = 1u << 12,
  LOOM_FUNC_DECL_BUILD_FLAG_HAS_PREDICATES = 1u << 13,
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
    loom_optional uint8_t effect_contract,
    loom_optional uint8_t temperature,
    loom_optional uint8_t inline_policy,
    loom_optional loom_symbol_ref_t target,
    loom_optional loom_attribute_t source_signature,
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

// LOOM_OP_FUNC_CALL: Function-like symbol call. Runtime calls target func.def/func.decl; required-inline exact template calls are consumed before executable lowering.
// %r = func.call @add(%a, %b) : (f32, f32) -> (f32)
LOOM_DEFINE_ISA(loom_func_call_isa, LOOM_OP_FUNC_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_func_call_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_func_call_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_func_call_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_call_effect_contract, 1, loom_callable_effect_contract_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_call_temperature, 2, loom_func_temperature_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_func_call_inline_policy, 3, loom_inline_policy_t)
enum loom_func_call_build_flag_bits_e {
  LOOM_FUNC_CALL_BUILD_FLAG_HAS_EFFECT_CONTRACT = 1u << 0,
  LOOM_FUNC_CALL_BUILD_FLAG_HAS_TEMPERATURE = 1u << 1,
  LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY = 1u << 2,
};
typedef uint32_t loom_func_call_build_flags_t;
iree_status_t loom_func_call_build(
    loom_builder_t* builder,
    loom_func_call_build_flags_t build_flags,
    loom_optional uint8_t effect_contract,
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
loom_trait_flags_t loom_func_call_effective_traits(
    const loom_module_t* module, const loom_op_t* op);
iree_status_t loom_func_call_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_func_call_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

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

// LOOM_OP_FUNC_LOCATION: Materialize immutable captured source provenance as a read-only buffer. The required postorder node array ends with the root location; children refer to earlier nodes. All captured data is semantic and survives debug stripping. This operation never reads its own debug annotation. Equal complete captures may share executable rodata.
// %site = func.location [#func.location.file<"example.cc", range = [12, 3, 12, 28]>] : buffer
LOOM_DEFINE_ISA(loom_func_location_isa, LOOM_OP_FUNC_LOCATION)
LOOM_DEFINE_RESULT(loom_func_location_result, 0)
LOOM_DEFINE_ATTR_PARAMETERIZED_ARRAY(loom_func_location_nodes, 0)
iree_status_t loom_func_location_build(
    loom_builder_t* builder,
    loom_parameterized_attr_array_t nodes,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_func_location_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the func dialect.
const loom_op_vtable_t* const* loom_func_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the func dialect.
const loom_op_semantics_t* loom_func_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a func op kind, or empty metadata.
loom_op_semantics_t loom_func_op_semantics(
    loom_op_kind_t kind);

// Returns parameterized attribute descriptors for the func dialect.
const loom_parameterized_attr_descriptor_t* loom_func_dialect_parameterized_attrs(
    iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/ops/func/ops.inc"

#endif  // LOOM_OPS_FUNC_OPS_H_
