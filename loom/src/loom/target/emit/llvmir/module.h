// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured LLVM IR module model.
//
// This is Loom's target-side LLVM IR representation. It stores types, values,
// functions, blocks, instructions, attributes, and metadata as structured
// records so text and future bitcode writers can serialize the same object
// model without reparsing printed LLVM syntax.

#ifndef LOOM_TARGET_LLVMIR_MODULE_H_
#define LOOM_TARGET_LLVMIR_MODULE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LLVMIR_TYPE_ID_INVALID ((uint32_t)UINT32_MAX)
#define LOOM_LLVMIR_VALUE_ID_INVALID ((uint32_t)UINT32_MAX)
#define LOOM_LLVMIR_BLOCK_ID_INVALID ((uint32_t)UINT32_MAX)
#define LOOM_LLVMIR_FUNCTION_ID_INVALID ((uint32_t)UINT32_MAX)
#define LOOM_LLVMIR_GLOBAL_ID_INVALID ((uint32_t)UINT32_MAX)
#define LOOM_LLVMIR_ATTR_GROUP_ID_INVALID ((uint32_t)UINT32_MAX)
#define LOOM_LLVMIR_METADATA_ID_INVALID ((uint32_t)UINT32_MAX)

typedef uint32_t loom_llvmir_type_id_t;
typedef uint32_t loom_llvmir_value_id_t;
typedef uint32_t loom_llvmir_block_id_t;
typedef uint32_t loom_llvmir_function_id_t;
typedef uint32_t loom_llvmir_global_id_t;
typedef uint32_t loom_llvmir_attr_group_id_t;
typedef uint32_t loom_llvmir_metadata_id_t;

typedef struct loom_llvmir_module_t loom_llvmir_module_t;
typedef struct loom_llvmir_function_t loom_llvmir_function_t;
typedef struct loom_llvmir_global_t loom_llvmir_global_t;
typedef struct loom_llvmir_block_t loom_llvmir_block_t;

typedef enum loom_llvmir_float_kind_e {
  LOOM_LLVMIR_FLOAT_F16 = 0,
  LOOM_LLVMIR_FLOAT_BF16 = 1,
  LOOM_LLVMIR_FLOAT_F32 = 2,
  LOOM_LLVMIR_FLOAT_F64 = 3,
} loom_llvmir_float_kind_t;

typedef enum loom_llvmir_type_kind_e {
  LOOM_LLVMIR_TYPE_VOID = 0,
  LOOM_LLVMIR_TYPE_INTEGER = 1,
  LOOM_LLVMIR_TYPE_FLOAT = 2,
  LOOM_LLVMIR_TYPE_POINTER = 3,
  LOOM_LLVMIR_TYPE_VECTOR = 4,
  LOOM_LLVMIR_TYPE_STRUCT = 5,
} loom_llvmir_type_kind_t;

typedef enum loom_llvmir_value_kind_e {
  LOOM_LLVMIR_VALUE_GLOBAL = 0,
  LOOM_LLVMIR_VALUE_PARAMETER = 1,
  LOOM_LLVMIR_VALUE_CONSTANT_INTEGER = 2,
  LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS = 3,
  LOOM_LLVMIR_VALUE_CONSTANT_NULL = 4,
  LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR = 5,
  LOOM_LLVMIR_VALUE_CONSTANT_POISON = 6,
  LOOM_LLVMIR_VALUE_INSTRUCTION = 7,
} loom_llvmir_value_kind_t;

typedef enum loom_llvmir_function_kind_e {
  LOOM_LLVMIR_FUNCTION_DECLARATION = 0,
  LOOM_LLVMIR_FUNCTION_DEFINITION = 1,
} loom_llvmir_function_kind_t;

typedef enum loom_llvmir_linkage_e {
  LOOM_LLVMIR_LINKAGE_DEFAULT = 0,
  LOOM_LLVMIR_LINKAGE_DSO_LOCAL = 1,
  LOOM_LLVMIR_LINKAGE_INTERNAL = 2,
  LOOM_LLVMIR_LINKAGE_PRIVATE = 3,
} loom_llvmir_linkage_t;

typedef enum loom_llvmir_calling_convention_e {
  LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT = 0,
  LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL = 1,
} loom_llvmir_calling_convention_t;

typedef enum loom_llvmir_attr_kind_e {
  LOOM_LLVMIR_ATTR_ALIGN = 0,
  LOOM_LLVMIR_ATTR_NOALIAS = 1,
  LOOM_LLVMIR_ATTR_READONLY = 2,
  LOOM_LLVMIR_ATTR_WRITEONLY = 3,
  LOOM_LLVMIR_ATTR_READNONE = 4,
  LOOM_LLVMIR_ATTR_NOUNDEF = 5,
  LOOM_LLVMIR_ATTR_NONNULL = 6,
  LOOM_LLVMIR_ATTR_INREG = 7,
  LOOM_LLVMIR_ATTR_ALWAYSINLINE = 8,
  LOOM_LLVMIR_ATTR_RANGE = 9,
  LOOM_LLVMIR_ATTR_NOCAPTURE = 10,
  LOOM_LLVMIR_ATTR_IMMARG = 11,
  LOOM_LLVMIR_ATTR_STRING_KEY = 12,
  LOOM_LLVMIR_ATTR_STRING_KEY_VALUE = 13,
} loom_llvmir_attr_kind_t;

typedef struct loom_llvmir_attr_t {
  // Attribute kind selecting the active payload.
  loom_llvmir_attr_kind_t kind;
  // Integer payload for ALIGN, or lower bound for RANGE.
  uint64_t value;
  // Upper bound for RANGE.
  uint64_t value2;
  // Scalar type used by RANGE.
  loom_llvmir_type_id_t type_id;
  // String key for target/string attributes.
  iree_string_view_t key;
  // String value for key/value attributes.
  iree_string_view_t string_value;
} loom_llvmir_attr_t;

typedef struct loom_llvmir_parameter_desc_t {
  // Parameter result type.
  loom_llvmir_type_id_t type_id;
  // Optional debug/text name without the leading percent sign.
  iree_string_view_t name;
  // Structured parameter attributes.
  const loom_llvmir_attr_t* attrs;
  // Number of entries in |attrs|.
  iree_host_size_t attr_count;
} loom_llvmir_parameter_desc_t;

typedef struct loom_llvmir_target_config_t {
  // Optional LLVM source_filename module field.
  iree_string_view_t source_name;
  // Optional LLVM target triple.
  iree_string_view_t target_triple;
  // Optional LLVM data layout.
  iree_string_view_t data_layout;
  // Optional producer string for future metadata/debug output.
  iree_string_view_t producer;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
} loom_llvmir_target_config_t;

typedef struct loom_llvmir_function_desc_t {
  // Function declaration or definition.
  loom_llvmir_function_kind_t kind;
  // Function symbol name without the leading at sign.
  iree_string_view_t name;
  // LLVM return type.
  loom_llvmir_type_id_t return_type;
  // Linkage/preemption token.
  loom_llvmir_linkage_t linkage;
  // Calling convention token.
  loom_llvmir_calling_convention_t calling_convention;
  // Numbered module attribute group, or INVALID for no group.
  loom_llvmir_attr_group_id_t attr_group_id;
} loom_llvmir_function_desc_t;

typedef struct loom_llvmir_global_desc_t {
  // Global symbol name without the leading at sign.
  iree_string_view_t name;
  // Linkage/preemption token.
  loom_llvmir_linkage_t linkage;
  // Stored value type.
  loom_llvmir_type_id_t value_type;
  // Pointer address space for references to this global.
  uint32_t address_space;
  // True when the global is immutable LLVM constant storage.
  bool is_constant;
  // Constant initializer value.
  loom_llvmir_value_id_t initializer;
  // Optional byte alignment, or zero to omit the attribute.
  uint32_t alignment;
} loom_llvmir_global_desc_t;

typedef struct loom_llvmir_metadata_i32_tuple_t {
  // Integer values in tuple order.
  const int32_t* values;
  // Number of entries in |values|.
  iree_host_size_t value_count;
} loom_llvmir_metadata_i32_tuple_t;

typedef struct loom_llvmir_metadata_attachment_t {
  // Metadata attachment name without the leading exclamation mark.
  iree_string_view_t name;
  // Numbered metadata node referenced by the attachment.
  loom_llvmir_metadata_id_t metadata_id;
} loom_llvmir_metadata_attachment_t;

// Creates a new empty LLVM IR module. The module owns all copied strings and
// child records until loom_llvmir_module_free().
iree_status_t loom_llvmir_module_allocate(
    const loom_llvmir_target_config_t* target_config,
    iree_allocator_t allocator, loom_llvmir_module_t** out_module);

void loom_llvmir_module_free(loom_llvmir_module_t* module);

iree_status_t loom_llvmir_module_get_void_type(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_module_get_integer_type(
    loom_llvmir_module_t* module, uint32_t bit_width,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_module_get_float_type(
    loom_llvmir_module_t* module, loom_llvmir_float_kind_t float_kind,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_module_get_pointer_type(
    loom_llvmir_module_t* module, uint32_t address_space,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_module_get_vector_type(
    loom_llvmir_module_t* module, uint32_t element_count,
    loom_llvmir_type_id_t element_type_id, loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_module_get_struct_type(
    loom_llvmir_module_t* module, const loom_llvmir_type_id_t* element_type_ids,
    iree_host_size_t element_count, loom_llvmir_type_id_t* out_type_id);

// Queries the element type and lane count of a VECTOR type.
iree_status_t loom_llvmir_module_get_vector_type_info(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id,
    uint32_t* out_element_count, loom_llvmir_type_id_t* out_element_type_id);

iree_status_t loom_llvmir_module_add_integer_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id, uint64_t value,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_module_add_integer_vector_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t vector_type_id,
    const uint64_t* values, iree_host_size_t value_count,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_module_add_float_bits_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id, uint64_t bits,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_module_add_null_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_module_add_poison_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_module_add_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_attr_t* attrs,
    iree_host_size_t attr_count, loom_llvmir_attr_group_id_t* out_group_id);

iree_status_t loom_llvmir_module_add_metadata_i32_tuple(
    loom_llvmir_module_t* module,
    const loom_llvmir_metadata_i32_tuple_t* metadata,
    loom_llvmir_metadata_id_t* out_metadata_id);

iree_status_t loom_llvmir_module_add_global(
    loom_llvmir_module_t* module, const loom_llvmir_global_desc_t* desc,
    loom_llvmir_global_t** out_global);

loom_llvmir_global_id_t loom_llvmir_global_id(
    const loom_llvmir_global_t* global);

loom_llvmir_value_id_t loom_llvmir_global_value_id(
    const loom_llvmir_global_t* global);

iree_status_t loom_llvmir_module_add_function(
    loom_llvmir_module_t* module, const loom_llvmir_function_desc_t* desc,
    loom_llvmir_function_t** out_function);

// Finds a function declaration or definition by symbol name.
loom_llvmir_function_t* loom_llvmir_module_find_function(
    loom_llvmir_module_t* module, iree_string_view_t name);

loom_llvmir_function_id_t loom_llvmir_function_id(
    const loom_llvmir_function_t* function);

// Returns the parent module that owns |function|, or NULL for NULL.
loom_llvmir_module_t* loom_llvmir_function_module(
    const loom_llvmir_function_t* function);

iree_status_t loom_llvmir_function_add_parameter(
    loom_llvmir_function_t* function, const loom_llvmir_parameter_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_function_add_metadata_attachment(
    loom_llvmir_function_t* function,
    const loom_llvmir_metadata_attachment_t* attachment);

iree_status_t loom_llvmir_function_add_block(loom_llvmir_function_t* function,
                                             iree_string_view_t name,
                                             loom_llvmir_block_t** out_block);

loom_llvmir_block_id_t loom_llvmir_block_id(const loom_llvmir_block_t* block);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_MODULE_H_
