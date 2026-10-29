// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed attribute readers used by generated operation APIs.

#ifndef LOOM_OPS_ATTRIBUTE_ACCESSORS_H_
#define LOOM_OPS_ATTRIBUTE_ACCESSORS_H_

#include "loom/error/emitter.h"
#include "loom/ir/attribute_schema.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Defines a function that reads an i64 attribute by index.
#define LOOM_DEFINE_ATTR_I64(func_name, index)           \
  static inline int64_t func_name(const loom_op_t* op) { \
    return loom_attr_as_i64(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an f64 attribute by index.
#define LOOM_DEFINE_ATTR_F64(func_name, index)           \
  static inline double func_name(const loom_op_t* op) {  \
    return loom_attr_as_f64(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an f64-array attribute by index.
#define LOOM_DEFINE_ATTR_F64_ARRAY(func_name, index)              \
  static inline loom_f64_array_t func_name(const loom_op_t* op) { \
    return loom_attr_as_f64_array(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads an enum attribute by index.
// Returns the enum case index as a uint8_t.
#define LOOM_DEFINE_ATTR_ENUM(func_name, index)           \
  static inline uint8_t func_name(const loom_op_t* op) {  \
    return loom_attr_as_enum(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an enum attribute by index.
// Returns the enum case index as |enum_type|.
#define LOOM_DEFINE_ATTR_ENUM_TYPED(func_name, index, enum_type)     \
  static inline enum_type func_name(const loom_op_t* op) {           \
    return (enum_type)loom_attr_as_enum(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an enum array attribute by index.
#define LOOM_DEFINE_ATTR_ENUM_ARRAY(func_name, index)              \
  static inline loom_enum_array_t func_name(const loom_op_t* op) { \
    return loom_attr_as_enum_array(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads a signed enum-set attribute by index.
#define LOOM_DEFINE_ATTR_SIGNED_ENUM_SET(func_name, index)              \
  static inline loom_signed_enum_set_t func_name(const loom_op_t* op) { \
    return loom_attr_as_signed_enum_set(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads a representation-scoped enum by index.
#define LOOM_DEFINE_ATTR_SCOPED_ENUM(func_name, index)           \
  static inline uint32_t func_name(const loom_op_t* op) {        \
    return loom_attr_as_scoped_enum(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a symbol attribute by index.
#define LOOM_DEFINE_ATTR_SYMBOL(func_name, index)                  \
  static inline loom_symbol_ref_t func_name(const loom_op_t* op) { \
    return loom_attr_as_symbol(loom_op_attrs(op)[(index)]);        \
  }

// Defines a function that reads a symbol-array attribute by index.
#define LOOM_DEFINE_ATTR_SYMBOL_ARRAY(func_name, index)                  \
  static inline loom_symbol_ref_array_t func_name(const loom_op_t* op) { \
    return loom_attr_as_symbol_array(loom_op_attrs(op)[(index)]);        \
  }

// Defines a function that reads a symbol-set attribute by index.
#define LOOM_DEFINE_ATTR_SYMBOL_SET(func_name, index)                    \
  static inline loom_symbol_ref_array_t func_name(const loom_op_t* op) { \
    return loom_attr_as_symbol_set(loom_op_attrs(op)[(index)]);          \
  }

// Defines a function that reads a string attribute by index.
#define LOOM_DEFINE_ATTR_STRING(func_name, index)                 \
  static inline loom_string_id_t func_name(const loom_op_t* op) { \
    return loom_attr_as_string_id(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads a string-array attribute by index.
#define LOOM_DEFINE_ATTR_STRING_ARRAY(func_name, index)                 \
  static inline loom_string_id_array_t func_name(const loom_op_t* op) { \
    return loom_attr_as_string_array(loom_op_attrs(op)[(index)]);       \
  }

// Defines a function that reads a bool attribute by index.
#define LOOM_DEFINE_ATTR_BOOL(func_name, index)           \
  static inline bool func_name(const loom_op_t* op) {     \
    return loom_attr_as_bool(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a static encoding attribute by index.
#define LOOM_DEFINE_ATTR_ENCODING(func_name, index)              \
  static inline uint16_t func_name(const loom_op_t* op) {        \
    return loom_attr_as_encoding_id(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a byte payload attribute by index.
#define LOOM_DEFINE_ATTR_BYTES(func_name, index)                        \
  static inline iree_const_byte_span_t func_name(const loom_op_t* op) { \
    return loom_attr_as_bytes(loom_op_attrs(op)[(index)]);              \
  }

// Defines a function that reads a type-table attribute by index.
#define LOOM_DEFINE_ATTR_TYPE(func_name, index)                 \
  static inline loom_type_id_t func_name(const loom_op_t* op) { \
    return loom_attr_as_type_id(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads an i64 array attribute by index.
#define LOOM_DEFINE_ATTR_I64_ARRAY(func_name, index)              \
  static inline loom_attribute_t func_name(const loom_op_t* op) { \
    return loom_op_attrs(op)[(index)];                            \
  }

// Defines a function that reads a predicate list attribute by index.
#define LOOM_DEFINE_ATTR_PREDICATE_LIST(func_name, index)         \
  static inline loom_attribute_t func_name(const loom_op_t* op) { \
    return loom_op_attrs(op)[(index)];                            \
  }

// Defines a function that reads a DICT attribute by index.
#define LOOM_DEFINE_ATTR_DICT(func_name, index)                          \
  static inline loom_named_attr_slice_t func_name(const loom_op_t* op) { \
    return loom_attr_as_dict(loom_op_attrs(op)[(index)]);                \
  }

// Defines a function that reads a parameterized attribute by index.
#define LOOM_DEFINE_ATTR_PARAMETERIZED(func_name, index)          \
  static inline loom_attribute_t func_name(const loom_op_t* op) { \
    return loom_op_attrs(op)[(index)];                            \
  }

// Defines a function that reads a parameterized attribute array by index.
#define LOOM_DEFINE_ATTR_PARAMETERIZED_ARRAY(func_name, index)           \
  static inline loom_parameterized_attr_array_t func_name(               \
      const loom_op_t* op) {                                             \
    return loom_attr_as_parameterized_array(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a generic attribute payload by index.
#define LOOM_DEFINE_ATTR_ANY(func_name, index)                    \
  static inline loom_attribute_t func_name(const loom_op_t* op) { \
    return loom_op_attrs(op)[(index)];                            \
  }

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ATTRIBUTE_ACCESSORS_H_
