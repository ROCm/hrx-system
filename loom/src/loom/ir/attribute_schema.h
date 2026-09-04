// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed attribute field schemas shared by operations,
// parameterized attributes, and descriptor-backed types.

#ifndef LOOM_IR_ATTRIBUTE_SCHEMA_H_
#define LOOM_IR_ATTRIBUTE_SCHEMA_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_attr_flag_bits_e {
  LOOM_ATTR_OPTIONAL = 1u << 0,
  // Enum values are ordinal-preserving across bytecode and generic
  // verification. Consumers still own sentinel and supported-case policy.
  LOOM_ATTR_OPEN_ENUM = 1u << 1,
  // Text formats may omit this required scalar attribute when the present
  // value equals the zero/false default. Parsers restore the explicit value.
  LOOM_ATTR_ELIDE_DEFAULT = 1u << 2,
  // Descriptor-aware text formats spell string values as bare identifiers.
  LOOM_ATTR_BARE_IDENTIFIER = 1u << 3,
};
typedef uint8_t loom_attr_flags_t;

typedef uint32_t loom_symbol_interface_flags_t;

enum loom_symbol_interface_bits_e {
  // Symbol implements the generated function-like interface.
  LOOM_SYMBOL_INTERFACE_FUNC_LIKE = 1u << 0,
  // Symbol implements the generated global-like contract.
  LOOM_SYMBOL_INTERFACE_GLOBAL = 1u << 1,
  // Symbol names a target executable/package-like entity.
  LOOM_SYMBOL_INTERFACE_EXECUTABLE = 1u << 2,
  // Symbol names a generic module-level record.
  LOOM_SYMBOL_INTERFACE_RECORD = 1u << 3,
  // Symbol names a target environment record.
  LOOM_SYMBOL_INTERFACE_TARGET = 1u << 4,
  // Symbol names a compile/link-time configuration value.
  LOOM_SYMBOL_INTERFACE_CONFIG = 1u << 5,
  // Symbol names a read-only executable data payload.
  LOOM_SYMBOL_INTERFACE_RODATA = 1u << 6,
  // Symbol defines or declares a host-launchable kernel contract.
  LOOM_SYMBOL_INTERFACE_KERNEL = 1u << 7,
  // Function-like symbol may be targeted by an ordinary call operation.
  LOOM_SYMBOL_INTERFACE_CALLABLE = 1u << 8,
  // Function-like symbol defines or declares a reusable command program.
  LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM = 1u << 9,
  // Symbol defines an abstract compile-time callable family.
  LOOM_SYMBOL_INTERFACE_TEMPLATE_FAMILY = 1u << 10,
  // Symbol defines a concrete compile-time template implementation.
  LOOM_SYMBOL_INTERFACE_TEMPLATE_PROVIDER = 1u << 11,
  // Symbol defines or declares an executable kernel entry ABI.
  LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY = 1u << 12,
  // Function-like symbol defines or declares a persistent pipeline program.
  LOOM_SYMBOL_INTERFACE_PIPELINE = 1u << 13,
  // All symbol interface bits understood by this compiler version.
  LOOM_SYMBOL_INTERFACE_FLAG_MASK = (1u << 14) - 1,
};

// Compact operation-local product classification for durable symbol roots.
// Enum-backed carriers occupy [0, UINT8_MAX]. The wider representation leaves
// one sentinel for symbol definitions that have no product-carrier contract.
typedef uint16_t loom_symbol_product_carrier_t;
#define LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED \
  ((loom_symbol_product_carrier_t)UINT16_MAX)

enum loom_symbol_reference_role_e {
  // Zero/default: the reference contributes to reachability and link closure.
  LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY = 0,
  // The reference records where a symbol may be found without retaining it.
  LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY = 1,
};
typedef uint8_t loom_symbol_reference_role_t;

// Generated metadata for a symbol-reference attribute.
typedef struct loom_symbol_reference_descriptor_t {
  // Human-readable expected symbol class used in diagnostics.
  loom_bstring_t name;
  // Structural symbol interfaces accepted by this reference.
  loom_symbol_interface_flags_t interfaces;
  // Compile-time graph role of each reference occurrence.
  loom_symbol_reference_role_t role;
} loom_symbol_reference_descriptor_t;

static_assert(sizeof(loom_symbol_reference_descriptor_t) == 16,
              "symbol reference descriptors must remain 16 bytes");

static inline iree_string_view_t loom_symbol_reference_descriptor_name(
    const loom_symbol_reference_descriptor_t* descriptor) {
  return descriptor ? loom_bstring_view(descriptor->name) : IREE_SV("symbol");
}

// Descriptor for one named parameter or operation attribute field.
typedef struct loom_attr_descriptor_t {
  // Author-facing field name used in diagnostics and stable formats.
  loom_bstring_t name;
  // Runtime attribute payload kind.
  loom_attr_kind_t attr_kind;
  // Attribute structural flags such as optional.
  loom_attr_flags_t flags;
  // Largest valid index in |enum_case_names|, or 0 when the table is NULL.
  uint8_t enum_max_value;
  // Dense enum value to keyword table, or NULL for non-enum attrs.
  const loom_bstring_t* enum_case_names;
  // Mutually exclusive reference contract selected by |attr_kind|.
  union {
    // Expected symbol target contract for SYMBOL, SYMBOL_ARRAY, and SYMBOL_SET
    // attributes.
    const loom_symbol_reference_descriptor_t* symbol_ref;
    // Expected family kind for PARAMETERIZED attributes and every element of
    // PARAMETERIZED_ARRAY attributes, or LOOM_PARAMETERIZED_ATTR_KIND_ANY when
    // open.
    loom_parameterized_attr_kind_t parameterized_attr_kind;
  } reference;
} loom_attr_descriptor_t;

// Returns the number of sparse enum keyword slots owned by |descriptor|.
static inline iree_host_size_t loom_attr_descriptor_enum_case_span(
    const loom_attr_descriptor_t* descriptor) {
  return descriptor && descriptor->enum_case_names
             ? (iree_host_size_t)descriptor->enum_max_value + 1
             : 0;
}

// Returns the keyword for |value| or NULL when the value is not declared.
static inline loom_bstring_t loom_attr_descriptor_enum_case_name(
    const loom_attr_descriptor_t* descriptor, uint8_t value) {
  return descriptor && descriptor->enum_case_names &&
                 value <= descriptor->enum_max_value
             ? descriptor->enum_case_names[value]
             : NULL;
}

// Returns true when |value| is declared by |descriptor|.
static inline bool loom_attr_descriptor_has_enum_case(
    const loom_attr_descriptor_t* descriptor, uint8_t value) {
  return loom_attr_descriptor_enum_case_name(descriptor, value) != NULL;
}

// Resolves a declared enum keyword to its byte value. Returns false when the
// keyword is not declared by |descriptor|.
static inline bool loom_attr_descriptor_find_enum_case(
    const loom_attr_descriptor_t* descriptor, iree_string_view_t keyword,
    uint8_t* out_value) {
  const iree_host_size_t case_span =
      loom_attr_descriptor_enum_case_span(descriptor);
  for (iree_host_size_t i = 0; i < case_span; ++i) {
    loom_bstring_t case_name = descriptor->enum_case_names[i];
    if (case_name && loom_bstring_equal(case_name, keyword)) {
      *out_value = (uint8_t)i;
      return true;
    }
  }
  return false;
}

// Returns the attribute name as a string view.
static inline iree_string_view_t loom_attr_descriptor_name(
    const loom_attr_descriptor_t* descriptor) {
  return loom_bstring_view(descriptor->name);
}

// Finds a descriptor by its stable field name. Descriptor arrays are small
// schema metadata tables; callers retain the ordinal when they need repeated
// access to the selected field.
static inline const loom_attr_descriptor_t* loom_attr_descriptor_find_by_name(
    const loom_attr_descriptor_t* descriptors, uint8_t descriptor_count,
    iree_string_view_t name, uint8_t* out_descriptor_index) {
  for (uint8_t i = 0; i < descriptor_count; ++i) {
    if (iree_string_view_equal(loom_attr_descriptor_name(&descriptors[i]),
                               name)) {
      *out_descriptor_index = i;
      return &descriptors[i];
    }
  }
  return NULL;
}

// Returns true when |kind| satisfies the payload-kind contract of
// |descriptor|. Open ANY fields still exclude descriptor-dependent payloads
// that cannot be interpreted without their owning field schema.
static inline bool loom_attr_descriptor_accepts_kind(
    const loom_attr_descriptor_t* descriptor, loom_attr_kind_t kind) {
  if (descriptor->attr_kind != LOOM_ATTR_ANY) {
    return kind == descriptor->attr_kind;
  }
  return kind > LOOM_ATTR_ABSENT && kind < LOOM_ATTR_COUNT_ &&
         kind != LOOM_ATTR_ANY && kind != LOOM_ATTR_SCOPED_ENUM &&
         kind != LOOM_ATTR_ENUM_ARRAY && kind != LOOM_ATTR_SIGNED_ENUM_SET &&
         kind != LOOM_ATTR_PARAMETERIZED_ARRAY &&
         kind != LOOM_ATTR_SYMBOL_ARRAY && kind != LOOM_ATTR_SYMBOL_SET;
}

// Returns the explicit zero/false scalar value implied by ELIDE_DEFAULT.
static inline loom_attribute_t loom_attr_descriptor_default_value(
    const loom_attr_descriptor_t* descriptor) {
  switch ((loom_attr_kind_t)descriptor->attr_kind) {
    case LOOM_ATTR_I64:
      return loom_attr_i64(0);
    case LOOM_ATTR_BOOL:
      return loom_attr_bool(false);
    default:
      return loom_attr_absent();
  }
}

// Returns true when |attr| is the elidable text default for |descriptor|.
static inline bool loom_attr_descriptor_elides_value(
    const loom_attr_descriptor_t* descriptor, const loom_attribute_t* attr) {
  if (!iree_any_bit_set(descriptor->flags, LOOM_ATTR_ELIDE_DEFAULT)) {
    return false;
  }
  loom_attribute_t default_value =
      loom_attr_descriptor_default_value(descriptor);
  return !loom_attr_is_absent(default_value) &&
         loom_attribute_equal(attr, &default_value);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_ATTRIBUTE_SCHEMA_H_
