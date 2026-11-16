// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed type metadata shared by common and dialect-owned type
// registries.

#ifndef LOOM_IR_TYPE_DESCRIPTOR_H_
#define LOOM_IR_TYPE_DESCRIPTOR_H_

#include "iree/base/api.h"
#include "loom/ir/attribute_schema.h"
#include "loom/ir/semantics.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;

enum loom_parameterized_type_flag_bits_e {
  // A type whose optional parameters are all absent omits the entire <...>
  // parameter list in canonical text.
  LOOM_PARAMETERIZED_TYPE_OMIT_EMPTY_PARAMETER_LIST = 1u << 0,
};
typedef uint8_t loom_parameterized_type_flags_t;

// Generated metadata for one descriptor-backed type family.
typedef struct loom_parameterized_type_descriptor_t {
  // Immutable owning type descriptor, including its name and text format. The
  // owner and parameter schema share a lifetime and refer to each other.
  const struct loom_type_descriptor_t* type_descriptor;

  // Parameter descriptors in stable declaration order.
  const loom_attr_descriptor_t* parameter_descriptors;

  // Type-owned facts used by joins and transport of this parameterized family.
  const struct loom_value_fact_domain_t* fact_domain;

  // Runtime type kind carrying the parameters. LOOM_TYPE_PARAMETERIZED uses
  // indirect immutable slots; compact kinds use their own representation.
  loom_type_kind_t ir_kind;

  // loom_type_flags_t bits used by a compact inline-enum representation.
  uint8_t type_flags;

  // Number of descriptor-indexed parameter slots.
  uint8_t parameter_count;

  // Text and representation behavior bits.
  loom_parameterized_type_flags_t flags;

  // Exact-build generated indirect family identity, or NONE for compact
  // representations and handwritten descriptors without a generated family.
  loom_parameterized_type_kind_t kind;
  // Logical handle mode for LOOM_TYPE_REFERENCE, otherwise zero.
  loom_reference_ownership_t reference_ownership;
} loom_parameterized_type_descriptor_t;

static_assert(sizeof(loom_parameterized_type_descriptor_t) ==
                  (IREE_PTR_SIZE == 8 ? 32 : 20),
              "parameterized type descriptor must remain compact");

// Format element kinds for type interiors inside `<...>`.
typedef enum loom_type_format_kind_e {
  // Dimension list such as `4x[%M]x...`.
  LOOM_TYPE_FMT_SHAPE = 0,
  // Element type keyword such as `f32` or `i8`.
  LOOM_TYPE_FMT_SCALAR = 1,
  // Encoding reference such as `#test.schema` or `%encoding`.
  LOOM_TYPE_FMT_ENCODING = 2,
  // Recursive type such as `test.ref<T>`.
  LOOM_TYPE_FMT_TYPE = 3,
  // Bare identifier attribute.
  LOOM_TYPE_FMT_ATTR = 4,
  // Literal punctuation or keyword.
  LOOM_TYPE_FMT_KEYWORD = 5,
  // Conditional element range.
  LOOM_TYPE_FMT_OPTIONAL = 6,
  // Suppresses whitespace before the next element.
  LOOM_TYPE_FMT_GLUE = 7,
  // Descriptor-backed attribute value.
  LOOM_TYPE_FMT_PARAM = 8,
  // Descriptor-backed parameter name.
  LOOM_TYPE_FMT_PARAM_KEY = 9,
  // Explicit view element-access alignment in bytes.
  LOOM_TYPE_FMT_ALIGNMENT = 10,
} loom_type_format_kind_t;

// Compact format instruction for a descriptor-backed type interior.
typedef struct loom_type_format_element_t {
  // Format opcode encoded as a `loom_type_format_kind_t`.
  uint8_t kind;
  // Parameter index consumed by this element.
  uint8_t field_index;
  // Kind-specific payload such as a keyword ID or skip count.
  uint16_t data;
} loom_type_format_element_t;

// Descriptor type optional groups use the high bit for polarity, the next
// seven bits for the child count, and the low byte for the anchor parameter.
#define LOOM_TYPE_FORMAT_OPTIONAL_INVERTED ((uint16_t)(1u << 15))
#define LOOM_TYPE_FORMAT_OPTIONAL_DATA(skip_count, anchor_index, inverted) \
  ((uint16_t)((((uint16_t)(skip_count) & 0x7Fu) << 8) |                    \
              (uint16_t)(anchor_index) |                                   \
              ((inverted) ? LOOM_TYPE_FORMAT_OPTIONAL_INVERTED : 0u)))
#define LOOM_TYPE_FORMAT_OPTIONAL_SKIP_COUNT(data) \
  ((uint16_t)(((data) >> 8) & 0x7Fu))
#define LOOM_TYPE_FORMAT_OPTIONAL_IS_INVERTED(data) \
  iree_any_bit_set((data), LOOM_TYPE_FORMAT_OPTIONAL_INVERTED)

// External identity of an exact managed reference type. Names are nonempty,
// NUL-free UTF-8 and independent of the type's source spelling. Providers bind
// this identity to their native ownership implementation; compiler metadata
// never contains runtime object or descriptor pointers.
typedef struct loom_type_reference_key_t {
  // Namespace owned by the external type provider.
  iree_string_view_t namespace_name;
  // Exact type name within the provider namespace.
  iree_string_view_t type_name;
} loom_type_reference_key_t;

// Generated metadata for one registered type spelling and representation.
typedef struct loom_type_descriptor_t {
  // B-string name with a trailing `<` outside its declared length.
  const uint8_t* name;
  // Runtime IR type kind constructed when parsing.
  loom_type_kind_t ir_kind;
  // Number of declared parameters.
  uint8_t param_count;
  // Optional type-owned value fact domain.
  const loom_value_fact_domain_t* fact_domain;
  // Semantic role and target-contract families for this type.
  loom_type_semantics_t semantics;
  // Format instructions for the type interior, or NULL for opaque types.
  const loom_type_format_element_t* format_elements;
  // Number of entries in `format_elements`.
  uint8_t format_element_count;
  // Descriptor-backed parameter schema whose type_descriptor refers to this
  // descriptor, or NULL when not declared.
  const loom_parameterized_type_descriptor_t* parameterized;
  // Static payload identity for an opaque managed dialect type, otherwise NULL.
  // Declaration-backed ref/ptr types carry their resolved identity in the type
  // value instead of sharing one payload key across the family descriptor.
  const loom_type_reference_key_t* reference;
} loom_type_descriptor_t;

// One name-to-descriptor row in a generated type registry table.
typedef struct loom_type_registry_entry_t {
  // Stable public type spelling.
  iree_string_view_t name;
  // Generated descriptor for `name`.
  const loom_type_descriptor_t* descriptor;
} loom_type_registry_entry_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_TYPE_DESCRIPTOR_H_
