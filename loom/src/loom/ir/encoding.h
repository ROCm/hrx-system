// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static encoding records and context-owned encoding family vtables.
//
// Tile/tensor encodings and view layouts carry either:
//   - a dynamic encoding SSA binding (`%enc`) in `loom_type_t.encoding_id`
//     with `LOOM_ENCODING_FLAG_SSA`, or
//   - a 1-based index into a module-owned `loom_encoding_table_t`.
// Vector types are shaped but intentionally cannot carry this attachment slot.
// Encoding SSA values may use role-qualified types such as `encoding<layout>`
// and `encoding<schema>` so producers and consumers declare the exact encoding
// category they accept at the type boundary.
//
// Each static module encoding entry is identified by `(name_id, attributes)`.
// `alias_id` is a display-only spelling hint used by the text printer and is
// deliberately excluded from structural equality and hashing.

#ifndef LOOM_IR_ENCODING_H_
#define LOOM_IR_ENCODING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/attribute_schema.h"
#include "loom/ir/type_constraint.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_op_t loom_op_t;
typedef struct loom_encoding_define_param_view_t
    loom_encoding_define_param_view_t;

// Context-local identity for a registered encoding family. IDs are dense and
// one-based so zero can represent an unregistered family in permissive
// contexts. This value never serializes.
typedef uint16_t loom_encoding_family_id_t;

#define LOOM_ENCODING_FAMILY_ID_INVALID ((loom_encoding_family_id_t)0)

enum loom_encoding_family_flag_bits_e {
  // Every present static parameter is bound to a family descriptor and has
  // the descriptor's declared payload kind.
  LOOM_ENCODING_FAMILY_STATIC_PARAMETERS_VALID = 1u << 0,
  // Family-specific semantics of the structurally valid parameters hold.
  LOOM_ENCODING_FAMILY_STATIC_SEMANTICS_VALID = 1u << 1,
};
typedef uint8_t loom_encoding_family_flags_t;

// A single static encoding instance, such as `#q8_0<block=32>`.
//
// The encoding family name and optional file-local alias are interned string
// IDs in the owning module. Parameters are named attributes so families can
// carry structured values such as integers, strings, arrays, and dicts.
struct loom_encoding_t {
  // Interned public family name in the owning module.
  loom_string_id_t name_id;
  // File-local alias spelling without '#', or LOOM_STRING_ID_INVALID when no
  // alias should be preferred for printing.
  loom_string_id_t alias_id;
  // Number of canonical sparse entries in |attributes|.
  uint8_t attribute_count;
  // Context-local family binding derived during module construction.
  struct {
    // Dense one-based identity in the owning context's encoding registry.
    // This never participates in structural identity or serialization.
    loom_encoding_family_id_t id;
    // Construction facts derived from the registered family descriptor.
    loom_encoding_family_flags_t flags;
  } family;
  // Arena-owned canonical parameter array. NULL when attribute_count is 0.
  const loom_named_attr_t* attributes;
};

typedef struct loom_encoding_t loom_encoding_t;

static_assert(sizeof(loom_encoding_t) == 24,
              "loom_encoding_t must remain 24 bytes");

// Sentinel returned for an encoding parameter that is not bound to a family
// descriptor. Unbound parameters are retained until an input boundary emits
// the corresponding unknown-parameter diagnostic.
#define LOOM_ENCODING_PARAMETER_INDEX_INVALID UINT8_MAX

// Returns the zero-based family descriptor index bound to |parameter|, or
// LOOM_ENCODING_PARAMETER_INDEX_INVALID when the parameter is unbound.
static inline uint8_t loom_encoding_parameter_descriptor_index(
    const loom_named_attr_t* parameter) {
  const uint8_t ordinal = (uint8_t)parameter->reserved;
  return ordinal == 0 ? LOOM_ENCODING_PARAMETER_INDEX_INVALID
                      : (uint8_t)(ordinal - 1);
}

// Binds |parameter| to a zero-based family descriptor index. Descriptor
// metadata is non-semantic and never participates in text, bytecode, hashing,
// or equality.
static inline void loom_encoding_parameter_bind_descriptor(
    loom_named_attr_t* parameter, uint8_t descriptor_index) {
  IREE_ASSERT(descriptor_index < UINT8_MAX);
  parameter->reserved = (uint32_t)descriptor_index + 1;
}

// Returns true when every present static parameter satisfies its registered
// family schema. Unregistered permissive encodings never carry this fact.
static inline bool loom_encoding_static_parameters_are_valid(
    const loom_encoding_t* encoding) {
  return iree_all_bits_set(encoding->family.flags,
                           LOOM_ENCODING_FAMILY_STATIC_PARAMETERS_VALID);
}

// Returns true when the registered family has accepted both the generated
// parameter schema and its family-specific static semantics.
static inline bool loom_encoding_static_is_valid(
    const loom_encoding_t* encoding) {
  return iree_all_bits_set(encoding->family.flags,
                           LOOM_ENCODING_FAMILY_STATIC_PARAMETERS_VALID |
                               LOOM_ENCODING_FAMILY_STATIC_SEMANTICS_VALID);
}

// Materializes the sparse parameter array into caller-owned descriptor-indexed
// slots. The encoding must have passed its registered static parameter schema,
// and |slot_count| must match that family's descriptor count. This is transient
// query state and never becomes part of encoding identity or module storage.
static inline void loom_encoding_collect_parameter_slots(
    const loom_encoding_t* encoding, uint8_t slot_count,
    const loom_named_attr_t** out_slots) {
  IREE_ASSERT(loom_encoding_static_parameters_are_valid(encoding));
  for (uint8_t i = 0; i < slot_count; ++i) out_slots[i] = NULL;
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* parameter = &encoding->attributes[i];
    const uint8_t descriptor_index =
        loom_encoding_parameter_descriptor_index(parameter);
    IREE_ASSERT(descriptor_index < slot_count);
    out_slots[descriptor_index] = parameter;
  }
}

// Describes one accepted dynamic parameter on encoding.define.
typedef struct loom_encoding_dynamic_parameter_descriptor_t {
  // Stable public parameter name.
  loom_bstring_t name;
  // Broad SSA type constraint checked before family semantics.
  loom_type_constraint_t type_constraint;
} loom_encoding_dynamic_parameter_descriptor_t;

// Generated structural metadata for one registered encoding family.
typedef struct loom_encoding_family_descriptor_t {
  // Stable public family name without a leading '#'.
  loom_bstring_t name;
  // Semantic role carried by instances of the family.
  loom_encoding_role_t role;
  // Number of lexically ordered descriptors in |parameter_descriptors|.
  uint8_t parameter_count;
  // Lexically ordered static parameter descriptors, or NULL when empty.
  const loom_attr_descriptor_t* parameter_descriptors;
  // Number of lexically ordered descriptors in |dynamic_parameter_descriptors|.
  uint8_t dynamic_parameter_count;
  // Lexically ordered dynamic parameter descriptors, or NULL when empty.
  const loom_encoding_dynamic_parameter_descriptor_t*
      dynamic_parameter_descriptors;
} loom_encoding_family_descriptor_t;

// A module-owned table of unique static encoding instances.
typedef struct loom_encoding_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_encoding_t* entries;
} loom_encoding_table_t;

// Vtable for one encoding family (`q6_k`, `q8_0`, `dense`, etc.).
//
// Module encoding entries store only static family name + canonical parameter
// attrs. Dynamic parameters are ordinary SSA operands on encoding.define, named
// by OperandDict metadata so the merged parameter view is explicit in the IR
// instead of hidden inside attribute payloads.
//
// The context-owned family vtable supplies runtime hooks for interpreting
// static instances: parameter verification, storage sizing, and byte
// encode/decode. Text and bytecode syntax are generic named attrs, so
// parsing/printing do not go through the family vtable.
typedef struct loom_encoding_vtable_t {
  // Generated family schema. Required and process-lifetime stable.
  const loom_encoding_family_descriptor_t* descriptor;

  // Classifies family-specific semantics during module construction. Called
  // only after the generated parameter schema has matched. May be NULL when
  // every structurally valid instance is semantically valid.
  bool (*is_static_valid)(const loom_module_t* module,
                          const loom_encoding_t* encoding);

  // Diagnoses a static encoding rejected by is_static_valid. Called only on
  // malformed IR after the generated parameter schema has matched. A non-OK
  // status is reserved for diagnostic sink failure. Must be present whenever
  // is_static_valid is present.
  iree_status_t (*diagnose_static)(const loom_module_t* module,
                                   const loom_encoding_t* encoding,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter);

  // Verifies one encoding.define op after generic OperandDict validation.
  // Receives the merged static/dynamic parameter view so families can enforce
  // required dynamic operands, reject unknown parameters, and diagnose type
  // mismatches at the op that introduced the dynamic encoding value.
  // Authored contract violations are emitted as structured diagnostics; a
  // non-OK status is reserved for diagnostic sink failure.
  //
  // May be NULL when the family has no dynamic-parameter contract. Static
  // parsing/printing stays generic; family-specific logic belongs here.
  iree_status_t (*verify_define)(
      const loom_module_t* module, const loom_op_t* op,
      const loom_encoding_define_param_view_t* params,
      iree_diagnostic_emitter_t emitter);

  // Computes storage size in bytes for `element_count` logical elements.
  // May be NULL for families that are compile-time-only metadata.
  iree_status_t (*storage_size)(const loom_module_t* module,
                                const loom_encoding_t* encoding,
                                iree_host_size_t element_count,
                                iree_host_size_t* out_storage_size);

  // Decodes encoded bytes into dense elements.
  // May be NULL if the family has no host-side codec yet.
  iree_status_t (*decode)(const loom_module_t* module,
                          const loom_encoding_t* encoding,
                          iree_const_byte_span_t encoded_data,
                          iree_byte_span_t decoded_data,
                          iree_host_size_t element_count);

  // Encodes dense elements into the family-specific byte layout.
  // May be NULL if the family has no host-side codec yet.
  iree_status_t (*encode)(const loom_module_t* module,
                          const loom_encoding_t* encoding,
                          iree_const_byte_span_t decoded_data,
                          iree_byte_span_t encoded_data,
                          iree_host_size_t element_count);
} loom_encoding_vtable_t;

// Context-owned dense list of registered encoding family vtables.
typedef struct loom_encoding_vtable_list_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  const loom_encoding_vtable_t** entries;
} loom_encoding_vtable_list_t;

// Returns the canonical parameter slice for `encoding`.
static inline loom_named_attr_slice_t loom_encoding_attrs(
    const loom_encoding_t* encoding) {
  return loom_make_named_attr_slice(encoding->attributes,
                                    encoding->attribute_count);
}

// Returns true if two static encoding instances are structurally equal.
//
// `alias_id` is ignored so multiple file-local aliases that name the same
// family and parameterization collapse to one canonical module entry.
bool loom_encoding_equal(const loom_encoding_t* a, const loom_encoding_t* b);

// Returns a content hash compatible with `loom_encoding_equal()`.
uint32_t loom_encoding_hash(const loom_encoding_t* encoding);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_ENCODING_H_
