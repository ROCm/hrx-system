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
#include "loom/ir/attribute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_encoding_define_param_view_t
    loom_encoding_define_param_view_t;

// A single static encoding instance, such as `#q8_0<block=32>`.
//
// The encoding family name and optional file-local alias are interned string
// IDs in the owning module. Parameters are named attributes so families can
// carry structured values such as integers, strings, arrays, and dicts.
struct loom_encoding_t {
  loom_string_id_t name_id;
  // File-local alias spelling without '#', or LOOM_STRING_ID_INVALID when no
  // alias should be preferred for printing.
  loom_string_id_t alias_id;
  uint8_t attribute_count;
  uint8_t reserved[3];
  // Arena-owned canonical parameter array. NULL when attribute_count is 0.
  const loom_named_attr_t* attributes;
};

typedef struct loom_encoding_t loom_encoding_t;

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
  // Encoding family name for lookup and printing, without a leading '#'.
  iree_string_view_t name;

  // Semantic role for this family. Leave UNKNOWN for unconstrained families.
  loom_encoding_role_t role;

  // Verifies that `encoding` carries a valid parameter set for this family.
  // May be NULL when the family accepts any canonical named attrs.
  iree_status_t (*verify)(const loom_module_t* module,
                          const loom_encoding_t* encoding);

  // Verifies one encoding.define op after generic OperandDict validation.
  // Receives the merged static/dynamic parameter view so families can enforce
  // required dynamic operands, reject unknown parameters, and diagnose type
  // mismatches at the op that introduced the dynamic encoding value.
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
