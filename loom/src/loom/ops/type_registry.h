// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_TYPE_REGISTRY_H_
#define LOOM_OPS_TYPE_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;

// Format element kinds for type interiors (inside <...>).
// These are separate from op format elements because type
// interiors have different semantics (shape dims, element
// types, encodings) than op bodies (operand refs, attr values).
typedef enum loom_type_format_kind_e {
  LOOM_TYPE_FMT_SHAPE = 0,      // Dimension list: 4x[%M]x...
  LOOM_TYPE_FMT_SCALAR = 1,      // Element type keyword: f32, i8.
  LOOM_TYPE_FMT_ENCODING = 2,    // Encoding ref: #q8_0 or %enc.
  LOOM_TYPE_FMT_TYPE = 3,         // Recursive type: vm.ref<T>.
  LOOM_TYPE_FMT_ATTR = 4,         // Bare identifier: group<workgroup>.
  LOOM_TYPE_FMT_KEYWORD = 5,      // Literal punctuation/word.
  LOOM_TYPE_FMT_OPTIONAL = 6,     // Conditional elements.
  LOOM_TYPE_FMT_GLUE = 7,         // Suppress space.
} loom_type_format_kind_t;

// A 4-byte format element for type interiors. Same layout
// as loom_format_element_t for consistent handling.
typedef struct loom_type_format_element_t {
  // Format opcode, encoded as loom_type_format_kind_t.
  uint8_t kind;

  // Parameter index consumed by this element.
  uint8_t field_index;

  // Kind-specific payload such as a keyword ID or skip count.
  uint16_t data;
} loom_type_format_element_t;

// Descriptor for a registered type. Contains the name,
// the IR type kind to construct, parameter count, and
// format elements describing the type interior syntax.
typedef struct loom_type_descriptor_t {
  // B-string name: [length]"tile", [length]"hal.buffer".
  const uint8_t* name;

  // What IR type kind to construct when parsing.
  loom_type_kind_t ir_kind;

  // Number of declared parameters.
  uint8_t param_count;

  // Optional type-owned value fact domain. NULL means the type only has generic
  // scalar facts or uses the domain-free extension behavior.
  const loom_value_fact_domain_t* fact_domain;

  // Semantic role and target-contract families for this type.
  loom_type_semantics_t semantics;

  // Format element array for the type interior (inside <...>).
  // NULL for opaque types (no angle brackets).
  const loom_type_format_element_t* format_elements;

  // Number of entries in |format_elements|.
  uint8_t format_element_count;
} loom_type_descriptor_t;

// Entry in the sorted type registry.
typedef struct loom_type_registry_entry_t {
  iree_string_view_t name;
  const loom_type_descriptor_t* descriptor;
} loom_type_registry_entry_t;

// Returns the number of entries in the global type registry.
iree_host_size_t loom_type_registry_count(void);

// Returns the sorted registry array (for iteration/testing).
const loom_type_registry_entry_t* loom_type_registry_entries(void);

// Looks up a type descriptor by name (e.g., "tile", "hal.buffer").
// Returns the descriptor on success, NULL if not found.
const loom_type_descriptor_t* loom_type_registry_lookup(
    iree_string_view_t name);

// Resolves the type-owned value fact domain for |type|, or NULL if the
// registered type has no extension fact domain.
const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type);

// Installs the generated type-registry fact-domain resolver on |context|.
void loom_type_registry_configure_fact_context(
    loom_fact_context_t* context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TYPE_REGISTRY_H_
