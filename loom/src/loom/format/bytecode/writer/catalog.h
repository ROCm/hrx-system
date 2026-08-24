// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable bytecode catalogs and IR-to-wire numbering.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_CATALOG_H_
#define LOOM_FORMAT_BYTECODE_WRITER_CATALOG_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/low_repr.h"
#include "loom/ir/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// String originating outside of the module string table.
typedef struct loom_bytecode_external_string_t {
  // External string contents.
  iree_string_view_t view;
  // Dense bytecode string-table ID assigned to |view|.
  uint32_t writer_id;
} loom_bytecode_external_string_t;

// Operation kind registered in the bytecode operation table.
typedef struct loom_bytecode_op_entry_t {
  // Internal operation kind represented by this entry.
  loom_op_kind_t kind;
  // Dense bytecode operation-table ID.
  uint32_t writer_op_id;
  // Bytecode string-table ID naming |kind|.
  uint32_t string_writer_id;
} loom_bytecode_op_entry_t;

// Structural type lookup entry used while assigning bytecode type IDs.
typedef struct loom_bytecode_type_index_entry_t {
  // Structural hash of the module type.
  uint32_t hash;
  // Module type-table index or UINT32_MAX for an empty slot.
  uint32_t module_index;
} loom_bytecode_type_index_entry_t;

// First-use-ordered bytecode catalogs derived while streaming one module.
typedef struct loom_bytecode_numbering_t {
  // Module being serialized.
  const loom_module_t* module;
  // Temporary arena owning all catalog storage.
  iree_arena_allocator_t* arena;
  // File-level location mode controlling operation location references.
  loom_bytecode_location_mode_t location_mode;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
  // Representation contract active while numbering one Low function.
  const loom_low_repr_descriptor_set_t* active_low_descriptor_set;

  // Bidirectional stable module ID and presentation-order mapping.
  struct {
    // Module symbol IDs indexed by presentation-ordered wire ordinal.
    loom_symbol_id_t* module_ids;
    // Presentation-ordered wire ordinals indexed by module symbol ID.
    loom_symbol_id_t* wire_ordinals;
  } symbol_order;

  // Bytecode string-table entries in first-use order.
  iree_string_view_t* string_entries;
  // Number of assigned bytecode string IDs.
  iree_host_size_t string_count;
  // Allocated capacity of |string_entries|.
  iree_host_size_t string_capacity;
  // Bytecode string IDs indexed by module string ID.
  uint32_t* module_string_map;

  // Strings originating outside of the module string table.
  loom_bytecode_external_string_t* external_strings;
  // Number of populated |external_strings| entries.
  iree_host_size_t external_string_count;
  // Allocated capacity of |external_strings|.
  iree_host_size_t external_string_capacity;

  // Bytecode type IDs indexed by module type-table index.
  uint32_t* type_map;
  // Structural reverse lookup from wire type to module type-table index.
  loom_bytecode_type_index_entry_t* type_index_entries;
  // Power-of-two capacity of |type_index_entries|.
  iree_host_size_t type_index_capacity;
  // Module type-table indices indexed by bytecode type ID.
  iree_host_size_t* type_order;
  // Number of assigned bytecode type IDs.
  iree_host_size_t type_count;
  // Allocated capacity of |type_order|.
  iree_host_size_t type_order_capacity;

  // Operation kinds in first-use bytecode table order.
  loom_bytecode_op_entry_t* op_entries;
  // Number of assigned bytecode operation IDs.
  iree_host_size_t op_count;
  // Allocated capacity of |op_entries|.
  iree_host_size_t op_capacity;
} loom_bytecode_numbering_t;

// Initializes empty catalogs and the stable symbol-order projection.
iree_status_t loom_bytecode_numbering_initialize(
    loom_bytecode_numbering_t* numbering, const loom_module_t* module,
    iree_arena_allocator_t* arena);

// Returns the module symbol ID assigned to |wire_ordinal|.
static inline loom_symbol_id_t loom_bytecode_module_symbol_id(
    const loom_bytecode_numbering_t* numbering, loom_symbol_id_t wire_ordinal) {
  IREE_ASSERT(wire_ordinal < numbering->module->symbols.count);
  return numbering->symbol_order.module_ids[wire_ordinal];
}

// Returns the wire ordinal assigned to |module_symbol_id|.
static inline loom_symbol_id_t loom_bytecode_wire_symbol_ordinal(
    const loom_bytecode_numbering_t* numbering,
    loom_symbol_id_t module_symbol_id) {
  IREE_ASSERT(module_symbol_id < numbering->module->symbols.count);
  return numbering->symbol_order.wire_ordinals[module_symbol_id];
}

// Interns a module-owned string into the bytecode string catalog.
iree_status_t loom_bytecode_numbering_intern_module_string(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id,
    uint32_t* out_writer_id);

// Interns an arbitrary borrowed string into the bytecode string catalog.
iree_status_t loom_bytecode_numbering_intern_string_view(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id);

// Interns a structural type and all of its dependencies.
iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id);

// Interns the registered kind of |op| into the operation catalog.
iree_status_t loom_bytecode_numbering_intern_op(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    uint32_t* out_writer_op_id);

// Validates and returns a closed or open enum ordinal.
iree_status_t loom_bytecode_get_enum_ordinal(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t* out_ordinal);

// Validates and returns a descriptor-backed enum array.
iree_status_t loom_bytecode_get_enum_array(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_enum_array_t* out_array);

// Validates and returns a descriptor-backed signed enum set.
iree_status_t loom_bytecode_get_signed_enum_set(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_signed_enum_set_t* out_set);

// Validates and returns a module-local symbol collection.
iree_status_t loom_bytecode_get_symbol_collection(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_ref_array_t* out_array);

// Resolves and validates one registered parameterized attribute family.
iree_status_t loom_bytecode_get_parameterized_attr(
    const loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind,
    const loom_parameterized_attr_descriptor_t** out_family_descriptor);

// Resolves whether one parameterized attribute slot is present.
iree_status_t loom_bytecode_parameter_is_present(
    const loom_parameterized_attr_descriptor_t* family_descriptor,
    loom_attribute_t value, uint8_t parameter_index, bool* out_present);

// Resolves the representation contract selected by |func_like|.
iree_status_t loom_bytecode_resolve_function_low_descriptor_set(
    const loom_bytecode_numbering_t* numbering, loom_func_like_t func_like,
    const loom_low_repr_descriptor_set_t** out_descriptor_set);

// Numbers all transitive catalog dependencies of one attribute value.
iree_status_t loom_bytecode_number_attr_value(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor);

// Numbers all transitive catalog dependencies of one static encoding.
iree_status_t loom_bytecode_number_encoding(
    loom_bytecode_numbering_t* numbering, uint16_t encoding_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_CATALOG_H_
