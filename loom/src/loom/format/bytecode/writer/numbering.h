// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Body-local SSA numbering and structural catalog discovery.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_NUMBERING_H_
#define LOOM_FORMAT_BYTECODE_WRITER_NUMBERING_H_

#include "loom/format/bytecode/writer/catalog.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum supported serialized region nesting depth.
#define LOOM_BYTECODE_WRITER_MAX_REGION_DEPTH 256

// One module value mapped into a body-local SSA namespace.
typedef struct loom_bytecode_value_numbering_entry_t {
  // Module value ID being mapped.
  loom_value_id_t value_id;
  // Body-local value number assigned to |value_id|.
  uint32_t number;
} loom_bytecode_value_numbering_entry_t;

// Dense body-local SSA namespace constructed in definition order.
typedef struct loom_bytecode_value_numbering_t {
  // Module containing numbered values.
  const loom_module_t* module;
  // Arena owning |entries|.
  iree_arena_allocator_t* arena;
  // Sorted map from module value ID to active local value number.
  loom_bytecode_value_numbering_entry_t* entries;
  // Number of initialized |entries|.
  iree_host_size_t count;
  // Allocated capacity of |entries|.
  iree_host_size_t capacity;
  // Next body-local value number to assign.
  uint32_t next_number;
} loom_bytecode_value_numbering_t;

// Declaration-local value closure used by a global symbol payload.
typedef struct loom_bytecode_global_value_list_t {
  // Temporary arena owning |values|.
  iree_arena_allocator_t* arena;
  // Module whose value table owns all IDs in |values|.
  const loom_module_t* module;
  // Ordered declaration-local values used by one global payload.
  loom_value_id_t* values;
  // Number of populated |values| entries.
  iree_host_size_t count;
  // Allocated capacity of |values|.
  iree_host_size_t capacity;
} loom_bytecode_global_value_list_t;

// Initializes an empty body-local SSA namespace.
void loom_bytecode_value_numbering_initialize(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_module_t* module, iree_arena_allocator_t* arena);

// Reserves storage for at least |minimum_capacity| local values.
iree_status_t loom_bytecode_value_numbering_ensure_capacity(
    loom_bytecode_value_numbering_t* value_numbering,
    iree_host_size_t minimum_capacity);

// Assigns the next body-local number to |value_id| when not already assigned.
iree_status_t loom_bytecode_value_numbering_assign_value(
    loom_bytecode_value_numbering_t* value_numbering, loom_value_id_t value_id);

// Resolves |value_id| in the active body-local SSA namespace.
iree_status_t loom_bytecode_resolve_value_number(
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id, uint32_t* out_number);

// Assigns body-local numbers to every definition nested under |region|.
iree_status_t loom_bytecode_value_numbering_assign_region(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region);

// Resolves whether one operation attribute participates in serialization.
iree_status_t loom_bytecode_op_attr_is_present(
    const loom_op_t* op, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t attr, bool* out_present);

// Returns true when |attr_index| carries the enclosing symbol identity.
bool loom_bytecode_attr_is_symbol_identity(const loom_op_vtable_t* vtable,
                                           uint8_t attr_index);

// Collects the declaration-local value closure of one global definition.
iree_status_t loom_bytecode_collect_global_values(
    iree_arena_allocator_t* arena, const loom_module_t* module,
    const loom_op_t* op, loom_bytecode_global_value_list_t* out_values);

// Numbers the catalogs referenced by one global definition and value closure.
iree_status_t loom_bytecode_number_global(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    const loom_bytecode_global_value_list_t* local_values);

// Validates the structural contract of a record symbol definition.
iree_status_t loom_bytecode_validate_record_symbol_op(
    const loom_module_t* module, const loom_op_t* op);

// Numbers the catalogs referenced by one record symbol definition.
iree_status_t loom_bytecode_number_record(loom_bytecode_numbering_t* numbering,
                                          const loom_op_t* op);

// Numbers the signature and nested-region catalogs of one function definition.
iree_status_t loom_bytecode_number_function(
    loom_bytecode_numbering_t* numbering, loom_func_like_t func_like);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_NUMBERING_H_
