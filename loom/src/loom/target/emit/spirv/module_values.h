// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local SPIR-V value refs.
//
// SPIR-V module emission needs a dense, function-local map from Loom SSA values
// to emitted SPIR-V result IDs and target-local value types. The map is indexed
// by an already-acquired local value domain so packet, ABI, and CFG emission
// can share O(1) lookups without owning the module's global value scratch.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_VALUES_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_VALUES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/emit/spirv/module_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_module_value_ref_t {
  // SPIR-V result ID carrying a low SSA value.
  uint32_t id;
  // SPIR-V type ID assigned to id.
  uint32_t type_id;
  // Target-local value type used by packet emitters.
  loom_spirv_value_type_t value_type;
} loom_spirv_module_value_ref_t;

typedef struct loom_spirv_module_value_table_t {
  // Function-local value domain that owns the dense ordinal map.
  const loom_local_value_domain_t* value_domain;
  // SPIR-V refs indexed by function-local value ordinal.
  loom_spirv_module_value_ref_t* refs;
  // Number of entries in refs.
  iree_host_size_t ref_count;
} loom_spirv_module_value_table_t;

// Initializes a zeroed value-ref table covering every value in |value_domain|.
iree_status_t loom_spirv_module_value_table_initialize(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_value_table_t* out_table,
    iree_arena_allocator_t* scratch_arena);

// Defines |value_id| as |value_ref|.
//
// |value_id| must be covered by the local value domain. A previous identical
// reservation is accepted; a previous incompatible definition is a compiler
// invariant violation.
void loom_spirv_module_value_table_define(
    loom_spirv_module_value_table_t* table, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref);

// Reserves or returns a SPIR-V result ID for |value_id| with the exact type
// contract supplied by the caller.
uint32_t loom_spirv_module_value_table_reserve(
    loom_spirv_module_value_table_t* table,
    loom_spirv_module_builder_t* builder, loom_value_id_t value_id,
    uint32_t type_id, loom_spirv_value_type_t value_type);

// Looks up the emitted SPIR-V value ref for |value_id|.
//
// |value_id| must have already been defined or reserved.
loom_spirv_module_value_ref_t loom_spirv_module_value_table_lookup(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id);

// Returns true when |value_id| already has a reserved or defined SPIR-V ref.
bool loom_spirv_module_value_table_exists(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_VALUES_H_
