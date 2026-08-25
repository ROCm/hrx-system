// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V emission for one target-low function.
//
// Function emission appends declarations, entry-point metadata, and function
// instructions to a module builder owned by the caller. The context carries
// only the module-wide caches and ABI state that functions intentionally
// share; function-local value and storage state remains private to the
// implementation.

#ifndef LOOM_TARGET_EMIT_SPIRV_FUNCTION_EMITTER_H_
#define LOOM_TARGET_EMIT_SPIRV_FUNCTION_EMITTER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/target/emit/spirv/module_abi.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_storage.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_SPIRV_BUILTIN_VARIABLE_COUNT = 3,
};

// Module-owned resources shared by each function emitted into one module.
typedef struct loom_spirv_function_emission_context_t {
  // Module containing the target-low function.
  loom_module_t* module;
  // Scratch arena reset and managed by the module-emission call.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder receiving the emitted function.
  loom_spirv_module_builder_t* builder;
  // Module-wide SPIR-V type and constant cache.
  loom_spirv_type_context_t* type_context;
  // Module-wide raw-BDA ABI layout shared by HAL kernel entries.
  loom_spirv_module_raw_bda_layout_t* raw_bda_layout;
  // Module-wide Input variable IDs indexed by supported builtin kind.
  uint32_t* builtin_variable_ids;
} loom_spirv_function_emission_context_t;

// Mutable state for one function emission.
//
// Structural-control emission is factored into its own translation unit and
// operates directly on this state. The state remains function-local and is
// initialized and released only by loom_spirv_emit_low_function.
typedef struct loom_spirv_emit_state_t {
  // Module-owned resources shared by every emitted function.
  loom_spirv_function_emission_context_t* context;
  // Module containing the emitted low function.
  loom_module_t* module;
  // Target-low function definition being emitted.
  loom_op_t* function_op;
  // Target-low function body being emitted.
  const loom_region_t* body;
  // Resolved target record and descriptor set for function_op.
  const loom_low_resolved_target_t* target;
  // Function-local scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder.
  loom_spirv_module_builder_t* builder;
  // Function-local value domain for dense value tables.
  loom_local_value_domain_t value_domain;
  // Function-local Loom value to SPIR-V value-ref table.
  loom_spirv_module_value_table_t value_table;
  // Module string IDs for descriptor-set immediate rows.
  loom_string_id_t* immediate_name_ids;
  // Number of entries in immediate_name_ids.
  iree_host_size_t immediate_name_id_count;
  // SPIR-V type and constant emission cache shared by the module.
  loom_spirv_type_context_t* type_context;
  // SPIR-V ID assigned to the function.
  uint32_t function_id;
  // SPIR-V label ID of the currently open function-section block.
  uint32_t current_label_id;
  // Selected ABI plan for entry materialization.
  loom_spirv_module_abi_plan_t abi_plan;
  // Function-local Workgroup storage materialization state.
  loom_spirv_module_workgroup_storage_state_t workgroup_storage;
  // Input variables referenced by this function's entry-point interface.
  uint32_t builtin_interface_variable_ids[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];
} loom_spirv_emit_state_t;

// Returns the writer for one logical module section.
static inline loom_spirv_binary_writer_t* loom_spirv_emit_section(
    loom_spirv_emit_state_t* state, loom_spirv_module_section_t section) {
  return loom_spirv_module_builder_section(state->builder, section);
}

// Allocates one fresh result ID from the containing module.
static inline uint32_t loom_spirv_emit_allocate_id(
    loom_spirv_emit_state_t* state) {
  return loom_spirv_module_builder_allocate_id(state->builder);
}

// Opens a function block with label_id and records it as the current block.
iree_status_t loom_spirv_emit_label_id(loom_spirv_emit_state_t* state,
                                       uint32_t label_id);

// Defines a function-local value and optionally emits its debug name.
iree_status_t loom_spirv_emit_define_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref, bool emit_name);

// Reserves a typed SPIR-V result ID for a function-local value.
iree_status_t loom_spirv_emit_reserve_value_ref(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id, uint32_t type_id,
    loom_spirv_value_type_t value_type, uint32_t* out_result_id);

// Looks up the SPIR-V ref assigned to a function-local value.
iree_status_t loom_spirv_emit_lookup_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t* out_value_ref);

// Returns true when a function-local value already has a SPIR-V ref.
static inline bool loom_spirv_emit_value_ref_exists(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id) {
  return loom_spirv_module_value_table_exists(&state->value_table, value_id);
}

// Emits one verified low op, recursively entering structured regions.
iree_status_t loom_spirv_emit_low_op(loom_spirv_emit_state_t* state,
                                     const loom_op_t* op);

// Emits one target-low function into |context->builder|.
//
// |function_op| and |target| are borrowed for the call. Module-wide IDs and
// ABI state are retained in |context| for subsequent functions; all
// function-local state is released before return.
iree_status_t loom_spirv_emit_low_function(
    loom_spirv_function_emission_context_t* context, loom_op_t* function_op,
    const loom_low_resolved_target_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_FUNCTION_EMITTER_H_
