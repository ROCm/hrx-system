// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V function ABI planning and materialization.
//
// This boundary owns the representation that turns verified low function
// boundary values into SPIR-V entry resources. The shader-entry ABI maps entry
// arguments and results to one descriptor slot each. The HAL raw-BDA ABI maps
// direct scalar arguments to inline push-constant words and materializes
// low.resource bindings from the hidden BDA root. The module emitter remains
// responsible for function traversal and packet emission.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_ABI_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_ABI_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"
#include "loom/target/emit/spirv/program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_module_abi_plan_kind_e {
  // Descriptor-slot shader-entry ABI used by low.func.def fixtures.
  LOOM_SPIRV_MODULE_ABI_PLAN_SHADER_ENTRY = 0,
  // Vulkan raw-BDA HAL kernel ABI used by low.kernel.def entries.
  LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA = 1,
} loom_spirv_module_abi_plan_kind_t;

typedef struct loom_spirv_module_abi_slot_t {
  // Low function value materialized by this ABI slot.
  loom_value_id_t value_id;
  // SPIR-V value type materialized for value_id.
  loom_spirv_value_type_t value_type;
  // Descriptor binding assigned to this shader descriptor slot.
  uint32_t binding;
  // Module-scope StorageBuffer variable ID for this descriptor slot.
  uint32_t variable_id;
  // Raw-BDA inline constant word offset for direct scalar arguments.
  uint16_t constant_word_offset;
  // Number of 32-bit inline constant words consumed by this argument.
  uint8_t constant_word_count;
} loom_spirv_module_abi_slot_t;

typedef struct loom_spirv_module_bda_root_t {
  // Module-scope PushConstant root variable ID.
  uint32_t variable_id;
  // Function-local PhysicalStorageBuffer pointer to the binding table.
  uint32_t binding_table_pointer_id;
  // Function-local uint32_t binding-base value loaded from the root.
  uint32_t binding_base_id;
} loom_spirv_module_bda_root_t;

typedef struct loom_spirv_module_shared_bda_root_t {
  // Module-scope PushConstant root variable ID shared by raw-BDA functions.
  uint32_t root_variable_id;
  // Maximum inline constant storage, in 32-bit words, across all entries.
  // Each entry retains its exact logical counts in its own ABI plan.
  uint16_t constant_word_count;
} loom_spirv_module_shared_bda_root_t;

typedef struct loom_spirv_module_abi_plan_t {
  // ABI materialization algorithm selected for this function.
  loom_spirv_module_abi_plan_kind_t kind;
  // Materialization slots for low function entry block arguments.
  loom_spirv_module_abi_slot_t* args;
  // Number of entries in args.
  iree_host_size_t arg_count;
  // Materialization slots for low function results.
  loom_spirv_module_abi_slot_t* results;
  // Number of entries in results.
  iree_host_size_t result_count;
  // Number of HAL binding-table entries required by raw-BDA resources.
  uint16_t bda_binding_count;
  // Number of 32-bit HAL inline constants consumed by direct scalar arguments.
  uint16_t bda_constant_word_count;
  // Hidden raw-BDA PushConstant root state.
  loom_spirv_module_bda_root_t bda_root;
} loom_spirv_module_abi_plan_t;

typedef struct loom_spirv_module_abi_context_t {
  // Module containing the emitted low function.
  const loom_module_t* module;
  // Target-low function definition being emitted.
  const loom_op_t* function_op;
  // Prepared target and representation binding for |function_op|.
  const loom_spirv_function_plan_t* function_plan;
  // Source-scoped diagnostic sink for this function's encoding failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Borrowed module acceptance cell, cleared by a rejected encoding.
  bool* accepted;
  // Function scratch arena used for transient ABI plans.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder.
  loom_spirv_module_builder_t* builder;
  // SPIR-V type and constant emission cache.
  loom_spirv_type_context_t* type_context;
  // Physical push-constant storage shared by emitted HAL kernel entries.
  loom_spirv_module_shared_bda_root_t* shared_bda_root;
  // Function-local Loom value to SPIR-V value-ref table.
  loom_spirv_module_value_table_t* value_table;
} loom_spirv_module_abi_context_t;

// Builds descriptor declarations or reserves shared raw-BDA storage for |plan|.
iree_status_t loom_spirv_module_abi_build_plan(
    loom_spirv_module_abi_context_t* context, const loom_block_t* entry_block,
    loom_spirv_module_abi_plan_t* plan);

// Materializes entry block arguments according to |plan|.
iree_status_t loom_spirv_module_abi_materialize_entry_args(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan);

// Materializes a low.resource operation from the raw-BDA binding table.
iree_status_t loom_spirv_module_abi_materialize_resource(
    loom_spirv_module_abi_context_t* context,
    loom_spirv_module_abi_plan_t* plan, const loom_op_t* op);

// Stores return operands into ABI result slots.
iree_status_t loom_spirv_module_abi_store_return_values(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan, const loom_op_t* op);

// Emits dispatch metadata keyed by the same name as OpEntryPoint.
iree_status_t loom_spirv_module_abi_emit_entry_metadata(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_abi_plan_t* plan, iree_string_view_t entry_name);

// Declares the shared physical root after every entry has reserved its storage.
// The declaration covers the maximum inline constant capacity; it does not
// change the logical argument counts recorded in individual entry metadata.
iree_status_t loom_spirv_module_abi_emit_shared_bda_root(
    loom_spirv_module_abi_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_ABI_H_
