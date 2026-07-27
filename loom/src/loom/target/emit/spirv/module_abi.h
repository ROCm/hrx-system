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
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"

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

typedef struct loom_spirv_module_raw_bda_layout_t {
  // Module-scope PushConstant root variable ID shared by raw-BDA functions.
  uint32_t root_variable_id;
  // Number of HAL binding-table entries required by every raw-BDA function.
  uint16_t binding_count;
  // Number of 32-bit HAL inline constants consumed by every raw-BDA function.
  uint16_t constant_word_count;
} loom_spirv_module_raw_bda_layout_t;

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
  // Decoded ABI value types for entry block arguments.
  loom_spirv_value_type_t* arg_value_types;
  // Number of decoded argument value types.
  iree_host_size_t arg_value_type_count;
  // Decoded ABI value types for function results.
  loom_spirv_value_type_t* result_value_types;
  // Number of decoded result value types.
  iree_host_size_t result_value_type_count;
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
  // Resolved target record and descriptor set for function_op.
  const loom_low_resolved_target_t* target;
  // Function scratch arena used for transient ABI plans.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder.
  loom_spirv_module_builder_t* builder;
  // SPIR-V type and constant emission cache.
  loom_spirv_type_context_t* type_context;
  // Module-level raw-BDA layout shared by emitted HAL kernel entries.
  loom_spirv_module_raw_bda_layout_t* raw_bda_layout;
  // Function-local Loom value to SPIR-V value-ref table.
  loom_spirv_module_value_table_t* value_table;
} loom_spirv_module_abi_context_t;

// Decodes boundary value types from low function ABI attributes into |plan|.
iree_status_t loom_spirv_module_abi_prepare_value_types(
    const loom_spirv_module_abi_context_t* context,
    const loom_block_t* entry_block, loom_spirv_module_abi_plan_t* plan);

// Builds descriptor declarations or raw-BDA root declarations for |plan|.
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

// Emits module metadata consumed by the raw-BDA HAL artifact loader.
iree_status_t loom_spirv_module_abi_emit_metadata(
    loom_spirv_module_abi_context_t* context,
    const loom_spirv_module_raw_bda_layout_t* raw_bda_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_ABI_H_
