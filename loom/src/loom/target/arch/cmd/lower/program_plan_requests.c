// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan_requests.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/ops/kernel/ops.h"

typedef struct loom_cmd_program_kernel_request_bridge_t {
  // Plan receiving one requirement per published semantic class.
  loom_cmd_program_plan_t* plan;

  // Configured entry declaration shared by every site in the class group.
  const loom_op_t* declaration_op;

  // Outer owner accepting the command-specific request.
  loom_cmd_program_kernel_request_sink_t sink;

  // Scratch class-to-requirement projection sized to the group site count.
  uint32_t* requirement_by_class;

  // Number of writable entries in |requirement_by_class|.
  iree_host_size_t class_capacity;

  // Block pool used for independently owned request modules.
  iree_arena_block_pool_t* block_pool;

  // Host allocator used for independently owned request modules.
  iree_allocator_t allocator;
} loom_cmd_program_kernel_request_bridge_t;

static iree_status_t loom_cmd_program_kernel_request_publish(
    void* user_data, const loom_kernel_request_t* source_request) {
  loom_cmd_program_kernel_request_bridge_t* bridge =
      (loom_cmd_program_kernel_request_bridge_t*)user_data;
  IREE_ASSERT_LT(bridge->plan->entry_requirement_count, UINT32_MAX);
  const uint32_t requirement_index =
      (uint32_t)bridge->plan->entry_requirement_count++;
  bridge->plan->entry_requirements[requirement_index] =
      (loom_cmd_entry_requirement_t){
          .declaration_op = bridge->declaration_op,
      };
  IREE_ASSERT_LT(source_request->class_ordinal, bridge->class_capacity);
  bridge->requirement_by_class[source_request->class_ordinal] =
      requirement_index;

  loom_kernel_class_product_t product = {0};
  IREE_RETURN_IF_ERROR(loom_kernel_request_materialize(
      source_request, bridge->block_pool, bridge->allocator, &product));
  return bridge->sink.publish(
      bridge->sink.user_data,
      (loom_cmd_program_kernel_request_t){
          .entry_requirement_index = requirement_index,
          .source_symbol_ordinal = source_request->source_symbol_ordinal,
          .class_ordinal = source_request->class_ordinal,
          .member_count = source_request->member_count,
          .product = product,
      });
}

iree_status_t loom_cmd_program_plan_publish_kernel_requests(
    loom_cmd_program_plan_t* plan, const loom_module_t* preparation_module,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_program_kernel_source_t* kernel_source,
    loom_cmd_program_kernel_site_root_t* roots, iree_host_size_t root_count,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(preparation_module);
  IREE_ASSERT_ARGUMENT(source_facts);
  IREE_ASSERT_ARGUMENT(kernel_source);
  IREE_ASSERT_ARGUMENT(kernel_source->producer);
  IREE_ASSERT_ARGUMENT(kernel_source->environment);
  IREE_ASSERT_ARGUMENT(kernel_source->source_definitions.values);
  IREE_ASSERT_ARGUMENT(kernel_source->sink.publish);
  IREE_ASSERT_ARGUMENT(roots);
  IREE_ASSERT_GT(root_count, 0u);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  const iree_host_size_t symbol_count = kernel_source->source_definitions.count;
  IREE_ASSERT_EQ(symbol_count, preparation_module->symbols.count);
  iree_host_size_t* site_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, symbol_count, sizeof(*site_counts), (void**)&site_counts));
  memset(site_counts, 0, symbol_count * sizeof(*site_counts));

  iree_host_size_t source_site_count = 0;
  for (iree_host_size_t root_ordinal = 0; root_ordinal < root_count;
       ++root_ordinal) {
    loom_cmd_program_kernel_site_root_t* root = &roots[root_ordinal];
    const iree_host_size_t command_count = root->schedule->command_count;
    if (command_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          scratch_arena, command_count, sizeof(*root->requirement_indices),
          (void**)&root->requirement_indices));
      memset(root->requirement_indices, 0xFF,
             command_count * sizeof(*root->requirement_indices));
    }
    for (iree_host_size_t command_ordinal = 0; command_ordinal < command_count;
         ++command_ordinal) {
      const loom_cmd_schedule_command_t* command =
          &root->schedule->commands[command_ordinal];
      IREE_ASSERT(loom_symbol_ref_is_valid(command->callee));
      IREE_ASSERT_EQ(command->callee.module_id, 0u);
      IREE_ASSERT_LT(command->callee.symbol_id, symbol_count);
      const iree_host_size_t source_symbol_ordinal =
          kernel_source->source_definitions.values[command->callee.symbol_id];
      if (source_symbol_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
        continue;
      }
      ++site_counts[command->callee.symbol_id];
      ++source_site_count;
    }
  }
  if (source_site_count == 0) return iree_ok_status();

  iree_host_size_t* site_offsets = NULL;
  iree_host_size_t* site_cursors = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, symbol_count + 1,
                                sizeof(*site_offsets), (void**)&site_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, symbol_count,
                                                 sizeof(*site_cursors),
                                                 (void**)&site_cursors));
  site_offsets[0] = 0;
  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    site_offsets[symbol_id + 1] =
        site_offsets[symbol_id] + site_counts[symbol_id];
    site_cursors[symbol_id] = site_offsets[symbol_id];
  }
  IREE_ASSERT_EQ(site_offsets[symbol_count], source_site_count);

  loom_kernel_class_site_t* sites = NULL;
  uint32_t** requirement_locations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, source_site_count, sizeof(*sites), (void**)&sites));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, source_site_count, sizeof(*requirement_locations),
      (void**)&requirement_locations));
  for (iree_host_size_t root_ordinal = 0; root_ordinal < root_count;
       ++root_ordinal) {
    loom_cmd_program_kernel_site_root_t* root = &roots[root_ordinal];
    for (iree_host_size_t command_ordinal = 0;
         command_ordinal < root->schedule->command_count; ++command_ordinal) {
      const loom_cmd_schedule_command_t* command =
          &root->schedule->commands[command_ordinal];
      const loom_symbol_id_t symbol_id = command->callee.symbol_id;
      if (kernel_source->source_definitions.values[symbol_id] ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
        continue;
      }
      const iree_host_size_t site_ordinal = site_cursors[symbol_id]++;
      sites[site_ordinal] = (loom_kernel_class_site_t){
          .facts = source_facts,
          .argument_values = command->arguments.values,
      };
      requirement_locations[site_ordinal] =
          &root->requirement_indices[command_ordinal];
    }
  }

  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    const iree_host_size_t site_count = site_counts[symbol_id];
    if (site_count == 0) continue;
    const iree_arena_checkpoint_t kernel_checkpoint =
        iree_arena_checkpoint_save(scratch_arena);
    const iree_host_size_t site_offset = site_offsets[symbol_id];
    const iree_host_size_t source_symbol_ordinal =
        kernel_source->source_definitions.values[symbol_id];
    IREE_ASSERT_NE(source_symbol_ordinal,
                   LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
    const loom_op_t* declaration_op =
        preparation_module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(loom_kernel_entry_decl_isa(declaration_op));

    uint32_t* requirement_by_class = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, site_count, sizeof(*requirement_by_class),
        (void**)&requirement_by_class));
    loom_cmd_program_kernel_request_bridge_t bridge = {
        .plan = plan,
        .declaration_op = declaration_op,
        .sink = kernel_source->sink,
        .requirement_by_class = requirement_by_class,
        .class_capacity = site_count,
        .block_pool = kernel_source->environment->block_pool,
        .allocator = kernel_source->environment->allocator,
    };
    loom_kernel_class_collection_t collection = {0};
    IREE_RETURN_IF_ERROR(loom_kernel_request_producer_publish(
        kernel_source->producer, kernel_source->environment,
        source_symbol_ordinal, &sites[site_offset], site_count,
        &kernel_source->collection_options,
        (loom_kernel_request_sink_t){
            .publish = loom_cmd_program_kernel_request_publish,
            .user_data = &bridge,
        },
        scratch_arena, &collection));
    for (iree_host_size_t site_index = 0; site_index < site_count;
         ++site_index) {
      const loom_decision_class_ordinal_t class_ordinal =
          collection.site_classes[site_index];
      IREE_ASSERT_LT(class_ordinal, collection.class_count);
      *requirement_locations[site_offset + site_index] =
          requirement_by_class[class_ordinal];
    }
    iree_arena_checkpoint_restore(&kernel_checkpoint);
  }
  return iree_ok_status();
}
