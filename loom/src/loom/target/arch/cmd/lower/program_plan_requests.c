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
#include "loom/ops/op_defs.h"
#include "loom/rewrite/module_projection.h"

static uint32_t loom_cmd_program_plan_append_source_requirement(
    loom_cmd_program_plan_t* plan, const loom_op_t* declaration_op) {
  IREE_ASSERT_LT(plan->entry_requirement_count, UINT32_MAX);
  const uint32_t requirement_index = (uint32_t)plan->entry_requirement_count++;
  plan->entry_requirements[requirement_index] =
      (loom_cmd_entry_requirement_t){.declaration_op = declaration_op};
  return requirement_index;
}

iree_status_t loom_cmd_program_kernel_request_materialize(
    const loom_cmd_program_kernel_request_t* request,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_kernel_class_product_t* out_product) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_product);
  *out_product = (loom_kernel_class_product_t){0};

  switch (request->kind) {
    case LOOM_CMD_PROGRAM_KERNEL_REQUEST_KIND_INDEXED:
      IREE_ASSERT_ARGUMENT(request->source.indexed);
      return loom_kernel_request_materialize(
          request->source.indexed, block_pool, allocator, out_product);
    case LOOM_CMD_PROGRAM_KERNEL_REQUEST_KIND_GENERATED: {
      IREE_ASSERT_ARGUMENT(request->source.generated);
      IREE_ASSERT_ARGUMENT(request->source.generated->module);
      IREE_ASSERT(loom_symbol_ref_is_valid(request->source.generated->kernel));
      iree_arena_allocator_t scratch_arena;
      iree_arena_initialize(block_pool, &scratch_arena);
      loom_ir_module_projection_t projection = {0};
      loom_module_t* cloned_module = NULL;
      iree_status_t status = loom_ir_module_clone(
          request->source.generated->module, /*options=*/NULL, block_pool,
          &scratch_arena, allocator, &projection, &cloned_module);
      if (iree_status_is_ok(status)) {
        *out_product = (loom_kernel_class_product_t){
            .module = cloned_module,
            .kernel = loom_ir_module_projection_target_symbol(
                &projection, request->source.generated->kernel.symbol_id),
        };
      }
      iree_arena_deinitialize(&scratch_arena);
      return status;
    }
  }
  IREE_ASSERT_UNREACHABLE();
  return iree_ok_status();
}

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
  // Optional destination for summaries needed by this kernel's multi-wave
  // roots.
  loom_cmd_program_kernel_accesses_t* accesses;
} loom_cmd_program_kernel_request_bridge_t;

static iree_status_t loom_cmd_program_kernel_request_publish(
    void* user_data, const loom_kernel_request_t* source_request) {
  loom_cmd_program_kernel_request_bridge_t* bridge =
      (loom_cmd_program_kernel_request_bridge_t*)user_data;
  const uint32_t requirement_index =
      loom_cmd_program_plan_append_source_requirement(bridge->plan,
                                                      bridge->declaration_op);
  IREE_ASSERT_LT(source_request->class_ordinal, bridge->class_capacity);
  bridge->requirement_by_class[source_request->class_ordinal] =
      requirement_index;

  if (bridge->accesses) {
    const loom_storage_access_summary_t* source_summary = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_request_require_storage_access(
        source_request, &source_summary));
    const iree_host_size_t argument_bytes =
        source_summary->argument_count * sizeof(*source_summary->arguments);
    loom_storage_access_summary_t* summary = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(bridge->accesses->arena,
                                             sizeof(*summary) + argument_bytes,
                                             (void**)&summary));
    *summary = *source_summary;
    loom_storage_access_argument_t* arguments =
        (loom_storage_access_argument_t*)(summary + 1);
    memcpy(arguments, source_summary->arguments, argument_bytes);
    summary->arguments = arguments;
    bridge->accesses->by_requirement[requirement_index] = summary;
  }

  const loom_cmd_program_kernel_request_t request = {
      .entry_requirement_index = requirement_index,
      .kind = LOOM_CMD_PROGRAM_KERNEL_REQUEST_KIND_INDEXED,
      .source.indexed = source_request,
  };
  return bridge->sink.publish(bridge->sink.user_data, &request);
}

iree_status_t loom_cmd_program_plan_publish_kernel_requests(
    loom_cmd_program_plan_t* plan, const loom_module_t* plan_module,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_program_kernel_source_t* kernel_source,
    loom_cmd_program_kernel_site_root_t* roots, iree_host_size_t root_count,
    loom_cmd_program_kernel_accesses_t* accesses,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(plan_module);
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
  IREE_ASSERT_EQ(symbol_count, plan_module->symbols.count);
  iree_host_size_t* site_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, symbol_count, sizeof(*site_counts), (void**)&site_counts));
  memset(site_counts, 0, symbol_count * sizeof(*site_counts));
  uint8_t* access_demand = NULL;
  if (accesses->by_requirement) {
    const iree_host_size_t bytes = (symbol_count + 7) / 8;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(scratch_arena, bytes, (void**)&access_demand));
    memset(access_demand, 0, bytes);
  }

  iree_host_size_t source_site_count = 0;
  for (iree_host_size_t root_ordinal = 0; root_ordinal < root_count;
       ++root_ordinal) {
    loom_cmd_program_kernel_site_root_t* root = &roots[root_ordinal];
    const iree_host_size_t command_count = root->schedule->command_count;
    IREE_ASSERT(command_count == 0 || root->requirement_indices != NULL);
    for (iree_host_size_t command_ordinal = 0; command_ordinal < command_count;
         ++command_ordinal) {
      const loom_cmd_schedule_command_t* command =
          &root->schedule->commands[command_ordinal];
      if (command->kind == LOOM_CMD_SCHEDULE_COMMAND_KIND_TIMESTAMP) {
        continue;
      }
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
      if (access_demand && root->schedule->wave_count > 1) {
        access_demand[command->callee.symbol_id / 8] |=
            1u << (command->callee.symbol_id % 8);
      }
    }
  }
  if (source_site_count == 0) {
    return iree_ok_status();
  }

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
      if (command->kind == LOOM_CMD_SCHEDULE_COMMAND_KIND_TIMESTAMP) {
        continue;
      }
      const loom_symbol_id_t symbol_id = command->callee.symbol_id;
      if (kernel_source->source_definitions.values[symbol_id] ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
        continue;
      }
      const iree_host_size_t site_ordinal = site_cursors[symbol_id]++;
      sites[site_ordinal] = (loom_kernel_class_site_t){
          .facts = source_facts,
          .argument_values = command->source_arguments.values,
      };
      requirement_locations[site_ordinal] =
          &root->requirement_indices[command_ordinal];
    }
  }

  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    const iree_host_size_t site_count = site_counts[symbol_id];
    if (site_count == 0) {
      continue;
    }
    const iree_arena_checkpoint_t kernel_checkpoint =
        iree_arena_checkpoint_save(scratch_arena);
    const iree_host_size_t site_offset = site_offsets[symbol_id];
    const iree_host_size_t source_symbol_ordinal =
        kernel_source->source_definitions.values[symbol_id];
    IREE_ASSERT_NE(source_symbol_ordinal,
                   LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
    const loom_op_t* declaration_op =
        plan_module->symbols.entries[symbol_id].defining_op;

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
        .accesses = access_demand && (access_demand[symbol_id / 8] &
                                      (1u << (symbol_id % 8)))
                        ? accesses
                        : NULL,
    };
    loom_kernel_class_collection_t collection = {0};
    IREE_RETURN_IF_ERROR(loom_kernel_request_producer_publish(
        kernel_source->producer, kernel_source->environment,
        source_symbol_ordinal, plan_module, &sites[site_offset], site_count,
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

iree_status_t loom_cmd_program_plan_publish_generated_kernel_requests(
    loom_cmd_program_plan_t* plan, const loom_module_t* plan_module,
    const loom_cmd_launch_count_realization_t* realization,
    loom_cmd_program_kernel_request_sink_t sink,
    loom_cmd_program_kernel_site_root_t* roots, iree_host_size_t root_count,
    loom_cmd_program_kernel_accesses_t* accesses,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(plan_module);
  IREE_ASSERT_ARGUMENT(realization);
  IREE_ASSERT_GT(realization->kernel_count, 0u);
  IREE_ASSERT_EQ(realization->symbol_count, plan_module->symbols.count);
  IREE_ASSERT_ARGUMENT(sink.publish);
  IREE_ASSERT_ARGUMENT(roots);
  IREE_ASSERT_GT(root_count, 0u);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  uint32_t* requirement_by_symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, realization->symbol_count, sizeof(*requirement_by_symbol),
      (void**)&requirement_by_symbol));
  memset(requirement_by_symbol, 0xFF,
         realization->symbol_count * sizeof(*requirement_by_symbol));

  for (iree_host_size_t root_ordinal = 0; root_ordinal < root_count;
       ++root_ordinal) {
    loom_cmd_program_kernel_site_root_t* root = &roots[root_ordinal];
    IREE_ASSERT(root->schedule->command_count == 0 ||
                root->requirement_indices != NULL);
    for (iree_host_size_t command_ordinal = 0;
         command_ordinal < root->schedule->command_count; ++command_ordinal) {
      const loom_cmd_schedule_command_t* command =
          &root->schedule->commands[command_ordinal];
      if (command->kind == LOOM_CMD_SCHEDULE_COMMAND_KIND_TIMESTAMP) {
        continue;
      }
      const loom_cmd_launch_count_kernel_t* generated =
          loom_cmd_launch_count_realization_lookup(realization,
                                                   command->callee);
      if (generated == NULL) {
        continue;
      }
      uint32_t requirement_index =
          requirement_by_symbol[command->callee.symbol_id];
      if (requirement_index == UINT32_MAX) {
        const loom_op_t* declaration_op =
            plan_module->symbols.entries[command->callee.symbol_id].defining_op;
        IREE_ASSERT(loom_kernel_entry_decl_isa(declaration_op));
        requirement_index = loom_cmd_program_plan_append_source_requirement(
            plan, declaration_op);
        requirement_by_symbol[command->callee.symbol_id] = requirement_index;

        const loom_cmd_program_kernel_request_t request = {
            .entry_requirement_index = requirement_index,
            .kind = LOOM_CMD_PROGRAM_KERNEL_REQUEST_KIND_GENERATED,
            .source.generated = &generated->product,
        };
        IREE_RETURN_IF_ERROR(sink.publish(sink.user_data, &request));
      }
      root->requirement_indices[command_ordinal] = requirement_index;
      if (accesses->by_requirement && root->schedule->wave_count > 1 &&
          !accesses->by_requirement[requirement_index]) {
        const iree_arena_checkpoint_t checkpoint =
            iree_arena_checkpoint_save(scratch_arena);
        loom_module_t* module = generated->product.module;
        const loom_func_like_t kernel = loom_func_like_cast(
            module, module->symbols.entries[generated->product.kernel.symbol_id]
                        .defining_op);
        loom_storage_access_scope_t scope;
        loom_storage_access_scope_initialize(module, scratch_arena, &scope);
        loom_local_value_domain_t domain = {0};
        iree_status_t status = loom_local_value_domain_acquire_for_region_tree(
            module, loom_func_like_body(kernel), scratch_arena, &domain);
        if (iree_status_is_ok(status)) {
          status = loom_storage_access_require_summary(
              &scope, kernel, &domain, accesses->arena,
              &accesses->by_requirement[requirement_index]);
        }
        loom_local_value_domain_release(&domain);
        iree_arena_checkpoint_restore(&checkpoint);
        IREE_RETURN_IF_ERROR(status);
      }
    }
  }
  return iree_ok_status();
}
