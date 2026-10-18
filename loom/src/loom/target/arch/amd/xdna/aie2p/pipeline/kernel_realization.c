// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/kernel_realization.h"

#include <string.h>

#include "loom/analysis/storage_access.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/view_regions.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/target/arch/amd/xdna/device/profile.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/target/function_version.h"

typedef struct loom_aie2p_kernel_binding_plan_t {
  // Source kernel argument replaced by invocation-local storage.
  loom_value_id_t source_argument;

  // Corresponding external pipeline binding.
  loom_value_id_t external_binding;

  // External binding after applying the array ABI alignment contract.
  loom_value_id_t external_buffer;

  // Local ingress allocation and channel.
  loom_value_id_t input_buffer;
  loom_value_id_t input_channel;

  // Local egress allocation and channel for writable bindings.
  loom_value_id_t output_buffer;
  loom_value_id_t output_channel;

  // Channel obligations materialized inside the compute strand.
  loom_value_id_t input_read;
  loom_value_id_t output_write;

  // External view used by both halves of the DMA strand.
  loom_value_id_t external_view;

  // Complete staged footprint in bytes.
  int64_t byte_length;

  // Required alignment of each invocation-local staging allocation.
  uint64_t byte_alignment;

  // Derived memory effects through this binding.
  loom_view_access_flags_t access;

  // Conservative root identity and alias partition across accessed views.
  loom_view_region_t root_region;

  // True when the source kernel observes this binding.
  bool used;

  // Typed views and channel obligations for one byte record.
  loom_type_t record_view_type;
  loom_type_t read_type;
  loom_type_t write_type;
} loom_aie2p_kernel_binding_plan_t;

typedef struct loom_aie2p_kernel_access_binding_t {
  // Binding referenced by this source memory operation.
  uint16_t binding_index;

  // Aggregate read/write effects through the binding at this operation.
  loom_view_access_flags_t access;

  // Mutable channel obligation held while the source operation executes.
  loom_value_id_t read;

  // Borrowed complete record used for coherent egress.
  loom_value_id_t view;
} loom_aie2p_kernel_access_binding_t;

typedef struct loom_aie2p_kernel_access_plan_t {
  // Source memory operation surrounded by coherent transfers.
  loom_op_t* operation;

  // First row in the realization plan's access_bindings table.
  iree_host_size_t binding_start;

  // Number of distinct external bindings referenced by the operation.
  uint16_t binding_count;
} loom_aie2p_kernel_access_plan_t;

typedef enum loom_aie2p_kernel_realization_mode_e {
  // Bulk ingress, independent compute, and bulk egress are source-equivalent.
  LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT = 0,

  // Each source memory operation refreshes and flushes its local mirrors.
  LOOM_AIE2P_KERNEL_REALIZATION_COHERENT = 1,
} loom_aie2p_kernel_realization_mode_t;

typedef struct loom_aie2p_kernel_realization_plan_t {
  // Source kernel being replaced.
  loom_func_like_t function;

  // One plan entry per source kernel argument.
  loom_aie2p_kernel_binding_plan_t* bindings;

  // Number of source kernel arguments and binding entries.
  uint16_t binding_count;

  // Source memory operations that access external kernel bindings.
  loom_aie2p_kernel_access_plan_t* accesses;

  // Number of source memory operations in accesses.
  iree_host_size_t access_count;

  // Allocated entry capacity for accesses during planning.
  iree_host_size_t access_capacity;

  // Flat per-operation binding effects in source walk order.
  loom_aie2p_kernel_access_binding_t* access_bindings;

  // Number of initialized rows in access_bindings.
  iree_host_size_t access_binding_count;

  // Allocated row capacity for access_bindings during planning.
  iree_host_size_t access_binding_capacity;

  // Analysis-owned refinement rows for source roots replaced by local storage.
  struct {
    // Borrowed rows compacted in the analysis arena after planning completes.
    loom_view_memory_space_refinement_t* entries;
    // Number of refinements requiring translation during root replacement.
    iree_host_size_t count;
  } memory_space_refinements;

  // Selected alias-preserving realization mechanics.
  loom_aie2p_kernel_realization_mode_t mode;

  // Rows selected from the exact physical array profile.
  uint16_t dma_row;
  uint16_t compute_row;

  // Required endpoint and local-allocation alignment.
  uint64_t byte_alignment;

  // True only after every admission requirement has been proven.
  bool admitted;

  // True when binding analysis emitted an admission diagnostic.
  bool rejected;
} loom_aie2p_kernel_realization_plan_t;

typedef struct loom_aie2p_kernel_realization_builder_t {
  loom_pass_t* pass;
  loom_module_t* module;
  loom_rewriter_t rewriter;
  loom_aie2p_kernel_realization_plan_t* plan;
  loom_location_id_t location;
  loom_value_id_t byte_origin;
  loom_value_id_t capacity;
  loom_value_id_t memory;
  loom_type_t async_token_type;
  loom_type_t async_group_type;
} loom_aie2p_kernel_realization_builder_t;

static iree_status_t loom_aie2p_kernel_reject(loom_pass_t* pass,
                                              const loom_op_t* op,
                                              iree_string_view_t requirement) {
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static bool loom_aie2p_kernel_is_unit_launch(
    const loom_target_dispatch_workgroup_count_t* count,
    const loom_target_workgroup_size_t* size) {
  return count->x == 1 && count->y == 1 && count->z == 1 && size->x == 1 &&
         size->y == 1 && size->z == 1;
}

static loom_value_id_t loom_aie2p_kernel_access_root(
    const loom_value_fact_table_t* facts,
    const loom_view_region_table_t* regions, loom_value_id_t value_id) {
  const loom_view_region_t* region = NULL;
  if (loom_view_region_table_try_lookup(regions, value_id, &region)) {
    return region->root_value_id;
  }
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value_id);
  const loom_fact_context_t* context = &facts->context;
  loom_value_fact_buffer_reference_t buffer_reference;
  if (loom_value_facts_query_buffer_reference(context, value_facts,
                                              &buffer_reference)) {
    return loom_value_fact_buffer_reference_resolve_root_value(buffer_reference,
                                                               value_id);
  }
  loom_value_fact_view_reference_t view_reference;
  if (loom_value_facts_query_view_reference(context, value_facts,
                                            &view_reference)) {
    return loom_value_fact_view_reference_resolve_root_value(view_reference,
                                                             value_id);
  }
  return LOOM_VALUE_ID_INVALID;
}

static uint16_t loom_aie2p_kernel_find_binding(
    const loom_module_t* module,
    const loom_aie2p_kernel_realization_plan_t* plan,
    loom_value_id_t root_value_id) {
  if (root_value_id == LOOM_VALUE_ID_INVALID) {
    return UINT16_MAX;
  }
  const loom_value_t* root = loom_module_value(module, root_value_id);
  if (!loom_value_is_block_arg(root)) {
    return UINT16_MAX;
  }
  const uint16_t index = loom_value_def_index(root);
  return index < plan->binding_count &&
                 plan->bindings[index].source_argument == root_value_id
             ? index
             : UINT16_MAX;
}

static iree_status_t loom_aie2p_kernel_append_access(
    loom_pass_t* pass, loom_aie2p_kernel_realization_plan_t* plan,
    loom_op_t* operation, uint16_t binding_index,
    loom_view_access_flags_t access) {
  loom_aie2p_kernel_access_plan_t* operation_plan = NULL;
  if (plan->access_count != 0 &&
      plan->accesses[plan->access_count - 1].operation == operation) {
    operation_plan = &plan->accesses[plan->access_count - 1];
  } else {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(pass->arena, plan->access_count,
                              plan->access_count + 1, sizeof(*plan->accesses),
                              &plan->access_capacity, (void**)&plan->accesses));
    operation_plan = &plan->accesses[plan->access_count++];
    *operation_plan = (loom_aie2p_kernel_access_plan_t){
        .operation = operation,
        .binding_start = plan->access_binding_count,
    };
  }
  for (uint16_t i = 0; i < operation_plan->binding_count; ++i) {
    loom_aie2p_kernel_access_binding_t* existing =
        &plan->access_bindings[operation_plan->binding_start + i];
    if (existing->binding_index == binding_index) {
      existing->access |= access;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      pass->arena, plan->access_binding_count, plan->access_binding_count + 1,
      sizeof(*plan->access_bindings), &plan->access_binding_capacity,
      (void**)&plan->access_bindings));
  plan->access_bindings[plan->access_binding_count++] =
      (loom_aie2p_kernel_access_binding_t){
          .binding_index = binding_index,
          .access = access,
      };
  ++operation_plan->binding_count;
  return iree_ok_status();
}

static bool loom_aie2p_kernel_bindings_are_disjoint(
    const loom_aie2p_kernel_realization_plan_t* plan, uint16_t left,
    uint16_t right) {
  return loom_view_region_roots_are_disjoint(
      &plan->bindings[left].root_region, &plan->bindings[right].root_region);
}

static iree_status_t loom_aie2p_kernel_select_realization_mode(
    loom_pass_t* pass, loom_aie2p_kernel_realization_plan_t* plan) {
  bool alias_sensitive = false;
  for (uint16_t i = 0; i < plan->binding_count; ++i) {
    if (!plan->bindings[i].used) {
      continue;
    }
    for (uint16_t j = (uint16_t)(i + 1); j < plan->binding_count; ++j) {
      if (!plan->bindings[j].used ||
          !iree_any_bit_set(plan->bindings[i].access | plan->bindings[j].access,
                            LOOM_VIEW_ACCESS_WRITE)) {
        continue;
      }
      if (!loom_aie2p_kernel_bindings_are_disjoint(plan, i, j)) {
        alias_sensitive = true;
      }
      if (iree_any_bit_set(plan->bindings[i].access, LOOM_VIEW_ACCESS_WRITE) &&
          iree_any_bit_set(plan->bindings[j].access, LOOM_VIEW_ACCESS_WRITE) &&
          !loom_aie2p_kernel_bindings_are_disjoint(plan, i, j)) {
        plan->mode = LOOM_AIE2P_KERNEL_REALIZATION_COHERENT;
        return iree_ok_status();
      }
    }
  }
  if (!alias_sensitive) {
    plan->mode = LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT;
    return iree_ok_status();
  }

  bool* written = NULL;
  if (plan->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, plan->binding_count, sizeof(*written), (void**)&written));
    memset(written, 0, plan->binding_count * sizeof(*written));
  }
  const loom_block_t* entry =
      loom_region_entry_block(loom_func_like_body(plan->function));
  for (iree_host_size_t i = 0; i < plan->access_count; ++i) {
    const loom_aie2p_kernel_access_plan_t* access = &plan->accesses[i];
    if (access->operation->parent_block != entry) {
      plan->mode = LOOM_AIE2P_KERNEL_REALIZATION_COHERENT;
      return iree_ok_status();
    }
    for (uint16_t j = 0; j < access->binding_count; ++j) {
      const loom_aie2p_kernel_access_binding_t* binding =
          &plan->access_bindings[access->binding_start + j];
      if (!iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_READ)) {
        continue;
      }
      for (uint16_t k = 0; k < plan->binding_count; ++k) {
        if (k != binding->binding_index && written[k] &&
            !loom_aie2p_kernel_bindings_are_disjoint(
                plan, binding->binding_index, k)) {
          plan->mode = LOOM_AIE2P_KERNEL_REALIZATION_COHERENT;
          return iree_ok_status();
        }
      }
    }
    for (uint16_t j = 0; j < access->binding_count; ++j) {
      const loom_aie2p_kernel_access_binding_t* binding =
          &plan->access_bindings[access->binding_start + j];
      if (iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE)) {
        written[binding->binding_index] = true;
      }
    }
  }
  plan->mode = LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_plan_bindings(
    loom_pass_t* pass, loom_module_t* module,
    loom_aie2p_kernel_realization_plan_t* plan,
    const loom_value_fact_table_t* facts) {
  loom_region_t* body = loom_func_like_body(plan->function);
  loom_local_value_domain_t domain = {0};
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      module, body, pass->arena, &domain));

  loom_symbolic_expr_context_t expressions;
  loom_symbolic_expr_context_initialize(module, &domain, facts, pass->arena,
                                        &expressions);
  loom_view_region_table_t regions;
  iree_status_t status =
      loom_view_region_table_initialize(&domain, &expressions, &regions);
  if (iree_status_is_ok(status)) {
    status = loom_view_region_table_analyze(&regions);
  }

  for (uint16_t i = 0;
       i < plan->binding_count && iree_status_is_ok(status) && !plan->rejected;
       ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &plan->bindings[i];
    const loom_value_t* argument =
        loom_module_value(module, binding->source_argument);
    if (loom_value_has_no_uses(argument)) {
      continue;
    }
    if (iree_any_bit_set(loom_view_region_table_root_use_flags(
                             &regions, binding->source_argument),
                         LOOM_VIEW_ROOT_USE_OPAQUE)) {
      plan->rejected = true;
      status = loom_aie2p_kernel_reject(
          pass, plan->function.op,
          IREE_SV("buffer arguments reaching only typed views through "
                  "root-preserving refinements"));
      break;
    }

    int64_t maximum_end = 0;
    bool has_region = false;
    binding->byte_alignment = plan->byte_alignment;
    for (iree_host_size_t r = 0; r < regions.region_count; ++r) {
      const loom_view_region_t* region = &regions.regions[r];
      if (region->root_value_id != binding->source_argument ||
          region->access_flags == 0) {
        continue;
      }
      if (!loom_symbolic_expr_is_constant(&region->begin_byte_offset) ||
          !loom_symbolic_expr_is_constant(&region->end_byte_offset) ||
          region->begin_byte_offset.constant < 0 ||
          region->end_byte_offset.constant <=
              region->begin_byte_offset.constant) {
        plan->rejected = true;
        status = loom_aie2p_kernel_reject(
            pass, plan->function.op,
            IREE_SV("statically bounded nonempty typed-view footprints"));
        break;
      }
      maximum_end = iree_max(maximum_end, region->end_byte_offset.constant);
      binding->byte_alignment =
          iree_max(binding->byte_alignment, region->root_minimum_alignment);
      if (!has_region) {
        binding->root_region = *region;
        has_region = true;
      } else {
        if (binding->root_region.alias_scope_id != region->alias_scope_id) {
          binding->root_region.alias_scope_id =
              LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
        }
        if (binding->root_region.memory_space != region->memory_space) {
          binding->root_region.memory_space =
              LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
        }
        binding->root_region.origin = loom_value_fact_reference_origin_meet(
            binding->root_region.origin, region->origin);
      }
    }
    if (!iree_status_is_ok(status) || plan->rejected) {
      break;
    }
    binding->access = loom_view_region_table_root_access_flags(
        &regions, binding->source_argument);
    if (binding->access == 0 || maximum_end == 0 || !has_region) {
      plan->rejected = true;
      status = loom_aie2p_kernel_reject(
          pass, plan->function.op,
          IREE_SV("an observable statically bounded access for each used "
                  "buffer argument"));
      break;
    }
    binding->used = true;
    binding->byte_length = maximum_end;
    binding->root_region.access_flags = binding->access;
  }

  const loom_storage_access_function_t* storage_access = NULL;
  if (iree_status_is_ok(status) && !plan->rejected) {
    loom_storage_access_scope_t storage_scope;
    loom_storage_access_scope_initialize(module, pass->arena, &storage_scope);
    status = loom_storage_access_require_function(
        &storage_scope, plan->function, &domain, &storage_access);
  }
  if (iree_status_is_ok(status) && !plan->rejected &&
      storage_access->has_unknown_memory_access) {
    plan->rejected = true;
    status = loom_aie2p_kernel_reject(
        pass, plan->function.op,
        IREE_SV("fully described external memory effects"));
  }
  for (const loom_storage_access_use_t* use =
           storage_access && !plan->rejected ? storage_access->uses : NULL;
       use && iree_status_is_ok(status); use = use->next_in_function) {
    const loom_value_id_t root =
        loom_aie2p_kernel_access_root(facts, &regions, use->value_id);
    const uint16_t binding_index =
        loom_aie2p_kernel_find_binding(module, plan, root);
    if (binding_index == UINT16_MAX) {
      continue;
    }
    const loom_storage_access_effects_t effects =
        loom_storage_access_use_effects(use);
    if (iree_any_bit_set(
            effects, LOOM_STORAGE_ACCESS_ESCAPE | LOOM_STORAGE_ACCESS_ASYNC)) {
      plan->rejected = true;
      status = loom_aie2p_kernel_reject(
          pass, plan->function.op,
          IREE_SV("synchronous nonescaping external buffer accesses"));
      break;
    }
    const loom_memory_access_t memory_access =
        loom_memory_access_cast(module, use->operation);
    if (loom_memory_access_isa(memory_access) &&
        loom_memory_access_operation_kind_is_atomic(
            loom_memory_access_operation_kind(memory_access))) {
      plan->rejected = true;
      status = loom_aie2p_kernel_reject(
          pass, plan->function.op,
          IREE_SV("non-atomic external buffer accesses"));
      break;
    }
    loom_view_access_flags_t access = 0;
    if (iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_READ)) {
      access |= LOOM_VIEW_ACCESS_READ;
    }
    if (iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_WRITE)) {
      access |= LOOM_VIEW_ACCESS_WRITE;
    }
    if (access) {
      status = loom_aie2p_kernel_append_access(
          pass, plan, (loom_op_t*)use->operation, binding_index, access);
    }
  }
  if (iree_status_is_ok(status) && !plan->rejected) {
    for (uint16_t i = 0; i < plan->binding_count; ++i) {
      bool found = false;
      for (iree_host_size_t j = 0; j < plan->access_binding_count; ++j) {
        found |= plan->access_bindings[j].binding_index == i;
      }
      if (plan->bindings[i].used && !found) {
        plan->rejected = true;
        status = loom_aie2p_kernel_reject(
            pass, plan->function.op,
            IREE_SV("indexed external buffer accesses"));
        break;
      }
    }
  }
  if (iree_status_is_ok(status) && !plan->rejected) {
    status = loom_aie2p_kernel_select_realization_mode(pass, plan);
  }
  if (iree_status_is_ok(status) && !plan->rejected) {
    // The analysis arena outlives application. Compact its retained rows to the
    // roots this plan replaces; unrelated local refinements remain untouched.
    plan->memory_space_refinements.entries =
        regions.memory_space_refinements.entries;
    for (iree_host_size_t i = 0; i < regions.memory_space_refinements.count;
         ++i) {
      const loom_view_memory_space_refinement_t refinement =
          regions.memory_space_refinements.entries[i];
      if (loom_aie2p_kernel_find_binding(
              module, plan, refinement.root_value_id) != UINT16_MAX) {
        plan->memory_space_refinements
            .entries[plan->memory_space_refinements.count++] = refinement;
      }
    }
  }

  loom_local_value_domain_release(&domain);
  return status;
}

static iree_status_t loom_aie2p_kernel_plan(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_aie2p_kernel_realization_plan_t* out_plan) {
  *out_plan = (loom_aie2p_kernel_realization_plan_t){.function = function};
  const loom_target_facts_t* function_target_facts =
      loom_target_function_version_target_facts(pass->function_version);
  const loom_aie2p_target_facts_t* target =
      loom_aie2p_target_facts_cast(function_target_facts);
  if (target == NULL) {
    return loom_aie2p_kernel_reject(pass, function.op,
                                    IREE_SV("an AIE2P array-program target"));
  }
  if (target->device_profile == NULL) {
    return loom_aie2p_kernel_reject(pass, function.op,
                                    IREE_SV("an exact deployment profile"));
  }
  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(target->device_profile);
  const loom_xdna_tile_facts_t* compute_facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  const loom_xdna_tile_facts_t* shim_facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  if (family->column_count == 0 || compute_facts->row_count < 2) {
    return loom_aie2p_kernel_reject(
        pass, function.op,
        IREE_SV("two compute rows in the selected array partition"));
  }
  out_plan->dma_row = compute_facts->first_row;
  out_plan->compute_row = (uint16_t)(compute_facts->first_row + 1);
  out_plan->byte_alignment =
      iree_max((uint64_t)compute_facts->dma.address_alignment,
               (uint64_t)shim_facts->dma.address_alignment);

  const loom_value_slice_t workload =
      loom_kernel_workload_arg_ids(module, function.op);
  if (workload.count != 0) {
    return loom_aie2p_kernel_reject(
        pass, function.op, IREE_SV("a kernel without configuration arguments"));
  }
  uint8_t export_linkage = 0;
  if (loom_func_like_export_symbol(function) != LOOM_STRING_ID_INVALID ||
      loom_func_like_export_linkage(function, &export_linkage)) {
    return loom_aie2p_kernel_reject(
        pass, function.op,
        IREE_SV("the default kernel export name and linkage"));
  }
  loom_region_t* body = loom_func_like_body(function);
  if (body == NULL || body->block_count != 1 ||
      !loom_kernel_return_isa(loom_region_entry_block(body)->last_op)) {
    return loom_aie2p_kernel_reject(
        pass, function.op, IREE_SV("a single-block result-free kernel body"));
  }

  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(function,
                                                     function_target_facts),
      &facts));
  loom_target_dispatch_workgroup_count_t workgroup_count = {0};
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_kernel_def_static_workgroup_count_from_facts(
          module, function.op, facts, &workgroup_count) ||
      !loom_kernel_def_static_workgroup_size_from_facts(
          module, function.op, facts, &workgroup_size) ||
      !loom_aie2p_kernel_is_unit_launch(&workgroup_count, &workgroup_size)) {
    return loom_aie2p_kernel_reject(
        pass, function.op,
        IREE_SV("a static one-workgroup, one-workitem launch"));
  }
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(function.op);
  if (loom_kernel_launch_config_has_workgroup_cluster_size(launch_config)) {
    loom_target_workgroup_cluster_size_t cluster_size = {0};
    if (!loom_kernel_def_static_workgroup_cluster_size_from_facts(
            module, function.op, facts, &cluster_size) ||
        cluster_size.x != 1 || cluster_size.y != 1 || cluster_size.z != 1) {
      return loom_aie2p_kernel_reject(pass, function.op,
                                      IREE_SV("a unit workgroup cluster"));
    }
  }

  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &out_plan->binding_count);
  if (out_plan->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, out_plan->binding_count, sizeof(*out_plan->bindings),
        (void**)&out_plan->bindings));
    memset(out_plan->bindings, 0,
           out_plan->binding_count * sizeof(*out_plan->bindings));
  }
  for (uint16_t i = 0; i < out_plan->binding_count; ++i) {
    if (!loom_type_is_buffer(loom_module_value_type(module, arguments[i]))) {
      return loom_aie2p_kernel_reject(
          pass, function.op,
          IREE_SV("buffer-only invocation arguments after specialization"));
    }
    out_plan->bindings[i].source_argument = arguments[i];
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_kernel_plan_bindings(pass, module, out_plan, facts));
  if (out_plan->rejected) {
    return iree_ok_status();
  }

  uint64_t local_byte_length = 0;
  for (uint16_t i = 0; i < out_plan->binding_count; ++i) {
    const loom_aie2p_kernel_binding_plan_t* binding = &out_plan->bindings[i];
    if (!binding->used) {
      continue;
    }
    if (!iree_checked_align_u64(local_byte_length, binding->byte_alignment,
                                &local_byte_length) ||
        !iree_checked_add_u64(local_byte_length, (uint64_t)binding->byte_length,
                              &local_byte_length)) {
      return loom_aie2p_kernel_reject(
          pass, function.op,
          IREE_SV("staged binding footprints representable in local memory"));
    }
    if (out_plan->mode == LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT &&
        iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE) &&
        (!iree_checked_align_u64(local_byte_length, binding->byte_alignment,
                                 &local_byte_length) ||
         !iree_checked_add_u64(local_byte_length,
                               (uint64_t)binding->byte_length,
                               &local_byte_length))) {
      return loom_aie2p_kernel_reject(
          pass, function.op,
          IREE_SV("staged binding footprints representable in local memory"));
    }
  }
  if (local_byte_length > compute_facts->memory.local_capacity) {
    return loom_aie2p_kernel_reject(
        pass, function.op,
        IREE_SV("enough tile-local memory for staged kernel bindings"));
  }
  out_plan->admitted = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_build_constant(
    loom_builder_t* builder, int64_t value, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, loom_attr_i64(value),
                                                 type, location, &op));
  *out_value = loom_index_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_make_opaque_type(loom_builder_t* builder,
                                                        iree_string_view_t name,
                                                        loom_type_t* out_type) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_type = loom_type_dialect_opaque(name_id);
  return loom_module_intern_type(builder->module, *out_type, out_type);
}

static iree_status_t loom_aie2p_kernel_build_async_transfer(
    loom_aie2p_kernel_realization_builder_t* builder, loom_value_id_t source,
    loom_value_id_t target, loom_kernel_direction_t direction) {
  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_async_copy_build(
      &builder->rewriter.builder, source, target, LOOM_CACHE_SCOPE_WORKGROUP,
      LOOM_CACHE_TEMPORAL_REGULAR, direction, builder->async_token_type,
      builder->location, &copy_op));
  const loom_value_id_t token = loom_kernel_async_copy_token(copy_op);
  loom_op_t* group_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_async_group_build(
      &builder->rewriter.builder, &token, 1, builder->async_group_type,
      builder->location, &group_op));
  loom_op_t* wait_op = NULL;
  return loom_kernel_async_wait_build(&builder->rewriter.builder,
                                      loom_kernel_async_group_group(group_op),
                                      0, builder->location, &wait_op);
}

static iree_status_t loom_aie2p_kernel_build_local_buffer(
    loom_aie2p_kernel_realization_builder_t* builder,
    loom_aie2p_kernel_binding_plan_t* binding, loom_value_id_t* out_buffer) {
  loom_builder_t* ir = &builder->rewriter.builder;
  loom_value_id_t byte_length = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_constant(
      ir, binding->byte_length, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      builder->location, &byte_length));
  loom_op_t* allocation_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      ir, LOOM_BUFFER_ALLOCA_BUILD_FLAG_HAS_POOL,
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, builder->memory,
      binding->byte_alignment, byte_length, loom_type_buffer(),
      builder->location, &allocation_op));
  *out_buffer = loom_buffer_alloca_result(allocation_op);
  binding->record_view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(binding->byte_length), 0);
  return loom_module_intern_type(builder->module, binding->record_view_type,
                                 &binding->record_view_type);
}

static iree_status_t loom_aie2p_kernel_build_channel(
    loom_aie2p_kernel_realization_builder_t* builder,
    loom_aie2p_kernel_binding_plan_t* binding, bool output) {
  loom_builder_t* ir = &builder->rewriter.builder;
  loom_value_id_t buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_kernel_build_local_buffer(builder, binding, &buffer));

  const loom_type_t slot_view_type = loom_type_shaped_2d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(1),
      loom_dim_pack_static(binding->byte_length), 0);
  loom_type_t interned_slot_view_type = slot_view_type;
  IREE_RETURN_IF_ERROR(loom_module_intern_type(builder->module, slot_view_type,
                                               &interned_slot_view_type));
  loom_op_t* slots_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(ir, buffer, builder->byte_origin,
                                              interned_slot_view_type,
                                              builder->location, &slots_op));

  const loom_type_t payload_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(binding->byte_length), 0);
  loom_type_id_t payload_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_type_id(builder->module, payload_type,
                                                  &payload_type_id));
  loom_type_t channel_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_channel_type_make(builder->module, payload_type_id, &channel_type));
  const bool coherent =
      builder->plan->mode == LOOM_AIE2P_KERNEL_REALIZATION_COHERENT;
  IREE_RETURN_IF_ERROR(loom_read_type_make(
      builder->module, coherent ? LOOM_READ_TYPE_BUILD_FLAG_HAS_MODE : 0,
      payload_type_id, coherent ? LOOM_READ_TYPE_MODE_MUTABLE : 0,
      &binding->read_type));
  IREE_RETURN_IF_ERROR(loom_write_type_make(builder->module, payload_type_id,
                                            &binding->write_type));

  loom_op_t* channel_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_bind_build(
      ir, LOOM_CHANNEL_BIND_DISCIPLINE_FIFO, loom_buffer_view_result(slots_op),
      builder->capacity, channel_type, builder->location, &channel_op));
  if (output) {
    binding->output_buffer = buffer;
    binding->output_channel = loom_channel_bind_result(channel_op);
  } else {
    binding->input_buffer = buffer;
    binding->input_channel = loom_channel_bind_result(channel_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_build_storage(
    loom_aie2p_kernel_realization_builder_t* builder,
    loom_block_t* pipeline_entry) {
  loom_builder_t* ir = &builder->rewriter.builder;
  const int64_t memory_coordinates[] = {0, builder->plan->compute_row};
  loom_op_t* memory_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_memory_build(
      ir, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, NULL, 0, memory_coordinates,
      IREE_ARRAYSIZE(memory_coordinates), loom_type_pool(), builder->location,
      &memory_op));
  builder->memory = loom_pipeline_memory_result(memory_op);

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(builder->module, index_type, &index_type));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(builder->module, offset_type, &offset_type));
  IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_constant(
      ir, 0, offset_type, builder->location, &builder->byte_origin));
  IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_constant(
      ir, 1, index_type, builder->location, &builder->capacity));

  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    binding->external_binding = pipeline_entry->arg_ids[i];
    if (!binding->used) {
      continue;
    }
    const loom_type_t buffer_type = loom_type_buffer();
    loom_op_t* alignment_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_assume_alignment_build(
        ir, &binding->external_binding, 1, builder->plan->byte_alignment,
        &buffer_type, 1, builder->location, &alignment_op));
    binding->external_buffer = loom_op_results(alignment_op)[0];
    IREE_RETURN_IF_ERROR(
        loom_aie2p_kernel_build_channel(builder, binding, false));
    if (builder->plan->mode == LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT &&
        iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE)) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_kernel_build_channel(builder, binding, true));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_build_dma_strand(
    loom_aie2p_kernel_realization_builder_t* builder) {
  loom_builder_t* ir = &builder->rewriter.builder;
  const int64_t origins[] = {0, builder->plan->dma_row};
  const int64_t counts[] = {1, 1};
  const int64_t strides[] = {1, 1};
  loom_op_t* strand_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_strand_build(
      ir, 0, loom_symbol_ref_null(), NULL, 0, origins, IREE_ARRAYSIZE(origins),
      NULL, 0, counts, IREE_ARRAYSIZE(counts), NULL, 0, strides,
      IREE_ARRAYSIZE(strides), builder->location, &strand_op));
  const loom_builder_ip_t pipeline_ip = loom_builder_enter_region(
      ir, strand_op, loom_pipeline_strand_body(strand_op));

  // Admit every input before waiting for any output so several bindings cannot
  // form a producer/consumer cycle across the two finite strands.
  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    if (!binding->used) {
      continue;
    }
    loom_op_t* external_view_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        ir, binding->external_buffer, builder->byte_origin,
        binding->record_view_type, builder->location, &external_view_op));
    binding->external_view = loom_buffer_view_result(external_view_op);
    loom_op_t* reserve_op = NULL;
    IREE_RETURN_IF_ERROR(loom_channel_reserve_build(
        ir, binding->input_channel, binding->write_type,
        binding->record_view_type, builder->location, &reserve_op));
    IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_async_transfer(
        builder, binding->external_view, loom_channel_reserve_view(reserve_op),
        LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP));
    loom_op_t* publish_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_channel_publish_build(ir, loom_channel_reserve_write(reserve_op),
                                   builder->location, &publish_op));
  }

  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    if (!binding->used ||
        !iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE)) {
      continue;
    }
    loom_op_t* acquire_op = NULL;
    IREE_RETURN_IF_ERROR(loom_channel_acquire_build(
        ir, 0, 0, binding->output_channel, binding->read_type,
        binding->record_view_type, builder->location, &acquire_op));
    IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_async_transfer(
        builder, loom_channel_acquire_view(acquire_op), binding->external_view,
        LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL));
    loom_op_t* release_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_channel_release_build(ir, loom_channel_acquire_read(acquire_op),
                                   builder->location, &release_op));
  }
  loom_op_t* end_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_end_build(ir, builder->location, &end_op));
  loom_builder_restore(ir, pipeline_ip);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_move_source_body(
    loom_aie2p_kernel_realization_builder_t* builder, loom_op_t* end_op) {
  loom_block_t* source_block =
      loom_region_entry_block(loom_func_like_body(builder->plan->function));
  for (loom_op_t* op = source_block->first_op; op != NULL;) {
    loom_op_t* next_op = op->next_op;
    if (!loom_kernel_return_isa(op)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_move_before(&builder->rewriter, op, end_op));
    }
    op = next_op;
  }
  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    if (!binding->used) {
      continue;
    }
    const loom_value_id_t replacement =
        builder->plan->mode == LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT &&
                iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE)
            ? binding->output_buffer
            : binding->input_buffer;
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_except(
        &builder->rewriter, binding->source_argument, replacement,
        builder->plan->function.op));
  }
  // Original spaces have already informed alias and transfer planning. The
  // copied contents now live in workgroup storage, including immutable inputs;
  // retaining source constant/private/global facts would misdescribe that copy.
  for (iree_host_size_t i = 0;
       i < builder->plan->memory_space_refinements.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_set_attr(
        &builder->rewriter,
        (loom_op_t*)builder->plan->memory_space_refinements.entries[i]
            .operation,
        0, loom_attr_enum(LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP)));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_build_compute_strand(
    loom_aie2p_kernel_realization_builder_t* builder) {
  loom_builder_t* ir = &builder->rewriter.builder;
  const int64_t origins[] = {0, builder->plan->compute_row};
  const int64_t counts[] = {1, 1};
  const int64_t strides[] = {1, 1};
  loom_op_t* strand_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_strand_build(
      ir, 0, loom_symbol_ref_null(), NULL, 0, origins, IREE_ARRAYSIZE(origins),
      NULL, 0, counts, IREE_ARRAYSIZE(counts), NULL, 0, strides,
      IREE_ARRAYSIZE(strides), builder->location, &strand_op));
  const loom_builder_ip_t pipeline_ip = loom_builder_enter_region(
      ir, strand_op, loom_pipeline_strand_body(strand_op));

  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    if (!binding->used) {
      continue;
    }
    loom_op_t* acquire_op = NULL;
    IREE_RETURN_IF_ERROR(loom_channel_acquire_build(
        ir, 0, 0, binding->input_channel, binding->read_type,
        binding->record_view_type, builder->location, &acquire_op));
    binding->input_read = loom_channel_acquire_read(acquire_op);
    if (iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE)) {
      loom_op_t* reserve_op = NULL;
      IREE_RETURN_IF_ERROR(loom_channel_reserve_build(
          ir, binding->output_channel, binding->write_type,
          binding->record_view_type, builder->location, &reserve_op));
      binding->output_write = loom_channel_reserve_write(reserve_op);
      loom_value_id_t byte_length = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_constant(
          ir, binding->byte_length, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
          builder->location, &byte_length));
      loom_op_t* copy_op = NULL;
      IREE_RETURN_IF_ERROR(loom_buffer_copy_build(
          ir, binding->input_buffer, builder->byte_origin,
          binding->output_buffer, builder->byte_origin, byte_length,
          builder->location, &copy_op));
    }
  }

  loom_op_t* end_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_end_build(ir, builder->location, &end_op));
  loom_builder_restore(ir, pipeline_ip);
  IREE_RETURN_IF_ERROR(loom_aie2p_kernel_move_source_body(builder, end_op));

  loom_builder_set_before(ir, end_op);
  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    if (!binding->used) {
      continue;
    }
    if (iree_any_bit_set(binding->access, LOOM_VIEW_ACCESS_WRITE)) {
      loom_op_t* publish_op = NULL;
      IREE_RETURN_IF_ERROR(loom_channel_publish_build(
          ir, binding->output_write, builder->location, &publish_op));
    }
    loom_op_t* release_op = NULL;
    IREE_RETURN_IF_ERROR(loom_channel_release_build(
        ir, binding->input_read, builder->location, &release_op));
  }
  loom_builder_restore(ir, pipeline_ip);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_build_coherent_strand(
    loom_aie2p_kernel_realization_builder_t* builder) {
  loom_builder_t* ir = &builder->rewriter.builder;
  const int64_t origins[] = {0, builder->plan->compute_row};
  const int64_t counts[] = {1, 1};
  const int64_t strides[] = {1, 1};
  loom_op_t* strand_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_strand_build(
      ir, 0, loom_symbol_ref_null(), NULL, 0, origins, IREE_ARRAYSIZE(origins),
      NULL, 0, counts, IREE_ARRAYSIZE(counts), NULL, 0, strides,
      IREE_ARRAYSIZE(strides), builder->location, &strand_op));
  const loom_builder_ip_t pipeline_ip = loom_builder_enter_region(
      ir, strand_op, loom_pipeline_strand_body(strand_op));

  for (uint16_t i = 0; i < builder->plan->binding_count; ++i) {
    loom_aie2p_kernel_binding_plan_t* binding = &builder->plan->bindings[i];
    if (!binding->used) {
      continue;
    }
    loom_op_t* external_view_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        ir, binding->external_buffer, builder->byte_origin,
        binding->record_view_type, builder->location, &external_view_op));
    binding->external_view = loom_buffer_view_result(external_view_op);
  }

  loom_op_t* end_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_end_build(ir, builder->location, &end_op));
  loom_builder_restore(ir, pipeline_ip);
  IREE_RETURN_IF_ERROR(loom_aie2p_kernel_move_source_body(builder, end_op));

  for (iree_host_size_t i = 0; i < builder->plan->access_count; ++i) {
    loom_aie2p_kernel_access_plan_t* access = &builder->plan->accesses[i];
    loom_builder_set_before(ir, access->operation);
    for (uint16_t j = 0; j < access->binding_count; ++j) {
      loom_aie2p_kernel_access_binding_t* access_binding =
          &builder->plan->access_bindings[access->binding_start + j];
      loom_aie2p_kernel_binding_plan_t* binding =
          &builder->plan->bindings[access_binding->binding_index];
      loom_op_t* reserve_op = NULL;
      IREE_RETURN_IF_ERROR(loom_channel_reserve_build(
          ir, binding->input_channel, binding->write_type,
          binding->record_view_type, builder->location, &reserve_op));
      IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_async_transfer(
          builder, binding->external_view,
          loom_channel_reserve_view(reserve_op),
          LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP));
      loom_op_t* publish_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_channel_publish_build(ir, loom_channel_reserve_write(reserve_op),
                                     builder->location, &publish_op));
      loom_op_t* acquire_op = NULL;
      IREE_RETURN_IF_ERROR(loom_channel_acquire_build(
          ir, LOOM_CHANNEL_ACQUIRE_BUILD_FLAG_HAS_MODE,
          LOOM_CHANNEL_MODE_MUTABLE, binding->input_channel, binding->read_type,
          binding->record_view_type, builder->location, &acquire_op));
      access_binding->read = loom_channel_acquire_read(acquire_op);
      access_binding->view = loom_channel_acquire_view(acquire_op);
    }
    loom_builder_set_after(ir, access->operation);
    for (uint16_t j = 0; j < access->binding_count; ++j) {
      const loom_aie2p_kernel_access_binding_t* access_binding =
          &builder->plan->access_bindings[access->binding_start + j];
      if (!iree_any_bit_set(access_binding->access, LOOM_VIEW_ACCESS_WRITE)) {
        continue;
      }
      loom_aie2p_kernel_binding_plan_t* binding =
          &builder->plan->bindings[access_binding->binding_index];
      IREE_RETURN_IF_ERROR(loom_aie2p_kernel_build_async_transfer(
          builder, access_binding->view, binding->external_view,
          LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL));
    }
    for (uint16_t j = 0; j < access->binding_count; ++j) {
      const loom_aie2p_kernel_access_binding_t* access_binding =
          &builder->plan->access_bindings[access->binding_start + j];
      loom_op_t* release_op = NULL;
      IREE_RETURN_IF_ERROR(loom_channel_release_build(
          ir, access_binding->read, builder->location, &release_op));
    }
  }
  loom_builder_restore(ir, pipeline_ip);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_remap_predicates(
    loom_module_t* module, const loom_value_id_t* source_arguments,
    const loom_value_id_t* target_arguments, uint16_t argument_count,
    loom_func_like_t target) {
  if (target.vtable->predicates_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  const uint8_t attribute_index = target.vtable->predicates_attr_index;
  for (uint16_t i = 0; i < argument_count; ++i) {
    loom_attribute_t remapped = loom_op_const_attrs(target.op)[attribute_index];
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_references(
        module, remapped, source_arguments[i], target_arguments[i], &remapped,
        &changed));
    if (changed) {
      IREE_RETURN_IF_ERROR(
          loom_op_set_attr(module, target.op, attribute_index, remapped));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_kernel_apply(
    loom_pass_t* pass, loom_module_t* module,
    loom_aie2p_kernel_realization_plan_t* plan) {
  loom_aie2p_kernel_realization_builder_t builder = {
      .pass = pass,
      .module = module,
      .plan = plan,
      .location = plan->function.op->location,
      .byte_origin = LOOM_VALUE_ID_INVALID,
      .capacity = LOOM_VALUE_ID_INVALID,
      .memory = LOOM_VALUE_ID_INVALID,
  };
  loom_rewriter_initialize(&builder.rewriter, module, pass->arena);
  loom_builder_t* ir = &builder.rewriter.builder;

  const loom_symbol_ref_t callee = loom_func_like_callee(plan->function);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(plan->function, &source_argument_count);
  loom_type_t* binding_types = NULL;
  iree_status_t status = iree_ok_status();
  if (source_argument_count != 0) {
    status = iree_arena_allocate_array(pass->arena, source_argument_count,
                                       sizeof(*binding_types),
                                       (void**)&binding_types);
    if (iree_status_is_ok(status)) {
      for (uint16_t i = 0; i < source_argument_count; ++i) {
        binding_types[i] = loom_type_buffer();
      }
    }
  }

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(plan->function, &predicate_count);
  const loom_symbol_ref_t target = loom_func_like_target(plan->function);
  loom_pipeline_def_build_flags_t build_flags =
      LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_SCOPE |
      LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_VISIBILITY;
  const bool retained = pass->function_version != NULL &&
                        iree_any_bit_set(pass->function_version->flags,
                                         LOOM_FUNCTION_VERSION_FLAG_RETAIN);
  if (retained) {
    build_flags |= LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_RETAIN;
  }
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_TARGET;
  }
  if (predicate_count != 0) {
    build_flags |= LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_PREDICATES;
  }
  loom_op_t* pipeline_op = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_pipeline_def_build(
        ir, build_flags, LOOM_PIPELINE_SCOPE_KERNEL,
        LOOM_PIPELINE_VISIBILITY_PUBLIC,
        retained ? LOOM_PIPELINE_RETAIN_RETAIN : 0, target, callee, NULL, 0,
        binding_types, source_argument_count, predicates, predicate_count,
        builder.location, &pipeline_op);
  }
  loom_func_like_t pipeline = {0};
  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments = NULL;
  if (iree_status_is_ok(status)) {
    pipeline = loom_func_like_cast(module, pipeline_op);
    target_arguments = loom_func_like_arg_ids(pipeline, &target_argument_count);
  }
  if (iree_status_is_ok(status) &&
      target_argument_count != source_argument_count) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "generated AIE2P pipeline argument count is inconsistent");
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_kernel_remap_predicates(
        module, source_arguments, target_arguments, source_argument_count,
        pipeline);
  }

  if (iree_status_is_ok(status)) {
    iree_host_size_t comment_count = 0;
    const iree_string_view_t* comments =
        loom_module_op_comments(module, plan->function.op, &comment_count);
    if (comment_count != 0) {
      status = loom_module_attach_op_comments(module, pipeline_op, comments,
                                              comment_count);
    }
  }

  loom_builder_ip_t module_ip = {0};
  bool entered_pipeline = false;
  if (iree_status_is_ok(status)) {
    module_ip = loom_builder_enter_region(ir, pipeline_op,
                                          loom_pipeline_def_body(pipeline_op));
    entered_pipeline = true;
    status = loom_aie2p_kernel_make_opaque_type(
        ir, IREE_SV("kernel.async.token"), &builder.async_token_type);
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_kernel_make_opaque_type(
        ir, IREE_SV("kernel.async.group"), &builder.async_group_type);
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_kernel_build_storage(
        &builder, loom_region_entry_block(loom_pipeline_def_body(pipeline_op)));
  }
  if (iree_status_is_ok(status)) {
    if (plan->mode == LOOM_AIE2P_KERNEL_REALIZATION_SNAPSHOT) {
      status = loom_aie2p_kernel_build_dma_strand(&builder);
      if (iree_status_is_ok(status)) {
        status = loom_aie2p_kernel_build_compute_strand(&builder);
      }
    } else {
      status = loom_aie2p_kernel_build_coherent_strand(&builder);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* finish_op = NULL;
    status = loom_pipeline_finish_build(ir, builder.location, &finish_op);
  }
  if (entered_pipeline) {
    loom_builder_restore(ir, module_ip);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(&builder.rewriter, plan->function.op);
  }
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(module, pipeline_op,
                                        loom_op_vtable(module, pipeline_op));
    loom_function_version_update(pass->function_version, pipeline);
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&builder.rewriter);
  return status;
}

iree_status_t loom_aie2p_kernel_realize(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function) {
  loom_aie2p_kernel_realization_plan_t plan;
  IREE_RETURN_IF_ERROR(loom_aie2p_kernel_plan(pass, module, function, &plan));
  if (!plan.admitted) {
    return iree_ok_status();
  }
  return loom_aie2p_kernel_apply(pass, module, &plan);
}
