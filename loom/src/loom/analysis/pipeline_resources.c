// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_resources.h"

#include <stdlib.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/pipeline/ops.h"

static iree_status_t loom_pipeline_resources_reject(
    const loom_op_t* op, iree_string_view_t requirement,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_LOWERING_066,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static int loom_pipeline_resources_compare_root(const void* lhs,
                                                const void* rhs) {
  const loom_value_id_t lhs_root =
      ((const loom_pipeline_resource_allocation_t*)lhs)->root_value_id;
  const loom_value_id_t rhs_root =
      ((const loom_pipeline_resource_allocation_t*)rhs)->root_value_id;
  return (lhs_root > rhs_root) - (lhs_root < rhs_root);
}

static int loom_pipeline_resources_compare_channel(const void* lhs,
                                                   const void* rhs) {
  const loom_value_id_t lhs_id =
      ((const loom_pipeline_resource_channel_t*)lhs)->identity.value_id;
  const loom_value_id_t rhs_id =
      ((const loom_pipeline_resource_channel_t*)rhs)->identity.value_id;
  return (lhs_id > rhs_id) - (lhs_id < rhs_id);
}

static int loom_pipeline_resources_compare_pool(const void* lhs,
                                                const void* rhs) {
  const loom_value_id_t lhs_id =
      ((const loom_pipeline_resource_pool_binding_t*)lhs)->value_id;
  const loom_value_id_t rhs_id =
      ((const loom_pipeline_resource_pool_binding_t*)rhs)->value_id;
  return (lhs_id > rhs_id) - (lhs_id < rhs_id);
}

static iree_status_t loom_pipeline_resources_select_memory(
    const loom_op_t* op, const loom_value_fact_table_t* facts,
    const loom_pipeline_resource_pool_t* pools,
    const loom_pipeline_resource_memory_t* memories,
    iree_host_size_t memory_count, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* arena, uint32_t* out_pool_index) {
  *out_pool_index = UINT32_MAX;
  const loom_attribute_t static_coordinates =
      loom_pipeline_memory_static_coordinates(op);
  const loom_value_slice_t dynamic_coordinates =
      loom_pipeline_memory_coordinates(op);
  uint64_t* coordinates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, static_coordinates.count,
                                sizeof(*coordinates), (void**)&coordinates));
  uint16_t dynamic_index = 0;
  for (uint32_t i = 0; i < static_coordinates.count; ++i) {
    int64_t coordinate = static_coordinates.i64_array[i];
    if (coordinate == INT64_MIN &&
        !loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(
                facts, dynamic_coordinates.values[dynamic_index++]),
            &coordinate)) {
      return loom_pipeline_resources_reject(
          op, IREE_SV("specialized memory-selection coordinates"),
          diagnostic_emitter);
    }
    if (coordinate < 0) {
      return loom_pipeline_resources_reject(
          op, IREE_SV("nonnegative memory-selection coordinates"),
          diagnostic_emitter);
    }
    coordinates[i] = (uint64_t)coordinate;
  }
  const loom_value_fact_memory_space_t memory_space =
      loom_pipeline_memory_memory_space(op);
  for (iree_host_size_t i = 0; i < memory_count; ++i) {
    const loom_pipeline_resource_memory_t* memory = &memories[i];
    if (memory->rank == static_coordinates.count &&
        pools[memory->pool_index].memory_space == memory_space &&
        memcmp(memory->coordinates, coordinates,
               memory->rank * sizeof(*coordinates)) == 0) {
      *out_pool_index = memory->pool_index;
      return iree_ok_status();
    }
  }
  return loom_pipeline_resources_reject(
      op, IREE_SV("memory at the selected coordinates in this invocation"),
      diagnostic_emitter);
}

iree_status_t loom_pipeline_resources_build(
    loom_module_t* module, loom_func_like_t pipeline,
    const loom_value_fact_table_t* facts,
    const loom_pipeline_resource_pool_t* pools, iree_host_size_t pool_count,
    const loom_pipeline_resource_memory_t* memories,
    iree_host_size_t memory_count,
    const loom_pipeline_resource_pool_binding_t* bindings,
    iree_host_size_t binding_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_resources_t* out_resources, bool* out_valid) {
  *out_resources = (loom_pipeline_resources_t){0};
  *out_valid = false;
  const loom_region_t* body = loom_pipeline_def_body(pipeline.op);
  if (body->block_count != 1) {
    return loom_pipeline_resources_reject(
        pipeline.op,
        IREE_SV("specialized straight-line allocation construction"),
        diagnostic_emitter);
  }

  loom_pipeline_resources_t resources = {
      .pools = pools,
      .pool_count = pool_count,
  };
  loom_pipeline_resource_allocation_t* allocations = NULL;
  iree_host_size_t allocation_capacity = 0;
  loom_pipeline_resource_channel_t* channels = NULL;
  iree_host_size_t channel_capacity = 0;
  loom_pipeline_resource_strand_t* strands = NULL;
  iree_host_size_t strand_capacity = 0;
  loom_op_t** compositions = NULL;
  iree_host_size_t composition_capacity = 0;
  loom_pipeline_resource_pool_binding_t* pool_bindings = NULL;
  iree_host_size_t pool_binding_capacity = 0;
  if (binding_count) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, binding_count, sizeof(*pool_bindings),
                              &pool_binding_capacity, (void**)&pool_bindings));
    memcpy(pool_bindings, bindings, binding_count * sizeof(*pool_bindings));
    resources.pool_binding_count = binding_count;
  }
  const loom_block_t* entry = loom_region_const_entry_block(body);
  for (loom_op_t* op = entry->first_op; op; op = op->next_op) {
    if (loom_pipeline_memory_isa(op)) {
      uint32_t pool_index = UINT32_MAX;
      IREE_RETURN_IF_ERROR(loom_pipeline_resources_select_memory(
          op, facts, pools, memories, memory_count, diagnostic_emitter, arena,
          &pool_index));
      if (pool_index == UINT32_MAX) {
        return iree_ok_status();
      }
      if (resources.pool_binding_count == pool_binding_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, resources.pool_binding_count,
            resources.pool_binding_count + 1, sizeof(*pool_bindings),
            &pool_binding_capacity, (void**)&pool_bindings));
      }
      pool_bindings[resources.pool_binding_count++] =
          (loom_pipeline_resource_pool_binding_t){
              .value_id = loom_pipeline_memory_result(op),
              .pool_index = pool_index,
          };
      continue;
    }
    if (loom_pipeline_strand_isa(op)) {
      const loom_region_t* strand_body = loom_pipeline_strand_body(op);
      loom_op_t* call = loom_region_const_entry_block(strand_body)->first_op;
      if (strand_body->block_count == 1 && loom_pipeline_end_isa(call)) {
        continue;
      }
      if (strand_body->block_count != 1 || !loom_func_call_isa(call) ||
          !loom_pipeline_end_isa(call->next_op)) {
        return loom_pipeline_resources_reject(
            op,
            IREE_SV("outlined strand bodies with explicit callable captures"),
            diagnostic_emitter);
      }
      if (resources.strand_count == strand_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, resources.strand_count, resources.strand_count + 1,
            sizeof(*strands), &strand_capacity, (void**)&strands));
      }
      strands[resources.strand_count++] = (loom_pipeline_resource_strand_t){
          .declaration = op,
          .call = call,
      };
      continue;
    }
    if (loom_pipeline_compose_isa(op)) {
      if (resources.composition_count == composition_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, resources.composition_count, resources.composition_count + 1,
            sizeof(*compositions), &composition_capacity,
            (void**)&compositions));
      }
      compositions[resources.composition_count++] = op;
      continue;
    }
    if (op->region_count) {
      return loom_pipeline_resources_reject(
          op, IREE_SV("composed and specialized allocation construction"),
          diagnostic_emitter);
    }
    if (loom_channel_bind_isa(op)) {
      loom_pipeline_resource_channel_t channel = {
          .binding = op,
          .identity = {.value_id = loom_channel_bind_result(op)},
      };
      if (!loom_value_facts_query_view_reference(
              &facts->context,
              loom_value_fact_table_lookup(facts,
                                           loom_channel_bind_storage(op)),
              &channel.storage)) {
        return loom_pipeline_resources_reject(
            op, IREE_SV("an explicit storage projection for channel slots"),
            diagnostic_emitter);
      }
      int64_t capacity = 0;
      if (!loom_value_facts_as_exact_i64(
              loom_value_fact_table_lookup(facts,
                                           loom_channel_bind_capacity(op)),
              &capacity) ||
          capacity <= 0) {
        return loom_pipeline_resources_reject(
            op, IREE_SV("a positive specialized channel capacity"),
            diagnostic_emitter);
      }
      channel.capacity = (uint64_t)capacity;
      if (resources.channel_count == channel_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, resources.channel_count, resources.channel_count + 1,
            sizeof(*channels), &channel_capacity, (void**)&channels));
      }
      channels[resources.channel_count++] = channel;
      continue;
    }
    if (!loom_buffer_alloca_isa(op)) {
      const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
      if (!loom_pipeline_finish_isa(op) &&
          (!iree_any_bit_set(traits, LOOM_TRAIT_PURE) ||
           loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
           iree_any_bit_set(
               traits, LOOM_TRAIT_OBSERVABLE_EFFECT | LOOM_TRAIT_CONVERGENT))) {
        return loom_pipeline_resources_reject(
            op, IREE_SV("an execution placement for construction effects"),
            diagnostic_emitter);
      }
      continue;
    }

    const loom_value_id_t pool_value_id = loom_buffer_alloca_pool(op);
    iree_host_size_t binding = 0;
    for (; binding < resources.pool_binding_count; ++binding) {
      if (pool_bindings[binding].value_id == pool_value_id) {
        break;
      }
    }
    if (binding == resources.pool_binding_count) {
      return loom_pipeline_resources_reject(
          op, IREE_SV("an explicit admitted binding for the allocation pool"),
          diagnostic_emitter);
    }
    const uint32_t pool_index = pool_bindings[binding].pool_index;
    if (loom_buffer_alloca_memory_space(op) != pools[pool_index].memory_space) {
      return loom_pipeline_resources_reject(
          op, IREE_SV("an allocation memory space matching its selected pool"),
          diagnostic_emitter);
    }
    const loom_value_id_t root_value_id = loom_buffer_alloca_result(op);
    loom_value_fact_buffer_reference_t reference = {0};
    int64_t byte_length = 0;
    if (!loom_value_facts_query_buffer_reference(
            &facts->context, loom_value_fact_table_lookup(facts, root_value_id),
            &reference) ||
        !loom_value_facts_as_exact_i64(reference.maximum_byte_extent,
                                       &byte_length) ||
        byte_length < 0) {
      return loom_pipeline_resources_reject(
          op, IREE_SV("a finite specialized allocation extent"),
          diagnostic_emitter);
    }
    if (resources.allocation_count == allocation_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, resources.allocation_count, resources.allocation_count + 1,
          sizeof(*allocations), &allocation_capacity, (void**)&allocations));
    }
    loom_pipeline_resource_allocation_t allocation = {
        .root_value_id = root_value_id,
        .pool_value_id = pool_value_id,
        .pool_index = pool_index,
        .byte_length = (uint64_t)byte_length,
        .byte_alignment = reference.minimum_alignment,
    };
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
        pools[pool_index].packing, allocation.byte_length,
        allocation.byte_alignment, NULL, 0, &allocation.byte_offset));
    allocations[resources.allocation_count++] = allocation;
  }
  if (resources.allocation_count > 1) {
    qsort(allocations, resources.allocation_count, sizeof(*allocations),
          loom_pipeline_resources_compare_root);
  }
  resources.allocations = allocations;
  if (resources.channel_count > 1) {
    qsort(channels, resources.channel_count, sizeof(*channels),
          loom_pipeline_resources_compare_channel);
  }
  resources.channels = channels;
  resources.strands = strands;
  resources.compositions = compositions;
  if (resources.pool_binding_count > 1) {
    qsort(pool_bindings, resources.pool_binding_count, sizeof(*pool_bindings),
          loom_pipeline_resources_compare_pool);
  }
  resources.pool_bindings = pool_bindings;
  *out_resources = resources;
  *out_valid = true;
  return iree_ok_status();
}

const loom_pipeline_resource_pool_binding_t*
loom_pipeline_resources_lookup_pool(const loom_pipeline_resources_t* resources,
                                    loom_value_id_t value_id) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = resources->pool_binding_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_resource_pool_binding_t* binding =
        &resources->pool_bindings[middle];
    if (binding->value_id == value_id) {
      return binding;
    }
    if (binding->value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return NULL;
}

const loom_pipeline_resource_channel_t* loom_pipeline_resources_lookup_channel(
    const loom_pipeline_resources_t* resources, loom_value_id_t value_id) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = resources->channel_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_resource_channel_t* channel =
        &resources->channels[middle];
    if (channel->identity.value_id == value_id) {
      return channel;
    }
    if (channel->identity.value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return NULL;
}

const loom_pipeline_resource_allocation_t*
loom_pipeline_resources_lookup_allocation(
    const loom_pipeline_resources_t* resources, loom_value_id_t root_value_id) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = resources->allocation_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_resource_allocation_t* allocation =
        &resources->allocations[middle];
    if (allocation->root_value_id == root_value_id) {
      return allocation;
    }
    if (allocation->root_value_id < root_value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return NULL;
}

iree_status_t loom_pipeline_resources_check_capacity(
    const loom_pipeline_resources_t* resources, const loom_op_t* entry,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid) {
  *out_valid = true;
  for (iree_host_size_t i = 0; i < resources->pool_count; ++i) {
    const uint64_t required =
        loom_source_storage_packing_requirement(resources->pools[i].packing)
            .byte_length;
    if (required <= resources->pools[i].byte_capacity) {
      continue;
    }
    *out_valid = false;
    const loom_diagnostic_param_t params[] = {
        loom_param_u64(i),
        loom_param_u64(required),
        loom_param_u64(resources->pools[i].byte_capacity),
    };
    const loom_diagnostic_emission_t emission = {
        .op = entry,
        .error = LOOM_ERR_LOWERING_067,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(diagnostic_emitter, &emission));
  }
  return iree_ok_status();
}
