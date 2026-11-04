// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/invocations.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_invocation_frame_t {
  // Stable public occurrence, published before discovering descendants.
  loom_pipeline_invocation_t invocation;
  // Enclosing active traversal frame.
  struct loom_pipeline_invocation_frame_t* parent;
  // First child awaiting expansion, in retained construction order.
  struct loom_pipeline_invocation_frame_t* first;
  // Final child in the pending list.
  struct loom_pipeline_invocation_frame_t* last;
  // Next sibling awaiting expansion.
  struct loom_pipeline_invocation_frame_t* next;
  // Next fully bound occurrence in parent-before-child inventory order.
  struct loom_pipeline_invocation_frame_t* ordered_next;
} loom_pipeline_invocation_frame_t;

typedef struct loom_pipeline_invocation_scope_t {
  // Bound source scope whose steps are awaiting argument binding.
  loom_pipeline_invocation_schedule_t* schedule;
  // Next nested scope discovered from retained control.
  struct loom_pipeline_invocation_scope_t* next;
} loom_pipeline_invocation_scope_t;

const loom_pipeline_resource_allocation_t*
loom_pipeline_invocation_lookup_allocation(
    const loom_pipeline_invocation_t* invocation, loom_value_id_t root) {
  iree_host_size_t begin = 0, end = invocation->allocation_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_invocation_allocation_t* entry =
        &invocation->allocations[middle];
    if (entry->root_value_id < root) {
      begin = middle + 1;
    } else if (entry->root_value_id > root) {
      end = middle;
    } else {
      return entry->allocation;
    }
  }
  return NULL;
}

static iree_status_t loom_pipeline_invocations_bind_allocations(
    loom_pipeline_invocation_t* invocation, iree_arena_allocator_t* arena) {
  const loom_pipeline_invocation_t* parent = invocation->parent;
  const iree_host_size_t inherited = parent ? parent->allocation_count : 0;
  const loom_pipeline_resources_t* resources = invocation->resources;
  invocation->allocation_count = inherited + resources->allocation_count;
  if (!resources->allocation_count) {
    invocation->allocations = parent ? parent->allocations : NULL;
    return iree_ok_status();
  }
  loom_pipeline_invocation_allocation_t* allocations = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, invocation->allocation_count,
                                sizeof(*allocations), (void**)&allocations));
  invocation->allocations = allocations;
  iree_host_size_t imported = 0, local = 0;
  for (iree_host_size_t i = 0; i < invocation->allocation_count; ++i) {
    if (imported < inherited &&
        (local == resources->allocation_count ||
         parent->allocations[imported].root_value_id <
             resources->allocations[local].root_value_id)) {
      allocations[i] = parent->allocations[imported++];
    } else {
      const loom_pipeline_resource_allocation_t* allocation =
          &resources->allocations[local++];
      allocations[i] = (loom_pipeline_invocation_allocation_t){
          .root_value_id = allocation->root_value_id, .allocation = allocation};
    }
  }
  return iree_ok_status();
}

static bool loom_pipeline_invocations_channel_less(
    const loom_pipeline_invocation_channel_t* channels,
    const iree_host_size_t* left, const iree_host_size_t* right) {
  return (uintptr_t)&channels[*left].source->identity <
         (uintptr_t)&channels[*right].source->identity;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(
    loom_pipeline_invocations_sort_channels, iree_host_size_t,
    const loom_pipeline_invocation_channel_t*,
    loom_pipeline_invocations_channel_less)

iree_host_size_t loom_pipeline_invocations_channel_index(
    const loom_pipeline_invocations_t* invocations,
    const loom_channel_identity_t* identity) {
  iree_host_size_t begin = 0, end = invocations->channel_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const iree_host_size_t index = invocations->channel_order[middle];
    const loom_channel_identity_t* candidate =
        &invocations->channels[index].source->identity;
    if ((uintptr_t)candidate < (uintptr_t)identity) {
      begin = middle + 1;
    } else if ((uintptr_t)candidate > (uintptr_t)identity) {
      end = middle;
    } else {
      return index;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_pipeline_invocations_finish(
    const loom_pipeline_invocation_frame_t* root, iree_arena_allocator_t* arena,
    loom_pipeline_invocations_t* invocations) {
  const loom_pipeline_invocation_t** entries = NULL;
  loom_pipeline_invocation_worker_t* workers = NULL;
  loom_pipeline_invocation_channel_t* channels = NULL;
  iree_host_size_t* channel_order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->count, sizeof(*entries), (void**)&entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->worker_count, sizeof(*workers), (void**)&workers));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->channel_count, sizeof(*channels), (void**)&channels));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->channel_count, sizeof(*channel_order),
      (void**)&channel_order));
  invocations->entries = entries;
  invocations->workers = workers;
  invocations->channels = channels;
  invocations->channel_order = channel_order;
  iree_host_size_t worker_index = 0, channel_index = 0;
  for (const loom_pipeline_invocation_frame_t* frame = root; frame;
       frame = frame->ordered_next) {
    const loom_pipeline_invocation_t* invocation = &frame->invocation;
    const loom_pipeline_resources_t* resources = invocation->resources;
    entries[invocation->ordinal] = invocation;
    for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
      workers[worker_index++] = (loom_pipeline_invocation_worker_t){
          .invocation = invocation, .strand = &resources->strands[i]};
    }
    for (iree_host_size_t i = 0; i < resources->channel_count; ++i) {
      const loom_pipeline_resource_channel_t* source = &resources->channels[i];
      channels[channel_index] = (loom_pipeline_invocation_channel_t){
          .invocation = invocation,
          .source = source,
          .allocation = loom_pipeline_invocation_lookup_allocation(
              invocation, source->storage.root_value_id)};
      channel_order[channel_index] = channel_index;
      ++channel_index;
    }
  }
  loom_pipeline_invocations_sort_channels(channels, channel_order,
                                          invocations->channel_count);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_invocations_reject(
    const loom_op_t* operation, iree_string_view_t requirement,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  const loom_diagnostic_param_t parameters[] = {loom_param_string(requirement)};
  return iree_diagnostic_emit(diagnostic_emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = operation,
                                  .error = LOOM_ERR_LOWERING_066,
                                  .params = parameters,
                                  .param_count = IREE_ARRAYSIZE(parameters),
                              });
}

static iree_status_t loom_pipeline_invocations_append(
    loom_module_t* module, loom_pipeline_invocation_frame_t* parent,
    loom_op_t* call, const loom_pipeline_schedule_t* schedule,
    iree_arena_allocator_t* arena,
    const loom_pipeline_invocation_t** out_invocation) {
  loom_pipeline_invocation_frame_t* child = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*child), (void**)&child));
  const loom_symbol_ref_t callee = loom_pipeline_compose_callee(call);
  *child = (loom_pipeline_invocation_frame_t){
      .invocation =
          {
              .function = loom_func_like_cast(
                  module,
                  module->symbols.entries[callee.symbol_id].defining_op),
              .call = call,
              .parent = &parent->invocation,
              .schedule = schedule,
          },
      .parent = parent,
  };
  if (parent->last) {
    parent->last->next = child;
  } else {
    parent->first = child;
  }
  parent->last = child;
  *out_invocation = &child->invocation;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_invocations_collect(
    loom_module_t* module, loom_pipeline_invocation_frame_t* frame,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid) {
  *out_valid = false;
  loom_pipeline_invocation_t* invocation = &frame->invocation;
  const loom_pipeline_resources_t* resources = invocation->resources;
  const loom_pipeline_invocation_t** compositions = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, resources->composition_count,
                                sizeof(*compositions), (void**)&compositions));
  invocation->compositions = compositions;
  for (iree_host_size_t i = 0; i < resources->composition_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_invocations_append(
        module, frame, resources->compositions[i], NULL, arena,
        &compositions[i]));
  }
  loom_pipeline_invocation_schedule_t* schedules = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, resources->schedule_count,
                                sizeof(*schedules), (void**)&schedules));
  invocation->schedules = schedules;
  loom_pipeline_invocation_scope_t* first = NULL;
  loom_pipeline_invocation_scope_t* last = NULL;
  for (iree_host_size_t i = 0; i < resources->schedule_count; ++i) {
    schedules[i] = (loom_pipeline_invocation_schedule_t){
        .source = resources->schedules[i]};
    loom_pipeline_invocation_scope_t* scope = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
    *scope = (loom_pipeline_invocation_scope_t){.schedule = &schedules[i]};
    if (last) {
      last->next = scope;
    } else {
      first = scope;
    }
    last = scope;
  }
  for (loom_pipeline_invocation_scope_t* scope = first; scope;
       scope = scope->next) {
    loom_pipeline_invocation_schedule_t* schedule = scope->schedule;
    const loom_pipeline_schedule_t* source = schedule->source;
    const loom_pipeline_invocation_step_t** blocks = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, source->control->graph.block_count,
                                  sizeof(*blocks), (void**)&blocks));
    schedule->blocks = blocks;
    for (iree_host_size_t b = 0; b < source->control->graph.block_count; ++b) {
      const loom_pipeline_schedule_block_t* block = &source->blocks[b];
      loom_pipeline_invocation_step_t* bindings = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, block->step_count, sizeof(*bindings), (void**)&bindings));
      memset(bindings, 0, block->step_count * sizeof(*bindings));
      blocks[b] = bindings;
      for (iree_host_size_t i = 0; i < block->step_count; ++i) {
        const loom_pipeline_schedule_step_t* step = &block->steps[i];
        if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION) {
          IREE_RETURN_IF_ERROR(loom_pipeline_invocations_append(
              module, frame, step->operation, source, arena,
              &bindings[i].child));
        } else if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_SERIAL) {
          loom_pipeline_invocation_schedule_t* nested = NULL;
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(arena, sizeof(*nested), (void**)&nested));
          *nested =
              (loom_pipeline_invocation_schedule_t){.source = step->nested};
          bindings[i].nested = nested;
          loom_pipeline_invocation_scope_t* pending = NULL;
          IREE_RETURN_IF_ERROR(
              iree_arena_allocate(arena, sizeof(*pending), (void**)&pending));
          *pending = (loom_pipeline_invocation_scope_t){.schedule = nested};
          last->next = pending;
          last = pending;
        } else if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE) {
          const loom_value_id_t channel =
              loom_channel_acquire_channel(step->operation);
          bindings[i].channel =
              loom_pipeline_resources_lookup_channel(resources, channel);
          if (!bindings[i].channel) {
            const loom_pipeline_channel_binding_t* incoming =
                loom_pipeline_arguments_lookup_channel(
                    invocation->arguments.channels,
                    invocation->arguments.channel_count, channel);
            if (incoming) {
              bindings[i].channel = incoming->channel;
            }
          }
          if (!bindings[i].channel) {
            return loom_pipeline_invocations_reject(
                step->operation,
                IREE_SV("an invocation-owned collective channel binding"),
                diagnostic_emitter);
          }
        }
      }
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_pipeline_invocations_build(
    loom_module_t* module, loom_func_like_t root_function,
    const loom_value_fact_table_t* root_facts,
    const loom_pipeline_resources_t* root_resources,
    const loom_pipeline_resource_memory_t* memories,
    iree_host_size_t memory_count,
    loom_pipeline_invocation_environment_t environment,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_invocations_t* out_invocations, bool* out_valid) {
  *out_invocations = (loom_pipeline_invocations_t){
      .count = 1,
      .worker_count = root_resources->strand_count,
      .channel_count = root_resources->channel_count};
  *out_valid = false;
  loom_pipeline_invocation_frame_t* root = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*root), (void**)&root));
  *root = (loom_pipeline_invocation_frame_t){
      .invocation = {.function = root_function,
                     .facts = root_facts,
                     .resources = root_resources},
  };
  IREE_RETURN_IF_ERROR(
      loom_pipeline_invocations_bind_allocations(&root->invocation, arena));
  if (!root_resources->composition_count && !root_resources->schedule_count) {
    IREE_RETURN_IF_ERROR(
        loom_pipeline_invocations_finish(root, arena, out_invocations));
    *out_valid = true;
    return iree_ok_status();
  }
  bool* active = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*active), (void**)&active));
  memset(active, 0, module->symbols.count * sizeof(*active));
  active[loom_func_like_callee(root_function).symbol_id] = true;
  bool collected = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_invocations_collect(
      module, root, diagnostic_emitter, arena, &collected));
  if (!collected) {
    return iree_ok_status();
  }
  loom_pipeline_invocation_frame_t* frame = root;
  loom_pipeline_invocation_frame_t* last = root;
  while (frame) {
    if (!frame->first) {
      active[loom_func_like_callee(frame->invocation.function).symbol_id] =
          false;
      frame = frame->parent;
      continue;
    }
    loom_pipeline_invocation_frame_t* child = frame->first;
    frame->first = child->next;
    loom_pipeline_invocation_t* invocation = &child->invocation;
    const loom_symbol_id_t symbol =
        loom_func_like_callee(invocation->function).symbol_id;
    if (!loom_pipeline_def_isa(invocation->function.op)) {
      return loom_pipeline_invocations_reject(
          invocation->call, IREE_SV("a visible child pipeline construction"),
          diagnostic_emitter);
    }
    if (active[symbol]) {
      return loom_pipeline_invocations_reject(
          invocation->call,
          IREE_SV("finite nonrecursive pipeline construction"),
          diagnostic_emitter);
    }
    active[symbol] = true;
    const loom_pipeline_invocation_t* parent = invocation->parent;
    loom_value_fact_table_t* facts = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*facts), (void**)&facts));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(facts, arena, 0));
    facts->context.resolve_type_domain =
        parent->facts->context.resolve_type_domain;
    facts->context.resolve_region_target =
        parent->facts->context.resolve_region_target;
    facts->context.target_facts =
        environment.target_facts
            ? environment.target_facts(environment.user_data,
                                       invocation->function)
            : loom_value_fact_table_block_target_facts(
                  parent->facts, invocation->call->parent_block);
    invocation->facts = facts;
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_pipeline_arguments_bind(
        module, invocation->call, invocation->function, parent->resources,
        parent->facts, parent->arguments.channels,
        parent->arguments.channel_count, facts, diagnostic_emitter, arena,
        &invocation->arguments, &valid));
    if (!valid) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_compute(facts, module, invocation->function));
    loom_pipeline_resources_t* resources = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*resources), (void**)&resources));
    invocation->resources = resources;
    IREE_RETURN_IF_ERROR(loom_pipeline_resources_build(
        module, invocation->function, facts, root_resources->pools,
        root_resources->pool_count, memories, memory_count,
        invocation->arguments.pools, invocation->arguments.pool_count,
        diagnostic_emitter, arena, resources, &valid));
    if (!valid) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_pipeline_invocations_bind_allocations(invocation, arena));
    invocation->first_worker = out_invocations->worker_count;
    invocation->ordinal = out_invocations->count++;
    out_invocations->worker_count += resources->strand_count;
    out_invocations->channel_count += resources->channel_count;
    last->ordered_next = child;
    last = child;
    IREE_RETURN_IF_ERROR(loom_pipeline_invocations_collect(
        module, child, diagnostic_emitter, arena, &collected));
    if (!collected) {
      return iree_ok_status();
    }
    frame = child;
  }
  IREE_RETURN_IF_ERROR(
      loom_pipeline_invocations_finish(root, arena, out_invocations));
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_pipeline_invocations_analyze_workers(
    loom_module_t* module, const loom_pipeline_invocations_t* invocations,
    const loom_func_like_t* functions,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_worker_t** out_workers, bool* out_valid) {
  *out_workers = NULL;
  *out_valid = false;
  loom_storage_access_scope_t storage_access;
  loom_storage_access_scope_initialize(module, arena, &storage_access);
  loom_pipeline_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->worker_count, sizeof(*workers), (void**)&workers));
  for (iree_host_size_t i = 0; i < invocations->count; ++i) {
    const loom_pipeline_invocation_t* invocation = invocations->entries[i];
    if (!invocation->resources->strand_count) {
      continue;
    }
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_pipeline_workers_build(
        module, invocation->resources, invocation->facts,
        invocation->arguments.channels, invocation->arguments.channel_count,
        functions ? functions + invocation->first_worker : NULL, NULL,
        &storage_access, diagnostic_emitter, arena,
        workers + invocation->first_worker, &valid));
    if (!valid) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_workers_prove_sources(
      &storage_access, workers, invocations->worker_count));
  *out_workers = workers;
  *out_valid = true;
  return iree_ok_status();
}
