// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/control.h"

#include <string.h>

#include "loom/ir/symbol_map.h"
#include "loom/ir/value_replacement.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/util/adaptive_sort.h"

typedef enum loom_pipeline_control_site_kind_e {
  LOOM_PIPELINE_CONTROL_STRAND,
  LOOM_PIPELINE_CONTROL_EMPTY,
  LOOM_PIPELINE_CONTROL_COMPOSITION,
  LOOM_PIPELINE_CONTROL_SERIAL,
  LOOM_PIPELINE_CONTROL_COLLECTIVE,
  LOOM_PIPELINE_CONTROL_COLLECTIVE_MEMBER,
  LOOM_PIPELINE_CONTROL_YIELD,
  LOOM_PIPELINE_CONTROL_END,
} loom_pipeline_control_site_kind_t;

typedef struct loom_pipeline_control_site_t {
  // Retained construction operation, independent of any projected engine.
  const loom_op_t* source;
  // Exact clone for the engine currently being materialized.
  loom_op_t* target;
  // Rewrite selected from the retained construction inventory.
  loom_pipeline_control_site_kind_t kind;
  // Worker ordinal for a strand site.
  iree_host_size_t worker;
  // Complete child occurrence for a composition site.
  const loom_pipeline_invocation_t* child;
  // Joining scope for a composition, or the scope being flattened for SERIAL.
  const loom_pipeline_invocation_schedule_t* scope;
  // Parent serial scope ordering a nested SERIAL site, or NULL at the root.
  const loom_pipeline_invocation_schedule_t* enclosing_scope;
  // Closed immutable record read, present only on its acquisition site.
  const loom_pipeline_collective_t* collective;
  // An autonomous single-child choice split around concurrent parent work.
  bool deferred;
} loom_pipeline_control_site_t;

typedef struct loom_pipeline_control_sites_t {
  // Pointer-indexed source sites, populated before cloning starts.
  loom_pipeline_control_site_t* entries;
  // Number of construction sites in this invocation.
  iree_host_size_t count;
  // Allocated source-site capacity.
  iree_host_size_t capacity;
  // Exact clone visitation order, used for consuming nested regions first.
  loom_pipeline_control_site_t** visits;
  // Number of sites observed during the current clone.
  iree_host_size_t visit_count;
  // At least one independently progressing scope needs boundary projections.
  bool has_deferred;
} loom_pipeline_control_sites_t;

typedef enum loom_pipeline_control_projection_e {
  LOOM_PIPELINE_CONTROL_COMPLETE,
  LOOM_PIPELINE_CONTROL_START,
  LOOM_PIPELINE_CONTROL_RETIRE,
} loom_pipeline_control_projection_t;

static bool loom_pipeline_control_site_less(
    const loom_pipeline_control_site_t* left,
    const loom_pipeline_control_site_t* right) {
  return (uintptr_t)left->source < (uintptr_t)right->source;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_control_sort_sites,
                          loom_pipeline_control_site_t,
                          loom_pipeline_control_site_less)

static iree_status_t loom_pipeline_control_add_site(
    iree_arena_allocator_t* arena, loom_pipeline_control_sites_t* sites,
    loom_pipeline_control_site_t site) {
  if (sites->count == sites->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, sites->count, sites->count + 1, sizeof(*sites->entries),
        &sites->capacity, (void**)&sites->entries));
  }
  sites->entries[sites->count++] = site;
  return iree_ok_status();
}

static loom_pipeline_control_site_t* loom_pipeline_control_find_site(
    const loom_pipeline_control_sites_t* sites, const loom_op_t* source) {
  iree_host_size_t begin = 0, end = sites->count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    loom_pipeline_control_site_t* site = &sites->entries[middle];
    if ((uintptr_t)site->source < (uintptr_t)source) {
      begin = middle + 1;
    } else if ((uintptr_t)site->source > (uintptr_t)source) {
      end = middle;
    } else {
      return site;
    }
  }
  return NULL;
}

static iree_status_t loom_pipeline_control_observe(void* user_data,
                                                   const loom_op_t* source,
                                                   loom_op_t* target) {
  loom_pipeline_control_sites_t* sites = user_data;
  loom_pipeline_control_site_t* site =
      loom_pipeline_control_find_site(sites, source);
  if (site) {
    site->target = target;
    sites->visits[sites->visit_count++] = site;
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_control_collect(
    const loom_pipeline_invocation_t* invocation,
    const loom_pipeline_execution_t* execution,
    const loom_pipeline_control_options_t* options,
    iree_arena_allocator_t* arena, loom_pipeline_control_sites_t* sites) {
  const loom_pipeline_resources_t* resources = invocation->resources;
  for (iree_host_size_t i = 0; i < resources->empty_strand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_control_add_site(
        arena, sites,
        (loom_pipeline_control_site_t){.source = resources->empty_strands[i],
                                       .kind = LOOM_PIPELINE_CONTROL_EMPTY}));
  }
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_control_add_site(
        arena, sites,
        (loom_pipeline_control_site_t){
            .source = resources->strands[i].declaration,
            .kind = LOOM_PIPELINE_CONTROL_STRAND,
            .worker = invocation->first_worker + i}));
  }
  for (iree_host_size_t i = 0; i < resources->composition_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_control_add_site(
        arena, sites,
        (loom_pipeline_control_site_t){
            .source = resources->compositions[i],
            .kind = LOOM_PIPELINE_CONTROL_COMPOSITION,
            .child = invocation->compositions[i]}));
  }
  for (iree_host_size_t i = 0; i < resources->schedule_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_control_add_site(
        arena, sites,
        (loom_pipeline_control_site_t){
            .source = invocation->schedules[i].source->declaration,
            .kind = LOOM_PIPELINE_CONTROL_SERIAL,
            .scope = &invocation->schedules[i]}));
  }
  // The retained queue contains source classifications and occurrence bindings.
  // Appending nested scopes visits each once, without recursion or IR walks.
  for (iree_host_size_t i = 0; i < sites->count; ++i) {
    if (sites->entries[i].kind != LOOM_PIPELINE_CONTROL_SERIAL) {
      continue;
    }
    const loom_pipeline_invocation_schedule_t* scope = sites->entries[i].scope;
    const iree_host_size_t first_step = sites->count;
    bool autonomous = false, instructions = false;
    for (iree_host_size_t b = 0; b < scope->source->control->graph.block_count;
         ++b) {
      const loom_pipeline_schedule_block_t* block = &scope->source->blocks[b];
      for (iree_host_size_t s = 0; s < block->step_count; ++s) {
        const loom_pipeline_schedule_step_t* step = &block->steps[s];
        loom_pipeline_control_site_t site = {.source = step->operation,
                                             .scope = scope};
        switch (step->kind) {
          case LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION:
            site.kind = LOOM_PIPELINE_CONTROL_COMPOSITION;
            site.child = scope->blocks[b][s].child;
            if (options->boundary_workers) {
              for (iree_host_size_t engine = 0;
                   engine < execution->engine_count; ++engine) {
                const iree_host_size_t index =
                    site.child->ordinal * execution->engine_count + engine;
                instructions |= execution->workers[index] != IREE_HOST_SIZE_MAX;
                autonomous |=
                    options->boundary_workers[index] != IREE_HOST_SIZE_MAX &&
                    execution->workers[index] == IREE_HOST_SIZE_MAX;
              }
            }
            break;
          case LOOM_PIPELINE_SCHEDULE_STEP_SERIAL:
            site.kind = LOOM_PIPELINE_CONTROL_SERIAL;
            site.scope = scope->blocks[b][s].nested;
            site.enclosing_scope = scope;
            break;
          case LOOM_PIPELINE_SCHEDULE_STEP_YIELD:
            site.kind = LOOM_PIPELINE_CONTROL_YIELD;
            break;
          case LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE:
            site.kind = LOOM_PIPELINE_CONTROL_COLLECTIVE;
            site.collective = step->collective;
            instructions = true;
            break;
          case LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE_MEMBER:
            site.kind = LOOM_PIPELINE_CONTROL_COLLECTIVE_MEMBER;
            break;
          default:
            continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_pipeline_control_add_site(arena, sites, site));
      }
    }
    // The retained single-child choice can bracket concurrent parent work only
    // when its children progress autonomously. Classify its boundary while its
    // exact step range is available, before the pointer index reorders sites.
    if (autonomous) {
      if (instructions || sites->entries[i].enclosing_scope ||
          !scope->source->single_child_choice) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "autonomous projection requires an instruction-free acyclic "
            "outer single-child choice with parent-owned resources");
      }
      sites->has_deferred = true;
      sites->entries[i].deferred = true;
      for (iree_host_size_t j = first_step; j < sites->count; ++j) {
        sites->entries[j].deferred = true;
      }
    }
  }
  const loom_region_t* body = loom_func_like_body(invocation->function);
  IREE_RETURN_IF_ERROR(loom_pipeline_control_add_site(
      arena, sites,
      (loom_pipeline_control_site_t){
          .source = loom_region_const_entry_block(body)->last_op,
          .kind = LOOM_PIPELINE_CONTROL_END}));
  loom_pipeline_control_sort_sites(sites->entries, sites->count);
  return iree_arena_allocate_array(arena, sites->count, sizeof(*sites->visits),
                                   (void**)&sites->visits);
}

// A serial region has no result tuple. Move its CFG into the current callable;
// each retained yield branches to the original continuation. Inner serial
// regions have already been consumed in reverse clone visitation order.
static iree_status_t loom_pipeline_control_flatten_scope(
    loom_rewriter_t* rewriter, const loom_pipeline_control_sites_t* sites,
    const loom_pipeline_control_site_t* site) {
  loom_op_t* serial = site->target;
  loom_region_t* source = loom_pipeline_serial_body(serial);
  loom_region_t* target = serial->parent_block->parent_region;
  const loom_op_t* parent = serial->parent_op;
  uint16_t position = 0;
  loom_region_try_block_index(target, serial->parent_block, &position);
  const uint16_t count = source->block_count;
  loom_block_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_rewriter_move_region_blocks(
      rewriter, source, serial, target, position + 1, serial->parent_op,
      &entry));
  loom_block_t* continuation = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      rewriter->module, target, position + count + 1, &continuation));
  while (serial->next_op) {
    IREE_RETURN_IF_ERROR(loom_rewriter_move_to_block_end(
        rewriter, serial->next_op, continuation, (loom_op_t*)parent));
  }
  const loom_pipeline_schedule_t* schedule = site->scope->source;
  for (iree_host_size_t b = 0; b < schedule->control->graph.block_count; ++b) {
    const loom_pipeline_schedule_block_t* block = &schedule->blocks[b];
    const loom_pipeline_schedule_step_t* last =
        &block->steps[block->step_count - 1];
    if (last->kind != LOOM_PIPELINE_SCHEDULE_STEP_YIELD) {
      continue;
    }
    loom_op_t* yield =
        loom_pipeline_control_find_site(sites, last->operation)->target;
    loom_builder_set_before(&rewriter->builder, yield);
    loom_op_t* branch = NULL;
    IREE_RETURN_IF_ERROR(loom_cfg_br_build(&rewriter->builder, continuation,
                                           NULL, 0, yield->location, &branch));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, yield));
  }
  loom_builder_set_before(&rewriter->builder, serial);
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_br_build(&rewriter->builder, entry, NULL, 0,
                                         serial->location, &branch));
  return loom_rewriter_erase(rewriter, serial);
}

static iree_status_t loom_pipeline_control_call_worker(
    loom_rewriter_t* rewriter, const loom_pipeline_realization_t* realization,
    const loom_pipeline_control_options_t* options,
    const loom_pipeline_control_site_t* site,
    const loom_value_id_t* state_arguments) {
  loom_channel_materialization_instance_t* worker =
      &realization->instances[site->worker].channels;
  const loom_op_t* call =
      loom_region_entry_block(loom_pipeline_strand_body(site->target))
          ->first_op;
  const loom_value_slice_t actuals = loom_func_call_operands(call);
  uint16_t count = 0;
  const loom_value_id_t* formals =
      loom_func_like_arg_ids(worker->function, &count);
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, count, sizeof(*arguments), (void**)&arguments));
  loom_local_value_domain_restore(&worker->value_domain);
  for (uint16_t i = 0; i < count; ++i) {
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_try_ordinal(&worker->value_domain, formals[i]);
    if (ordinal != LOOM_VALUE_ORDINAL_INVALID) {
      arguments[i] = actuals.values[ordinal];
    } else {
      // New formals belong to the physical owner's explicit state ABI.
      uint16_t state = 0;
      while (options->worker_state_arguments[site->worker][state] !=
             formals[i]) {
        ++state;
      }
      arguments[i] = state_arguments[state];
    }
  }
  loom_local_value_domain_release(&worker->value_domain);
  loom_op_t* emitted = NULL;
  return loom_func_call_build(
      &rewriter->builder, 0, 0, 0, 0, loom_func_like_callee(worker->function),
      arguments, count, NULL, 0, NULL, 0, site->source->location, &emitted);
}

static iree_status_t loom_pipeline_control_call_child(
    loom_rewriter_t* rewriter, const loom_pipeline_control_site_t* site,
    const loom_pipeline_control_function_t* function,
    const loom_value_id_t* state_arguments) {
  const loom_value_slice_t actuals = loom_call_like_operands(
      loom_call_like_cast(rewriter->module, site->target));
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, function->argument_count,
                                sizeof(*arguments), (void**)&arguments));
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    const uint16_t source = function->arguments[i];
    arguments[i] = source < actuals.count
                       ? actuals.values[source]
                       : state_arguments[source - actuals.count];
  }
  loom_op_t* emitted = NULL;
  return loom_func_call_build(&rewriter->builder, 0, 0, 0, 0,
                              loom_func_like_callee(function->function),
                              arguments, function->argument_count, NULL, 0,
                              NULL, 0, site->source->location, &emitted);
}

static iree_status_t loom_pipeline_control_call_projection(
    loom_rewriter_t* rewriter, const loom_pipeline_control_function_t* function,
    const loom_value_id_t* source_arguments) {
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, function->argument_count,
                                sizeof(*arguments), (void**)&arguments));
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    arguments[i] = source_arguments[function->arguments[i]];
  }
  loom_op_t* call = NULL;
  return loom_func_call_build(&rewriter->builder, 0, 0, 0, 0,
                              loom_func_like_callee(function->function),
                              arguments, function->argument_count, NULL, 0,
                              NULL, 0, LOOM_LOCATION_UNKNOWN, &call);
}

static iree_status_t loom_pipeline_control_collective(
    loom_rewriter_t* rewriter, const loom_pipeline_control_options_t* options,
    const loom_pipeline_control_sites_t* sites,
    const loom_pipeline_control_site_t* site, iree_host_size_t engine,
    const loom_value_id_t* state_arguments) {
  const loom_pipeline_collective_t* collective = site->collective;
  const iree_host_size_t count = collective->operation_count;
  loom_op_t** operations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, count, sizeof(*operations), (void**)&operations));
  for (iree_host_size_t i = 0; i < count; ++i) {
    operations[i] =
        loom_pipeline_control_find_site(sites, collective->operations[i])
            ->target;
  }
  loom_value_id_t payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(options->emit.collective_begin(
      options->emit.user_data, &rewriter->builder, engine, site->scope,
      collective, state_arguments,
      loom_module_value_type(rewriter->module,
                             loom_channel_acquire_view(operations[0])),
      &payload));
  const bool reader = payload != LOOM_VALUE_ID_INVALID;
  if (reader) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, loom_channel_acquire_view(operations[0]), payload));
  }
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, collective->result_count,
                                sizeof(*values), (void**)&values));
  for (iree_host_size_t i = 0; i < collective->result_count; ++i) {
    const loom_pipeline_collective_result_t* result = &collective->results[i];
    values[i] =
        reader ? loom_op_results(operations[result->operation])[result->result]
               : LOOM_VALUE_ID_INVALID;
  }
  loom_builder_set_before(&rewriter->builder, operations[count - 1]);
  IREE_RETURN_IF_ERROR(
      options->emit.collective_end(options->emit.user_data, &rewriter->builder,
                                   engine, site->scope, collective, values));
  for (iree_host_size_t i = 0; i < collective->result_count; ++i) {
    const loom_pipeline_collective_result_t* result = &collective->results[i];
    const loom_value_id_t original =
        loom_op_results(operations[result->operation])[result->result];
    if (original != values[i]) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_replace_all_uses_with(rewriter, original, values[i]));
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, operations[count - 1]));
  if (!reader) {
    for (iree_host_size_t i = count - 1; i > 1; --i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, operations[i - 1]));
    }
  }
  return loom_rewriter_erase(rewriter, operations[0]);
}

static iree_status_t loom_pipeline_control_project(
    loom_rewriter_t* rewriter, const loom_pipeline_realization_t* realization,
    const loom_pipeline_execution_t* execution,
    const loom_pipeline_control_options_t* options,
    const loom_pipeline_invocation_t* invocation, iree_host_size_t engine,
    loom_symbol_ref_t symbol, loom_pipeline_control_sites_t* sites,
    loom_pipeline_control_function_t* functions,
    loom_pipeline_control_projection_t projection,
    const loom_pipeline_control_function_t* boundaries,
    loom_pipeline_control_function_t* out_function) {
  loom_module_t* module = rewriter->module;
  uint16_t source_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(invocation->function, &source_count);
  const iree_host_size_t total_count = source_count + options->state_count;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, total_count, sizeof(*types), (void**)&types));
  for (uint16_t i = 0; i < source_count; ++i) {
    types[i] = loom_module_value_type(module, source_arguments[i]);
  }
  for (uint16_t i = 0; i < options->state_count; ++i) {
    types[source_count + i] = options->state_types[i];
  }
  loom_builder_enter_region(&rewriter->builder, NULL, module->body);
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &rewriter->builder, LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY, 0, 0, 0,
      0, 0, 0, LOOM_INLINE_POLICY_INLINE, loom_symbol_ref_null(),
      loom_attr_absent(), 0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, types,
      total_count, NULL, 0, NULL, 0, NULL, 0, invocation->function.op->location,
      &function));
  loom_region_t* body = loom_func_def_body(function);
  loom_block_t* entry = loom_region_entry_block(body);
  IREE_RETURN_IF_ERROR(loom_module_assign_value_types(
      module, source_arguments, entry->arg_ids, source_count));
  // Keep stable IDs while DCE and argument removal shrink the entry tuple.
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, total_count, sizeof(*arguments), (void**)&arguments));
  if (total_count) {
    memcpy(arguments, entry->arg_ids, total_count * sizeof(*arguments));
  }
  const loom_value_id_t* state_arguments =
      total_count ? arguments + source_count : NULL;
  sites->visit_count = 0;
  const loom_ir_remap_options_t remap_options = {
      .clone_observer = {.fn = loom_pipeline_control_observe,
                         .user_data = sites}};
  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, rewriter->arena,
                                                &remap_options, &remap));
  loom_builder_enter_region(&rewriter->builder, function, body);
  IREE_RETURN_IF_ERROR(loom_ir_clone_region_blocks(
      &rewriter->builder, loom_func_like_body(invocation->function), body, 1,
      &remap));
  IREE_ASSERT_EQ(sites->visit_count, sites->count);
  loom_block_t* cloned_entry = body->blocks[1];
  for (uint16_t i = 0; i < source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, cloned_entry->arg_ids[i], arguments[i]));
  }
  for (uint16_t i = source_count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(loom_block_remove_arg(module, cloned_entry, i - 1));
  }
  loom_builder_set_block(&rewriter->builder, entry);
  rewriter->builder.ip.parent_op = function;
  if (projection == LOOM_PIPELINE_CONTROL_COMPLETE && sites->has_deferred) {
    IREE_RETURN_IF_ERROR(loom_pipeline_control_call_projection(
        rewriter, &boundaries[0], arguments));
  }
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_br_build(&rewriter->builder, cloned_entry, NULL,
                                         0, function->location, &branch));
  for (iree_host_size_t reverse = sites->visit_count; reverse > 0; --reverse) {
    const loom_pipeline_control_site_t* site = sites->visits[reverse - 1];
    loom_builder_set_before(&rewriter->builder, site->target);
    switch (site->kind) {
      case LOOM_PIPELINE_CONTROL_STRAND:
        if (projection == LOOM_PIPELINE_CONTROL_COMPLETE &&
            options->worker_engines[site->worker] == engine) {
          IREE_RETURN_IF_ERROR(loom_pipeline_control_call_worker(
              rewriter, realization, options, site, state_arguments));
        }
        IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
        break;
      case LOOM_PIPELINE_CONTROL_EMPTY: {
        IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
        break;
      }
      case LOOM_PIPELINE_CONTROL_COLLECTIVE:
        if (projection == LOOM_PIPELINE_CONTROL_COMPLETE &&
            loom_pipeline_execution_schedule(execution, site->scope)
                    ->workers[engine] != IREE_HOST_SIZE_MAX) {
          IREE_RETURN_IF_ERROR(loom_pipeline_control_collective(
              rewriter, options, sites, site, engine, state_arguments));
        }
        break;
      case LOOM_PIPELINE_CONTROL_COLLECTIVE_MEMBER:
        break;  // The acquire site consumes the complete closed borrow.
      case LOOM_PIPELINE_CONTROL_COMPOSITION: {
        if (site->deferred) {
          if (projection != LOOM_PIPELINE_CONTROL_COMPLETE &&
              options->boundary_workers[site->child->ordinal *
                                            execution->engine_count +
                                        engine] != IREE_HOST_SIZE_MAX) {
            IREE_RETURN_IF_ERROR(options->emit.boundary(
                options->emit.user_data, &rewriter->builder, engine, NULL,
                site->child, projection == LOOM_PIPELINE_CONTROL_START));
          }
          IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
          break;
        }
        if (projection != LOOM_PIPELINE_CONTROL_COMPLETE) {
          IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
          break;
        }
        if (site->scope &&
            loom_pipeline_execution_schedule(execution, site->scope)
                    ->workers[engine] == IREE_HOST_SIZE_MAX) {
          IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
          break;
        }
        if (options->emit.boundary) {
          IREE_RETURN_IF_ERROR(options->emit.boundary(
              options->emit.user_data, &rewriter->builder, engine, site->scope,
              site->child, true));
        }
        const loom_pipeline_control_function_t* child =
            &functions[site->child->ordinal * execution->engine_count + engine];
        if (loom_func_like_isa(child->function)) {
          IREE_RETURN_IF_ERROR(loom_pipeline_control_call_child(
              rewriter, site, child, state_arguments));
        }
        if (options->emit.boundary) {
          IREE_RETURN_IF_ERROR(options->emit.boundary(
              options->emit.user_data, &rewriter->builder, engine, site->scope,
              site->child, false));
        }
        IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
        break;
      }
      case LOOM_PIPELINE_CONTROL_SERIAL:
        if (site->deferred) {
          if (projection == LOOM_PIPELINE_CONTROL_COMPLETE) {
            IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
          } else {
            IREE_RETURN_IF_ERROR(
                loom_pipeline_control_flatten_scope(rewriter, sites, site));
          }
          break;
        }
        if (projection != LOOM_PIPELINE_CONTROL_COMPLETE) {
          IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
          break;
        }
        if (site->enclosing_scope && options->emit.boundary &&
            loom_pipeline_execution_schedule(execution, site->enclosing_scope)
                    ->workers[engine] != IREE_HOST_SIZE_MAX) {
          IREE_RETURN_IF_ERROR(options->emit.boundary(
              options->emit.user_data, &rewriter->builder, engine,
              site->enclosing_scope, NULL, true));
          loom_builder_set_before(&rewriter->builder, site->target->next_op);
          IREE_RETURN_IF_ERROR(options->emit.boundary(
              options->emit.user_data, &rewriter->builder, engine,
              site->enclosing_scope, NULL, false));
        }
        if (loom_pipeline_execution_schedule(execution, site->scope)
                ->workers[engine] == IREE_HOST_SIZE_MAX) {
          IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_pipeline_control_flatten_scope(rewriter, sites, site));
        }
        break;
      case LOOM_PIPELINE_CONTROL_YIELD:
        break;  // The enclosing serial splice owns its continuation.
      case LOOM_PIPELINE_CONTROL_END: {
        if (projection == LOOM_PIPELINE_CONTROL_COMPLETE &&
            sites->has_deferred) {
          IREE_RETURN_IF_ERROR(loom_pipeline_control_call_projection(
              rewriter, &boundaries[1], arguments));
        }
        loom_op_t* returned = NULL;
        IREE_RETURN_IF_ERROR(loom_func_return_build(
            &rewriter->builder, NULL, 0, site->target->location, &returned));
        IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, site->target));
        break;
      }
    }
  }
  const loom_func_like_t callable = loom_func_like_cast(module, function);
  IREE_RETURN_IF_ERROR(loom_rewriter_seed_function(rewriter, callable));
  loom_op_t* dead = NULL;
  while ((dead = loom_rewriter_pop(rewriter))) {
    bool erased = false;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, dead, &erased));
  }
  uint16_t* retained = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, total_count, sizeof(*retained), (void**)&retained));
  uint16_t retained_count = 0;
  for (iree_host_size_t i = 0; i < total_count; ++i) {
    if (loom_module_value_has_uses(module, arguments[i])) {
      retained[retained_count++] = (uint16_t)i;
    }
  }
  for (uint16_t i = entry->arg_count; i > 0; --i) {
    if (!loom_module_value_has_uses(module, entry->arg_ids[i - 1])) {
      IREE_RETURN_IF_ERROR(loom_block_remove_arg(module, entry, i - 1));
    }
  }
  *out_function =
      (loom_pipeline_control_function_t){.function = callable,
                                         .arguments = retained,
                                         .argument_count = retained_count};
  return options->emit.function
             ? options->emit.function(options->emit.user_data, engine, callable)
             : iree_ok_status();
}

iree_status_t loom_pipeline_control_materialize(
    loom_rewriter_t* rewriter, const loom_pipeline_realization_t* realization,
    const loom_pipeline_execution_t* execution,
    const loom_pipeline_control_options_t* options,
    loom_pipeline_control_function_t** out_functions) {
  *out_functions = NULL;
  if (!execution->engine_count) {
    return iree_ok_status();
  }
  const loom_pipeline_invocations_t* invocations = &realization->invocations;
  loom_pipeline_control_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, invocations->count,
      execution->engine_count * sizeof(*functions), (void**)&functions));
  memset(functions, 0,
         invocations->count * execution->engine_count * sizeof(*functions));
  *out_functions = functions;
  loom_symbol_map_t names = {0};
  loom_module_t* module = rewriter->module;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_string_id_t name = module->symbols.entries[i].name_id;
    if (name != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_symbol_map_insert(&names, rewriter->arena, name,
                                                  (loom_symbol_id_t)i));
    }
  }
  uint32_t next_name = 0;
  for (iree_host_size_t reverse = invocations->count; reverse > 0; --reverse) {
    const loom_pipeline_invocation_t* invocation =
        invocations->entries[reverse - 1];
    loom_pipeline_control_sites_t sites = {0};
    IREE_RETURN_IF_ERROR(loom_pipeline_control_collect(
        invocation, execution, options, rewriter->arena, &sites));
    for (iree_host_size_t engine = 0; engine < execution->engine_count;
         ++engine) {
      if (execution->workers[invocation->ordinal * execution->engine_count +
                             engine] == IREE_HOST_SIZE_MAX) {
        continue;
      }
      loom_pipeline_control_function_t boundaries[2] = {0};
      for (int phase = sites.has_deferred ? 0 : 2; phase < 3; ++phase) {
        loom_string_id_t name = LOOM_STRING_ID_INVALID;
        do {
          char text[48];
          iree_snprintf(text, sizeof(text), "__pipeline_engine_%u",
                        next_name++);
          IREE_RETURN_IF_ERROR(loom_module_intern_string(
              module, iree_make_cstring_view(text), &name));
        } while (loom_symbol_map_find(&names, name) != LOOM_SYMBOL_ID_INVALID);
        loom_symbol_ref_t symbol = {.module_id = 0};
        IREE_RETURN_IF_ERROR(
            loom_module_add_symbol(module, name, &symbol.symbol_id));
        IREE_RETURN_IF_ERROR(loom_symbol_map_insert(&names, rewriter->arena,
                                                    name, symbol.symbol_id));
        IREE_RETURN_IF_ERROR(loom_pipeline_control_project(
            rewriter, realization, execution, options, invocation, engine,
            symbol, &sites, functions,
            phase == 0   ? LOOM_PIPELINE_CONTROL_START
            : phase == 1 ? LOOM_PIPELINE_CONTROL_RETIRE
                         : LOOM_PIPELINE_CONTROL_COMPLETE,
            boundaries,
            phase < 2
                ? &boundaries[phase]
                : &functions[invocation->ordinal * execution->engine_count +
                             engine]));
      }
    }
  }
  return iree_ok_status();
}
