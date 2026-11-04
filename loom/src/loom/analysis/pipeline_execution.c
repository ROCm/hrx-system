// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_execution.h"

#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_execution_component_t {
  // Parent component, or this component's own ordinal at a root.
  iree_host_size_t parent;
  // Number of worker occurrences represented by this root.
  iree_host_size_t count;
  // Lowest worker ordinal represented by this root.
  iree_host_size_t representative;
} loom_pipeline_execution_component_t;

static iree_host_size_t loom_pipeline_execution_component_root(
    loom_pipeline_execution_component_t* components, iree_host_size_t worker) {
  while (components[worker].parent != worker) {
    const iree_host_size_t parent = components[worker].parent;
    components[worker].parent = components[parent].parent;
    worker = components[worker].parent;
  }
  return worker;
}

iree_status_t loom_pipeline_execution_coalesce(
    iree_host_size_t worker_count,
    const loom_pipeline_execution_sharing_t* sharing,
    iree_host_size_t sharing_count, iree_arena_allocator_t* arena,
    iree_host_size_t** out_representatives) {
  iree_host_size_t* representatives = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, worker_count, sizeof(*representatives), (void**)&representatives));
  if (!sharing_count) {
    for (iree_host_size_t i = 0; i < worker_count; ++i) {
      representatives[i] = i;
    }
    *out_representatives = representatives;
    return iree_ok_status();
  }
  loom_pipeline_execution_component_t* components = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, worker_count, sizeof(*components), (void**)&components));
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    components[i] = (loom_pipeline_execution_component_t){
        .parent = i, .count = 1, .representative = i};
  }
  for (iree_host_size_t i = 0; i < sharing_count; ++i) {
    iree_host_size_t first = loom_pipeline_execution_component_root(
        components, sharing[i].first_worker);
    iree_host_size_t second = loom_pipeline_execution_component_root(
        components, sharing[i].second_worker);
    if (first == second) {
      continue;
    }
    if (components[first].count < components[second].count) {
      const iree_host_size_t temporary = first;
      first = second;
      second = temporary;
    }
    components[second].parent = first;
    components[first].count += components[second].count;
    components[first].representative = iree_min(
        components[first].representative, components[second].representative);
  }
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    representatives[i] =
        components[loom_pipeline_execution_component_root(components, i)]
            .representative;
  }
  *out_representatives = representatives;
  return iree_ok_status();
}

typedef struct loom_pipeline_execution_scope_t {
  // Published participant row, independent of temporary traversal links.
  loom_pipeline_execution_schedule_t schedule;
  // Mutable representative workers while collecting this scope's children.
  iree_host_size_t* workers;
  // Enclosing scope index, or IREE_HOST_SIZE_MAX for an independent scope.
  iree_host_size_t parent;
} loom_pipeline_execution_scope_t;

static bool loom_pipeline_execution_schedule_less(
    const loom_pipeline_execution_schedule_t* left,
    const loom_pipeline_execution_schedule_t* right) {
  return (uintptr_t)left->source < (uintptr_t)right->source;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_execution_sort_schedules,
                          loom_pipeline_execution_schedule_t,
                          loom_pipeline_execution_schedule_less)

// A serial union retains one representative for later conflict reporting;
// a concurrent union rejects every duplicate, including two copies of one
// authored definition. Source symbol equality grants no execution sharing.
static bool loom_pipeline_execution_merge(
    iree_host_size_t* destination, const iree_host_size_t* source,
    iree_host_size_t count, bool concurrent,
    loom_pipeline_execution_conflict_t* conflict) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (source[i] == IREE_HOST_SIZE_MAX) {
      continue;
    }
    if (concurrent && destination[i] != IREE_HOST_SIZE_MAX) {
      *conflict = (loom_pipeline_execution_conflict_t){
          .first_worker = destination[i], .second_worker = source[i]};
      return false;
    }
    if (destination[i] == IREE_HOST_SIZE_MAX) {
      destination[i] = source[i];
    }
  }
  return true;
}

static iree_status_t loom_pipeline_execution_append_scope(
    const loom_pipeline_invocation_schedule_t* source, iree_host_size_t parent,
    iree_host_size_t engine_count, iree_arena_allocator_t* arena,
    loom_pipeline_execution_scope_t** scopes, iree_host_size_t* count,
    iree_host_size_t* capacity) {
  if (*count == *capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, *count, *count + 1, sizeof(**scopes), capacity, (void**)scopes));
  }
  iree_host_size_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, engine_count, sizeof(*workers), (void**)&workers));
  for (iree_host_size_t i = 0; i < engine_count; ++i) {
    workers[i] = IREE_HOST_SIZE_MAX;
  }
  (*scopes)[(*count)++] = (loom_pipeline_execution_scope_t){
      .schedule = {.source = source, .workers = workers},
      .workers = workers,
      .parent = parent};
  return iree_ok_status();
}

iree_status_t loom_pipeline_execution_build(
    const loom_pipeline_invocations_t* invocations,
    const iree_host_size_t* worker_engines, iree_host_size_t engine_count,
    iree_arena_allocator_t* arena, loom_pipeline_execution_t* out_execution,
    loom_pipeline_execution_conflict_t* out_conflict) {
  *out_execution = (loom_pipeline_execution_t){.engine_count = engine_count};
  *out_conflict = (loom_pipeline_execution_conflict_t){
      .first_worker = IREE_HOST_SIZE_MAX, .second_worker = IREE_HOST_SIZE_MAX};
  iree_host_size_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, invocations->count, engine_count * sizeof(*workers),
      (void**)&workers));
  out_execution->workers = workers;
  for (iree_host_size_t i = 0; i < invocations->count * engine_count; ++i) {
    workers[i] = IREE_HOST_SIZE_MAX;
  }
  loom_pipeline_execution_scope_t* scopes = NULL;
  iree_host_size_t scope_count = 0, scope_capacity = 0;
  for (iree_host_size_t reverse = invocations->count; reverse > 0; --reverse) {
    const loom_pipeline_invocation_t* invocation =
        invocations->entries[reverse - 1];
    const loom_pipeline_resources_t* resources = invocation->resources;
    iree_host_size_t* row =
        engine_count ? workers + invocation->ordinal * engine_count : NULL;
    for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
      const iree_host_size_t worker = invocation->first_worker + i;
      const iree_host_size_t engine = worker_engines[worker];
      if (engine == IREE_HOST_SIZE_MAX) {
        continue;
      }
      if (row[engine] != IREE_HOST_SIZE_MAX) {
        *out_conflict = (loom_pipeline_execution_conflict_t){
            .first_worker = row[engine], .second_worker = worker};
        return iree_ok_status();
      }
      row[engine] = worker;
    }
    for (iree_host_size_t i = 0; i < resources->composition_count; ++i) {
      const iree_host_size_t* child =
          engine_count
              ? workers + invocation->compositions[i]->ordinal * engine_count
              : NULL;
      if (!loom_pipeline_execution_merge(row, child, engine_count, true,
                                         out_conflict)) {
        return iree_ok_status();
      }
    }
    const iree_host_size_t first_scope = scope_count;
    for (iree_host_size_t i = 0; i < resources->schedule_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_pipeline_execution_append_scope(
          &invocation->schedules[i], IREE_HOST_SIZE_MAX, engine_count, arena,
          &scopes, &scope_count, &scope_capacity));
    }
    // Discover nested scopes in retained control order. Child occurrence rows
    // already exist because the construction inventory is parent-before-child.
    for (iree_host_size_t i = first_scope; i < scope_count; ++i) {
      const loom_pipeline_invocation_schedule_t* schedule =
          scopes[i].schedule.source;
      iree_host_size_t* participants = scopes[i].workers;
      for (iree_host_size_t b = 0;
           b < schedule->source->control->graph.block_count; ++b) {
        const loom_pipeline_schedule_block_t* block =
            &schedule->source->blocks[b];
        for (iree_host_size_t s = 0; s < block->step_count; ++s) {
          const loom_pipeline_invocation_step_t binding =
              schedule->blocks[b][s];
          switch (block->steps[s].kind) {
            case LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION:
              loom_pipeline_execution_merge(
                  participants,
                  engine_count ? workers + binding.child->ordinal * engine_count
                               : NULL,
                  engine_count, false, out_conflict);
              break;
            case LOOM_PIPELINE_SCHEDULE_STEP_SERIAL: {
              IREE_RETURN_IF_ERROR(loom_pipeline_execution_append_scope(
                  binding.nested, i, engine_count, arena, &scopes, &scope_count,
                  &scope_capacity));
              break;
            }
            default:
              break;
          }
        }
      }
    }
    // Nested scopes are sequential in their parent. Distinct outer scopes
    // progress concurrently with each other and with the invocation's strands.
    for (iree_host_size_t i = scope_count; i > first_scope; --i) {
      const loom_pipeline_execution_scope_t* scope = &scopes[i - 1];
      const bool outer = scope->parent == IREE_HOST_SIZE_MAX;
      iree_host_size_t* destination =
          outer ? row : scopes[scope->parent].workers;
      if (!loom_pipeline_execution_merge(destination, scope->workers,
                                         engine_count, outer, out_conflict)) {
        return iree_ok_status();
      }
    }
  }
  loom_pipeline_execution_schedule_t* schedules = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, scope_count, sizeof(*schedules), (void**)&schedules));
  for (iree_host_size_t i = 0; i < scope_count; ++i) {
    schedules[i] = scopes[i].schedule;
  }
  loom_pipeline_execution_sort_schedules(schedules, scope_count);
  out_execution->schedules = schedules;
  out_execution->schedule_count = scope_count;
  return iree_ok_status();
}

const loom_pipeline_execution_schedule_t* loom_pipeline_execution_schedule(
    const loom_pipeline_execution_t* execution,
    const loom_pipeline_invocation_schedule_t* schedule) {
  iree_host_size_t begin = 0, end = execution->schedule_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_execution_schedule_t* candidate =
        &execution->schedules[middle];
    if ((uintptr_t)candidate->source < (uintptr_t)schedule) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  IREE_ASSERT(begin < execution->schedule_count &&
                  execution->schedules[begin].source == schedule,
              "execution scope belongs to the admitted inventory");
  return &execution->schedules[begin];
}
