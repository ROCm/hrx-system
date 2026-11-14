// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#include <string.h>

iree_status_t loom_aie2p_execution_plan_copy_resources(
    const loom_aie2p_execution_plan_t* source, iree_arena_allocator_t* arena,
    loom_aie2p_execution_plan_t* target) {
  const iree_host_size_t tile_count =
      (iree_host_size_t)source->routing.family->column_count *
      source->routing.family->row_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*target->tiles), (void**)&target->tiles));
  memcpy(target->tiles, source->tiles, tile_count * sizeof(*target->tiles));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_clone(
      &source->routing,
      iree_max(source->routing.route_count,
               source->routing.prefixes.node_count),
      arena, &target->routing));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->routing.route_count, sizeof(*target->routes),
      (void**)&target->routes));
  for (iree_host_size_t i = 0; i < source->routing.route_count; ++i) {
    target->routes[i] = source->routes[i];
    target->routes[i].edge = &target->routing.routes[i];
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count * 2, sizeof(*target->control_sources),
      (void**)&target->control_sources));
  memcpy(target->control_sources, source->control_sources,
         tile_count * 2 * sizeof(*target->control_sources));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count * 2, sizeof(*target->control_destination_sources),
      (void**)&target->control_destination_sources));
  memcpy(target->control_destination_sources,
         source->control_destination_sources,
         tile_count * 2 * sizeof(*target->control_destination_sources));
  target->switches = NULL;
  loom_aie2p_native_switch_t** tail = &target->switches;
  for (const loom_aie2p_native_switch_t* state = source->switches; state;
       state = state->next) {
    loom_aie2p_native_switch_t* copy = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*copy), (void**)&copy));
    *copy = *state;
    copy->next = NULL;
    for (unsigned i = 0; i < LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT; ++i) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, state->banks[i].word_count, sizeof(uint32_t),
          (void**)&copy->banks[i].words));
      memcpy(copy->banks[i].words, state->banks[i].words,
             state->banks[i].word_count * sizeof(uint32_t));
    }
    *tail = copy;
    tail = &copy->next;
  }
  target->connection_count = source->connection_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->connection_count, sizeof(*target->connections),
      (void**)&target->connections));
  memcpy(target->connections, source->connections,
         source->connection_count * sizeof(*target->connections));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_execution_plan_clone(
    const loom_pipeline_entry_scope_t* source, iree_arena_allocator_t* arena,
    loom_pipeline_entry_scope_t* target) {
  return loom_aie2p_execution_plan_copy_resources(
      (const loom_aie2p_execution_plan_t*)source, arena,
      (loom_aie2p_execution_plan_t*)target);
}

const loom_pipeline_entry_scope_type_t loom_aie2p_execution_plan_type = {
    .size = sizeof(loom_aie2p_execution_plan_t),
    .clone = loom_aie2p_execution_plan_clone,
};
