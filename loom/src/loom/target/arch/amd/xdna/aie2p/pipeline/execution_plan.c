// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#include <string.h>

iree_status_t loom_aie2p_execution_plan_copy_resources(
    const loom_aie2p_execution_plan_t* source,
    iree_host_size_t added_connection_capacity, iree_arena_allocator_t* arena,
    loom_aie2p_execution_plan_t* target) {
  const iree_host_size_t tile_count =
      loom_xdna_array_physical_tile_count(source->routing.family);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*target->tiles), (void**)&target->tiles));
  memcpy(target->tiles, source->tiles, tile_count * sizeof(*target->tiles));
  const loom_xdna_array_family_t* family = source->routing.family;
  const iree_host_size_t route_capacity =
      iree_max(source->routing.route_count,
               source->routing.prefixes.node_count) +
      added_connection_capacity *
          (2 * family->row_count + family->column_count + 2);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_clone(
      &source->routing, route_capacity, arena, &target->routing));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, route_capacity, sizeof(*target->routes), (void**)&target->routes));
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
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count, sizeof(*target->control_paths),
      (void**)&target->control_paths));
  memcpy(target->control_paths, source->control_paths,
         family->column_count * sizeof(*target->control_paths));
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
  loom_aie2p_native_connection_t* groups = NULL;
  loom_aie2p_native_connection_endpoint_t* destinations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->connections.group_count + added_connection_capacity,
      sizeof(*groups), (void**)&groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->connections.destination_count + added_connection_capacity,
      sizeof(*destinations), (void**)&destinations));
  if (source->connections.group_count) {
    memcpy(groups, source->connections.groups,
           source->connections.group_count * sizeof(*groups));
  }
  if (source->connections.destination_count) {
    memcpy(destinations, source->connections.destinations,
           source->connections.destination_count * sizeof(*destinations));
  }
  target->connections = (loom_aie2p_native_connections_t){
      .groups = groups,
      .group_count = source->connections.group_count,
      .destinations = destinations,
      .destination_count = source->connections.destination_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_execution_plan_clone(
    const loom_pipeline_entry_scope_t* source, iree_arena_allocator_t* arena,
    loom_pipeline_entry_scope_t* target) {
  return loom_aie2p_execution_plan_copy_resources(
      (const loom_aie2p_execution_plan_t*)source, 0, arena,
      (loom_aie2p_execution_plan_t*)target);
}

const loom_pipeline_entry_scope_type_t loom_aie2p_execution_plan_type = {
    .size = sizeof(loom_aie2p_execution_plan_t),
    .clone = loom_aie2p_execution_plan_clone,
};
