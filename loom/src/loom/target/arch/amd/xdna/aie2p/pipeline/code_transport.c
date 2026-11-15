// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/code_transport.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"

typedef struct loom_aie2p_code_transport_sender_t {
  // Source-owned prefix shared by the sender's packet-discriminated routes.
  loom_aie2p_array_route_tree_t tree;
  // SRAM memory-to-stream channel; meaningful when worker_count is nonzero.
  uint16_t engine;
  // Loaders sharing the queue, each with at most one incomplete request.
  uint8_t worker_count;
  // Private hardware-command suffix available to this column's load requests.
  loom_aie2p_array_route_sharing_t commands;
  // Identities excluded by the carrier source and previous load requests.
  uint32_t command_packets;
} loom_aie2p_code_transport_sender_t;

iree_status_t loom_aie2p_code_transport_initialize(
    const loom_aie2p_execution_plan_t* source, iree_host_size_t worker_capacity,
    iree_arena_allocator_t* arena, loom_aie2p_code_transport_t* out_transport) {
  *out_transport = (loom_aie2p_code_transport_t){0};
  loom_aie2p_execution_plan_t* plan = &out_transport->plan;
  plan->base = source->base;
  IREE_RETURN_IF_ERROR(loom_aie2p_execution_plan_copy_resources(
      source, 3 * worker_capacity, arena, plan));
  out_transport->connections =
      (loom_aie2p_native_connection_t*)plan->connections.groups;
  out_transport->destinations =
      (loom_aie2p_native_connection_endpoint_t*)plan->connections.destinations;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->routing.family->column_count,
      sizeof(*out_transport->senders), (void**)&out_transport->senders));
  memset(
      out_transport->senders, 0,
      source->routing.family->column_count * sizeof(*out_transport->senders));
  const uint16_t memory_row =
      loom_xdna_array_tile_kind_facts(source->routing.family,
                                      LOOM_XDNA_TILE_KIND_MEMORY)
          ->first_row;
  for (uint16_t column = 0; column < source->routing.family->column_count;
       ++column) {
    const loom_aie2p_native_control_path_t* path = &plan->control_paths[column];
    if (!path->count) {
      continue;
    }
    loom_aie2p_code_transport_sender_t* sender =
        &out_transport->senders[column];
    IREE_RETURN_IF_ERROR(loom_aie2p_array_route_sharing_initialize(
        &plan->routing, path->first, path->count,
        (loom_xdna_tile_coordinate_t){column, memory_row},
        LOOM_AIE2P_ARRAY_ROUTE_CARRIER_HARDWARE_COMMAND, &sender->commands));
    sender->command_packets = plan->control_sources[path->source].packet_mask;
  }
  return iree_ok_status();
}

static uint32_t loom_aie2p_code_transport_endpoint(
    const loom_xdna_array_family_t* family,
    loom_aie2p_native_connection_endpoint_t endpoint) {
  return loom_xdna_array_physical_tile_index(family, endpoint.coordinate) * 2 +
         (endpoint.port == LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
}

static bool loom_aie2p_code_transport_route(
    loom_aie2p_code_transport_t* transport,
    loom_aie2p_native_connection_endpoint_t source,
    loom_aie2p_native_connection_endpoint_t destination, uint8_t packet,
    loom_aie2p_array_route_tree_t* dma_tree,
    loom_aie2p_array_route_sharing_t* commands) {
  loom_aie2p_execution_plan_t* plan = &transport->plan;
  const loom_xdna_array_family_t* family = plan->routing.family;
  const uint32_t destination_index =
      loom_aie2p_code_transport_endpoint(family, destination);
  uint32_t source_index = UINT32_MAX;
  loom_aie2p_array_route_tree_t* tree = dma_tree;
  uint32_t* owner = &plan->control_destination_sources[destination_index];
  if (source.port == LOOM_XDNA_STREAM_PORT_DMA) {
    *owner = UINT32_MAX - 1;
  } else {
    source_index = loom_aie2p_code_transport_endpoint(family, source);
    loom_aie2p_native_control_source_t* control =
        &plan->control_sources[source_index];
    tree = &control->tree;
    if (!tree->root_index) {
      control->destination = destination_index;
    } else if (control->destination != destination_index) {
      control->destination = UINT32_MAX;
    }
    if (*owner == UINT32_MAX) {
      *owner = source_index;
    } else if (*owner != source_index) {
      *owner = UINT32_MAX - 1;
    }
    control->packet_mask |= UINT32_C(1) << packet;
  }
  const iree_host_size_t first = plan->routing.route_count;
  const bool routed =
      commands ? loom_aie2p_array_route_control(
                     &plan->routing, (uint32_t)plan->connections.group_count,
                     tree, source.coordinate, destination.coordinate, commands)
               : loom_aie2p_array_route_stream(
                     &plan->routing, (uint32_t)plan->connections.group_count,
                     tree, source.coordinate, source.port, source.channel,
                     destination.coordinate, destination.port,
                     destination.channel, true);
  if (!routed) {
    return false;
  }
  if (commands && commands->used) {
    plan->control_sources[source_index].destination = UINT32_MAX;
    plan->control_sources[plan->control_paths[source.coordinate.column].source]
        .destination = UINT32_MAX;
  }
  for (iree_host_size_t i = first; i < plan->routing.route_count; ++i) {
    plan->routes[i] =
        (loom_aie2p_native_route_t){.edge = &plan->routing.routes[i],
                                    .packet = packet,
                                    .control_source = source_index};
  }
  transport->connections[plan->connections.group_count++] =
      (loom_aie2p_native_connection_t){
          .source = source,
          .packet = packet,
          .retain_header = true,
          .destination_offset = plan->connections.destination_count,
          .destination_count = 1};
  transport->destinations[plan->connections.destination_count++] = destination;
  return true;
}

void loom_aie2p_code_transport_bind(
    loom_aie2p_code_transport_t* transport, loom_xdna_tile_coordinate_t compute,
    iree_host_size_t descriptor_count,
    loom_aie2p_code_transport_binding_t* out_binding,
    iree_string_view_t* out_requirement) {
  *out_binding = (loom_aie2p_code_transport_binding_t){0};
  *out_requirement = iree_string_view_empty();
  loom_aie2p_execution_plan_t* plan = &transport->plan;
  const loom_xdna_array_family_t* family = plan->routing.family;
  const loom_xdna_tile_facts_t* facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_MEMORY);
  const loom_xdna_tile_coordinate_t memory = {compute.column, facts->first_row};
  loom_aie2p_execution_tile_t* occupancy =
      &plan->tiles[loom_xdna_array_physical_tile_index(family, memory)];
  const uint16_t bank_capacity = facts->dma.buffer_descriptor_count / 2;
  loom_aie2p_code_transport_sender_t* sender =
      &transport->senders[compute.column];
  if (!sender->worker_count ||
      sender->worker_count == facts->dma.task_queue_depth ||
      sender->worker_count == 32 ||
      descriptor_count >
          bank_capacity - occupancy->descriptor_counts[sender->engine & 1]) {
    if (occupancy->mm2s_count == facts->dma.channel_count_per_direction) {
      *out_requirement =
          IREE_SV("a free SRAM memory-to-stream engine for code delivery");
      return;
    }
    sender->tree = (loom_aie2p_array_route_tree_t){0};
    sender->engine = occupancy->mm2s_count++;
    sender->worker_count = 0;
  }
  const uint16_t engine = sender->engine;
  const uint16_t bank = engine & 1;
  if (descriptor_count > bank_capacity - occupancy->descriptor_counts[bank]) {
    *out_requirement =
        IREE_SV("SRAM descriptors for the complete native code catalog");
    return;
  }
  const uint16_t first_descriptor =
      bank * bank_capacity + occupancy->descriptor_counts[bank];
  occupancy->descriptor_counts[bank] += (uint16_t)descriptor_count;
  const uint32_t core_index =
      loom_xdna_array_physical_tile_index(family, compute) * 2;
  const uint32_t queue_available = ~(
      plan->control_sources[core_index].packet_mask | sender->command_packets);
  const uint32_t reply_available =
      ~plan->control_sources[core_index + 1].packet_mask;
  if (!queue_available || !reply_available) {
    *out_requirement = IREE_SV(
        "unused control packet identities for code loading and completion");
    return;
  }
  const uint8_t queue_packet =
      (uint8_t)iree_math_count_trailing_zeros_u32(queue_available);
  const uint8_t reply_packet =
      (uint8_t)iree_math_count_trailing_zeros_u32(reply_available);
  const uint8_t code_packet = sender->worker_count++;
  if (sender->commands.domain) {
    sender->command_packets |= UINT32_C(1) << queue_packet;
  }
  const loom_aie2p_native_connection_endpoint_t core = {
      compute, LOOM_XDNA_STREAM_PORT_CORE, 0};
  const loom_aie2p_native_connection_endpoint_t control = {
      compute, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  const loom_aie2p_native_connection_endpoint_t queue = {
      memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0};
  const loom_aie2p_native_connection_endpoint_t dma = {
      memory, LOOM_XDNA_STREAM_PORT_DMA, (uint8_t)engine};
  if (!loom_aie2p_code_transport_route(
          transport, core, queue, queue_packet, NULL,
          sender->commands.domain ? &sender->commands : NULL) ||
      !loom_aie2p_code_transport_route(transport, dma, control, code_packet,
                                       &sender->tree, NULL) ||
      !loom_aie2p_code_transport_route(transport, control, core, reply_packet,
                                       NULL, NULL)) {
    *out_requirement =
        IREE_SV("independent queue, code-delivery, and completion routes");
    return;
  }
  *out_binding = (loom_aie2p_code_transport_binding_t){
      .compute = compute,
      .memory = memory,
      .engine = engine,
      .first_descriptor = first_descriptor,
      .descriptor_count = (uint16_t)descriptor_count,
      .queue_packet = queue_packet,
      .code_packet = code_packet,
      .reply_packet = reply_packet};
}

iree_status_t loom_aie2p_code_transport_finish(
    loom_aie2p_code_transport_t* transport, iree_arena_allocator_t* arena,
    iree_string_view_t* out_requirement) {
  loom_aie2p_execution_plan_t* plan = &transport->plan;
  const loom_aie2p_native_switch_input_t input = {
      .routing = &plan->routing,
      .routes = plan->routes,
      .control_sources = plan->control_sources,
      .control_destination_sources = plan->control_destination_sources};
  IREE_RETURN_IF_ERROR(loom_aie2p_native_switches_build(
      &input, arena, &plan->switches, out_requirement));
  if (iree_string_view_is_empty(*out_requirement)) {
    return loom_aie2p_native_connections_verify(&plan->routing, plan->switches,
                                                &plan->connections, arena);
  }
  return iree_ok_status();
}
