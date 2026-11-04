// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/route.h"

#include "iree/base/internal/math.h"

typedef enum loom_aie2p_array_route_direction_e {
  LOOM_AIE2P_ARRAY_ROUTE_NORTH = 0,
  LOOM_AIE2P_ARRAY_ROUTE_SOUTH = 1,
  LOOM_AIE2P_ARRAY_ROUTE_WEST = 2,
  LOOM_AIE2P_ARRAY_ROUTE_EAST = 3,
  LOOM_AIE2P_ARRAY_ROUTE_DIRECTION_COUNT = 4,
} loom_aie2p_array_route_direction_t;

// One physical arrival in a canonical source's prefix tree. Separate roots
// and separate prefixes never merge even when they traverse the same tile.
typedef struct loom_aie2p_array_route_node_t {
  // Tile reached by this prefix.
  loom_xdna_tile_coordinate_t coordinate;
  // Switch slave port carrying this source's stream.
  loom_xdna_stream_port_t incoming_port;
  // Channel ordinal within incoming_port.
  uint8_t incoming_channel;
  // Retained next-hop node per direction, or IREE_HOST_SIZE_MAX before use.
  iree_host_size_t children[LOOM_AIE2P_ARRAY_ROUTE_DIRECTION_COUNT];
} loom_aie2p_array_route_node_t;

// One uncommitted hop after the source's already reserved multicast prefix.
typedef struct loom_aie2p_array_route_step_t {
  // Tile reached by crossing this hop's physical link.
  loom_xdna_tile_coordinate_t destination;
  // Source tile's incoming slave range.
  const loom_xdna_stream_port_range_t* input;
  // Source tile's outgoing master range.
  const loom_xdna_stream_port_range_t* output;
  // Destination tile's incoming slave range.
  const loom_xdna_stream_port_range_t* next_input;
  // Occupied channels on the directed physical link.
  uint32_t* occupied;
  // Free link channels that can reach the requested final destination.
  uint32_t feasible;
  // Prefix-tree child selected by the hop.
  loom_aie2p_array_route_direction_t direction;
} loom_aie2p_array_route_step_t;

iree_status_t loom_aie2p_array_route_builder_initialize(
    const loom_xdna_array_family_t* family, iree_host_size_t channel_count,
    iree_host_size_t route_capacity, iree_arena_allocator_t* arena,
    loom_aie2p_array_route_builder_t* out_builder) {
  *out_builder = (loom_aie2p_array_route_builder_t){.family = family};
  if (route_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, route_capacity,
                                                 sizeof(*out_builder->routes),
                                                 (void**)&out_builder->routes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, route_capacity, sizeof(*out_builder->prefixes.nodes),
      (void**)&out_builder->prefixes.nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, channel_count, sizeof(*out_builder->prefixes.roots),
      (void**)&out_builder->prefixes.roots));
  memset(out_builder->prefixes.roots, 0xFF,
         channel_count * sizeof(*out_builder->prefixes.roots));

  const iree_host_size_t tile_count =
      (iree_host_size_t)family->column_count * family->row_count;
  uint32_t* occupied_channels = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count * LOOM_AIE2P_ARRAY_ROUTE_DIRECTION_COUNT,
      sizeof(*occupied_channels), (void**)&occupied_channels));
  memset(occupied_channels, 0,
         tile_count * LOOM_AIE2P_ARRAY_ROUTE_DIRECTION_COUNT *
             sizeof(*occupied_channels));
  out_builder->link_channels.northbound = occupied_channels;
  out_builder->link_channels.southbound = occupied_channels + tile_count;
  out_builder->link_channels.westbound = occupied_channels + tile_count * 2u;
  out_builder->link_channels.eastbound = occupied_channels + tile_count * 3u;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count + family->row_count - 2,
      sizeof(*out_builder->steps), (void**)&out_builder->steps));
  return iree_ok_status();
}

static iree_host_size_t loom_aie2p_array_create_route_node(
    loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate,
    loom_xdna_stream_port_t incoming_port, uint8_t incoming_channel) {
  const iree_host_size_t node_index = builder->prefixes.node_count++;
  builder->prefixes.nodes[node_index] = (loom_aie2p_array_route_node_t){
      .coordinate = coordinate,
      .incoming_port = incoming_port,
      .incoming_channel = incoming_channel,
      .children = {IREE_HOST_SIZE_MAX, IREE_HOST_SIZE_MAX, IREE_HOST_SIZE_MAX,
                   IREE_HOST_SIZE_MAX},
  };
  return node_index;
}

static const loom_xdna_stream_port_range_t* loom_aie2p_array_port_range(
    const loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port) {
  const loom_xdna_tile_facts_t* facts =
      loom_xdna_array_tile_facts(builder->family, coordinate);
  return loom_xdna_array_stream_port_range(builder->family, facts->kind,
                                           direction, port);
}

static void loom_aie2p_array_append_route(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_switch_kind_t switch_kind,
    loom_xdna_stream_port_t source_port, uint8_t source_channel,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel) {
  builder->routes[builder->route_count++] = (loom_aie2p_array_route_plan_t){
      .channel_index = channel_index,
      .coordinate = coordinate,
      .switch_kind = switch_kind,
      .source_port = source_port,
      .source_channel = source_channel,
      .destination_port = destination_port,
      .destination_channel = destination_channel,
  };
}

static uint8_t loom_aie2p_array_dma_stream_channel(
    const loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate, loom_xdna_dma_direction_t direction,
    uint8_t dma_channel) {
  const loom_xdna_tile_facts_t* tile_facts =
      loom_xdna_array_tile_facts(builder->family, coordinate);
  const uint8_t base = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                           ? tile_facts->dma.memory_to_stream_port_base
                           : tile_facts->dma.stream_to_memory_port_base;
  const uint8_t stride = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                             ? tile_facts->dma.memory_to_stream_port_stride
                             : tile_facts->dma.stream_to_memory_port_stride;
  return base + dma_channel * stride;
}

static iree_host_size_t loom_aie2p_array_vertical_link_index(
    const loom_aie2p_array_route_builder_t* builder, uint16_t column,
    uint16_t lower_row) {
  return (iree_host_size_t)lower_row * builder->family->column_count + column;
}

static iree_host_size_t loom_aie2p_array_horizontal_link_index(
    const loom_aie2p_array_route_builder_t* builder, uint16_t lower_column,
    uint16_t row) {
  return (iree_host_size_t)row * builder->family->column_count + lower_column;
}

static loom_aie2p_array_route_direction_t loom_aie2p_array_route_direction(
    loom_xdna_tile_coordinate_t source, loom_xdna_tile_coordinate_t destination,
    uint16_t horizontal_row) {
  uint16_t next_row = destination.row;
  if (source.column != destination.column) {
    if (source.row == horizontal_row) {
      return source.column > destination.column ? LOOM_AIE2P_ARRAY_ROUTE_WEST
                                                : LOOM_AIE2P_ARRAY_ROUTE_EAST;
    }
    next_row = horizontal_row;
  }
  return source.row < next_row ? LOOM_AIE2P_ARRAY_ROUTE_NORTH
                               : LOOM_AIE2P_ARRAY_ROUTE_SOUTH;
}

static loom_aie2p_array_route_step_t loom_aie2p_array_route_step(
    loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t source, loom_xdna_stream_port_t incoming_port,
    loom_aie2p_array_route_direction_t direction) {
  static const loom_xdna_stream_port_t outputs[] = {
      LOOM_XDNA_STREAM_PORT_NORTH, LOOM_XDNA_STREAM_PORT_SOUTH,
      LOOM_XDNA_STREAM_PORT_WEST, LOOM_XDNA_STREAM_PORT_EAST};
  static const loom_xdna_stream_port_t inputs[] = {
      LOOM_XDNA_STREAM_PORT_SOUTH, LOOM_XDNA_STREAM_PORT_NORTH,
      LOOM_XDNA_STREAM_PORT_EAST, LOOM_XDNA_STREAM_PORT_WEST};
  loom_xdna_tile_coordinate_t destination = source;
  uint32_t* occupied;
  if (direction <= LOOM_AIE2P_ARRAY_ROUTE_SOUTH) {
    const bool north = direction == LOOM_AIE2P_ARRAY_ROUTE_NORTH;
    destination.row += north ? 1 : -1;
    const iree_host_size_t index = loom_aie2p_array_vertical_link_index(
        builder, source.column, iree_min(source.row, destination.row));
    occupied = north ? &builder->link_channels.northbound[index]
                     : &builder->link_channels.southbound[index];
  } else {
    const bool west = direction == LOOM_AIE2P_ARRAY_ROUTE_WEST;
    destination.column += west ? -1 : 1;
    const iree_host_size_t index = loom_aie2p_array_horizontal_link_index(
        builder, iree_min(source.column, destination.column), source.row);
    occupied = west ? &builder->link_channels.westbound[index]
                    : &builder->link_channels.eastbound[index];
  }
  const loom_xdna_stream_port_range_t* output = loom_aie2p_array_port_range(
      builder, source, LOOM_XDNA_STREAM_DIRECTION_MASTER, outputs[direction]);
  const loom_xdna_stream_port_range_t* next_input = loom_aie2p_array_port_range(
      builder, destination, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
      inputs[direction]);
  return (loom_aie2p_array_route_step_t){
      .destination = destination,
      .input = loom_aie2p_array_port_range(
          builder, source, LOOM_XDNA_STREAM_DIRECTION_SLAVE, incoming_port),
      .output = output,
      .next_input = next_input,
      .occupied = occupied,
      .feasible =
          (UINT32_MAX >> (32 - iree_min(output->count, next_input->count))) &
          ~*occupied,
      .direction = direction,
  };
}

static uint32_t loom_aie2p_array_route_inputs(
    const loom_aie2p_array_route_builder_t* builder,
    const loom_xdna_stream_port_range_t* input,
    const loom_xdna_stream_port_range_t* output, uint32_t destinations) {
  uint32_t sources = 0;
  for (uint8_t i = 0; i < input->count; ++i) {
    if (loom_xdna_array_stream_destination_channels(builder->family, input, i,
                                                    output) &
        destinations) {
      sources |= UINT32_C(1) << i;
    }
  }
  return sources;
}

static bool loom_aie2p_array_plan_route(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    iree_host_size_t node_index, loom_xdna_tile_coordinate_t destination,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel,
    uint16_t horizontal_row, bool packet) {
  // Existing multicast prefixes retain their physical arrival and lane. Only
  // the unreserved suffix participates in lane selection.
  while (builder->prefixes.nodes[node_index].coordinate.column !=
             destination.column ||
         builder->prefixes.nodes[node_index].coordinate.row !=
             destination.row) {
    const loom_aie2p_array_route_node_t* node =
        &builder->prefixes.nodes[node_index];
    const loom_aie2p_array_route_direction_t direction =
        loom_aie2p_array_route_direction(node->coordinate, destination,
                                         horizontal_row);
    if (node->children[direction] == IREE_HOST_SIZE_MAX) {
      break;
    }
    if (packet) {
      // Packet classes share the physical source tree to preserve stream
      // order. Every class retains its complete path so its filters also
      // admit that class on links already allocated for another branch.
      static const loom_xdna_stream_port_t outputs[] = {
          LOOM_XDNA_STREAM_PORT_NORTH, LOOM_XDNA_STREAM_PORT_SOUTH,
          LOOM_XDNA_STREAM_PORT_WEST, LOOM_XDNA_STREAM_PORT_EAST};
      const loom_aie2p_array_route_node_t* child =
          &builder->prefixes.nodes[node->children[direction]];
      loom_aie2p_array_append_route(
          builder, channel_index, node->coordinate,
          LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, node->incoming_port,
          node->incoming_channel, outputs[direction], child->incoming_channel);
    }
    node_index = node->children[direction];
  }
  const loom_aie2p_array_route_node_t* prefix =
      &builder->prefixes.nodes[node_index];
  loom_xdna_tile_coordinate_t coordinate = prefix->coordinate;
  loom_xdna_stream_port_t incoming_port = prefix->incoming_port;
  iree_host_size_t step_count = 0;
  while (coordinate.column != destination.column ||
         coordinate.row != destination.row) {
    const loom_aie2p_array_route_direction_t direction =
        loom_aie2p_array_route_direction(coordinate, destination,
                                         horizontal_row);
    loom_aie2p_array_route_step_t* step = &builder->steps[step_count++];
    *step = loom_aie2p_array_route_step(builder, coordinate, incoming_port,
                                        direction);
    coordinate = step->destination;
    incoming_port = step->next_input->port;
  }
  const loom_xdna_stream_port_range_t* input = loom_aie2p_array_port_range(
      builder, destination, LOOM_XDNA_STREAM_DIRECTION_SLAVE, incoming_port);
  const loom_xdna_stream_port_range_t* output = loom_aie2p_array_port_range(
      builder, destination, LOOM_XDNA_STREAM_DIRECTION_MASTER,
      destination_port);
  uint32_t feasible = loom_aie2p_array_route_inputs(
      builder, input, output, UINT32_C(1) << destination_channel);
  // Propagate reachability before reserving any link. Memory-tile north/south
  // passthrough, for example, couples the lanes on both sides of that tile.
  for (iree_host_size_t i = step_count; i > 0; --i) {
    loom_aie2p_array_route_step_t* step = &builder->steps[i - 1];
    step->feasible &= feasible;
    feasible = loom_aie2p_array_route_inputs(builder, step->input, step->output,
                                             step->feasible);
  }
  if (!(feasible & (UINT32_C(1) << prefix->incoming_channel))) {
    return false;
  }
  for (iree_host_size_t i = 0; i < step_count; ++i) {
    loom_aie2p_array_route_node_t* node = &builder->prefixes.nodes[node_index];
    const loom_aie2p_array_route_step_t* step = &builder->steps[i];
    const uint32_t channels =
        step->feasible &
        loom_xdna_array_stream_destination_channels(
            builder->family, step->input, node->incoming_channel, step->output);
    const uint8_t channel = iree_math_count_trailing_zeros_u32(channels);
    *step->occupied |= UINT32_C(1) << channel;
    loom_aie2p_array_append_route(builder, channel_index, node->coordinate,
                                  LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH,
                                  node->incoming_port, node->incoming_channel,
                                  step->output->port, channel);
    node->children[step->direction] = loom_aie2p_array_create_route_node(
        builder, step->destination, step->next_input->port, channel);
    node_index = node->children[step->direction];
  }
  const loom_aie2p_array_route_node_t* node =
      &builder->prefixes.nodes[node_index];
  loom_aie2p_array_append_route(builder, channel_index, node->coordinate,
                                LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH,
                                node->incoming_port, node->incoming_channel,
                                destination_port, destination_channel);
  return true;
}

bool loom_aie2p_array_route_ingress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint32_t source_channel_index, loom_xdna_tile_coordinate_t shim_coordinate,
    uint8_t shim_dma_channel, loom_xdna_tile_coordinate_t worker_coordinate,
    uint8_t worker_dma_channel) {
  iree_host_size_t* root = &builder->prefixes.roots[source_channel_index];
  if (*root == IREE_HOST_SIZE_MAX) {
    const uint8_t current_channel = loom_aie2p_array_dma_stream_channel(
        builder, shim_coordinate, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
        shim_dma_channel);
    loom_aie2p_array_append_route(builder, channel_index, shim_coordinate,
                                  LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX,
                                  LOOM_XDNA_STREAM_PORT_DMA, shim_dma_channel,
                                  LOOM_XDNA_STREAM_PORT_NORTH, current_channel);
    *root = loom_aie2p_array_create_route_node(
        builder, shim_coordinate, LOOM_XDNA_STREAM_PORT_SOUTH, current_channel);
  }
  return loom_aie2p_array_plan_route(
      builder, channel_index, *root, worker_coordinate,
      LOOM_XDNA_STREAM_PORT_DMA, worker_dma_channel, shim_coordinate.row,
      false);
}

static iree_host_size_t loom_aie2p_array_route_worker_source(
    loom_aie2p_array_route_builder_t* builder, uint32_t source_channel_index,
    loom_xdna_tile_coordinate_t coordinate, uint8_t dma_channel) {
  iree_host_size_t* root = &builder->prefixes.roots[source_channel_index];
  if (*root == IREE_HOST_SIZE_MAX) {
    const uint8_t stream_channel = loom_aie2p_array_dma_stream_channel(
        builder, coordinate, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
        dma_channel);
    *root = loom_aie2p_array_create_route_node(
        builder, coordinate, LOOM_XDNA_STREAM_PORT_DMA, stream_channel);
  }
  return *root;
}

bool loom_aie2p_array_route_egress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint32_t source_channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel) {
  iree_host_size_t node_index = loom_aie2p_array_route_worker_source(
      builder, source_channel_index, worker_coordinate, worker_dma_channel);
  const uint8_t shim_link_channel = loom_aie2p_array_dma_stream_channel(
      builder, shim_coordinate, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
      shim_dma_channel);
  if (!loom_aie2p_array_plan_route(builder, channel_index, node_index,
                                   shim_coordinate, LOOM_XDNA_STREAM_PORT_SOUTH,
                                   shim_link_channel, shim_coordinate.row,
                                   false)) {
    return false;
  }
  loom_aie2p_array_append_route(builder, channel_index, shim_coordinate,
                                LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX,
                                LOOM_XDNA_STREAM_PORT_NORTH, shim_link_channel,
                                LOOM_XDNA_STREAM_PORT_DMA, shim_dma_channel);
  return true;
}

bool loom_aie2p_array_route_stream(loom_aie2p_array_route_builder_t* builder,
                                   uint32_t channel_index,
                                   uint32_t source_channel_index,
                                   loom_xdna_tile_coordinate_t source,
                                   loom_xdna_stream_port_t source_port,
                                   uint8_t source_channel,
                                   loom_xdna_tile_coordinate_t destination,
                                   loom_xdna_stream_port_t destination_port,
                                   uint8_t destination_channel, bool packet) {
  iree_host_size_t* root = &builder->prefixes.roots[source_channel_index];
  if (*root == IREE_HOST_SIZE_MAX) {
    *root = loom_aie2p_array_create_route_node(builder, source, source_port,
                                               source_channel);
  }
  // SRAM switches have only north/south links. Cross-column traffic turns on
  // the receiving core's row, or on the first compute row for SRAM-to-SRAM
  // copies. These are switch hops, not additional DMA or core execution.
  uint16_t horizontal_row = source.row;
  if (source.column != destination.column &&
      loom_xdna_array_tile_facts(builder->family, source)->kind ==
          LOOM_XDNA_TILE_KIND_MEMORY) {
    horizontal_row =
        loom_xdna_array_tile_facts(builder->family, destination)->kind ==
                LOOM_XDNA_TILE_KIND_MEMORY
            ? loom_xdna_array_tile_kind_facts(builder->family,
                                              LOOM_XDNA_TILE_KIND_COMPUTE)
                  ->first_row
            : destination.row;
  }
  return loom_aie2p_array_plan_route(builder, channel_index, *root, destination,
                                     destination_port, destination_channel,
                                     horizontal_row, packet);
}

bool loom_aie2p_array_route_workers(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint32_t source_channel_index,
    loom_xdna_tile_coordinate_t sender_coordinate, uint8_t sender_dma_channel,
    loom_xdna_tile_coordinate_t receiver_coordinate,
    uint8_t receiver_dma_channel, bool packet) {
  const uint8_t sender_stream_channel = loom_aie2p_array_dma_stream_channel(
      builder, sender_coordinate, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
      sender_dma_channel);
  const uint8_t receiver_stream_channel = loom_aie2p_array_dma_stream_channel(
      builder, receiver_coordinate, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
      receiver_dma_channel);
  return loom_aie2p_array_route_stream(
      builder, channel_index, source_channel_index, sender_coordinate,
      LOOM_XDNA_STREAM_PORT_DMA, sender_stream_channel, receiver_coordinate,
      LOOM_XDNA_STREAM_PORT_DMA, receiver_stream_channel, packet);
}
