// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/route.h"

static iree_status_t loom_aie2p_array_allocate_link_channel(
    uint8_t* next_channel, uint8_t capacity, uint8_t* out_channel) {
  if (*next_channel < capacity) {
    *out_channel = (*next_channel)++;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "AIE2P stream link channels are exhausted");
}

static iree_status_t loom_aie2p_array_port_capacity(
    const loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port,
    uint8_t* out_capacity) {
  const loom_xdna_tile_facts_t* tile_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_array_tile_facts(builder->family, coordinate, &tile_facts));
  const loom_xdna_stream_port_range_t* range = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_array_stream_port_range(
      builder->family, tile_facts->kind, direction, port, &range));
  *out_capacity = range->count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_allocate_physical_link(
    loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t source_coordinate,
    loom_xdna_stream_port_t source_port,
    loom_xdna_tile_coordinate_t destination_coordinate,
    loom_xdna_stream_port_t destination_port, uint8_t* next_channel,
    uint8_t* out_channel) {
  uint8_t source_capacity = 0;
  uint8_t destination_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
      builder, source_coordinate, LOOM_XDNA_STREAM_DIRECTION_MASTER,
      source_port, &source_capacity));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
      builder, destination_coordinate, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
      destination_port, &destination_capacity));
  const uint8_t capacity = source_capacity < destination_capacity
                               ? source_capacity
                               : destination_capacity;
  return loom_aie2p_array_allocate_link_channel(next_channel, capacity,
                                                out_channel);
}

static iree_status_t loom_aie2p_array_append_route(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_switch_kind_t switch_kind,
    loom_xdna_stream_port_t source_port, uint8_t source_channel,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel) {
  if (switch_kind == LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH) {
    uint8_t source_capacity = 0;
    uint8_t destination_capacity = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
        builder, coordinate, LOOM_XDNA_STREAM_DIRECTION_SLAVE, source_port,
        &source_capacity));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_port_capacity(
        builder, coordinate, LOOM_XDNA_STREAM_DIRECTION_MASTER,
        destination_port, &destination_capacity));
    if (source_channel >= source_capacity ||
        destination_channel >= destination_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AIE2P route exceeds a stream port range");
    }
  }
  builder->routes[builder->route_count++] = (loom_aie2p_array_route_plan_t){
      .channel_index = channel_index,
      .coordinate = coordinate,
      .switch_kind = switch_kind,
      .source_port = source_port,
      .source_channel = source_channel,
      .destination_port = destination_port,
      .destination_channel = destination_channel,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_dma_stream_channel(
    const loom_aie2p_array_route_builder_t* builder,
    loom_xdna_tile_coordinate_t coordinate,
    loom_aie2p_array_dma_direction_t direction, uint8_t dma_channel,
    uint8_t* out_stream_channel) {
  const loom_xdna_tile_facts_t* tile_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_array_tile_facts(builder->family, coordinate, &tile_facts));
  if (dma_channel >= tile_facts->dma.channel_count_per_direction) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P DMA channel exceeds the tile engine");
  }
  const uint8_t base =
      direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
          ? tile_facts->dma.memory_to_stream_port_base
          : tile_facts->dma.stream_to_memory_port_base;
  const uint8_t stride =
      direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
          ? tile_facts->dma.memory_to_stream_port_stride
          : tile_facts->dma.stream_to_memory_port_stride;
  *out_stream_channel = base + dma_channel * stride;
  return iree_ok_status();
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

static iree_status_t loom_aie2p_array_plan_horizontal_route_segment(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint16_t target_column, loom_xdna_tile_coordinate_t* inout_coordinate,
    loom_xdna_stream_port_t* inout_incoming_port,
    uint8_t* inout_stream_channel) {
  while (inout_coordinate->column != target_column) {
    const bool move_west = inout_coordinate->column > target_column;
    const uint16_t next_column =
        move_west ? (uint16_t)(inout_coordinate->column - 1u)
                  : (uint16_t)(inout_coordinate->column + 1u);
    const uint16_t lower_column = inout_coordinate->column < next_column
                                      ? inout_coordinate->column
                                      : next_column;
    const iree_host_size_t link_index = loom_aie2p_array_horizontal_link_index(
        builder, lower_column, inout_coordinate->row);
    uint8_t* next_channel = move_west
                                ? &builder->link_channels.westbound[link_index]
                                : &builder->link_channels.eastbound[link_index];
    const loom_xdna_stream_port_t outgoing_port =
        move_west ? LOOM_XDNA_STREAM_PORT_WEST : LOOM_XDNA_STREAM_PORT_EAST;
    const loom_xdna_stream_port_t next_incoming_port =
        move_west ? LOOM_XDNA_STREAM_PORT_EAST : LOOM_XDNA_STREAM_PORT_WEST;
    const loom_xdna_tile_coordinate_t next_coordinate = {
        next_column,
        inout_coordinate->row,
    };
    uint8_t link_channel = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_link(
        builder, *inout_coordinate, outgoing_port, next_coordinate,
        next_incoming_port, next_channel, &link_channel));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
        builder, channel_index, *inout_coordinate,
        LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, *inout_incoming_port,
        *inout_stream_channel, outgoing_port, link_channel));
    *inout_coordinate = next_coordinate;
    *inout_incoming_port = next_incoming_port;
    *inout_stream_channel = link_channel;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_vertical_route_segment(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint16_t target_row, loom_xdna_tile_coordinate_t* inout_coordinate,
    loom_xdna_stream_port_t* inout_incoming_port,
    uint8_t* inout_stream_channel) {
  while (inout_coordinate->row != target_row) {
    const bool move_north = inout_coordinate->row < target_row;
    const uint16_t next_row = move_north
                                  ? (uint16_t)(inout_coordinate->row + 1u)
                                  : (uint16_t)(inout_coordinate->row - 1u);
    const uint16_t lower_row =
        inout_coordinate->row < next_row ? inout_coordinate->row : next_row;
    const iree_host_size_t link_index = loom_aie2p_array_vertical_link_index(
        builder, inout_coordinate->column, lower_row);
    uint8_t* next_channel =
        move_north ? &builder->link_channels.northbound[link_index]
                   : &builder->link_channels.southbound[link_index];
    const loom_xdna_stream_port_t outgoing_port =
        move_north ? LOOM_XDNA_STREAM_PORT_NORTH : LOOM_XDNA_STREAM_PORT_SOUTH;
    const loom_xdna_stream_port_t next_incoming_port =
        move_north ? LOOM_XDNA_STREAM_PORT_SOUTH : LOOM_XDNA_STREAM_PORT_NORTH;
    const loom_xdna_tile_coordinate_t next_coordinate = {
        inout_coordinate->column,
        next_row,
    };
    uint8_t link_channel = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_allocate_physical_link(
        builder, *inout_coordinate, outgoing_port, next_coordinate,
        next_incoming_port, next_channel, &link_channel));
    IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
        builder, channel_index, *inout_coordinate,
        LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, *inout_incoming_port,
        *inout_stream_channel, outgoing_port, link_channel));
    *inout_coordinate = next_coordinate;
    *inout_incoming_port = next_incoming_port;
    *inout_stream_channel = link_channel;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_route_destination(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t coordinate,
    loom_xdna_stream_port_t incoming_port, uint8_t incoming_channel,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel) {
  return loom_aie2p_array_append_route(
      builder, channel_index, coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
      incoming_channel, destination_port, destination_channel);
}

iree_status_t loom_aie2p_array_route_ingress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel) {
  uint8_t current_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_dma_stream_channel(
      builder, shim_coordinate, LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM,
      shim_dma_channel, &current_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
      builder, channel_index, shim_coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX, LOOM_XDNA_STREAM_PORT_DMA,
      shim_dma_channel, LOOM_XDNA_STREAM_PORT_NORTH, current_channel));

  loom_xdna_tile_coordinate_t coordinate = shim_coordinate;
  loom_xdna_stream_port_t incoming_port = LOOM_XDNA_STREAM_PORT_SOUTH;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_horizontal_route_segment(
      builder, channel_index, worker_coordinate.column, &coordinate,
      &incoming_port, &current_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_vertical_route_segment(
      builder, channel_index, worker_coordinate.row, &coordinate,
      &incoming_port, &current_channel));
  return loom_aie2p_array_plan_route_destination(
      builder, channel_index, coordinate, incoming_port, current_channel,
      LOOM_XDNA_STREAM_PORT_DMA, worker_dma_channel);
}

iree_status_t loom_aie2p_array_route_egress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel) {
  loom_xdna_tile_coordinate_t coordinate = worker_coordinate;
  uint8_t current_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_dma_stream_channel(
      builder, worker_coordinate,
      LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM, worker_dma_channel,
      &current_channel));
  loom_xdna_stream_port_t incoming_port = LOOM_XDNA_STREAM_PORT_DMA;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_vertical_route_segment(
      builder, channel_index, shim_coordinate.row, &coordinate, &incoming_port,
      &current_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_horizontal_route_segment(
      builder, channel_index, shim_coordinate.column, &coordinate,
      &incoming_port, &current_channel));

  uint8_t shim_link_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_dma_stream_channel(
      builder, shim_coordinate, LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY,
      shim_dma_channel, &shim_link_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
      builder, channel_index, coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH, incoming_port,
      current_channel, LOOM_XDNA_STREAM_PORT_SOUTH, shim_link_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_append_route(
      builder, channel_index, shim_coordinate,
      LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX, LOOM_XDNA_STREAM_PORT_NORTH,
      shim_link_channel, LOOM_XDNA_STREAM_PORT_DMA, shim_dma_channel));
  return iree_ok_status();
}

iree_status_t loom_aie2p_array_route_workers(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t sender_coordinate, uint8_t sender_dma_channel,
    loom_xdna_tile_coordinate_t receiver_coordinate,
    uint8_t receiver_dma_channel) {
  loom_xdna_tile_coordinate_t coordinate = sender_coordinate;
  loom_xdna_stream_port_t incoming_port = LOOM_XDNA_STREAM_PORT_DMA;
  uint8_t current_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_dma_stream_channel(
      builder, sender_coordinate,
      LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM, sender_dma_channel,
      &current_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_horizontal_route_segment(
      builder, channel_index, receiver_coordinate.column, &coordinate,
      &incoming_port, &current_channel));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_vertical_route_segment(
      builder, channel_index, receiver_coordinate.row, &coordinate,
      &incoming_port, &current_channel));
  uint8_t receiver_stream_channel = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_dma_stream_channel(
      builder, receiver_coordinate,
      LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY, receiver_dma_channel,
      &receiver_stream_channel));
  return loom_aie2p_array_plan_route_destination(
      builder, channel_index, coordinate, incoming_port, current_channel,
      LOOM_XDNA_STREAM_PORT_DMA, receiver_stream_channel);
}
