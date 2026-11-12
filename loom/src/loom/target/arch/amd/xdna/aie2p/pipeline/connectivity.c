// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"

#include "iree/base/internal/math.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

// NPU2 switch ordinals fit in five bits, ports have at most eight channels,
// and packet filtering can either retain or remove the one source header.
enum {
  LOOM_AIE2P_CONNECTIVITY_CHANNEL_COUNT = 8,
  LOOM_AIE2P_CONNECTIVITY_ARRIVALS_PER_TILE = 64,
};

typedef struct loom_aie2p_native_connection_arrival_t {
  // Switch input reached by the selected hardware configuration.
  loom_aie2p_native_connection_endpoint_t endpoint;
  // The source packet header is still present at this input.
  bool framed;
} loom_aie2p_native_connection_arrival_t;

typedef struct loom_aie2p_native_connectivity_tile_t {
  // Completed switch bank values, or NULL for an untouched tile.
  const loom_aie2p_native_switch_t* state;
  // Physical tile role used for ordinal interpretation.
  loom_xdna_tile_kind_t kind;
  // Selected switch-input lane masks per shim DMA sender.
  uint8_t ingress[LOOM_AIE2P_CONNECTIVITY_CHANNEL_COUNT];
  // Selected DMA receiver masks per shim switch-output lane.
  uint8_t egress[LOOM_AIE2P_CONNECTIVITY_CHANNEL_COUNT];
  // Arrivals already encountered by the current source trace.
  uint64_t visited;
  // Intended endpoints not yet delivered, indexed by port * 8 + channel.
  uint64_t remaining[2];
} loom_aie2p_native_connectivity_tile_t;

struct loom_aie2p_native_connectivity_t {
  // Immutable selected array geometry and port facts.
  const loom_xdna_array_family_t* family;
  // Column-major physical network view and per-query scratch.
  loom_aie2p_native_connectivity_tile_t* tiles;
  // Number of physical tiles in that view.
  iree_host_size_t tile_count;
  // Inverse switch ordinal maps, indexed by tile kind and direction minus one.
  loom_aie2p_native_connection_endpoint_t ports[3][2][32];
  // First ordinal of each architectural port in the same domains.
  uint8_t ordinals[3][2][10];
  // Bounded worklist: each switch input and framing state enters once.
  loom_aie2p_native_connection_arrival_t* arrivals;
  // Number of populated worklist entries for this source.
  iree_host_size_t arrival_count;
  // First connection failure in this source's traversal.
  loom_aie2p_native_connectivity_result_t result;
};

static loom_aie2p_native_connectivity_tile_t* loom_aie2p_connectivity_tile(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_xdna_tile_coordinate_t coordinate) {
  return &connectivity
              ->tiles[coordinate.column * connectivity->family->row_count +
                      coordinate.row];
}

iree_status_t loom_aie2p_native_connectivity_allocate(
    const loom_xdna_array_family_t* family,
    const loom_aie2p_native_switch_t* switches,
    const loom_aie2p_array_route_plan_t* routes, iree_host_size_t route_count,
    iree_arena_allocator_t* arena,
    loom_aie2p_native_connectivity_t** out_connectivity) {
  loom_aie2p_native_connectivity_t* connectivity;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*connectivity), (void**)&connectivity));
  *connectivity = (loom_aie2p_native_connectivity_t){
      .family = family, .tile_count = family->column_count * family->row_count};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, connectivity->tile_count, sizeof(*connectivity->tiles),
      (void**)&connectivity->tiles));
  memset(connectivity->tiles, 0,
         connectivity->tile_count * sizeof(*connectivity->tiles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena,
      connectivity->tile_count * LOOM_AIE2P_CONNECTIVITY_ARRIVALS_PER_TILE,
      sizeof(*connectivity->arrivals), (void**)&connectivity->arrivals));
  for (uint16_t column = 0; column < family->column_count; ++column) {
    for (uint16_t row = 0; row < family->row_count; ++row) {
      const loom_xdna_tile_coordinate_t coordinate = {column, row};
      loom_aie2p_connectivity_tile(connectivity, coordinate)->kind =
          loom_xdna_array_tile_facts(family, coordinate)->kind;
    }
  }
  for (const loom_aie2p_native_switch_t* state = switches; state;
       state = state->next) {
    loom_aie2p_connectivity_tile(connectivity, state->coordinate)->state =
        state;
  }
  for (uint16_t i = 0; i < family->stream_port_count; ++i) {
    const loom_xdna_stream_port_range_t* range = &family->stream_ports[i];
    connectivity
        ->ordinals[range->tile_kind - 1][range->direction - 1][range->port] =
        range->ordinal;
    for (uint8_t channel = 0; channel < range->count; ++channel) {
      connectivity->ports[range->tile_kind - 1][range->direction - 1]
                         [range->ordinal + channel] =
          (loom_aie2p_native_connection_endpoint_t){.port = range->port,
                                                    .channel = channel};
    }
  }
  // These selected mux edges are the configuration emitter's input. Stream
  // edges are deliberately ignored: their realized banks are the oracle here.
  for (iree_host_size_t i = 0; i < route_count; ++i) {
    const loom_aie2p_array_route_plan_t* edge = &routes[i];
    if (edge->switch_kind != LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX) {
      continue;
    }
    loom_aie2p_native_connectivity_tile_t* tile =
        loom_aie2p_connectivity_tile(connectivity, edge->coordinate);
    if (edge->source_port == LOOM_XDNA_STREAM_PORT_DMA) {
      tile->ingress[edge->source_channel] |= 1u << edge->destination_channel;
    } else {
      tile->egress[edge->source_channel] |= 1u << edge->destination_channel;
    }
  }
  *out_connectivity = connectivity;
  return iree_ok_status();
}

static void loom_aie2p_connectivity_fail(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connectivity_failure_t failure,
    loom_aie2p_native_connection_endpoint_t endpoint) {
  if (connectivity->result.failure == LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE) {
    connectivity->result =
        (loom_aie2p_native_connectivity_result_t){failure, endpoint};
  }
}

static void loom_aie2p_connectivity_enqueue(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_endpoint_t endpoint, bool framed) {
  loom_aie2p_native_connectivity_tile_t* tile =
      loom_aie2p_connectivity_tile(connectivity, endpoint.coordinate);
  const uint8_t ordinal =
      connectivity->ordinals[tile->kind - 1][LOOM_XDNA_STREAM_DIRECTION_SLAVE -
                                             1][endpoint.port] +
      endpoint.channel;
  const uint64_t bit = UINT64_C(1) << (ordinal + (framed ? 32 : 0));
  if (tile->visited & bit) {
    loom_aie2p_connectivity_fail(
        connectivity, LOOM_AIE2P_NATIVE_CONNECTIVITY_REPEATED_ARRIVAL,
        endpoint);
    return;
  }
  tile->visited |= bit;
  connectivity->arrivals[connectivity->arrival_count++] =
      (loom_aie2p_native_connection_arrival_t){endpoint, framed};
}

static void loom_aie2p_connectivity_deliver(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_endpoint_t endpoint, bool framed,
    bool retain_header) {
  loom_aie2p_native_connectivity_tile_t* tile =
      loom_aie2p_connectivity_tile(connectivity, endpoint.coordinate);
  const uint8_t index = endpoint.port * 8 + endpoint.channel;
  const uint64_t bit = UINT64_C(1) << (index % 64);
  if (!(tile->remaining[index / 64] & bit)) {
    loom_aie2p_connectivity_fail(
        connectivity, LOOM_AIE2P_NATIVE_CONNECTIVITY_UNEXPECTED_DESTINATION,
        endpoint);
  } else if (framed != retain_header) {
    loom_aie2p_connectivity_fail(
        connectivity, LOOM_AIE2P_NATIVE_CONNECTIVITY_HEADER, endpoint);
  }
  tile->remaining[index / 64] &= ~bit;
}

static void loom_aie2p_connectivity_follow(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_endpoint_t endpoint, bool framed,
    bool retain_header) {
  loom_aie2p_native_connectivity_tile_t* tile =
      loom_aie2p_connectivity_tile(connectivity, endpoint.coordinate);
  if (tile->kind == LOOM_XDNA_TILE_KIND_SHIM_NOC &&
      endpoint.port == LOOM_XDNA_STREAM_PORT_SOUTH) {
    uint8_t engines = tile->egress[endpoint.channel];
    if (!engines) {
      // Configuration completion can terminate at an external shim port.
      loom_aie2p_connectivity_deliver(connectivity, endpoint, framed,
                                      retain_header);
    }
    while (engines) {
      endpoint.port = LOOM_XDNA_STREAM_PORT_DMA;
      endpoint.channel = (uint8_t)iree_math_count_trailing_zeros_u32(engines);
      engines &= engines - 1;
      loom_aie2p_connectivity_deliver(connectivity, endpoint, framed,
                                      retain_header);
    }
    return;
  }
  switch (endpoint.port) {
    case LOOM_XDNA_STREAM_PORT_NORTH:
      ++endpoint.coordinate.row;
      endpoint.port = LOOM_XDNA_STREAM_PORT_SOUTH;
      break;
    case LOOM_XDNA_STREAM_PORT_SOUTH:
      --endpoint.coordinate.row;
      endpoint.port = LOOM_XDNA_STREAM_PORT_NORTH;
      break;
    case LOOM_XDNA_STREAM_PORT_EAST:
      ++endpoint.coordinate.column;
      endpoint.port = LOOM_XDNA_STREAM_PORT_WEST;
      break;
    case LOOM_XDNA_STREAM_PORT_WEST:
      --endpoint.coordinate.column;
      endpoint.port = LOOM_XDNA_STREAM_PORT_EAST;
      break;
    case LOOM_XDNA_STREAM_PORT_FIFO:
      break;
    default:
      loom_aie2p_connectivity_deliver(connectivity, endpoint, framed,
                                      retain_header);
      return;
  }
  if (endpoint.coordinate.column >= connectivity->family->column_count ||
      endpoint.coordinate.row >= connectivity->family->row_count) {
    loom_aie2p_connectivity_fail(
        connectivity, LOOM_AIE2P_NATIVE_CONNECTIVITY_ARRAY_BOUNDARY, endpoint);
    return;
  }
  loom_aie2p_connectivity_enqueue(connectivity, endpoint, framed);
}

static void loom_aie2p_connectivity_visit(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_arrival_t arrival, uint8_t packet,
    bool retain_header) {
  loom_aie2p_native_connectivity_tile_t* tile =
      loom_aie2p_connectivity_tile(connectivity, arrival.endpoint.coordinate);
  const uint8_t slave =
      connectivity->ordinals[tile->kind - 1][LOOM_XDNA_STREAM_DIRECTION_SLAVE -
                                             1][arrival.endpoint.port] +
      arrival.endpoint.channel;
  const uint32_t slave_word =
      tile->state ? tile->state->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES]
                        .words[slave]
                  : 0;
  if (!(slave_word & (UINT32_C(1) << 31))) {
    loom_aie2p_connectivity_fail(connectivity,
                                 LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END,
                                 arrival.endpoint);
    return;
  }
  const bool packet_mode = (slave_word & (UINT32_C(1) << 30)) != 0;
  uint8_t arbiter = UINT8_MAX;
  uint8_t selection = 0;
  if (packet_mode) {
    if (!arrival.framed) {
      loom_aie2p_connectivity_fail(connectivity,
                                   LOOM_AIE2P_NATIVE_CONNECTIVITY_HEADER,
                                   arrival.endpoint);
      return;
    }
    const loom_aie2p_native_register_bank_t* filters =
        &tile->state->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS];
    const uint32_t slot_count =
        filters->word_count /
        tile->state->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].word_count;
    for (uint32_t slot = 0; slot < slot_count; ++slot) {
      const uint32_t word = filters->words[slave * slot_count + slot];
      if ((word & (1u << 8)) &&
          (((word >> 24) ^ packet) & ((word >> 16) & 31)) == 0) {
        arbiter = word & 7;
        selection = (word >> 4) & 3;
        break;
      }
    }
  }
  const loom_aie2p_native_register_bank_t* masters =
      &tile->state->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS];
  uint32_t output_count = 0;
  for (uint32_t master = 0; master < masters->word_count; ++master) {
    const uint32_t word = masters->words[master];
    if (!(word & (UINT32_C(1) << 31)) ||
        ((word & (UINT32_C(1) << 30)) != 0) != packet_mode) {
      continue;
    }
    if (packet_mode
            ? ((word & 7) != arbiter || !(word & (1u << (selection + 3))))
            : (word & 127) != slave) {
      continue;
    }
    ++output_count;
    loom_aie2p_native_connection_endpoint_t endpoint =
        connectivity->ports[tile->kind - 1]
                           [LOOM_XDNA_STREAM_DIRECTION_MASTER - 1][master];
    endpoint.coordinate = arrival.endpoint.coordinate;
    loom_aie2p_connectivity_follow(
        connectivity, endpoint,
        arrival.framed && !(packet_mode && (word & (1u << 7))), retain_header);
  }
  if (!output_count) {
    loom_aie2p_connectivity_fail(connectivity,
                                 LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END,
                                 arrival.endpoint);
  }
}

loom_aie2p_native_connectivity_result_t loom_aie2p_native_connectivity_check(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_endpoint_t source, uint8_t packet,
    const loom_aie2p_native_connection_endpoint_t* destinations,
    iree_host_size_t destination_count, bool retain_header) {
  connectivity->arrival_count = 0;
  connectivity->result = (loom_aie2p_native_connectivity_result_t){0};
  for (iree_host_size_t i = 0; i < connectivity->tile_count; ++i) {
    connectivity->tiles[i].visited = 0;
    memset(connectivity->tiles[i].remaining, 0,
           sizeof(connectivity->tiles[i].remaining));
  }
  for (iree_host_size_t i = 0; i < destination_count; ++i) {
    const loom_aie2p_native_connection_endpoint_t endpoint = destinations[i];
    const uint8_t index = endpoint.port * 8 + endpoint.channel;
    loom_aie2p_connectivity_tile(connectivity, endpoint.coordinate)
        ->remaining[index / 64] |= UINT64_C(1) << (index % 64);
  }
  const loom_aie2p_native_connectivity_tile_t* tile =
      loom_aie2p_connectivity_tile(connectivity, source.coordinate);
  if (tile->kind == LOOM_XDNA_TILE_KIND_SHIM_NOC &&
      source.port == LOOM_XDNA_STREAM_PORT_DMA) {
    uint8_t lanes = tile->ingress[source.channel];
    if (!lanes) {
      loom_aie2p_connectivity_fail(
          connectivity, LOOM_AIE2P_NATIVE_CONNECTIVITY_DEAD_END, source);
    }
    while (lanes) {
      source.port = LOOM_XDNA_STREAM_PORT_SOUTH;
      source.channel = (uint8_t)iree_math_count_trailing_zeros_u32(lanes);
      lanes &= lanes - 1;
      loom_aie2p_connectivity_enqueue(connectivity, source,
                                      packet != UINT8_MAX);
    }
  } else {
    loom_aie2p_connectivity_enqueue(connectivity, source, packet != UINT8_MAX);
  }
  for (iree_host_size_t i = 0;
       i < connectivity->arrival_count &&
       connectivity->result.failure == LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE;
       ++i) {
    loom_aie2p_connectivity_visit(connectivity, connectivity->arrivals[i],
                                  packet, retain_header);
  }
  for (iree_host_size_t i = 0;
       i < destination_count &&
       connectivity->result.failure == LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE;
       ++i) {
    const loom_aie2p_native_connection_endpoint_t endpoint = destinations[i];
    const uint8_t index = endpoint.port * 8 + endpoint.channel;
    if (loom_aie2p_connectivity_tile(connectivity, endpoint.coordinate)
            ->remaining[index / 64] &
        (UINT64_C(1) << (index % 64))) {
      loom_aie2p_connectivity_fail(
          connectivity, LOOM_AIE2P_NATIVE_CONNECTIVITY_MISSING_DESTINATION,
          endpoint);
    }
  }
  return connectivity->result;
}

static const char* loom_aie2p_connectivity_port_name(
    loom_xdna_stream_port_t port) {
  static const char* const names[] = {
      "invalid", "core", "dma",   "tile-control", "fifo",
      "south",   "west", "north", "east",         "trace"};
  return names[port];
}

static void loom_aie2p_connectivity_verify_delivery(
    loom_aie2p_native_connectivity_t* connectivity,
    loom_aie2p_native_connection_endpoint_t source, uint8_t packet,
    const loom_aie2p_native_connection_endpoint_t* destinations,
    iree_host_size_t destination_count, bool retain_header) {
  const loom_aie2p_native_connectivity_result_t result =
      loom_aie2p_native_connectivity_check(connectivity, source, packet,
                                           destinations, destination_count,
                                           retain_header);
  if (result.failure != LOOM_AIE2P_NATIVE_CONNECTIVITY_NONE) {
    static const char* const failures[] = {"none",
                                           "dead end",
                                           "repeated arrival",
                                           "unexpected destination",
                                           "missing destination",
                                           "incorrect header framing",
                                           "array boundary"};
    iree_status_abort(iree_make_status(
        IREE_STATUS_INTERNAL,
        "AIE2P connection invariant: source (%u,%u) %s[%u], packet %u: "
        "%s at (%u,%u) %s[%u]",
        source.coordinate.column, source.coordinate.row,
        loom_aie2p_connectivity_port_name(source.port), source.channel, packet,
        failures[result.failure], result.endpoint.coordinate.column,
        result.endpoint.coordinate.row,
        loom_aie2p_connectivity_port_name(result.endpoint.port),
        result.endpoint.channel));
  }
}

static iree_status_t loom_aie2p_native_connections_verify_impl(
    const loom_aie2p_native_context_t* context) {
  loom_aie2p_native_connectivity_t* connectivity;
  iree_arena_allocator_t* arena = context->pass->arena;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_connectivity_allocate(
      context->family, context->switches, context->routing.routes,
      context->routing.route_count, arena, &connectivity));
  iree_host_size_t capacity = context->control_connection_count;
  for (const loom_aie2p_native_payload_source_t* source =
           context->payload_sources;
       source; source = source->next) {
    for (const loom_aie2p_native_payload_class_t* payload = source->classes;
         payload; payload = payload->next) {
      capacity = iree_max(capacity, payload->destination_count);
    }
  }
  loom_aie2p_native_connection_endpoint_t* destinations;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*destinations), (void**)&destinations));
  for (const loom_aie2p_native_payload_source_t* source =
           context->payload_sources;
       source; source = source->next) {
    for (const loom_aie2p_native_payload_class_t* payload = source->classes;
         payload; payload = payload->next) {
      if (!payload->routed) {
        continue;
      }
      for (iree_host_size_t i = 0; i < payload->destination_count; ++i) {
        destinations[i] = (loom_aie2p_native_connection_endpoint_t){
            payload->destinations[i].tile->coordinate,
            LOOM_XDNA_STREAM_PORT_DMA, payload->destinations[i].engine};
      }
      const loom_aie2p_native_connection_endpoint_t origin = {
          source->endpoint.tile->coordinate, LOOM_XDNA_STREAM_PORT_DMA,
          source->endpoint.engine};
      loom_aie2p_connectivity_verify_delivery(
          connectivity, origin, source->framed ? payload->packet : UINT8_MAX,
          destinations, payload->destination_count, false);
    }
  }
  // Group retained control requests by physical sender and packet identity.
  // Destination membership comes from request ownership, never route edges.
  const iree_host_size_t group_count = connectivity->tile_count * 2 * 32;
  iree_host_size_t* first;
  iree_host_size_t* next;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*first), (void**)&first));
  memset(first, 0xff, group_count * sizeof(*first));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, context->control_connection_count, sizeof(*next), (void**)&next));
  for (iree_host_size_t i = 0; i < context->control_connection_count; ++i) {
    const loom_aie2p_native_control_connection_t* request =
        &context->control_connections[i];
    const iree_host_size_t tile =
        request->source.coordinate.column * context->family->row_count +
        request->source.coordinate.row;
    const iree_host_size_t group =
        (tile * 2 +
         (request->source.port == LOOM_XDNA_STREAM_PORT_TILE_CONTROL)) *
            32 +
        request->packet;
    next[i] = first[group];
    first[group] = i;
  }
  for (iree_host_size_t group = 0; group < group_count; ++group) {
    if (first[group] == IREE_HOST_SIZE_MAX) {
      continue;
    }
    const loom_aie2p_native_control_connection_t* request =
        &context->control_connections[first[group]];
    iree_host_size_t count = 0;
    for (iree_host_size_t i = first[group]; i != IREE_HOST_SIZE_MAX;
         i = next[i]) {
      destinations[count++] = context->control_connections[i].destination;
    }
    loom_aie2p_connectivity_verify_delivery(connectivity, request->source,
                                            request->packet, destinations,
                                            count, true);
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_connections_verify(
    const loom_aie2p_native_context_t* context) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(context->pass->arena);
  iree_status_t status = loom_aie2p_native_connections_verify_impl(context);
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
