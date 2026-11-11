// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

typedef struct loom_aie2p_native_payload_inventory_t {
  // Dense physical queue offsets indexed by tile ordinal.
  iree_host_size_t* bases;
  // Canonical sender at each memory-to-stream slot.
  loom_aie2p_native_payload_source_t** sources;
  // First sender reaching each stream-to-memory slot, for framing admission.
  loom_aie2p_native_payload_source_t** arrivals;
  // Append position preserving the first occurrence of each sender.
  loom_aie2p_native_payload_source_t** tail;
} loom_aie2p_native_payload_inventory_t;

static uint32_t loom_aie2p_native_payload_distance(
    const loom_aie2p_native_tile_t* source,
    const loom_aie2p_native_tile_t* destination) {
  const uint16_t a[] = {source->coordinate.column, source->coordinate.row};
  const uint16_t b[] = {destination->coordinate.column,
                        destination->coordinate.row};
  return (a[0] > b[0] ? a[0] - b[0] : b[0] - a[0]) +
         (a[1] > b[1] ? a[1] - b[1] : b[1] - a[1]);
}

static bool loom_aie2p_native_payload_before(
    const loom_aie2p_native_tile_t* source,
    loom_aie2p_native_payload_endpoint_t a,
    loom_aie2p_native_payload_endpoint_t b) {
  const uint32_t da = loom_aie2p_native_payload_distance(source, a.tile);
  const uint32_t db = loom_aie2p_native_payload_distance(source, b.tile);
  return da != db           ? da > db
         : a.tile != b.tile ? a.tile < b.tile
                            : a.engine < b.engine;
}

static iree_status_t loom_aie2p_native_intern_payload(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_payload_inventory_t* inventory,
    loom_aie2p_native_payload_endpoint_t sender,
    loom_aie2p_native_payload_endpoint_t* destinations,
    iree_host_size_t destination_count, bool framed, bool distribution,
    const loom_op_t* op, loom_aie2p_native_payload_class_t** out_payload) {
  *out_payload = NULL;
  const iree_host_size_t slot =
      inventory->bases[sender.tile - context->tiles] + sender.engine;
  loom_aie2p_native_payload_source_t* source = inventory->sources[slot];
  if (!source) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                             sizeof(*source), (void**)&source));
    *source = (loom_aie2p_native_payload_source_t){.endpoint = sender,
                                                   .route = UINT32_MAX};
    inventory->sources[slot] = *inventory->tail = source;
    inventory->tail = &source->next;
  }
  source->framed |= framed;
  // Canonical destination order also grows long shared prefixes before their
  // shorter branches. No transfer order changes within a receiver's stream.
  for (iree_host_size_t i = 1; i < destination_count; ++i) {
    const loom_aie2p_native_payload_endpoint_t endpoint = destinations[i];
    iree_host_size_t j = i;
    while (j && loom_aie2p_native_payload_before(sender.tile, endpoint,
                                                 destinations[j - 1])) {
      destinations[j] = destinations[j - 1];
      --j;
    }
    destinations[j] = endpoint;
  }
  loom_aie2p_native_payload_class_t** position = &source->classes;
  for (; *position; position = &(*position)->next) {
    loom_aie2p_native_payload_class_t* candidate = *position;
    if (candidate->destination_count != destination_count) {
      continue;
    }
    iree_host_size_t i = 0;
    while (i < destination_count &&
           candidate->destinations[i].tile == destinations[i].tile &&
           candidate->destinations[i].engine == destinations[i].engine) {
      ++i;
    }
    if (i == destination_count) {
      candidate->distribution_only &= distribution;
      *out_payload = candidate;
      return iree_ok_status();
    }
  }
  if (source->class_count == 32) {
    return loom_aie2p_native_reject(
        context, op,
        IREE_SV(
            "physical sender destination classes within the packet ID range"));
  }
  loom_aie2p_native_payload_class_t* payload;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      context->pass->arena,
      sizeof(*payload) + destination_count * sizeof(*destinations),
      (void**)&payload));
  *payload = (loom_aie2p_native_payload_class_t){
      .source = source,
      .destinations = (loom_aie2p_native_payload_endpoint_t*)(payload + 1),
      .destination_count = destination_count,
      .packet = source->class_count++,
      .distribution_only = distribution};
  memcpy(payload->destinations, destinations,
         destination_count * sizeof(*destinations));
  *position = *out_payload = payload;
  source->framed |= source->class_count > 1;
  for (iree_host_size_t i = 0; i < destination_count; ++i) {
    const iree_host_size_t destination_slot =
        inventory->bases[destinations[i].tile - context->tiles] +
        destinations[i].engine;
    loom_aie2p_native_payload_source_t* previous =
        inventory->arrivals[destination_slot];
    if (previous && previous != source) {
      previous->framed = source->framed = true;
    }
    inventory->arrivals[destination_slot] = source;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_form_payloads(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
  loom_aie2p_native_payload_inventory_t inventory = {
      .tail = &context->payload_sources};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, tile_count, sizeof(*inventory.bases),
      (void**)&inventory.bases));
  iree_host_size_t slots = 0;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    inventory.bases[i] = slots;
    slots += context->tiles[i].facts->dma.channel_count_per_direction;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, slots * 2, sizeof(*inventory.sources),
      (void**)&inventory.sources));
  memset(inventory.sources, 0, slots * 2 * sizeof(*inventory.sources));
  inventory.arrivals = inventory.sources + slots;
  loom_aie2p_native_payload_endpoint_t* destinations;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, realization->invocations.channel_count + 1,
      sizeof(*destinations), (void**)&destinations));
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    const loom_op_t* op = realization->function.op;
    for (loom_aie2p_native_dma_path_t* path = worker->transfer_paths; path;
         path = path->next_demand) {
      const loom_aie2p_native_payload_endpoint_t local = {
          .tile = path->local, .engine = path->local_engine};
      const loom_aie2p_native_payload_endpoint_t shim = {
          .tile = path->shim, .engine = path->shim_engine};
      destinations[0] = path->ingress ? local : shim;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_intern_payload(
          context, &inventory, path->ingress ? shim : local, destinations, 1,
          false, false, op, &path->payload));
      if (!path->payload) {
        return iree_ok_status();
      }
    }
    for (loom_aie2p_native_distribution_t* distribution =
             worker->distribution_plan ? worker->distribution_plan->geometries
                                       : NULL;
         distribution; distribution = distribution->next_demand) {
      loom_aie2p_native_distribution_path_t* path = distribution->path;
      if (path->first != distribution) {
        continue;
      }
      const bool packet = path->packet_count != 0;
      const iree_host_size_t count = packet ? path->packet_count : 1;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          context->pass->arena, count, sizeof(*path->payloads),
          (void**)&path->payloads));
      const loom_aie2p_native_payload_endpoint_t sender = {
          .tile = distribution->sender.tile,
          .engine = distribution->sender.engine};
      for (iree_host_size_t c = 0; c < count; ++c) {
        const iree_host_size_t receivers =
            packet ? path->packets[c].destination_count
                   : distribution->receiver_count;
        for (iree_host_size_t j = 0; j < receivers; ++j) {
          const iree_host_size_t ordinal =
              packet ? path->packets[c].receivers[j] : j;
          destinations[j] = (loom_aie2p_native_payload_endpoint_t){
              .tile = distribution->receivers[ordinal].tile,
              .engine = distribution->receivers[ordinal].engine};
        }
        IREE_RETURN_IF_ERROR(loom_aie2p_native_intern_payload(
            context, &inventory, sender, destinations, receivers, packet, true,
            op, &path->payloads[c]));
        if (!path->payloads[c]) {
          return iree_ok_status();
        }
      }
    }
    loom_aie2p_native_input_t* input = worker->input;
    if (input) {
      const loom_aie2p_native_payload_endpoint_t sender = {
          .tile = input->path->shim, .engine = input->path->input_engine};
      for (iree_host_size_t j = 0; j < input->source->phase_count; ++j) {
        loom_aie2p_native_input_phase_t* phase = &input->phases[j];
        destinations[0] = (loom_aie2p_native_payload_endpoint_t){
            .tile = context->pool_tiles[phase->destination->pool_index],
            .engine = phase->receiver.engine};
        IREE_RETURN_IF_ERROR(loom_aie2p_native_intern_payload(
            context, &inventory, sender, destinations, 1, true, false, op,
            &phase->payload));
        if (!phase->payload) {
          return iree_ok_status();
        }
      }
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_route_payload(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_payload_class_t* payload, const loom_op_t* op,
    uint32_t* next_source, bool* out_valid) {
  *out_valid = false;
  if (payload->routed) {
    *out_valid = true;
    return iree_ok_status();
  }
  loom_aie2p_native_payload_source_t* source = payload->source;
  if (source->route == UINT32_MAX) {
    // The first destination owns the tree identity. A separate identity would
    // exceed the route inventory's one source slot per destination demand.
    source->route = *next_source;
  }
  payload->route_first = context->routing.route_count;
  for (iree_host_size_t i = 0; i < payload->destination_count; ++i) {
    const loom_aie2p_native_payload_endpoint_t* destination =
        &payload->destinations[i];
    const iree_host_size_t first = context->routing.route_count;
    if (!loom_aie2p_array_route_stream(
            &context->routing, (*next_source)++, source->route,
            source->endpoint.tile->coordinate, LOOM_XDNA_STREAM_PORT_DMA,
            source->endpoint.engine, destination->tile->coordinate,
            LOOM_XDNA_STREAM_PORT_DMA, destination->engine, source->framed)) {
      return loom_aie2p_native_reject(
          context, op, IREE_SV("available physical payload stream links"));
    }
    loom_aie2p_native_route_label(context, first,
                                  source->framed ? payload->packet : UINT8_MAX,
                                  UINT32_MAX);
  }
  payload->route_count = context->routing.route_count - payload->route_first;
  payload->routed = true;
  *out_valid = true;
  return iree_ok_status();
}

void loom_aie2p_native_frame_payload(
    const loom_aie2p_native_payload_class_t* payload, uint32_t words[8]) {
  if (!payload->source->framed) {
    return;
  }
  const loom_aie2p_native_memory_layout_t* layout =
      payload->source->endpoint.tile->memory_layout;
  const uint8_t word = layout ? layout->packet.word : 2;
  const loom_xdna_register_field_id_t enable =
      layout ? layout->packet.enable
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_ENABLE_PACKET;
  const loom_xdna_register_field_id_t id =
      layout ? layout->packet.id
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_PACKET_ID;
  words[word] |= loom_xdna_register_field_encode(enable, 1) |
                 loom_xdna_register_field_encode(id, payload->packet);
}
