// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

typedef struct loom_aie2p_native_payload_multicast_t {
  // Next eligible SRAM multicast in the formation inventory.
  struct loom_aie2p_native_payload_multicast_t* next;
  // Physical class owning this receiver set.
  loom_aie2p_native_payload_class_t* payload;
  // Source-independent receiver identity, indexed by physical DMA queue.
  uint64_t* receivers;
  // First member with equal receivers and disjoint construction lifetimes.
  struct loom_aie2p_native_payload_multicast_t* group;
  // Next distinct group; populated only on its first member.
  struct loom_aie2p_native_payload_multicast_t* next_group;
  // Number of compatible physical senders, populated only on the first member.
  iree_host_size_t member_count;
  // Shared packet lanes, initialized only for groups with several senders.
  loom_aie2p_array_route_multicast_t routes;
} loom_aie2p_native_payload_multicast_t;

typedef struct loom_aie2p_native_payload_inventory_t {
  // Dense physical queue offsets indexed by tile ordinal.
  iree_host_size_t* bases;
  // Canonical sender at each memory-to-stream slot.
  loom_aie2p_native_payload_source_t** sources;
  // First sender reaching each stream-to-memory slot, for framing admission.
  loom_aie2p_native_payload_source_t** arrivals;
  // Append position preserving the first occurrence of each sender.
  loom_aie2p_native_payload_source_t** tail;
  // Serial constructions can reuse physical receiver queues.
  bool has_serial;
  // Receiver bitset length in uint64_t words.
  iree_host_size_t receiver_words;
  // SRAM multicast classes with source-independent receiver identities.
  loom_aie2p_native_payload_multicast_t* multicasts;
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
    *source = (loom_aie2p_native_payload_source_t){.endpoint = sender};
    inventory->sources[slot] = *inventory->tail = source;
    inventory->tail = &source->next;
  }
  source->framed |= framed;
  // Canonical destination order also grows long shared prefixes before their
  // shorter branches. No transfer order changes within a receiver's stream.
  bool compute =
      destinations[0].tile->facts->kind == LOOM_XDNA_TILE_KIND_COMPUTE;
  bool cross_column =
      destinations[0].tile->coordinate.column != sender.tile->coordinate.column;
  for (iree_host_size_t i = 1; i < destination_count; ++i) {
    const loom_aie2p_native_payload_endpoint_t endpoint = destinations[i];
    compute &= endpoint.tile->facts->kind == LOOM_XDNA_TILE_KIND_COMPUTE;
    cross_column |=
        endpoint.tile->coordinate.column != sender.tile->coordinate.column;
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
  source->nonlocal_class_count += !compute || cross_column;
  if (inventory->has_serial && distribution && compute && cross_column &&
      destination_count > 1 &&
      sender.tile->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY) {
    loom_aie2p_native_payload_multicast_t* multicast;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        context->pass->arena,
        sizeof(*multicast) + inventory->receiver_words * sizeof(uint64_t),
        (void**)&multicast));
    *multicast = (loom_aie2p_native_payload_multicast_t){
        .next = inventory->multicasts,
        .payload = payload,
        .receivers = (uint64_t*)(multicast + 1)};
    memset(multicast->receivers, 0,
           inventory->receiver_words * sizeof(*multicast->receivers));
    payload->multicast = inventory->multicasts = multicast;
  }
  *position = *out_payload = payload;
  source->framed |= source->class_count > 1;
  for (iree_host_size_t i = 0; i < destination_count; ++i) {
    const iree_host_size_t destination_slot =
        inventory->bases[destinations[i].tile - context->tiles] +
        destinations[i].engine;
    if (payload->multicast) {
      payload->multicast->receivers[destination_slot / 64] |=
          UINT64_C(1) << (destination_slot % 64);
    }
    loom_aie2p_native_payload_source_t* previous =
        inventory->arrivals[destination_slot];
    if (previous && previous != source) {
      previous->framed = source->framed = true;
    }
    inventory->arrivals[destination_slot] = source;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_share_payloads(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_payload_inventory_t* inventory) {
  // Queue assignment is the exclusion proof: distinct physical senders can
  // reach the same receiver queue only in disjoint construction lifetimes.
  // Equal complete receiver sets therefore establish both packet identity and
  // serialization. Reallocating lifetimes here would duplicate that proof.
  loom_aie2p_native_payload_multicast_t* groups = NULL;
  for (loom_aie2p_native_payload_multicast_t* multicast = inventory->multicasts;
       multicast; multicast = multicast->next) {
    // A second nonlocal class could merge into this group's packet namespace.
    // Local upward classes never enter its horizontal or descending links.
    if (!multicast->payload->distribution_only ||
        multicast->payload->source->nonlocal_class_count != 1) {
      continue;
    }
    loom_aie2p_native_payload_multicast_t* group = groups;
    while (group &&
           memcmp(group->receivers, multicast->receivers,
                  inventory->receiver_words * sizeof(*group->receivers))) {
      group = group->next_group;
    }
    if (!group) {
      group = multicast;
      group->next_group = groups;
      groups = group;
    }
    multicast->group = group;
    ++group->member_count;
  }
  for (loom_aie2p_native_payload_multicast_t* group = groups; group;
       group = group->next_group) {
    if (group->member_count > 1) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_route_multicast_initialize(
          &context->routing, context->pass->arena, &group->routes));
    }
  }
  for (loom_aie2p_native_payload_multicast_t* multicast = inventory->multicasts;
       multicast; multicast = multicast->next) {
    if (!multicast->group || multicast->group->member_count < 2) {
      continue;
    }
    // All merged inputs see exactly the same multicast identity. Other classes
    // remain private to their source column and retain distinct local IDs.
    loom_aie2p_native_payload_source_t* source = multicast->payload->source;
    source->framed = true;
    uint8_t packet = 1;
    for (loom_aie2p_native_payload_class_t* payload = source->classes; payload;
         payload = payload->next) {
      payload->packet = payload == multicast->payload ? 0 : packet++;
    }
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
      .tail = &context->payload_sources,
      .has_serial = realization->invocations.lifetimes.has_serial};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, tile_count, sizeof(*inventory.bases),
      (void**)&inventory.bases));
  iree_host_size_t slots = 0;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    inventory.bases[i] = slots;
    slots += context->tiles[i].facts->dma.channel_count_per_direction;
  }
  inventory.receiver_words = (slots + 63) / 64;
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
  IREE_RETURN_IF_ERROR(loom_aie2p_native_share_payloads(context, &inventory));
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
  loom_aie2p_array_route_multicast_t* multicast =
      payload->multicast && payload->multicast->group &&
              payload->multicast->group->member_count > 1
          ? &payload->multicast->group->routes
          : NULL;
  payload->route_first = context->routing.route_count;
  for (iree_host_size_t i = 0; i < payload->destination_count; ++i) {
    const loom_aie2p_native_payload_endpoint_t* destination =
        &payload->destinations[i];
    const iree_host_size_t first = context->routing.route_count;
    const uint32_t label = (*next_source)++;
    const bool routed =
        multicast
            ? loom_aie2p_array_route_multicast(
                  &context->routing, label, &source->tree,
                  source->endpoint.tile->coordinate, source->endpoint.engine,
                  destination->tile->coordinate, destination->engine, multicast)
            : loom_aie2p_array_route_stream(
                  &context->routing, label, &source->tree,
                  source->endpoint.tile->coordinate, LOOM_XDNA_STREAM_PORT_DMA,
                  source->endpoint.engine, destination->tile->coordinate,
                  LOOM_XDNA_STREAM_PORT_DMA, destination->engine,
                  source->framed);
    if (!routed) {
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
