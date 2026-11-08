// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"

#include "iree/base/internal/math.h"

uint8_t loom_aie2p_native_delivery_packet(
    loom_aie2p_native_delivery_t* delivery, uint32_t occupied,
    const loom_aie2p_native_tile_t* tile, loom_xdna_stream_port_t port,
    uint8_t channel) {
  uint32_t available =
      ~(occupied | (UINT32_C(1) << LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID));
  while (available) {
    const uint8_t packet = iree_math_count_trailing_zeros_u32(available);
    available &= available - 1;
    if (delivery && delivery->packets[packet].tile) {
      if (delivery->packets[packet].tile != tile ||
          delivery->packets[packet].port != port ||
          delivery->packets[packet].channel != channel) {
        continue;
      }
    }
    if (delivery) {
      delivery->packets[packet].tile = tile;
      delivery->packets[packet].port = port;
      delivery->packets[packet].channel = channel;
    }
    return packet;
  }
  return UINT8_MAX;
}

uint8_t loom_aie2p_native_collective_packet(uint32_t* occupied) {
  const uint32_t available = ~*occupied;
  if (!available) {
    return UINT8_MAX;
  }
  const uint8_t packet = 31 - iree_math_count_leading_zeros_u32(available);
  *occupied |= UINT32_C(1) << packet;
  return packet;
}

iree_status_t loom_aie2p_native_prepare_sharing(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization) {
  bool has_controls = false;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    has_controls |= context->workers[i].refills != NULL ||
                    context->workers[i].channels.remote_count != 0;
  }
  if (!has_controls) {
    return iree_ok_status();
  }
  iree_arena_allocator_t* arena = context->pass->arena;
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, tile_count,
                                                 sizeof(*context->deliveries),
                                                 (void**)&context->deliveries));
  memset(context->deliveries, 0, tile_count * sizeof(*context->deliveries));
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->distribution_phases) {
      continue;
    }
    for (loom_aie2p_native_distribution_t* distribution = worker->distributions;
         distribution; distribution = distribution->next) {
      if (distribution->receiver_count != 1 || !distribution->repetitions ||
          distribution->source->record_count != 1 ||
          distribution->source->records->release ||
          distribution->source->records->admissions !=
              distribution->repetitions ||
          distribution->sender.channel->source->capacity != 1 ||
          distribution->sender.channel->replay.record_count ||
          !distribution->routes.count) {
        continue;
      }
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &distribution->receivers[0];
      const loom_aie2p_native_channel_t* channel = receiver->channel;
      const loom_aie2p_native_cursor_t* reader = &channel->cursor.reader;
      const bool core_receiver =
          reader->worker != UINT32_MAX && !reader->shared &&
          reader->exact_admissions &&
          reader->maximum_admissions == distribution->repetitions &&
          realization->invocations.workers[reader->worker].invocation ==
              realization->invocations.workers[i].invocation &&
          context->workers[reader->worker].execution ==
              LOOM_AIE2P_NATIVE_EXECUTION_CORE;
      if (channel->source->capacity != 1 || channel->replay.record_count ||
          (distribution->repetitions > 1 && !core_receiver &&
           channel->configuration_returns.count != distribution->repetitions) ||
          distribution->sender.tile == receiver->tile) {
        continue;
      }
      // One record uses the initially free slot and needs no later credit
      // return. Repeated delivery requires the exact receiver continuation.
      const iree_host_size_t tile_index = receiver->tile - context->tiles;
      loom_aie2p_native_delivery_t** selected =
          &context->deliveries[tile_index];
      if (!*selected) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(arena, sizeof(**selected), (void**)selected));
        **selected = (loom_aie2p_native_delivery_t){
            .distribution = distribution,
            .reader_worker = core_receiver ? reader->worker : UINT32_MAX};
      } else if (distribution->routes.count <
                 (*selected)->distribution->routes.count) {
        (*selected)->distribution = distribution;
        (*selected)->reader_worker =
            core_receiver ? reader->worker : UINT32_MAX;
      }
    }
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_native_delivery_t* delivery = context->deliveries[i];
    if (!delivery) {
      continue;
    }
    const loom_aie2p_native_distribution_t* distribution =
        delivery->distribution;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_route_sharing_initialize(
        &context->routing, distribution->routes.first,
        distribution->routes.count, distribution->receivers[0].tile->coordinate,
        arena, &delivery->sharing));
  }
  return iree_ok_status();
}

void loom_aie2p_native_commit_sharing(loom_aie2p_native_context_t* context) {
  if (!context->deliveries) {
    return;
  }
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
  // Control selection reserves this payload identity across every shared
  // domain; the remaining IDs name distinct control or credit destinations.
  const uint8_t payload_packet = LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_native_delivery_t* delivery = context->deliveries[i];
    if (!delivery || !delivery->sharing.used) {
      continue;
    }
    loom_aie2p_native_distribution_t* distribution = delivery->distribution;
    loom_aie2p_native_distribution_endpoint_t* sender = &distribution->sender;
    const loom_aie2p_native_memory_layout_t* layout =
        sender->tile->memory_layout;
    const uint16_t engine = sender->engine;
    delivery->address = loom_xdna_register_field_address(
        context->family, layout->directions[0].start, sender->tile->coordinate,
        &engine);
    delivery->task = loom_xdna_register_field_encode(
        layout->directions[0].start, sender->sequence.descriptor);
    // A singleton's finite startup plan may unroll several identical records
    // into a descriptor chain. A receiver grant admits exactly one record,
    // so keep only its first descriptor and remove that chain continuation.
    sender->sequence.descriptor_count = 1;
    sender->sequence.words[0][layout->chain.word] &=
        ~loom_xdna_register_field_encode(layout->chain.enable, 1);
    sender->sequence.task_count = 1;
    sender->sequence.tasks[0] = delivery->task;
    sender->sequence.words[0][layout->packet.word] |=
        loom_xdna_register_field_encode(layout->packet.enable, 1) |
        loom_xdna_register_field_encode(layout->packet.id, payload_packet);
    sender->wait_for_idle = true;
    for (iree_host_size_t j = distribution->routes.first;
         j < distribution->routes.first + distribution->routes.count; ++j) {
      context->routes[j].packet = payload_packet;
    }
    const iree_host_size_t channel_index =
        distribution->receivers[0].channel - context->channels;
    context->channels[channel_index].delivery = delivery;
  }
}

iree_status_t loom_aie2p_native_prepare_grant(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_delivery_t* delivery,
    loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, NULL, 0, &builder, out_symbol, &function));
  const loom_xdna_tile_coordinate_t sender =
      delivery->distribution->sender.tile->coordinate;
  const uint32_t origin = sender.column << context->family->column_shift |
                          sender.row << context->family->row_shift;
  loom_value_id_t task;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_constant(code, &builder, delivery->task, &task));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, delivery->packet, delivery->address - origin, &task, 1));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}
