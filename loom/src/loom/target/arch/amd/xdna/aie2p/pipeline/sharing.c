// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"

#include "iree/base/internal/math.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/payload.h"

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

// Sharing a wire changes its framing for every geometry. Admission is per
// receiver: each must reserve a complete record before starting its sender.
static bool loom_aie2p_native_select_grant_reader(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    loom_aie2p_native_distribution_t* distribution) {
  if ((distribution->ordered && distribution->transport->batch_count > 1) ||
      distribution->receiver_count != 1 || !distribution->repetitions ||
      distribution->source->record_count != 1 ||
      distribution->source->records->release ||
      distribution->source->records->admissions != distribution->repetitions ||
      distribution->sender.channel->source->capacity != 1 ||
      distribution->sender.channel->replay.record_count) {
    return false;
  }
  const loom_aie2p_native_distribution_endpoint_t* receiver =
      &distribution->receivers[0];
  const loom_aie2p_native_channel_t* channel = receiver->channel;
  const loom_aie2p_native_cursor_t* reader = &channel->cursor.reader;
  const bool core_receiver =
      reader->worker != UINT32_MAX &&
      loom_aie2p_native_cursor_has_exact_admissions(
          reader, distribution->repetitions) &&
      realization->invocations.workers[reader->worker].invocation ==
          distribution->invocation &&
      context->workers[reader->worker].execution ==
          LOOM_AIE2P_NATIVE_EXECUTION_CORE;
  if (channel->replay.record_count ||
      (distribution->repetitions > 1 && !core_receiver &&
       (channel->source->capacity != 1 ||
        channel->configuration_returns.count != distribution->repetitions)) ||
      distribution->sender.tile == receiver->tile) {
    return false;
  }
  distribution->grant_reader = core_receiver ? reader->worker : UINT32_MAX;
  return true;
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
    for (loom_aie2p_native_distribution_t* first =
             context->workers[i].distributions;
         first; first = first->next) {
      loom_aie2p_native_distribution_path_t* path = first->path;
      if (path->first != first || !path->routes.count) {
        continue;
      }
      const loom_aie2p_native_payload_class_t* payload = path->payloads[0];
      if (!payload->distribution_only || payload->source->class_count != 1 ||
          payload->distribution_path_count != 1) {
        continue;
      }
      bool eligible = true;
      for (loom_aie2p_native_distribution_t* geometry = path->geometries;
           geometry; geometry = geometry->next_geometry) {
        if (!loom_aie2p_native_select_grant_reader(context, realization,
                                                   geometry)) {
          eligible = false;
          break;
        }
      }
      if (!eligible) {
        continue;
      }
      const iree_host_size_t tile_index =
          first->receivers[0].tile - context->tiles;
      loom_aie2p_native_delivery_t** selected =
          &context->deliveries[tile_index];
      if (!*selected) {
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(arena, sizeof(**selected), (void**)selected));
        **selected = (loom_aie2p_native_delivery_t){.path = path};
      } else if (path->routes.count < (*selected)->path->routes.count) {
        (*selected)->path = path;
      }
    }
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    loom_aie2p_native_delivery_t* delivery = context->deliveries[i];
    if (!delivery) {
      continue;
    }
    const loom_aie2p_native_distribution_path_t* path = delivery->path;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_route_sharing_initialize(
        &context->routing, path->routes.first, path->routes.count,
        path->first->receivers[0].tile->coordinate, arena, &delivery->sharing));
    for (loom_aie2p_native_distribution_t* geometry = path->geometries;
         geometry; geometry = geometry->next_geometry) {
      loom_aie2p_native_grant_t* grant;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, sizeof(*grant), (void**)&grant));
      *grant = (loom_aie2p_native_grant_t){
          .next = delivery->grants,
          .distribution = geometry,
          .window_capacity =
              iree_min(geometry->receivers[0].channel->source->capacity,
                       geometry->sender.tile->facts->dma.task_queue_depth)};
      delivery->grants = grant;
    }
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
    for (loom_aie2p_native_grant_t* grant = delivery->grants; grant;
         grant = grant->next) {
      loom_aie2p_native_distribution_t* distribution = grant->distribution;
      loom_aie2p_native_distribution_endpoint_t* sender = &distribution->sender;
      const loom_aie2p_native_memory_layout_t* layout =
          sender->tile->memory_layout;
      const uint16_t engine = sender->engine;
      grant->address = loom_xdna_register_field_address(
          context->family, layout->directions[0].start,
          sender->tile->coordinate, &engine);
      grant->task = loom_xdna_register_field_encode(
          layout->directions[0].start, sender->sequence.descriptor);
      // Each startup record owns an initially free receiver slot. Subsequent
      // returns grant one successor each; the receiver suppresses grants for
      // the final window. The singleton sender always uses this descriptor.
      sender->sequence.descriptor_count = 1;
      sender->sequence.words[0][layout->chain.word] &=
          ~loom_xdna_register_field_encode(layout->chain.enable, 1);
      sender->sequence.task_count = 1;
      const uint32_t initial_records =
          iree_min(distribution->repetitions, grant->window_capacity);
      sender->sequence.tasks[0] =
          grant->task | loom_xdna_register_field_encode(
                            layout->directions[0].repeat, initial_records - 1);
      sender->sequence.words[0][layout->packet.word] |=
          loom_xdna_register_field_encode(layout->packet.enable, 1) |
          loom_xdna_register_field_encode(layout->packet.id, payload_packet);
      sender->wait_for_idle = true;
      const iree_host_size_t channel_index =
          distribution->receivers[0].channel - context->channels;
      context->channels[channel_index].grant = grant;
    }
    const loom_aie2p_native_distribution_path_t* path = delivery->path;
    for (iree_host_size_t j = path->routes.first;
         j < path->routes.first + path->routes.count; ++j) {
      context->routes[j].packet = payload_packet;
    }
  }
}

iree_status_t loom_aie2p_native_prepare_grant(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_grant_t* grant, loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, NULL, 0, &builder, out_symbol, &function));
  const loom_xdna_tile_coordinate_t sender =
      grant->distribution->sender.tile->coordinate;
  const uint32_t origin = sender.column << context->family->column_shift |
                          sender.row << context->family->row_shift;
  loom_value_id_t task;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_constant(code, &builder, grant->task, &task));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, grant->packet, grant->address - origin, &task, 1));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}
