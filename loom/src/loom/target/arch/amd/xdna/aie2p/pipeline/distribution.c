// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/distribution.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/storage_geometry.h"
#include "loom/ops/channel/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

iree_status_t loom_aie2p_native_select_channel_endpoint(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_invocation_t* invocation,
    const loom_aie2p_native_channel_t* channel,
    loom_xdna_dma_direction_t direction, uint32_t repetitions,
    uint32_t credit_interval, uint32_t reserved_tasks,
    const loom_aie2p_native_distribution_endpoint_t* shared,
    loom_aie2p_native_distribution_endpoint_t* out_endpoint,
    iree_string_view_t* out_reason, bool* out_valid) {
  *out_valid = false;
  loom_aie2p_native_tile_t* tile = context->pool_tiles[channel->pool_index];
  const loom_xdna_dma_facts_t* dma = &tile->facts->dma;
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  const bool sender = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM;
  const uint32_t batch_records = channel->replay.record_count;
  // Receiver lengths delimit their records independently of the sender's
  // packet boundary. A retained contiguous batch therefore needs one complete
  // transfer per replay. A single retained lifetime brackets the whole finite
  // sequence; autonomous repeated lifetimes use one credit per replay. A
  // retained input receiver publishes only after the complete batch arrives.
  const uint64_t transfer_bytes = (uint64_t)channel->record_byte_length *
                                  (batch_records ? batch_records : 1);
  if (!channel->record_dense ||
      (channel->source->capacity > 1 &&
       channel->byte_stride < channel->record_byte_length)) {
    *out_reason = IREE_SV("contiguous non-overlapping channel-copy records");
    return iree_ok_status();
  }
  const uint64_t length_units =
      transfer_bytes / dma->transfer_length_granularity;
  const uint64_t minimum_length_units =
      iree_max(1u, dma->transfer_length_offset);
  const uint64_t maximum_length_units =
      (uint64_t)dma->maximum_encoded_transfer_length +
      dma->transfer_length_offset;
  if (transfer_bytes % dma->transfer_length_granularity ||
      length_units < minimum_length_units ||
      length_units > maximum_length_units) {
    *out_reason =
        IREE_SV("a channel record length representable by its memory DMA");
    return iree_ok_status();
  }
  // A retained batch is one contiguous transfer. Single-slot transfers also
  // return to the same address, regardless of the logical record stride.
  const uint32_t capacity = batch_records ? 1 : channel->source->capacity;
  const uint32_t byte_stride = capacity > 1 ? channel->byte_stride : 0;
  const uint64_t step_units = byte_stride / dma->transfer_length_granularity;
  if (channel->byte_offset % dma->address_alignment ||
      byte_stride % dma->transfer_length_granularity ||
      capacity > (UINT64_C(1) << dma->iteration_bits) ||
      step_units > (UINT64_C(1) << dma->step_size_bits)) {
    *out_reason = IREE_SV(
        "DMA-aligned channel storage with a representable "
        "rotating slot stride");
    return iree_ok_status();
  }
  uint8_t* next_engine = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                             ? &tile->next_mm2s
                             : &tile->next_s2mm;
  const uint8_t engine = shared ? shared->engine : *next_engine;
  if (engine == dma->channel_count_per_direction) {
    *out_reason = IREE_SV("an available on-chip DMA engine");
    return iree_ok_status();
  }
  uint32_t words[8] = {0};
  words[0] = length_units - dma->transfer_length_offset;
  words[layout->lock_word] =
      loom_xdna_register_field_encode(
          layout->locks.acquire_id,
          layout->local_lock_base +
              (sender ? channel->ready_lock : channel->free_lock)) |
      loom_xdna_register_field_encode(layout->locks.acquire_value, -1) |
      loom_xdna_register_field_encode(layout->locks.acquire_enable, 1) |
      loom_xdna_register_field_encode(
          layout->locks.release_id,
          layout->local_lock_base +
              (sender ? channel->free_lock : channel->ready_lock)) |
      loom_xdna_register_field_encode(layout->locks.release_value, 1) |
      loom_xdna_register_field_encode(layout->locks.valid, 1);
  *out_endpoint = (loom_aie2p_native_distribution_endpoint_t){
      .channel = channel, .tile = tile, .engine = engine};
  if (!shared) {
    ++*next_engine;
  }
  const loom_pipeline_invocation_t* owner =
      invocations->channels[channel - context->channels].invocation;
  const uint32_t controller =
      sender ? channel->cursor.writer.worker : channel->cursor.reader.worker;
  const iree_host_size_t controller_engine =
      controller == UINT32_MAX ? IREE_HOST_SIZE_MAX
                               : context->core_engines[controller];
  const bool private_geometry =
      invocations->lifetimes.has_serial && owner == invocation &&
      invocation != invocations->entries[0] &&
      controller_engine != IREE_HOST_SIZE_MAX &&
      invocations->workers[controller].invocation == invocation;
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_count);
  loom_aie2p_native_endpoint_storage_t** ranges =
      &tile->descriptor_storage[next_descriptor - tile->descriptor_counts];
  if (private_geometry && reserved_tasks < dma->task_queue_depth) {
    loom_xdna_dma_facts_t available = *dma;
    available.task_queue_depth -= reserved_tasks;
    for (loom_aie2p_native_endpoint_storage_t* storage = *ranges; storage;
         storage = storage->next) {
      if (storage->last->ordinal >= invocation->ordinal ||
          storage->controller_engine != controller_engine) {
        continue;
      }
      const loom_aie2p_native_dma_sequence_t* first = &storage->first->sequence;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_build_sequence(
          &available, layout, first->descriptor, storage->descriptor_count,
          direction, batch_records ? repetitions / batch_records : repetitions,
          capacity, channel->byte_offset, byte_stride, credit_interval, words,
          context->pass->arena, &out_endpoint->sequence, out_reason,
          out_valid));
      if (*out_valid) {
        out_endpoint->storage = storage;
        storage->last = invocation;
        out_endpoint->next_geometry = storage->geometries;
        storage->geometries = out_endpoint;
        return iree_ok_status();
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_select_sequence(
      context, tile, out_endpoint->engine, direction,
      shared || batch_records ? NULL : channel,
      batch_records ? repetitions / batch_records : repetitions, capacity,
      channel->byte_offset, byte_stride, credit_interval, reserved_tasks, words,
      &out_endpoint->sequence, out_reason, out_valid));
  if (*out_valid && private_geometry) {
    loom_aie2p_native_endpoint_storage_t* storage;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        context->pass->arena, sizeof(*storage), (void**)&storage));
    loom_aie2p_native_endpoint_storage_t** position = ranges;
    while (*position && (*position)->descriptor_count <=
                            out_endpoint->sequence.descriptor_count) {
      position = &(*position)->next;
    }
    *storage = (loom_aie2p_native_endpoint_storage_t){
        .next = *position,
        .first = out_endpoint,
        .geometries = out_endpoint,
        .descriptor_count = out_endpoint->sequence.descriptor_count,
        .controller_engine = controller_engine,
        .last = invocation};
    *position = out_endpoint->storage = storage;
  }
  return iree_ok_status();
}

// A contiguous source range delivered to one immutable packet class.
typedef struct loom_aie2p_native_distribution_range_t {
  // First source record within the retained batch.
  uint32_t first;
  // Consecutive records coalesced into one DMA packet.
  uint32_t count;
  // Index in the distribution's retained packet-class table.
  uint8_t packet;
} loom_aie2p_native_distribution_range_t;

static bool loom_aie2p_native_distribution_destinations_equal(
    const loom_aie2p_native_distribution_packet_t* a,
    const loom_aie2p_native_distribution_packet_t* b) {
  if (a->destination_count != b->destination_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < a->destination_count; ++i) {
    bool found = false;
    for (iree_host_size_t j = 0; j < b->destination_count; ++j) {
      if (a->receivers[i] == b->receivers[j]) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

// Retains logical receiver correspondence before physical path selection.
// Existing phases keep their established order even when first use reverses.
static iree_status_t loom_aie2p_native_packet_receivers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_transport_batch_t* batch,
    const loom_aie2p_native_distribution_t* shared,
    loom_aie2p_native_distribution_t* distribution, uint32_t** out_totals,
    uint32_t** out_copy_receivers) {
  iree_arena_allocator_t* arena = context->pass->arena;
  const iree_host_size_t channel_count = realization->invocations.channel_count;
  uint32_t* totals;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, channel_count, sizeof(*totals), (void**)&totals));
  memset(totals, 0, channel_count * sizeof(*totals));
  if (shared) {
    distribution->receiver_ordinals = shared->receiver_ordinals;
  } else {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, channel_count, sizeof(*distribution->receiver_ordinals),
        (void**)&distribution->receiver_ordinals));
    memset(distribution->receiver_ordinals, UINT8_MAX,
           channel_count * sizeof(*distribution->receiver_ordinals));
  }
  uint32_t* ordinals = distribution->receiver_ordinals;
  uint32_t* copy_receivers;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, batch->copy_count,
                                                 sizeof(*copy_receivers),
                                                 (void**)&copy_receivers));
  for (iree_host_size_t i = 0; i < batch->copy_count; ++i) {
    const loom_pipeline_transport_copy_t* copy = &batch->copies[i];
    const iree_host_size_t index = loom_pipeline_invocations_channel_index(
        &realization->invocations, copy->destination->channel);
    if (!totals[index]) {
      if (shared && shared->receiver_ordinals[index] == UINT32_MAX) {
        return loom_aie2p_native_reject(
            context, copy->destination->copy,
            IREE_SV("the same packet destinations in every shared multicast "
                    "phase"));
      }
      if (!shared) {
        ordinals[index] = (uint32_t)distribution->receiver_count;
      }
      ++distribution->receiver_count;
    }
    totals[index] += (uint32_t)copy->executions;
    copy_receivers[i] = ordinals[index];
  }
  if (shared && shared->receiver_count != distribution->receiver_count) {
    return loom_aie2p_native_reject(
        context, batch->records->admission->op,
        IREE_SV(
            "the same packet destinations in every shared multicast phase"));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, distribution->receiver_count, sizeof(*distribution->receivers),
      (void**)&distribution->receivers));
  for (iree_host_size_t i = 0; i < channel_count; ++i) {
    if (totals[i]) {
      distribution->receivers[ordinals[i]] =
          (loom_aie2p_native_distribution_endpoint_t){
              .channel = &context->channels[i]};
    }
  }
  *out_totals = totals;
  *out_copy_receivers = copy_receivers;
  return iree_ok_status();
}

// A common admitted instruction engine proves complete child nonoverlap.
// Topology selects compatible wires; addresses never establish ordering.
static iree_host_size_t loom_aie2p_native_distribution_controller(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocations_t* invocations,
    const loom_aie2p_native_distribution_t* distribution,
    const loom_aie2p_native_channel_t* source) {
  const loom_pipeline_invocation_t* invocation = distribution->invocation;
  if (!invocations->lifetimes.has_serial ||
      invocation == invocations->entries[0]) {
    return IREE_HOST_SIZE_MAX;
  }
  const uint32_t controller = source->cursor.writer.worker;
  if (controller == UINT32_MAX ||
      invocations->workers[controller].invocation != invocation) {
    return IREE_HOST_SIZE_MAX;
  }
  for (iree_host_size_t i = 0; i <= distribution->receiver_count; ++i) {
    const loom_aie2p_native_channel_t* channel =
        i ? distribution->receivers[i - 1].channel : source;
    if (invocations->channels[channel - context->channels].invocation !=
        invocation) {
      return IREE_HOST_SIZE_MAX;
    }
  }
  return context->core_engines[controller];
}

static loom_aie2p_native_distribution_t*
loom_aie2p_native_reuse_distribution_path(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_tile_t* tile,
    const loom_aie2p_native_distribution_t* distribution,
    iree_host_size_t controller) {
  if (controller == IREE_HOST_SIZE_MAX) {
    return NULL;
  }
  for (loom_aie2p_native_distribution_path_t* path = tile->distribution_paths;
       path; path = path->next) {
    const loom_aie2p_native_distribution_t* first = path->first;
    if (path->controller_engine != controller ||
        path->last->ordinal >= distribution->invocation->ordinal ||
        (!first->repetitions) != (!distribution->repetitions) ||
        first->receiver_count != distribution->receiver_count) {
      continue;
    }
    iree_host_size_t i = 0;
    while (
        i < distribution->receiver_count &&
        first->receivers[i].tile ==
            context
                ->pool_tiles[distribution->receivers[i].channel->pool_index]) {
      ++i;
    }
    if (i == distribution->receiver_count) {
      return path->first;
    }
  }
  return NULL;
}

static bool loom_aie2p_native_distribution_program_equal(
    const loom_pipeline_transport_t* a_transport,
    const loom_pipeline_transport_batch_t* a,
    const loom_pipeline_transport_t* b_transport,
    const loom_pipeline_transport_batch_t* b) {
  if (a->record_count != b->record_count || a->copy_count != b->copy_count ||
      a->replays != b->replays) {
    return false;
  }
  for (iree_host_size_t i = 0; i < a->copy_count; ++i) {
    if (a->copies[i].source - a->records != b->copies[i].source - b->records ||
        a->copies[i].destination->channel !=
            b->copies[i].destination->channel ||
        a->copies[i].executions != b->copies[i].executions) {
      return false;
    }
    if (a->replays) {
      continue;
    }
    uint16_t a_scope = a->copies[i].scope;
    uint16_t b_scope = b->copies[i].scope;
    while (a_scope != a->records->admission_scope &&
           b_scope != b->records->admission_scope) {
      const loom_pipeline_transport_loop_t* a_loop =
          &a_transport->copy_loops[a_scope];
      const loom_pipeline_transport_loop_t* b_loop =
          &b_transport->copy_loops[b_scope];
      if (a_loop->begin - a->copies != b_loop->begin - b->copies ||
          a_loop->end - a->copies != b_loop->end - b->copies ||
          a_loop->iterations != b_loop->iterations) {
        return false;
      }
      a_scope = a_loop->parent;
      b_scope = b_loop->parent;
    }
    if (a_scope != a->records->admission_scope ||
        b_scope != b->records->admission_scope) {
      return false;
    }
  }
  return true;
}

// Uniform replay already has a compact order contract; only nonuniform copy
// programs need their retained loop interpreter. Both yield the same visits
// to packet selection without recovering or expanding source control flow.
static const loom_pipeline_transport_copy_t*
loom_aie2p_native_distribution_next_copy(
    const loom_pipeline_transport_batch_t* batch,
    loom_pipeline_transport_copy_cursor_t* cursor, uint64_t* remaining) {
  if (!batch->replays) {
    return loom_pipeline_transport_copy_cursor_next(cursor);
  }
  if (!*remaining) {
    return NULL;
  }
  const loom_pipeline_transport_copy_t* copy = cursor->next++;
  if (cursor->next == cursor->end) {
    --*remaining;
    cursor->next = batch->copies;
  }
  return copy;
}

// Selects a complete retained source lifetime with independent receiver counts.
// Physical packets coalesce adjacent records only when their destination sets
// agree; source admission and reclamation bracket the entire descriptor chain.
static iree_status_t loom_aie2p_native_select_packet_distribution(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_transport_t* transport,
    const loom_pipeline_transport_batch_t* batch,
    const loom_aie2p_native_channel_t* channel,
    loom_aie2p_native_distribution_t* path, uint32_t* next_source,
    loom_aie2p_native_distribution_t* distribution, const uint32_t* totals,
    const uint32_t* copy_receivers, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_transport_cohort_t* first = batch->records;
  if (!first->release) {
    return loom_aie2p_native_reject(
        context, first->admission->op,
        IREE_SV("a complete retained source lifetime per packet program"));
  }
  loom_aie2p_native_tile_t* tile = context->pool_tiles[channel->pool_index];
  const loom_xdna_dma_facts_t* dma = &tile->facts->dma;
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  const uint8_t engine = path ? path->sender.engine : tile->next_mm2s;
  uint16_t descriptor_base, descriptor_capacity;
  uint16_t* used = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_capacity);
  if (engine == dma->channel_count_per_direction ||
      *used == descriptor_capacity) {
    return loom_aie2p_native_reject(
        context, first->admission->op,
        IREE_SV("an available on-chip DMA engine and descriptor"));
  }
  const uint64_t maximum_bytes =
      ((uint64_t)dma->maximum_encoded_transfer_length +
       dma->transfer_length_offset) *
      dma->transfer_length_granularity;
  if (channel->record_byte_length % dma->transfer_length_granularity ||
      channel->record_byte_length > maximum_bytes ||
      channel->byte_offset % dma->address_alignment) {
    return loom_aie2p_native_reject(
        context, first->admission->op,
        IREE_SV(
            "DMA-aligned retained records with representable packet lengths"));
  }
  const uint32_t maximum_records = maximum_bytes / channel->record_byte_length;
  const uint32_t task_count = (uint32_t)iree_host_size_ceil_div(
      first->admissions, dma->maximum_task_repeat_count);
  if (task_count > dma->task_queue_depth) {
    return loom_aie2p_native_reject(
        context, first->admission->op,
        IREE_SV(
            "all retained packet lifetimes within the finite DMA task queue"));
  }
  iree_arena_allocator_t* arena = context->pass->arena;
  loom_aie2p_native_distribution_path_t* packet_path = distribution->path;
  const iree_host_size_t channel_count = realization->invocations.channel_count;
  const uint32_t* ordinals = distribution->receiver_ordinals;
  distribution->sender =
      (loom_aie2p_native_distribution_endpoint_t){.channel = channel,
                                                  .tile = tile,
                                                  .engine = engine,
                                                  .wait_for_idle = true};
  if (!path) {
    ++tile->next_mm2s;
  }
  for (iree_host_size_t i = 0; i < channel_count; ++i) {
    if (!totals[i]) {
      continue;
    }
    const uint32_t ordinal = ordinals[i];
    const loom_aie2p_native_channel_t* receiver_channel = &context->channels[i];
    if (((path || transport->batch_count > 1) &&
         totals[i] % receiver_channel->source->capacity) ||
        (path && path->receivers[ordinal].sequence.terminal_slot)) {
      return loom_aie2p_native_reject(
          context, first->admission->op,
          IREE_SV("complete destination-ring turns on the same packet path"));
    }
    iree_string_view_t reason;
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channel_endpoint(
        context, &realization->invocations, distribution->invocation,
        receiver_channel, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY, totals[i],
        1, 0, path ? &path->receivers[ordinal] : NULL,
        &distribution->receivers[ordinal], &reason, &valid));
    if (!valid) {
      return loom_aie2p_native_reject(context, first->admission->op, reason);
    }
    loom_aie2p_native_distribution_endpoint_t* receiver =
        &distribution->receivers[ordinal];
    receiver->wait_for_idle = true;
    if (tile == receiver->tile &&
        (engine >= dma->loopback_channel_count || engine != receiver->engine)) {
      return loom_aie2p_native_reject(
          context, first->admission->op,
          IREE_SV("matching same-tile DMA channels"));
    }
  }
  // Packet identities belong to the physical path, so every phase can select
  // a different subset without reprogramming filters or renumbering earlier
  // descriptors. Filter/arbiter admission enforces the smaller switch limits.
  if (!path) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, 32, sizeof(*packet_path->packets),
                                  (void**)&packet_path->packets));
    packet_path->source_route = (*next_source)++;
  }
  const uint8_t first_packet = packet_path->packet_count;
  loom_aie2p_native_distribution_range_t* ranges;
  const uint16_t available = descriptor_capacity - *used;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, available, sizeof(*ranges), (void**)&ranges));
  uint32_t* visit_receivers;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, distribution->receiver_count, sizeof(*visit_receivers),
      (void**)&visit_receivers));
  uint16_t range_count = 0;
  loom_pipeline_transport_copy_cursor_t cursor = {
      .next = batch->copies, .end = batch->copies + batch->copy_count};
  uint64_t remaining = batch->replays;
  if (!batch->replays) {
    IREE_RETURN_IF_ERROR(loom_pipeline_transport_copy_cursor_initialize(
        transport, batch, arena, &cursor));
  }
  const loom_pipeline_transport_copy_t* copy =
      loom_aie2p_native_distribution_next_copy(batch, &cursor, &remaining);
  while (copy) {
    loom_aie2p_native_distribution_packet_t visit = {
        .destinations = copy,
        .receivers = visit_receivers,
        .destination_count = 1};
    visit_receivers[0] = copy_receivers[copy - batch->copies];
    const loom_pipeline_transport_copy_t* next =
        loom_aie2p_native_distribution_next_copy(batch, &cursor, &remaining);
    while (next && next == copy + visit.destination_count &&
           next->source == copy->source && next->scope == copy->scope) {
      bool repeated = false;
      const uint32_t receiver = copy_receivers[next - batch->copies];
      for (iree_host_size_t k = 0; k < visit.destination_count; ++k) {
        repeated |= visit_receivers[k] == receiver;
      }
      if (repeated) {
        break;
      }
      visit_receivers[visit.destination_count++] = receiver;
      next =
          loom_aie2p_native_distribution_next_copy(batch, &cursor, &remaining);
    }
    uint8_t packet = 0;
    for (; packet < packet_path->packet_count; ++packet) {
      if (loom_aie2p_native_distribution_destinations_equal(
              &packet_path->packets[packet], &visit)) {
        break;
      }
    }
    if (packet == packet_path->packet_count) {
      if (packet == 32) {
        return loom_aie2p_native_reject(
            context, copy->destination->copy,
            IREE_SV("packet destination classes within the hardware ID range"));
      }
      uint32_t* receivers;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, visit.destination_count,
                                    sizeof(*receivers), (void**)&receivers));
      memcpy(receivers, visit_receivers,
             visit.destination_count * sizeof(*receivers));
      visit.receivers = receivers;
      packet_path->packets[packet_path->packet_count++] = visit;
    }
    const uint32_t record = (uint32_t)(copy->source - batch->records);
    if (range_count && ranges[range_count - 1].packet == packet &&
        ranges[range_count - 1].first + ranges[range_count - 1].count ==
            record &&
        ranges[range_count - 1].count < maximum_records) {
      ++ranges[range_count - 1].count;
    } else {
      if (range_count == available) {
        return loom_aie2p_native_reject(
            context, copy->destination->copy,
            IREE_SV("a complete ordered packet program within its DMA "
                    "descriptor bank"));
      }
      ranges[range_count++] = (loom_aie2p_native_distribution_range_t){
          .first = record, .count = 1, .packet = packet};
    }
    copy = next;
  }
  for (uint8_t i = first_packet; i < packet_path->packet_count; ++i) {
    const loom_aie2p_native_distribution_packet_t* packet =
        &packet_path->packets[i];
    for (iree_host_size_t j = 0; j < packet->destination_count; ++j) {
      const loom_pipeline_transport_destination_t* destination =
          packet->destinations[j].destination;
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &distribution->receivers[packet->receivers[j]];
      const iree_host_size_t route_first = context->routing.route_count;
      if (!loom_aie2p_array_route_workers(
              &context->routing, (*next_source)++, packet_path->source_route,
              tile->coordinate, engine, receiver->tile->coordinate,
              receiver->engine, true)) {
        return loom_aie2p_native_reject(
            context, destination->copy,
            IREE_SV("available on-die packet multicast links"));
      }
      loom_aie2p_native_route_label(context, route_first, i, UINT32_MAX);
    }
  }
  loom_aie2p_native_dma_sequence_t* sequence = &distribution->sender.sequence;
  *sequence =
      (loom_aie2p_native_dma_sequence_t){.descriptor = descriptor_base + *used,
                                         .descriptor_count = range_count,
                                         .task_count = (uint8_t)task_count};
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena,
                          range_count * sizeof(*sequence->words) +
                              task_count * sizeof(*sequence->tasks),
                          (void**)&sequence->words));
  sequence->tasks = (uint32_t*)(sequence->words + range_count);
  uint32_t admissions = (uint32_t)first->admissions;
  for (uint32_t i = 0; i < task_count; ++i) {
    const uint32_t count = iree_min(admissions, dma->maximum_task_repeat_count);
    sequence->tasks[i] =
        loom_xdna_register_field_encode(
            layout->directions[LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM - 1]
                .start,
            sequence->descriptor) |
        loom_xdna_register_field_encode(
            layout->directions[LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM - 1]
                .repeat,
            count - 1);
    admissions -= count;
  }
  for (uint16_t i = 0; i < range_count; ++i) {
    uint32_t* words = sequence->words[i];
    memset(words, 0, sizeof(*sequence->words));
    const loom_aie2p_native_distribution_range_t* range = &ranges[i];
    words[0] = range->count * channel->record_byte_length /
                   dma->transfer_length_granularity -
               dma->transfer_length_offset;
    words[layout->address_word] |= loom_xdna_register_field_encode(
        layout->address, (layout->local_address_base + channel->byte_offset +
                          range->first * channel->byte_stride) >>
                             dma->address_encoding_shift);
    words[layout->packet.word] |=
        loom_xdna_register_field_encode(layout->packet.enable, 1) |
        loom_xdna_register_field_encode(layout->packet.id, range->packet);
    words[layout->lock_word] |=
        loom_xdna_register_field_encode(layout->locks.valid, 1);
    if (!i) {
      words[layout->lock_word] |=
          loom_xdna_register_field_encode(
              layout->locks.acquire_id,
              layout->local_lock_base + channel->ready_lock) |
          loom_xdna_register_field_encode(layout->locks.acquire_value, -1) |
          loom_xdna_register_field_encode(layout->locks.acquire_enable, 1);
    }
    if (i + 1 == range_count) {
      words[layout->lock_word] |=
          loom_xdna_register_field_encode(
              layout->locks.release_id,
              layout->local_lock_base + channel->free_lock) |
          loom_xdna_register_field_encode(layout->locks.release_value, 1);
    } else {
      words[layout->chain.word] |=
          loom_xdna_register_field_encode(layout->chain.next,
                                          sequence->descriptor + i + 1) |
          loom_xdna_register_field_encode(layout->chain.enable, 1);
    }
  }
  *used += range_count;
  *out_valid = true;
  return iree_ok_status();
}

// Circuit geometry owns the destination storage and semaphore identities as
// well as its transfer count. Ordered phases can share a wire without sharing
// those descriptors.
static bool loom_aie2p_native_distribution_geometry_equal(
    const loom_aie2p_native_distribution_t* distribution,
    const loom_pipeline_transport_cohort_t* cohort, uint32_t repetitions) {
  if (distribution->repetitions != repetitions ||
      distribution->receiver_count != cohort->destination_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < cohort->destination_count; ++i) {
    if (distribution->source->records->destinations[i].channel !=
        cohort->destinations[i].channel) {
      return false;
    }
  }
  return true;
}

loom_aie2p_native_distribution_t* loom_aie2p_native_reusable_distribution(
    const loom_aie2p_native_worker_t* owner,
    const loom_aie2p_native_channel_t* source,
    const loom_pipeline_transport_t* transport,
    const loom_pipeline_transport_batch_t* batch, uint32_t repetitions) {
  for (loom_aie2p_native_distribution_t* candidate = owner->distributions;
       candidate; candidate = candidate->next) {
    if (candidate->sender.channel == source &&
        (owner->distribution_packets
             ? loom_aie2p_native_distribution_program_equal(
                   candidate->transport, candidate->source, transport, batch)
             : loom_aie2p_native_distribution_geometry_equal(
                   candidate, batch->records, repetitions))) {
      return candidate;
    }
  }
  return NULL;
}

static const loom_aie2p_native_channel_t*
loom_aie2p_native_distribution_phase_channel(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_transport_cohort_t* cohort, iree_host_size_t endpoint) {
  return loom_aie2p_native_channel(
      context, realization,
      endpoint ? cohort->destinations[endpoint - 1].channel
               : cohort->admission->channel);
}

static iree_status_t loom_aie2p_native_distribution_windows(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_transport_t* transport, uint32_t** out_windows) {
  *out_windows = NULL;
  if (!transport->prefix_cycles) {
    return iree_ok_status();
  }
  const iree_host_size_t tail = transport->batch_count - 1;
  const loom_pipeline_transport_cohort_t* bulk =
      transport->batches[tail].records;
  for (iree_host_size_t i = 0; i < tail; ++i) {
    if (transport->batches[i].records->destination_count !=
        bulk->destination_count) {
      return iree_ok_status();
    }
  }
  uint32_t* windows;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, bulk->destination_count + 1, sizeof(*windows),
      (void**)&windows));
  for (iree_host_size_t i = 0; i <= bulk->destination_count; ++i) {
    const loom_aie2p_native_channel_t* channel =
        loom_aie2p_native_distribution_phase_channel(context, realization, bulk,
                                                     i);
    windows[i] = loom_aie2p_native_phased_window(
        context, channel, tail, (uint32_t)transport->prefix_cycles,
        (uint32_t)(bulk->admissions / transport->prefix_cycles));
    if (!windows[i]) {
      return iree_ok_status();
    }
    for (iree_host_size_t j = 0; j < tail; ++j) {
      if (loom_aie2p_native_distribution_phase_channel(
              context, realization, transport->batches[j].records, i)
              ->pool_index != channel->pool_index) {
        return iree_ok_status();
      }
    }
  }
  *out_windows = windows;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_distribution_refills(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_transport_t* transport, const uint32_t* windows,
    const loom_aie2p_native_distribution_t* first,
    const loom_aie2p_native_distribution_t* bulk) {
  for (iree_host_size_t i = 0; i <= bulk->receiver_count; ++i) {
    if (bulk->repetitions <= windows[i]) {
      continue;
    }
    const loom_aie2p_native_distribution_endpoint_t* endpoint =
        i ? &bulk->receivers[i - 1] : &bulk->sender;
    loom_aie2p_native_refill_queue_t* queue;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_add_refill(
        context, endpoint->tile, endpoint->engine,
        i ? LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY
          : LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
        endpoint->channel, bulk->repetitions, windows[i], &endpoint->sequence,
        &queue));
    if (transport->prefix_cycles == 1) {
      continue;
    }
    const iree_host_size_t count = transport->batch_count - 1;
    uint32_t* prefixes;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, count, sizeof(*prefixes), (void**)&prefixes));
    const loom_aie2p_native_distribution_t* phase = first;
    for (iree_host_size_t j = 0; j < count; ++j, phase = phase->next) {
      const loom_aie2p_native_distribution_endpoint_t* prefix =
          i ? &phase->receivers[i - 1] : &phase->sender;
      prefixes[j] = prefix->sequence.tasks[0];
    }
    queue->prefixes = prefixes;
    queue->prefix_count = (uint32_t)count;
    queue->cycle_windows =
        (uint32_t)(bulk->repetitions / transport->prefix_cycles) / windows[i];
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_select_worker_distributions(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, uint32_t* next_source, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_worker_t* source = &realization->workers[worker_index];
  loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  const loom_pipeline_invocation_t* invocation =
      realization->invocations.workers[worker_index].invocation;
  const bool child = invocation->ordinal != 0;
  loom_aie2p_native_worker_t* owner =
      &context
           ->workers[context->distribution_execution
                         .workers[context->distribution_engines[worker_index]]];
  loom_aie2p_native_distribution_t** tail = &owner->distributions;
  while (*tail) {
    tail = &(*tail)->next;
  }
  loom_aie2p_native_distribution_phase_t** phase_tail =
      &worker->distribution_phases;
  loom_aie2p_native_distribution_phase_t* previous_phase = NULL;
  // A cursor's retained total exceeds one batch's contribution precisely when
  // this worker uses that endpoint in more than one phase. Independent
  // distributions retain their autonomous paths.
  const bool packet_program = owner->distribution_packets;
  bool ordered = child || (packet_program && source->transport.batch_count > 1);
  for (iree_host_size_t i = 0; i < source->transport.batch_count; ++i) {
    const loom_pipeline_transport_batch_t* batch =
        &source->transport.batches[i];
    const loom_pipeline_transport_cohort_t* cohort = batch->records;
    const loom_aie2p_native_channel_t* source_channel =
        loom_aie2p_native_channel(context, realization,
                                  cohort->admission->channel);
    ordered |= source_channel->cursor.reader.maximum_admissions >
               cohort->admissions * batch->record_count;
    for (iree_host_size_t j = 0;
         !packet_program && j < cohort->destination_count; ++j) {
      const loom_aie2p_native_channel_t* destination =
          loom_aie2p_native_channel(context, realization,
                                    cohort->destinations[j].channel);
      ordered |= destination->cursor.writer.maximum_admissions >
                 cohort->transfers * batch->record_count;
    }
  }
  uint32_t* queued_tasks = NULL;
  if (child && !packet_program) {
    const iree_host_size_t endpoint_count =
        source->transport.batches[0].records->destination_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, endpoint_count, sizeof(*queued_tasks),
        (void**)&queued_tasks));
    memset(queued_tasks, 0, endpoint_count * sizeof(*queued_tasks));
  }
  uint32_t* windows = NULL;
  if (!packet_program && (!ordered || child)) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_distribution_windows(
        context, realization, &source->transport, &windows));
  }
  loom_aie2p_native_distribution_t* first_phase = NULL;
  for (iree_host_size_t i = 0; i < source->transport.batch_count; ++i) {
    const loom_pipeline_transport_batch_t* batch =
        &source->transport.batches[i];
    const loom_pipeline_transport_cohort_t* cohort = batch->records;
    const loom_aie2p_native_channel_t* source_channel =
        loom_aie2p_native_channel(context, realization,
                                  cohort->admission->channel);
    const uint32_t repetitions =
        (uint32_t)(cohort->transfers * batch->record_count);
    loom_aie2p_native_distribution_t* path = ordered   ? owner->distributions
                                             : windows ? first_phase
                                                       : NULL;
    if (ordered) {
      const uint32_t producer = source_channel->cursor.writer.worker;
      if (producer == UINT32_MAX ||
          (!child &&
           (!cohort->release || cohort->admissions != 1 ||
            (path && producer != path->sender.channel->cursor.writer.worker) ||
            context->workers[producer].execution !=
                LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION))) {
        return loom_aie2p_native_reject(
            context, cohort->admission->op,
            IREE_SV("successive retained batches with one controller-owned "
                    "publication per phase"));
      }
      const loom_pipeline_transport_cohort_t* first =
          source->transport.batches[0].records;
      if ((!packet_program &&
           (cohort->destination_count != first->destination_count ||
            (path && cohort->destination_count != path->receiver_count))) ||
          (path && context->pool_tiles[source_channel->pool_index] !=
                       path->sender.tile)) {
        return loom_aie2p_native_reject(
            context, cohort->admission->op,
            IREE_SV("one source memory tile and destination set for an "
                    "ordered multicast path"));
      }
    }
    // Geometry variants are bounded by the physical descriptor bank. A later
    // phase can reuse one only after its prior endpoints have retired.
    loom_aie2p_native_distribution_t* distribution =
        ordered
            ? loom_aie2p_native_reusable_distribution(
                  owner, source_channel, &source->transport, batch, repetitions)
            : NULL;
    if (!distribution) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*distribution), (void**)&distribution));
      *distribution = (loom_aie2p_native_distribution_t){
          .invocation = invocation,
          .source = batch,
          .transport = &source->transport,
          .receiver_count = packet_program ? 0 : cohort->destination_count,
          .repetitions = packet_program ? 0 : repetitions,
          .ordered = ordered,
      };
      uint32_t* totals = NULL;
      uint32_t* copy_receivers = NULL;
      if (packet_program) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_packet_receivers(
            context, realization, batch, path, distribution, &totals,
            &copy_receivers));
      } else {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            context->pass->arena, cohort->destination_count,
            sizeof(*distribution->receivers),
            (void**)&distribution->receivers));
        for (iree_host_size_t j = 0; j < cohort->destination_count; ++j) {
          distribution->receivers[j] =
              (loom_aie2p_native_distribution_endpoint_t){
                  .channel = loom_aie2p_native_channel(
                      context, realization, cohort->destinations[j].channel)};
        }
      }
      const iree_host_size_t controller =
          loom_aie2p_native_distribution_controller(
              context, &realization->invocations, distribution, source_channel);
      loom_aie2p_native_tile_t* source_tile =
          context->pool_tiles[source_channel->pool_index];
      if (!path) {
        path = loom_aie2p_native_reuse_distribution_path(
            context, source_tile, distribution, controller);
      }
      if (path) {
        distribution->path = path->path;
        if (distribution->path->controller_engine != controller) {
          distribution->path->controller_engine = IREE_HOST_SIZE_MAX;
        }
        if (distribution->path->last->ordinal < invocation->ordinal) {
          distribution->path->last = invocation;
        }
      } else {
        IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                                 sizeof(*distribution->path),
                                                 (void**)&distribution->path));
        *distribution->path = (loom_aie2p_native_distribution_path_t){
            .first = distribution,
            .last = invocation,
            .controller_engine = controller};
        // Preserve source-engine order so equivalent fresh streams select the
        // same queues instead of exchanging them on each child occurrence.
        loom_aie2p_native_distribution_path_t** position =
            &source_tile->distribution_paths;
        while (*position) {
          position = &(*position)->next;
        }
        *position = distribution->path;
      }
      distribution->next_geometry = distribution->path->geometries;
      distribution->path->geometries = distribution;
      if (packet_program) {
        bool valid = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_packet_distribution(
            context, realization, &source->transport, batch, source_channel,
            path, next_source, distribution, totals, copy_receivers, &valid));
        if (!valid) {
          return iree_ok_status();
        }
      } else {
        iree_string_view_t reason;
        bool valid = false;
        const bool bulk = i + 1 == source->transport.batch_count;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channel_endpoint(
            context, &realization->invocations, invocation, source_channel,
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
            windows ? (bulk ? windows[0] : 1) : distribution->repetitions,
            (uint32_t)(cohort->admissions == 1 ? batch->replays : 1),
            child ? queued_tasks[0] + source->transport.batch_count - i - 1 : 0,
            path ? &path->sender : NULL, &distribution->sender, &reason,
            &valid));
        if (!valid) {
          return loom_aie2p_native_reject(context, cohort->admission->op,
                                          reason);
        }
        const uint32_t source_route = path ? 0 : (*next_source)++;
        const iree_host_size_t route_begin = context->routing.route_count;
        for (iree_host_size_t j = 0; j < cohort->destination_count; ++j) {
          const loom_pipeline_transport_destination_t* destination =
              &cohort->destinations[j];
          const loom_aie2p_native_channel_t* destination_channel =
              loom_aie2p_native_channel(context, realization,
                                        destination->channel);
          if (path && context->pool_tiles[destination_channel->pool_index] !=
                          path->receivers[j].tile) {
            return loom_aie2p_native_reject(
                context, destination->copy,
                IREE_SV("the same ordered physical destinations in every "
                        "shared multicast phase"));
          }
          if (ordered && repetitions % destination_channel->source->capacity) {
            return loom_aie2p_native_reject(
                context, destination->copy,
                IREE_SV("complete destination-ring turns at a shared multicast "
                        "phase boundary"));
          }
          IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channel_endpoint(
              context, &realization->invocations, invocation,
              destination_channel, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
              windows ? (bulk ? windows[j + 1] : 1) : distribution->repetitions,
              1,
              child
                  ? queued_tasks[j + 1] + source->transport.batch_count - i - 1
                  : 0,
              path ? &path->receivers[j] : NULL, &distribution->receivers[j],
              &reason, &valid));
          if (!valid) {
            return loom_aie2p_native_reject(context, destination->copy, reason);
          }
          const loom_aie2p_native_distribution_endpoint_t* sender =
              &distribution->sender;
          loom_aie2p_native_distribution_endpoint_t* receiver =
              &distribution->receivers[j];
          receiver->wait_for_idle =
              ordered || windows ||
              !loom_aie2p_native_cursor_has_read_completion(
                  context, &destination_channel->cursor.reader,
                  distribution->repetitions);
          distribution->sender.wait_for_idle |= receiver->wait_for_idle;
          if (sender->tile == receiver->tile &&
              (sender->engine >=
                   sender->tile->facts->dma.loopback_channel_count ||
               sender->engine != receiver->engine)) {
            return loom_aie2p_native_reject(
                context, destination->copy,
                IREE_SV(
                    "matching DMA channel indices for same-tile channel copy"));
          }
          if (path) {
            continue;
          }
          const iree_host_size_t route_first = context->routing.route_count;
          const uint32_t route = (*next_source)++;
          if (!loom_aie2p_array_route_workers(
                  &context->routing, route, source_route,
                  distribution->sender.tile->coordinate,
                  distribution->sender.engine,
                  distribution->receivers[j].tile->coordinate,
                  distribution->receivers[j].engine, false)) {
            return loom_aie2p_native_reject(
                context, destination->copy,
                IREE_SV("available on-die multicast payload links"));
          }
          loom_aie2p_native_route_label(context, route_first, UINT8_MAX,
                                        UINT32_MAX);
        }
        if (!path) {
          distribution->path->routes.first = route_begin;
          distribution->path->routes.count =
              context->routing.route_count - route_begin;
        }
      }
      *tail = distribution;
      tail = &distribution->next;
    }
    if (queued_tasks) {
      for (iree_host_size_t j = 0; j <= distribution->receiver_count; ++j) {
        const loom_aie2p_native_distribution_endpoint_t* endpoint =
            j ? &distribution->receivers[j - 1] : &distribution->sender;
        queued_tasks[j] += endpoint->sequence.task_count;
        // Reused geometry may have been selected for a less occupied queue.
        if (queued_tasks[j] > endpoint->tile->facts->dma.task_queue_depth) {
          return loom_aie2p_native_reject(
              context, cohort->admission->op,
              IREE_SV("complete child phase tasks within DMA queue capacity"));
        }
      }
    }
    if (windows) {
      if (!first_phase) {
        first_phase = distribution;
      }
      if (i + 1 == source->transport.batch_count) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_distribution_refills(
            context, &source->transport, windows, first_phase, distribution));
      }
    }
    if (ordered) {
      loom_aie2p_native_distribution_phase_t* phase;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                               sizeof(*phase), (void**)&phase));
      *phase = (loom_aie2p_native_distribution_phase_t){
          .previous = previous_phase,
          .source = batch,
          .distribution = distribution};
      *phase_tail = phase;
      phase_tail = &phase->next;
      previous_phase = phase;
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

// The controller and distributor retain the same sequence of complete FIFO
// turns. Bind those existing ownership events once; emission uses the selected
// phase directly and never searches source code or infers cross-worker order.
iree_status_t loom_aie2p_native_bind_distribution_phases(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  const iree_host_size_t count = realization->invocations.worker_count;
  const iree_host_size_t controller_count =
      realization->invocations.count * context->execution.engine_count;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, controller_count,
                                sizeof(*context->distribution_controllers),
                                (void**)&context->distribution_controllers));
  for (iree_host_size_t i = 0; i < controller_count; ++i) {
    context->distribution_controllers[i] = IREE_HOST_SIZE_MAX;
  }
  const loom_aie2p_native_distribution_phase_t** next;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->pass->arena, count,
                                                 sizeof(*next), (void**)&next));
  loom_aie2p_native_distribution_command_t*** command_tails;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->pass->arena, count,
                                                 sizeof(*command_tails),
                                                 (void**)&command_tails));
  for (iree_host_size_t i = 0; i < count; ++i) {
    command_tails[i] = &context->workers[i].distribution_commands;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    next[i] = context->workers[i].distribution_phases;
    const loom_pipeline_invocation_t* invocation =
        realization->invocations.workers[i].invocation;
    if (!invocation->ordinal || !next[i]) {
      continue;
    }
    for (const loom_aie2p_native_distribution_phase_t* phase = next[i]; phase;
         phase = phase->next) {
      loom_aie2p_native_distribution_t* distribution = phase->distribution;
      // Preserve phase order in every shared queue while publishing receiver
      // tasks before the corresponding sender, as for a single child phase.
      for (iree_host_size_t remaining = distribution->receiver_count + 1;
           remaining; --remaining) {
        const iree_host_size_t j = remaining - 1;
        const bool sender = j == 0;
        loom_aie2p_native_distribution_endpoint_t* endpoint =
            sender ? &distribution->sender : &distribution->receivers[j - 1];
        uint32_t cursor = sender ? endpoint->channel->cursor.writer.worker
                                 : endpoint->channel->cursor.reader.worker;
        if (cursor != UINT32_MAX && context->workers[cursor].input) {
          cursor = context->workers[cursor].input->controller;
        }
        iree_host_size_t engine = cursor == UINT32_MAX
                                      ? IREE_HOST_SIZE_MAX
                                      : context->core_engines[cursor];
        // An autonomous distributor has no instruction cursor owner. Its
        // compute-memory endpoint is controlled by that tile's core.
        if (engine == IREE_HOST_SIZE_MAX &&
            endpoint->tile->facts->kind == LOOM_XDNA_TILE_KIND_COMPUTE) {
          engine = (iree_host_size_t)(endpoint->tile - context->tiles);
        }
        iree_host_size_t controller =
            engine == IREE_HOST_SIZE_MAX
                ? IREE_HOST_SIZE_MAX
                : context->execution
                      .workers[invocation->ordinal *
                                   context->execution.engine_count +
                               engine];
        if (controller == IREE_HOST_SIZE_MAX && engine != IREE_HOST_SIZE_MAX &&
            invocation->schedule) {
          controller =
              context->execution.workers[invocation->parent->ordinal *
                                             context->execution.engine_count +
                                         engine];
          if (controller != IREE_HOST_SIZE_MAX &&
              realization->invocations.workers[controller].invocation !=
                  invocation->parent) {
            controller = IREE_HOST_SIZE_MAX;
          }
        }
        // Nested phase controllers also execute the child's outer boundary.
        if (controller == IREE_HOST_SIZE_MAX) {
          return loom_aie2p_native_reject(
              context, phase->source->records->admission->op,
              IREE_SV("a producer and each receiver controlled by the complete "
                      "child or its concurrent parent invocation"));
        }
        endpoint->controller_engine = engine;
        context->distribution_controllers[invocation->ordinal *
                                              context->execution.engine_count +
                                          engine] = controller;
        loom_aie2p_native_distribution_command_t* command;
        IREE_RETURN_IF_ERROR(iree_arena_allocate(
            context->pass->arena, sizeof(*command), (void**)&command));
        *command = (loom_aie2p_native_distribution_command_t){
            .invocation = invocation,
            .endpoint = endpoint,
            .direction = sender ? LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                                : LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
            .retire = phase->next == NULL};
        *command_tails[controller] = command;
        command_tails[controller] = &command->next;
      }
    }
    next[i] = NULL;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION) {
      continue;
    }
    const loom_pipeline_worker_t* source = &realization->workers[i];
    for (iree_host_size_t j = 0; j < source->transport.count; ++j) {
      const loom_pipeline_transport_step_t* step = &source->transport.steps[j];
      if (step->kind != LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL ||
          step->source.channel->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_PUBLISH) {
        continue;
      }
      loom_aie2p_native_configuration_action_t* action =
          &worker->configuration_actions[step->source.channel -
                                         source->channels.actions];
      const uint32_t reader = action->channel->cursor.reader.worker;
      if (action->value < 0 || reader == UINT32_MAX ||
          !context->workers[reader].distribution_phases) {
        continue;
      }
      const loom_aie2p_native_distribution_phase_t* phase = next[reader];
      if (!phase || phase->distribution->sender.channel != action->channel) {
        return loom_aie2p_native_reject(
            context, step->op,
            IREE_SV("controller publications in the shared multicast phase "
                    "order"));
      }
      action->distribution_phase = phase;
      next[reader] = phase->next;
    }
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (next[i]) {
      return loom_aie2p_native_reject(
          context, next[i]->source->records->admission->op,
          IREE_SV(
              "a complete controller publication for each multicast phase"));
    }
  }
  *out_valid = true;
  return iree_ok_status();
}
