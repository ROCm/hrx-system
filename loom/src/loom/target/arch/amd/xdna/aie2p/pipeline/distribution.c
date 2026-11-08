// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/analysis/storage_geometry.h"
#include "loom/ops/channel/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

static iree_status_t loom_aie2p_native_select_distribution_endpoint(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel,
    loom_xdna_dma_direction_t direction, uint32_t repetitions,
    uint32_t credit_interval,
    const loom_aie2p_native_distribution_endpoint_t* shared,
    loom_aie2p_native_distribution_endpoint_t* out_endpoint,
    iree_string_view_t* out_reason, bool* out_valid) {
  *out_valid = false;
  loom_aie2p_native_tile_t* tile = context->pool_tiles[channel->pool_index];
  const loom_xdna_dma_facts_t* dma = &tile->facts->dma;
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  const bool sender = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM;
  const uint32_t batch_records = sender ? channel->replay.record_count : 0;
  // Receiver lengths delimit their records independently of the sender's
  // packet boundary. A retained contiguous batch therefore needs one complete
  // transfer per replay. A single retained lifetime brackets the whole finite
  // sequence; autonomous repeated lifetimes use one credit per replay.
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
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_count);
  if (engine == dma->channel_count_per_direction ||
      *next_descriptor == descriptor_count) {
    *out_reason = IREE_SV("an available on-chip DMA engine and descriptor");
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
  return loom_aie2p_native_select_sequence(
      context, tile, out_endpoint->engine, direction,
      shared || batch_records ? NULL : channel,
      batch_records ? repetitions / batch_records : repetitions, capacity,
      channel->byte_offset, byte_stride, credit_interval, words,
      &out_endpoint->sequence, out_reason, out_valid);
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
      if (a->destinations[i].destination->channel ==
          b->destinations[j].destination->channel) {
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
    loom_aie2p_native_distribution_t* distribution, bool* out_valid) {
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
  loom_aie2p_native_distribution_t* packet_path = path ? path : distribution;
  uint32_t* totals;
  const iree_host_size_t channel_count = realization->invocations.channel_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, channel_count, sizeof(*totals), (void**)&totals));
  if (!path) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, channel_count, sizeof(*packet_path->receiver_ordinals),
        (void**)&packet_path->receiver_ordinals));
    memset(packet_path->receiver_ordinals, UINT8_MAX,
           channel_count * sizeof(*packet_path->receiver_ordinals));
  }
  uint32_t* ordinals = packet_path->receiver_ordinals;
  memset(totals, 0, channel_count * sizeof(*totals));
  for (iree_host_size_t i = 0; i < batch->copy_count; ++i) {
    const loom_pipeline_transport_copy_t* copy = &batch->copies[i];
    const iree_host_size_t index = loom_pipeline_invocations_channel_index(
        &realization->invocations, copy->destination->channel);
    if (!totals[index]) {
      if (path && ordinals[index] == UINT32_MAX) {
        return loom_aie2p_native_reject(
            context, copy->destination->copy,
            IREE_SV("the same packet destinations in every shared multicast "
                    "phase"));
      }
      if (!path) {
        ordinals[index] = (uint32_t)distribution->receiver_count;
      }
      ++distribution->receiver_count;
    }
    totals[index] += (uint32_t)copy->executions;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, distribution->receiver_count, sizeof(*distribution->receivers),
      (void**)&distribution->receivers));
  if (path && path->receiver_count != distribution->receiver_count) {
    return loom_aie2p_native_reject(
        context, first->admission->op,
        IREE_SV(
            "the same packet destinations in every shared multicast phase"));
  }
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
    if ((path || transport->batch_count > 1) &&
        totals[i] % receiver_channel->source->capacity) {
      return loom_aie2p_native_reject(
          context, first->admission->op,
          IREE_SV("complete destination-ring turns on the same packet path"));
    }
    iree_string_view_t reason;
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_distribution_endpoint(
        context, receiver_channel, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
        totals[i], 1, path ? &path->receivers[ordinal] : NULL,
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
    loom_aie2p_native_distribution_packet_t visit = {.destinations = copy,
                                                     .destination_count = 1};
    const loom_pipeline_transport_copy_t* next =
        loom_aie2p_native_distribution_next_copy(batch, &cursor, &remaining);
    while (next && next == copy + visit.destination_count &&
           next->source == copy->source && next->scope == copy->scope) {
      bool repeated = false;
      for (iree_host_size_t k = 0; k < visit.destination_count; ++k) {
        repeated |= copy[k].destination->channel == next->destination->channel;
      }
      if (repeated) {
        break;
      }
      ++visit.destination_count;
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
      const iree_host_size_t index = loom_pipeline_invocations_channel_index(
          &realization->invocations, destination->channel);
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &distribution->receivers[ordinals[index]];
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
  for (iree_host_size_t i = 0; i < source->transport.batch_count; ++i) {
    const loom_pipeline_transport_batch_t* batch =
        &source->transport.batches[i];
    const loom_pipeline_transport_cohort_t* cohort = batch->records;
    const loom_aie2p_native_channel_t* source_channel =
        loom_aie2p_native_channel(context, realization,
                                  cohort->admission->channel);
    const uint32_t repetitions =
        (uint32_t)(cohort->transfers * batch->record_count);
    loom_aie2p_native_distribution_t* path =
        ordered ? owner->distributions : NULL;
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
          (path &&
           source_channel->pool_index != path->sender.channel->pool_index)) {
        return loom_aie2p_native_reject(
            context, cohort->admission->op,
            IREE_SV("one source memory tile and destination set for an "
                    "ordered multicast path"));
      }
      for (iree_host_size_t j = 0;
           !packet_program && j < cohort->destination_count; ++j) {
        if (cohort->destinations[j].channel != first->destinations[j].channel ||
            (path &&
             loom_aie2p_native_channel(context, realization,
                                       cohort->destinations[j].channel) !=
                 path->receivers[j].channel)) {
          return loom_aie2p_native_reject(
              context, cohort->destinations[j].copy,
              IREE_SV("the same ordered destinations in every shared "
                      "multicast phase"));
        }
      }
    }
    // Geometry variants are bounded by the physical descriptor bank. A later
    // phase can reuse one only after its prior endpoints have retired.
    loom_aie2p_native_distribution_t* distribution = NULL;
    for (loom_aie2p_native_distribution_t* candidate =
             ordered ? owner->distributions : NULL;
         candidate; candidate = candidate->next) {
      if (candidate->sender.channel == source_channel &&
          (packet_program ? loom_aie2p_native_distribution_program_equal(
                                candidate->transport, candidate->source,
                                &source->transport, batch)
                          : candidate->repetitions == repetitions)) {
        distribution = candidate;
        break;
      }
    }
    if (!distribution) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*distribution), (void**)&distribution));
      *distribution = (loom_aie2p_native_distribution_t){
          .path = path ? path : distribution,
          .source = batch,
          .transport = &source->transport,
          .receiver_count = packet_program ? 0 : cohort->destination_count,
          .repetitions = packet_program ? 0 : repetitions,
      };
      if (packet_program) {
        bool valid = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_packet_distribution(
            context, realization, &source->transport, batch, source_channel,
            path, next_source, distribution, &valid));
        if (!valid) {
          return iree_ok_status();
        }
      } else {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            context->pass->arena, cohort->destination_count,
            sizeof(*distribution->receivers),
            (void**)&distribution->receivers));
        iree_string_view_t reason;
        bool valid = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_distribution_endpoint(
            context, source_channel, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
            distribution->repetitions,
            (uint32_t)(cohort->admissions == 1 ? batch->replays : 1),
            path ? &path->sender : NULL, &distribution->sender, &reason,
            &valid));
        if (!valid) {
          return loom_aie2p_native_reject(context, cohort->admission->op,
                                          reason);
        }
        const uint32_t source_route = path ? 0 : (*next_source)++;
        distribution->routes.first = context->routing.route_count;
        for (iree_host_size_t j = 0; j < cohort->destination_count; ++j) {
          const loom_pipeline_transport_destination_t* destination =
              &cohort->destinations[j];
          const loom_aie2p_native_channel_t* destination_channel =
              loom_aie2p_native_channel(context, realization,
                                        destination->channel);
          if (ordered && repetitions % destination_channel->source->capacity) {
            return loom_aie2p_native_reject(
                context, destination->copy,
                IREE_SV("complete destination-ring turns at a shared multicast "
                        "phase boundary"));
          }
          IREE_RETURN_IF_ERROR(loom_aie2p_native_select_distribution_endpoint(
              context, destination_channel,
              LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
              distribution->repetitions, 1, path ? &path->receivers[j] : NULL,
              &distribution->receivers[j], &reason, &valid));
          if (!valid) {
            return loom_aie2p_native_reject(context, destination->copy, reason);
          }
          const loom_aie2p_native_distribution_endpoint_t* sender =
              &distribution->sender;
          loom_aie2p_native_distribution_endpoint_t* receiver =
              &distribution->receivers[j];
          // Existing core/configuration/egress completion retires a receiver
          // only when that direct reader consumes every published record.
          // Distribution readers have no independent completion anchor; observe
          // their input DMA directly instead of reconstructing a transitive
          // dependency graph.
          const loom_aie2p_native_cursor_t* reader =
              &destination_channel->cursor.reader;
          receiver->wait_for_idle =
              ordered || reader->worker == UINT32_MAX ||
              reader->maximum_admissions != distribution->repetitions ||
              !realization->workers[reader->worker].block_execution_counts ||
              context->workers[reader->worker].execution ==
                  LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION;
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
        distribution->routes.count =
            context->routing.route_count - distribution->routes.first;
      }
      *tail = distribution;
      tail = &distribution->next;
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
  for (iree_host_size_t i = 0; i < count; ++i) {
    next[i] = context->workers[i].distribution_phases;
    const loom_pipeline_invocation_t* invocation =
        realization->invocations.workers[i].invocation;
    if (!invocation->ordinal || !next[i]) {
      continue;
    }
    const loom_aie2p_native_distribution_phase_t* phase = next[i];
    if (phase->next) {
      return loom_aie2p_native_reject(
          context, phase->source->records->admission->op,
          IREE_SV("one complete retained distribution lifetime per child"));
    }
    const loom_aie2p_native_distribution_t* distribution = phase->distribution;
    for (iree_host_size_t j = 0; j <= distribution->receiver_count; ++j) {
      const bool sender = j == 0;
      const loom_aie2p_native_distribution_endpoint_t* endpoint =
          sender ? &distribution->sender : &distribution->receivers[j - 1];
      const uint32_t cursor = sender ? endpoint->channel->cursor.writer.worker
                                     : endpoint->channel->cursor.reader.worker;
      iree_host_size_t engine = cursor == UINT32_MAX
                                    ? IREE_HOST_SIZE_MAX
                                    : context->core_engines[cursor];
      // An autonomous distributor has no instruction cursor owner. Its
      // compute-memory endpoint can instead be controlled by that tile's core.
      if (engine == IREE_HOST_SIZE_MAX &&
          endpoint->tile->facts->kind == LOOM_XDNA_TILE_KIND_COMPUTE) {
        engine = (iree_host_size_t)(endpoint->tile - context->tiles);
      }
      iree_host_size_t controller =
          engine == IREE_HOST_SIZE_MAX
              ? IREE_HOST_SIZE_MAX
              : context->execution.workers[invocation->ordinal *
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
      // The execution row includes every nested phase of this child. Those
      // controllers execute its outer start/retirement boundary as well.
      if (controller == IREE_HOST_SIZE_MAX) {
        return loom_aie2p_native_reject(
            context, phase->source->records->admission->op,
            IREE_SV("a producer and each receiver controlled by the complete "
                    "child or its concurrent parent invocation"));
      }
      context->distribution_controllers[invocation->ordinal *
                                            context->execution.engine_count +
                                        engine] = controller;
      loom_aie2p_native_distribution_command_t* command;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*command), (void**)&command));
      loom_aie2p_native_worker_t* worker = &context->workers[controller];
      *command = (loom_aie2p_native_distribution_command_t){
          .next = worker->distribution_commands,
          .invocation = invocation,
          .endpoint = endpoint,
          .direction = sender ? LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                              : LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY};
      worker->distribution_commands = command;
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
