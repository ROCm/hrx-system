// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"

#include "loom/analysis/storage_geometry.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/distribution.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"

iree_status_t loom_aie2p_native_select_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t worker,
    bool* out_eligible) {
  *out_eligible = false;
  const loom_pipeline_worker_t* source = &realization->workers[worker];
  const loom_pipeline_transport_input_t* plan = source->transport.input;
  if (!plan) {
    return iree_ok_status();
  }
  if (plan->acquisition || !source->transport.sources_stable ||
      plan->phase_count == 1) {
    // One stable phase retains autonomous repetition. Phased input admission
    // belongs to an existing instruction worker.
    return iree_ok_status();
  }
  // Prefixes fit independently before the first bulk demand, even if the
  // observer reads a later phase first. The final phase may span many windows;
  // its exact arrivals drive additional address periods and final reclamation.
  const loom_pipeline_transport_input_phase_t* tail =
      &plan->phases[plan->phase_count - 1];
  const bool invocation_owned =
      plan->admissions == 1 && tail->period == tail->transfers;
  for (iree_host_size_t i = 0; i < plan->phase_count; ++i) {
    const loom_pipeline_transport_input_phase_t* phase = &plan->phases[i];
    const loom_aie2p_native_channel_t* local = loom_aie2p_native_channel(
        context, realization, phase->reservation->channel);
    if (phase->transfer->request.source.memory_space !=
            LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
        phase->transfer->request.dest.memory_space !=
            LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
        phase->destination_byte_offset ||
        phase->transfer->request.transferred_byte_count !=
            local->record_byte_length ||
        !iree_any_bit_set(phase->transfer->request.source.flags,
                          LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH) ||
        !phase->period || phase->period > UINT32_MAX ||
        phase->transfers > UINT32_MAX || !plan->admissions ||
        plan->admissions > UINT32_MAX / phase->transfers ||
        phase->transfers % phase->period ||
        (i + 1 < plan->phase_count &&
         (phase->transfers > local->source->capacity ||
          phase->transfers != phase->period ||
          (!invocation_owned && phase->transfers != 1))) ||
        phase->source_byte_offset < 0 || phase->source_byte_stride < 0 ||
        phase->source_byte_stride > UINT32_MAX) {
      return iree_ok_status();
    }
  }
  loom_aie2p_native_input_t* input;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena, sizeof(*input),
                                           (void**)&input));
  *input = (loom_aie2p_native_input_t){
      .worker = (uint32_t)worker,
      .invocation = realization->invocations.workers[worker].invocation,
      .controller = UINT32_MAX,
      .begin = loom_symbol_ref_null(),
      .start = loom_symbol_ref_null(),
      .retire = loom_symbol_ref_null(),
      .source = plan,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, plan->phase_count, sizeof(*input->phases),
      (void**)&input->phases));
  for (iree_host_size_t i = 0; i < plan->phase_count; ++i) {
    input->phases[i] = (loom_aie2p_native_input_phase_t){
        .source = &plan->phases[i],
        .destination = loom_aie2p_native_channel(
            context, realization, plan->phases[i].reservation->channel),
    };
  }
  context->workers[worker].input = input;
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_input_descriptor(
    loom_aie2p_native_context_t* context, const loom_aie2p_native_tile_t* shim,
    uint16_t descriptor, uint32_t binding, uint64_t offset, uint64_t length,
    const uint32_t words[8]) {
  loom_aie2p_native_input_descriptor_t* selected;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      context->pass->arena, sizeof(*selected), (void**)&selected));
  *selected = (loom_aie2p_native_input_descriptor_t){
      .next = context->input_descriptors,
      .tile = shim,
      .descriptor = descriptor,
      .binding = binding,
      .byte_offset = offset,
      .byte_length = length,
  };
  memcpy(selected->words, words, sizeof(selected->words));
  context->input_descriptors = selected;
  return iree_ok_status();
}

static bool loom_aie2p_native_input_route(
    loom_aie2p_native_context_t* context, uint32_t* next_source,
    uint32_t canonical, loom_xdna_tile_coordinate_t source,
    loom_xdna_stream_port_t source_port, uint8_t source_channel,
    loom_xdna_tile_coordinate_t destination,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel,
    uint8_t packet) {
  const iree_host_size_t first = context->routing.route_count;
  if (!loom_aie2p_array_route_stream(
          &context->routing, (*next_source)++, canonical, source, source_port,
          source_channel, destination, destination_port, destination_channel,
          true)) {
    return false;
  }
  loom_aie2p_native_route_label(context, first, packet, UINT32_MAX);
  return true;
}

// The input proof supplies prefixes followed by a bulk phase in each source
// lifetime. A bulk arrival proves those prefixes have left the shared queue.
// Counting the active task, the next prefixes and window require P+2 slots.
static uint32_t loom_aie2p_native_input_receive_window(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_input_t* input) {
  const iree_host_size_t tail = input->source->phase_count - 1;
  const loom_aie2p_native_input_phase_t* bulk = &input->phases[tail];
  const loom_aie2p_native_channel_t* channel = bulk->destination;
  for (iree_host_size_t i = 0; i < tail; ++i) {
    if (input->phases[i].destination->pool_index != channel->pool_index) {
      return 0;
    }
  }
  if (channel->replay.record_count) {
    // The finite retained batch coalesces into one receive task. Its prefix
    // tasks and complete payload fit together without per-record refills.
    const loom_xdna_dma_facts_t* dma =
        &context->pool_tiles[channel->pool_index]->facts->dma;
    return input->source->phase_count <= dma->task_queue_depth
               ? (uint32_t)bulk->source->transfers
               : 0;
  }
  return loom_aie2p_native_phased_window(context, channel, tail,
                                         (uint32_t)input->source->admissions,
                                         (uint32_t)bulk->source->transfers);
}

static iree_status_t loom_aie2p_native_input_receivers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    loom_aie2p_native_input_t* input, const loom_aie2p_native_input_t* shared,
    uint32_t window, iree_string_view_t* reason, bool* out_valid) {
  const iree_host_size_t tail = input->source->phase_count - 1;
  for (iree_host_size_t i = 0; i <= tail; ++i) {
    loom_aie2p_native_input_phase_t* phase = &input->phases[i];
    if (phase->destination->replay.record_count &&
        (input->source->admissions != 1 ||
         phase->source->transfers != phase->destination->replay.record_count ||
         phase->source->period != phase->source->transfers)) {
      *reason =
          IREE_SV("one complete retained input batch per source lifetime");
      *out_valid = false;
      return iree_ok_status();
    }
    const uint32_t repetitions =
        window
            ? (i == tail ? window : (uint32_t)phase->source->transfers)
            : (uint32_t)(input->source->admissions * phase->source->transfers);
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channel_endpoint(
        context, &realization->invocations, input->invocation,
        phase->destination, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
        repetitions, 1, 0,
        shared ? &shared->phases[i].receiver
               : (window && i ? &input->phases[0].receiver : NULL),
        window ? NULL : phase->destination, &phase->receiver, reason,
        out_valid));
    if (!*out_valid) {
      return iree_ok_status();
    }
    phase->receiver.wait_for_idle = true;
    phase->receiver.controller_engine =
        context->core_engines[input->controller];
  }
  const loom_aie2p_native_input_phase_t* bulk = &input->phases[tail];
  const uint32_t repetitions =
      (uint32_t)(input->source->admissions * bulk->source->transfers);
  if (window && repetitions > window) {
    loom_aie2p_native_refill_queue_t* queue;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_add_refill(
        context, context->pool_tiles[bulk->destination->pool_index],
        bulk->receiver.engine, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
        bulk->destination, repetitions, window, &bulk->receiver.sequence,
        &queue));
    if (input->source->admissions > 1) {
      uint32_t* prefixes;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          context->pass->arena, tail, sizeof(*prefixes), (void**)&prefixes));
      for (iree_host_size_t i = 0; i < tail; ++i) {
        prefixes[i] = input->phases[i].receiver.sequence.tasks[0];
      }
      queue->prefixes = prefixes;
      queue->prefix_count = (uint32_t)tail;
      queue->cycle_windows = (uint32_t)bulk->source->transfers / window;
    }
  }
  return iree_ok_status();
}

static loom_aie2p_native_input_path_t* loom_aie2p_native_find_input_path(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocations_t* invocations,
    const loom_aie2p_native_worker_t* owner,
    const loom_aie2p_native_tile_t* shim,
    const loom_aie2p_native_input_t* input) {
  // The list is bounded by physical shim queues, not worker occurrences.
  // Common execution admission already proved nonoverlap on this engine.
  // Borrow it only when each input has its controller's lifetime; two inputs
  // in the same invocation cannot share a terminal retirement boundary.
  for (loom_aie2p_native_input_path_t* path = owner->input_paths; path;
       path = path->next) {
    const loom_aie2p_native_input_t* first = path->first;
    const loom_aie2p_native_input_t* last = path->last;
    if (path->shim != shim ||
        first->shared_receiver != input->shared_receiver ||
        first->source->phase_count != input->source->phase_count ||
        last->controller == input->controller ||
        invocations->workers[last->controller].invocation != last->invocation ||
        invocations->workers[input->controller].invocation !=
            input->invocation) {
      continue;
    }
    bool same_receivers = true;
    for (iree_host_size_t i = 0; i < input->source->phase_count; ++i) {
      same_receivers &=
          first->phases[i].receiver.tile ==
          context->pool_tiles[input->phases[i].destination->pool_index];
    }
    if (same_receivers) {
      return path;
    }
  }
  return NULL;
}

iree_status_t loom_aie2p_native_route_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    loom_aie2p_native_input_t* input, uint32_t* next_source,
    uint8_t* next_controller, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_worker_t* source = &realization->workers[input->worker];
  const loom_op_t* declaration = source->source->declaration;
  loom_aie2p_native_tile_t* shim =
      &context->tiles[context->workers[input->worker].tile->coordinate.column *
                      context->family->row_count];
  loom_aie2p_native_worker_t* owner =
      &context->workers[context->execution
                            .workers[context->core_engines[input->controller]]];
  const uint32_t window =
      loom_aie2p_native_input_receive_window(context, input);
  input->shared_receiver = window != 0;
  loom_aie2p_native_input_path_t* path = loom_aie2p_native_find_input_path(
      context, &realization->invocations, owner, shim, input);
  const bool shared_path = path != NULL;
  const loom_xdna_dma_facts_t* dma = &shim->facts->dma;
  const uint64_t descriptors = input->source->phase_count;
  uint16_t descriptor_base, descriptor_count;
  uint16_t* used = loom_aie2p_native_descriptor_bank(
      shim, path ? path->input_engine : shim->next_mm2s, &descriptor_base,
      &descriptor_count);
  if ((!path && shim->next_mm2s == dma->channel_count_per_direction) ||
      descriptors > descriptor_count - *used ||
      input->source->phase_count > dma->task_queue_depth) {
    return loom_aie2p_native_reject(
        context, declaration,
        IREE_SV("shim queues and immutable descriptors for phased input"));
  }
  const uint64_t origin = (uint64_t)shim->coordinate.column
                          << context->family->column_shift;
  if (!path) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                             sizeof(*path), (void**)&path));
    *path = (loom_aie2p_native_input_path_t){
        .next = owner->input_paths,
        .first = input,
        .shim = shim,
        .input_engine = shim->next_mm2s++,
    };
    owner->input_paths = path;
    const uint16_t engine = path->input_engine;
    path->input_queue =
        (uint32_t)(loom_xdna_register_field_address(
                       context->family,
                       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID,
                       shim->coordinate, &engine) -
                   origin);
  }
  input->path = path;
  path->last = input;
  iree_string_view_t reason;
  bool receivers_valid = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_receivers(
      context, realization, input, shared_path ? path->first : NULL, window,
      &reason, &receivers_valid));
  if (!receivers_valid) {
    return loom_aie2p_native_reject(context, declaration, reason);
  }
  const uint32_t canonical = *next_source;
  for (iree_host_size_t i = 0; i < input->source->phase_count; ++i) {
    loom_aie2p_native_input_phase_t* phase = &input->phases[i];
    const loom_pipeline_transport_input_phase_t* plan = phase->source;
    if (loom_movement_endpoint_minimum_byte_alignment(
            &plan->transfer->request.source) < dma->address_alignment) {
      return loom_aie2p_native_reject(
          context, plan->transfer->request.op,
          IREE_SV("aligned phased input DMA sources"));
    }
    loom_storage_geometry_t geometry;
    if (!loom_storage_geometry_query(
            &source->facts.context, context->code.module,
            plan->transfer->request.source.type, &geometry) ||
        !loom_aie2p_native_encode_ingress(
            context->family, &geometry, dma,
            (uint32_t)plan->transfer->request.transferred_byte_count,
            (uint32_t)plan->source_byte_stride, (uint32_t)plan->period,
            (uint32_t)plan->period, phase->words, &phase->repetitions) ||
        phase->repetitions > dma->maximum_task_repeat_count) {
      return loom_aie2p_native_reject(
          context, declaration,
          IREE_SV(
              "a phased input address period representable by one shim task"));
    }
    phase->words[2] |=
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_ENABLE_PACKET, 1) |
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_PACKET_ID,
            input->shared_receiver ? 0 : i);
    phase->descriptor = descriptor_base + *used;
    ++*used;
    const uint64_t span =
        (uint64_t)plan->source_byte_stride * (plan->period - 1) +
        plan->transfer->request.source.static_byte_length;
    const loom_value_t* root = loom_module_value(
        context->code.module, plan->transfer->request.source.root_value_id);
    const loom_block_t* entry =
        loom_region_entry_block(loom_func_like_body(realization->function));
    if (!loom_value_is_block_arg(root) || loom_value_def_block(root) != entry) {
      return loom_aie2p_native_reject(
          context, plan->transfer->request.op,
          IREE_SV("a caller-buffer root for stable phased input"));
    }
    const uint32_t binding = loom_value_def_index(root);
    const uint64_t offset = plan->source_byte_offset;
    context->bindings[binding].byte_length =
        iree_max(context->bindings[binding].byte_length, offset + span);
    context->bindings[binding].byte_alignment =
        iree_max(context->bindings[binding].byte_alignment,
                 plan->transfer->request.source.root_minimum_alignment);
    context->bindings[binding].access |= LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_input_descriptor(
        context, shim, phase->descriptor, binding, offset, span, phase->words));
    if (!shared_path && (!input->shared_receiver || !i) &&
        !loom_aie2p_native_input_route(
            context, next_source, canonical, shim->coordinate,
            LOOM_XDNA_STREAM_PORT_DMA, path->input_engine,
            phase->receiver.tile->coordinate, LOOM_XDNA_STREAM_PORT_DMA,
            phase->receiver.engine, (uint8_t)i)) {
      return loom_aie2p_native_reject(
          context, declaration,
          IREE_SV("packet routes from phased input to each local destination"));
    }
  }
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
      context, realization, input->controller, shim,
      input->invocation->ordinal != 0, next_source, next_controller, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  if (input->invocation->ordinal) {
    for (iree_host_size_t i = 0; i < input->source->phase_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
          context, realization, input->controller,
          input->phases[i].receiver.tile, true, next_source, next_controller,
          &valid));
      if (!valid) {
        return iree_ok_status();
      }
    }
  }
  path->control_packet = owner->control_packets[shim - context->tiles];
  *out_valid = true;
  return iree_ok_status();
}
