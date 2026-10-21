// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_geometry.h"
#include "loom/ops/channel/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

// DMA queue admission consumes already established completion edges. It never
// inserts a wait or changes the author's grouping to hide resource exhaustion.
typedef struct loom_aie2p_native_queue_t {
  // Selected engine pair, allocated at first use.
  loom_aie2p_native_dma_path_t* path;
  // Retained completion sites for tasks not yet retired in this block.
  const loom_op_t** completions;
  // Number of retained completion sites.
  uint8_t count;
  // Joint queue capacity of the two endpoints.
  uint8_t capacity;
  // Block whose straight-line issue interval is being admitted.
  const loom_block_t* block;
  // Authored group currently accumulating one physical task on this path.
  const loom_kernel_async_group_t* group;
  // Task accumulating members from group, or NULL before its first member.
  loom_aie2p_native_transfer_task_t* task;
} loom_aie2p_native_queue_t;

// SRAM channels of equal parity share one of two 24-descriptor banks.
// Compute and shim channels share a single bank of all descriptors.
static uint16_t* loom_aie2p_native_descriptor_bank(
    loom_aie2p_native_tile_t* tile, uint8_t engine, uint16_t* out_base,
    uint16_t* out_count) {
  const bool memory = tile->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY;
  const uint8_t bank = memory ? engine & 1 : 0;
  *out_count = tile->facts->dma.buffer_descriptor_count / (memory ? 2 : 1);
  *out_base = bank * *out_count;
  return &tile->descriptor_counts[bank];
}

static bool loom_aie2p_native_descriptor(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_worker_t* worker,
    const loom_movement_endpoint_t* endpoint, const loom_xdna_dma_facts_t* dma,
    uint32_t bytes, loom_xdna_tile_kind_t kind, uint32_t* words,
    iree_string_view_t* reason) {
  loom_storage_geometry_t geometry;
  if (!loom_storage_geometry_query(&worker->facts.context, context->code.module,
                                   endpoint->type, &geometry)) {
    *reason = IREE_SV("specialized DMA endpoint shape and strides");
    return false;
  }
  return loom_aie2p_native_encode_descriptor(context->family, &geometry, dma,
                                             bytes, 1, kind, words, reason);
}

bool loom_aie2p_native_select_ingress_descriptor(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_worker_t* source,
    const loom_movement_request_t* request, uint32_t byte_stride,
    uint32_t repetitions, loom_aie2p_native_worker_t* worker) {
  const loom_xdna_dma_facts_t* dma =
      &context
           ->tiles[worker->tile->coordinate.column * context->family->row_count]
           .facts->dma;
  loom_storage_geometry_t record;
  if (!loom_storage_geometry_query(&source->facts.context, context->code.module,
                                   request->source.type, &record)) {
    return false;
  }
  const uint32_t record_bytes = (uint32_t)request->transferred_byte_count;
  return loom_aie2p_native_encode_ingress(
      context->family, &record, dma, record_bytes, byte_stride, repetitions,
      worker->repetition.shim_words, &worker->repetition.shim_repetitions);
}

// Labels only newly appended edges; prefix sharing belongs to a single route
// source. Packet routes use distinct sources so filters remain unambiguous.
static void loom_aie2p_native_route_label(loom_aie2p_native_context_t* context,
                                          iree_host_size_t first,
                                          uint8_t packet) {
  for (iree_host_size_t i = first; i < context->routing.route_count; ++i) {
    context->routes[i] = (loom_aie2p_native_route_t){
        .edge = &context->routing.routes[i], .packet = packet};
  }
}

static bool loom_aie2p_native_packet_route(
    loom_aie2p_native_context_t* context, uint32_t* next_source,
    loom_xdna_tile_coordinate_t source, loom_xdna_stream_port_t source_port,
    loom_xdna_tile_coordinate_t destination,
    loom_xdna_stream_port_t destination_port, uint8_t packet) {
  const iree_host_size_t first = context->routing.route_count;
  const uint32_t index = (*next_source)++;
  if (!loom_aie2p_array_route_stream(&context->routing, index, index, source,
                                     source_port, 0, destination,
                                     destination_port, 0)) {
    return false;
  }
  loom_aie2p_native_route_label(context, first, packet);
  return true;
}

// Workers have flat CFGs, and async legality retires every group within its
// block. Different streams therefore cannot overlap. Within a stream, the
// retained completion edge determines when a physical resource becomes
// reusable.
static bool loom_aie2p_native_transfer_retired(
    const loom_aie2p_native_transfer_t* transfer, const loom_op_t* issue) {
  return transfer->completion->parent_block != issue->parent_block ||
         transfer->completion->block_ordinal <= issue->block_ordinal;
}

static void loom_aie2p_native_task_descriptor_words(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor,
    const loom_aie2p_native_transfer_t* transfer, uint32_t out_words[8]) {
  memcpy(out_words,
         descriptor->shim ? transfer->shim_words : transfer->local_words,
         sizeof(transfer->local_words));
  if (transfer->descriptor_ordinal + 1 == task->transfer_count) {
    return;
  }
  const loom_aie2p_native_memory_layout_t* layout =
      descriptor->shim ? NULL : task->path->local->memory_layout;
  const uint8_t word = layout ? layout->chain.word : 7;
  const loom_xdna_register_field_id_t next =
      layout ? layout->chain.next
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_NEXT_BD;
  const loom_xdna_register_field_id_t enable =
      layout ? layout->chain.enable
             : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_USE_NEXT_BD;
  out_words[word] |= loom_xdna_register_field_encode_admitted(
                         next, descriptor->sequence.descriptor +
                                   transfer->descriptor_ordinal + 1) |
                     loom_xdna_register_field_encode_admitted(enable, 1);
}

static bool loom_aie2p_native_task_descriptor_matches(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor) {
  if (descriptor->sequence.descriptor_count != task->transfer_count) {
    return false;
  }
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    uint32_t words[8];
    loom_aie2p_native_task_descriptor_words(task, descriptor, transfer, words);
    if (memcmp(descriptor->sequence.words[transfer->descriptor_ordinal], words,
               sizeof(words)) != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_native_task_external_address_differs(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_transfer_t* previous) {
  const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
  for (; transfer && previous;
       transfer = transfer->task_next, previous = previous->task_next) {
    if (transfer->binding != previous->binding ||
        transfer->external_view->begin_byte_offset.constant !=
            previous->external_view->begin_byte_offset.constant) {
      return true;
    }
  }
  return transfer || previous;
}

// Each task endpoint has an immutable descriptor sequence and latest retained
// completion. Local and shim sequences can be reused independently.
static loom_aie2p_native_dma_descriptor_t*
loom_aie2p_native_reusable_descriptor(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_transfer_task_t* task, bool shim) {
  if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
    // Autonomous descriptors retain iteration state. Configuration-only waits
    // observe completion credits without consuming them for a subsequent use.
    return NULL;
  }
  const loom_op_t* issue = task->first_transfer->source->request.op;
  for (loom_aie2p_native_dma_descriptor_t* candidate = worker->descriptors;
       candidate; candidate = candidate->next) {
    const loom_aie2p_native_transfer_t* previous = candidate->last_transfer;
    if (candidate->shim != shim || candidate->path != task->path ||
        !loom_aie2p_native_transfer_retired(previous, issue) ||
        (candidate->completion &&
         !loom_aie2p_native_transfer_retired(
             candidate->completion->last_transfer, issue)) ||
        !loom_aie2p_native_task_descriptor_matches(task, candidate)) {
      continue;
    }
    if (shim) {
      candidate->address_dynamic |=
          task->shim_address_dynamic ||
          loom_aie2p_native_task_external_address_differs(task, previous);
    } else if (candidate->address_dynamic != task->local_address_dynamic) {
      continue;
    }
    return candidate;
  }
  return NULL;
}

// Core waits consume arrival credits, so a retired credit can serve a different
// descriptor geometry on the same tile. Configuration waits leave their credits
// set and cannot share them. The search is bounded by the physical lock count.
static loom_aie2p_native_dma_completion_t*
loom_aie2p_native_reusable_completion(const loom_aie2p_native_worker_t* worker,
                                      const loom_aie2p_native_tile_t* tile,
                                      const loom_op_t* issue) {
  if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
    return NULL;
  }
  for (loom_aie2p_native_dma_completion_t* completion = worker->dma_completions;
       completion; completion = completion->next) {
    if (completion->tile == tile &&
        loom_aie2p_native_transfer_retired(completion->last_transfer, issue)) {
      return completion;
    }
  }
  return NULL;
}

static iree_status_t loom_aie2p_native_select_sequence(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    const uint32_t template_words[8],
    loom_aie2p_native_dma_sequence_t* out_sequence,
    iree_string_view_t* out_reason, bool* out_valid);

static iree_status_t loom_aie2p_native_allocate_task_descriptor(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* worker,
    loom_aie2p_native_transfer_task_t* task, bool shim,
    loom_aie2p_native_dma_descriptor_t*** descriptor_tail,
    loom_aie2p_native_dma_completion_t*** completion_tail,
    loom_aie2p_native_dma_descriptor_t** out_descriptor) {
  loom_aie2p_native_tile_t* tile = shim ? task->path->shim : task->path->local;
  const uint8_t engine =
      shim ? task->path->shim_engine : task->path->local_engine;
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_count);
  if (task->transfer_count > descriptor_count - *next_descriptor) {
    return loom_aie2p_native_reject(
        context, task->first_transfer->source->request.op,
        shim ? IREE_SV("available native shim descriptors for one async group")
             : IREE_SV("available native descriptors in the DMA channel's bank "
                       "for one async group"));
  }
  const iree_host_size_t words_bytes =
      task->transfer_count * sizeof(uint32_t[8]);
  loom_aie2p_native_dma_descriptor_t* descriptor;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                           sizeof(*descriptor) + words_bytes,
                                           (void**)&descriptor));
  *descriptor = (loom_aie2p_native_dma_descriptor_t){
      .path = task->path,
      .sequence =
          {
              .descriptor = descriptor_base + *next_descriptor,
              .descriptor_count = (uint16_t)task->transfer_count,
              .words = (uint32_t (*)[8])(descriptor + 1),
          },
      .shim = shim,
      .address_dynamic =
          shim ? task->shim_address_dynamic : task->local_address_dynamic,
  };
  *next_descriptor += (uint16_t)task->transfer_count;
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    loom_aie2p_native_task_descriptor_words(
        task, descriptor, transfer,
        descriptor->sequence.words[transfer->descriptor_ordinal]);
  }
  if (!shim && task->path->ingress) {
    descriptor->completion = loom_aie2p_native_reusable_completion(
        worker, tile, task->first_transfer->source->request.op);
    if (!descriptor->completion) {
      if (tile->next_lock == tile->facts->lock_count) {
        return loom_aie2p_native_reject(
            context, task->first_transfer->source->request.op,
            IREE_SV("an available native ingress completion semaphore"));
      }
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*descriptor->completion),
          (void**)&descriptor->completion));
      *descriptor->completion = (loom_aie2p_native_dma_completion_t){
          .tile = tile, .lock = tile->next_lock++};
      if (worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
        const loom_aie2p_native_channel_t* channel =
            task->first_transfer->local_channel;
        const uint32_t channel_index = (uint32_t)(channel - context->channels);
        const loom_aie2p_native_channel_access_t* access =
            &worker->channels.accesses[worker->channels.indices[channel_index]];
        descriptor->completion->selector =
            access->window->lock_selector_base + descriptor->completion->lock;
      }
      **completion_tail = descriptor->completion;
      *completion_tail = &descriptor->completion->next;
    }
  }
  **descriptor_tail = descriptor;
  *descriptor_tail = &descriptor->next;
  *out_descriptor = descriptor;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_task_descriptors(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* worker,
    loom_aie2p_native_transfer_task_t* task, bool autonomous,
    loom_aie2p_native_dma_descriptor_t*** descriptor_tail,
    loom_aie2p_native_dma_completion_t*** completion_tail, bool* out_valid) {
  *out_valid = false;
  if (autonomous) {
    // Autonomous repetition currently admits one source transfer and builds its
    // finite ring program through the dedicated sequence planner.
    const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
    for (unsigned endpoint = 0; endpoint < 2; ++endpoint) {
      const bool shim = endpoint != 0;
      loom_aie2p_native_tile_t* tile =
          shim ? task->path->shim : task->path->local;
      loom_aie2p_native_dma_descriptor_t* descriptor;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          context->pass->arena, sizeof(*descriptor), (void**)&descriptor));
      *descriptor = (loom_aie2p_native_dma_descriptor_t){.path = task->path,
                                                         .shim = shim};
      iree_string_view_t reason;
      bool valid;
      if (shim) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_sequence(
            context, tile, task->path->shim_engine,
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
            worker->repetition.shim_repetitions, 1, 0, 0, transfer->shim_words,
            &descriptor->sequence, &reason, &valid));
      } else {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_sequence(
            context, tile, task->path->local_engine,
            LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY, worker->repetition.count,
            transfer->local_channel->source->capacity,
            transfer->local_channel->byte_offset +
                (uint32_t)transfer->local_view->projection_byte_offset.constant,
            transfer->local_channel->byte_stride, transfer->local_words,
            &descriptor->sequence, &reason, &valid));
      }
      if (!valid) {
        return loom_aie2p_native_reject(
            context, task->first_transfer->source->request.op, reason);
      }
      **descriptor_tail = descriptor;
      *descriptor_tail = &descriptor->next;
      if (shim) {
        task->shim_descriptor = descriptor;
      } else {
        task->local_descriptor = descriptor;
      }
    }
  } else {
    task->local_descriptor =
        loom_aie2p_native_reusable_descriptor(worker, task, false);
    if (!task->local_descriptor) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_allocate_task_descriptor(
          context, worker, task, false, descriptor_tail, completion_tail,
          &task->local_descriptor));
      if (!task->local_descriptor) {
        return iree_ok_status();
      }
    }
    task->shim_descriptor =
        loom_aie2p_native_reusable_descriptor(worker, task, true);
    if (!task->shim_descriptor) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_allocate_task_descriptor(
          context, worker, task, true, descriptor_tail, completion_tail,
          &task->shim_descriptor));
      if (!task->shim_descriptor) {
        return iree_ok_status();
      }
    }
  }
  task->local_descriptor->last_transfer = task->first_transfer;
  task->shim_descriptor->last_transfer = task->first_transfer;
  if (task->local_descriptor->completion) {
    task->local_descriptor->completion->last_transfer = task->first_transfer;
  }
  for (loom_aie2p_native_transfer_t* transfer = task->first_transfer; transfer;
       transfer = transfer->task_next) {
    transfer->local_descriptor = task->local_descriptor;
    transfer->shim_descriptor = task->shim_descriptor;
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_sequence(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    const uint32_t template_words[8],
    loom_aie2p_native_dma_sequence_t* out_sequence,
    iree_string_view_t* out_reason, bool* out_valid) {
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_native_build_sequence(
      &tile->facts->dma, tile->memory_layout,
      descriptor_base + *next_descriptor, descriptor_count - *next_descriptor,
      direction, repetitions, capacity, byte_offset, byte_stride,
      template_words, out_sequence, out_reason, out_valid,
      context->pass->arena));
  if (*out_valid) {
    *next_descriptor += out_sequence->descriptor_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_worker_transfers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t i,
    uint32_t* next_source, uint8_t* next_controller, bool* out_valid) {
  *out_valid = false;
  iree_arena_allocator_t* arena = context->pass->arena;
  loom_module_t* module = context->code.module;
  const iree_host_size_t tile_count =
      context->family->column_count * context->family->row_count;
  const loom_block_t* entry =
      loom_region_entry_block(loom_func_like_body(realization->function));
  const loom_pipeline_worker_t* source = &realization->workers[i];
  loom_aie2p_native_worker_t* worker = &context->workers[i];
  const bool configuration =
      worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
  const bool autonomous = worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA;
  loom_aie2p_native_queue_t* queues;
  uint8_t* controls;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count * 2, sizeof(*queues), (void**)&queues));
  memset(queues, 0, tile_count * 2 * sizeof(*queues));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, tile_count, (void**)&controls));
  memset(controls, 0xff, tile_count);
  uint8_t next_control = 0;
  bool has_egress = false;
  iree_host_size_t transport_step = 0;
  loom_aie2p_native_transfer_t** tail = &worker->transfers;
  loom_aie2p_native_transfer_task_t** task_tail = &worker->transfer_tasks;
  loom_aie2p_native_dma_descriptor_t** descriptor_tail = &worker->descriptors;
  loom_aie2p_native_dma_completion_t** completion_tail =
      &worker->dma_completions;
  iree_host_size_t stream_index = 0;
  for (const loom_kernel_async_stream_t* stream =
           configuration ? source->transport.streams[0]
                         : source->asynchronous.streams;
       stream;
       stream = configuration ? (++stream_index < source->transport.stream_count
                                     ? source->transport.streams[stream_index]
                                     : NULL)
                              : stream->next) {
    for (iree_host_size_t j = 0; j < stream->transfer_count; ++j) {
      const loom_kernel_async_transfer_t* transfer = &stream->transfers[j];
      const loom_movement_request_t* request = &transfer->request;
      const bool ingress =
          request->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
      const loom_movement_endpoint_t* external =
          ingress ? &request->source : &request->dest;
      const loom_movement_endpoint_t* local =
          ingress ? &request->dest : &request->source;
      if (request->kind != LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY ||
          external->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
          local->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
        return loom_aie2p_native_reject(
            context, request->op,
            IREE_SV("a dense or strided borrowed copy between caller memory "
                    "and tile memory"));
      }
      const loom_view_region_t *external_view = NULL, *local_view = NULL;
      loom_view_region_table_try_lookup(
          &source->asynchronous.movement.view_regions, external->value_id,
          &external_view);
      loom_view_region_table_try_lookup(
          &source->asynchronous.movement.view_regions, local->value_id,
          &local_view);
      loom_channel_payload_record_t payload_record = {
          .value_id = LOOM_VALUE_ID_INVALID,
      };
      const bool has_payload_record =
          local_view &&
          loom_channel_completion_try_lookup_payload_record(
              &source->completion, local->value_id, &payload_record);
      const loom_channel_identity_t* payload_channel =
          has_payload_record ? loom_channel_plan_channel(
                                   &source->channels, payload_record.value_id)
                             : NULL;
      const loom_pipeline_resource_channel_t* bound =
          payload_channel
              ? loom_pipeline_resources_lookup_channel(
                    &realization->resources, payload_channel->value_id)
              : NULL;
      const loom_aie2p_native_channel_t* local_channel =
          bound ? &context->channels[bound - realization->resources.channels]
                : NULL;
      const bool record_dynamic =
          bound && bound->capacity > 1 && payload_record.maximum_admissions > 1;
      const loom_value_t* root =
          loom_module_value(module, external->root_value_id);
      if (!local_channel || !loom_value_is_block_arg(root) ||
          loom_value_def_block(root) != entry ||
          !loom_symbolic_expr_is_linear(&external_view->begin_byte_offset) ||
          !loom_symbolic_expr_is_linear(&local_view->projection_byte_offset) ||
          external->begin_byte_offset.facts.range_lo < 0 ||
          external->end_byte_offset.facts.range_hi > UINT32_MAX ||
          local_view->projection_byte_offset.facts.range_lo < 0 ||
          local_view->projection_byte_offset.facts.range_hi > UINT32_MAX) {
        return loom_aie2p_native_reject(
            context, request->op,
            IREE_SV("a caller-buffer root and a borrowed channel record with "
                    "bounded native offsets"));
      }
      loom_aie2p_native_tile_t* local_tile =
          context->pool_tiles[local_channel->pool_index];
      const loom_aie2p_native_memory_layout_t* layout =
          local_tile->memory_layout;
      loom_aie2p_native_tile_t* shim =
          &context->tiles[worker->tile->coordinate.column *
                          context->family->row_count];
      if (loom_movement_endpoint_minimum_byte_alignment(external) <
              shim->facts->dma.address_alignment ||
          loom_movement_endpoint_minimum_byte_alignment(local) <
              local_tile->facts->dma.address_alignment) {
        return loom_aie2p_native_reject(context, request->op,
                                        IREE_SV("aligned DMA endpoints"));
      }
      const iree_host_size_t local_index = local_tile - context->tiles;
      loom_aie2p_native_queue_t* queue =
          &queues[local_index * 2 + (ingress ? 0 : 1)];
      if (!queue->path) {
        uint8_t* local_engine =
            ingress ? &local_tile->next_s2mm : &local_tile->next_mm2s;
        uint8_t* shim_engine = ingress ? &shim->next_mm2s : &shim->next_s2mm;
        if (*local_engine ==
                local_tile->facts->dma.channel_count_per_direction ||
            *shim_engine == shim->facts->dma.channel_count_per_direction ||
            (!ingress && has_egress && !configuration)) {
          return loom_aie2p_native_reject(
              context, request->op,
              IREE_SV("an independent DMA engine pair and one external "
                      "completion stream per worker"));
        }
        IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*queue->path),
                                                 (void**)&queue->path));
        *queue->path =
            (loom_aie2p_native_dma_path_t){.next = worker->paths,
                                           .local = local_tile,
                                           .shim = shim,
                                           .local_engine = (*local_engine)++,
                                           .shim_engine = (*shim_engine)++,
                                           .ingress = ingress};
        worker->paths = queue->path;
        queue->capacity = iree_min(local_tile->facts->dma.task_queue_depth,
                                   shim->facts->dma.task_queue_depth);
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            arena, queue->capacity, sizeof(*queue->completions),
            (void**)&queue->completions));
        const iree_host_size_t route_first = context->routing.route_count;
        const uint32_t route = (*next_source)++;
        const bool routed =
            ingress ? loom_aie2p_array_route_ingress(
                          &context->routing, route, route, shim->coordinate,
                          queue->path->shim_engine, local_tile->coordinate,
                          queue->path->local_engine)
                    : loom_aie2p_array_route_egress(
                          &context->routing, route, route,
                          local_tile->coordinate, queue->path->local_engine,
                          shim->coordinate, queue->path->shim_engine);
        if (!routed) {
          return loom_aie2p_native_reject(
              context, request->op, IREE_SV("available payload stream links"));
        }
        loom_aie2p_native_route_label(context, route_first, UINT8_MAX);
        loom_aie2p_native_tile_t* destinations[] = {local_tile, shim};
        uint8_t* packets[] = {&queue->path->control.local,
                              &queue->path->control.shim};
        for (unsigned k = 0;
             worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE && k < 2;
             ++k) {
          const iree_host_size_t tile_index = destinations[k] - context->tiles;
          if (controls[tile_index] == UINT8_MAX) {
            if (next_control == 4 ||
                !loom_aie2p_native_packet_route(
                    context, next_source, worker->tile->coordinate,
                    LOOM_XDNA_STREAM_PORT_CORE, destinations[k]->coordinate,
                    LOOM_XDNA_STREAM_PORT_TILE_CONTROL, next_control)) {
              return loom_aie2p_native_reject(
                  context, request->op,
                  IREE_SV("available worker control-packet routes"));
            }
            controls[tile_index] = next_control++;
          }
          *packets[k] = controls[tile_index];
        }
        if (!ingress) {
          has_egress = true;
          queue->path->completion_packet =
              configuration ? 0 : next_controller[shim->coordinate.column]++;
          if (!loom_aie2p_native_packet_route(
                  context, next_source, shim->coordinate,
                  LOOM_XDNA_STREAM_PORT_TILE_CONTROL,
                  configuration ? shim->coordinate : worker->tile->coordinate,
                  configuration ? LOOM_XDNA_STREAM_PORT_SOUTH
                                : LOOM_XDNA_STREAM_PORT_CORE,
                  queue->path->completion_packet)) {
            return loom_aie2p_native_reject(
                context, request->op,
                IREE_SV("an independent external-completion return route"));
          }
        }
      }
      if (queue->block != stream->block) {
        queue->count = 0;
        queue->block = stream->block;
        queue->group = NULL;
        queue->task = NULL;
      }
      const loom_kernel_async_group_t* group =
          &stream->groups[transfer->group_index];
      if (queue->group != group) {
        uint8_t pending = 0;
        for (uint8_t k = 0; k < queue->count; ++k) {
          if (queue->completions[k]->block_ordinal >
              request->op->block_ordinal) {
            queue->completions[pending++] = queue->completions[k];
          }
        }
        queue->count = pending;
        if (pending == queue->capacity) {
          return loom_aie2p_native_reject(
              context, request->op,
              IREE_SV(
                  "DMA group retirement before the native task queues fill"));
        }
        queue->completions[queue->count++] = group->completion;
        loom_aie2p_native_transfer_task_t* task;
        IREE_RETURN_IF_ERROR(
            iree_arena_allocate(arena, sizeof(*task), (void**)&task));
        *task = (loom_aie2p_native_transfer_task_t){.path = queue->path};
        *task_tail = task;
        task_tail = &task->next;
        queue->group = group;
        queue->task = task;
      }
      loom_aie2p_native_transfer_t* selected;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, sizeof(*selected), (void**)&selected));
      *selected = (loom_aie2p_native_transfer_t){
          .task = queue->task,
          .source = transfer,
          .completion = group->completion,
          .binding = loom_value_def_index(root),
          .external_view = external_view,
          .local_view = local_view,
          .local_channel = local_channel,
          .local_record = payload_record.value_id,
          .record_dynamic = record_dynamic,
          .descriptor_ordinal = queue->task->transfer_count++};
      *tail = selected;
      tail = &selected->next;
      if (queue->task->last_transfer) {
        queue->task->last_transfer->task_next = selected;
      } else {
        queue->task->first_transfer = selected;
      }
      queue->task->last_transfer = selected;
      queue->task->local_address_dynamic |=
          !autonomous &&
          (record_dynamic || !loom_symbolic_expr_is_constant(
                                 &local_view->projection_byte_offset));
      queue->task->shim_address_dynamic |=
          !autonomous &&
          !loom_symbolic_expr_is_constant(&external_view->begin_byte_offset);
      iree_string_view_t reason;
      if (autonomous) {
        memcpy(selected->shim_words, worker->repetition.shim_words,
               sizeof(selected->shim_words));
      }
      if ((!autonomous &&
           !loom_aie2p_native_descriptor(
               context, source, external, &shim->facts->dma,
               (uint32_t)request->transferred_byte_count, shim->facts->kind,
               selected->shim_words, &reason)) ||
          !loom_aie2p_native_descriptor(
              context, source, local, &local_tile->facts->dma,
              (uint32_t)request->transferred_byte_count,
              local_tile->facts->kind, selected->local_words, &reason)) {
        return loom_aie2p_native_reject(context, request->op, reason);
      }
      // A fixed slot and projection need no per-issue address patch. Reuse
      // preserves that address in the immutable descriptor template.
      if (!autonomous && !record_dynamic &&
          loom_symbolic_expr_is_constant(&local_view->projection_byte_offset)) {
        const uint32_t address =
            local_channel->byte_offset +
            (uint32_t)local_view->projection_byte_offset.constant;
        selected->local_words[layout->address_word] |=
            loom_xdna_register_field_encode_admitted(
                layout->address,
                (layout->local_address_base + address) >>
                    local_tile->facts->dma.address_encoding_shift);
      }
      if (autonomous) {
        selected->local_words[layout->lock_word] |=
            loom_xdna_register_field_encode_admitted(
                layout->locks.acquire_id,
                layout->local_lock_base + local_channel->free_lock) |
            loom_xdna_register_field_encode_admitted(
                layout->locks.acquire_value, -1) |
            loom_xdna_register_field_encode_admitted(
                layout->locks.acquire_enable, 1) |
            loom_xdna_register_field_encode_admitted(
                layout->locks.release_id,
                layout->local_lock_base + local_channel->ready_lock) |
            loom_xdna_register_field_encode_admitted(
                layout->locks.release_value, 1);
      }
      if (configuration) {
        while (source->transport.steps[transport_step].op != request->op) {
          ++transport_step;
        }
        const loom_op_t* preceding =
            transport_step ? source->transport.steps[transport_step - 1].op
                           : NULL;
        if (!ingress && preceding && loom_channel_acquire_isa(preceding) &&
            transport_step + 1 < source->transport.count &&
            source->transport.steps[transport_step + 1].op ==
                group->completion &&
            loom_op_const_results(preceding)[0] == payload_record.value_id) {
          selected->admission = preceding;
          selected->local_words[layout->lock_word] |=
              loom_xdna_register_field_encode_admitted(
                  layout->locks.acquire_id,
                  layout->local_lock_base + local_channel->ready_lock) |
              loom_xdna_register_field_encode_admitted(
                  layout->locks.acquire_value, -1) |
              loom_xdna_register_field_encode_admitted(
                  layout->locks.acquire_enable, 1);
        }
      }
      loom_aie2p_native_binding_t* binding =
          &context->bindings[selected->binding];
      binding->byte_length =
          iree_max(binding->byte_length,
                   (uint64_t)external->end_byte_offset.facts.range_hi);
      binding->byte_alignment =
          iree_max(binding->byte_alignment, external->root_minimum_alignment);
      binding->access |= ingress ? LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ
                                 : LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE;
      const bool group_end =
          j + 1 == stream->transfer_count ||
          stream->transfers[j + 1].group_index != transfer->group_index;
      if (group_end) {
        for (iree_host_size_t k = 0; k < tile_count * 2; ++k) {
          loom_aie2p_native_queue_t* group_queue = &queues[k];
          if (group_queue->group != group) {
            continue;
          }
          bool task_valid;
          IREE_RETURN_IF_ERROR(loom_aie2p_native_select_task_descriptors(
              context, worker, group_queue->task, autonomous, &descriptor_tail,
              &completion_tail, &task_valid));
          if (!task_valid) {
            return iree_ok_status();
          }
          group_queue->group = NULL;
          group_queue->task = NULL;
        }
      }
    }
  }
  // Sharing can turn an initially fixed address into a per-issue patch. Assign
  // relocated storage only after every shim descriptor's address policy is
  // final.
  for (loom_aie2p_native_transfer_t* transfer = worker->transfers; transfer;
       transfer = transfer->next) {
    if (!transfer->shim_descriptor->address_dynamic) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
        context->inventory.pools[worker->tile->pool_index].packing, 8, 8, NULL,
        0, &transfer->base_storage_offset));
    transfer->base_load_address = worker->tile->facts->memory.local_load_base +
                                  (uint32_t)transfer->base_storage_offset;
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_distribution_endpoint(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel,
    loom_xdna_dma_direction_t direction, uint32_t repetitions,
    loom_aie2p_native_distribution_endpoint_t* out_endpoint,
    iree_string_view_t* out_reason, bool* out_valid) {
  *out_valid = false;
  loom_aie2p_native_tile_t* tile = context->pool_tiles[channel->pool_index];
  const loom_xdna_dma_facts_t* dma = &tile->facts->dma;
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  if (!channel->record_dense ||
      (channel->source->capacity > 1 &&
       channel->byte_stride < channel->record_byte_length)) {
    *out_reason = IREE_SV("contiguous non-overlapping channel-copy records");
    return iree_ok_status();
  }
  const uint64_t length_units =
      channel->record_byte_length / dma->transfer_length_granularity;
  const uint64_t minimum_length_units =
      iree_max(1u, dma->transfer_length_offset);
  const uint64_t maximum_length_units =
      (uint64_t)dma->maximum_encoded_transfer_length +
      dma->transfer_length_offset;
  if (channel->record_byte_length % dma->transfer_length_granularity ||
      length_units < minimum_length_units ||
      length_units > maximum_length_units) {
    *out_reason =
        IREE_SV("a channel record length representable by its memory DMA");
    return iree_ok_status();
  }
  const uint64_t step_units =
      channel->byte_stride / dma->transfer_length_granularity;
  if (channel->byte_offset % dma->address_alignment ||
      channel->byte_stride % dma->transfer_length_granularity ||
      channel->source->capacity > (UINT64_C(1) << dma->iteration_bits) ||
      step_units > (UINT64_C(1) << dma->step_size_bits)) {
    *out_reason = IREE_SV(
        "DMA-aligned channel storage with a representable "
        "rotating slot stride");
    return iree_ok_status();
  }
  uint8_t* next_engine = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM
                             ? &tile->next_mm2s
                             : &tile->next_s2mm;
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, *next_engine, &descriptor_base, &descriptor_count);
  if (*next_engine == dma->channel_count_per_direction ||
      *next_descriptor == descriptor_count) {
    *out_reason = IREE_SV("an available on-chip DMA engine and descriptor");
    return iree_ok_status();
  }
  uint32_t words[8] = {0};
  words[0] = length_units - dma->transfer_length_offset;
  const bool sender = direction == LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM;
  words[layout->lock_word] =
      loom_xdna_register_field_encode_admitted(
          layout->locks.acquire_id,
          layout->local_lock_base +
              (sender ? channel->ready_lock : channel->free_lock)) |
      loom_xdna_register_field_encode_admitted(layout->locks.acquire_value,
                                               -1) |
      loom_xdna_register_field_encode_admitted(layout->locks.acquire_enable,
                                               1) |
      loom_xdna_register_field_encode_admitted(
          layout->locks.release_id,
          layout->local_lock_base +
              (sender ? channel->free_lock : channel->ready_lock)) |
      loom_xdna_register_field_encode_admitted(layout->locks.release_value, 1) |
      loom_xdna_register_field_encode_admitted(layout->locks.valid, 1);
  *out_endpoint = (loom_aie2p_native_distribution_endpoint_t){
      .channel = channel, .tile = tile, .engine = (*next_engine)++};
  return loom_aie2p_native_select_sequence(
      context, tile, out_endpoint->engine, direction, repetitions,
      channel->source->capacity, channel->byte_offset, channel->byte_stride,
      words, &out_endpoint->sequence, out_reason, out_valid);
}

static iree_status_t loom_aie2p_native_select_worker_distributions(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, uint32_t* next_source, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_worker_t* source = &realization->workers[worker_index];
  loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  loom_aie2p_native_distribution_t** tail = &worker->distributions;
  for (iree_host_size_t i = 0; i < source->transport.cohort_count; ++i) {
    const loom_pipeline_transport_cohort_t* cohort =
        &source->transport.cohorts[i];
    loom_aie2p_native_distribution_t* distribution;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        context->pass->arena, sizeof(*distribution), (void**)&distribution));
    *distribution = (loom_aie2p_native_distribution_t){
        .receiver_count = cohort->destination_count,
        .repetitions = (uint32_t)source->transport.repetitions,
    };
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, cohort->destination_count,
        sizeof(*distribution->receivers), (void**)&distribution->receivers));
    const loom_pipeline_resource_channel_t* source_binding =
        loom_pipeline_resources_lookup_channel(
            &realization->resources, cohort->admission->channel->value_id);
    const loom_aie2p_native_channel_t* source_channel =
        &context->channels[source_binding - realization->resources.channels];
    iree_string_view_t reason;
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_distribution_endpoint(
        context, source_channel, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
        distribution->repetitions, &distribution->sender, &reason, &valid));
    if (!valid) {
      return loom_aie2p_native_reject(context, cohort->admission->op, reason);
    }
    const uint32_t source_route = (*next_source)++;
    for (iree_host_size_t j = 0; j < cohort->destination_count; ++j) {
      const loom_pipeline_transport_destination_t* destination =
          &cohort->destinations[j];
      const loom_pipeline_resource_channel_t* destination_binding =
          loom_pipeline_resources_lookup_channel(
              &realization->resources, destination->channel->value_id);
      const loom_aie2p_native_channel_t* destination_channel =
          &context->channels[destination_binding -
                             realization->resources.channels];
      IREE_RETURN_IF_ERROR(loom_aie2p_native_select_distribution_endpoint(
          context, destination_channel,
          LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY, distribution->repetitions,
          &distribution->receivers[j], &reason, &valid));
      if (!valid) {
        return loom_aie2p_native_reject(context, destination->copy, reason);
      }
      const loom_aie2p_native_distribution_endpoint_t* sender =
          &distribution->sender;
      loom_aie2p_native_distribution_endpoint_t* receiver =
          &distribution->receivers[j];
      // Existing core/configuration/egress completion retires a receiver only
      // when that direct reader consumes every published record. Distribution
      // readers have no independent completion anchor; observe their input DMA
      // directly instead of reconstructing a transitive dependency graph.
      const loom_aie2p_native_cursor_t* reader =
          &destination_channel->cursor.reader;
      receiver->wait_for_idle =
          reader->worker == UINT32_MAX ||
          reader->maximum_admissions != distribution->repetitions ||
          !realization->workers[reader->worker].block_execution_counts ||
          context->workers[reader->worker].execution ==
              LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION;
      distribution->sender.wait_for_idle |= receiver->wait_for_idle;
      if (sender->tile->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY &&
          sender->tile->coordinate.column !=
              receiver->tile->coordinate.column) {
        return loom_aie2p_native_reject(
            context, destination->copy,
            IREE_SV("same-column channel-copy routes from SRAM storage"));
      }
      if (sender->tile == receiver->tile &&
          (sender->engine >= sender->tile->facts->dma.loopback_channel_count ||
           sender->engine != receiver->engine)) {
        return loom_aie2p_native_reject(
            context, destination->copy,
            IREE_SV("matching DMA channel indices for same-tile channel copy"));
      }
      const iree_host_size_t route_first = context->routing.route_count;
      const uint32_t route = (*next_source)++;
      if (!loom_aie2p_array_route_workers(
              &context->routing, route, source_route,
              distribution->sender.tile->coordinate,
              distribution->sender.engine,
              distribution->receivers[j].tile->coordinate,
              distribution->receivers[j].engine)) {
        return loom_aie2p_native_reject(
            context, destination->copy,
            IREE_SV("available on-die multicast payload links"));
      }
      loom_aie2p_native_route_label(context, route_first, UINT8_MAX);
    }
    *tail = distribution;
    tail = &distribution->next;
  }
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_select_transfers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  iree_arena_allocator_t* arena = context->pass->arena;
  iree_host_size_t transfer_count = 0;
  iree_host_size_t cohort_count = 0;
  iree_host_size_t destination_count = 0;
  for (iree_host_size_t i = 0; i < realization->resources.strand_count; ++i) {
    for (const loom_kernel_async_stream_t* stream =
             realization->workers[i].asynchronous.streams;
         stream; stream = stream->next) {
      transfer_count += stream->transfer_count;
    }
    cohort_count += realization->workers[i].transport.cohort_count;
    for (iree_host_size_t j = 0;
         j < realization->workers[i].transport.cohort_count; ++j) {
      destination_count +=
          realization->workers[i].transport.cohorts[j].destination_count;
    }
  }
  const iree_host_size_t source_count =
      transfer_count * 4 + cohort_count + destination_count;
  const iree_host_size_t route_capacity =
      source_count *
      (context->family->column_count + context->family->row_count + 2);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_initialize(
      context->family, source_count, route_capacity, arena, &context->routing));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, route_capacity,
                                                 sizeof(*context->routes),
                                                 (void**)&context->routes));
  uint32_t next_source = 0;
  uint8_t* next_controller;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, context->family->column_count,
                                           (void**)&next_controller));
  memset(next_controller, 1, context->family->column_count);
  iree_status_t status = iree_ok_status();
  bool valid = true;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && valid &&
                               i < realization->resources.strand_count;
       ++i) {
    if (context->workers[i].execution ==
        LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION) {
      status = loom_aie2p_native_select_worker_distributions(
          context, realization, i, &next_source, &valid);
      continue;
    }
    loom_pipeline_worker_t* worker = &realization->workers[i];
    loom_local_value_domain_restore(&worker->value_domain);
    status = loom_aie2p_native_select_worker_transfers(
        context, realization, i, &next_source, next_controller, &valid);
    loom_local_value_domain_release(&worker->value_domain);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_aie2p_native_switches_build(context, realization->function.op,
                                              out_valid);
  }
  return status;
}
