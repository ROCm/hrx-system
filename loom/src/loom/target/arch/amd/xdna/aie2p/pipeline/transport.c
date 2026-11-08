// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_geometry.h"
#include "loom/ops/channel/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_resources.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"
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
uint16_t* loom_aie2p_native_descriptor_bank(loom_aie2p_native_tile_t* tile,
                                            uint8_t engine, uint16_t* out_base,
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
    uint32_t bytes, uint32_t records, loom_xdna_tile_kind_t kind,
    uint32_t* words, iree_string_view_t* reason) {
  loom_storage_geometry_t geometry;
  if (!loom_storage_geometry_query(&worker->facts.context, context->code.module,
                                   endpoint->type, &geometry)) {
    *reason = IREE_SV("specialized DMA endpoint shape and strides");
    return false;
  }
  if (records > 1) {
    // Retained-batch admission proved whole contiguous channel records.
    geometry.layout_kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE;
  }
  return loom_aie2p_native_encode_descriptor(
      context->family, &geometry, dma, bytes, records, kind, words, reason);
}

bool loom_aie2p_native_select_ingress_descriptor(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_worker_t* source,
    const loom_movement_request_t* request, uint32_t byte_stride,
    uint32_t repetitions, uint32_t period, loom_aie2p_native_worker_t* worker) {
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
      period, worker->repetition.shim_words,
      &worker->repetition.shim_repetitions);
}

// Labels appended edges, including reused prefixes retained by packet routes.
// Distinct packet classes share one physical source tree and carry their own
// filter identity along its complete path.
void loom_aie2p_native_route_label(loom_aie2p_native_context_t* context,
                                   iree_host_size_t first, uint8_t packet,
                                   uint32_t control_source) {
  for (iree_host_size_t i = first; i < context->routing.route_count; ++i) {
    context->routes[i] =
        (loom_aie2p_native_route_t){.edge = &context->routing.routes[i],
                                    .packet = packet,
                                    .control_source = control_source};
  }
}

bool loom_aie2p_native_packet_route(loom_aie2p_native_context_t* context,
                                    uint32_t* next_source,
                                    loom_xdna_tile_coordinate_t source,
                                    loom_xdna_stream_port_t source_port,
                                    loom_xdna_tile_coordinate_t destination,
                                    loom_xdna_stream_port_t destination_port,
                                    uint8_t packet) {
  const iree_host_size_t first = context->routing.route_count;
  const uint32_t index = (*next_source)++;
  const iree_host_size_t tile =
      source.column * context->family->row_count + source.row;
  const uint32_t source_index =
      tile * 2 + (source_port == LOOM_XDNA_STREAM_PORT_TILE_CONTROL);
  loom_aie2p_native_control_source_t* control =
      &context->control_sources[source_index];
  const uint32_t destination_index =
      destination_port == LOOM_XDNA_STREAM_PORT_CORE ||
              destination_port == LOOM_XDNA_STREAM_PORT_TILE_CONTROL
          ? (destination.column * context->family->row_count +
             destination.row) *
                    2 +
                (destination_port == LOOM_XDNA_STREAM_PORT_TILE_CONTROL)
          : UINT32_MAX;
  if (control->route == UINT32_MAX) {
    control->route = index;
    control->destination = destination_index;
  } else if (control->destination != destination_index) {
    control->destination = UINT32_MAX;
  }
  if (destination_index != UINT32_MAX) {
    uint32_t* owner = &context->control_destination_sources[destination_index];
    if (*owner == UINT32_MAX) {
      *owner = source_index;
    } else if (*owner != source_index) {
      *owner = UINT32_MAX - 1;
    }
  }
  const iree_host_size_t destination_tile =
      destination.column * context->family->row_count + destination.row;
  loom_aie2p_native_delivery_t* delivery =
      context->deliveries && source_port == LOOM_XDNA_STREAM_PORT_CORE &&
              destination_port == LOOM_XDNA_STREAM_PORT_TILE_CONTROL
          ? context->deliveries[destination_tile]
          : NULL;
  const bool routed =
      delivery ? loom_aie2p_array_route_control(&context->routing, index,
                                                control->route, source,
                                                &delivery->sharing)
               : loom_aie2p_array_route_stream(
                     &context->routing, index, control->route, source,
                     source_port, 0, destination, destination_port, 0, true);
  if (!routed) {
    return false;
  }
  if (delivery && delivery->sharing.used) {
    // A private endpoint still needs packet filtering when any path to it
    // borrows payload lanes. Retain that exclusion for the entire source tree.
    control->destination = UINT32_MAX;
  }
  loom_aie2p_native_route_label(context, first, packet, source_index);
  return true;
}

iree_status_t loom_aie2p_native_select_control_route(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, const loom_aie2p_native_tile_t* tile,
    bool reply, uint32_t* next_source, uint8_t* next_controller,
    bool* out_valid) {
  *out_valid = false;
  loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  loom_aie2p_native_worker_t* owner =
      &context->workers[context->execution
                            .workers[context->core_engines[worker_index]]];
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
  if (!owner->control_packets) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        context->pass->arena, tile_count * 2, (void**)&owner->control_packets));
    memset(owner->control_packets, UINT8_MAX, tile_count * 2);
    owner->control_replies = owner->control_packets + tile_count;
  }
  const iree_host_size_t index = tile - context->tiles;
  if (owner->control_packets[index] == UINT8_MAX) {
    if (owner->control_packet_count == 4 ||
        !loom_aie2p_native_packet_route(
            context, next_source, worker->tile->coordinate,
            LOOM_XDNA_STREAM_PORT_CORE, tile->coordinate,
            LOOM_XDNA_STREAM_PORT_TILE_CONTROL, owner->control_packet_count)) {
      return loom_aie2p_native_reject(
          context, realization->workers[worker_index].source->declaration,
          IREE_SV("available worker control-packet routes"));
    }
    owner->control_packets[index] = owner->control_packet_count++;
  }
  if (reply && owner->control_replies[index] == UINT8_MAX) {
    uint8_t* packet = &next_controller[tile->coordinate.column];
    if (*packet == 32 ||
        !loom_aie2p_native_packet_route(context, next_source, tile->coordinate,
                                        LOOM_XDNA_STREAM_PORT_TILE_CONTROL,
                                        worker->tile->coordinate,
                                        LOOM_XDNA_STREAM_PORT_CORE, *packet)) {
      return loom_aie2p_native_reject(
          context, realization->workers[worker_index].source->declaration,
          IREE_SV("a tile-control status reply route to its resident core"));
    }
    owner->control_replies[index] = (*packet)++;
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_allocate_task_descriptor(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* worker,
    loom_aie2p_native_worker_t* resource_owner,
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
  if (!shim && task->path->ingress && !task->path->completion_tokens) {
    descriptor->completion = loom_aie2p_native_reusable_completion(
        resource_owner, tile, task->first_transfer);
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
    loom_aie2p_native_worker_t* resource_owner,
    loom_aie2p_native_transfer_task_t* task, bool autonomous,
    loom_aie2p_native_dma_descriptor_t*** descriptor_tail,
    loom_aie2p_native_dma_completion_t*** completion_tail, bool* out_valid) {
  *out_valid = false;
  if (autonomous) {
    // Autonomous repetition admits one source transfer and builds its immutable
    // ring program through the sequence planner.
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
            LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM, NULL,
            worker->repetition.shim_repetitions, 1, 0, 0, 1,
            transfer->shim_words, &descriptor->sequence, &reason, &valid));
      } else {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_sequence(
            context, tile, task->path->local_engine,
            LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY, transfer->local_channel,
            worker->repetition.count,
            transfer->local_channel->replay.record_count
                ? 1
                : transfer->local_channel->source->capacity,
            transfer->local_channel->byte_offset +
                (uint32_t)transfer->local_view->projection_byte_offset.constant,
            transfer->local_channel->byte_stride, 1, transfer->local_words,
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
        loom_aie2p_native_reusable_descriptor(resource_owner, task, false);
    if (!task->local_descriptor) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_allocate_task_descriptor(
          context, worker, resource_owner, task, false, descriptor_tail,
          completion_tail, &task->local_descriptor));
      if (!task->local_descriptor) {
        return iree_ok_status();
      }
    }
    task->shim_descriptor =
        loom_aie2p_native_reusable_descriptor(resource_owner, task, true);
    if (!task->shim_descriptor) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_allocate_task_descriptor(
          context, worker, resource_owner, task, true, descriptor_tail,
          completion_tail, &task->shim_descriptor));
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

iree_status_t loom_aie2p_native_select_sequence(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction,
    const loom_aie2p_native_channel_t* progress_channel, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    uint32_t credit_interval, const uint32_t template_words[8],
    loom_aie2p_native_dma_sequence_t* out_sequence,
    iree_string_view_t* out_reason, bool* out_valid) {
  uint16_t descriptor_base, descriptor_count;
  uint16_t* next_descriptor = loom_aie2p_native_descriptor_bank(
      tile, engine, &descriptor_base, &descriptor_count);
  IREE_RETURN_IF_ERROR(loom_aie2p_native_build_sequence(
      &tile->facts->dma, tile->memory_layout,
      descriptor_base + *next_descriptor, descriptor_count - *next_descriptor,
      direction, repetitions, capacity, byte_offset, byte_stride,
      credit_interval, template_words, context->pass->arena, out_sequence,
      out_reason, out_valid));
  if (!*out_valid) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_refill(
        context, tile, engine, direction, progress_channel, repetitions,
        capacity, byte_offset, byte_stride, credit_interval,
        descriptor_base + *next_descriptor, descriptor_count - *next_descriptor,
        template_words, out_sequence, out_reason, out_valid));
  }
  if (*out_valid) {
    *next_descriptor += out_sequence->descriptor_count;
  }
  return iree_ok_status();
}

// The selected controller owns a complete retained batch from its first
// reservation until its full-prefix publication. Dense adjacent fills can
// complete together at the first wait: no reader observes an intermediate
// record, and no other observable handoff may occur inside this interval.
// Independent tile fills may interleave inside that same unpublished interval.
// This consumes canonical transport, completion, and physical slot facts;
// emitted code never reconstructs the grouping from the source operations.
static uint32_t loom_aie2p_native_ingress_records(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* source,
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_channel_t* channel,
    const loom_kernel_async_stream_t* stream, iree_host_size_t first_transfer,
    iree_host_size_t first_step, iree_host_size_t completed_groups,
    iree_host_size_t* members) {
  const loom_module_t* module = context->code.module;
  const uint32_t count = channel->replay.record_count;
  if (count < 2 || !channel->record_dense ||
      channel->byte_stride != channel->record_byte_length ||
      stream->transfer_count - first_transfer < count ||
      stream->transfers[first_transfer].group_index != completed_groups) {
    return 1;
  }
  // Reading later records early must not cross an outstanding external write.
  // Earlier members of this group can only be independent ingress copies.
  const loom_kernel_async_group_t* first_group =
      &stream->groups[completed_groups];
  for (iree_host_size_t i = first_group->first_transfer; i < first_transfer;
       ++i) {
    const loom_movement_request_t* pending = &stream->transfers[i].request;
    loom_channel_payload_record_t record;
    if (pending->kind != LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY ||
        pending->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
        pending->dest.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
        !loom_channel_completion_try_lookup_payload_record(
            &source->completion, pending->dest.value_id, &record)) {
      return 1;
    }
    const loom_channel_identity_t* identity =
        loom_channel_plan_channel(&source->channels, record.value_id);
    const loom_aie2p_native_channel_t* other_channel =
        loom_aie2p_native_channel(context, realization, identity);
    if (other_channel->pool_index == channel->pool_index) {
      return 1;
    }
  }
  const loom_movement_endpoint_t* first =
      &stream->transfers[first_transfer].request.source;
  uint32_t copies = 0;
  for (iree_host_size_t i = first_step; i < source->transport.count; ++i) {
    const loom_pipeline_transport_step_t* step = &source->transport.steps[i];
    if (step->op->parent_block != stream->block) {
      return 1;
    }
    if (step->kind == LOOM_PIPELINE_TRANSPORT_STEP_WAIT) {
      continue;
    }
    if (step->kind == LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL) {
      const loom_channel_plan_action_t* source_action = step->source.channel;
      const loom_aie2p_native_configuration_action_t* action =
          &worker->configuration_actions[source_action -
                                         source->channels.actions];
      if ((source_action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE &&
           source_action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_PUBLISH) ||
          action->expected >= 0) {
        return 1;
      }
      if (action->value >= 0) {
        return copies == count ? count : 1;
      }
      continue;
    }
    const loom_kernel_async_transfer_t* transfer = step->source.transfer;
    const loom_movement_request_t* request = &transfer->request;
    if (request->kind != LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY ||
        request->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
        request->dest.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      return 1;
    }
    loom_channel_payload_record_t record;
    if (!loom_channel_completion_try_lookup_payload_record(
            &source->completion, request->dest.value_id, &record)) {
      return 1;
    }
    const loom_channel_identity_t* identity =
        loom_channel_plan_channel(&source->channels, record.value_id);
    if (identity != &channel->source->identity) {
      const loom_aie2p_native_channel_t* other_channel =
          loom_aie2p_native_channel(context, realization, identity);
      if (other_channel->pool_index == channel->pool_index) {
        return 1;
      }
      continue;
    }
    const uint64_t copy_byte_offset = (uint64_t)copies * channel->byte_stride;
    if (copy_byte_offset > UINT32_MAX) {
      return 1;
    }
    const int64_t offset =
        first->begin_byte_offset.facts.range_lo + (int64_t)copy_byte_offset;
    if (copies == count ||
        request->source.root_value_id != first->root_value_id ||
        request->transferred_byte_count != channel->record_byte_length ||
        request->source.begin_byte_offset.facts.range_lo != offset ||
        request->source.end_byte_offset.facts.range_hi !=
            offset + channel->record_byte_length ||
        request->source.end_byte_offset.facts.range_hi > UINT32_MAX) {
      return 1;
    }
    const loom_view_region_t* local_view;
    loom_view_region_table_try_lookup(
        &source->asynchronous.movement.view_regions, request->dest.value_id,
        &local_view);
    if (local_view->projection_byte_offset.constant != 0) {
      return 1;
    }
    const loom_value_id_t reservation =
        loom_channel_plan_admission(&source->channels, record.value_id);
    if (worker->configuration_records[loom_local_value_domain_ordinal(
            &source->value_domain, reservation)] !=
        (uint32_t)copy_byte_offset) {
      return 1;
    }
    const loom_type_t endpoints[] = {request->source.type, request->dest.type};
    for (iree_host_size_t endpoint = 0; endpoint < IREE_ARRAYSIZE(endpoints);
         ++endpoint) {
      loom_storage_geometry_t geometry;
      loom_storage_geometry_span_t span;
      if (!loom_storage_geometry_query(&source->facts.context, module,
                                       endpoints[endpoint], &geometry) ||
          !loom_storage_geometry_measure(&geometry, 0, &span) || !span.dense) {
        return 1;
      }
    }
    members[copies++] = transfer - stream->transfers;
  }
  return 1;
}

static iree_status_t loom_aie2p_native_select_worker_transfers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t i,
    uint32_t* next_source, uint8_t* next_controller, bool* out_valid) {
  *out_valid = false;
  iree_arena_allocator_t* arena = context->pass->arena;
  loom_module_t* module = context->code.module;
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
  const loom_block_t* entry =
      loom_region_entry_block(loom_func_like_body(realization->function));
  const loom_pipeline_worker_t* source = &realization->workers[i];
  loom_aie2p_native_worker_t* worker = &context->workers[i];
  if (!source->asynchronous.streams) {
    *out_valid = true;
    return iree_ok_status();
  }
  // Ordered occurrences on one core share its finite engines, routes, and
  // descriptor banks. The common execution plan has already proved that their
  // complete lifetimes do not overlap; source groups retain their own tasks.
  loom_aie2p_native_worker_t* resource_owner =
      worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE
          ? &context
                 ->workers[context->execution.workers[context->core_engines[i]]]
          : worker;
  const bool configuration =
      worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
  const bool autonomous = worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_DMA;
  loom_aie2p_native_queue_t* queues;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count * 2, sizeof(*queues), (void**)&queues));
  memset(queues, 0, tile_count * 2 * sizeof(*queues));
  if (!resource_owner->paths_by_tile) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, tile_count * 2, sizeof(*resource_owner->paths_by_tile),
        (void**)&resource_owner->paths_by_tile));
    memset(resource_owner->paths_by_tile, 0,
           tile_count * 2 * sizeof(*resource_owner->paths_by_tile));
  }
  iree_host_size_t transport_step = 0;
  loom_aie2p_native_transfer_t** tail = &worker->transfers;
  loom_aie2p_native_transfer_task_t** task_tail = &worker->transfer_tasks;
  loom_aie2p_native_dma_descriptor_t** descriptor_tail =
      &resource_owner->descriptors;
  while (*descriptor_tail) {
    descriptor_tail = &(*descriptor_tail)->next;
  }
  loom_aie2p_native_dma_completion_t** completion_tail =
      &resource_owner->dma_completions;
  while (*completion_tail) {
    completion_tail = &(*completion_tail)->next;
  }
  iree_host_size_t* members = NULL;
  uint32_t member_capacity = 0;
  iree_host_size_t stream_index = 0;
  for (const loom_kernel_async_stream_t* stream =
           configuration ? source->transport.streams[0]
                         : source->asynchronous.streams;
       stream;
       stream = configuration ? (++stream_index < source->transport.stream_count
                                     ? source->transport.streams[stream_index]
                                     : NULL)
                              : stream->next) {
    bool* merged = NULL;
    if (configuration) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, stream->transfer_count, sizeof(*merged), (void**)&merged));
      memset(merged, 0, stream->transfer_count * sizeof(*merged));
    }
    iree_host_size_t completed_groups = 0;
    for (iree_host_size_t j = 0; j < stream->transfer_count; ++j) {
      if (merged && merged[j]) {
        continue;
      }
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
      const loom_aie2p_native_channel_t* local_channel =
          payload_channel
              ? loom_aie2p_native_channel(context, realization, payload_channel)
              : NULL;
      const bool record_dynamic = !configuration && local_channel &&
                                  local_channel->source->capacity > 1 &&
                                  payload_record.maximum_admissions > 1;
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
      uint32_t record_offset = 0;
      if (configuration) {
        const loom_value_id_t admission = loom_channel_plan_admission(
            &source->channels, payload_record.value_id);
        record_offset =
            worker->configuration_records[loom_local_value_domain_ordinal(
                &source->value_domain, admission)];
        while (source->transport.steps[transport_step].op != request->op) {
          const loom_op_t* op = source->transport.steps[transport_step].op;
          while (completed_groups < stream->group_count &&
                 stream->groups[completed_groups].completion == op) {
            ++completed_groups;
          }
          ++transport_step;
        }
        if (local_channel->replay.record_count > member_capacity) {
          member_capacity = local_channel->replay.record_count;
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              arena, member_capacity, sizeof(*members), (void**)&members));
        }
      }
      const uint32_t source_count =
          configuration && ingress && !record_offset
              ? loom_aie2p_native_ingress_records(
                    context, realization, source, worker, local_channel, stream,
                    j, transport_step, completed_groups, members)
              : 1;
      for (uint32_t k = 1; k < source_count; ++k) {
        merged[members[k]] = true;
      }
      const loom_aie2p_native_memory_layout_t* layout =
          local_tile->memory_layout;
      loom_aie2p_native_tile_t* shim =
          &context->tiles[(configuration ? local_tile : worker->tile)
                              ->coordinate.column *
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
      loom_aie2p_native_dma_path_t** shared_path =
          &resource_owner->paths_by_tile[local_index * 2 + (ingress ? 0 : 1)];
      queue->path = *shared_path;
      if (!queue->path) {
        uint8_t* local_engine =
            ingress ? &local_tile->next_s2mm : &local_tile->next_mm2s;
        uint8_t* shim_engine = ingress ? &shim->next_mm2s : &shim->next_s2mm;
        // A full descriptor bank cannot serve any additional engine of that
        // parity. Keep allocation within a bank that still has storage.
        while (*local_engine <
               local_tile->facts->dma.channel_count_per_direction) {
          uint16_t descriptor_base, descriptor_count;
          const uint16_t* used = loom_aie2p_native_descriptor_bank(
              local_tile, *local_engine, &descriptor_base, &descriptor_count);
          if (*used < descriptor_count) {
            break;
          }
          ++*local_engine;
        }
        if (*local_engine ==
                local_tile->facts->dma.channel_count_per_direction ||
            *shim_engine == shim->facts->dma.channel_count_per_direction ||
            (!ingress && resource_owner->has_egress && !configuration)) {
          return loom_aie2p_native_reject(
              context, request->op,
              IREE_SV("an independent DMA engine pair and one external "
                      "completion stream per worker"));
        }
        IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*queue->path),
                                                 (void**)&queue->path));
        *queue->path = (loom_aie2p_native_dma_path_t){
            .next = resource_owner->paths,
            .local = local_tile,
            .shim = shim,
            .local_engine = (*local_engine)++,
            .shim_engine = (*shim_engine)++,
            .completion_tokens =
                !configuration && !autonomous &&
                        local_tile->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY
                    ? 2
                    : (ingress ? 0 : 1),
            .ingress = ingress};
        resource_owner->paths = queue->path;
        *shared_path = queue->path;
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
        loom_aie2p_native_route_label(context, route_first, UINT8_MAX,
                                      UINT32_MAX);
        loom_aie2p_native_tile_t* destinations[] = {local_tile, shim};
        uint8_t* packets[] = {&queue->path->control.local,
                              &queue->path->control.shim};
        for (unsigned k = 0;
             worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE && k < 2;
             ++k) {
          bool valid = false;
          IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
              context, realization, i, destinations[k], false, next_source,
              next_controller, &valid));
          if (!valid) {
            return iree_ok_status();
          }
          *packets[k] =
              resource_owner->control_packets[destinations[k] - context->tiles];
        }
        if (queue->path->completion_tokens) {
          resource_owner->has_egress |= !ingress;
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
          if (queue->path->completion_tokens == 2 &&
              !loom_aie2p_native_packet_route(
                  context, next_source, local_tile->coordinate,
                  LOOM_XDNA_STREAM_PORT_TILE_CONTROL, worker->tile->coordinate,
                  LOOM_XDNA_STREAM_PORT_CORE, queue->path->completion_packet)) {
            return loom_aie2p_native_reject(
                context, request->op,
                IREE_SV("an SRAM task-completion route to its issuing core"));
          }
          if (!configuration) {
            resource_owner->control_replies[shim - context->tiles] =
                queue->path->completion_packet;
            if (queue->path->completion_tokens == 2) {
              resource_owner->control_replies[local_tile - context->tiles] =
                  queue->path->completion_packet;
            }
          }
        }
      }
      if (!queue->capacity) {
        queue->capacity = iree_min(local_tile->facts->dma.task_queue_depth,
                                   shim->facts->dma.task_queue_depth);
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            arena, queue->capacity, sizeof(*queue->completions),
            (void**)&queue->completions));
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
          .worker_index = i,
          .binding = loom_value_def_index(root),
          .source_count = source_count,
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
               (uint32_t)request->transferred_byte_count, source_count,
               shim->facts->kind, selected->shim_words, &reason)) ||
          !loom_aie2p_native_descriptor(
              context, source, local, &local_tile->facts->dma,
              (uint32_t)request->transferred_byte_count,
              autonomous && local_channel->replay.record_count
                  ? local_channel->replay.record_count
                  : source_count,
              local_tile->facts->kind, selected->local_words, &reason)) {
        return loom_aie2p_native_reject(context, request->op, reason);
      }
      // A fixed slot and projection need no per-issue address patch. Reuse
      // preserves that address in the immutable descriptor template.
      if (!autonomous && !record_dynamic &&
          loom_symbolic_expr_is_constant(&local_view->projection_byte_offset)) {
        const uint32_t address =
            local_channel->byte_offset + record_offset +
            (uint32_t)local_view->projection_byte_offset.constant;
        selected->local_words[layout->address_word] |=
            loom_xdna_register_field_encode(
                layout->address,
                (layout->local_address_base + address) >>
                    local_tile->facts->dma.address_encoding_shift);
      }
      if (autonomous) {
        const int32_t credits = local_channel->replay.record_count
                                    ? (int32_t)local_channel->replay.credits
                                    : 1;
        selected->local_words[layout->lock_word] |=
            loom_xdna_register_field_encode(
                layout->locks.acquire_id,
                layout->local_lock_base + local_channel->free_lock) |
            loom_xdna_register_field_encode(layout->locks.acquire_value,
                                            -credits) |
            loom_xdna_register_field_encode(layout->locks.acquire_enable, 1) |
            loom_xdna_register_field_encode(
                layout->locks.release_id,
                layout->local_lock_base + local_channel->ready_lock) |
            loom_xdna_register_field_encode(layout->locks.release_value,
                                            credits);
      }
      if (configuration) {
        const loom_pipeline_transport_step_t* preceding =
            transport_step ? &source->transport.steps[transport_step - 1]
                           : NULL;
        if (!ingress && preceding &&
            preceding->kind == LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL &&
            preceding->source.channel->kind ==
                LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE &&
            transport_step + 1 < source->transport.count &&
            source->transport.steps[transport_step + 1].op ==
                group->completion &&
            loom_op_const_results(preceding->op)[0] ==
                payload_record.value_id) {
          selected->admission = preceding->op;
          selected->local_words[layout->lock_word] |=
              loom_xdna_register_field_encode(
                  layout->locks.acquire_id,
                  layout->local_lock_base + local_channel->ready_lock) |
              loom_xdna_register_field_encode(layout->locks.acquire_value, -1) |
              loom_xdna_register_field_encode(layout->locks.acquire_enable, 1);
        }
      }
      loom_aie2p_native_binding_t* binding =
          &context->bindings[selected->binding];
      binding->byte_length = iree_max(
          binding->byte_length,
          (uint64_t)external->end_byte_offset.facts.range_hi +
              (uint64_t)(source_count - 1) * local_channel->byte_stride);
      binding->byte_alignment =
          iree_max(binding->byte_alignment, external->root_minimum_alignment);
      binding->access |= ingress ? LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ
                                 : LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE;
      const bool group_end =
          source_count > 1 || j + 1 == stream->transfer_count ||
          stream->transfers[j + 1].group_index != transfer->group_index;
      if (group_end) {
        for (iree_host_size_t k = 0; k < tile_count * 2; ++k) {
          loom_aie2p_native_queue_t* group_queue = &queues[k];
          if (group_queue->group != group) {
            continue;
          }
          bool task_valid;
          IREE_RETURN_IF_ERROR(loom_aie2p_native_select_task_descriptors(
              context, worker, resource_owner, group_queue->task, autonomous,
              &descriptor_tail, &completion_tail, &task_valid));
          if (!task_valid) {
            return iree_ok_status();
          }
          group_queue->group = NULL;
          group_queue->task = NULL;
        }
      }
    }
  }
  // One core receives a single ordered token stream. Separate task paths may
  // share it only when the previous wait retires before the next issue, or
  // when the same source wait retires every task together.
  for (const loom_aie2p_native_transfer_task_t* task =
           worker->execution == LOOM_AIE2P_NATIVE_EXECUTION_CORE
               ? worker->transfer_tasks
               : NULL;
       task; task = task->next) {
    if (!task->path->completion_tokens) {
      continue;
    }
    const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
    for (const loom_aie2p_native_dma_path_t* path = resource_owner->paths; path;
         path = path->next) {
      const loom_aie2p_native_transfer_t* previous = path->last_token_transfer;
      if (previous && path != task->path &&
          previous->completion != transfer->completion &&
          !loom_aie2p_native_transfer_retired(previous, i,
                                              transfer->source->request.op)) {
        return loom_aie2p_native_reject(
            context, transfer->source->request.op,
            IREE_SV("ordered retirement of DMA paths sharing one core token "
                    "stream"));
      }
    }
    task->path->last_token_transfer = transfer;
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_reserve_binding(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* owner,
    loom_aie2p_native_transfer_t* transfer) {
  if (!owner->relocations) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, context->binding_count,
        sizeof(*owner->relocations), (void**)&owner->relocations));
    memset(owner->relocations, 0,
           context->binding_count * sizeof(*owner->relocations));
  }
  loom_aie2p_native_relocation_t* relocation =
      &owner->relocations[transfer->binding];
  if (!relocation->first_transfer) {
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_reserve(
        context->inventory.pools[owner->tile->pool_index].packing, 8, 8, NULL,
        0, &relocation->storage_offset));
    relocation->load_address = owner->tile->facts->memory.local_load_base +
                               (uint32_t)relocation->storage_offset;
    relocation->first_transfer = transfer;
  }
  transfer->relocation = relocation;
  return iree_ok_status();
}

// A later child can promote an earlier child's immutable external address to a
// per-issue patch. Reserve relocated bindings after all occurrences have chosen
// their shared descriptors, so every generated submit observes the final
// policy. Every dynamic copy adds its complete offset to an immutable caller
// base, so one relocated address per core and binding serves all occurrences.
static iree_status_t loom_aie2p_native_reserve_transfer_bindings(
    loom_aie2p_native_context_t* context, iree_host_size_t worker_count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < worker_count;
       ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    loom_aie2p_native_worker_t* owner =
        &context->workers[context->execution.workers[context->core_engines[i]]];
    for (loom_aie2p_native_transfer_t* transfer = worker->transfers;
         iree_status_is_ok(status) && transfer; transfer = transfer->next) {
      if (!transfer->shim_descriptor->address_dynamic) {
        continue;
      }
      status = loom_aie2p_native_reserve_binding(context, owner, transfer);
    }
  }
  return status;
}

// Final descriptor policies define the submission signature. A local
// descriptor already proves matching routes, geometry, and local patch words.
// Its physical core owns the complete program, including ordered occurrences
// that reuse the endpoint after the preceding phase has retired.
static iree_status_t loom_aie2p_native_group_submissions(
    loom_aie2p_native_context_t* context, iree_host_size_t worker_count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < worker_count;
       ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    for (loom_aie2p_native_transfer_task_t* task = worker->transfer_tasks;
         iree_status_is_ok(status) && task; task = task->next) {
      // Queue-only and local-only patches are too short to reliably amortize
      // native call overhead. Their constants also share within the caller.
      if (!task->shim_descriptor->address_dynamic) {
        continue;
      }
      const loom_aie2p_call_abi_requirements_t requirements = {
          .scalar_count =
              (uint16_t)(task->transfer_count *
                             (2 + task->local_descriptor->address_dynamic) +
                         1),
          .pointer_count = (uint16_t)task->transfer_count,
      };
      if (!loom_aie2p_call_abi_requirements_fit(requirements)) {
        continue;
      }
      loom_aie2p_native_dma_submit_t* group =
          task->local_descriptor->submit_group;
      if (!group) {
        status = iree_arena_allocate(context->pass->arena, sizeof(*group),
                                     (void**)&group);
        if (iree_status_is_ok(status)) {
          *group = (loom_aie2p_native_dma_submit_t){.function =
                                                        loom_symbol_ref_null()};
          task->local_descriptor->submit_group = group;
        }
      }
      if (iree_status_is_ok(status)) {
        ++group->use_count;
        task->submit_group = group;
      }
    }
  }
  return status;
}

iree_status_t loom_aie2p_native_select_transfers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  iree_arena_allocator_t* arena = context->pass->arena;
  iree_host_size_t transfer_count = 0;
  iree_host_size_t cohort_count = 0;
  iree_host_size_t destination_count = 0;
  iree_host_size_t child_control_count = 0;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    for (const loom_kernel_async_stream_t* stream =
             realization->workers[i].asynchronous.streams;
         stream; stream = stream->next) {
      transfer_count += stream->transfer_count;
    }
    cohort_count += realization->workers[i].transport.cohort_count;
    const bool child =
        realization->invocations.workers[i].invocation->ordinal != 0;
    for (iree_host_size_t j = 0;
         j < realization->workers[i].transport.cohort_count; ++j) {
      destination_count +=
          realization->workers[i].transport.cohorts[j].destination_count;
      if (child) {
        child_control_count +=
            2 *
            (1 +
             realization->workers[i].transport.cohorts[j].destination_count);
      }
    }
  }
  iree_host_size_t source_count =
      transfer_count * 4 + cohort_count + destination_count +
      child_control_count +
      context->collective_count * context->execution.engine_count;
  for (iree_host_size_t i = 0; i < realization->invocations.channel_count;
       ++i) {
    // One receiver and one sender queue per equal-ordinal channel. Shared
    // ordered phases retain their own task admission and do not use refills.
    source_count += context->channels[i].progress ? 2 : 0;
  }
  const iree_host_size_t route_capacity =
      source_count *
      (context->family->column_count + context->family->row_count + 2);
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_initialize(
      context->family, source_count, route_capacity, arena, &context->routing));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, route_capacity,
                                                 sizeof(*context->routes),
                                                 (void**)&context->routes));
  const iree_host_size_t control_source_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count * 2;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, control_source_count, sizeof(*context->control_sources),
      (void**)&context->control_sources));
  memset(context->control_sources, 0xff,
         control_source_count * sizeof(*context->control_sources));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, control_source_count,
                                sizeof(*context->control_destination_sources),
                                (void**)&context->control_destination_sources));
  memset(context->control_destination_sources, 0xff,
         control_source_count * sizeof(*context->control_destination_sources));
  uint32_t next_source = 0;
  uint8_t* next_controller;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, context->family->column_count,
                                           (void**)&next_controller));
  memset(next_controller, 1, context->family->column_count);
  iree_status_t status = iree_ok_status();
  bool valid = true;
  bool ordered_distributions = false;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    if (context->workers[i].execution !=
        LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION) {
      continue;
    }
    loom_aie2p_native_worker_t* owner =
        &context->workers[context->distribution_execution
                              .workers[context->distribution_engines[i]]];
    const loom_pipeline_transport_t* transport =
        &realization->workers[i].transport;
    for (iree_host_size_t j = 0; j < transport->batch_count; ++j) {
      owner->distribution_packets |= transport->batches[j].replays == 0;
    }
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && valid &&
                               i < realization->invocations.worker_count;
       ++i) {
    const loom_pipeline_transport_t* transport =
        &realization->workers[i].transport;
    for (iree_host_size_t j = 0; j < transport->batch_count; ++j) {
      if (transport->batches[j].records->release) {
        status = loom_aie2p_native_select_worker_distributions(
            context, realization, i, &next_source, &valid);
        ordered_distributions |=
            context->workers[i].distribution_phases != NULL;
        break;
      }
    }
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && valid &&
                               i < realization->invocations.worker_count;
       ++i) {
    if (context->workers[i].execution ==
        LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION) {
      if (context->workers[i].distributions ||
          context->workers[i].distribution_phases) {
        continue;
      }
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
  if (iree_status_is_ok(status) && valid && ordered_distributions) {
    status = loom_aie2p_native_bind_distribution_phases(context, realization,
                                                        &valid);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && valid &&
                               i < realization->invocations.worker_count;
       ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
      continue;
    }
    loom_aie2p_native_worker_t* owner =
        &context->workers[context->execution.workers[context->core_engines[i]]];
    for (loom_aie2p_native_distribution_command_t* command =
             worker->distribution_commands;
         iree_status_is_ok(status) && valid && command;
         command = command->next) {
      status = loom_aie2p_native_select_control_route(
          context, realization, i, command->endpoint->tile, true, &next_source,
          next_controller, &valid);
      if (iree_status_is_ok(status) && valid) {
        loom_aie2p_native_retirement_t* group = owner->retirements;
        for (; group; group = group->next) {
          const loom_aie2p_native_distribution_command_t* source =
              group->source;
          if (source->endpoint->tile == command->endpoint->tile &&
              source->endpoint->engine == command->endpoint->engine &&
              source->direction == command->direction) {
            break;
          }
        }
        if (!group) {
          status = iree_arena_allocate(arena, sizeof(*group), (void**)&group);
          if (iree_status_is_ok(status)) {
            *group = (loom_aie2p_native_retirement_t){
                .next = owner->retirements,
                .source = command,
                .function = loom_symbol_ref_null()};
            owner->retirements = group;
          }
        }
        if (iree_status_is_ok(status)) {
          ++group->use_count;
          command->retirement = group;
        }
      }
    }
    for (iree_host_size_t j = 0; j < worker->channels.count; ++j) {
      const loom_aie2p_native_channel_access_t* access =
          &worker->channels.accesses[j];
      if (access->window) {
        continue;
      }
      const iree_host_size_t tile =
          context->pool_tiles[access->channel->pool_index] - context->tiles;
      if (!owner->control_packets ||
          owner->control_packets[tile] == UINT8_MAX) {
        return loom_aie2p_native_reject(
            context, realization->workers[i].source->declaration,
            IREE_SV("an SRAM DMA path to carry complete-batch publication"));
      }
    }
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_aie2p_native_reserve_transfer_bindings(
        context, realization->invocations.worker_count);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_aie2p_native_prepare_sharing(context, realization);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_aie2p_native_route_refills(context, realization, &next_source,
                                             &valid);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_aie2p_native_group_submissions(
        context, realization->invocations.worker_count);
  }
  if (iree_status_is_ok(status) && valid) {
    status = loom_aie2p_native_route_collectives(context, &next_source, &valid);
  }
  if (iree_status_is_ok(status) && valid) {
    loom_aie2p_native_commit_sharing(context);
    status = loom_aie2p_native_switches_build(context, realization->function.op,
                                              out_valid);
  }
  return status;
}
