// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#include "loom/analysis/storage_geometry.h"
#include "loom/error/parameter.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/target/function_version.h"
#include "loom/util/fact_extensions.h"

iree_status_t loom_aie2p_native_reject(
    const loom_aie2p_native_context_t* context, const loom_op_t* op,
    iree_string_view_t requirement) {
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_XDNA_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(context->pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_native_inventory(
    loom_aie2p_native_context_t* context) {
  const loom_xdna_array_family_t* family = context->family;
  const iree_host_size_t tile_count = family->column_count * family->row_count;
  iree_arena_allocator_t* arena = context->pass->arena;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*context->tiles), (void**)&context->tiles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, tile_count,
                                                 sizeof(*context->pool_tiles),
                                                 (void**)&context->pool_tiles));
  loom_pipeline_resource_pool_t* pools = NULL;
  loom_pipeline_resource_memory_t* memories = NULL;
  uint64_t* coordinates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*pools), (void**)&pools));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*memories), (void**)&memories));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, 2 * sizeof(*coordinates), (void**)&coordinates));
  context->inventory.pools = pools;
  context->inventory.memories = memories;
  for (uint16_t column = 0; column < family->column_count; ++column) {
    for (uint16_t row = 0; row < family->row_count; ++row) {
      loom_aie2p_native_tile_t* tile =
          &context->tiles[column * family->row_count + row];
      *tile = (loom_aie2p_native_tile_t){.coordinate = {column, row},
                                         .pool_index = UINT32_MAX};
      tile->facts = loom_xdna_array_tile_facts(family, tile->coordinate);
      if (tile->facts->kind == LOOM_XDNA_TILE_KIND_SHIM_NOC) {
        continue;
      }
      tile->memory_layout = tile->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY
                                ? &loom_aie2p_sram_memory
                                : &loom_aie2p_compute_memory;
      const uint32_t pool = (uint32_t)context->inventory.pool_count++;
      tile->pool_index = pool;
      context->pool_tiles[pool] = tile;
      pools[pool] = (loom_pipeline_resource_pool_t){
          .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
          .byte_capacity = tile->facts->memory.local_capacity};
      IREE_RETURN_IF_ERROR(loom_source_storage_packing_allocate(
          (loom_source_storage_packing_interference_callback_t){0}, NULL, 0,
          arena, &pools[pool].packing));
      coordinates[pool * 2] = column;
      coordinates[pool * 2 + 1] = row;
      memories[pool] = (loom_pipeline_resource_memory_t){
          .coordinates = &coordinates[pool * 2], .rank = 2, .pool_index = pool};
      ++context->inventory.memory_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_channels(
    loom_aie2p_native_context_t* context, loom_module_t* module,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  const loom_pipeline_resources_t* resources = &realization->resources;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, resources->channel_count,
      sizeof(*context->channels), (void**)&context->channels));
  for (iree_host_size_t i = 0; i < resources->channel_count; ++i) {
    const loom_pipeline_resource_channel_t* source = &resources->channels[i];
    const loom_pipeline_resource_allocation_t* allocation =
        loom_pipeline_resources_lookup_allocation(
            resources, source->storage.root_value_id);
    if (!allocation) {
      return loom_aie2p_native_reject(
          context, source->binding,
          IREE_SV("invocation-owned on-chip channel storage"));
    }
    loom_aie2p_native_tile_t* owner =
        context->pool_tiles[allocation->pool_index];
    const loom_type_t storage_type = loom_module_value_type(
        module, loom_channel_bind_storage(source->binding));
    loom_storage_geometry_t geometry;
    loom_storage_geometry_span_t record;
    int64_t offset = 0;
    uint64_t stride_bits = 0, record_bits = 0, extent = 0;
    if (!loom_storage_geometry_query(&realization->facts->context, module,
                                     storage_type, &geometry) ||
        geometry.rank == 0 ||
        !loom_storage_geometry_measure(&geometry, 1, &record) ||
        !loom_value_facts_as_exact_i64(source->storage.base_byte_offset,
                                       &offset) ||
        offset < 0 ||
        !iree_checked_mul_u64(geometry.axes[0].element_stride,
                              geometry.element_bit_count, &stride_bits) ||
        !iree_checked_mul_u64(record.element_span, geometry.element_bit_count,
                              &record_bits) ||
        stride_bits % 8 || record_bits % 8 ||
        !iree_checked_mul_u64(source->capacity - 1, stride_bits / 8, &extent) ||
        !iree_checked_add_u64(extent, record_bits / 8, &extent) ||
        (uint64_t)offset > allocation->byte_length ||
        extent > allocation->byte_length - (uint64_t)offset ||
        stride_bits / 8 > UINT32_MAX ||
        allocation->byte_offset > UINT32_MAX - (uint64_t)offset) {
      return loom_aie2p_native_reject(
          context, source->binding,
          IREE_SV("specialized byte-addressable channel slot geometry"));
    }
    if (source->capacity > (uint64_t)owner->facts->lock_value_maximum ||
        owner->next_lock + 2 > owner->facts->lock_count) {
      return loom_aie2p_native_reject(
          context, source->binding,
          IREE_SV("a hardware semaphore pair with enough slot credits"));
    }
    context->channels[i] = (loom_aie2p_native_channel_t){
        .source = source,
        .pool_index = allocation->pool_index,
        .byte_offset = (uint32_t)(allocation->byte_offset + (uint64_t)offset),
        .byte_stride = (uint32_t)(stride_bits / 8),
        .record_byte_length = (uint32_t)(record_bits / 8),
        .record_dense = record.dense,
        .free_lock = owner->next_lock++,
        .ready_lock = owner->next_lock++,
        .cursor = {.reader = {.worker = UINT32_MAX},
                   .writer = {.worker = UINT32_MAX}}};
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_channel_accesses(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* source, loom_aie2p_native_worker_t* worker,
    bool* out_valid) {
  *out_valid = false;
  iree_arena_allocator_t* arena = context->pass->arena;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, realization->resources.channel_count,
      sizeof(*worker->channels.indices), (void**)&worker->channels.indices));
  memset(
      worker->channels.indices, 0xff,
      realization->resources.channel_count * sizeof(*worker->channels.indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->binding_count, sizeof(*worker->channels.accesses),
      (void**)&worker->channels.accesses));
  for (iree_host_size_t i = 0; i < source->binding_count; ++i) {
    const iree_host_size_t index =
        source->bindings[i].channel - realization->resources.channels;
    if (worker->channels.indices[index] != UINT32_MAX) {
      continue;
    }
    worker->channels.indices[index] = (uint32_t)worker->channels.count;
    loom_aie2p_native_channel_access_t* access =
        &worker->channels.accesses[worker->channels.count++];
    const loom_aie2p_native_channel_t* channel = &context->channels[index];
    access->channel = channel;
    if (context->pool_tiles[channel->pool_index]->facts->kind ==
        LOOM_XDNA_TILE_KIND_MEMORY) {
      return loom_aie2p_native_reject(
          context, source->source->declaration,
          IREE_SV("DMA-owned SRAM storage with compute-memory handoffs"));
    }
    const loom_xdna_tile_coordinate_t owner =
        context->pool_tiles[channel->pool_index]->coordinate;
    access->window = loom_xdna_array_load_window(
        context->family, worker->tile->coordinate, owner);
    if (!access->window) {
      return loom_aie2p_native_reject(
          context, source->source->declaration,
          IREE_SV("channel storage directly visible from its consuming core"));
    }
    access->address = access->window->base + channel->byte_offset;
    const uint64_t alignment_bits = access->address | channel->byte_stride;
    access->alignment =
        alignment_bits ? alignment_bits & (~alignment_bits + 1) : 64;
    access->locks.free =
        access->window->lock_selector_base + channel->free_lock;
    access->locks.ready =
        access->window->lock_selector_base + channel->ready_lock;
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_select_buffer_accesses(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* source, loom_aie2p_native_worker_t* worker,
    bool* out_valid) {
  *out_valid = false;
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(source->function, &argument_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, argument_count, sizeof(*worker->buffers.accesses),
      (void**)&worker->buffers.accesses));
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_id_t argument = arguments[i];
    if (!loom_type_is_buffer(
            loom_module_value_type(context->code.module, argument)) ||
        !loom_module_value_has_uses(context->code.module, argument)) {
      continue;
    }
    loom_value_fact_buffer_reference_t reference;
    if (!loom_value_facts_query_buffer_reference(
            &source->facts.context,
            loom_value_fact_table_lookup(&source->facts, argument),
            &reference)) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("a resolved storage root for each core buffer capture"));
    }
    const loom_value_id_t root =
        loom_value_fact_buffer_reference_resolve_root_value(reference,
                                                            argument);
    if (reference.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      continue;
    }
    const loom_pipeline_resource_allocation_t* allocation =
        loom_pipeline_resources_lookup_allocation(&realization->resources,
                                                  root);
    if (!allocation || allocation->byte_length == 0 ||
        allocation->byte_offset > UINT32_MAX ||
        allocation->byte_length > UINT32_MAX) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("an invocation-owned finite local allocation for each core "
                  "buffer capture"));
    }
    const loom_aie2p_native_tile_t* owner =
        context->pool_tiles[allocation->pool_index];
    if (owner->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("DMA-owned SRAM storage with compute-memory handoffs"));
    }
    const loom_xdna_address_window_t* window = loom_xdna_array_load_window(
        context->family, worker->tile->coordinate, owner->coordinate);
    if (!window) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("captured storage directly visible from its consuming core"));
    }
    loom_aie2p_native_buffer_access_t* access =
        &worker->buffers.accesses[worker->buffers.count++];
    access->argument = argument;
    access->alignment = allocation->byte_alignment;
    access->address = window->base + (uint32_t)allocation->byte_offset;
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_claim_transport_cursor(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, uint32_t worker_index,
    const loom_channel_identity_t* identity, const loom_op_t* op, bool reader,
    uint64_t admissions, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_resource_channel_t* bound =
      loom_pipeline_resources_lookup_channel(&realization->resources,
                                             identity->value_id);
  loom_aie2p_native_channel_t* channel =
      &context->channels[bound - realization->resources.channels];
  loom_aie2p_native_cursor_t* cursor =
      reader ? &channel->cursor.reader : &channel->cursor.writer;
  if (!admissions || admissions > UINT32_MAX) {
    return loom_aie2p_native_reject(
        context, op,
        IREE_SV(
            "a positive channel-copy count representable by native cursors"));
  }
  if (cursor->worker != UINT32_MAX) {
    return loom_aie2p_native_reject(
        context, op,
        IREE_SV("one ordered transport stream per FIFO read or write cursor"));
  }
  cursor->worker = worker_index;
  cursor->maximum_admissions = (uint32_t)admissions;
  cursor->advances =
      channel->source->capacity > 1 && cursor->maximum_admissions > 1;
  *out_valid = true;
  return iree_ok_status();
}

static bool loom_aie2p_native_can_configure(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* worker) {
  if (!worker->transport.steps || worker->transport.induction) {
    return false;
  }
  // A singleton returns its entire credit on each turn. A retained reader
  // likewise returns a complete batch after replay. Ordinary multi-slot rings
  // can have concurrent partial publications and need atomic credit updates.
  const uint32_t worker_index = (uint32_t)(worker - realization->workers);
  for (iree_host_size_t i = 0; i < worker->binding_count; ++i) {
    const loom_aie2p_native_channel_t* channel =
        &context->channels[worker->bindings[i].channel -
                           realization->resources.channels];
    if (channel->source->capacity > 1 &&
        (channel->cursor.reader.maximum_admissions > 1 ||
         channel->cursor.writer.maximum_admissions > 1) &&
        !(channel->replay.record_count &&
          channel->cursor.writer.worker == worker_index &&
          channel->cursor.writer.maximum_admissions %
                  channel->replay.record_count ==
              0)) {
      return false;
    }
  }
  for (const loom_kernel_async_stream_t* stream = worker->asynchronous.streams;
       stream; stream = stream->next) {
    for (iree_host_size_t i = 0; i < stream->transfer_count; ++i) {
      const loom_movement_request_t* request = &stream->transfers[i].request;
      if (request->kind != LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY ||
          !((request->source.memory_space ==
                 LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
             request->dest.memory_space ==
                 LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) ||
            (request->source.memory_space ==
                 LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
             request->dest.memory_space ==
                 LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL))) {
        return false;
      }
      const loom_view_region_t* source;
      const loom_view_region_t* destination;
      loom_view_region_table_try_lookup(
          &worker->asynchronous.movement.view_regions, request->source.value_id,
          &source);
      loom_view_region_table_try_lookup(
          &worker->asynchronous.movement.view_regions, request->dest.value_id,
          &destination);
      const bool ingress =
          request->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
      const loom_view_region_t* external = ingress ? source : destination;
      const loom_view_region_t* local = ingress ? destination : source;
      if (!loom_symbolic_expr_is_constant(&external->begin_byte_offset) ||
          !loom_symbolic_expr_is_constant(&local->projection_byte_offset)) {
        return false;
      }
    }
  }
  return true;
}

typedef struct loom_aie2p_native_configuration_cursor_t {
  // Next reserved physical slot.
  uint32_t reservation;
  // Published prefix within the current retained batch.
  uint32_t publication;
} loom_aie2p_native_configuration_cursor_t;

static iree_status_t loom_aie2p_native_configure_channels(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* source, loom_aie2p_native_worker_t* worker) {
  // These are physical cursor positions, assigned once from the retained
  // straight-line effect order. Emission consumes the selected turns directly.
  loom_aie2p_native_configuration_cursor_t* positions;
  loom_aie2p_native_configuration_action_t* actions;
  uint32_t* records;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, realization->resources.channel_count,
      sizeof(*positions), (void**)&positions));
  memset(positions, 0,
         realization->resources.channel_count * sizeof(*positions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, source->channels.action_count, sizeof(*actions),
      (void**)&actions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, source->channels.value_count, sizeof(*records),
      (void**)&records));
  for (iree_host_size_t i = 0; i < source->transport.count; ++i) {
    const loom_pipeline_transport_step_t* step = &source->transport.steps[i];
    if (step->kind != LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL) {
      continue;
    }
    const loom_channel_plan_action_t* source_action = step->source.channel;
    const iree_host_size_t action_index =
        source_action - source->channels.actions;
    const loom_pipeline_resource_channel_t* binding =
        loom_pipeline_resources_lookup_channel(
            &realization->resources, source_action->channel->value_id);
    const iree_host_size_t channel_index =
        binding - realization->resources.channels;
    const loom_aie2p_native_channel_t* channel =
        &context->channels[channel_index];
    loom_aie2p_native_configuration_cursor_t* position =
        &positions[channel_index];
    loom_aie2p_native_configuration_action_t* action = &actions[action_index];
    *action = (loom_aie2p_native_configuration_action_t){
        .channel = channel, .expected = -1, .value = -1};
    const loom_op_t* op = step->op;
    const bool reserve = loom_channel_reserve_isa(op);
    const bool acquire = loom_channel_acquire_isa(op);
    if (reserve || acquire) {
      const loom_value_id_t record = loom_op_const_results(op)[0];
      records[loom_local_value_domain_ordinal(&source->value_domain, record)] =
          reserve ? position->reservation * channel->byte_stride : 0;
    }
    if (channel->replay.record_count) {
      if (reserve) {
        if (position->reservation == 0) {
          action->lock = channel->free_lock;
          action->expected = (int16_t)channel->replay.count;
          action->value = 0;
        }
        position->reservation =
            (position->reservation + 1) % channel->replay.record_count;
      } else if (loom_channel_publish_isa(op)) {
        position->publication +=
            source->completion.actions[action_index].credits;
        if (position->publication == channel->replay.record_count) {
          action->lock = channel->ready_lock;
          action->value = (int16_t)channel->replay.count;
          position->publication = 0;
        }
      }
    } else if (channel->source->capacity == 1 &&
               source->completion.actions[action_index].maximum_admissions >
                   1) {
      action->lock = reserve || loom_channel_release_isa(op)
                         ? channel->free_lock
                         : channel->ready_lock;
      if (reserve || acquire) {
        action->expected = 1;
        action->value = 0;
      } else if (source->completion.actions[action_index].credits) {
        action->value = 1;
      }
    } else if (acquire || loom_channel_publish_isa(op)) {
      // A single admission from an ordinary ring retains the original
      // initial-credit debit and unobservable final reclamation.
      action->lock = channel->ready_lock;
      action->expected = acquire ? 1 : -1;
      action->value = acquire ? 0 : 1;
    }
  }
  worker->configuration_actions = actions;
  worker->configuration_records = records;
  return iree_ok_status();
}

static bool loom_aie2p_native_can_repeat(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* source, loom_aie2p_native_worker_t* worker) {
  const loom_pipeline_transport_t* transport = &source->transport;
  if (!transport->induction || !transport->sources_stable ||
      transport->count != 4 || transport->stream_count != 1 ||
      transport->streams[0]->transfer_count != 1 ||
      transport->repetitions < 2) {
    return false;
  }
  const loom_pipeline_transport_step_t* steps = transport->steps;
  if (!loom_channel_reserve_isa(steps[0].op) ||
      steps[1].kind != LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER ||
      steps[2].kind != LOOM_PIPELINE_TRANSPORT_STEP_WAIT ||
      !loom_channel_publish_isa(steps[3].op) ||
      loom_channel_publish_write(steps[3].op) !=
          loom_channel_reserve_write(steps[0].op)) {
    return false;
  }
  const loom_movement_request_t* request = &steps[1].source.transfer->request;
  if (request->kind != LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY ||
      request->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
      request->dest.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  const loom_pipeline_resource_channel_t* binding =
      loom_pipeline_resources_lookup_channel(
          &realization->resources, steps[0].source.channel->channel->value_id);
  const loom_aie2p_native_channel_t* channel =
      &context->channels[binding - realization->resources.channels];
  // Consumer completion observes every DMA publication, and therefore the last
  // external read and local write, before the invocation can retire storage.
  // Egress must instead observe global-write completion before releasing its
  // source slot: a subsequent reservation can observe the external output.
  if (channel->cursor.reader.worker == UINT32_MAX ||
      channel->cursor.reader.maximum_admissions != transport->repetitions ||
      !realization->workers[channel->cursor.reader.worker]
           .block_execution_counts) {
    return false;
  }
  const loom_view_region_t *external, *local;
  loom_view_region_table_try_lookup(&source->asynchronous.movement.view_regions,
                                    request->source.value_id, &external);
  loom_view_region_table_try_lookup(&source->asynchronous.movement.view_regions,
                                    request->dest.value_id, &local);
  if (local->base_view_value_id != loom_channel_reserve_view(steps[0].op) ||
      !loom_symbolic_expr_is_constant(&local->projection_byte_offset)) {
    return false;
  }
  const loom_symbolic_expr_t* offset = &external->begin_byte_offset;
  int64_t begin = offset->constant, stride = 0;
  if (offset->term_count) {
    int64_t initial, increment, displacement;
    if (!loom_symbolic_expr_is_linear(offset) || offset->term_count != 1 ||
        offset->terms[0].value_id != transport->induction->value ||
        !loom_value_facts_as_exact_i64(
            loom_value_fact_recurrence_operand_facts(
                &source->facts, transport->induction->initial_value),
            &initial) ||
        !loom_value_facts_as_exact_i64(
            loom_value_fact_recurrence_operand_facts(
                &source->facts, transport->induction->step),
            &increment) ||
        !iree_checked_mul_i64(offset->terms[0].coefficient, initial,
                              &displacement) ||
        !iree_checked_add_i64(begin, displacement, &begin) ||
        !iree_checked_mul_i64(offset->terms[0].coefficient, increment,
                              &stride)) {
      return false;
    }
  }
  const loom_xdna_dma_facts_t* local_dma =
      &context->pool_tiles[channel->pool_index]->facts->dma;
  const loom_xdna_dma_facts_t* shim_dma =
      &context
           ->tiles[worker->tile->coordinate.column * context->family->row_count]
           .facts->dma;
  if (channel->replay.record_count &&
      ((uint64_t)request->transferred_byte_count !=
           channel->record_byte_length ||
       local->projection_byte_offset.constant != 0 ||
       stride != channel->record_byte_length ||
       transport->repetitions % channel->replay.record_count)) {
    return false;
  }
  if (begin < 0 || begin > UINT32_MAX || stride < 0 || stride > UINT32_MAX ||
      transport->repetitions > UINT32_MAX ||
      binding->capacity > (UINT64_C(1) << local_dma->iteration_bits) ||
      (channel->byte_stride % local_dma->transfer_length_granularity) ||
      (stride % shim_dma->transfer_length_granularity) ||
      channel->byte_stride / local_dma->transfer_length_granularity >
          (UINT64_C(1) << local_dma->step_size_bits) ||
      !loom_aie2p_native_select_ingress_descriptor(
          context, source, request, (uint32_t)stride,
          (uint32_t)transport->repetitions, worker)) {
    return false;
  }
  worker->repetition.count = (uint32_t)transport->repetitions;
  if (channel->replay.record_count) {
    worker->repetition.count /= channel->replay.record_count;
  }
  worker->repetition.external_byte_offset = (uint32_t)begin;
  return true;
}

static iree_status_t loom_aie2p_native_select(
    void* user_data, loom_module_t* module,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  loom_aie2p_native_context_t* context = user_data;
  *out_valid = false;
  if (realization->resources.composition_count) {
    return loom_aie2p_native_reject(
        context, realization->resources.compositions[0],
        IREE_SV("child execution boundaries to be realized separately"));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channels(
      context, module, realization, out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  *out_valid = false;
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(realization->function, &argument_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, argument_count, sizeof(*context->bindings),
      (void**)&context->bindings));
  context->binding_count = argument_count;
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (!loom_type_is_buffer(loom_module_value_type(module, arguments[i]))) {
      return loom_aie2p_native_reject(
          context, realization->function.op,
          IREE_SV("buffer invocation arguments after specialization"));
    }
    context->bindings[i] = (loom_aie2p_native_binding_t){
        .root = arguments[i], .byte_length = 1, .byte_alignment = 4};
  }
  const iree_host_size_t worker_count = realization->resources.strand_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, worker_count, sizeof(*context->workers),
      (void**)&context->workers));
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    const loom_pipeline_worker_t* source = &realization->workers[i];
    if (source->rank != 2 || source->axes[0].count != 1 ||
        source->axes[1].count != 1 ||
        source->axes[0].origin >= context->family->column_count ||
        source->axes[1].origin >= context->family->row_count) {
      return loom_aie2p_native_reject(
          context, source->source->declaration,
          IREE_SV("a single placed compute tile per strand"));
    }
    loom_aie2p_native_tile_t* tile =
        &context->tiles[source->axes[0].origin * context->family->row_count +
                        source->axes[1].origin];
    if (tile->facts->kind != LOOM_XDNA_TILE_KIND_COMPUTE) {
      return loom_aie2p_native_reject(
          context, source->source->declaration,
          IREE_SV("a compute-tile execution location"));
    }
    context->workers[i] = (loom_aie2p_native_worker_t){.tile = tile};
    if (source->transport.cohort_count) {
      for (iree_host_size_t j = 0; j < source->transport.batch_count; ++j) {
        const loom_pipeline_transport_batch_t* batch =
            &source->transport.batches[j];
        const loom_pipeline_transport_cohort_t* first = batch->records;
        if (!first->release) {
          continue;
        }
        const loom_pipeline_resource_channel_t* bound =
            loom_pipeline_resources_lookup_channel(
                &realization->resources, first->admission->channel->value_id);
        loom_aie2p_native_channel_t* channel =
            &context->channels[bound - realization->resources.channels];
        const loom_aie2p_native_tile_t* owner =
            context->pool_tiles[channel->pool_index];
        if (channel->replay.record_count ||
            batch->record_count != bound->capacity || !channel->record_dense ||
            channel->byte_stride != channel->record_byte_length ||
            batch->replays > (uint64_t)owner->facts->lock_value_maximum) {
          return loom_aie2p_native_reject(
              context, first->admission->op,
              IREE_SV("one contiguous retained FIFO batch with a representable "
                      "replay credit count"));
        }
        channel->replay.record_count = (uint32_t)batch->record_count;
        channel->replay.count = (uint32_t)batch->replays;
      }
      for (iree_host_size_t j = 0; j < source->transport.batch_count; ++j) {
        const loom_pipeline_transport_batch_t* batch =
            &source->transport.batches[j];
        const loom_pipeline_transport_cohort_t* cohort = batch->records;
        if (cohort->transfers > UINT32_MAX / batch->record_count) {
          return loom_aie2p_native_reject(
              context, cohort->admission->op,
              IREE_SV("a complete channel-copy count representable by native "
                      "cursors"));
        }
        bool valid = false;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_claim_transport_cursor(
            context, realization, (uint32_t)i, cohort->admission->channel,
            cohort->admission->op, true,
            cohort->admissions * batch->record_count, &valid));
        if (!valid) {
          return iree_ok_status();
        }
        for (iree_host_size_t k = 0; k < cohort->destination_count; ++k) {
          IREE_RETURN_IF_ERROR(loom_aie2p_native_claim_transport_cursor(
              context, realization, (uint32_t)i,
              cohort->destinations[k].channel,
              cohort->destinations[k].reservation, false,
              cohort->transfers * batch->record_count, &valid));
          if (!valid) {
            return iree_ok_status();
          }
        }
      }
      continue;
    }
    if (source->completion.requirement !=
        LOOM_CHANNEL_COMPLETION_REQUIREMENT_NONE) {
      static const iree_string_view_t requirements[] = {
          IREE_SVL(""),
          IREE_SVL("FIFO acquire/reserve/wait/publish/release channel actions"),
          IREE_SVL("statically resolved FIFO record completion identities"),
          IREE_SVL("compatible FIFO completion frontiers at control joins"),
          IREE_SVL("independent slot reuse across blocking channel admission"),
      };
      return loom_aie2p_native_reject(
          context, source->completion.op,
          requirements[source->completion.requirement]);
    }
    for (iree_host_size_t j = 0; j < source->channels.action_count; ++j) {
      const loom_channel_plan_action_t* action = &source->channels.actions[j];
      const loom_op_t* op = action->op;
      if (!loom_channel_acquire_isa(op) && !loom_channel_reserve_isa(op) &&
          !loom_channel_publish_isa(op) && !loom_channel_release_isa(op) &&
          !loom_channel_wait_isa(op)) {
        return loom_aie2p_native_reject(
            context, op,
            IREE_SV(
                "FIFO acquire/reserve/wait/publish/release channel actions"));
      }
      if (loom_channel_acquire_isa(op) || loom_channel_reserve_isa(op)) {
        const loom_pipeline_resource_channel_t* bound =
            loom_pipeline_resources_lookup_channel(&realization->resources,
                                                   action->channel->value_id);
        loom_aie2p_native_channel_t* channel =
            &context->channels[bound - realization->resources.channels];
        loom_aie2p_native_cursor_t* cursor = loom_channel_acquire_isa(op)
                                                 ? &channel->cursor.reader
                                                 : &channel->cursor.writer;
        if (cursor->worker != UINT32_MAX && cursor->worker != i) {
          return loom_aie2p_native_reject(
              context, op,
              IREE_SV("one strand owning each FIFO's read or write cursor"));
        }
        *cursor = (loom_aie2p_native_cursor_t){
            .worker = (uint32_t)i,
            .maximum_admissions =
                source->completion.actions[j].maximum_admissions,
            .advances = channel->source->capacity > 1 &&
                        source->completion.actions[j].maximum_admissions > 1};
      }
    }
    for (iree_host_size_t j = 0; j < source->channels.action_count; ++j) {
      const loom_channel_plan_action_t* action = &source->channels.actions[j];
      if (source->completion.actions[j].credits >
          (uint32_t)tile->facts->lock_value_maximum) {
        return loom_aie2p_native_reject(
            context, action->op,
            IREE_SV("a completed FIFO prefix within semaphore credit range"));
      }
    }
  }
  // The invocation command stream is one sequential engine. Selecting at most
  // one strand preserves concurrency between independently authored strands.
  // Every other strand retains its own execution engine.
  bool configuration_used = false;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    loom_pipeline_worker_t* source = &realization->workers[i];
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    if (source->transport.cohort_count) {
      worker->execution = LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION;
      continue;
    }
    bool configure = false;
    bool repeat = false;
    if (source->transport.steps) {
      loom_local_value_domain_restore(&source->value_domain);
      repeat =
          loom_aie2p_native_can_repeat(context, realization, source, worker);
      configure = !repeat && !configuration_used &&
                  loom_aie2p_native_can_configure(context, realization, source);
      loom_local_value_domain_release(&source->value_domain);
    }
    if (configure) {
      worker->execution = LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
      configuration_used = true;
      loom_local_value_domain_restore(&source->value_domain);
      iree_status_t status = loom_aie2p_native_configure_channels(
          context, realization, source, worker);
      loom_local_value_domain_release(&source->value_domain);
      IREE_RETURN_IF_ERROR(status);
      continue;
    }
    if (repeat) {
      worker->execution = LOOM_AIE2P_NATIVE_EXECUTION_DMA;
      continue;
    }
    loom_aie2p_native_tile_t* tile = worker->tile;
    if (tile->has_worker) {
      return loom_aie2p_native_reject(
          context, source->source->declaration,
          IREE_SV("an independently assigned compute tile for each "
                  "instruction-executing strand"));
    }
    if (tile->next_lock == tile->facts->lock_count) {
      return loom_aie2p_native_reject(
          context, source->source->declaration,
          IREE_SV("one semaphore for worker completion"));
    }
    tile->has_worker = true;
    worker->completion_lock = tile->next_lock++;
    const loom_xdna_address_window_t* self_window = loom_xdna_array_load_window(
        context->family, tile->coordinate, tile->coordinate);
    worker->completion_selector =
        self_window->lock_selector_base + worker->completion_lock;
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channel_accesses(
        context, realization, source, worker, &valid));
    if (!valid) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_buffer_accesses(
        context, realization, source, worker, &valid));
    if (!valid) {
      return iree_ok_status();
    }
  }
  for (iree_host_size_t i = 0; i < realization->resources.channel_count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    if (channel->replay.record_count &&
        (channel->cursor.writer.worker == UINT32_MAX ||
         (context->workers[channel->cursor.writer.worker].execution !=
              LOOM_AIE2P_NATIVE_EXECUTION_DMA &&
          context->workers[channel->cursor.writer.worker].execution !=
              LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION))) {
      return loom_aie2p_native_reject(
          context, realization->function.op,
          IREE_SV("a DMA-owned ingress for retained FIFO replay"));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_select_transfers(context, realization, out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  return loom_pipeline_resources_check_capacity(
      &realization->resources, realization->function.op,
      context->pass->diagnostic_emitter, out_valid);
}

iree_status_t loom_aie2p_pipeline_realize(loom_pass_t* pass,
                                          loom_module_t* module,
                                          loom_func_like_t function) {
  const loom_aie2p_target_facts_t* target = loom_aie2p_target_facts_cast(
      loom_target_function_version_target_facts(pass->function_version));
  loom_aie2p_native_context_t context = {.pass = pass, .target = target};
  if (!target || !target->device_profile) {
    return loom_aie2p_native_reject(&context, function.op,
                                    IREE_SV("an exact deployment profile"));
  }
  context.family =
      loom_xdna_device_profile_array_family(target->device_profile);
  IREE_RETURN_IF_ERROR(loom_aie2p_native_inventory(&context));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_builder_initialize(module, pass->arena, &context.code));
  const loom_pipeline_realization_callback_t callback = {
      .select = loom_aie2p_native_select,
      .worker = loom_aie2p_native_emit_worker,
      .entry = loom_aie2p_native_emit_configuration,
      .user_data = &context};
  return loom_pipeline_realize(pass, module, function, &context.inventory,
                               &callback);
}
