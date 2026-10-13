// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#include "loom/analysis/storage_geometry.h"
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
      if (tile->facts->kind != LOOM_XDNA_TILE_KIND_COMPUTE) {
        continue;
      }
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

static bool loom_aie2p_native_can_configure(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_pipeline_worker_t* worker) {
  if (!worker->transport.steps) {
    return false;
  }
  // One admission at each private endpoint makes its initial free credit
  // unconditional and its final reclamation unobservable. Publication and
  // external-write completion remain explicit ordered effects.
  for (iree_host_size_t i = 0; i < worker->binding_count; ++i) {
    const loom_aie2p_native_channel_t* channel =
        &context->channels[worker->bindings[i].channel -
                           realization->resources.channels];
    if (channel->cursor.reader.maximum_admissions > 1 ||
        channel->cursor.writer.maximum_admissions > 1) {
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
  if (begin < 0 || begin > UINT32_MAX || stride < 0 || stride > UINT32_MAX ||
      transport->repetitions > local_dma->maximum_task_repeat_count ||
      transport->repetitions > shim_dma->maximum_task_repeat_count ||
      binding->capacity > (UINT64_C(1) << local_dma->iteration_bits) ||
      (stride &&
       transport->repetitions > (UINT64_C(1) << shim_dma->iteration_bits)) ||
      (channel->byte_stride % local_dma->transfer_length_granularity) ||
      (stride % shim_dma->transfer_length_granularity) ||
      channel->byte_stride / local_dma->transfer_length_granularity >
          (UINT64_C(1) << local_dma->step_size_bits) ||
      (uint64_t)stride / shim_dma->transfer_length_granularity >
          (UINT64_C(1) << shim_dma->step_size_bits)) {
    return false;
  }
  worker->repetition.count = (uint32_t)transport->repetitions;
  worker->repetition.external_byte_offset = (uint32_t)begin;
  worker->repetition.external_byte_stride = (uint32_t)stride;
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
    bool configure = false;
    bool repeat = false;
    if (source->transport.steps) {
      loom_local_value_domain_restore(&source->value_domain);
      configure = !configuration_used &&
                  loom_aie2p_native_can_configure(context, realization, source);
      repeat = !configure && loom_aie2p_native_can_repeat(context, realization,
                                                          source, worker);
      loom_local_value_domain_release(&source->value_domain);
    }
    if (configure) {
      worker->execution = LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION;
      configuration_used = true;
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
