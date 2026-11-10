// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"
#include "loom/util/adaptive_sort.h"

static bool loom_aie2p_native_collective_less(
    const loom_aie2p_native_collective_t* left,
    const loom_aie2p_native_collective_t* right) {
  if (left->scope != right->scope) {
    return (uintptr_t)left->scope < (uintptr_t)right->scope;
  }
  return (uintptr_t)left->source < (uintptr_t)right->source;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_aie2p_native_collective_sort,
                          loom_aie2p_native_collective_t,
                          loom_aie2p_native_collective_less)

static const loom_aie2p_native_collective_t* loom_aie2p_native_collective_find(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* source) {
  const loom_aie2p_native_collective_t key = {.scope = scope, .source = source};
  iree_host_size_t begin = 0, end = context->collective_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (loom_aie2p_native_collective_less(&context->collectives[middle],
                                          &key)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return &context->collectives[begin];
}

iree_status_t loom_aie2p_native_select_collectives(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  iree_host_size_t capacity = 0;
  const loom_pipeline_execution_t* execution = &context->execution;
  for (iree_host_size_t s = 0; s < execution->schedule_count; ++s) {
    const loom_pipeline_execution_schedule_t* participants =
        &execution->schedules[s];
    const loom_pipeline_invocation_schedule_t* scope = participants->source;
    for (iree_host_size_t b = 0; b < scope->source->control->graph.block_count;
         ++b) {
      const loom_pipeline_schedule_block_t* block = &scope->source->blocks[b];
      for (iree_host_size_t i = 0; i < block->step_count; ++i) {
        const loom_pipeline_schedule_step_t* step = &block->steps[i];
        if (step->kind != LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE) {
          continue;
        }
        const loom_pipeline_collective_t* source = step->collective;
        const iree_host_size_t index = loom_pipeline_invocations_channel_index(
            &realization->invocations, &scope->blocks[b][i].channel->identity);
        loom_aie2p_native_channel_t* channel = &context->channels[index];
        const uint32_t worker = channel->cursor.reader.worker;
        if (worker == UINT32_MAX ||
            context->core_engines[worker] == IREE_HOST_SIZE_MAX ||
            participants->workers[context->core_engines[worker]] ==
                IREE_HOST_SIZE_MAX) {
          return loom_aie2p_native_reject(
              context, step->operation,
              IREE_SV("a collective with a participating FIFO reader"));
        }
        for (iree_host_size_t r = 0; r < source->result_count; ++r) {
          const loom_pipeline_collective_result_t* result = &source->results[r];
          const loom_type_t type = loom_module_value_type(
              context->code.module,
              loom_op_results(
                  source->operations[result->operation])[result->result]);
          loom_aie2p_call_abi_requirements_t carrier = {0};
          if (!loom_aie2p_call_abi_add_source_type(type, &carrier) ||
              carrier.scalar_count != 1) {
            return loom_aie2p_native_reject(
                context, source->operations[result->operation],
                IREE_SV("collective values with one scalar stream word"));
          }
        }
        if (context->collective_count == capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              context->pass->arena, context->collective_count,
              context->collective_count + 1, sizeof(*context->collectives),
              &capacity, (void**)&context->collectives));
        }
        const iree_host_size_t reader = context->core_engines[worker];
        const loom_aie2p_native_tile_t* owner =
            context->pool_tiles[channel->pool_index];
        const loom_xdna_address_window_t* window = loom_xdna_array_load_window(
            context->family, context->workers[worker].tile->coordinate,
            owner->coordinate);
        if (!window) {
          return loom_aie2p_native_reject(
              context, step->operation,
              IREE_SV(
                  "collective storage visible to its admitted FIFO reader"));
        }
        const iree_host_size_t owner_engine = owner - context->tiles;
        const iree_host_size_t relay =
            participants->workers[owner_engine] != IREE_HOST_SIZE_MAX
                ? owner_engine
                : reader;
        iree_host_size_t peer_count = 0;
        for (iree_host_size_t e = 0; e < execution->engine_count; ++e) {
          peer_count +=
              e != reader && participants->workers[e] != IREE_HOST_SIZE_MAX;
        }
        context->collectives[context->collective_count++] =
            (loom_aie2p_native_collective_t){
                .scope = scope,
                .source = source,
                .channel = channel,
                .reader = reader,
                .relay = relay,
                .peer_count = peer_count,
                .address = window->base + channel->byte_offset,
                .ready_lock = window->lock_selector_base + channel->ready_lock,
                .free_lock = window->lock_selector_base + channel->free_lock};
        // The collective and ordinary child consume successive records on
        // the same engine. Their cursor belongs to the shared entry frame.
        channel->cursor.reader.shared = true;
        channel->cursor.reader.advances = channel->source->capacity > 1;
      }
    }
  }
  loom_aie2p_native_collective_sort(context->collectives,
                                    context->collective_count);
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_route_collectives(
    loom_aie2p_native_context_t* context, uint32_t* next_source,
    bool* out_valid) {
  *out_valid = false;
  if (!context->collective_count) {
    *out_valid = true;
    return iree_ok_status();
  }
  const loom_pipeline_execution_t* execution = &context->execution;
  for (iree_host_size_t i = 0; i < context->collective_count; ++i) {
    loom_aie2p_native_collective_t* collective = &context->collectives[i];
    if (!collective->source->result_count || !collective->peer_count) {
      continue;
    }
    const loom_pipeline_execution_schedule_t* participants =
        loom_pipeline_execution_schedule(execution, collective->scope);
    const iree_host_size_t reader = collective->reader;
    const iree_host_size_t relay = collective->relay;
    loom_aie2p_native_worker_t* reader_worker =
        &context->workers[execution->workers[reader]];
    loom_aie2p_native_worker_t* relay_worker =
        &context->workers[execution->workers[relay]];
    collective->reader_packet =
        loom_aie2p_native_collective_packet(&reader_worker->packet_mask);
    collective->relay_packet =
        relay == reader
            ? collective->reader_packet
            : loom_aie2p_native_collective_packet(&relay_worker->packet_mask);
    if (collective->reader_packet == UINT8_MAX ||
        collective->relay_packet == UINT8_MAX) {
      return loom_aie2p_native_reject(
          context, collective->source->operations[0],
          IREE_SV("available collective packet identities"));
    }
    for (iree_host_size_t e = 0; e < execution->engine_count; ++e) {
      if (e == reader || participants->workers[e] == IREE_HOST_SIZE_MAX) {
        continue;
      }
      const bool first_hop = e == relay || relay == reader;
      if (!loom_aie2p_native_packet_route(
              context, next_source,
              (first_hop ? reader_worker : relay_worker)->tile->coordinate,
              LOOM_XDNA_STREAM_PORT_CORE,
              context->workers[participants->workers[e]].tile->coordinate,
              LOOM_XDNA_STREAM_PORT_CORE,
              first_hop ? collective->reader_packet : collective->relay_packet,
              NULL)) {
        return loom_aie2p_native_reject(
            context, collective->source->operations[0],
            IREE_SV(
                "collective scalar-stream routes alongside model transport"));
      }
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_collective_begin(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* source,
    const loom_value_id_t* state_arguments, loom_type_t payload_type,
    loom_value_id_t* out_payload) {
  loom_aie2p_native_context_t* context = user_data;
  const loom_aie2p_native_collective_t* collective =
      loom_aie2p_native_collective_find(context, scope, source);
  *out_payload = LOOM_VALUE_ID_INVALID;
  if (engine != collective->reader) {
    return iree_ok_status();
  }
  loom_aie2p_worker_builder_t* code = &context->code;
  code->worker_index = context->execution.workers[engine];
  loom_builder_t helper;
  loom_symbol_ref_t symbol;
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, &code->address_type, 1, &helper, &symbol, &function));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_lock(
      code, &helper, AIE2P_CORE_DESCRIPTOR_REF_LOCK_ACQUIRE_IMMEDIATE,
      collective->ready_lock, -1));
  const loom_named_attr_t attribute = {
      .name_id = code->integer_name,
      .value = loom_attr_i64(collective->address)};
  loom_value_id_t address;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &helper, AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
      NULL, 0, loom_make_named_attr_slice(&attribute, 1), &code->address_type,
      &address));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, &address, 1));
  loom_op_t *call, *offset, *view;
  const loom_type_t buffer_type = loom_type_buffer();
  IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, 0, symbol, NULL,
                                             0, &buffer_type, 1, NULL, 0,
                                             LOOM_LOCATION_UNKNOWN, &call));
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &offset));
  loom_value_id_t record = loom_index_constant_result(offset);
  if (collective->channel->cursor.reader.advances) {
    loom_value_id_t cursor_view;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_cursor_view(
        builder, state_arguments[0], collective->channel - context->channels,
        false, &cursor_view));
    const int64_t zero = 0;
    const loom_type_t scalar_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    loom_op_t *load, *cast, *next_cast, *store;
    IREE_RETURN_IF_ERROR(loom_view_load_build(builder, 0, 0, cursor_view, NULL,
                                              0, &zero, 1, 0, 0, scalar_type,
                                              LOOM_LOCATION_UNKNOWN, &load));
    IREE_RETURN_IF_ERROR(
        loom_index_cast_build(builder, loom_view_load_result(load), scalar_type,
                              offset_type, LOOM_LOCATION_UNKNOWN, &cast));
    loom_value_id_t next;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_cursor_advance(
        builder, collective->channel, record, loom_index_cast_result(cast),
        LOOM_LOCATION_UNKNOWN, &record, &next));
    IREE_RETURN_IF_ERROR(
        loom_index_cast_build(builder, next, offset_type, scalar_type,
                              LOOM_LOCATION_UNKNOWN, &next_cast));
    IREE_RETURN_IF_ERROR(loom_view_store_build(
        builder, 0, 0, loom_index_cast_result(next_cast), cursor_view, NULL, 0,
        &zero, 1, 0, 0, LOOM_LOCATION_UNKNOWN, &store));
  }
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(builder, loom_op_results(call)[0],
                                              record, payload_type,
                                              LOOM_LOCATION_UNKNOWN, &view));
  *out_payload = loom_buffer_view_result(view);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_collective_end(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* source, loom_value_id_t* values) {
  loom_aie2p_native_context_t* context = user_data;
  const loom_aie2p_native_collective_t* collective =
      loom_aie2p_native_collective_find(context, scope, source);
  loom_aie2p_worker_builder_t* code = &context->code;
  code->worker_index = context->execution.workers[engine];
  const bool reader = engine == collective->reader;
  const iree_host_size_t count = source->result_count;
  if (count && collective->peer_count) {
    loom_type_t* types = NULL;
    loom_type_t* result_types = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        code->arena, count, sizeof(*types), (void**)&types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        code->arena, count, sizeof(*result_types), (void**)&result_types));
    for (iree_host_size_t i = 0; i < count; ++i) {
      types[i] = code->scalar_type;
      const loom_pipeline_collective_result_t* result = &source->results[i];
      result_types[i] = loom_module_value_type(
          code->module,
          loom_op_results(
              source->operations[result->operation])[result->result]);
    }
    loom_builder_t helper;
    loom_symbol_ref_t symbol;
    loom_op_t* function = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
        code, reader ? types : NULL, reader ? count : 0, reader ? NULL : types,
        reader ? 0 : count, &helper, &symbol, &function));
    loom_value_id_t* words = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        code->arena, count, sizeof(*words), (void**)&words));
    if (reader) {
      memcpy(words,
             loom_region_entry_block(loom_low_func_def_body(function))->arg_ids,
             count * sizeof(*words));
    } else {
      loom_value_id_t header;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
          code, &helper, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
          loom_named_attr_slice_empty(), &code->scalar_type, &header));
      for (iree_host_size_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
            code, &helper, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
            loom_named_attr_slice_empty(), &code->scalar_type, &words[i]));
      }
    }
    if (reader || (engine == collective->relay && collective->peer_count > 1)) {
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_packet_write(
          code, &helper,
          reader ? collective->reader_packet : collective->relay_packet, words,
          count));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(
        &helper, reader ? NULL : words, reader ? 0 : count));
    loom_op_t* call = NULL;
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(
        builder, 0, 0, 0, 0, symbol, reader ? values : NULL, reader ? count : 0,
        reader ? NULL : result_types, reader ? 0 : count, NULL, 0,
        LOOM_LOCATION_UNKNOWN, &call));
    if (!reader) {
      memcpy(values, loom_op_results(call), count * sizeof(*values));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_join_boundary(context, builder, engine, scope));
  if (reader) {
    loom_builder_t helper;
    loom_symbol_ref_t symbol;
    loom_op_t* function = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, NULL, 0, NULL, 0,
                                                  &helper, &symbol, &function));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_lock(
        code, &helper, AIE2P_CORE_DESCRIPTOR_REF_LOCK_RELEASE_IMMEDIATE,
        collective->free_lock, 1));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
    loom_op_t* call = NULL;
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, 0, symbol,
                                               NULL, 0, NULL, 0, NULL, 0,
                                               LOOM_LOCATION_UNKNOWN, &call));
  }
  return iree_ok_status();
}
