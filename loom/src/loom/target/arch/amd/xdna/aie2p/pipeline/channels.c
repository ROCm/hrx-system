// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"
#include "loom/transforms/pipeline/channel_materialization.h"

typedef struct loom_aie2p_native_endpoint_t {
  // Physical channel access and lock selectors from this worker's tile.
  const loom_aie2p_native_channel_access_t* access;
  // Initial physical slot address helper.
  loom_symbol_ref_t base;
  // Backing buffer address retained across borrows, or invalid when the
  // worker's endpoint set fills or exceeds the physical address-register
  // capacity.
  loom_value_id_t buffer;
  // Mutable state indices for directions owned by this worker. Fixed-slot
  // endpoints use zero displacement and require no threaded state.
  struct {
    // Current read byte offset; used only by an advancing owned acquire.
    iree_host_size_t reader;
    // Current write byte offset; used only by an advancing owned reserve.
    iree_host_size_t writer;
  } cursor;
  // Local publication prefix of a finite SRAM batch; not persisted across
  // invocations because admission excludes producer-side slot reuse.
  iree_host_size_t publication;
  // DMA task windows advanced by this endpoint's read admissions.
  const loom_aie2p_native_refill_t* refills;
  // Native consuming-credit acquire helper, created on first use.
  loom_symbol_ref_t acquire;
  // Native producing-credit acquire helper, created on first use.
  loom_symbol_ref_t reserve;
  // Native ready-credit release helper, created on first use.
  loom_symbol_ref_t publish;
  // Native free-credit release helper, created on first use.
  loom_symbol_ref_t release;
  // Alignment common to every physical record address.
  uint64_t alignment;
} loom_aie2p_native_endpoint_t;

typedef struct loom_aie2p_native_channel_emitter_t {
  // Physical selection and typed helper emitter.
  loom_aie2p_native_context_t* context;
  // Complete static protocol inventory, retaining child ownership.
  const loom_pipeline_invocations_t* invocations;
  // Endpoint index by construction channel index; unused entries are
  // UINT32_MAX.
  const uint32_t* indices;
  // Bound endpoints in the worker's retained channel-access order.
  loom_aie2p_native_endpoint_t* endpoints;
  // Worker-owned source actions and completion deltas, retained before
  // rewriting.
  const loom_pipeline_worker_t* worker;
  // Projected actions in the same ordinal order as source completion deltas.
  const loom_channel_plan_action_t* actions;
  // Initial byte displacement for every channel cursor.
  loom_value_id_t origin;
  // Invocation-private scalar views in the channel materializer's state order.
  const loom_value_id_t* state_views;
  // Canonical consuming-credit delta supplied to lock helpers.
  loom_value_id_t acquire_delta;
  // Canonical single-credit completion delta supplied to lock helpers.
  loom_value_id_t release_delta;
} loom_aie2p_native_channel_emitter_t;

static iree_status_t loom_aie2p_native_invoke(loom_builder_t* builder,
                                              loom_symbol_ref_t callee,
                                              const loom_value_id_t* arguments,
                                              iree_host_size_t argument_count,
                                              const loom_type_t* result_type,
                                              loom_value_id_t* out_result) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_invoke_build(
      builder, 0, 0, 0, 0, callee, arguments, argument_count, result_type,
      result_type ? 1 : 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  if (out_result) {
    *out_result = loom_op_results(op)[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_address(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint32_t address, loom_value_id_t* out_value) {
  const loom_named_attr_t attribute = {.name_id = context->integer_name,
                                       .value = loom_attr_i64(address)};
  return loom_aie2p_worker_op(
      context, builder, AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
      NULL, 0, loom_make_named_attr_slice(&attribute, 1),
      &context->address_type, out_value);
}

static iree_status_t loom_aie2p_native_address_helper(
    loom_aie2p_native_context_t* context, uint32_t address,
    loom_symbol_ref_t* out_symbol) {
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      &context->code, NULL, 0, &context->code.address_type, 1, &builder,
      out_symbol, &function));
  loom_value_id_t result;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_address(&context->code, &builder, address, &result));
  return loom_aie2p_worker_return(&builder, &result, 1);
}

static iree_status_t loom_aie2p_native_lock_helper(
    loom_aie2p_worker_builder_t* context, uint32_t descriptor,
    uint16_t selector, loom_symbol_ref_t* out_symbol) {
  loom_builder_t builder;
  loom_op_t* function;
  const loom_type_t argument = context->scalar_type;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      context, &argument, 1, NULL, 0, &builder, out_symbol, &function));
  const loom_value_id_t delta =
      loom_region_entry_block(loom_low_func_def_body(function))->arg_ids[0];
  const loom_named_attr_t attribute = {.name_id = context->lock_name,
                                       .value = loom_attr_i64(selector)};
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      context, &builder, descriptor, &delta, 1,
      loom_make_named_attr_slice(&attribute, 1), NULL, NULL));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

static iree_status_t loom_aie2p_native_acquire_delta_helper(
    loom_aie2p_worker_builder_t* context, loom_symbol_ref_t* out_symbol) {
  loom_builder_t builder;
  loom_op_t* function;
  const loom_type_t scalar_type = context->scalar_type;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      context, NULL, 0, &scalar_type, 1, &builder, out_symbol, &function));
  loom_value_id_t delta;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_constant(context, &builder, -1, &delta));
  return loom_aie2p_worker_return(&builder, &delta, 1);
}

static iree_status_t loom_aie2p_native_endpoint(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_access_t* access,
    loom_aie2p_native_endpoint_t* endpoint) {
  *endpoint = (loom_aie2p_native_endpoint_t){.access = access,
                                             .buffer = LOOM_VALUE_ID_INVALID,
                                             .acquire = loom_symbol_ref_null(),
                                             .reserve = loom_symbol_ref_null(),
                                             .publish = loom_symbol_ref_null(),
                                             .release = loom_symbol_ref_null(),
                                             .alignment = access->alignment};
  return access->window ? loom_aie2p_native_address_helper(
                              context, access->address, &endpoint->base)
                        : iree_ok_status();
}

static iree_status_t loom_aie2p_native_sram_publish_helper(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel, uint8_t packet,
    loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, NULL, 0, &builder, out_symbol, &function));
  const loom_aie2p_native_tile_t* tile =
      context->pool_tiles[channel->pool_index];
  const uint64_t origin =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  const uint16_t locks[] = {channel->free_lock, channel->ready_lock};
  // Complete producer and consumer lifetimes alternate. Clearing the free
  // credit and publishing the full batch therefore cannot race a partial
  // consumer release; these are ownership transitions, not atomic additions.
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(locks); ++i) {
    loom_value_id_t value;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, i, &value));
    const uint64_t address = loom_xdna_register_field_address(
        context->family, tile->memory_layout->lock_value, tile->coordinate,
        &locks[i]);
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        code, &builder, packet, (uint32_t)(address - origin), &value, 1));
  }
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

static iree_status_t loom_aie2p_native_materialize_buffer_accesses(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_aie2p_native_worker_t* worker,
    const loom_pipeline_worker_materialization_t* instance) {
  const loom_type_t buffer_type = loom_type_buffer();
  for (iree_host_size_t i = 0; i < worker->buffers.count; ++i) {
    const loom_aie2p_native_buffer_access_t* access =
        &worker->buffers.accesses[i];
    loom_symbol_ref_t helper;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_address_helper(context, access->address, &helper));

    loom_value_id_t buffer;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_invoke(
        &rewriter->builder, helper, NULL, 0, &buffer_type, &buffer));
    loom_op_t* aligned;
    IREE_RETURN_IF_ERROR(loom_buffer_assume_alignment_build(
        &rewriter->builder, &buffer, 1, access->alignment, &buffer_type, 1,
        LOOM_LOCATION_UNKNOWN, &aligned));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter,
        loom_pipeline_worker_materialization_value(instance, access->argument),
        loom_op_results(aligned)[0]));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_cursor_advance(
    loom_builder_t* builder, const loom_aie2p_native_channel_t* channel,
    loom_value_id_t origin, loom_value_id_t cursor, loom_location_id_t location,
    loom_value_id_t* out_record, loom_value_id_t* out_next) {
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  // Cursors start at zero and advance one record, wrapping at capacity.
  // Retain that induction invariant at the borrow so source CFG widening
  // cannot discard the selected address width and record alignment.
  const loom_predicate_t predicates[] = {
      {.kind = LOOM_PREDICATE_RANGE,
       .arg_count = 3,
       .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                    LOOM_PRED_ARG_CONST},
       .args = {cursor, 0,
                channel->byte_stride * (channel->source->capacity - 1)}},
      {.kind = LOOM_PREDICATE_MULTIPLE_OF,
       .arg_count = 2,
       .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                    LOOM_PRED_ARG_NONE},
       .args = {cursor, channel->byte_stride, 0}},
  };
  loom_op_t* bounded;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(
      builder, &cursor, 1, predicates, IREE_ARRAYSIZE(predicates), &offset_type,
      1, location, &bounded));
  *out_record = loom_index_assume_results(bounded).values[0];
  loom_op_t *stride, *limit, *next, *wrap, *selected;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(channel->byte_stride),
                                offset_type, location, &stride));
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(channel->byte_stride * channel->source->capacity),
      offset_type, location, &limit));
  IREE_RETURN_IF_ERROR(loom_index_add_build(builder, *out_record,
                                            loom_index_constant_result(stride),
                                            offset_type, location, &next));
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder, LOOM_INDEX_CMP_PREDICATE_EQ, loom_index_add_result(next),
      loom_index_constant_result(limit), location, &wrap));
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_index_cmp_result(wrap), origin, loom_index_add_result(next),
      offset_type, location, &selected));
  *out_next = loom_scf_select_result(selected);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_channel_action(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_channel_plan_action_t* action, loom_value_id_t* state,
    iree_host_size_t state_count, loom_value_id_t* results) {
  (void)state_count;
  loom_aie2p_native_channel_emitter_t* emitter = user_data;
  const uint32_t index =
      emitter->indices[loom_pipeline_invocations_channel_index(
          emitter->invocations, action->channel)];
  loom_aie2p_native_endpoint_t* endpoint = &emitter->endpoints[index];
  loom_builder_t* builder = &rewriter->builder;
  if (action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE ||
      action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE ||
      action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_WAIT) {
    loom_value_id_t record;
    uint16_t view_result = 0;
    if (action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_WAIT) {
      record = loom_channel_wait_read(action->op);
    } else {
      const bool write = action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE;
      loom_symbol_ref_t* acquire =
          write ? &endpoint->reserve : &endpoint->acquire;
      if (endpoint->access->window && !loom_symbol_ref_is_valid(*acquire)) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_lock_helper(
            &emitter->context->code,
            AIE2P_CORE_DESCRIPTOR_REF_LOCK_ACQUIRE_IMMEDIATE,
            write ? endpoint->access->locks.free
                  : endpoint->access->locks.ready,
            acquire));
      }
      if (endpoint->access->window) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_invoke(
            builder, *acquire, &emitter->acquire_delta, 1, NULL, NULL));
      }
      if (!write) {
        for (const loom_aie2p_native_refill_t* refill = endpoint->refills;
             refill; refill = refill->next_reader) {
          IREE_RETURN_IF_ERROR(
              loom_aie2p_native_emit_refill(builder, refill, state));
        }
      }
      record = emitter->origin;
      const loom_aie2p_native_channel_t* physical = endpoint->access->channel;
      if ((write ? physical->cursor.writer : physical->cursor.reader)
              .advances) {
        const iree_host_size_t cursor =
            write ? endpoint->cursor.writer : endpoint->cursor.reader;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_cursor_advance(
            builder, physical, emitter->origin, state[cursor],
            action->op->location, &record, &state[cursor]));
      }
      results[0] = record;
      view_result = 1;
    }
    if (!endpoint->access->window) {
      // DMA consumed the payload and its pure projections before this rewrite.
      // The zero-bit result has no core address to materialize.
      return iree_ok_status();
    }
    const loom_type_t buffer_type = loom_type_buffer();
    loom_value_id_t buffer = endpoint->buffer;
    if (buffer == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_invoke(
          builder, endpoint->base, NULL, 0, &buffer_type, &buffer));
    }
    loom_op_t* aligned;
    IREE_RETURN_IF_ERROR(loom_buffer_assume_alignment_build(
        builder, &buffer, 1, endpoint->alignment, &buffer_type, 1,
        action->op->location, &aligned));
    loom_op_t* view;
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        builder, loom_op_results(aligned)[0], record,
        loom_module_value_type(rewriter->module,
                               loom_op_results(action->op)[view_result]),
        action->op->location, &view));
    results[view_result] = loom_buffer_view_result(view);
    return iree_ok_status();
  }
  const uint32_t credits =
      emitter->worker->completion.actions[action - emitter->actions].credits;
  if (!credits) {
    return iree_ok_status();
  }
  if (!endpoint->access->window) {
    const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    loom_op_t *delta, *prefix, *capacity, *complete, *publish;
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(builder, loom_attr_i64(credits), offset_type,
                                  action->op->location, &delta));
    IREE_RETURN_IF_ERROR(
        loom_index_add_build(builder, state[endpoint->publication],
                             loom_index_constant_result(delta), offset_type,
                             action->op->location, &prefix));
    state[endpoint->publication] = loom_index_add_result(prefix);
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(endpoint->access->channel->replay.record_count),
        offset_type, action->op->location, &capacity));
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_EQ, loom_index_add_result(prefix),
        loom_index_constant_result(capacity), action->op->location, &complete));
    IREE_RETURN_IF_ERROR(
        loom_scf_if_build(builder, 0, loom_index_cmp_result(complete), NULL, 0,
                          NULL, 0, action->op->location, &publish));
    const loom_builder_ip_t saved = loom_builder_enter_region(
        builder, publish, loom_scf_if_then_region(publish));
    iree_status_t status = loom_aie2p_native_invoke(builder, endpoint->publish,
                                                    NULL, 0, NULL, NULL);
    if (iree_status_is_ok(status)) {
      loom_op_t* yield;
      status =
          loom_scf_yield_build(builder, NULL, 0, action->op->location, &yield);
    }
    loom_builder_restore(builder, saved);
    return status;
  }
  const bool publish = action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_PUBLISH;
  loom_symbol_ref_t additional = loom_symbol_ref_null();
  loom_symbol_ref_t* completion =
      publish ? &endpoint->publish : &endpoint->release;
  if (credits != 1) {
    completion = &additional;
  }
  if (!loom_symbol_ref_is_valid(*completion)) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_lock_helper(
        &emitter->context->code,
        AIE2P_CORE_DESCRIPTOR_REF_LOCK_RELEASE_IMMEDIATE,
        publish ? endpoint->access->locks.ready : endpoint->access->locks.free,
        completion));
  }
  loom_value_id_t delta = emitter->release_delta;
  if (credits != 1) {
    loom_op_t* constant;
    IREE_RETURN_IF_ERROR(
        loom_scalar_constant_build(builder, loom_attr_i64((int32_t)credits),
                                   loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                   action->op->location, &constant));
    delta = loom_scalar_constant_result(constant);
  }
  return loom_aie2p_native_invoke(builder, *completion, &delta, 1, NULL, NULL);
}

static iree_status_t loom_aie2p_native_channel_exit(
    void* user_data, loom_rewriter_t* rewriter, const loom_op_t* terminator,
    const loom_value_id_t* state, iree_host_size_t state_count) {
  (void)terminator;
  loom_aie2p_native_channel_emitter_t* emitter = user_data;
  const int64_t zero = 0;
  for (iree_host_size_t i = 0; i < state_count; ++i) {
    if (emitter->state_views[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_op_t* cast = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        &rewriter->builder, state[i], loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &cast));
    loom_op_t* store = NULL;
    IREE_RETURN_IF_ERROR(loom_view_store_build(
        &rewriter->builder, 0, 0, loom_index_cast_result(cast),
        emitter->state_views[i], NULL, 0, &zero, 1, 0, 0, LOOM_LOCATION_UNKNOWN,
        &store));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_emit_worker(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index) {
  loom_aie2p_native_context_t* context = user_data;
  context->code.realization = realization;
  context->code.worker_index = worker_index;
  context->code.diagnostic_emitter = context->pass->diagnostic_emitter;
  const loom_pipeline_worker_t* source = &realization->workers[worker_index];
  const loom_pipeline_worker_materialization_t* instance =
      &realization->instances[worker_index];
  const loom_channel_materialization_instance_t* channels = &instance->channels;
  loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  loom_aie2p_native_channel_emitter_t emitter = {
      .context = context,
      .invocations = &realization->invocations,
      .worker = source,
      .actions = channels->plan.actions,
      .indices = worker->channels.indices};
  const iree_host_size_t endpoint_count = worker->channels.count;
  bool has_sram = false;
  for (iree_host_size_t i = 0; i < endpoint_count; ++i) {
    has_sram |= !worker->channels.accesses[i].window;
  }
  iree_arena_allocator_t* arena = context->pass->arena;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, endpoint_count,
                                                 sizeof(*emitter.endpoints),
                                                 (void**)&emitter.endpoints));
  for (iree_host_size_t i = 0; i < endpoint_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint(
        context, &worker->channels.accesses[i], &emitter.endpoints[i]));
  }
  iree_host_size_t refill_count = 0;
  for (loom_aie2p_native_refill_t* refill = worker->refills; refill;
       refill = refill->next) {
    const iree_host_size_t channel = refill->progress - context->channels;
    loom_aie2p_native_endpoint_t* endpoint =
        &emitter.endpoints[worker->channels.indices[channel]];
    refill->next_reader = endpoint->refills;
    endpoint->refills = refill;
    ++refill_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_prepare_refills(context, worker->refills));
  loom_aie2p_worker_builder_t* code = &context->code;
  if (has_sram) {
    IREE_RETURN_IF_ERROR(loom_rewriter_enable_worklist(rewriter));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_emit_transfers(
      context, rewriter, realization, worker_index));
  uint8_t* erased_values = NULL;
  if (has_sram) {
    loom_op_t* candidate;
    while ((candidate = loom_rewriter_pop(rewriter))) {
      bool erased;
      IREE_RETURN_IF_ERROR(
          loom_rewriter_erase_if_dead(rewriter, candidate, &erased));
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, channels->plan.value_count, sizeof(*erased_values),
        (void**)&erased_values));
    memset(erased_values, 0, channels->plan.value_count);
    for (iree_host_size_t i = 0; i < channels->plan.action_count; ++i) {
      const loom_channel_plan_action_t* action = &channels->plan.actions[i];
      if (action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE) {
        continue;
      }
      const iree_host_size_t channel_index =
          loom_pipeline_invocations_channel_index(&realization->invocations,
                                                  action->channel);
      if (worker->channels.accesses[worker->channels.indices[channel_index]]
              .window) {
        continue;
      }
      const loom_value_id_t payload = loom_op_results(action->op)[1];
      if (loom_module_value_has_uses(code->module, payload)) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_reject(
            context, action->op,
            IREE_SV("SRAM payload borrows consumed exclusively by DMA")));
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      }
      erased_values[loom_local_value_domain_ordinal(&channels->value_domain,
                                                    payload)] = 1;
    }
  }
  loom_block_t* entry =
      loom_region_entry_block(loom_func_like_body(channels->function));
  loom_builder_set_before(&rewriter->builder, entry->first_op);
  IREE_RETURN_IF_ERROR(loom_aie2p_native_materialize_buffer_accesses(
      context, rewriter, worker, instance));
  const loom_type_t delta_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  // Establish one canonical +/-1 pair in every channel-bearing worker. Both
  // lower to replayable constants: low-pressure workers retain and reuse them,
  // while constrained workers can materialize them at lock-helper calls instead
  // of consuming call-preserved registers across the complete worker. The
  // acquire helper introduces its target-typed constant after helper expansion;
  // release +1 remains available to source cursor and loop arithmetic.
  if (endpoint_count != 0) {
    loom_op_t* release_delta;
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        &rewriter->builder, loom_attr_i64(1), delta_type, LOOM_LOCATION_UNKNOWN,
        &release_delta));
    emitter.release_delta = loom_scalar_constant_result(release_delta);
    loom_symbol_ref_t helper;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_acquire_delta_helper(code, &helper));
    IREE_RETURN_IF_ERROR(loom_aie2p_native_invoke(&rewriter->builder, helper,
                                                  NULL, 0, &delta_type,
                                                  &emitter.acquire_delta));
  }
  loom_op_t* origin;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(&rewriter->builder, loom_attr_i64(0),
                                loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                LOOM_LOCATION_UNKNOWN, &origin));
  emitter.origin = loom_index_constant_result(origin);
  loom_value_id_t* state = NULL;
  const iree_host_size_t state_capacity =
      endpoint_count * (has_sram ? 3 : 2) + refill_count * 2;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, state_capacity, sizeof(*state), (void**)&state));
  loom_value_id_t* state_views = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, state_capacity, sizeof(*state_views), (void**)&state_views));
  emitter.state_views = state_views;
  iree_host_size_t state_count = 0;
  const loom_type_t buffer_type = loom_type_buffer();
  const uint16_t address_register_capacity =
      code->descriptors->reg_classes[AIE2P_CORE_REG_CLASS_ID_AIE2P_EP]
          .allocatable_count;
  // Retain every address when the physical file can hold the complete set and
  // still leave a register for payload addressing, so repeated borrows pay no
  // setup. Sets filling the file materialize at each borrow; ordinary CSE
  // shares adjacent phases without making disjoint endpoints simultaneously
  // live.
  const bool retain_endpoint_addresses =
      endpoint_count < address_register_capacity;
  for (iree_host_size_t i = 0; i < endpoint_count; ++i) {
    loom_aie2p_native_endpoint_t* endpoint = &emitter.endpoints[i];
    if (retain_endpoint_addresses && endpoint->access->window) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_invoke(&rewriter->builder, endpoint->base, NULL, 0,
                                   &buffer_type, &endpoint->buffer));
    }
    const loom_aie2p_native_channel_t* channel = endpoint->access->channel;
    if (!endpoint->access->window) {
      endpoint->publication = state_count;
      state_views[state_count] = LOOM_VALUE_ID_INVALID;
      state[state_count++] = emitter.origin;
      const loom_aie2p_native_worker_t* owner =
          &context->workers[context->execution
                                .workers[context->core_engines[worker_index]]];
      const iree_host_size_t tile_index =
          context->pool_tiles[channel->pool_index] - context->tiles;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_sram_publish_helper(
          context, channel, owner->control_packets[tile_index],
          &endpoint->publish));
    }
    for (uint32_t direction = 0; direction < 2; ++direction) {
      const loom_aie2p_native_cursor_t* cursor =
          direction ? &channel->cursor.writer : &channel->cursor.reader;
      if (!cursor->advances || cursor->worker == UINT32_MAX ||
          context->core_engines[cursor->worker] !=
              context->core_engines[worker_index]) {
        continue;
      }
      if (direction) {
        endpoint->cursor.writer = state_count;
      } else {
        endpoint->cursor.reader = state_count;
      }
      const iree_host_size_t channel_index = channel - context->channels;
      if (!cursor->shared &&
          emitter.invocations->channels[channel_index].invocation ==
              emitter.invocations->workers[worker_index].invocation) {
        state_views[state_count] = LOOM_VALUE_ID_INVALID;
        state[state_count++] = emitter.origin;
        continue;
      }
      if (worker->cursor_state_argument == LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_module_define_value(
            code->module, buffer_type, &worker->cursor_state_argument));
      }
      IREE_RETURN_IF_ERROR(loom_aie2p_native_cursor_view(
          &rewriter->builder, worker->cursor_state_argument,
          channel - context->channels, direction != 0,
          &state_views[state_count]));
      const int64_t zero = 0;
      loom_op_t *load = NULL, *cast = NULL;
      IREE_RETURN_IF_ERROR(loom_view_load_build(
          &rewriter->builder, 0, 0, state_views[state_count], NULL, 0, &zero, 1,
          0, 0, loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN,
          &load));
      IREE_RETURN_IF_ERROR(
          loom_index_cast_build(&rewriter->builder, loom_view_load_result(load),
                                loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                LOOM_LOCATION_UNKNOWN, &cast));
      state[state_count++] = loom_index_cast_result(cast);
    }
  }
  for (loom_aie2p_native_refill_t* refill = worker->refills; refill;
       refill = refill->next) {
    refill->state_index = state_count;
    const uint32_t initial[] = {0, refill->remaining_tasks};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(initial); ++i) {
      loom_op_t* value;
      IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
          &rewriter->builder, loom_attr_i64((int32_t)initial[i]), delta_type,
          LOOM_LOCATION_UNKNOWN, &value));
      state_views[state_count] = LOOM_VALUE_ID_INVALID;
      state[state_count++] = loom_scalar_constant_result(value);
    }
  }
  loom_type_t* carriers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, channels->plan.value_count, sizeof(*carriers), (void**)&carriers));
  memset(carriers, 0, channels->plan.value_count * sizeof(*carriers));
  for (loom_value_ordinal_t i = 0; i < channels->plan.value_count; ++i) {
    const loom_type_t type = loom_module_value_type(
        code->module, channels->value_domain.value_ids[i]);
    if (loom_read_type_isa(type) || loom_write_type_isa(type)) {
      carriers[i] = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    }
  }
  const loom_channel_materialization_options_t options = {
      .carrier_types = carriers,
      .erased_values = erased_values,
      .initial_state = state,
      .state_count = state_count,
      .emit = {.fn = loom_aie2p_native_channel_action,
               .exit = loom_aie2p_native_channel_exit,
               .user_data = &emitter}};
  IREE_RETURN_IF_ERROR(loom_channel_materialize(rewriter, &channels->plan,
                                                &channels->graph, &options));
  // The materializer consumes the original block-argument inventory. Attach
  // the generated frame formal after that consuming rewrite has finished.
  if (worker->cursor_state_argument != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_block_add_arg(code->module, entry, worker->cursor_state_argument));
  }
  return iree_ok_status();
}
