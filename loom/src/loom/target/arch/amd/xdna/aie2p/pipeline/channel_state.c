// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"

#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/view/ops.h"

static void loom_aie2p_native_accumulate_cursor_owner(
    loom_aie2p_native_channel_t* channel) {
  const loom_aie2p_native_cursor_t* reader = &channel->cursor.reader;
  const loom_aie2p_native_cursor_t* writer = &channel->cursor.writer;
  if (reader->worker == channel->balance.worker) {
    channel->balance.reads += reader->maximum_admissions;
    channel->balance.exact &= reader->exact_admissions;
  }
  if (writer->worker == channel->balance.worker) {
    channel->balance.writes += writer->maximum_admissions;
    channel->balance.exact &= writer->exact_admissions;
  }
}

void loom_aie2p_native_begin_cursor_owner(
    loom_aie2p_native_channel_t* channel,
    const loom_pipeline_invocations_t* invocations, uint32_t worker) {
  if (channel->balance.worker == worker) {
    return;
  }
  if (channel->balance.worker != UINT32_MAX) {
    loom_aie2p_native_accumulate_cursor_owner(channel);
    if (invocations->workers[channel->balance.worker].invocation !=
        invocations->workers[worker].invocation) {
      channel->balance.balanced &=
          channel->balance.exact &&
          channel->balance.reads == channel->balance.writes;
      channel->balance.reads = channel->balance.writes = 0;
      channel->balance.exact = true;
    }
  }
  channel->balance.worker = worker;
}

void loom_aie2p_native_finish_cursor_owners(
    loom_aie2p_native_channel_t* channel) {
  if (channel->balance.worker != UINT32_MAX) {
    loom_aie2p_native_accumulate_cursor_owner(channel);
    channel->balance.balanced &=
        channel->balance.exact &&
        channel->balance.reads == channel->balance.writes;
  }
}

void loom_aie2p_native_cursor_bind(loom_aie2p_native_cursor_t* cursor,
                                   uint64_t capacity, uint32_t worker,
                                   uint32_t maximum_admissions, bool site_exact,
                                   bool imported) {
  const bool shared = cursor->shared || (cursor->worker != UINT32_MAX &&
                                         cursor->worker != worker);
  if (cursor->worker == worker) {
    maximum_admissions =
        iree_max(cursor->maximum_admissions, maximum_admissions);
    site_exact &= cursor->exact_admissions;
  }
  *cursor = (loom_aie2p_native_cursor_t){
      .worker = worker,
      .maximum_admissions = maximum_admissions,
      .exact_admissions = site_exact && maximum_admissions != UINT32_MAX,
      .shared = shared,
      .advances =
          capacity > 1 && (maximum_admissions > 1 || shared || imported),
  };
}

bool loom_aie2p_native_cursor_has_exact_admissions(
    const loom_aie2p_native_cursor_t* cursor, uint64_t count) {
  return !cursor->shared && cursor->exact_admissions &&
         cursor->maximum_admissions == count;
}

bool loom_aie2p_native_cursor_persistent(
    const loom_pipeline_invocations_t* invocations, iree_host_size_t channel,
    const loom_aie2p_native_cursor_t* cursor) {
  return cursor->advances &&
         (cursor->shared ||
          invocations->channels[channel].invocation !=
              invocations->workers[cursor->worker].invocation);
}

const loom_aie2p_native_cursor_t* loom_aie2p_native_record_cursor(
    const loom_aie2p_native_channel_t* channel, loom_type_t capability) {
  return loom_read_type_isa(capability) ? &channel->cursor.reader
                                        : &channel->cursor.writer;
}

iree_status_t loom_aie2p_native_cursor_view(loom_builder_t* builder,
                                            loom_value_id_t frame,
                                            iree_host_size_t channel,
                                            bool writer,
                                            loom_value_id_t* out_view) {
  loom_op_t* offset = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64((channel * 2 + writer) * sizeof(uint32_t)),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN,
      &offset));
  loom_op_t* view = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      builder, frame, loom_index_constant_result(offset),
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(1), 0),
      LOOM_LOCATION_UNKNOWN, &view));
  *out_view = loom_buffer_view_result(view);
  return iree_ok_status();
}

// Frame cells live with their channel construction, even when nested calls
// share them. Each cell has one physical engine owner; resetting it at that
// construction's entry needs no inter-engine synchronization.
iree_status_t loom_aie2p_native_initialize_cursors(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_value_id_t frame, iree_host_size_t engine,
    const loom_pipeline_invocation_t* owner) {
  const loom_pipeline_invocations_t* invocations =
      &context->code.realization->invocations;
  loom_op_t* zero = NULL;
  const iree_host_size_t end =
      owner->first_channel + owner->resources->channel_count;
  for (iree_host_size_t i = owner->first_channel; i < end; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    for (uint32_t direction = 0; direction < 2; ++direction) {
      const loom_aie2p_native_cursor_t* cursor =
          direction ? &channel->cursor.writer : &channel->cursor.reader;
      if (cursor->worker == UINT32_MAX) {
        continue;
      }
      const bool persistent =
          loom_aie2p_native_cursor_persistent(invocations, i, cursor);
      if (!persistent || context->core_engines[cursor->worker] != engine) {
        continue;
      }
      if (!zero) {
        IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
            builder, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
            LOOM_LOCATION_UNKNOWN, &zero));
      }
      loom_value_id_t view = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_cursor_view(
          builder, frame, i, direction != 0, &view));
      const int64_t index = 0;
      loom_op_t* store = NULL;
      IREE_RETURN_IF_ERROR(loom_view_store_build(
          builder, 0, 0, loom_scalar_constant_result(zero), view, NULL, 0,
          &index, 1, 0, 0, LOOM_LOCATION_UNKNOWN, &store));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_bind_endpoint_initializer(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_aie2p_native_distribution_endpoint_t* endpoint,
    const loom_pipeline_invocation_t* invocation, bool* out_valid) {
  const loom_pipeline_invocations_t* invocations = &realization->invocations;
  const iree_host_size_t channel = endpoint->channel - context->channels;
  const loom_pipeline_invocation_t* owner =
      invocations->channels[channel].invocation;
  const loom_aie2p_native_dma_sequence_t* sequence = &endpoint->sequence;
  if (sequence->terminal_slot && !sequence->rewind_iteration &&
      owner != invocation) {
    *out_valid = false;
    return loom_aie2p_native_reject(
        context, endpoint->channel->source->binding,
        IREE_SV("a continuing DMA slot iterator for an imported partial ring"));
  }
  // Root channel state is initialized by the entry configuration.
  // A complete turn needs no restoration at any construction boundary.
  if (!sequence->rewind_iteration || !owner->ordinal) {
    return iree_ok_status();
  }
  if (!context->endpoint_initializers) {
    const iree_host_size_t count =
        invocations->count * context->execution.engine_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, count, sizeof(*context->endpoint_initializers),
        (void**)&context->endpoint_initializers));
    memset(context->endpoint_initializers, 0,
           count * sizeof(*context->endpoint_initializers));
  }
  // Task admission already selected this controller and its physical routes.
  // Its engine participates in the constructing owner's retained lifetime.
  const iree_host_size_t index =
      owner->ordinal * context->execution.engine_count +
      endpoint->controller_engine;
  loom_aie2p_native_endpoint_initialization_t* initializer;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      context->pass->arena, sizeof(*initializer), (void**)&initializer));
  *initializer = (loom_aie2p_native_endpoint_initialization_t){
      .next = context->endpoint_initializers[index], .endpoint = endpoint};
  context->endpoint_initializers[index] = initializer;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_select_initializers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && *out_valid &&
                               i < realization->invocations.worker_count;
       ++i) {
    const loom_aie2p_native_worker_t* worker = &context->workers[i];
    // Physical owners retain each interned geometry once, even when multiple
    // child sites submit it. Initialization therefore does not grow per use.
    for (const loom_aie2p_native_distribution_t* distribution =
             worker->distributions;
         iree_status_is_ok(status) && *out_valid && distribution;
         distribution = distribution->next) {
      for (iree_host_size_t e = 0; iree_status_is_ok(status) && *out_valid &&
                                   e <= distribution->receiver_count;
           ++e) {
        const loom_aie2p_native_distribution_endpoint_t* endpoint =
            e ? &distribution->receivers[e - 1] : &distribution->sender;
        status = loom_aie2p_native_bind_endpoint_initializer(
            context, realization, endpoint, distribution->invocation,
            out_valid);
      }
    }
  }
  return status;
}

iree_status_t loom_aie2p_native_initialize_endpoints(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t engine, const loom_pipeline_invocation_t* invocation) {
  const loom_aie2p_native_endpoint_initialization_t* initializer =
      context->endpoint_initializers
          ? context->endpoint_initializers[invocation->ordinal *
                                               context->execution.engine_count +
                                           engine]
          : NULL;
  if (!initializer) {
    return iree_ok_status();
  }
  loom_aie2p_worker_builder_t* code = &context->code;
  code->worker_index = context->execution.workers[engine];
  const loom_aie2p_native_worker_t* controller =
      &context->workers[code->worker_index];
  loom_builder_t helper;
  loom_symbol_ref_t symbol;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, NULL, 0, NULL, 0, &helper,
                                                &symbol, &function));
  for (; initializer; initializer = initializer->next) {
    const loom_aie2p_native_distribution_endpoint_t* endpoint =
        initializer->endpoint;
    const loom_aie2p_native_memory_layout_t* layout =
        endpoint->tile->memory_layout;
    const loom_xdna_tile_coordinate_t tile = endpoint->tile->coordinate;
    const uint64_t origin = (uint64_t)tile.column
                                << context->family->column_shift |
                            (uint64_t)tile.row << context->family->row_shift;
    const uint16_t descriptor = endpoint->sequence.descriptor;
    const uint32_t address =
        (uint32_t)(loom_xdna_register_field_address(context->family,
                                                    layout->iteration.current,
                                                    tile, &descriptor) -
                   origin);
    loom_value_id_t word;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &helper, endpoint->sequence.words[0][layout->iteration_word],
        &word));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        code, &helper,
        controller->control_packets[endpoint->tile - context->tiles], address,
        &word, 1));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
  loom_op_t* call;
  return loom_low_invoke_build(builder, 0, 0, 0, 0, symbol, NULL, 0, NULL, 0,
                               NULL, 0, LOOM_LOCATION_UNKNOWN, &call);
}
