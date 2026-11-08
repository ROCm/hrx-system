// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/worker_materialization.h"

#include <stdlib.h>

#include "loom/transforms/view/offset_expression.h"

typedef struct loom_pipeline_worker_clone_state_t {
  // Next retained block stream in clone visitation order.
  const loom_kernel_async_stream_t* stream;
  // Next transfer in the current stream.
  iree_host_size_t transfer;
  // Next group in the current stream.
  iree_host_size_t group;
  // Next explicit wait in the current stream.
  iree_host_size_t wait;
  // Caller-owned selected correspondence populated by the clone observer.
  loom_pipeline_worker_materialization_t* instance;
} loom_pipeline_worker_clone_state_t;

static iree_status_t loom_pipeline_worker_clone_op(void* user_data,
                                                   const loom_op_t* source,
                                                   loom_op_t* target) {
  loom_pipeline_worker_clone_state_t* state = user_data;
  const loom_kernel_async_stream_t* stream = state->stream;
  if (!stream) {
    return iree_ok_status();
  }
  if (state->transfer < stream->transfer_count &&
      stream->transfers[state->transfer].request.op == source) {
    ++state->transfer;
  } else if (state->group < stream->group_count &&
             stream->groups[state->group].op == source) {
    ++state->group;
  } else if (state->wait < stream->wait_count &&
             stream->waits[state->wait] == source) {
    ++state->wait;
  } else {
    return iree_ok_status();
  }
  state->instance->operations[state->instance->operation_count++] =
      (loom_ir_remap_op_projection_t){source, target};
  if (state->transfer == stream->transfer_count &&
      state->group == stream->group_count &&
      state->wait == stream->wait_count) {
    state->stream = stream->next;
    state->transfer = state->group = state->wait = 0;
  }
  return iree_ok_status();
}

static int loom_pipeline_worker_compare_operations(const void* lhs,
                                                   const void* rhs) {
  const uintptr_t a =
      (uintptr_t)((const loom_ir_remap_op_projection_t*)lhs)->source_op;
  const uintptr_t b =
      (uintptr_t)((const loom_ir_remap_op_projection_t*)rhs)->source_op;
  return (a > b) - (a < b);
}

iree_status_t loom_pipeline_worker_materialization_clone(
    loom_rewriter_t* rewriter, const loom_pipeline_worker_t* source,
    loom_symbol_ref_t symbol,
    loom_pipeline_worker_materialization_t* out_instance) {
  *out_instance = (loom_pipeline_worker_materialization_t){0};
  iree_host_size_t count = 0;
  for (const loom_kernel_async_stream_t* stream = source->asynchronous.streams;
       stream; stream = stream->next) {
    count += stream->transfer_count + stream->group_count + stream->wait_count;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, count, sizeof(*out_instance->operations),
      (void**)&out_instance->operations));
  loom_pipeline_worker_clone_state_t state = {
      .stream = source->asynchronous.streams, .instance = out_instance};
  const loom_ir_clone_observer_t observer = {
      .fn = count ? loom_pipeline_worker_clone_op : NULL, .user_data = &state};
  IREE_RETURN_IF_ERROR(loom_channel_materialization_clone(
      rewriter, source->function, &source->channels, source->graph, symbol,
      observer, &out_instance->channels));
  // Selected effects may never retain pointers into the shared source body.
  IREE_ASSERT_EQ(out_instance->operation_count, count);
  if (count > 1) {
    qsort(out_instance->operations, count, sizeof(*out_instance->operations),
          loom_pipeline_worker_compare_operations);
  }
  return iree_ok_status();
}

loom_op_t* loom_pipeline_worker_materialization_op(
    const loom_pipeline_worker_materialization_t* instance,
    const loom_op_t* source) {
  iree_host_size_t begin = 0, end = instance->operation_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if ((uintptr_t)instance->operations[middle].source_op < (uintptr_t)source) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  IREE_ASSERT_LT(begin, instance->operation_count);
  IREE_ASSERT_EQ(instance->operations[begin].source_op, source);
  return instance->operations[begin].target_op;
}

loom_value_id_t loom_pipeline_worker_materialization_value(
    const loom_pipeline_worker_materialization_t* instance,
    loom_value_id_t source) {
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  const bool mapped =
      loom_ir_remap_try_lookup_value(&instance->channels.remap, source, &value);
  // Emission must never introduce a reference to another occurrence's value.
  IREE_ASSERT(mapped);
  (void)mapped;
  return value;
}

iree_status_t loom_pipeline_worker_materialization_offset(
    const loom_pipeline_worker_materialization_t* instance,
    loom_builder_t* builder, const loom_symbolic_expr_t* source,
    loom_value_id_t anchor, loom_value_id_t* out_value) {
  loom_symbolic_expr_t expression = *source;
  loom_symbolic_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      instance->channels.remap.arena, source->term_count, sizeof(*terms),
      (void**)&terms));
  for (iree_host_size_t i = 0; i < source->term_count; ++i) {
    terms[i] = source->terms[i];
    terms[i].value_id =
        loom_pipeline_worker_materialization_value(instance, terms[i].value_id);
    terms[i].relation_value_id = loom_pipeline_worker_materialization_value(
        instance, terms[i].relation_value_id);
  }
  expression.terms = terms;
  if (loom_symbolic_expr_has_relation_value(source)) {
    expression.relation_value_id = loom_pipeline_worker_materialization_value(
        instance, source->relation_value_id);
  }
  return loom_view_materialize_offset_expression(
      builder, &expression, LOOM_VALUE_ID_INVALID,
      loom_pipeline_worker_materialization_value(instance, anchor), out_value);
}
