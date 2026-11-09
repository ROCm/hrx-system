// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sram.h"

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_resources.h"

iree_status_t loom_aie2p_native_sram_publish_helper(
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

iree_status_t loom_aie2p_native_sram_lock_helper(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel, uint16_t lock, bool acquire,
    uint32_t credits, loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  const loom_aie2p_native_worker_t* owner =
      &context
           ->workers[context->execution
                         .workers[context->core_engines[code->worker_index]]];
  const loom_aie2p_native_tile_t* tile =
      context->pool_tiles[channel->pool_index];
  const iree_host_size_t tile_index = tile - context->tiles;
  const int32_t delta = acquire ? -(int32_t)credits : (int32_t)credits;
  const uint16_t indices[] = {lock, (uint16_t)delta & 127};
  const uint64_t origin =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  const loom_xdna_register_field_id_t field =
      acquire ? LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_LOCK_ACQUIRE_SUCCESS
              : LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_LOCK_RELEASE_SUCCESS;
  const uint32_t address =
      (uint32_t)(loom_xdna_register_field_address(context->family, field,
                                                  tile->coordinate, indices) -
                 origin);
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, NULL, 0, &builder, out_symbol, &function));
  loom_value_id_t zero;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, 0, &zero));
  loom_region_t* region = loom_low_func_def_body(function);
  loom_block_t *request, *complete;
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(code->module, region, &request));
  IREE_RETURN_IF_ERROR(
      loom_region_append_block(code->module, region, &complete));
  loom_op_t* branch;
  IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, request, NULL, 0,
                                         LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder, request);
  loom_value_id_t success, pending;
  // These addresses have read side effects. Read exactly one word: rounding
  // the address or reading a prefix would execute different lock operations.
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_read(
      code, &builder, owner->control_packets[tile_index],
      owner->control_replies[tile_index], address, 1, &success));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32, success, zero,
      &pending));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      &builder, pending, request, complete, LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder, complete);
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

const loom_op_t* loom_aie2p_native_sram_reply_conflict(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* source,
    const loom_aie2p_native_worker_t* worker, iree_host_size_t worker_index) {
  if (!worker->channels.remote_count ||
      worker->execution != LOOM_AIE2P_NATIVE_EXECUTION_CORE) {
    return NULL;
  }
  // Both retained sequences follow block/source order. A request after the
  // current wait remains at the frontier for the next task interval.
  iree_host_size_t action_index = 0;
  for (const loom_aie2p_native_transfer_task_t* task = worker->transfer_tasks;
       task; task = task->next) {
    if (!task->path->completion_tokens) {
      continue;
    }
    const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
    const loom_op_t* issue = transfer->source->request.op;
    while (action_index < source->channels.action_count) {
      const loom_channel_plan_action_t* action =
          &source->channels.actions[action_index];
      const uint16_t action_block = action->op->parent_block->region_index;
      const uint16_t issue_block = issue->parent_block->region_index;
      if (action_block > issue_block) {
        break;
      }
      if (action_block < issue_block ||
          action->op->block_ordinal <= issue->block_ordinal) {
        ++action_index;
        continue;
      }
      const iree_host_size_t channel =
          loom_pipeline_invocations_channel_index(invocations, action->channel);
      const loom_aie2p_native_channel_access_t* access =
          &worker->channels.accesses[worker->channels.indices[channel]];
      if (access->window || access->batch_publication ||
          (action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE &&
           action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE &&
           !source->completion.actions[action_index].credits)) {
        ++action_index;
        continue;
      }
      if (loom_aie2p_native_transfer_retired(transfer, worker_index,
                                             action->op)) {
        break;
      }
      return action->op;
    }
  }
  return NULL;
}
