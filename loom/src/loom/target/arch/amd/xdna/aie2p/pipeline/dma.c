// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_submission.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

loom_aie2p_native_queue_status_t loom_aie2p_native_queue_status(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_tile_t* tile, uint16_t engine,
    loom_xdna_dma_direction_t direction) {
  const loom_xdna_register_field_id_t* fields =
      tile->memory_layout
          ? tile->memory_layout->directions[direction - 1].status
          : NULL;
  static const loom_xdna_register_field_id_t shim_status[2][6] = {
      {LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_TASK_QUEUE_SIZE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_CHANNEL_RUNNING,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_ACQUIRE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_LOCK_RELEASE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_STREAM_BACKPRESSURE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_STATUS_STALLED_COMPLETION_TOKEN},
      {LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_TASK_QUEUE_SIZE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_CHANNEL_RUNNING,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_ACQUIRE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_LOCK_RELEASE,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_STREAM_STARVATION,
       LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_STATUS_STALLED_TOKEN_OR_COUNT_FIFO},
  };
  if (!fields) {
    fields = shim_status[direction - 1];
  }
  uint32_t mask = loom_xdna_register_field_encode(fields[0], 7);
  for (iree_host_size_t i = 1; i < 6; ++i) {
    mask |= loom_xdna_register_field_encode(fields[i], 1);
  }
  return (loom_aie2p_native_queue_status_t){
      .address = loom_xdna_register_field_address(context->family, fields[0],
                                                  tile->coordinate, &engine),
      .pending_mask = mask,
  };
}

static iree_status_t loom_aie2p_native_dma_local_word(
    const loom_aie2p_native_transfer_t* transfer,
    const loom_pipeline_worker_materialization_t* instance,
    loom_builder_t* builder, loom_value_id_t* out_word) {
  const loom_aie2p_native_dma_path_t* path = transfer->local_descriptor->path;
  const loom_aie2p_native_memory_layout_t* layout = path->local->memory_layout;
  const loom_location_id_t location = transfer->source->request.op->location;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t word_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const uint8_t encoding_shift = path->local->facts->dma.address_encoding_shift;
  loom_value_id_t projection;
  IREE_RETURN_IF_ERROR(loom_pipeline_worker_materialization_offset(
      instance, builder, &transfer->local_view->projection_byte_offset,
      transfer->local_view->view_value_id, &projection));
  loom_builder_set_before(builder, loom_pipeline_worker_materialization_op(
                                       instance, transfer->source->request.op));
  loom_op_t *base, *address;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder,
      loom_attr_i64(layout->local_address_base +
                    transfer->local_channel->byte_offset),
      offset_type, location, &base));
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(builder, loom_index_constant_result(base),
                           projection, offset_type, location, &address));
  if (transfer->record_dynamic) {
    // The consuming channel rewrite substitutes the record's byte offset in
    // this expression before any source analysis resumes.
    IREE_RETURN_IF_ERROR(
        loom_index_add_build(builder, loom_index_add_result(address),
                             loom_pipeline_worker_materialization_value(
                                 instance, transfer->local_record),
                             offset_type, location, &address));
  }
  const loom_xdna_register_field_info_t address_field =
      loom_xdna_register_field_info(layout->address);
  const int address_shift =
      address_field.least_significant_bit - encoding_shift;
  loom_op_t *integer, *shift, *shifted, *length, *word;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_index_add_result(address),
                            offset_type, word_type, location, &integer));
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder,
      loom_attr_i64(address_shift < 0 ? -address_shift : address_shift),
      word_type, location, &shift));
  loom_value_id_t address_bits;
  if (address_shift < 0) {
    IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
        builder, loom_index_cast_result(integer),
        loom_scalar_constant_result(shift), word_type, location, &shifted));
    address_bits = loom_scalar_shrui_result(shifted);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        builder, 0, loom_index_cast_result(integer),
        loom_scalar_constant_result(shift), word_type, location, &shifted));
    address_bits = loom_scalar_shli_result(shifted);
  }
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder,
      loom_attr_i64(
          (int32_t)transfer->local_descriptor->sequence
              .words[transfer->descriptor_ordinal][layout->address_word]),
      word_type, location, &length));
  IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
      builder, address_bits, loom_scalar_constant_result(length), word_type,
      location, &word));
  *out_word = loom_scalar_ori_result(word);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_dma_wait_helper(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_transfer_task_t* task) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, NULL, 0, &builder, &task->wait, &function));
  if (!task->path->completion_tokens) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_lock(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_LOCK_ACQUIRE_IMMEDIATE,
        task->local_descriptor->completion->selector, -1));
  } else {
    for (uint8_t i = 0; i < task->path->completion_tokens; ++i) {
      loom_value_id_t completed;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
          loom_named_attr_slice_empty(), &code->scalar_type, &completed));
    }
  }
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

iree_status_t loom_aie2p_native_emit_transfers(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index) {
  loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  const loom_pipeline_worker_t* source = &realization->workers[worker_index];
  const loom_pipeline_worker_materialization_t* instance =
      &realization->instances[worker_index];
  loom_builder_t* builder = &rewriter->builder;
  for (loom_aie2p_native_transfer_task_t* task = worker->transfer_tasks; task;
       task = task->next) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_emit_submission(context, task));
    IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_wait_helper(context, task));
    loom_value_id_t* arguments = NULL;
    const iree_host_size_t maximum_argument_count = task->transfer_count * 2;
    if (maximum_argument_count) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          context->pass->arena, maximum_argument_count, sizeof(*arguments),
          (void**)&arguments));
    }
    iree_host_size_t argument_count = 0;
    for (loom_aie2p_native_transfer_t* transfer = task->first_transfer;
         transfer; transfer = transfer->task_next) {
      if (task->shim_descriptor->address_dynamic) {
        loom_value_id_t offset;
        if (transfer->external_view->begin_value_id != LOOM_VALUE_ID_INVALID) {
          offset = loom_pipeline_worker_materialization_value(
              instance, transfer->external_view->begin_value_id);
        } else {
          IREE_RETURN_IF_ERROR(loom_pipeline_worker_materialization_offset(
              instance, builder, &transfer->external_view->begin_byte_offset,
              transfer->external_view->view_value_id, &offset));
        }
        loom_builder_set_before(builder,
                                loom_pipeline_worker_materialization_op(
                                    instance, transfer->source->request.op));
        loom_op_t *integer, *narrowed;
        IREE_RETURN_IF_ERROR(loom_index_cast_build(
            builder, offset, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
            loom_type_scalar(LOOM_SCALAR_TYPE_I64), LOOM_LOCATION_UNKNOWN,
            &integer));
        IREE_RETURN_IF_ERROR(
            loom_scalar_trunci_build(builder, loom_op_results(integer)[0],
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                                     loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                     LOOM_LOCATION_UNKNOWN, &narrowed));
        arguments[argument_count++] = loom_op_results(narrowed)[0];
      }
      if (task->local_descriptor->address_dynamic) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_local_word(
            transfer, instance, builder, &arguments[argument_count++]));
      }
    }
    loom_builder_set_before(
        builder, loom_pipeline_worker_materialization_op(
                     instance, task->last_transfer->source->request.op));
    loom_op_t* invoke;
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(
        builder, 0, 0, 0, 0, task->submit, arguments, argument_count, NULL, 0,
        NULL, 0, LOOM_LOCATION_UNKNOWN, &invoke));
    loom_builder_set_before(builder,
                            loom_pipeline_worker_materialization_op(
                                instance, task->first_transfer->completion));
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, 0, task->wait,
                                               NULL, 0, NULL, 0, NULL, 0,
                                               LOOM_LOCATION_UNKNOWN, &invoke));
  }
  // Remove tokens in dependency order after every admitted completion has
  // been emitted.
  for (const loom_kernel_async_stream_t* stream = source->asynchronous.streams;
       stream; stream = stream->next) {
    for (iree_host_size_t i = 0; i < stream->wait_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(
          rewriter,
          loom_pipeline_worker_materialization_op(instance, stream->waits[i])));
    }
    for (iree_host_size_t i = 0; i < stream->group_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_erase(rewriter, loom_pipeline_worker_materialization_op(
                                            instance, stream->groups[i].op)));
    }
    for (iree_host_size_t i = 0; i < stream->transfer_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(
          rewriter, loom_pipeline_worker_materialization_op(
                        instance, stream->transfers[i].request.op)));
    }
  }
  return iree_ok_status();
}
