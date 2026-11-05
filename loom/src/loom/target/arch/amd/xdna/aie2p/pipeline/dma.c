// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/array/registers.h"
#include "loom/transforms/view/offset_expression.h"

static uint32_t loom_aie2p_native_register_offset(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_tile_t* tile, loom_xdna_register_field_id_t field,
    uint16_t index) {
  const uint64_t address = loom_xdna_register_field_address(
      context->family, field, tile->coordinate, &index);
  const uint64_t origin =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  return (uint32_t)(address - origin);
}

static iree_status_t loom_aie2p_native_dma_local_word(
    const loom_aie2p_native_transfer_t* transfer, loom_builder_t* builder,
    loom_value_id_t* out_word) {
  const loom_aie2p_native_dma_path_t* path = transfer->local_descriptor->path;
  const loom_aie2p_native_memory_layout_t* layout = path->local->memory_layout;
  const loom_location_id_t location = transfer->source->request.op->location;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t word_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const uint8_t encoding_shift = path->local->facts->dma.address_encoding_shift;
  loom_value_id_t projection;
  IREE_RETURN_IF_ERROR(loom_view_materialize_offset_expression(
      builder, &transfer->local_view->projection_byte_offset,
      LOOM_VALUE_ID_INVALID, transfer->local_view->view_value_id, &projection));
  loom_builder_set_before(builder, transfer->source->request.op);
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
    IREE_RETURN_IF_ERROR(loom_index_add_build(
        builder, loom_index_add_result(address), transfer->local_record,
        offset_type, location, &address));
  }
  const loom_xdna_register_field_info_t address_field =
      loom_xdna_register_field_info(layout->address);
  loom_op_t *integer, *shift, *address_bits, *length, *word;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_index_add_result(address),
                            offset_type, word_type, location, &integer));
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder,
      loom_attr_i64(address_field.least_significant_bit - encoding_shift),
      word_type, location, &shift));
  IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
      builder, 0, loom_index_cast_result(integer),
      loom_scalar_constant_result(shift), word_type, location, &address_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder,
      loom_attr_i64(
          (int32_t)transfer->local_descriptor->sequence
              .words[transfer->descriptor_ordinal][layout->address_word]),
      word_type, location, &length));
  IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
      builder, loom_scalar_shli_result(address_bits),
      loom_scalar_constant_result(length), word_type, location, &word));
  *out_word = loom_scalar_ori_result(word);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_dma_submit_helper(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_transfer_task_t* task, bool shared,
    loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  const loom_aie2p_native_dma_path_t* path = task->path;
  const bool external_dynamic = task->shim_descriptor->address_dynamic;
  const bool local_dynamic = task->local_descriptor->address_dynamic;
  loom_type_t* types = NULL;
  iree_host_size_t argument_count = 0;
  const iree_host_size_t maximum_argument_count =
      task->transfer_count * (shared ? 4 : 2) + shared;
  if (maximum_argument_count) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->pass->arena, maximum_argument_count,
                                  sizeof(*types), (void**)&types));
  }
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    if (external_dynamic) {
      types[argument_count++] = code->scalar_type;
      if (shared) {
        types[argument_count++] = code->address_type;
        types[argument_count++] = code->scalar_type;
      }
    }
    if (local_dynamic) {
      types[argument_count++] = code->scalar_type;
    }
  }
  if (shared) {
    types[argument_count++] = code->scalar_type;
  }
  loom_builder_t builder;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, types, argument_count, NULL, 0, &builder, out_symbol, &function));
  if (shared) {
    const loom_func_like_t callable =
        loom_func_like_cast(code->module, function);
    loom_op_attrs(function)[callable.vtable->inline_policy_attr_index] =
        loom_attr_enum(LOOM_INLINE_POLICY_NOINLINE);
  }
  const loom_value_id_t* arguments =
      loom_region_entry_block(loom_low_func_def_body(function))->arg_ids;
  iree_host_size_t argument_index = 0;
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    const loom_value_id_t external_offset =
        external_dynamic ? arguments[argument_index++] : LOOM_VALUE_ID_INVALID;
    const loom_value_id_t relocation_address = shared && external_dynamic
                                                   ? arguments[argument_index++]
                                                   : LOOM_VALUE_ID_INVALID;
    const loom_value_id_t external_command = shared && external_dynamic
                                                 ? arguments[argument_index++]
                                                 : LOOM_VALUE_ID_INVALID;
    const loom_value_id_t local_word =
        local_dynamic ? arguments[argument_index++] : LOOM_VALUE_ID_INVALID;
    loom_value_id_t external_low = LOOM_VALUE_ID_INVALID;
    loom_value_id_t external_high = LOOM_VALUE_ID_INVALID;
    if (external_dynamic) {
      loom_value_id_t base = relocation_address, low, high;
      if (!shared) {
        const loom_named_attr_t address_attr = {
            .name_id = code->integer_name,
            .value = loom_attr_i64(transfer->relocation->load_address)};
        IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
            code, &builder,
            AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32, NULL, 0,
            loom_make_named_attr_slice(&address_attr, 1), &code->address_type,
            &base));
      }
      for (unsigned i = 0; i < 2; ++i) {
        const loom_named_attr_t displacement = {
            .name_id = code->displacement_name, .value = loom_attr_i64(i * 4)};
        IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
            code, &builder,
            AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_I32_INDEXED_IMMEDIATE, &base,
            1, loom_make_named_attr_slice(&displacement, 1), &code->scalar_type,
            i ? &high : &low));
      }
      loom_value_id_t carry;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, low,
          external_offset, &external_low));
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_ULT_I32, external_low,
          low, &carry));
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, high, carry,
          &external_high));
    }
    if (local_dynamic) {
      const uint16_t descriptor = task->local_descriptor->sequence.descriptor +
                                  (uint16_t)transfer->descriptor_ordinal;
      const uint32_t local_descriptor = loom_aie2p_native_register_offset(
          context, path->local, path->local->memory_layout->address,
          descriptor);
      IREE_RETURN_IF_ERROR(
          loom_aie2p_worker_control_write(code, &builder, path->control.local,
                                          local_descriptor, &local_word, 1));
    }
    if (external_dynamic) {
      const uint16_t descriptor = task->shim_descriptor->sequence.descriptor +
                                  (uint16_t)transfer->descriptor_ordinal;
      const uint32_t shim_descriptor = loom_aie2p_native_register_offset(
          context, path->shim,
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD1_BASE_ADDRESS_LOW,
          descriptor);
      const loom_value_id_t external_words[] = {external_low, external_high};
      if (shared) {
        IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write_encoded(
            code, &builder, path->control.shim, external_command,
            external_words, 2));
      } else {
        IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
            code, &builder, path->control.shim, shim_descriptor, external_words,
            2));
      }
    }
  }
  const loom_xdna_register_field_id_t local_queue =
      path->local->memory_layout->directions[path->ingress ? 1 : 0].start;
  const loom_xdna_register_field_id_t shim_queue =
      path->ingress
          ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
          : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID;
  loom_value_id_t local_task, shim_task;
  uint32_t local_bits = loom_xdna_register_field_encode(
      local_queue, task->local_descriptor->sequence.descriptor);
  if (path->completion_tokens == 2) {
    local_bits |= loom_xdna_register_field_encode(
        path->local->memory_layout->directions[path->ingress ? 1 : 0].token, 1);
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, &builder, (int32_t)local_bits, &local_task));
  if (shared) {
    shim_task = arguments[argument_index++];
  } else {
    uint32_t shim_bits = loom_xdna_register_field_encode(
        shim_queue, task->shim_descriptor->sequence.descriptor);
    if (path->completion_tokens) {
      shim_bits |= loom_xdna_register_field_encode(
          path->ingress
              ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_ENABLE_TOKEN_ISSUE
              : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE,
          1);
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, (int32_t)shim_bits, &shim_task));
  }
  // Queue the receiver before its sender. The two streams then advance without
  // involving the host, and the explicit wait owns descriptor/slot retirement.
  const uint32_t local_queue_address = loom_aie2p_native_register_offset(
      context, path->local, local_queue, path->local_engine);
  const uint32_t shim_queue_address = loom_aie2p_native_register_offset(
      context, path->shim, shim_queue, path->shim_engine);
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, path->ingress ? path->control.local : path->control.shim,
      path->ingress ? local_queue_address : shim_queue_address,
      path->ingress ? &local_task : &shim_task, 1));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, path->ingress ? path->control.shim : path->control.local,
      path->ingress ? shim_queue_address : local_queue_address,
      path->ingress ? &shim_task : &local_task, 1));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

static iree_status_t loom_aie2p_native_dma_submit_adapter(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_transfer_task_t* task) {
  loom_aie2p_worker_builder_t* code = &context->code;
  const loom_aie2p_native_dma_path_t* path = task->path;
  const bool local_dynamic = task->local_descriptor->address_dynamic;
  const iree_host_size_t call_argument_count =
      task->transfer_count * (3 + local_dynamic) + 1;
  const loom_xdna_register_field_id_t shim_queue =
      path->ingress
          ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
          : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID;
  loom_builder_t builder;
  loom_op_t* function;
  const iree_host_size_t source_argument_count =
      task->transfer_count * (1 + local_dynamic);
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, source_argument_count,
                                sizeof(*types), (void**)&types));
  for (iree_host_size_t i = 0; i < source_argument_count; ++i) {
    types[i] = code->scalar_type;
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_helper(code, types, source_argument_count, NULL, 0,
                               &builder, &task->submit, &function));
  const loom_func_like_t callable = loom_func_like_cast(code->module, function);
  loom_op_attrs(function)[callable.vtable->inline_policy_attr_index] =
      loom_attr_enum(LOOM_INLINE_POLICY_INLINE);
  const loom_value_id_t* source_arguments =
      loom_region_entry_block(loom_low_func_def_body(function))->arg_ids;
  loom_value_id_t* call_arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, call_argument_count, sizeof(*call_arguments),
      (void**)&call_arguments));
  iree_host_size_t source_index = 0;
  iree_host_size_t argument_count = 0;
  for (const loom_aie2p_native_transfer_t* transfer = task->first_transfer;
       transfer; transfer = transfer->task_next) {
    call_arguments[argument_count++] = source_arguments[source_index++];
    const loom_named_attr_t address_attr = {
        .name_id = code->integer_name,
        .value = loom_attr_i64(transfer->relocation->load_address)};
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
        NULL, 0, loom_make_named_attr_slice(&address_attr, 1),
        &code->address_type, &call_arguments[argument_count++]));
    const uint16_t descriptor = task->shim_descriptor->sequence.descriptor +
                                (uint16_t)transfer->descriptor_ordinal;
    const uint32_t command = loom_aie2p_worker_control_write_command(
        loom_aie2p_native_register_offset(
            context, path->shim,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD1_BASE_ADDRESS_LOW,
            descriptor),
        2);
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, (int32_t)command, &call_arguments[argument_count++]));
    if (local_dynamic) {
      call_arguments[argument_count++] = source_arguments[source_index++];
    }
  }
  uint32_t shim_bits = loom_xdna_register_field_encode(
      shim_queue, task->shim_descriptor->sequence.descriptor);
  if (path->completion_tokens) {
    shim_bits |= loom_xdna_register_field_encode(
        path->ingress
            ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_ENABLE_TOKEN_ISSUE
            : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE,
        1);
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, &builder, (int32_t)shim_bits, &call_arguments[argument_count++]));
  loom_op_t* call;
  IREE_RETURN_IF_ERROR(loom_low_func_call_build(
      &builder, 0, 0, 0, task->submit_group->function, call_arguments,
      argument_count, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &call));
  return loom_aie2p_worker_return(&builder, NULL, 0);
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
  loom_builder_t* builder = &rewriter->builder;
  for (loom_aie2p_native_transfer_task_t* task = worker->transfer_tasks; task;
       task = task->next) {
    // Short and rarely repeated bodies keep ordinary constant propagation and
    // avoid the call/return, argument moves, and helper alignment overhead.
    if (task->submit_group && task->submit_group->use_count >= 4) {
      if (!loom_symbol_ref_is_valid(task->submit_group->function)) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_submit_helper(
            context, task, true, &task->submit_group->function));
      }
      IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_submit_adapter(context, task));
    } else {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_submit_helper(
          context, task, false, &task->submit));
    }
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
        loom_value_id_t offset = transfer->external_view->begin_value_id;
        // Keep the authored address-domain arithmetic and its guarantees when
        // the view analysis retained a complete offset value.
        if (offset == LOOM_VALUE_ID_INVALID) {
          IREE_RETURN_IF_ERROR(loom_view_materialize_offset_expression(
              builder, &transfer->external_view->begin_byte_offset,
              LOOM_VALUE_ID_INVALID, transfer->external_view->view_value_id,
              &offset));
        }
        loom_builder_set_before(builder, transfer->source->request.op);
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
            transfer, builder, &arguments[argument_count++]));
      }
    }
    loom_builder_set_before(builder, task->last_transfer->source->request.op);
    loom_op_t* invoke;
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(
        builder, 0, 0, 0, task->submit, arguments, argument_count, NULL, 0,
        NULL, 0, LOOM_LOCATION_UNKNOWN, &invoke));
    loom_builder_set_before(builder, task->first_transfer->completion);
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, task->wait,
                                               NULL, 0, NULL, 0, NULL, 0,
                                               LOOM_LOCATION_UNKNOWN, &invoke));
  }
  // Remove tokens in dependency order after every admitted completion has
  // been emitted.
  for (const loom_kernel_async_stream_t* stream = source->asynchronous.streams;
       stream; stream = stream->next) {
    for (iree_host_size_t i = 0; i < stream->wait_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, stream->waits[i]));
    }
    for (iree_host_size_t i = 0; i < stream->group_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_erase(rewriter, (loom_op_t*)stream->groups[i].op));
    }
    for (iree_host_size_t i = 0; i < stream->transfer_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(
          rewriter, (loom_op_t*)stream->transfers[i].request.op));
    }
  }
  return iree_ok_status();
}
