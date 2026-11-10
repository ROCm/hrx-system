// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_submission.h"

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

static uint32_t loom_aie2p_native_dma_queue_word(
    const loom_aie2p_native_transfer_task_t* task, bool shim) {
  const loom_aie2p_native_dma_path_t* path = task->path;
  const loom_xdna_register_field_id_t queue =
      shim
          ? (path->ingress
                 ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID
                 : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID)
          : path->local->memory_layout->directions[path->ingress ? 1 : 0].start;
  const uint16_t descriptor = shim
                                  ? task->shim_descriptor->sequence.descriptor
                                  : task->local_descriptor->sequence.descriptor;
  uint32_t word = loom_xdna_register_field_encode(queue, descriptor);
  if (path->completion_tokens && (shim || path->completion_tokens == 2)) {
    const loom_xdna_register_field_id_t token =
        shim
            ? (path->ingress
                   ? LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_ENABLE_TOKEN_ISSUE
                   : LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE)
            : path->local->memory_layout->directions[path->ingress ? 1 : 0]
                  .token;
    word |= loom_xdna_register_field_encode(token, 1);
  }
  return word;
}

static iree_status_t loom_aie2p_native_dma_submit_helper(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_transfer_task_t* task, bool shared,
    loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  const loom_aie2p_native_dma_path_t* path = task->path;
  const bool external_dynamic = task->shim_descriptor->address_dynamic;
  const bool local_dynamic = task->local_descriptor->address_dynamic;
  const bool local_queue_argument =
      shared && task->submit_group->local_queue_argument;
  loom_type_t* types = NULL;
  iree_host_size_t argument_count = 0;
  const iree_host_size_t maximum_argument_count =
      task->transfer_count * (shared ? 4 : 2) + shared + local_queue_argument;
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
  if (local_queue_argument) {
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
      loom_value_id_t base = relocation_address;
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
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_external_address(
          code, &builder, base, external_offset, &external_low,
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
  if (!local_queue_argument) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, (int32_t)loom_aie2p_native_dma_queue_word(task, false),
        &local_task));
  }
  if (shared) {
    shim_task = arguments[argument_index++];
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, (int32_t)loom_aie2p_native_dma_queue_word(task, true),
        &shim_task));
  }
  if (local_queue_argument) {
    local_task = arguments[argument_index++];
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
  const bool local_queue_argument = task->submit_group->local_queue_argument;
  const iree_host_size_t call_argument_count =
      task->transfer_count * (3 + local_dynamic) + 1 + local_queue_argument;
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
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, &builder, (int32_t)loom_aie2p_native_dma_queue_word(task, true),
      &call_arguments[argument_count++]));
  if (local_queue_argument) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, (int32_t)loom_aie2p_native_dma_queue_word(task, false),
        &call_arguments[argument_count++]));
  }
  loom_op_t* call;
  IREE_RETURN_IF_ERROR(loom_low_func_call_build(
      &builder, 0, 0, 0, 0, task->submit_group->function, call_arguments,
      argument_count, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &call));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

iree_status_t loom_aie2p_native_emit_submission(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_transfer_task_t* task) {
  // Short and rarely repeated bodies keep ordinary constant propagation and
  // avoid the call/return, argument moves, and helper alignment overhead.
  if (task->submit_group && task->submit_group->use_count >=
                                LOOM_AIE2P_NATIVE_DMA_SUBMIT_MIN_SHARED_USES) {
    if (!loom_symbol_ref_is_valid(task->submit_group->function)) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_submit_helper(
          context, task, true, &task->submit_group->function));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_submit_adapter(context, task));
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_dma_submit_helper(
        context, task, false, &task->submit));
  }
  return iree_ok_status();
}
