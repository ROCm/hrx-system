// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/channel_state.h"
#include "loom/transforms/pipeline/control.h"

static iree_status_t loom_aie2p_native_program_register(
    void* user_data, iree_host_size_t engine, loom_func_like_t function) {
  loom_aie2p_native_context_t* context = user_data;
  return loom_pipeline_realization_register_helper(
      context->code.module, context->code.realization,
      context->execution.workers[engine], function,
      context->pass->diagnostic_emitter);
}

static iree_status_t loom_aie2p_native_program_boundary(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_invocation_t* child, bool before,
    const loom_value_id_t* state_arguments) {
  loom_aie2p_native_context_t* context = user_data;
  if (child && before && child->resources->channel_count) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_initialize_cursors(
        context, builder, state_arguments[0], engine, child));
    IREE_RETURN_IF_ERROR(loom_aie2p_native_initialize_endpoints(
        context, builder, engine, child));
  }
  if (child && context->distribution_controllers) {
    const iree_host_size_t worker =
        context->distribution_controllers[child->ordinal *
                                              context->execution.engine_count +
                                          engine];
    if (worker != IREE_HOST_SIZE_MAX) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_distribution_boundary(
          context, builder, worker, child, before));
    }
  }
  // Every scope participant joins, including engines absent from this child.
  // Local finite DMA retirement precedes arrival at the collective.
  if (scope && !before) {
    return loom_aie2p_native_join_boundary(context, builder, engine, scope);
  }
  return iree_ok_status();
}

// Completion belongs to the whole physical entry. A child returns to its
// caller instead of releasing this semaphore or entering the quiescent loop.
static iree_status_t loom_aie2p_native_program_completion(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_worker_t* worker, loom_symbol_ref_t* out_symbol) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, NULL, 0, NULL, 0, &builder, out_symbol, &function));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_lock(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_LOCK_RELEASE_IMMEDIATE,
      worker->completion_selector, 1));
  loom_block_t* park = NULL;
  IREE_RETURN_IF_ERROR(loom_region_append_block(
      code->module, loom_low_func_def_body(function), &park));
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, park, NULL, 0,
                                         LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder, park);
  loom_value_id_t unused = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
      loom_named_attr_slice_empty(), &code->scalar_type, &unused));
  return loom_low_br_build(&builder, park, NULL, 0, LOOM_LOCATION_UNKNOWN,
                           &branch);
}

static iree_status_t loom_aie2p_native_program_frame(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_pipeline_realization_t* realization, iree_host_size_t engine,
    loom_value_id_t* out_frame) {
  const iree_host_size_t channel_count = realization->invocations.channel_count;
  loom_op_t *size = NULL, *allocation = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(channel_count * 2 * sizeof(uint32_t)),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &size));
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      builder, 0, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE, LOOM_VALUE_ID_INVALID,
      sizeof(uint32_t), loom_index_constant_result(size), loom_type_buffer(),
      LOOM_LOCATION_UNKNOWN, &allocation));
  *out_frame = loom_buffer_alloca_result(allocation);
  return loom_aie2p_native_initialize_cursors(
      context, builder, *out_frame, engine,
      realization->invocations.entries[0]);
}

iree_status_t loom_aie2p_native_emit_programs(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  const loom_value_id_t** state_arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, realization->invocations.worker_count,
      sizeof(*state_arguments), (void**)&state_arguments));
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    state_arguments[i] = &context->workers[i].cursor_state_argument;
  }
  context->code.realization = realization;
  const loom_type_t frame_type = loom_type_buffer();
  const loom_pipeline_control_options_t options = {
      .worker_engines = context->core_engines,
      .boundary_workers = context->distribution_controllers,
      .state_types = &frame_type,
      .state_count = 1,
      .worker_state_arguments = state_arguments,
      .emit = {.function = loom_aie2p_native_program_register,
               .boundary = loom_aie2p_native_program_boundary,
               .collective_begin = loom_aie2p_native_collective_begin,
               .collective_end = loom_aie2p_native_collective_end,
               .user_data = context}};
  loom_pipeline_control_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_control_materialize(
      rewriter, realization, &context->execution, &options, &functions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, context->execution.engine_count,
      sizeof(*context->programs), (void**)&context->programs));
  for (iree_host_size_t engine = 0; engine < context->execution.engine_count;
       ++engine) {
    context->programs[engine] = (loom_func_like_t){0};
    const loom_pipeline_control_function_t* projected = &functions[engine];
    if (!loom_func_like_isa(projected->function)) {
      continue;
    }
    // Source arguments precede generated state in the retained ABI. DMA
    // materialization removes supported external buffer captures; observing
    // their metadata still requires a caller value that this entry cannot
    // supply. Never substitute private cursor storage for that buffer.
    if (projected->argument_count &&
        projected->arguments[0] < context->binding_count) {
      return loom_aie2p_native_reject(
          context, realization->function.op,
          IREE_SV("external buffer arguments used only through DMA endpoints"));
    }
    const iree_host_size_t representative = context->execution.workers[engine];
    const loom_location_id_t location =
        realization->workers[representative].source->declaration->location;
    context->code.worker_index = representative;
    loom_symbol_ref_t complete;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_program_completion(
        context, &context->workers[representative], &complete));
    loom_symbol_ref_t symbol;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_symbol(&context->code, &symbol));
    loom_builder_enter_region(&rewriter->builder, NULL, rewriter->module->body);
    loom_op_t* entry = NULL;
    IREE_RETURN_IF_ERROR(loom_func_def_build(
        &rewriter->builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, NULL, 0,
        NULL, 0, NULL, 0, NULL, 0, location, &entry));
    const loom_func_like_t function =
        loom_func_like_cast(rewriter->module, entry);
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_program_register(context, engine, function));
    loom_builder_enter_region(&rewriter->builder, entry,
                              loom_func_def_body(entry));
    // Physical materialization erased source buffer captures. Only the
    // invocation-private frame survives when the entry has advancing cursors.
    loom_value_id_t frame = LOOM_VALUE_ID_INVALID;
    if (projected->argument_count) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_program_frame(
          context, &rewriter->builder, realization, engine, &frame));
    }
    loom_op_t* call = NULL;
    IREE_RETURN_IF_ERROR(
        loom_func_call_build(&rewriter->builder, 0, 0, 0, 0,
                             loom_func_like_callee(projected->function), &frame,
                             projected->argument_count, NULL, 0, NULL, 0,
                             LOOM_LOCATION_UNKNOWN, &call));
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(&rewriter->builder, 0, 0, 0, 0, complete, NULL, 0,
                              NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &call));
    IREE_RETURN_IF_ERROR(loom_func_return_build(&rewriter->builder, NULL, 0,
                                                LOOM_LOCATION_UNKNOWN, &call));
    context->programs[engine] = function;
  }
  *out_valid = true;
  return iree_ok_status();
}
