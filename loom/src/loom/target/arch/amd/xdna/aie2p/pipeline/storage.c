// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/storage.h"

#include "loom/ops/buffer/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/util/fact_extensions.h"

iree_status_t loom_aie2p_native_select_storage_accesses(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_t* invocation,
    const loom_pipeline_worker_t* source, loom_aie2p_native_worker_t* worker,
    bool* out_valid) {
  *out_valid = false;
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(source->function, &argument_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, argument_count, sizeof(*worker->storage.accesses),
      (void**)&worker->storage.accesses));
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_id_t argument = arguments[i];
    const loom_type_t type =
        loom_module_value_type(context->code.module, argument);
    if ((!loom_type_is_buffer(type) && !loom_type_is_view(type)) ||
        !loom_module_value_has_uses(context->code.module, argument)) {
      continue;
    }
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(&source->facts, argument);
    loom_value_id_t root;
    loom_value_fact_memory_space_t memory_space;
    loom_value_facts_t byte_offset = loom_value_facts_exact_i64(0);
    if (loom_type_is_buffer(type)) {
      loom_value_fact_buffer_reference_t reference;
      if (!loom_value_facts_query_buffer_reference(&source->facts.context,
                                                   facts, &reference)) {
        return loom_aie2p_native_reject(
            context, source->source->call,
            IREE_SV("a resolved storage root for each core buffer capture"));
      }
      root = loom_value_fact_buffer_reference_resolve_root_value(reference,
                                                                 argument);
      memory_space = reference.memory_space;
    } else {
      loom_value_fact_view_reference_t reference;
      if (!loom_value_facts_query_view_reference(&source->facts.context, facts,
                                                 &reference)) {
        return loom_aie2p_native_reject(
            context, source->source->call,
            IREE_SV("a resolved storage root for each core view capture"));
      }
      root = loom_value_fact_view_reference_resolve_root_value(reference,
                                                               argument);
      memory_space = reference.memory_space;
      byte_offset = reference.base_byte_offset;
    }
    if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      continue;
    }
    int64_t offset;
    if (!loom_value_facts_as_exact_i64(byte_offset, &offset) || offset < 0 ||
        (uint64_t)offset > UINT32_MAX) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("a fixed local byte offset for each core storage capture"));
    }
    const loom_pipeline_storage_allocation_t* allocation =
        loom_pipeline_invocation_lookup_allocation(invocation, root);
    if (!allocation || allocation->source->byte_length == 0 ||
        allocation->byte_offset > UINT32_MAX ||
        allocation->source->byte_length > UINT32_MAX) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("an invocation-owned finite local allocation for each core "
                  "storage capture"));
    }
    const loom_aie2p_native_tile_t* owner =
        context->pool_tiles[allocation->source->pool_index];
    if (owner->facts->kind == LOOM_XDNA_TILE_KIND_MEMORY) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("DMA-owned SRAM storage with compute-memory handoffs"));
    }
    const loom_xdna_address_window_t* window = loom_xdna_array_load_window(
        context->family, worker->tile->coordinate, owner->coordinate);
    if (!window) {
      return loom_aie2p_native_reject(
          context, source->source->call,
          IREE_SV("captured storage directly visible from its consuming core"));
    }
    loom_aie2p_native_storage_access_t* access =
        &worker->storage.accesses[worker->storage.count++];
    access->argument = argument;
    access->alignment = allocation->source->byte_alignment;
    access->address = window->base + (uint32_t)allocation->byte_offset;
    access->byte_offset = (uint32_t)offset;
  }
  *out_valid = true;
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

iree_status_t loom_aie2p_native_address_helper(
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

iree_status_t loom_aie2p_native_materialize_storage_accesses(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_aie2p_native_worker_t* worker,
    const loom_pipeline_worker_materialization_t* instance) {
  const loom_type_t buffer_type = loom_type_buffer();
  for (iree_host_size_t i = 0; i < worker->storage.count; ++i) {
    const loom_aie2p_native_storage_access_t* access =
        &worker->storage.accesses[i];
    loom_symbol_ref_t helper;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_address_helper(context, access->address, &helper));

    loom_op_t* call;
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(
        &rewriter->builder, 0, 0, 0, 0, helper, NULL, 0, &buffer_type, 1, NULL,
        0, LOOM_LOCATION_UNKNOWN, &call));
    const loom_value_id_t buffer = loom_op_results(call)[0];
    loom_op_t* aligned;
    IREE_RETURN_IF_ERROR(loom_buffer_assume_alignment_build(
        &rewriter->builder, &buffer, 1, access->alignment, &buffer_type, 1,
        LOOM_LOCATION_UNKNOWN, &aligned));
    const loom_value_id_t argument =
        loom_pipeline_worker_materialization_value(instance, access->argument);
    const loom_type_t type = loom_module_value_type(rewriter->module, argument);
    loom_value_id_t replacement = loom_op_results(aligned)[0];
    if (loom_type_is_view(type)) {
      loom_op_t *origin, *view;
      // Preserve the common allocation base so ordinary view lowering can
      // share its address register and fold displacements into memory operands.
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &rewriter->builder, loom_attr_i64(access->byte_offset),
          loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN,
          &origin));
      IREE_RETURN_IF_ERROR(loom_buffer_view_build(
          &rewriter->builder, replacement, loom_index_constant_result(origin),
          type, LOOM_LOCATION_UNKNOWN, &view));
      replacement = loom_buffer_view_result(view);
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_with(rewriter, argument, replacement));
  }
  return iree_ok_status();
}
