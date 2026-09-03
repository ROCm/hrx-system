// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/array_plan.h"

#include <inttypes.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/printer/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/aie2p/array/program.h"
#include "loom/target/arch/amd/xdna/aie2p/array/resident.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

static bool loom_aie2p_array_plan_check_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("aie2p-array-plan"));
}

static iree_status_t loom_aie2p_array_plan_check_parse_symbol(
    iree_string_view_t target_options, iree_string_view_t* out_symbol_name) {
  target_options = iree_string_view_trim(target_options);
  if (!iree_string_view_starts_with(target_options, IREE_SV("@")) ||
      target_options.size == 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "aie2p-array-plan requires one array function symbol");
  }
  iree_string_view_t symbol = iree_string_view_empty();
  iree_string_view_t remaining = iree_string_view_empty();
  iree_string_view_split(target_options, ' ', &symbol, &remaining);
  if (!iree_string_view_is_empty(iree_string_view_trim(remaining))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aie2p-array-plan accepts no options after the "
                            "array function symbol");
  }
  *out_symbol_name = iree_string_view_substr(symbol, 1, IREE_HOST_SIZE_MAX);
  return iree_ok_status();
}

static bool loom_aie2p_array_plan_check_has_contract(
    const loom_module_t* module, loom_op_t* function_op,
    iree_string_view_t expected_contract) {
  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  return contract_id < module->strings.count &&
         iree_string_view_equal(module->strings.entries[contract_id],
                                expected_contract);
}

static iree_status_t loom_aie2p_array_plan_check_compile_leaves(
    const loom_check_emit_provider_request_t* request,
    iree_diagnostic_emitter_t diagnostic_emitter,
    loom_aie2p_array_leaf_t** out_leaves, iree_host_size_t* out_leaf_count) {
  *out_leaves = NULL;
  *out_leaf_count = 0;
  iree_host_size_t leaf_count = 0;
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(request->module, symbol) {
    if (symbol->defining_op && loom_low_func_def_isa(symbol->defining_op) &&
        loom_aie2p_array_plan_check_has_contract(
            request->module, symbol->defining_op,
            IREE_SV("amd.xdna.aie2p.core"))) {
      ++leaf_count;
    }
  }
  if (leaf_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AIE2P array module contains no core leaves");
  }

  loom_aie2p_array_leaf_t* leaves = NULL;
  loom_aie2p_leaf_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->case_arena, leaf_count, sizeof(*leaves), (void**)&leaves));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->case_arena, leaf_count, sizeof(*contributions),
      (void**)&contributions));

  const loom_aie2p_leaf_compile_options_t compile_options = {
      .descriptor_registry = &request->low_registry->registry,
      .diagnostic_emitter = diagnostic_emitter,
  };
  iree_host_size_t leaf_index = 0;
  loom_module_for_each_symbol(request->module, symbol) {
    if (!symbol->defining_op || !loom_low_func_def_isa(symbol->defining_op) ||
        !loom_aie2p_array_plan_check_has_contract(
            request->module, symbol->defining_op,
            IREE_SV("amd.xdna.aie2p.core"))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(
        request->module, symbol->defining_op, &compile_options,
        request->case_arena, &contributions[leaf_index]));
    leaves[leaf_index] = (loom_aie2p_array_leaf_t){
        .entry = {.module_id = 0,
                  .symbol_id =
                      (loom_symbol_id_t)(symbol -
                                         request->module->symbols.entries)},
        .contribution = &contributions[leaf_index],
    };
    ++leaf_index;
  }
  *out_leaves = leaves;
  *out_leaf_count = leaf_count;
  return iree_ok_status();
}

static const char* loom_aie2p_array_plan_check_access_name(
    loom_aie2p_array_binding_access_t access) {
  switch (access) {
    case LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ:
      return "read";
    case LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE:
      return "write";
    case LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ_WRITE:
      return "read-write";
  }
  IREE_ASSERT_UNREACHABLE("validated binding access");
  return "unknown";
}

static const char* loom_aie2p_array_plan_check_direction_name(
    loom_aie2p_array_endpoint_direction_t direction) {
  return direction == LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND ? "send"
                                                               : "receive";
}

static const char* loom_aie2p_array_plan_check_dma_direction_name(
    loom_aie2p_array_dma_direction_t direction) {
  return direction == LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM
             ? "memory-to-stream"
             : "stream-to-memory";
}

static const char* loom_aie2p_array_plan_check_transport_name(
    loom_aie2p_array_channel_transport_t transport) {
  return transport == LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA
             ? "external-dma"
             : "neighbor-memory";
}

static const char* loom_aie2p_array_plan_check_switch_name(
    loom_aie2p_array_switch_kind_t switch_kind) {
  return switch_kind == LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH ? "switch"
                                                                   : "shim-mux";
}

static const char* loom_aie2p_array_plan_check_port_name(
    loom_xdna_stream_port_t port) {
  switch (port) {
    case LOOM_XDNA_STREAM_PORT_CORE:
      return "core";
    case LOOM_XDNA_STREAM_PORT_DMA:
      return "dma";
    case LOOM_XDNA_STREAM_PORT_TILE_CONTROL:
      return "tile-control";
    case LOOM_XDNA_STREAM_PORT_FIFO:
      return "fifo";
    case LOOM_XDNA_STREAM_PORT_SOUTH:
      return "south";
    case LOOM_XDNA_STREAM_PORT_WEST:
      return "west";
    case LOOM_XDNA_STREAM_PORT_NORTH:
      return "north";
    case LOOM_XDNA_STREAM_PORT_EAST:
      return "east";
    case LOOM_XDNA_STREAM_PORT_TRACE:
      return "trace";
  }
  IREE_ASSERT_UNREACHABLE("validated stream port");
  return "unknown";
}

static iree_status_t loom_aie2p_array_plan_check_append_endpoint(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_endpoint_t* endpoint,
    iree_string_builder_t* builder) {
  const loom_aie2p_array_endpoint_t* base = endpoint;
  if (endpoint->partition_source_endpoint_index != UINT32_MAX) {
    base = &plan->endpoints[endpoint->partition_source_endpoint_index];
  }
  if (base->owner_kind == LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "binding[%" PRIu32 "]:%" PRIu32,
        plan->bindings[base->owner_index].ordinal, base->port));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "worker[%" PRIu32 "]:%" PRIu32, base->owner_index,
        base->port));
  }
  if (endpoint->partition_source_endpoint_index != UINT32_MAX) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " partition=%" PRIu32 "/%" PRIu32, endpoint->partition_lane,
        endpoint->partition_lane_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_check_format(
    const loom_module_t* module, const loom_aie2p_array_plan_t* plan,
    iree_string_builder_t* builder) {
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(module, plan->function_op);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "array @%.*s family=%s groups=%" PRIhsz " workers=%" PRIhsz
      " bindings=%" PRIhsz " channels=%" PRIhsz " slots=%" PRIhsz
      " locks=%" PRIhsz " dma=%" PRIhsz " routes=%" PRIhsz "\n",
      (int)function_name.size, function_name.data, plan->family->key,
      plan->group_count, plan->worker_count, plan->binding_count,
      plan->channel_count, plan->channel_slot_count, plan->lock_count,
      plan->dma_channel_count, plan->route_count));

  for (iree_host_size_t i = 0; i < plan->binding_count; ++i) {
    const loom_aie2p_array_binding_t* binding = &plan->bindings[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "binding index=%" PRIhsz " ordinal=%" PRIu32 " access=%s\n", i,
        binding->ordinal,
        loom_aie2p_array_plan_check_access_name(binding->access)));
  }
  for (iree_host_size_t i = 0; i < plan->worker_count; ++i) {
    const loom_aie2p_array_worker_t* worker = &plan->workers[i];
    const loom_aie2p_leaf_realization_t* realization =
        &plan->worker_plans[i].contribution->realization;
    const iree_string_view_t entry_name =
        loom_low_diagnostic_symbol_name(module, worker->entry);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "worker index=%" PRIhsz " group=%" PRIu32 " lane=%" PRIu32
        " entry=@%.*s tile=(%u,%u) code-bytes=%" PRIu64 " resources=%" PRIhsz
        " storage-domains=%" PRIhsz "\n",
        i, worker->group_index, worker->lane, (int)entry_name.size,
        entry_name.data, worker->coordinate.column, worker->coordinate.row,
        realization->code.byte_length, realization->resource_import_count,
        realization->storage_domain_count));
  }
  for (iree_host_size_t i = 0; i < plan->channel_count; ++i) {
    const loom_aie2p_array_channel_t* channel = &plan->channels[i];
    const loom_aie2p_array_endpoint_t* sender =
        &plan->endpoints[channel->sender_endpoint_index];
    const loom_aie2p_array_endpoint_t* receiver =
        &plan->endpoints[channel->receiver_endpoint_index];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "channel index=%" PRIhsz " transport=%s record-bytes=%" PRIu32
        " capacity=%" PRIu32 " sender=",
        i, loom_aie2p_array_plan_check_transport_name(channel->transport),
        channel->record_byte_length, channel->capacity));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_array_plan_check_append_endpoint(plan, sender, builder));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " receiver="));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_array_plan_check_append_endpoint(plan, receiver, builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  for (iree_host_size_t i = 0; i < plan->worker_storage_count; ++i) {
    const loom_aie2p_array_worker_storage_plan_t* storage =
        &plan->worker_storage[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "worker-storage worker=%" PRIu32 " space=%u offset=0x%05" PRIx32
        " load-address=0x%05" PRIx32 " bytes=%" PRIu32 "\n",
        storage->worker_index, storage->storage_space, storage->owner_offset,
        storage->load_address, storage->byte_length));
  }
  for (iree_host_size_t i = 0; i < plan->worker_port_count; ++i) {
    const loom_aie2p_array_worker_port_plan_t* port = &plan->worker_ports[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "worker-port worker=%" PRIu32 " port=%" PRIu32
        " direction=%s"
        " channel=%" PRIu32 " first-slot=%" PRIu32 "\n",
        port->worker_index, port->port,
        loom_aie2p_array_plan_check_direction_name(port->direction),
        port->channel_index, port->first_channel_slot));
  }
  for (iree_host_size_t i = 0; i < plan->channel_slot_count; ++i) {
    const loom_aie2p_array_channel_slot_t* slot = &plan->channel_slots[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "slot channel=%" PRIu32 " index=%" PRIu32
        " owner=(%u,%u)"
        " offset=0x%05" PRIx32 " bytes=%" PRIu32,
        slot->channel_index, slot->slot, slot->owner.column, slot->owner.row,
        slot->owner_offset, slot->byte_length));
    if (slot->sender_load_address == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " sender=external"));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " sender=0x%05" PRIx32, slot->sender_load_address));
    }
    if (slot->receiver_load_address == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " receiver=external"));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " receiver=0x%05" PRIx32, slot->receiver_load_address));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  for (iree_host_size_t i = 0; i < plan->lock_count; ++i) {
    const loom_aie2p_array_lock_plan_t* lock = &plan->locks[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "lock channel=%" PRIu32 " tile=(%u,%u) id=%u role=%s initial=%d\n",
        lock->channel_index, lock->coordinate.column, lock->coordinate.row,
        lock->lock_id, lock->consumer_ready ? "ready" : "credit",
        lock->initial_value));
  }
  for (iree_host_size_t i = 0; i < plan->dma_channel_count; ++i) {
    const loom_aie2p_array_dma_plan_t* dma = &plan->dma_channels[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "dma channel=%" PRIu32
        " tile=(%u,%u) side=%s direction=%s"
        " engine-channel=%u bd-start=%u bd-count=%u\n",
        dma->channel_index, dma->coordinate.column, dma->coordinate.row,
        dma->shim_side ? "shim" : "compute",
        loom_aie2p_array_plan_check_dma_direction_name(dma->direction),
        dma->dma_channel, dma->buffer_descriptor_start,
        dma->buffer_descriptor_count));
  }
  for (iree_host_size_t i = 0; i < plan->route_count; ++i) {
    const loom_aie2p_array_route_plan_t* route = &plan->routes[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "route channel=%" PRIu32
        " tile=(%u,%u) kind=%s source=%s[%u]"
        " destination=%s[%u]\n",
        route->channel_index, route->coordinate.column, route->coordinate.row,
        loom_aie2p_array_plan_check_switch_name(route->switch_kind),
        loom_aie2p_array_plan_check_port_name(route->source_port),
        route->source_channel,
        loom_aie2p_array_plan_check_port_name(route->destination_port),
        route->destination_channel));
  }
  for (iree_host_size_t i = 0; i < plan->binding_plan_count; ++i) {
    const loom_aie2p_array_binding_plan_t* binding = &plan->binding_plans[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "binding-patch ordinal=%" PRIu32 " channel=%" PRIu32
        " shim=(%u,%u) direction=%s dma-channel=%u partition=%" PRIu32
        "/%" PRIu32 "\n",
        plan->bindings[binding->binding_index].ordinal, binding->channel_index,
        binding->shim_coordinate.column, binding->shim_coordinate.row,
        loom_aie2p_array_plan_check_dma_direction_name(binding->direction),
        binding->dma_channel, binding->partition_lane,
        binding->partition_lane_count));
  }
  return iree_ok_status();
}

typedef struct loom_aie2p_array_program_check_record_counts_t {
  iree_host_size_t writes;
  iree_host_size_t masked_writes;
  iree_host_size_t block_writes;
  iree_host_size_t tile_loads;
  iree_host_size_t waits;
} loom_aie2p_array_program_check_record_counts_t;

static iree_status_t loom_aie2p_array_program_check_count_records(
    const loom_aie2p_program_record_t* records, iree_host_size_t record_count,
    loom_aie2p_array_program_check_record_counts_t* out_counts) {
  *out_counts = (loom_aie2p_array_program_check_record_counts_t){0};
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    switch (records[i].type) {
      case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
        ++out_counts->writes;
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
        ++out_counts->masked_writes;
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
        ++out_counts->block_writes;
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
        ++out_counts->tile_loads;
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
        ++out_counts->waits;
        break;
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "unknown AIE2P program record type %u",
                                (unsigned)records[i].type);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_program_check_append_block(
    iree_string_view_t program_name, iree_host_size_t record_index,
    const loom_aie2p_program_register_block_write32_t* block,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "%.*s-block record=%" PRIhsz " address=0x%08" PRIx32 " words=[",
      (int)program_name.size, program_name.data, record_index, block->address));
  for (iree_host_size_t i = 0; i < block->word_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s0x%08" PRIx32, i == 0 ? "" : ",", block->words[i]));
  }
  return iree_string_builder_append_cstring(builder, "]\n");
}

static iree_status_t loom_aie2p_array_program_check_format(
    const loom_aie2p_array_program_t* program,
    const loom_aie2p_encoded_array_program_t* encoded,
    iree_string_builder_t* builder) {
  loom_aie2p_array_program_check_record_counts_t array_counts = {0};
  loom_aie2p_array_program_check_record_counts_t control_counts = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_program_check_count_records(
      program->array_records, program->array_record_count, &array_counts));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_program_check_count_records(
      program->control_records, program->control_record_count,
      &control_counts));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "\narray-program records=%" PRIhsz " payload-bytes=%" PRIhsz
      " writes=%" PRIhsz " masked-writes=%" PRIhsz " block-writes=%" PRIhsz
      " tile-loads=%" PRIhsz " waits=%" PRIhsz "\n",
      program->array_record_count, encoded->array_payload.data_length,
      array_counts.writes, array_counts.masked_writes,
      array_counts.block_writes, array_counts.tile_loads, array_counts.waits));
  for (iree_host_size_t i = 0; i < program->array_record_count; ++i) {
    const loom_aie2p_program_record_t* record = &program->array_records[i];
    if (record->type == LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32) {
      IREE_RETURN_IF_ERROR(loom_aie2p_array_program_check_append_block(
          IREE_SV("array"), i, &record->value.register_block_write32, builder));
    } else if (record->type == LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "array-tile-load record=%" PRIhsz " tile-program=%" PRIu32 "\n", i,
          record->value.tile_program_load.tile_program_index));
    }
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "control-program records=%" PRIhsz " payload-bytes=%" PRIhsz
      " writes=%" PRIhsz " masked-writes=%" PRIhsz " block-writes=%" PRIhsz
      " tile-loads=%" PRIhsz " waits=%" PRIhsz " relocations=%" PRIhsz "\n",
      program->control_record_count, encoded->control_payload.data_length,
      control_counts.writes, control_counts.masked_writes,
      control_counts.block_writes, control_counts.tile_loads,
      control_counts.waits, program->relocation_count));
  for (iree_host_size_t i = 0; i < program->control_record_count; ++i) {
    const loom_aie2p_program_record_t* record = &program->control_records[i];
    switch (record->type) {
      case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "control-write record=%" PRIhsz " address=0x%08" PRIx32
            " value=0x%08" PRIx32 "\n",
            i, record->value.register_write32.address,
            record->value.register_write32.value));
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "control-mask-write record=%" PRIhsz " address=0x%08" PRIx32
            " mask=0x%08" PRIx32 " value=0x%08" PRIx32 "\n",
            i, record->value.register_mask_write32.address,
            record->value.register_mask_write32.mask,
            record->value.register_mask_write32.value));
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
        IREE_RETURN_IF_ERROR(loom_aie2p_array_program_check_append_block(
            IREE_SV("control"), i, &record->value.register_block_write32,
            builder));
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "control-wait record=%" PRIhsz
            " tile=(%u,%u) direction=%s"
            " dma-channel=%u columns=%u rows=%u\n",
            i, record->value.dma_task_wait.coordinate.column,
            record->value.dma_task_wait.coordinate.row,
            loom_aie2p_array_plan_check_dma_direction_name(
                record->value.dma_task_wait.direction),
            record->value.dma_task_wait.dma_channel,
            record->value.dma_task_wait.column_count,
            record->value.dma_task_wait.row_count));
        break;
      case LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P invocation-control program contains a tile load");
      default:
        IREE_ASSERT_UNREACHABLE("counted AIE2P control record type");
    }
  }
  for (iree_host_size_t i = 0; i < program->relocation_count; ++i) {
    const loom_aie2p_program_relocation_t* source = &program->relocations[i];
    const loom_xdna_elf_relocation_record_t* resolved =
        &encoded->relocations[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "control-relocation index=%" PRIhsz " record=%" PRIu32 " word=%" PRIu32
        " payload-offset=%" PRIu32 " binding=%" PRIu32 " addend=%" PRId64
        " width=%u"
        " alignment=%" PRIu64 "\n",
        i, source->target_record_index, source->target_word_index,
        resolved->target_byte_offset, source->binding_ordinal, source->addend,
        source->field_byte_width, source->required_alignment));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_check_resident_program(
    const loom_check_emit_provider_request_t* request,
    const loom_aie2p_array_plan_t* plan,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  loom_aie2p_array_resident_program_t program = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_materialize_resident_program(
      request->module, plan, request->case_arena, &program));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &request->result->actual_output,
      "\nresident-program workers=%" PRIhsz "\n", program.worker_count));

  const loom_aie2p_leaf_compile_options_t compile_options = {
      .descriptor_registry = &request->low_registry->registry,
      .diagnostic_emitter = diagnostic_emitter,
  };
  for (iree_host_size_t i = 0; i < program.worker_count; ++i) {
    const loom_aie2p_array_resident_worker_t* resident = &program.workers[i];
    loom_aie2p_leaf_contribution_t contribution = {0};
    IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(
        request->module, resident->function_op, &compile_options,
        request->case_arena, &contribution));
    if (contribution.realization.resource_import_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "materialized AIE2P resident worker retains resource imports");
    }
    const loom_xdna_tile_coordinate_t coordinate =
        plan->worker_plans[resident->worker_index].coordinate;
    const iree_string_view_t entry_name =
        loom_low_diagnostic_symbol_name(request->module, resident->entry);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &request->result->actual_output,
        "resident worker=%" PRIu32 " entry=@%.*s tile=(%u,%u) phases=%" PRIu32
        " code-bytes=%" PRIu64 " resources=%" PRIhsz " fixups=%" PRIhsz "\n",
        resident->worker_index, (int)entry_name.size, entry_name.data,
        coordinate.column, coordinate.row, resident->phase_count,
        contribution.realization.code.byte_length,
        contribution.realization.resource_import_count,
        contribution.object.fixup_count));
  }

  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_low_descriptor_text_asm_environment_initialize(
      &request->low_registry->registry, &low_asm_environment);
  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_REQUIRE_LOW_ASM,
      .low_asm_environment = low_asm_environment,
  };
  for (iree_host_size_t i = 0; i < program.worker_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &request->result->actual_output, "\n"));
    IREE_RETURN_IF_ERROR(loom_text_print_operation_to_builder_with_options(
        request->module, program.workers[i].function_op,
        &request->result->actual_output, &print_options));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_array_plan_check_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  iree_string_view_t function_symbol_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_check_parse_symbol(
      request->target_options, &function_symbol_name));

  loom_check_prepare_source_low_options_t prepare_options = {0};
  loom_check_prepare_source_low_options_initialize(&prepare_options);
  prepare_options.control_flow_lowering =
      LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
  IREE_RETURN_IF_ERROR(loom_check_prepare_source_low_module(
      request->module, &prepare_options, request->low_registry,
      request->environment, request->source_resolver,
      request->diagnostic_collector, request->block_pool));
  if (request->diagnostic_collector->count != 0) return iree_ok_status();

  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t diagnostic_emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &diagnostic_capture,
  };
  loom_op_t* array_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      request->module, function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, diagnostic_emitter,
      &array_function));
  if (array_function == NULL) return iree_ok_status();

  loom_aie2p_array_leaf_t* leaves = NULL;
  iree_host_size_t leaf_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_check_compile_leaves(
      request, diagnostic_emitter, &leaves, &leaf_count));
  if (request->diagnostic_collector->count != 0) return iree_ok_status();

  loom_aie2p_array_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_array_plan_build(request->module, array_function, leaves,
                                  leaf_count, request->case_arena, &plan));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_check_format(
      request->module, &plan, &request->result->actual_output));
  loom_aie2p_array_program_t array_program = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_program_build(
      &plan, request->case_arena, &array_program));
  loom_aie2p_encoded_array_program_t encoded_program = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_program_encode(
      &array_program, /*first_tile_program_header_ordinal=*/0,
      /*control_program_header_ordinal=*/0, request->case_arena,
      &encoded_program));
  IREE_RETURN_IF_ERROR(loom_aie2p_array_program_check_format(
      &array_program, &encoded_program, &request->result->actual_output));
  return loom_aie2p_array_plan_check_resident_program(request, &plan,
                                                      diagnostic_emitter);
}

static iree_status_t loom_aie2p_array_plan_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "aie2p-array-plan");
}

const loom_check_emit_provider_t loom_aie2p_array_plan_check_emit_provider = {
    .name = IREE_SVL("aie2p-array-plan"),
    .match = loom_aie2p_array_plan_check_matches,
    .execute = loom_aie2p_array_plan_check_execute,
    .append_names = loom_aie2p_array_plan_check_append_names,
};
