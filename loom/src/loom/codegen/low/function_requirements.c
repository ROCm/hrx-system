// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_requirements.h"

#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/immediates.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_read_only_data_builder_t {
  loom_low_read_only_data_requirement_t* records;
  iree_host_size_t record_capacity;
  uint16_t* ordinal_slots;
  iree_host_size_t slot_capacity;
} loom_low_read_only_data_builder_t;

static iree_host_size_t loom_low_symbol_ref_hash(
    loom_symbol_ref_t symbol, iree_host_size_t slot_capacity) {
  const uint32_t identity =
      ((uint32_t)symbol.module_id << 16) | (uint32_t)symbol.symbol_id;
  return (identity * UINT32_C(0x85EBCA77)) & (slot_capacity - 1);
}

static uint32_t loom_low_read_only_data_lookup(
    const loom_low_read_only_data_requirement_t* records,
    iree_host_size_t record_count, const uint16_t* ordinal_slots,
    iree_host_size_t slot_capacity, loom_symbol_ref_t symbol) {
  if (!loom_symbol_ref_is_valid(symbol)) {
    return UINT32_MAX;
  }
  if (slot_capacity == 0) {
    for (uint32_t ordinal = 0; ordinal < (uint32_t)record_count; ++ordinal) {
      const loom_symbol_ref_t existing = records[ordinal].symbol;
      if (existing.module_id == symbol.module_id &&
          existing.symbol_id == symbol.symbol_id) {
        return ordinal;
      }
    }
    return UINT32_MAX;
  }
  const iree_host_size_t slot_mask = slot_capacity - 1;
  iree_host_size_t slot_index = loom_low_symbol_ref_hash(symbol, slot_capacity);
  while (ordinal_slots[slot_index] != 0) {
    const uint32_t ordinal = ordinal_slots[slot_index] - 1;
    IREE_ASSERT_LT(ordinal, record_count);
    const loom_symbol_ref_t existing = records[ordinal].symbol;
    if (existing.module_id == symbol.module_id &&
        existing.symbol_id == symbol.symbol_id) {
      return ordinal;
    }
    slot_index = (slot_index + 1) & slot_mask;
  }
  return UINT32_MAX;
}

static void loom_low_read_only_data_insert_slot(
    const loom_low_read_only_data_requirement_t* records, uint16_t ordinal,
    uint16_t* ordinal_slots, iree_host_size_t slot_capacity) {
  const iree_host_size_t slot_mask = slot_capacity - 1;
  iree_host_size_t slot_index =
      loom_low_symbol_ref_hash(records[ordinal].symbol, slot_capacity);
  while (ordinal_slots[slot_index] != 0) {
    slot_index = (slot_index + 1) & slot_mask;
  }
  ordinal_slots[slot_index] = ordinal + 1;
}

static iree_status_t loom_low_function_requirements_append_read_only_data(
    const loom_module_t* module, loom_symbol_ref_t symbol,
    iree_arena_allocator_t* arena, loom_low_read_only_data_builder_t* builder,
    loom_low_function_requirements_t* out_requirements) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol));
  IREE_ASSERT_EQ(symbol.module_id, 0);
  IREE_ASSERT_LT(symbol.symbol_id, module->symbols.count);
  if (loom_low_read_only_data_lookup(
          builder->records, out_requirements->read_only_data_count,
          builder->ordinal_slots, builder->slot_capacity,
          symbol) != UINT32_MAX) {
    return iree_ok_status();
  }
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  IREE_ASSERT(loom_global_rodata_def_isa(definition));
  IREE_ASSERT_LT(out_requirements->read_only_data_count, UINT16_MAX);

  const iree_host_size_t minimum_capacity =
      out_requirements->read_only_data_count + 1;
  const bool grow_records = minimum_capacity > builder->record_capacity;
  const bool grow_slots =
      minimum_capacity > 4 && minimum_capacity > builder->slot_capacity / 2;
  loom_low_read_only_data_builder_t candidate = *builder;
  if (grow_records || grow_slots) {
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(arena);
    iree_status_t status = iree_ok_status();
    if (grow_records) {
      status = iree_arena_grow_array(
          arena, out_requirements->read_only_data_count,
          iree_max(minimum_capacity, 4u), sizeof(*candidate.records),
          &candidate.record_capacity, (void**)&candidate.records);
    }
    if (iree_status_is_ok(status) && grow_slots) {
      candidate.slot_capacity =
          candidate.slot_capacity == 0 ? 16 : candidate.slot_capacity * 2;
      status = iree_arena_allocate_array(arena, candidate.slot_capacity,
                                         sizeof(*candidate.ordinal_slots),
                                         (void**)&candidate.ordinal_slots);
      if (iree_status_is_ok(status)) {
        memset(candidate.ordinal_slots, 0,
               candidate.slot_capacity * sizeof(*candidate.ordinal_slots));
        for (uint16_t ordinal = 0;
             ordinal < out_requirements->read_only_data_count; ++ordinal) {
          loom_low_read_only_data_insert_slot(candidate.records, ordinal,
                                              candidate.ordinal_slots,
                                              candidate.slot_capacity);
        }
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_arena_checkpoint_restore(&checkpoint);
      return status;
    }
  }
  const uint16_t ordinal = (uint16_t)out_requirements->read_only_data_count;
  candidate.records[ordinal] = (loom_low_read_only_data_requirement_t){
      .symbol = symbol,
      .definition = definition,
      .contents = loom_global_rodata_def_contents(definition),
      .minimum_alignment =
          loom_global_rodata_def_has_alignment(definition)
              ? (uint64_t)loom_global_rodata_def_alignment(definition)
              : 1,
      .bank_conflicts = loom_global_rodata_def_bank_conflicts(definition),
  };
  if (candidate.slot_capacity != 0) {
    loom_low_read_only_data_insert_slot(candidate.records, ordinal,
                                        candidate.ordinal_slots,
                                        candidate.slot_capacity);
  }
  *builder = candidate;
  ++out_requirements->read_only_data_count;
  return iree_ok_status();
}

static iree_status_t loom_low_function_requirements_collect_read_only_data(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_low_read_only_data_builder_t* builder,
    loom_low_function_requirements_t* out_requirements) {
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(descriptor_set, op, &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE ||
      !iree_any_bit_set(packet.descriptor->flags,
                        LOOM_LOW_DESCRIPTOR_FLAG_READ_ONLY_DATA_IMMEDIATES)) {
    return iree_ok_status();
  }
  const loom_named_attr_slice_t attrs =
      loom_low_descriptor_packet_attrs(op, packet.kind);
  const uint32_t presence = loom_low_bind_immediate_presence(
      module, descriptor_set, packet.descriptor, attrs);
  const loom_low_descriptor_layout_t* layout =
      loom_low_descriptor_layout(descriptor_set, packet.descriptor);
  for (uint16_t i = 0; i < layout->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[layout->immediate_start + (uint32_t)i];
    if (!iree_any_bit_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_READ_ONLY_DATA)) {
      continue;
    }
    const loom_attribute_t value =
        loom_low_bound_immediate_attr(attrs, presence, immediate);
    if (value.kind == LOOM_ATTR_SYMBOL) {
      IREE_RETURN_IF_ERROR(loom_low_function_requirements_append_read_only_data(
          module, loom_attr_as_symbol(value), arena, builder,
          out_requirements));
    }
  }
  return iree_ok_status();
}

uint32_t loom_low_function_requirements_read_only_data_ordinal(
    const loom_low_function_requirements_t* requirements,
    loom_symbol_ref_t symbol) {
  return loom_low_read_only_data_lookup(
      requirements->read_only_data, requirements->read_only_data_count,
      requirements->read_only_data_ordinal_slots,
      requirements->read_only_data_ordinal_slot_count, symbol);
}

static iree_status_t loom_low_function_requirements_append_direct_callee(
    loom_symbol_ref_t callee, iree_arena_allocator_t* arena,
    loom_symbol_ref_t** inout_callees, iree_host_size_t* inout_capacity,
    loom_low_function_requirements_t* out_requirements) {
  for (iree_host_size_t i = 0; i < out_requirements->direct_callees.count;
       ++i) {
    const loom_symbol_ref_t existing = (*inout_callees)[i];
    if (existing.module_id == callee.module_id &&
        existing.symbol_id == callee.symbol_id) {
      return iree_ok_status();
    }
  }
  const iree_host_size_t minimum_capacity =
      out_requirements->direct_callees.count + 1;
  if (minimum_capacity > *inout_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, out_requirements->direct_callees.count,
        iree_max(minimum_capacity, 4u), sizeof(**inout_callees), inout_capacity,
        (void**)inout_callees));
  }
  (*inout_callees)[out_requirements->direct_callees.count++] = callee;
  return iree_ok_status();
}

iree_status_t loom_low_function_requirements_build(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_low_function_requirements_t* out_requirements) {
  *out_requirements = (loom_low_function_requirements_t){0};
  const loom_op_t** resources = NULL;
  iree_host_size_t resource_capacity = 0;
  loom_low_read_only_data_builder_t read_only_data_builder = {0};
  loom_symbol_ref_t* direct_callees = NULL;
  iree_host_size_t direct_callee_capacity = 0;
  loom_low_storage_layout_builder_t storage_builder;
  loom_low_storage_layout_builder_initialize(&storage_builder);
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  bool entry_traffic_prefix_open = true;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    out_requirements->node_count += block->op_count;
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (block == entry_block && entry_traffic_prefix_open &&
          !loom_low_storage_reserve_isa(op)) {
        entry_traffic_prefix_open = loom_low_function_entry_traffic_op(op);
        if (entry_traffic_prefix_open) {
          out_requirements->entry_traffic_end = op;
          if (loom_low_function_entry_preamble_op(op)) {
            out_requirements->entry_preamble_end = op;
          }
        }
      }
      switch (op->kind) {
        case LOOM_OP_LOW_RESOURCE: {
          const iree_host_size_t minimum_capacity =
              out_requirements->resource_count + 1;
          if (minimum_capacity > resource_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, out_requirements->resource_count,
                iree_max(minimum_capacity, 4u), sizeof(*resources),
                &resource_capacity, (void**)&resources));
          }
          resources[out_requirements->resource_count++] = op;
          break;
        }
        case LOOM_OP_LOW_LIVE_IN:
          break;
        case LOOM_OP_LOW_STORAGE_RESERVE:
        case LOOM_OP_LOW_STORAGE_VIEW: {
          IREE_RETURN_IF_ERROR(loom_low_storage_layout_builder_append(
              module, op, arena, &storage_builder));
          break;
        }
        case LOOM_OP_LOW_RETURN:
          ++out_requirements->return_count;
          break;
        case LOOM_OP_LOW_STORAGE_ADDRESS:
          ++out_requirements->storage_address_count;
          break;
        case LOOM_OP_LOW_FUNC_CALL: {
          IREE_RETURN_IF_ERROR(
              loom_low_function_requirements_append_direct_callee(
                  loom_low_func_call_callee(op), arena, &direct_callees,
                  &direct_callee_capacity, out_requirements));
          break;
        }
        default:
          break;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_function_requirements_collect_read_only_data(
              module, descriptor_set, op, arena, &read_only_data_builder,
              out_requirements));
    }
  }
  out_requirements->resources = resources;
  out_requirements->read_only_data = read_only_data_builder.records;
  out_requirements->direct_callees.values = direct_callees;
  out_requirements->read_only_data_ordinal_slots =
      read_only_data_builder.ordinal_slots;
  out_requirements->read_only_data_ordinal_slot_count =
      read_only_data_builder.slot_capacity;
  return loom_low_storage_layout_builder_finish(
      &storage_builder, arena, &out_requirements->storage_layout);
}
