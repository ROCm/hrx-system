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
#include "loom/ops/low/ops.h"

typedef struct loom_low_direct_callee_builder_t {
  loom_low_direct_callee_requirement_t* values;
  iree_host_size_t count;
  iree_host_size_t capacity;
  uint16_t* ordinal_slots;
  iree_host_size_t slot_capacity;
} loom_low_direct_callee_builder_t;

static bool loom_low_symbol_ref_equal(loom_symbol_ref_t lhs,
                                      loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static iree_host_size_t loom_low_symbol_ref_hash(
    loom_symbol_ref_t symbol, iree_host_size_t slot_capacity) {
  const uint32_t identity =
      ((uint32_t)symbol.module_id << 16) | (uint32_t)symbol.symbol_id;
  return (identity * UINT32_C(0x85EBCA77)) & (slot_capacity - 1);
}

static iree_status_t loom_low_function_requirements_collect_read_only_data(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_low_read_only_data_builder_t* builder) {
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
      IREE_RETURN_IF_ERROR(loom_low_read_only_data_builder_insert(
          builder, module, loom_attr_as_symbol(value), arena));
    }
  }
  return iree_ok_status();
}

static uint32_t loom_low_direct_callee_lookup(
    const loom_low_direct_callee_builder_t* builder, loom_symbol_ref_t callee) {
  if (builder->slot_capacity == 0) {
    for (uint32_t ordinal = 0; ordinal < (uint32_t)builder->count; ++ordinal) {
      if (loom_low_symbol_ref_equal(builder->values[ordinal].symbol, callee)) {
        return ordinal;
      }
    }
    return UINT32_MAX;
  }
  const iree_host_size_t slot_mask = builder->slot_capacity - 1;
  iree_host_size_t slot_index =
      loom_low_symbol_ref_hash(callee, builder->slot_capacity);
  while (builder->ordinal_slots[slot_index] != 0) {
    const uint32_t ordinal = builder->ordinal_slots[slot_index] - 1;
    IREE_ASSERT_LT(ordinal, builder->count);
    if (loom_low_symbol_ref_equal(builder->values[ordinal].symbol, callee)) {
      return ordinal;
    }
    slot_index = (slot_index + 1) & slot_mask;
  }
  return UINT32_MAX;
}

static void loom_low_direct_callee_insert_slot(
    loom_low_direct_callee_builder_t* builder, uint16_t ordinal) {
  const iree_host_size_t slot_mask = builder->slot_capacity - 1;
  iree_host_size_t slot_index = loom_low_symbol_ref_hash(
      builder->values[ordinal].symbol, builder->slot_capacity);
  while (builder->ordinal_slots[slot_index] != 0) {
    slot_index = (slot_index + 1) & slot_mask;
  }
  builder->ordinal_slots[slot_index] = ordinal + 1;
}

static iree_status_t loom_low_function_requirements_append_direct_callee(
    loom_symbol_ref_t callee, const loom_op_t* first_call_op,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_low_direct_callee_builder_t* builder) {
  if (loom_low_direct_callee_lookup(builder, callee) != UINT32_MAX) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(builder->count, UINT16_MAX);
  const iree_host_size_t minimum_capacity = builder->count + 1;
  const bool grow_values = minimum_capacity > builder->capacity;
  const bool grow_slots =
      minimum_capacity > 4 && minimum_capacity > builder->slot_capacity / 2;
  loom_low_direct_callee_builder_t candidate = *builder;
  if (grow_values || grow_slots) {
    const iree_arena_checkpoint_t arena_checkpoint =
        iree_arena_checkpoint_save(arena);
    const iree_arena_checkpoint_t scratch_checkpoint =
        iree_arena_checkpoint_save(scratch_arena);
    iree_status_t status = iree_ok_status();
    if (grow_values) {
      status = iree_arena_grow_array(
          arena, builder->count, iree_max(minimum_capacity, 4u),
          sizeof(*candidate.values), &candidate.capacity,
          (void**)&candidate.values);
    }
    if (iree_status_is_ok(status) && grow_slots) {
      candidate.slot_capacity =
          candidate.slot_capacity == 0 ? 16 : candidate.slot_capacity * 2;
      status = iree_arena_allocate_array(scratch_arena, candidate.slot_capacity,
                                         sizeof(*candidate.ordinal_slots),
                                         (void**)&candidate.ordinal_slots);
      if (iree_status_is_ok(status)) {
        memset(candidate.ordinal_slots, 0,
               candidate.slot_capacity * sizeof(*candidate.ordinal_slots));
        for (uint16_t ordinal = 0; ordinal < candidate.count; ++ordinal) {
          loom_low_direct_callee_insert_slot(&candidate, ordinal);
        }
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_arena_checkpoint_restore(&scratch_checkpoint);
      iree_arena_checkpoint_restore(&arena_checkpoint);
      return status;
    }
  }
  const uint16_t ordinal = (uint16_t)candidate.count++;
  candidate.values[ordinal] = (loom_low_direct_callee_requirement_t){
      .symbol = callee,
      .first_call_op = first_call_op,
  };
  if (candidate.slot_capacity != 0) {
    loom_low_direct_callee_insert_slot(&candidate, ordinal);
  }
  *builder = candidate;
  return iree_ok_status();
}

static iree_status_t loom_low_function_requirements_build_impl(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_region_t* body,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_low_function_requirements_t* out_requirements) {
  *out_requirements = (loom_low_function_requirements_t){0};
  const loom_op_t** resources = NULL;
  iree_host_size_t resource_capacity = 0;
  loom_low_read_only_data_builder_t read_only_data_builder = {0};
  loom_low_direct_callee_builder_t direct_callee_builder = {0};
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
        case LOOM_OP_LOW_BORROW:
          ++out_requirements->borrow_count;
          break;
        case LOOM_OP_LOW_FUNC_CALL: {
          IREE_RETURN_IF_ERROR(
              loom_low_function_requirements_append_direct_callee(
                  loom_low_func_call_callee(op), op, arena, scratch_arena,
                  &direct_callee_builder));
          break;
        }
        default:
          break;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_function_requirements_collect_read_only_data(
              module, descriptor_set, op, arena, &read_only_data_builder));
    }
  }
  out_requirements->resources = resources;
  out_requirements->read_only_data =
      loom_low_read_only_data_builder_finish(&read_only_data_builder);
  out_requirements->direct_callees =
      (loom_low_direct_callee_requirement_list_t){
          .values = direct_callee_builder.values,
          .count = direct_callee_builder.count,
      };
  return loom_low_storage_layout_builder_finish(
      &storage_builder, arena, &out_requirements->storage_layout);
}

iree_status_t loom_low_function_requirements_build(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_low_function_requirements_t* out_requirements) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_low_function_requirements_build_impl(
      module, descriptor_set, body, arena, &scratch_arena, out_requirements);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
