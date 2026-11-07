// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/read_only_data.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"

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

static uint32_t loom_low_read_only_data_lookup(
    const loom_low_read_only_data_requirement_t* records,
    iree_host_size_t record_count, const uint16_t* ordinal_slots,
    iree_host_size_t slot_capacity, loom_symbol_ref_t symbol) {
  if (!loom_symbol_ref_is_valid(symbol)) {
    return LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID;
  }
  if (slot_capacity == 0) {
    for (uint32_t ordinal = 0; ordinal < (uint32_t)record_count; ++ordinal) {
      const loom_symbol_ref_t existing = records[ordinal].symbol;
      if (loom_low_symbol_ref_equal(existing, symbol)) {
        return ordinal;
      }
    }
    return LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID;
  }
  const iree_host_size_t slot_mask = slot_capacity - 1;
  iree_host_size_t slot_index = loom_low_symbol_ref_hash(symbol, slot_capacity);
  while (ordinal_slots[slot_index] != 0) {
    const uint32_t ordinal = ordinal_slots[slot_index] - 1;
    IREE_ASSERT_LT(ordinal, record_count);
    const loom_symbol_ref_t existing = records[ordinal].symbol;
    if (loom_low_symbol_ref_equal(existing, symbol)) {
      return ordinal;
    }
    slot_index = (slot_index + 1) & slot_mask;
  }
  return LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID;
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

loom_low_read_only_data_set_t loom_low_read_only_data_builder_finish(
    const loom_low_read_only_data_builder_t* builder) {
  return (loom_low_read_only_data_set_t){
      .values = builder->values,
      .count = builder->count,
      .ordinal_slots = builder->ordinal_slots,
      .slot_count = builder->slot_count,
  };
}

uint32_t loom_low_read_only_data_ordinal(
    const loom_low_read_only_data_set_t* data, loom_symbol_ref_t symbol) {
  return loom_low_read_only_data_lookup(
      data->values, data->count, data->ordinal_slots, data->slot_count, symbol);
}

static iree_status_t loom_low_read_only_data_builder_append(
    loom_low_read_only_data_builder_t* builder, iree_arena_allocator_t* arena,
    loom_low_read_only_data_requirement_t** out_record) {
  IREE_ASSERT_LT(builder->count, LOOM_SYMBOL_ID_INVALID);
  const iree_host_size_t minimum_capacity = builder->count + 1;
  const bool grow_records = minimum_capacity > builder->capacity;
  const bool grow_slots =
      minimum_capacity > 4 && minimum_capacity > builder->slot_count / 2;
  loom_low_read_only_data_builder_t candidate = *builder;
  if (grow_records || grow_slots) {
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(arena);
    iree_status_t status = iree_ok_status();
    if (grow_records) {
      status = iree_arena_grow_array(
          arena, builder->count, iree_max(minimum_capacity, 4u),
          sizeof(*candidate.values), &candidate.capacity,
          (void**)&candidate.values);
    }
    if (iree_status_is_ok(status) && grow_slots) {
      candidate.slot_count =
          candidate.slot_count == 0 ? 16 : candidate.slot_count * 2;
      status = iree_arena_allocate_array(arena, candidate.slot_count,
                                         sizeof(*candidate.ordinal_slots),
                                         (void**)&candidate.ordinal_slots);
      if (iree_status_is_ok(status)) {
        memset(candidate.ordinal_slots, 0,
               candidate.slot_count * sizeof(*candidate.ordinal_slots));
        for (uint16_t ordinal = 0; ordinal < builder->count; ++ordinal) {
          loom_low_read_only_data_insert_slot(candidate.values, ordinal,
                                              candidate.ordinal_slots,
                                              candidate.slot_count);
        }
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_arena_checkpoint_restore(&checkpoint);
      return status;
    }
  }
  *builder = candidate;
  *out_record = &builder->values[builder->count];
  return iree_ok_status();
}

static void loom_low_read_only_data_builder_commit(
    loom_low_read_only_data_builder_t* builder) {
  if (builder->slot_count != 0) {
    loom_low_read_only_data_insert_slot(
        builder->values, (uint16_t)builder->count, builder->ordinal_slots,
        builder->slot_count);
  }
  ++builder->count;
}

iree_status_t loom_low_read_only_data_builder_insert(
    loom_low_read_only_data_builder_t* builder, const loom_module_t* module,
    loom_symbol_ref_t symbol, iree_arena_allocator_t* arena) {
  const loom_low_read_only_data_set_t data =
      loom_low_read_only_data_builder_finish(builder);
  if (loom_low_read_only_data_ordinal(&data, symbol) !=
      LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  loom_low_read_only_data_requirement_t* record = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_read_only_data_builder_append(builder, arena, &record));
  *record = (loom_low_read_only_data_requirement_t){
      .symbol = symbol,
      .definition = definition,
      .contents = loom_global_rodata_def_contents(definition),
      .minimum_alignment =
          loom_global_rodata_def_has_alignment(definition)
              ? (uint64_t)loom_global_rodata_def_alignment(definition)
              : 1,
      .bank_conflicts = loom_global_rodata_def_bank_conflicts(definition),
  };
  loom_low_read_only_data_builder_commit(builder);
  return iree_ok_status();
}

iree_status_t loom_low_read_only_data_builder_extend(
    loom_low_read_only_data_builder_t* builder,
    const loom_low_read_only_data_set_t* data, iree_arena_allocator_t* arena) {
  for (iree_host_size_t i = 0; i < data->count; ++i) {
    const loom_low_read_only_data_set_t current =
        loom_low_read_only_data_builder_finish(builder);
    if (loom_low_read_only_data_ordinal(&current, data->values[i].symbol) !=
        LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID) {
      continue;
    }
    loom_low_read_only_data_requirement_t* record = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_read_only_data_builder_append(builder, arena, &record));
    *record = data->values[i];
    loom_low_read_only_data_builder_commit(builder);
  }
  return iree_ok_status();
}
