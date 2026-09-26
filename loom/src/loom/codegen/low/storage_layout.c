// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_layout.h"

#include <string.h>

#include "loom/analysis/storage_layout.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/rewriter.h"

iree_host_size_t loom_low_storage_space_set_names(
    loom_low_storage_space_set_t set, iree_host_size_t capacity,
    iree_string_view_t* out_names) {
  static const loom_storage_space_t kStorageSpaceOrder[] = {
      LOOM_STORAGE_SPACE_STACK,
      LOOM_STORAGE_SPACE_SCRATCH,
      LOOM_STORAGE_SPACE_PRIVATE,
      LOOM_STORAGE_SPACE_WORKGROUP,
  };
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kStorageSpaceOrder); ++i) {
    const loom_storage_space_t storage_space = kStorageSpaceOrder[i];
    if (!loom_low_storage_space_set_contains(set, storage_space)) {
      continue;
    }
    if (count < capacity) {
      out_names[count] = loom_low_storage_type_space_name(storage_space);
    }
    ++count;
  }
  return count;
}

iree_status_t loom_low_storage_layout_hoist_reservations(
    loom_module_t* module, loom_region_t* body, iree_arena_allocator_t* arena) {
  loom_op_t* insertion_op = loom_region_entry_block(body)->first_op;
  // ABI imports must remain the entry preamble even when reservations move
  // out of later blocks. Their relative order does not affect storage packing.
  while (loom_low_live_in_isa(insertion_op) ||
         loom_low_resource_isa(insertion_op)) {
    insertion_op = insertion_op->next_op;
  }
  loom_rewriter_t rewriter = {0};
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < body->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_op_t* op = body->blocks[block_index]->first_op;
    while (op != NULL && iree_status_is_ok(status)) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_storage_reserve_isa(op)) {
        if (op == insertion_op) {
          insertion_op = next_op;
        } else {
          if (rewriter.module == NULL) {
            loom_rewriter_initialize(&rewriter, module, arena);
          }
          status = loom_rewriter_move_before(&rewriter, op, insertion_op);
        }
      }
      op = next_op;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static uint64_t* loom_low_storage_layout_space_size(
    loom_low_storage_layout_space_sizes_t* sizes, loom_storage_space_t space) {
  switch (space) {
    case LOOM_STORAGE_SPACE_STACK:
      return &sizes->stack_bytes;
    case LOOM_STORAGE_SPACE_SCRATCH:
      return &sizes->scratch_bytes;
    case LOOM_STORAGE_SPACE_PRIVATE:
      return &sizes->private_bytes;
    case LOOM_STORAGE_SPACE_WORKGROUP:
      return &sizes->workgroup_bytes;
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified storage reservation must have a valid storage space");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_storage_layout_pack_reservation(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_space_sizes_t* sizes,
    loom_low_storage_layout_reservation_t* out_reservation) {
  const loom_value_id_t storage_value_id =
      loom_low_storage_reserve_storage(reserve_op);
  const loom_type_t storage_type =
      loom_module_value_type(module, storage_value_id);
  const loom_storage_space_t storage_space =
      loom_type_storage_space(storage_type);
  uint64_t* space_size =
      loom_low_storage_layout_space_size(sizes, storage_space);
  const uint64_t byte_size =
      (uint64_t)loom_low_storage_reserve_byte_length(reserve_op);
  const uint64_t byte_alignment =
      (uint64_t)loom_low_storage_reserve_byte_alignment(reserve_op);
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_storage_layout_append(byte_size, byte_alignment,
                                                  space_size, &byte_offset));
  *out_reservation = (loom_low_storage_layout_reservation_t){
      .space = storage_space,
      .byte_offset = byte_offset,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
  };
  return iree_ok_status();
}

void loom_low_storage_layout_builder_initialize(
    const loom_low_storage_layout_t* workgroup_layout,
    loom_low_storage_layout_builder_t* out_builder) {
  *out_builder = (loom_low_storage_layout_builder_t){
      .workgroup_layout = workgroup_layout,
      .workgroup_tail_alignment =
          workgroup_layout ? workgroup_layout->workgroup_tail_alignment : 0,
  };
  if (workgroup_layout != NULL) {
    out_builder->space_sizes.workgroup_bytes =
        workgroup_layout->space_sizes.workgroup_bytes;
  }
}

loom_low_storage_layout_requirement_t loom_low_storage_layout_requirement(
    const loom_low_storage_layout_t* layout, loom_storage_space_t space) {
  loom_low_storage_layout_requirement_t requirement = {0};
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_low_storage_layout_reservation_t* reservation =
        &layout->records[i].reservation;
    if (reservation->space != space) {
      continue;
    }
    requirement.byte_length = reservation->byte_offset + reservation->byte_size;
    requirement.minimum_alignment =
        iree_max(requirement.minimum_alignment, reservation->byte_alignment);
  }
  if (space == LOOM_STORAGE_SPACE_WORKGROUP) {
    requirement.byte_length = layout->space_sizes.workgroup_bytes;
    requirement.minimum_alignment = iree_max(requirement.minimum_alignment,
                                             layout->workgroup_tail_alignment);
  }
  if (requirement.byte_length == 0 &&
      !(space == LOOM_STORAGE_SPACE_WORKGROUP &&
        layout->workgroup_tail_alignment != 0)) {
    requirement.minimum_alignment = 0;
  }
  return requirement;
}

iree_status_t loom_low_storage_layout_builder_append(
    const loom_module_t* module, const loom_op_t* reserve_op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_builder_t* builder) {
  loom_low_storage_layout_reservation_t reservation;
  const bool is_workgroup =
      loom_type_storage_space(loom_module_value_type(
          module, loom_low_storage_reserve_storage(reserve_op))) ==
      LOOM_STORAGE_SPACE_WORKGROUP;
  if (is_workgroup && builder->workgroup_layout != NULL) {
    const loom_low_storage_layout_record_t* records =
        builder->workgroup_layout->records;
    while (records[builder->workgroup_record_cursor].reservation.space !=
           LOOM_STORAGE_SPACE_WORKGROUP) {
      ++builder->workgroup_record_cursor;
    }
    reservation = records[builder->workgroup_record_cursor++].reservation;
  } else {
    IREE_RETURN_IF_ERROR(loom_low_storage_layout_pack_reservation(
        module, reserve_op, &builder->space_sizes, &reservation));
  }
  builder->workgroup_record_count += is_workgroup;
  const iree_host_size_t minimum_capacity = builder->record_count + 1;
  if (minimum_capacity > builder->record_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, builder->record_count, iree_max(minimum_capacity, 4u),
        sizeof(*builder->records), &builder->record_capacity,
        (void**)&builder->records));
  }
  builder->records[builder->record_count++] =
      (loom_low_storage_layout_record_t){
          .storage_value_id = loom_low_storage_reserve_storage(reserve_op),
          .reservation = reservation,
      };
  return iree_ok_status();
}

void loom_low_storage_layout_builder_require_workgroup_tail(
    uint64_t base_alignment, loom_low_storage_layout_builder_t* builder) {
  builder->workgroup_tail_alignment =
      iree_max(builder->workgroup_tail_alignment, base_alignment);
}

iree_status_t loom_low_storage_layout_builder_finish(
    const loom_low_storage_layout_builder_t* builder,
    loom_low_storage_layout_t* out_layout) {
  *out_layout = (loom_low_storage_layout_t){
      .space_sizes = builder->space_sizes,
      .records = builder->records,
      .record_count = builder->record_count,
      .workgroup_record_count = builder->workgroup_record_count,
      .workgroup_tail_alignment = builder->workgroup_tail_alignment,
  };
  if (builder->workgroup_layout == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_storage_layout_align_workgroup_tail(
        builder->workgroup_tail_alignment, &out_layout->space_sizes));
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_layout_align_workgroup_tail(
    uint64_t base_alignment, loom_low_storage_layout_space_sizes_t* sizes) {
  if (base_alignment != 0 &&
      !iree_checked_align_u64(sizes->workgroup_bytes, base_alignment,
                              &sizes->workgroup_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup tail alignment overflows");
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_layout_project_workgroup(
    const loom_low_storage_layout_t* layout, iree_arena_allocator_t* arena,
    loom_low_storage_layout_t* out_layout) {
  *out_layout = (loom_low_storage_layout_t){
      .space_sizes.workgroup_bytes = layout->space_sizes.workgroup_bytes,
      .workgroup_tail_alignment = layout->workgroup_tail_alignment,
      .workgroup_record_count = layout->workgroup_record_count,
      .record_count = layout->workgroup_record_count,
  };
  if (layout->workgroup_record_count == 0) {
    return iree_ok_status();
  }
  loom_low_storage_layout_record_t* records = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, layout->workgroup_record_count,
                                sizeof(*records), (void**)&records));
  iree_host_size_t ordinal = 0;
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    if (layout->records[i].reservation.space == LOOM_STORAGE_SPACE_WORKGROUP) {
      records[ordinal++] = layout->records[i];
    }
  }
  out_layout->records = records;
  return iree_ok_status();
}

iree_status_t loom_low_workgroup_layouts_initialize(
    iree_host_size_t symbol_count, iree_host_size_t entry_count,
    iree_arena_allocator_t* arena, loom_low_workgroup_layouts_t* out_layouts) {
  *out_layouts = (loom_low_workgroup_layouts_t){.symbol_count = symbol_count};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entry_count, sizeof(*out_layouts->entries),
      (void**)&out_layouts->entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, symbol_count, sizeof(*out_layouts->ordinals),
      (void**)&out_layouts->ordinals));
  memset(out_layouts->ordinals, 0,
         symbol_count * sizeof(*out_layouts->ordinals));
  return iree_ok_status();
}

iree_status_t loom_low_workgroup_layouts_insert(
    loom_symbol_id_t symbol_id, const loom_low_storage_layout_t* layout,
    iree_arena_allocator_t* arena, loom_low_workgroup_layouts_t* layouts) {
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_project_workgroup(
      layout, arena, &layouts->entries[layouts->count]));
  layouts->ordinals[symbol_id] = (uint32_t)++layouts->count;
  return iree_ok_status();
}

const loom_low_storage_layout_t* loom_low_workgroup_layouts_lookup(
    const loom_low_workgroup_layouts_t* layouts, loom_symbol_id_t symbol_id) {
  if (layouts == NULL || symbol_id >= layouts->symbol_count ||
      layouts->ordinals[symbol_id] == 0) {
    return NULL;
  }
  return &layouts->entries[layouts->ordinals[symbol_id] - 1];
}

iree_status_t loom_low_storage_layout_accumulate_reservation(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_space_sizes_t* sizes) {
  loom_low_storage_layout_reservation_t reservation;
  return loom_low_storage_layout_pack_reservation(module, reserve_op, sizes,
                                                  &reservation);
}

void loom_low_storage_layout_lookup_reference(
    const loom_low_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference) {
  uint64_t byte_offset = 0;
  uint64_t byte_length = 0;
  bool has_view = false;
  for (;;) {
    const loom_value_t* storage_value =
        loom_module_value(module, storage_value_id);
    const loom_op_t* defining_op = loom_value_def_op(storage_value);
    if (loom_low_storage_reserve_isa(defining_op)) {
      loom_low_storage_layout_reservation_t reservation;
      loom_low_storage_layout_lookup_reservation(layout, storage_value_id,
                                                 &reservation);
      *out_reference = (loom_low_storage_layout_reference_t){
          .reservation = reservation,
          .byte_offset = byte_offset,
          .byte_length = has_view ? byte_length : reservation.byte_size,
      };
      return;
    }

    IREE_ASSERT(defining_op != NULL && loom_low_storage_view_isa(defining_op),
                "verified storage references must be reserve/view chains");
    if (!has_view) {
      byte_length = (uint64_t)loom_low_storage_view_byte_length(defining_op);
      has_view = true;
    }
    byte_offset += (uint64_t)loom_low_storage_view_offset(defining_op);
    storage_value_id = loom_low_storage_view_source(defining_op);
  }
}

void loom_low_storage_layout_lookup_reservation(
    const loom_low_storage_layout_t* layout, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation) {
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_low_storage_layout_record_t* record = &layout->records[i];
    if (record->storage_value_id != storage_value_id) {
      continue;
    }
    *out_reservation = record->reservation;
    return;
  }
  IREE_ASSERT_UNREACHABLE(
      "storage layout and reference must belong to the same function");
  IREE_BUILTIN_UNREACHABLE();
}
