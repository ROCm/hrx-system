// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_layout.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_registry.h"

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
  uint64_t aligned_space_size = 0;
  if (!iree_checked_align_u64(*space_size, byte_alignment,
                              &aligned_space_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage layout alignment overflows");
  }
  uint64_t next_space_size = 0;
  if (!iree_checked_add_u64(aligned_space_size, byte_size, &next_space_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage layout space size overflows");
  }
  *out_reservation = (loom_low_storage_layout_reservation_t){
      .space = storage_space,
      .byte_offset = aligned_space_size,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
  };
  *space_size = next_space_size;
  return iree_ok_status();
}

void loom_low_storage_layout_builder_initialize(
    loom_low_storage_layout_builder_t* out_builder) {
  *out_builder = (loom_low_storage_layout_builder_t){0};
}

iree_status_t loom_low_storage_layout_builder_append(
    const loom_module_t* module, const loom_op_t* reserve_op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_builder_t* builder) {
  loom_low_storage_layout_reservation_t reservation;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_pack_reservation(
      module, reserve_op, &builder->space_sizes, &reservation));
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

void loom_low_storage_layout_builder_finish(
    const loom_low_storage_layout_builder_t* builder,
    loom_low_storage_layout_t* out_layout) {
  *out_layout = (loom_low_storage_layout_t){
      .space_sizes = builder->space_sizes,
      .records = builder->records,
      .record_count = builder->record_count,
  };
}

iree_status_t loom_low_storage_layout_accumulate_reservation(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_space_sizes_t* sizes) {
  loom_low_storage_layout_reservation_t reservation;
  return loom_low_storage_layout_pack_reservation(module, reserve_op, sizes,
                                                  &reservation);
}

static const loom_low_storage_layout_record_t*
loom_low_storage_layout_lookup_record(const loom_low_storage_layout_t* layout,
                                      loom_value_id_t storage_value_id,
                                      iree_host_size_t* out_ordinal) {
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_low_storage_layout_record_t* record = &layout->records[i];
    if (record->storage_value_id != storage_value_id) continue;
    if (out_ordinal != NULL) *out_ordinal = i;
    return record;
  }
  IREE_ASSERT_UNREACHABLE(
      "storage layout and reference must belong to the same function");
  IREE_BUILTIN_UNREACHABLE();
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
      iree_host_size_t reservation_ordinal = 0;
      const loom_low_storage_layout_record_t* record =
          loom_low_storage_layout_lookup_record(layout, storage_value_id,
                                                &reservation_ordinal);
      *out_reference = (loom_low_storage_layout_reference_t){
          .reservation_ordinal = reservation_ordinal,
          .reservation = record->reservation,
          .byte_offset = byte_offset,
          .byte_length = has_view ? byte_length : record->reservation.byte_size,
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
  *out_reservation =
      loom_low_storage_layout_lookup_record(layout, storage_value_id, NULL)
          ->reservation;
}
