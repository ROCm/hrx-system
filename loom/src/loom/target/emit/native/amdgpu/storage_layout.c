// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/storage_layout.h"

static iree_status_t loom_amdgpu_storage_layout_segment_size_ptr(
    loom_amdgpu_storage_layout_segment_sizes_t* sizes,
    loom_storage_space_t storage_space, uint64_t** out_segment_size) {
  *out_segment_size = NULL;
  switch (storage_space) {
    case LOOM_STORAGE_SPACE_WORKGROUP:
      *out_segment_size = &sizes->group_segment_fixed_size;
      return iree_ok_status();
    case LOOM_STORAGE_SPACE_PRIVATE:
    case LOOM_STORAGE_SPACE_SCRATCH:
      *out_segment_size = &sizes->private_segment_fixed_size;
      return iree_ok_status();
    case LOOM_STORAGE_SPACE_STACK:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU stack storage must be rejected before native storage layout");
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified storage reservation must have a valid storage space");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_storage_layout_project_reservation(
    const loom_low_storage_layout_record_t* source_record,
    loom_amdgpu_storage_layout_segment_sizes_t* sizes,
    loom_amdgpu_storage_layout_record_t* out_record) {
  const loom_low_storage_layout_reservation_t* source_reservation =
      &source_record->reservation;
  uint64_t* segment_size = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_segment_size_ptr(
      sizes, source_reservation->space, &segment_size));

  uint64_t aligned_segment_size = 0;
  if (!iree_checked_align_u64(*segment_size, source_reservation->byte_alignment,
                              &aligned_segment_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU storage layout alignment overflows");
  }
  uint64_t next_segment_size = 0;
  if (!iree_checked_add_u64(aligned_segment_size, source_reservation->byte_size,
                            &next_segment_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU storage layout fixed segment size overflows");
  }

  *out_record = (loom_amdgpu_storage_layout_record_t){
      .storage_value_id = source_record->storage_value_id,
      .reservation =
          {
              .space = source_reservation->space,
              .byte_offset = aligned_segment_size,
              .byte_size = source_reservation->byte_size,
              .byte_alignment = source_reservation->byte_alignment,
          },
  };
  *segment_size = next_segment_size;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_build(
    const loom_low_storage_layout_t* source_layout,
    iree_arena_allocator_t* arena, loom_amdgpu_storage_layout_t* out_layout) {
  loom_amdgpu_storage_layout_record_t* records = NULL;
  if (source_layout->record_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, source_layout->record_count,
                                  sizeof(*records), (void**)&records));
  }
  loom_amdgpu_storage_layout_segment_sizes_t sizes = {0};
  for (iree_host_size_t i = 0; i < source_layout->record_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project_reservation(
        &source_layout->records[i], &sizes, &records[i]));
  }
  *out_layout = (loom_amdgpu_storage_layout_t){
      .segment_sizes = sizes,
      .records = records,
      .record_count = source_layout->record_count,
  };
  return iree_ok_status();
}

void loom_amdgpu_storage_layout_lookup_reference(
    const loom_amdgpu_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  const loom_low_storage_layout_t low_layout = {
      .records = layout->records,
      .record_count = layout->record_count,
  };
  loom_low_storage_layout_lookup_reference(&low_layout, module,
                                           storage_value_id, out_reference);
}
