// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/storage_layout.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"

bool loom_amdgpu_storage_layout_try_project_reservation(
    const loom_low_storage_layout_reservation_t* source_reservation,
    loom_amdgpu_storage_layout_segment_sizes_t* sizes,
    loom_amdgpu_storage_layout_reservation_t* out_reservation) {
  uint64_t* segment_size =
      source_reservation->space == LOOM_STORAGE_SPACE_WORKGROUP
          ? &sizes->group_segment_fixed_size
          : &sizes->private_segment_fixed_size;

  uint64_t aligned_segment_size = 0;
  uint64_t next_segment_size = 0;
  if (!iree_checked_align_u64(*segment_size, source_reservation->byte_alignment,
                              &aligned_segment_size) ||
      !iree_checked_add_u64(aligned_segment_size, source_reservation->byte_size,
                            &next_segment_size)) {
    return false;
  }

  *out_reservation = (loom_amdgpu_storage_layout_reservation_t){
      .space = source_reservation->space,
      .byte_offset = aligned_segment_size,
      .byte_size = source_reservation->byte_size,
      .byte_alignment = source_reservation->byte_alignment,
  };
  *segment_size = next_segment_size;
  return true;
}

iree_status_t loom_amdgpu_storage_layout_emit_overflow(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t value_id, loom_storage_space_t space,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
      loom_param_string(space == LOOM_STORAGE_SPACE_WORKGROUP
                            ? IREE_SV("AMDGPU workgroup")
                            : IREE_SV("AMDGPU private/scratch")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = loom_low_diagnostic_value_origin_op(module, value_id, function_op),
      .error = LOOM_ERR_TARGET_096,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_amdgpu_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_storage_layout_t* source_layout,
    loom_amdgpu_storage_layout_segment_sizes_t segment_prefix,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    bool* out_accepted, loom_amdgpu_storage_layout_t* out_layout) {
  *out_accepted = true;
  *out_layout = (loom_amdgpu_storage_layout_t){0};
  loom_amdgpu_storage_layout_record_t* records = NULL;
  if (source_layout->record_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, source_layout->record_count,
                                  sizeof(*records), (void**)&records));
  }
  loom_amdgpu_storage_layout_segment_sizes_t sizes = segment_prefix;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < source_layout->record_count &&
                               *out_accepted && iree_status_is_ok(status);
       ++i) {
    const loom_low_storage_layout_record_t* source_record =
        &source_layout->records[i];
    records[i].storage_value_id = source_record->storage_value_id;
    if (!loom_amdgpu_storage_layout_try_project_reservation(
            &source_record->reservation, &sizes, &records[i].reservation)) {
      *out_accepted = false;
      status = loom_amdgpu_storage_layout_emit_overflow(
          module, function_op, source_record->storage_value_id,
          source_record->reservation.space, emitter);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (!*out_accepted) {
    return iree_ok_status();
  }
  *out_layout = (loom_amdgpu_storage_layout_t){
      .segment_sizes = sizes,
      .records = records,
      .record_count = source_layout->record_count,
      .index = source_layout->index,
  };
  return iree_ok_status();
}

void loom_amdgpu_storage_layout_lookup_reference(
    const loom_amdgpu_storage_layout_t* layout,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  loom_low_storage_layout_lookup_reference(&layout->index, layout->records,
                                           storage_value_id, out_reference);
}
