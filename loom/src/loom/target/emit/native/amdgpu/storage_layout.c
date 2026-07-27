// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/storage_layout.h"

typedef struct loom_amdgpu_storage_layout_projection_t {
  // Fixed segment sizes accumulated so far.
  loom_amdgpu_storage_layout_segment_sizes_t sizes;
  // Optional destination records to populate.
  loom_amdgpu_storage_layout_record_t* records;
  // Maximum number of records in |records|.
  iree_host_size_t record_capacity;
  // Number of storage records counted or populated so far.
  iree_host_size_t record_count;
  // Optional storage value to resolve while scanning.
  loom_value_id_t resolve_storage_value_id;
  // Optional resolved reservation destination.
  loom_amdgpu_storage_layout_reservation_t* resolved_reservation;
  // Whether |resolved_reservation| has been populated.
  bool resolved;
} loom_amdgpu_storage_layout_projection_t;

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
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU storage layout found an unknown storage space");
  }
}

static iree_status_t loom_amdgpu_storage_layout_project_reservation(
    void* user_data, loom_value_id_t storage_value_id,
    const loom_low_storage_layout_reservation_t* low_reservation) {
  loom_amdgpu_storage_layout_projection_t* projection =
      (loom_amdgpu_storage_layout_projection_t*)user_data;
  uint64_t* segment_size = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_segment_size_ptr(
      &projection->sizes, low_reservation->space, &segment_size));

  if (!iree_is_power_of_two_uint64(low_reservation->byte_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU storage layout alignment must be a power of two");
  }
  uint64_t aligned_segment_size = 0;
  if (!iree_checked_align_u64(*segment_size, low_reservation->byte_alignment,
                              &aligned_segment_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU storage layout alignment overflows");
  }
  uint64_t next_segment_size = 0;
  if (!iree_checked_add_u64(aligned_segment_size, low_reservation->byte_size,
                            &next_segment_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU storage layout fixed segment size overflows");
  }

  const loom_amdgpu_storage_layout_reservation_t reservation = {
      .space = low_reservation->space,
      .byte_offset = aligned_segment_size,
      .byte_size = low_reservation->byte_size,
      .byte_alignment = low_reservation->byte_alignment,
  };
  if (projection->records != NULL) {
    if (projection->record_count >= projection->record_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU storage layout record count changed while building");
    }
    projection->records[projection->record_count] =
        (loom_amdgpu_storage_layout_record_t){
            .storage_value_id = storage_value_id,
            .reservation = reservation,
        };
  }
  if (projection->resolved_reservation != NULL &&
      projection->resolve_storage_value_id == storage_value_id) {
    *projection->resolved_reservation = reservation;
    projection->resolved = true;
  }
  ++projection->record_count;
  *segment_size = next_segment_size;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_layout_project(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_record_t* records,
    iree_host_size_t record_capacity, loom_value_id_t resolve_storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* resolved_reservation,
    loom_amdgpu_storage_layout_projection_t* out_projection) {
  *out_projection = (loom_amdgpu_storage_layout_projection_t){
      .records = records,
      .record_capacity = record_capacity,
      .resolve_storage_value_id = resolve_storage_value_id,
      .resolved_reservation = resolved_reservation,
  };
  return loom_low_storage_layout_visit_reservations(
      module, function_op, loom_amdgpu_storage_layout_project_reservation,
      out_projection);
}

iree_status_t loom_amdgpu_storage_layout_collect_segment_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_segment_sizes_t* out_sizes) {
  *out_sizes = (loom_amdgpu_storage_layout_segment_sizes_t){0};
  loom_amdgpu_storage_layout_projection_t projection;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project(
      module, function_op, NULL, 0, LOOM_VALUE_ID_INVALID, NULL, &projection));
  *out_sizes = projection.sizes;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_amdgpu_storage_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_storage_layout_t){0};
  loom_amdgpu_storage_layout_projection_t count_projection;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project(
      module, function_op, NULL, 0, LOOM_VALUE_ID_INVALID, NULL,
      &count_projection));
  loom_amdgpu_storage_layout_record_t* records = NULL;
  if (count_projection.record_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, count_projection.record_count,
                                  sizeof(*records), (void**)&records));
  }
  loom_amdgpu_storage_layout_projection_t build_projection;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project(
      module, function_op, records, count_projection.record_count,
      LOOM_VALUE_ID_INVALID, NULL, &build_projection));
  if (build_projection.record_count != count_projection.record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU storage layout record count changed while building");
  }
  *out_layout = (loom_amdgpu_storage_layout_t){
      .segment_sizes = build_projection.sizes,
      .records = records,
      .record_count = build_projection.record_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_lookup(
    const loom_amdgpu_storage_layout_t* layout,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation) {
  const loom_low_storage_layout_t low_layout = {
      .records = layout->records,
      .record_count = layout->record_count,
  };
  return loom_low_storage_layout_lookup_reservation(
      &low_layout, storage_value_id, out_reservation);
}

iree_status_t loom_amdgpu_storage_layout_resolve(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation) {
  *out_reservation = (loom_amdgpu_storage_layout_reservation_t){0};
  loom_amdgpu_storage_layout_projection_t projection;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project(
      module, function_op, NULL, 0, storage_value_id, out_reservation,
      &projection));
  if (!projection.resolved) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU storage layout could not resolve referenced storage value");
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_lookup_reference(
    const loom_amdgpu_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  const loom_low_storage_layout_t low_layout = {
      .records = layout->records,
      .record_count = layout->record_count,
  };
  return loom_low_storage_layout_lookup_reference(
      &low_layout, module, storage_value_id, out_reference);
}

typedef struct loom_amdgpu_storage_layout_resolve_context_t {
  // Module containing the low function.
  const loom_module_t* module;
  // Low function whose reserves define the AMDGPU fixed-segment layout.
  const loom_op_t* function_op;
} loom_amdgpu_storage_layout_resolve_context_t;

static iree_status_t loom_amdgpu_storage_layout_resolve_callback(
    void* user_data, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation) {
  const loom_amdgpu_storage_layout_resolve_context_t* context =
      (const loom_amdgpu_storage_layout_resolve_context_t*)user_data;
  return loom_amdgpu_storage_layout_resolve(
      context->module, context->function_op, storage_value_id, out_reservation);
}

iree_status_t loom_amdgpu_storage_layout_resolve_reference(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  const loom_amdgpu_storage_layout_resolve_context_t context = {
      .module = module,
      .function_op = function_op,
  };
  return loom_low_storage_layout_resolve_reference_from_reservations(
      module, storage_value_id, loom_amdgpu_storage_layout_resolve_callback,
      (void*)&context, out_reference);
}
