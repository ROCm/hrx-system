// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_layout.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_storage_layout_scan_state_t {
  // Packed byte sizes accumulated for each storage space.
  loom_low_storage_layout_space_sizes_t space_sizes;
  // Optional destination records to populate.
  loom_low_storage_layout_record_t* records;
  // Maximum number of records in |records|.
  iree_host_size_t record_capacity;
  // Number of storage records counted or populated so far.
  iree_host_size_t record_count;
  // Optional storage value to resolve while scanning.
  loom_value_id_t resolve_storage_value_id;
  // Optional resolved reservation destination.
  loom_low_storage_layout_reservation_t* resolved_reservation;
  // Whether |resolved_reservation| has been populated.
  bool resolved;
  // Optional callback invoked once for each visited reservation.
  loom_low_storage_layout_reservation_callback_t callback;
  // User data passed to |callback|.
  void* callback_user_data;
} loom_low_storage_layout_scan_state_t;

static bool loom_low_storage_layout_u64_is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_low_storage_layout_checked_add_u64(
    uint64_t lhs, uint64_t rhs, uint64_t* out_result) {
  *out_result = 0;
  if (lhs > UINT64_MAX - rhs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage layout offset overflows");
  }
  *out_result = lhs + rhs;
  return iree_ok_status();
}

static iree_status_t loom_low_storage_layout_checked_align_u64(
    uint64_t value, uint64_t alignment, uint64_t* out_aligned_value) {
  *out_aligned_value = 0;
  if (!loom_low_storage_layout_u64_is_power_of_two(alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout alignment must be a power of two");
  }
  const uint64_t alignment_mask = alignment - 1;
  if (value > UINT64_MAX - alignment_mask) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage layout alignment overflows");
  }
  *out_aligned_value = (value + alignment_mask) & ~alignment_mask;
  return iree_ok_status();
}

static iree_status_t loom_low_storage_layout_space_size_ptr(
    loom_low_storage_layout_space_sizes_t* sizes, loom_storage_space_t space,
    uint64_t** out_size) {
  *out_size = NULL;
  switch (space) {
    case LOOM_STORAGE_SPACE_STACK:
      *out_size = &sizes->stack_bytes;
      return iree_ok_status();
    case LOOM_STORAGE_SPACE_SCRATCH:
      *out_size = &sizes->scratch_bytes;
      return iree_ok_status();
    case LOOM_STORAGE_SPACE_PRIVATE:
      *out_size = &sizes->private_bytes;
      return iree_ok_status();
    case LOOM_STORAGE_SPACE_WORKGROUP:
      *out_size = &sizes->workgroup_bytes;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low storage layout found an unknown storage "
                              "space");
  }
}

static iree_status_t loom_low_storage_layout_visit_reserve(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_scan_state_t* state) {
  const int64_t signed_byte_size =
      loom_low_storage_reserve_byte_length(reserve_op);
  const int64_t signed_byte_alignment =
      loom_low_storage_reserve_byte_alignment(reserve_op);
  if (signed_byte_size <= 0 || signed_byte_alignment <= 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout requires positive storage sizes and alignments");
  }

  const loom_value_id_t storage_value_id =
      loom_low_storage_reserve_storage(reserve_op);
  const loom_type_t storage_type =
      loom_module_value_type(module, storage_value_id);
  if (!loom_type_is_storage(storage_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout found a reserve with non-storage result type");
  }
  const loom_storage_space_t storage_space =
      loom_type_storage_space(storage_type);

  uint64_t* space_size = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_space_size_ptr(
      &state->space_sizes, storage_space, &space_size));

  const uint64_t byte_size = (uint64_t)signed_byte_size;
  const uint64_t byte_alignment = (uint64_t)signed_byte_alignment;
  uint64_t aligned_space_size = 0;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_checked_align_u64(
      *space_size, byte_alignment, &aligned_space_size));
  if (aligned_space_size > UINT64_MAX - byte_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage layout space size overflows");
  }

  const loom_low_storage_layout_reservation_t reservation = {
      .space = storage_space,
      .byte_offset = aligned_space_size,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
  };
  if (state->records != NULL) {
    if (state->record_count >= state->record_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low storage layout record count changed while building");
    }
    state->records[state->record_count] = (loom_low_storage_layout_record_t){
        .storage_value_id = storage_value_id,
        .reservation = reservation,
    };
  }
  if (state->resolved_reservation != NULL &&
      state->resolve_storage_value_id == storage_value_id) {
    *state->resolved_reservation = reservation;
    state->resolved = true;
  }
  ++state->record_count;
  *space_size = aligned_space_size + byte_size;
  if (state->callback != NULL) {
    IREE_RETURN_IF_ERROR(state->callback(state->callback_user_data,
                                         storage_value_id, &reservation));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_storage_layout_scan(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_low_storage_layout_record_t* records, iree_host_size_t record_capacity,
    loom_value_id_t resolve_storage_value_id,
    loom_low_storage_layout_reservation_t* resolved_reservation,
    loom_low_storage_layout_reservation_callback_t callback,
    void* callback_user_data, loom_low_storage_layout_scan_state_t* out_state) {
  *out_state = (loom_low_storage_layout_scan_state_t){
      .records = records,
      .record_capacity = record_capacity,
      .resolve_storage_value_id = resolve_storage_value_id,
      .resolved_reservation = resolved_reservation,
      .callback = callback,
      .callback_user_data = callback_user_data,
  };
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low storage layout requires a low function body");
  }
  const loom_block_t* block = NULL;
  const loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_storage_reserve_isa(op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_storage_layout_visit_reserve(module, op, out_state));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_storage_layout_lookup_reservation_callback(
    void* user_data, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation) {
  const loom_low_storage_layout_t* layout =
      (const loom_low_storage_layout_t*)user_data;
  return loom_low_storage_layout_lookup_reservation(layout, storage_value_id,
                                                    out_reservation);
}

typedef struct loom_low_storage_layout_resolve_context_t {
  // Module containing the low function.
  const loom_module_t* module;
  // Low function whose reserves define the storage layout.
  const loom_op_t* function_op;
} loom_low_storage_layout_resolve_context_t;

static iree_status_t loom_low_storage_layout_resolve_reservation_callback(
    void* user_data, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation) {
  const loom_low_storage_layout_resolve_context_t* context =
      (const loom_low_storage_layout_resolve_context_t*)user_data;
  return loom_low_storage_layout_resolve_reservation(
      context->module, context->function_op, storage_value_id, out_reservation);
}

iree_status_t loom_low_storage_layout_resolve_reference_from_reservations(
    const loom_module_t* module, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_lookup_fn_t lookup_reservation,
    void* lookup_user_data,
    loom_low_storage_layout_reference_t* out_reference) {
  *out_reference = (loom_low_storage_layout_reference_t){0};
  if (storage_value_id == LOOM_VALUE_ID_INVALID ||
      storage_value_id >= module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout found an invalid storage value");
  }
  const loom_value_t* storage_value =
      loom_module_value(module, storage_value_id);
  if (loom_value_is_block_arg(storage_value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout requires storage defined by low.storage.reserve or "
        "low.storage.view");
  }
  const loom_op_t* defining_op = loom_value_def_op(storage_value);
  if (defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout found storage without a defining op");
  }

  if (loom_low_storage_reserve_isa(defining_op)) {
    loom_low_storage_layout_reservation_t reservation = {0};
    IREE_RETURN_IF_ERROR(
        lookup_reservation(lookup_user_data, storage_value_id, &reservation));
    *out_reference = (loom_low_storage_layout_reference_t){
        .reservation = reservation,
        .byte_offset = 0,
        .byte_length = reservation.byte_size,
    };
    return iree_ok_status();
  }

  if (!loom_low_storage_view_isa(defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout requires storage defined by low.storage.reserve or "
        "low.storage.view");
  }

  loom_low_storage_layout_reference_t source_reference = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_storage_layout_resolve_reference_from_reservations(
          module, loom_low_storage_view_source(defining_op), lookup_reservation,
          lookup_user_data, &source_reference));
  const int64_t signed_offset = loom_low_storage_view_offset(defining_op);
  const int64_t signed_byte_length =
      loom_low_storage_view_byte_length(defining_op);
  if (signed_offset < 0 || signed_byte_length <= 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage layout requires non-negative storage view offsets and "
        "positive storage view lengths");
  }
  const uint64_t offset = (uint64_t)signed_offset;
  const uint64_t byte_length = (uint64_t)signed_byte_length;
  if (offset > source_reference.byte_length ||
      byte_length > source_reference.byte_length - offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low storage layout view range exceeds its source storage");
  }
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_checked_add_u64(
      source_reference.byte_offset, offset, &byte_offset));
  *out_reference = (loom_low_storage_layout_reference_t){
      .reservation = source_reference.reservation,
      .byte_offset = byte_offset,
      .byte_length = byte_length,
  };
  return iree_ok_status();
}

iree_status_t loom_low_storage_layout_visit_reservations(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_low_storage_layout_reservation_callback_t callback, void* user_data) {
  loom_low_storage_layout_scan_state_t state;
  return loom_low_storage_layout_scan(module, function_op, NULL, 0,
                                      LOOM_VALUE_ID_INVALID, NULL, callback,
                                      user_data, &state);
}

iree_status_t loom_low_storage_layout_collect_space_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_low_storage_layout_space_sizes_t* out_sizes) {
  *out_sizes = (loom_low_storage_layout_space_sizes_t){0};
  loom_low_storage_layout_scan_state_t state;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_scan(module, function_op, NULL,
                                                    0, LOOM_VALUE_ID_INVALID,
                                                    NULL, NULL, NULL, &state));
  *out_sizes = state.space_sizes;
  return iree_ok_status();
}

iree_status_t loom_low_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_t* out_layout) {
  *out_layout = (loom_low_storage_layout_t){0};
  loom_low_storage_layout_scan_state_t count_state;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_scan(
      module, function_op, NULL, 0, LOOM_VALUE_ID_INVALID, NULL, NULL, NULL,
      &count_state));
  loom_low_storage_layout_record_t* records = NULL;
  if (count_state.record_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, count_state.record_count, sizeof(*records), (void**)&records));
  }
  loom_low_storage_layout_scan_state_t build_state;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_scan(
      module, function_op, records, count_state.record_count,
      LOOM_VALUE_ID_INVALID, NULL, NULL, NULL, &build_state));
  if (build_state.record_count != count_state.record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low storage layout record count changed while building");
  }
  *out_layout = (loom_low_storage_layout_t){
      .space_sizes = build_state.space_sizes,
      .records = records,
      .record_count = build_state.record_count,
  };
  return iree_ok_status();
}

iree_status_t loom_low_storage_layout_lookup_reservation(
    const loom_low_storage_layout_t* layout, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation) {
  *out_reservation = (loom_low_storage_layout_reservation_t){0};
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_low_storage_layout_record_t* record = &layout->records[i];
    if (record->storage_value_id != storage_value_id) {
      continue;
    }
    *out_reservation = record->reservation;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "low storage layout could not resolve referenced storage value");
}

iree_status_t loom_low_storage_layout_resolve_reservation(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation) {
  *out_reservation = (loom_low_storage_layout_reservation_t){0};
  loom_low_storage_layout_scan_state_t state;
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_scan(
      module, function_op, NULL, 0, storage_value_id, out_reservation, NULL,
      NULL, &state));
  if (!state.resolved) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "low storage layout could not resolve referenced storage value");
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_layout_lookup_reference(
    const loom_low_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference) {
  return loom_low_storage_layout_resolve_reference_from_reservations(
      module, storage_value_id,
      loom_low_storage_layout_lookup_reservation_callback, (void*)layout,
      out_reference);
}

iree_status_t loom_low_storage_layout_resolve_reference(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference) {
  const loom_low_storage_layout_resolve_context_t context = {
      .module = module,
      .function_op = function_op,
  };
  return loom_low_storage_layout_resolve_reference_from_reservations(
      module, storage_value_id,
      loom_low_storage_layout_resolve_reservation_callback, (void*)&context,
      out_reference);
}
