// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/vm/sync.h"
#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/prefill_group.h"

struct loom_serve_text_resize_t {
  // Allocator for this prepared source result.
  iree_allocator_t host_allocator;
  // New resident directory capacity, independent of dispatch geometry.
  iree_host_size_t row_count;
  // Owned replacement source policy state, moved into the model at submission.
  iree_vm_buffer_t* control_state;
  // Validated source storage records; preparation acquires no device memory.
  loom_serve_text_storage_t storage;
  // Group plan exported from control_state, published with it at submission.
  loom_runtime_execution_plan_t* group_plan;
};

void loom_serve_text_resize_free(loom_serve_text_resize_t* resize) {
  if (!resize) {
    return;
  }
  iree_vm_buffer_release(resize->control_state);
  loom_serve_text_storage_deinitialize(&resize->storage);
  loom_runtime_execution_plan_release(resize->group_plan);
  iree_allocator_free(resize->host_allocator, resize);
}

// Only resident bindings and origins scale here. Epoch packet/feedback buffers
// stay bounded by the compiled cohort, not the number of retained histories.
static iree_status_t loom_serve_text_resize_host_buffer(
    loom_serve_text_model_t* model, iree_host_size_t slot,
    iree_host_size_t length, const void* initial) {
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(length, iree_alignof(int64_t),
                                             model->host_allocator, &buffer));
  iree_byte_span_t bytes = iree_byte_span_empty();
  iree_status_t status = iree_vm_buffer_map_write(buffer, 0, length, &bytes);
  if (iree_status_is_ok(status)) {
    if (length && initial) {
      memcpy(bytes.data, initial, length);
    }
    iree_vm_buffer_release(model->control.host.buffers[slot]);
    model->control.host.buffers[slot] = buffer;
    model->control.host.bytes[slot] = bytes;
  } else {
    iree_vm_buffer_release(buffer);
  }
  return status;
}

iree_status_t loom_serve_text_model_prepare_resize(
    loom_serve_text_model_t* model, iree_host_size_t row_count,
    loom_serve_text_resize_t** out_resize) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_resize = NULL;
  if (row_count <= model->row_count || row_count > INT32_MAX) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "resident growth requires a larger i32 capacity"));
  }
  if (model->state.prepared_reservations) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  loom_serve_text_resize_t* resize = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(model->host_allocator, sizeof(*resize),
                                (void**)&resize));
  resize->host_allocator = model->host_allocator;
  resize->row_count = row_count;
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->source.vm_types,
                                               model->source.control_state),
      iree_vm_variant_from_i64((int64_t)row_count)};
  iree_vm_variant_t results[1 + LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT] = {0};
  iree_status_t status = iree_vm_invoke(
      loom_runtime_program_invocation(model->control.program),
      model->control.resize_state, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->source.vm_types, &results[0], &resize->control_state);
  }
  if (iree_status_is_ok(status) && !resize->control_state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "resized source control state must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_storage_initialize(
        &model->source.vm_types, results + 1, row_count,
        model->context_capacity, model->state.cache.capacity,
        model->host_allocator, &resize->storage);
  }
  if (iree_status_is_ok(status) && model->two_bank) {
    const uint32_t slots[] = {0, 2, 4, 5};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
      const uint8_t* record =
          resize->storage.bytes[LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS].data +
          slots[i] * 24;
      if (iree_unaligned_load_le_u64(record) !=
          iree_hal_buffer_byte_length(model->banks[0].roots[i])) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "group resize must preserve bank packet geometry");
        break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    // The resized control state may describe different group leaves.
    status = loom_serve_text_prefill_group_prepare_plan(
        model, resize->control_state, &resize->group_plan);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_resize = resize;
  } else {
    loom_serve_text_resize_free(resize);
  }
  IREE_RETURN_AND_END_ZONE(z0, status);
}

iree_status_t loom_serve_text_model_submit_resize(
    loom_serve_text_model_t* model, loom_serve_text_resize_t* resize,
    loom_runtime_execution_frontier_t* out_frontier) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_frontier = (loom_runtime_execution_frontier_t){0};
  const iree_host_size_t row_count = resize->row_count;
  iree_status_t status =
      loom_serve_text_state_resize(&model->state, row_count, &resize->storage);
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_release(model->source.control_state);
    model->source.control_state = resize->control_state;
    resize->control_state = NULL;
    if (resize->group_plan) {
      loom_runtime_execution_plan_release(model->execution_plan);
      model->execution_plan = resize->group_plan;
      resize->group_plan = NULL;
    }
    status = loom_serve_text_resize_host_buffer(
        model, TEXT_HOST_STATE_BINDINGS, row_count * 4 * sizeof(int64_t), NULL);
  }
  if (iree_status_is_ok(status)) {
    const iree_const_byte_span_t origins =
        model->state.storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS];
    status = loom_serve_text_resize_host_buffer(
        model, TEXT_HOST_STATE_ORIGINS, origins.data_length, origins.data);
  }
  loom_serve_text_row_t** rows = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(model->host_allocator, row_count,
                                         sizeof(*rows), (void**)&rows);
  }
  if (iree_status_is_ok(status)) {
    memcpy(rows, model->rows, model->row_count * sizeof(*rows));
    iree_allocator_free(model->host_allocator, model->rows);
    model->rows = rows;
  }
  for (iree_host_size_t i = model->row_count;
       i < row_count && iree_status_is_ok(status); ++i) {
    status = iree_arena_allocate(&model->row_records, sizeof(**model->rows),
                                 (void**)&model->rows[i]);
    if (iree_status_is_ok(status)) {
      *model->rows[i] = (loom_serve_text_row_t){.model = model,
                                                .state = model->state.rows[i]};
      ++model->row_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_frontier = loom_runtime_execution_frontier(model->execution);
  }
  IREE_RETURN_AND_END_ZONE(z0, status);
}

void loom_serve_text_model_complete_resize(loom_serve_text_model_t* model) {
  loom_serve_text_state_complete_resize(&model->state);
}

iree_host_size_t loom_serve_text_model_row_count(
    const loom_serve_text_model_t* model) {
  return model->row_count;
}
