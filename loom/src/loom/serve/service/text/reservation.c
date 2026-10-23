// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/reservation.h"

#include <string.h>

#include "loom/serve/service/text/model_instance.h"

typedef struct loom_serve_text_reserve_context_t {
  // Borrowed model owning parameters and the selected row's state.
  loom_serve_text_model_t* model;
  // Caller-owned result, distinct from a local logical-ID shortage.
  loom_serve_text_reserve_result_t* result;
} loom_serve_text_reserve_context_t;

static iree_status_t loom_serve_text_admit_state(
    void* user_data, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_memory_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation) {
  loom_serve_text_reserve_context_t* context = user_data;
  loom_serve_residency_admission_t admission;
  const iree_status_t status = loom_serve_residency_try_admit_state(
      context->model->residency, range_count, ranges, readiness, value,
      &admission, out_operation);
  switch (admission) {
    case LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY:
      *context->result = LOOM_SERVE_TEXT_RESERVE_CAPACITY;
      *out_admission = LOOM_SERVE_MEMORY_ADMISSION_CAPACITY;
      break;
    case LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED:
      *context->result = LOOM_SERVE_TEXT_RESERVE_DEFERRED;
      *out_admission = LOOM_SERVE_MEMORY_ADMISSION_DEFERRED;
      break;
    case LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED:
      *context->result = LOOM_SERVE_TEXT_RESERVE_ACQUIRED;
      *out_admission = LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
      break;
  }
  return status;
}

iree_status_t loom_serve_text_reservation_prepare(
    loom_serve_text_row_t* row, loom_serve_text_reserve_mode_t mode,
    loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_text_reservation_t* out_reservation) {
  *out_reservation = (loom_serve_text_reservation_t){0};
  const iree_host_size_t block_size = row->model->state.storage.block_size;
  const iree_host_size_t maximum =
      iree_host_size_ceil_div(row->model->context_capacity, block_size) *
      block_size;
  if ((mode != LOOM_SERVE_TEXT_RESERVE_FRESH &&
       mode != LOOM_SERVE_TEXT_RESERVE_CONTINUE &&
       mode != LOOM_SERVE_TEXT_RESERVE_CHECKPOINT) ||
      ((checkpoint != NULL) != (mode == LOOM_SERVE_TEXT_RESERVE_CHECKPOINT)) ||
      (checkpoint && checkpoint->model != row->model) || !extent ||
      extent > maximum ||
      (checkpoint && extent < checkpoint->state->position) ||
      (mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE &&
       extent < row->state->position)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid text completion reservation");
  }
  if (row->state->reserved_extent ||
      (mode == LOOM_SERVE_TEXT_RESERVE_CONTINUE && row->state->snapshot)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "reservation requires an idle resident frontier");
  }
  loom_serve_text_state_reservation_t* state = NULL;
  // One cold restore installs the endpoint's newly selected physical IDs.
  // Another plan can share it after that restore has completed, not before.
  if (checkpoint && checkpoint->reservations && checkpoint->state->snapshot) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_prepare_reserve(
      row->state, mode, checkpoint ? checkpoint->state : NULL, extent, &state));
  if (state) {
    loom_serve_text_checkpoint_retain(checkpoint);
    if (checkpoint) {
      ++checkpoint->reservations;
    }
    *out_reservation = (loom_serve_text_reservation_t){
        .row = row, .state = state, .checkpoint = checkpoint, .mode = mode};
  }
  return iree_ok_status();
}

uint64_t loom_serve_text_reservation_memory_footprint(
    const loom_serve_text_reservation_t* reservation) {
  const loom_serve_text_model_t* model = reservation->row->model;
  const uint64_t extents[] = {
      loom_serve_residency_activation_footprint(model->residency),
      loom_serve_text_state_reservation_footprint(reservation->state),
      loom_serve_module_workspace_requirement(model->control.native_module)
          .allocation_size,
      2ull * model->source.prefill_capacity *
          LOOM_SERVE_TEXT_MEDIA_PACKET_RECORD_BYTES,
  };
  uint64_t bytes = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(extents); ++i) {
    bytes += iree_min(extents[i], UINT64_MAX - bytes);
  }
  return bytes;
}

iree_status_t loom_serve_text_reservation_submit(
    loom_serve_text_reservation_t* reservation,
    loom_serve_text_reserve_result_t* out_result,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_result = LOOM_SERVE_TEXT_RESERVE_STATE_CAPACITY;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  loom_serve_text_row_t* row = reservation->row;
  const loom_serve_text_checkpoint_t* checkpoint = reservation->checkpoint;
  loom_serve_text_reserve_context_t context = {
      .model = row->model,
      .result = out_result,
  };
  loom_serve_memory_admission_t admission;
  IREE_RETURN_IF_ERROR(loom_serve_text_state_reservation_submit(
      reservation->state, loom_serve_text_admit_state, &context, &admission));
  if (admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE) {
    if (reservation->mode == LOOM_SERVE_TEXT_RESERVE_FRESH) {
      row->has_prediction = false;
      memset(&row->transfer, 0, sizeof(row->transfer));
      memset(&row->metrics, 0, sizeof(row->metrics));
    } else if (checkpoint) {
      row->has_prediction = checkpoint->has_prediction;
      memset(&row->transfer, 0, sizeof(row->transfer));
      row->transfer.token = checkpoint->token;
      row->transfer.progress[0] = checkpoint->has_prediction;
      row->transfer.progress[7] =
          checkpoint->has_prediction &&
          checkpoint->token == row->model->source.eos_token;
    }
  }
  *out_frontier = loom_serve_execution_frontier(row->model->execution);
  return iree_ok_status();
}

void loom_serve_text_reservation_deinitialize(
    loom_serve_text_reservation_t* reservation) {
  loom_serve_text_state_reservation_free(reservation->state);
  if (reservation->checkpoint) {
    --reservation->checkpoint->reservations;
  }
  loom_serve_text_checkpoint_release(reservation->checkpoint);
  *reservation = (loom_serve_text_reservation_t){0};
}
