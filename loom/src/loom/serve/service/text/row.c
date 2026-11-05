// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/reservation.h"

iree_host_size_t loom_serve_text_row_index(const loom_serve_text_row_t* row) {
  return row->state->index;
}

iree_host_size_t loom_serve_text_row_pool_usage(
    const loom_serve_text_row_t* row) {
  return (iree_host_size_t)row->state->block_count *
         row->model->state.storage.block_size;
}

iree_host_size_t loom_serve_text_row_pool_growth(
    const loom_serve_text_row_t* row, iree_host_size_t extent) {
  if (!row->model->state.cache.capacity) {
    return 0;
  }
  return loom_serve_text_state_row_growth(
             row->state, (uint32_t)(extent - row->state->position)) *
         row->model->state.storage.block_size;
}

iree_host_size_t loom_serve_text_row_pool_reclaimable(
    const loom_serve_text_row_t* row) {
  const loom_serve_text_state_t* state = &row->model->state;
  iree_host_size_t count = 0;
  const iree_host_size_t map =
      row->state->index * state->storage.blocks_per_row;
  for (uint32_t i = 0; i < row->state->block_count; ++i) {
    count += !loom_serve_block_pool_is_shared(&state->cache.pool,
                                              state->cache.maps[map + i]);
  }
  return count * state->storage.block_size;
}

void loom_serve_text_row_reset_retired(loom_serve_text_row_t* row) {
  const bool reserved = row->state->reserved_extent != 0;
  loom_serve_text_state_row_reset(row->state);
  if (reserved) {
    loom_serve_text_model_release_parameters(row->model);
  }
  row->has_prediction = false;
  memset(&row->transfer, 0, sizeof(row->transfer));
  memset(&row->metrics, 0, sizeof(row->metrics));
}

iree_status_t loom_serve_text_row_reset(loom_serve_text_row_t* row) {
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain_all(row->model->device_count,
                                                      row->model->executions));
  loom_serve_text_row_reset_retired(row);
  return iree_ok_status();
}

iree_host_size_t loom_serve_text_row_suspended_bytes(
    const loom_serve_text_row_t* row) {
  return loom_serve_snapshot_size(row->state->snapshot);
}

iree_status_t loom_serve_text_row_suspend(loom_serve_text_row_t* row) {
  if (row->state->reserved_extent) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot suspend an active request reservation");
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain_all(row->model->device_count,
                                                      row->model->executions));
  return loom_serve_text_state_row_suspend(row->state);
}

iree_status_t loom_serve_text_row_try_resume(loom_serve_text_row_t* row,
                                             bool* out_resumed) {
  *out_resumed = false;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain_all(row->model->device_count,
                                                      row->model->executions));
  return loom_serve_text_state_row_try_resume(row->state, out_resumed);
}

iree_status_t loom_serve_text_row_submit_pin(
    loom_serve_text_row_t* row, loom_serve_text_checkpoint_t** out_checkpoint,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_checkpoint = NULL;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  if (!row->state->position || row->state->snapshot) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "checkpoint requires a nonempty resident row");
  }
  loom_serve_text_state_checkpoint_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_row_submit_pin(row->state, &state));
  if (!state) {
    return iree_ok_status();
  }
  const iree_host_size_t index =
      (iree_host_size_t)(state - row->model->state.checkpoints.values);
  loom_serve_text_checkpoint_t* checkpoint = &row->model->checkpoints[index];
  *checkpoint =
      (loom_serve_text_checkpoint_t){.model = row->model,
                                     .references = 1,
                                     .state = state,
                                     .has_prediction = row->has_prediction,
                                     .token = row->transfer.token};
  *out_checkpoint = checkpoint;
  *out_frontier = loom_serve_execution_frontier(row->model->executions[0]);
  return iree_ok_status();
}

iree_status_t loom_serve_text_row_try_pin(
    loom_serve_text_row_t* row, loom_serve_text_checkpoint_t** out_checkpoint) {
  *out_checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain_all(row->model->device_count,
                                                      row->model->executions));
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  loom_serve_execution_frontier_t frontier;
  iree_status_t status =
      loom_serve_text_row_submit_pin(row, &checkpoint, &frontier);
  status = iree_status_join(
      status, loom_serve_execution_drain_all(row->model->device_count,
                                             row->model->executions));
  if (iree_status_is_ok(status)) {
    *out_checkpoint = checkpoint;
  }
  return status;
}

iree_status_t loom_serve_text_row_try_restore(
    loom_serve_text_row_t* row, const loom_serve_text_checkpoint_t* checkpoint,
    bool* out_restored) {
  *out_restored = false;
  if (checkpoint->model != row->model) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint belongs to a different model");
  }
  if (row->state->reserved_extent) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot replace an active request reservation");
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain_all(row->model->device_count,
                                                      row->model->executions));
  IREE_RETURN_IF_ERROR(loom_serve_text_state_row_try_restore(
      row->state, checkpoint->state, out_restored));
  if (*out_restored) {
    row->has_prediction = checkpoint->has_prediction;
    memset(&row->transfer, 0, sizeof(row->transfer));
    row->transfer.token = checkpoint->token;
    row->transfer.progress[0] = checkpoint->has_prediction;
    row->transfer.progress[7] =
        checkpoint->has_prediction &&
        checkpoint->token == row->model->source.eos_token;
  }
  return iree_ok_status();
}

void loom_serve_text_row_complete_reserve(loom_serve_text_row_t* row) {
  loom_serve_text_state_row_complete_reserve(row->state);
  loom_serve_text_model_cancel_admission(row->model);
}

iree_status_t loom_serve_text_row_try_reserve(
    loom_serve_text_row_t* row, loom_serve_text_reserve_mode_t mode,
    const loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t extent,
    bool* out_admitted) {
  *out_admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain_all(row->model->device_count,
                                                      row->model->executions));
  loom_serve_text_reserve_result_t result;
  loom_serve_execution_frontier_t frontier;
  iree_status_t status = loom_serve_text_row_submit_reserve(
      row, mode, checkpoint, extent, &result, &frontier);
  loom_serve_text_model_cancel_admission(row->model);
  status = iree_status_join(
      status, loom_serve_execution_drain_all(row->model->device_count,
                                             row->model->executions));
  if (iree_status_is_ok(status) && result == LOOM_SERVE_TEXT_RESERVE_ACQUIRED) {
    loom_serve_text_row_complete_reserve(row);
    *out_admitted = true;
  }
  return status;
}

void loom_serve_text_row_release_reservation(loom_serve_text_row_t* row) {
  if (!row->state->reserved_extent) {
    return;
  }
  loom_serve_text_state_row_release_reservation(row->state);
  loom_serve_text_model_release_parameters(row->model);
}

void loom_serve_text_checkpoint_retain(
    loom_serve_text_checkpoint_t* checkpoint) {
  if (checkpoint) {
    ++checkpoint->references;
  }
}

void loom_serve_text_checkpoint_release(
    loom_serve_text_checkpoint_t* checkpoint) {
  if (!checkpoint || --checkpoint->references) {
    return;
  }
  loom_serve_text_state_checkpoint_release(checkpoint->state);
  checkpoint->state = NULL;
}

iree_status_t loom_serve_text_checkpoint_submit_suspend(
    loom_serve_text_checkpoint_t* checkpoint,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  if (checkpoint->reservations) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "checkpoint has a prepared continuation");
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_checkpoint_submit_suspend(checkpoint->state));
  *out_frontier =
      loom_serve_execution_frontier(checkpoint->model->executions[0]);
  return iree_ok_status();
}

void loom_serve_text_checkpoint_complete_suspend(
    loom_serve_text_checkpoint_t* checkpoint) {
  loom_serve_text_state_checkpoint_complete_suspend(checkpoint->state);
}

iree_status_t loom_serve_text_checkpoint_suspend(
    loom_serve_text_checkpoint_t* checkpoint) {
  loom_serve_text_model_t* model = checkpoint->model;
  IREE_RETURN_IF_ERROR(
      loom_serve_execution_drain_all(model->device_count, model->executions));
  loom_serve_execution_frontier_t frontier;
  iree_status_t status =
      loom_serve_text_checkpoint_submit_suspend(checkpoint, &frontier);
  status = iree_status_join(
      status,
      loom_serve_execution_drain_all(model->device_count, model->executions));
  if (iree_status_is_ok(status)) {
    loom_serve_text_checkpoint_complete_suspend(checkpoint);
  }
  return status;
}

iree_host_size_t loom_serve_text_checkpoint_suspended_bytes(
    const loom_serve_text_checkpoint_t* checkpoint) {
  return loom_serve_snapshot_size(checkpoint->state->snapshot);
}

iree_host_size_t loom_serve_text_model_checkpoint_suspended_bytes(
    const loom_serve_text_model_t* model) {
  iree_host_size_t bytes = 0;
  for (iree_host_size_t i = 0; i < model->state.checkpoints.capacity; ++i) {
    const loom_serve_text_checkpoint_t* checkpoint = &model->checkpoints[i];
    if (checkpoint->references) {
      bytes += loom_serve_text_checkpoint_suspended_bytes(checkpoint);
    }
  }
  return bytes;
}

int32_t loom_serve_text_row_token(const loom_serve_text_row_t* row) {
  return row->transfer.token;
}

bool loom_serve_text_row_is_eos(const loom_serve_text_row_t* row) {
  return row->transfer.progress[7] != 0;
}

iree_host_size_t loom_serve_text_row_position(
    const loom_serve_text_row_t* row) {
  return row->state->position;
}

loom_serve_text_metrics_t loom_serve_text_row_metrics(
    const loom_serve_text_row_t* row) {
  return row->metrics;
}
