// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/turn.h"

typedef struct loom_serve_realtime_turn_epoch_t {
  // Current continuous native activity input, closed independently of its loan.
  loom_serve_realtime_input_item_t* input;
  // Producer-owned start after earlier accepted encoded appends publish.
  loom_serve_realtime_capture_mark_t begin;
  // Stable original PCM loan, retained through accepted conversion release.
  loom_serve_realtime_capture_span_t span;
  // First original frame not yet offered to the detector converter.
  uint64_t position;
  // Source/converter retention geometry has passed capacity admission.
  bool ready;
} loom_serve_realtime_turn_epoch_t;

struct loom_serve_realtime_turn_t {
  // Allocator owning this object and its input converter.
  iree_allocator_t host_allocator;
  // Borrowed original microphone storage, shared with finite transcriptions.
  loom_serve_realtime_capture_t* capture;
  // Immutable raw ring capacity, in 24 kHz mono frames.
  iree_host_size_t capacity;
  // Copied assistant request share for each independently admitted activity
  // row.
  uint32_t execution_weight;
  // One finite conversion slot with independent numerical epoch histories.
  loom_serve_realtime_input_t* input;
  // Stable loan records for a current epoch and cancelled CPU borrowers.
  loom_serve_realtime_turn_epoch_t epochs[4];
  // Current epoch; NULL until begin or after clear.
  loom_serve_realtime_turn_epoch_t* current;
  // Selected policy, immutable through an epoch.
  loom_serve_realtime_turn_options_t options;
  // One pending onset/stop; the caller controls transition consumption.
  loom_serve_realtime_turn_event_t event;
  // Classified endpoint in the original microphone clock.
  uint64_t covered;
  // Previous committed stop or explicit epoch start, bounding pre-roll.
  uint64_t floor;
  // Start of the current run of classified silence, valid while quiet.
  uint64_t quiet_begin;
  // A detected utterance is currently open.
  bool speaking;
  // At least one consecutive inactive observation follows active speech.
  bool quiet;
};

iree_status_t loom_serve_realtime_turn_allocate(
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_realtime_binding_t binding,
    loom_serve_realtime_capture_t* capture, iree_host_size_t capture_capacity,
    uint32_t execution_weight, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator, loom_serve_realtime_turn_t** out_turn) {
  *out_turn = NULL;
  loom_serve_realtime_turn_t* turn = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*turn), (void**)&turn));
  turn->host_allocator = host_allocator;
  turn->capture = capture;
  turn->capacity = capture_capacity;
  turn->execution_weight = execution_weight;
  iree_status_t status = loom_serve_realtime_input_allocate(
      executor, registry, binding, observer, host_allocator, &turn->input);
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_turn_free(turn);
    return status;
  }
  *out_turn = turn;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_turn_begin(
    loom_serve_realtime_turn_t* turn,
    loom_serve_realtime_turn_options_t options) {
  loom_serve_realtime_turn_epoch_t* epoch = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(turn->epochs); ++i) {
    if (!turn->epochs[i].input && !turn->epochs[i].span.capture) {
      epoch = &turn->epochs[i];
      break;
    }
  }
  if (!epoch) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "cancelled activity conversion is still retiring");
  }
  *epoch = (loom_serve_realtime_turn_epoch_t){0};
  const loom_serve_recognition_request_t request = {
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY,
      .execution_weight = turn->execution_weight};
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_input_begin(turn->input, &request, &epoch->input));
  loom_serve_realtime_capture_mark(turn->capture, &epoch->begin);
  turn->current = epoch;
  turn->options = options;
  turn->covered = turn->floor = 0;
  turn->speaking = turn->quiet = false;
  return iree_ok_status();
}

void loom_serve_realtime_turn_clear(loom_serve_realtime_turn_t* turn) {
  if (turn->current) {
    loom_serve_realtime_input_close(turn->current->input);
    turn->current->input = NULL;
    loom_serve_realtime_capture_mark_cancel(&turn->current->begin);
    turn->current = NULL;
  }
  turn->event = (loom_serve_realtime_turn_event_t){0};
  turn->speaking = turn->quiet = false;
}

bool loom_serve_realtime_turn_is_open(const loom_serve_realtime_turn_t* turn) {
  return turn->current != NULL;
}

bool loom_serve_realtime_turn_is_speaking(
    const loom_serve_realtime_turn_t* turn) {
  return turn->speaking;
}

// Source positions are external clock values. Quotient/remainder conversion
// avoids overflowing a long-running source position before checking its sum.
static iree_status_t loom_serve_realtime_turn_position(uint64_t origin,
                                                       uint32_t rate,
                                                       uint64_t position,
                                                       bool ceil,
                                                       uint64_t* out_position) {
  const uint64_t whole = position / rate;
  const uint64_t fraction =
      (position % rate * 24000 + (ceil ? rate - 1 : 0)) / rate;
  if (origin > UINT64_MAX - fraction ||
      whole > (UINT64_MAX - origin - fraction) / 24000) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "activity position exceeds microphone clock");
  }
  *out_position = origin + whole * 24000 + fraction;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_turn_advance(loom_serve_realtime_turn_t* turn,
                                               bool stopping,
                                               bool* out_progress) {
  if (stopping) {
    loom_serve_realtime_turn_clear(turn);
  }
  loom_serve_realtime_input_advance(turn->input, stopping, out_progress);
  loom_serve_realtime_turn_epoch_t* epoch = turn->current;
  if (!epoch) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_input_status(epoch->input));
  const loom_serve_realtime_input_geometry_t* geometry =
      loom_serve_realtime_input_geometry(epoch->input);
  if (!geometry || !epoch->begin.reached) {
    return iree_ok_status();
  }
  if (!epoch->ready) {
    const uint64_t prefix = (uint64_t)turn->options.prefix_padding_ms * 24;
    if (prefix > turn->capacity || turn->capacity - prefix < 4096 ||
        geometry->retention_frames > turn->capacity - prefix - 4096) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "activity geometry and pre-roll exceed assistant input_capacity");
    }
    epoch->position = epoch->begin.position;
    turn->floor = turn->covered = epoch->begin.position;
    epoch->ready = true;
    *out_progress = true;
  }
  if (loom_serve_realtime_input_complete(epoch->input)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "activity source completed before microphone EOF");
  }
  if (turn->event.kind) {
    return iree_ok_status();
  }
  const loom_serve_recognition_activity_t* activity =
      loom_serve_realtime_input_activity(epoch->input);
  if (activity) {
    uint64_t begin = 0, end = 0;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_turn_position(
        epoch->begin.position, geometry->sample_rate, activity->begin, false,
        &begin));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_turn_position(
        epoch->begin.position, geometry->sample_rate, activity->end, true,
        &end));
    if (activity->speech_probability >= turn->options.threshold) {
      if (!turn->speaking) {
        const uint64_t prefix = (uint64_t)turn->options.prefix_padding_ms * 24;
        turn->event = (loom_serve_realtime_turn_event_t){
            .kind = LOOM_SERVE_REALTIME_TURN_STARTED,
            .position = iree_max(turn->floor, begin - iree_min(begin, prefix))};
        turn->speaking = true;
      }
      turn->quiet = false;
    } else if (turn->speaking) {
      if (!turn->quiet) {
        turn->quiet_begin = turn->covered;
        turn->quiet = true;
      }
      const uint64_t silence = (uint64_t)turn->options.silence_duration_ms * 24;
      if (end - turn->quiet_begin >= silence) {
        turn->event = (loom_serve_realtime_turn_event_t){
            .kind = LOOM_SERVE_REALTIME_TURN_STOPPED,
            .position = turn->quiet_begin + silence};
        turn->floor = turn->event.position;
        turn->speaking = turn->quiet = false;
      }
    }
    turn->covered = end;
    loom_serve_realtime_input_consume_activity(epoch->input);
    *out_progress = true;
  }
  if (!epoch->span.capture &&
      epoch->position < loom_serve_realtime_capture_end(turn->capture)) {
    loom_serve_realtime_capture_borrow(turn->capture, epoch->position, 4096,
                                       &epoch->span);
    const iree_host_size_t count = epoch->span.samples.data_length / 2;
    iree_status_t status = loom_serve_realtime_input_append(
        turn->input, epoch->input, epoch->span.samples,
        loom_serve_realtime_capture_release, &epoch->span);
    if (!iree_status_is_ok(status)) {
      loom_serve_realtime_capture_release(&epoch->span);
      return status;
    }
    epoch->position += count;
    *out_progress = true;
  }
  return iree_ok_status();
}

loom_serve_realtime_turn_event_t loom_serve_realtime_turn_peek(
    const loom_serve_realtime_turn_t* turn) {
  return turn->event;
}

void loom_serve_realtime_turn_consume(loom_serve_realtime_turn_t* turn) {
  turn->event = (loom_serve_realtime_turn_event_t){0};
}

uint64_t loom_serve_realtime_turn_covered(
    const loom_serve_realtime_turn_t* turn) {
  return turn->covered;
}

static uint64_t loom_serve_realtime_turn_prefix_position(
    const loom_serve_realtime_turn_t* turn) {
  if (!turn->current->ready) {
    return turn->current->begin.position;
  }
  const uint64_t prefix = (uint64_t)turn->options.prefix_padding_ms * 24;
  return iree_max(turn->floor, turn->covered - iree_min(turn->covered, prefix));
}

bool loom_serve_realtime_turn_prefix(const loom_serve_realtime_turn_t* turn,
                                     uint64_t* out_position) {
  if (!turn->current->begin.reached) {
    return false;
  }
  *out_position = loom_serve_realtime_turn_prefix_position(turn);
  return true;
}

uint64_t loom_serve_realtime_turn_retain(
    const loom_serve_realtime_turn_t* turn) {
  if (!turn->current || !turn->current->begin.reached) {
    return loom_serve_realtime_capture_end(turn->capture);
  }
  if (!turn->current->ready) {
    return turn->current->begin.position;
  }
  uint64_t retain = iree_min(turn->current->position,
                             loom_serve_realtime_turn_prefix_position(turn));
  if (turn->event.kind == LOOM_SERVE_REALTIME_TURN_STARTED) {
    retain = iree_min(retain, turn->event.position);
  }
  return retain;
}

bool loom_serve_realtime_turn_pending(const loom_serve_realtime_turn_t* turn) {
  return loom_serve_realtime_input_pending(turn->input);
}

void loom_serve_realtime_turn_free(loom_serve_realtime_turn_t* turn) {
  if (!turn) {
    return;
  }
  loom_serve_realtime_input_free(turn->input);
  iree_allocator_free(turn->host_allocator, turn);
}
