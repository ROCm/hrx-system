// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_TURN_H_
#define LOOM_SERVE_SERVER_REALTIME_TURN_H_

#include "loom/serve/server/realtime/capture.h"
#include "loom/serve/server/realtime/input.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_turn_options_t {
  // Source speech probability at or above which speech is active.
  float threshold;
  // Original microphone history included before an observed onset.
  uint32_t prefix_padding_ms;
  // Consecutive classified silence required to close an active utterance.
  uint32_t silence_duration_ms;
} loom_serve_realtime_turn_options_t;

typedef enum loom_serve_realtime_turn_event_kind_e {
  LOOM_SERVE_REALTIME_TURN_NONE,
  LOOM_SERVE_REALTIME_TURN_STARTED,
  LOOM_SERVE_REALTIME_TURN_STOPPED,
} loom_serve_realtime_turn_event_kind_t;

typedef struct loom_serve_realtime_turn_event_t {
  // Onset includes pre-roll; stop includes the selected trailing silence.
  loom_serve_realtime_turn_event_kind_t kind;
  // Exact position in the continuous original 24 kHz microphone clock.
  uint64_t position;
} loom_serve_realtime_turn_event_t;

typedef struct loom_serve_realtime_turn_t loom_serve_realtime_turn_t;

// Owns continuous detector conversion and bounded retiring epochs. Capture
// remains caller-owned through the final PCM release. Every call is serialized
// by the registry owner; finite numerical work uses the shared executor.
iree_status_t loom_serve_realtime_turn_allocate(
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_realtime_binding_t binding,
    loom_serve_realtime_capture_t* capture, iree_host_size_t capture_capacity,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_turn_t** out_turn);

// Begins an epoch at the current ordered append frontier. The caller invokes
// this before its first append and only when no epoch is open. Source geometry
// is admitted against capture capacity after asynchronous priming.
iree_status_t loom_serve_realtime_turn_begin(
    loom_serve_realtime_turn_t* turn,
    loom_serve_realtime_turn_options_t options);
// Ends current detector interest without resetting capture's raw clock.
// Accepted conversion and PCM loans retain their records through retirement.
void loom_serve_realtime_turn_clear(loom_serve_realtime_turn_t* turn);
bool loom_serve_realtime_turn_is_open(const loom_serve_realtime_turn_t* turn);
bool loom_serve_realtime_turn_is_speaking(
    const loom_serve_realtime_turn_t* turn);

iree_status_t loom_serve_realtime_turn_advance(loom_serve_realtime_turn_t* turn,
                                               bool stopping,
                                               bool* out_progress);
// One pending transition, or NONE. Consume only after the caller has retained
// the interval in its finite transcription or committed conversation owner.
loom_serve_realtime_turn_event_t loom_serve_realtime_turn_peek(
    const loom_serve_realtime_turn_t* turn);
void loom_serve_realtime_turn_consume(loom_serve_realtime_turn_t* turn);
// Classified raw endpoint, excluding input whose activity is still pending.
uint64_t loom_serve_realtime_turn_covered(
    const loom_serve_realtime_turn_t* turn);
// Earliest raw frame required by detector input, pre-roll, or pending onset.
// The caller combines this with independently retained transcription cursors.
uint64_t loom_serve_realtime_turn_retain(
    const loom_serve_realtime_turn_t* turn);
// Earliest uncommitted pre-roll; explicit commit can begin a finite interval
// here when no onset has occurred. False leaves out_position untouched while
// the epoch's original ordered start is still pending publication.
bool loom_serve_realtime_turn_prefix(const loom_serve_realtime_turn_t* turn,
                                     uint64_t* out_position);

bool loom_serve_realtime_turn_pending(const loom_serve_realtime_turn_t* turn);
// Requires no open epoch or accepted CPU borrowers. NULL is accepted.
void loom_serve_realtime_turn_free(loom_serve_realtime_turn_t* turn);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_TURN_H_
