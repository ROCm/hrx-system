// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_HISTORY_H_
#define LOOM_SERVE_SERVER_REALTIME_HISTORY_H_

#include "loom/serve/server/realtime/binding.h"
#include "loom/serve/service/text/checkpoint_control.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_realtime_history_phase_e {
  LOOM_SERVE_REALTIME_HISTORY_EMPTY,
  LOOM_SERVE_REALTIME_HISTORY_CAPTURE,
  LOOM_SERVE_REALTIME_HISTORY_PINNED,
  LOOM_SERVE_REALTIME_HISTORY_RELEASE,
} loom_serve_realtime_history_phase_t;

typedef struct loom_serve_realtime_history_pin_t {
  // Logical resource intent, separate from control receipt ownership.
  loom_serve_realtime_history_phase_t phase;
  // Owned internal checkpoint ID, stable through control admission/retirement.
  iree_string_builder_t name;
  // Owned source session identifying the completed native text request.
  iree_string_builder_t session;
  // Last included conversation creation identity, including generated output.
  uint64_t through;
  // Invalidation during capture requires observing its effect before release.
  bool invalidated;
  // Accepted control receipt, retained until completion even after disconnect.
  loom_serve_text_checkpoint_control_t* control;
} loom_serve_realtime_history_pin_t;

typedef struct loom_serve_realtime_history_t {
  // Borrowed registry; all operations run on its serialized owner.
  loom_serve_registry_t* registry;
  // Original text registration; cleanup never resolves a replacement namesake.
  loom_serve_realtime_binding_t binding;
  // Current completed prefix and its predecessor for playback reconciliation.
  loom_serve_realtime_history_pin_t pins[2];
} loom_serve_realtime_history_t;

void loom_serve_realtime_history_initialize(
    loom_serve_registry_t* registry, loom_serve_realtime_binding_t binding,
    iree_allocator_t host_allocator, loom_serve_realtime_history_t* history);

// Requests capture after native text completion. Pressure leaves accepted false
// while the oldest of two pins is released; the preceding turn remains pinned.
// The caller holds the completed generation until capture is accepted.
iree_status_t loom_serve_realtime_history_capture(
    loom_serve_realtime_history_t* history, iree_string_view_t session,
    uint64_t through, bool* accepted);
// Latest compatible completed prefix, borrowed until the next history advance.
iree_string_view_t loom_serve_realtime_history_checkpoint(
    const loom_serve_realtime_history_t* history);
// Invalidate pins including an edited item; zero invalidates every prefix.
void loom_serve_realtime_history_invalidate(
    loom_serve_realtime_history_t* history, uint64_t sequence);
// Observes accepted effects before cleanup, including shutdown while PIN runs.
// A rejected capture has no pin effect and fails only its session. A failed
// release sets out_cleanup_failed: the original model must retire the resource.
iree_status_t loom_serve_realtime_history_advance(
    loom_serve_realtime_history_t* history, bool stopping, bool* out_progress,
    bool* out_cleanup_failed);
bool loom_serve_realtime_history_pending(
    const loom_serve_realtime_history_t* history);
// A newly completed turn is still being captured; the next response waits for
// its definitive compatible-prefix choice instead of racing pin publication.
bool loom_serve_realtime_history_capturing(
    const loom_serve_realtime_history_t* history);
// Requires no pins or controls. NULL is not accepted.
void loom_serve_realtime_history_deinitialize(
    loom_serve_realtime_history_t* history);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_HISTORY_H_
