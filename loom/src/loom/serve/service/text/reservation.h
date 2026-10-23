// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_RESERVATION_H_
#define LOOM_SERVE_SERVICE_TEXT_RESERVATION_H_

#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// A prepared request's exact continuation, before parameter activation or
// device submission. Its owner excludes destination mutation until submission
// or cancellation; independent rows may progress. Zero initialization is empty.
typedef struct loom_serve_text_reservation_t {
  // Borrowed destination, outliving this logical reservation.
  loom_serve_text_row_t* row;
  // Owned KV/COW/recurrent plan; NULL means no logical capacity was acquired.
  struct loom_serve_text_state_reservation_t* state;
  // Retained immutable endpoint, or NULL for fresh/resident continuations.
  loom_serve_text_checkpoint_t* checkpoint;
  // Selected frontier semantics, applied only after successful admission.
  loom_serve_text_reserve_mode_t mode;
} loom_serve_text_reservation_t;

// Selects logical ownership once, with no parameter pin, mapping, or queue
// work. A successful empty result denotes logical capacity pressure. The
// selected row remains unchanged; cancelling this record returns all selected
// IDs. A retained checkpoint must not be suspended or otherwise mutated while
// held.
iree_status_t loom_serve_text_reservation_prepare(
    loom_serve_text_row_t* row, loom_serve_text_reserve_mode_t mode,
    loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_text_reservation_t* out_reservation);

// Conservative language-side completion footprint: parameter activation,
// the prepared state union, peak private commands, and both epoch packet
// headers. Image feature roots and their gathered payload are additional.
// Queries the logical plan without activating or pinning parameters.
uint64_t loom_serve_text_reservation_memory_footprint(
    const loom_serve_text_reservation_t* reservation);

// Submits the same plan after earlier model work has completed. Capacity or
// deferred admission leaves it intact; ACQUIRED transfers logical ownership
// into the row and requires successful ALL readiness and row_complete_reserve.
// Submission/readiness failure remains terminal, with native ownership held
// by the model. The prepared record may be deinitialized on every outcome.
iree_status_t loom_serve_text_reservation_submit(
    loom_serve_text_reservation_t* reservation,
    loom_serve_text_reserve_result_t* out_result,
    loom_serve_execution_frontier_t* out_frontier);

// Cancels unsubmitted ownership or releases submitted plan metadata. Neither
// path waits, releases an admitted request's pin, or mutates its row.
void loom_serve_text_reservation_deinitialize(
    loom_serve_text_reservation_t* reservation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_RESERVATION_H_
