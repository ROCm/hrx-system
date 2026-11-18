// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_RESERVATION_PLAN_H_
#define LOOM_SERVE_SERVICE_TEXT_RESERVATION_PLAN_H_

#include "loom/serve/service/text/state.h"

enum loom_serve_text_reservation_flag_bits_t {
  // The immutable checkpoint needs its contents restored before use.
  LOOM_SERVE_TEXT_RESERVATION_COLD = 1u << 0,
  // The selected destination already owns the borrowed prefix.
  LOOM_SERVE_TEXT_RESERVATION_CONTINUE = 1u << 1,
  // Admission transferred the tentative ownership into the destination.
  LOOM_SERVE_TEXT_RESERVATION_INSTALLED = 1u << 2,
  // The complete physical requirements have been materialized and retained.
  LOOM_SERVE_TEXT_RESERVATION_REQUIREMENTS = 1u << 3,
};
typedef uint32_t loom_serve_text_reservation_flags_t;

// Prepared logical transition shared by physical submission and cold
// forecasting. Forecasting reads this plan; it never changes its IDs or
// commitment ranges.
struct loom_serve_text_state_reservation_t {
  // Next prepared plan; the state borrows every live plan through this list.
  loom_serve_text_state_reservation_t* next;
  // Link naming this plan, permitting constant-time cancellation.
  loom_serve_text_state_reservation_t** previous_link;
  // Borrowed destination, unchanged until successful submission.
  loom_serve_text_state_row_t* row;
  // Borrowed restore endpoint, protected by the request through submission.
  loom_serve_text_state_checkpoint_t* checkpoint;
  // Cold checkpoint's prepared ownership, sharing the trailing block array.
  loom_serve_text_state_checkpoint_t restored;
  // Requested consumed-position high-water, including speculative slack.
  uint32_t extent;
  // Logical input frontier to install with the selected ownership.
  uint32_t position;
  // Source-owned frontier accompanying the logical input position.
  uint32_t source_position;
  // Consumed prefix pages preceding unused completion capacity.
  uint32_t prefix;
  // Complete selected page count, including the borrowed prefix.
  uint32_t count;
  // Leading IDs borrowed from the immutable row or resident checkpoint.
  uint32_t preserved;
  // Authoritative recurrent reader, or UINT32_MAX for a fresh row.
  uint32_t reader;
  // Private recurrent writer selected by the existing COW planner.
  uint32_t writer;
  // Original shared partial page retained until its copy retires.
  uint32_t tail;
  // Ownership and materialization state for this exact logical transition.
  loom_serve_text_reservation_flags_t flags;
  // Owned source restore descriptors; NULL without a cold checkpoint.
  loom_runtime_snapshot_range_t* restore_ranges;
  // Number of restore descriptors.
  iree_host_size_t restore_count;
  // Owned uncertain additions for native admission; admitted borrows are
  // absent.
  loom_runtime_memory_range_t* commitments;
  // Number of physical ranges, zero for fixed-backed state.
  iree_host_size_t commitment_count;
  // Lazily owned complete source requirements, merged in physical byte order.
  loom_runtime_snapshot_range_t* requirements;
  // Number of complete requirements, including admitted shared sources.
  iree_host_size_t requirement_count;
  // Complete physical footprint of this continuation and shared model state.
  uint64_t footprint_bytes;
  // Owned changed suffix followed by cold checkpoint pages when present.
  uint32_t blocks[];
};

// Borrows the unchanged prefix from its existing logical owner.
const uint32_t* loom_serve_text_state_reservation_prefix(
    const loom_serve_text_state_reservation_t* reservation);

// Materializes the complete physical requirements once for media admission or
// cold forecasting. Does not inspect or mutate the native backing directory.
iree_status_t loom_serve_text_state_reservation_requirements(
    loom_serve_text_state_reservation_t* reservation);

#endif  // LOOM_SERVE_SERVICE_TEXT_RESERVATION_PLAN_H_
