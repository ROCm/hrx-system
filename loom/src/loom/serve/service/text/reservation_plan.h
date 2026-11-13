// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_RESERVATION_PLAN_H_
#define LOOM_SERVE_SERVICE_TEXT_RESERVATION_PLAN_H_

#include "loom/serve/service/text/state.h"

// Owned logical selection shared by physical submission and cold forecasting.
// Forecasting reads this plan; it never changes its IDs or commitment ranges.
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
  // Complete selected page count in the trailing block array.
  uint32_t count;
  // Authoritative recurrent reader, or UINT32_MAX for a fresh row.
  uint32_t reader;
  // Private recurrent writer selected by the existing COW planner.
  uint32_t writer;
  // Original shared partial page retained until its copy retires.
  uint32_t tail;
  // Whether the checkpoint's contents need an asynchronous restore.
  bool cold;
  // Successful admission transferred logical ownership to row/checkpoint.
  bool installed;
  // Owned source restore descriptors; NULL without a cold checkpoint.
  loom_runtime_snapshot_range_t* restore_ranges;
  // Number of restore descriptors.
  iree_host_size_t restore_count;
  // Owned exact physical range union input for admission and commitment.
  loom_runtime_memory_range_t* commitments;
  // Source allocation per commitment, coallocated with the native ranges.
  iree_host_size_t* commitment_allocations;
  // Number of physical ranges, zero for fixed-backed state.
  iree_host_size_t commitment_count;
  // Complete physical footprint of this continuation and shared model state.
  uint64_t footprint_bytes;
  // Selected row pages followed by cold checkpoint pages when present.
  uint32_t blocks[];
};

#endif  // LOOM_SERVE_SERVICE_TEXT_RESERVATION_PLAN_H_
