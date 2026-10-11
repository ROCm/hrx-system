// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_EPOCH_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_EPOCH_H_

#include "experimental/loom_serve/text/model.h"
#include "experimental/loom_serve/text/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable model scheduling choices. Admission and output transport credit are
// settled by the caller before preparing an epoch.
typedef struct loom_serve_text_epoch_options_t {
  // Logical context bound of every resident row.
  iree_host_size_t context_capacity;
  // Maximum known input tokens granted to one row in an epoch.
  iree_host_size_t chunk_size;
  // One or two pre-issued device epochs per host publication boundary.
  iree_host_size_t continuation_epochs;
  // Packed, isolated or matched execution of the selected partition.
  loom_serve_text_schedule_mode_t schedule_mode;
  // Whether known inputs and verifiers may share a cohort.
  loom_serve_text_packing_mode_t packing_mode;
  // Borrowed cached shape catalog, stable through the model lifetime.
  const loom_serve_packing_shape_t* shapes;
  // Number of candidate token/span shapes.
  iree_host_size_t shape_count;
} loom_serve_text_epoch_options_t;

// Snapshot of a credited row at a completed model boundary. Only rows selected
// by the packed partition are read. Zero input_count identifies decode; a
// nonzero count is remaining prefill, including a one-token prompt tail.
typedef struct loom_serve_text_epoch_row_t {
  // Borrowed input IDs, stable until all cohort spans have been submitted.
  const int32_t* input_tokens;
  // Number of unconsumed known input IDs; zero for decode.
  iree_host_size_t input_count;
  // Remaining client output allowance, including terminal selection.
  iree_host_size_t output_count;
  // Current consumed position in the model's retained state.
  iree_host_size_t position;
  // Selected but unconsumed anchor for a decode row.
  int32_t token;
} loom_serve_text_epoch_row_t;

// One immutable cohort and its completed feedback. The owner retains this
// storage until successful device readiness and response CPU release. Planning
// has no HTTP side effects, allocations, native mutation or synchronization.
// Span anchors borrow decode_tokens, so a prepared cohort is not movable.
typedef struct loom_serve_text_epoch_t {
  // Number of admitted spans; zero when no cohort is retained.
  iree_host_size_t count;
  // Cached command-program bucket for the first epoch.
  iree_host_size_t shape_index;
  // Current span in isolated/matched execution.
  iree_host_size_t cursor;
  // First issued epoch identity and host-observed submission time.
  struct {
    // First device epoch identity, including pre-issued continuations.
    uint64_t serial;
    // Host observation timestamp in nanoseconds.
    iree_time_t time;
  } start;
  // Number of verifier spans in the first epoch.
  iree_host_size_t proposal_count;
  // Number of pre-issued model traversals for packed execution.
  iree_host_size_t device_epochs;
  // Sum of issued packed bucket capacities, including the continuation. Zero
  // for isolated/matched stages whose individual capacities are not retained.
  iree_host_size_t token_slots;
  // Known inputs scheduled in the second device epoch.
  iree_host_size_t continued_prefill_count;
  // Total first-epoch input slots, including speculative proposals.
  iree_host_size_t input_count;
  // Retained row associations, known IDs and selection flags.
  loom_serve_text_span_t spans[LOOM_SERVE_TEXT_ROW_CAPACITY];
  // Per-span feedback filled only after successful readiness.
  loom_serve_text_result_t results[LOOM_SERVE_TEXT_ROW_CAPACITY];
  // Stable anchors for later isolated submissions.
  int32_t decode_tokens[LOOM_SERVE_TEXT_ROW_CAPACITY];
  // Consumed positions before submission, used by accounting.
  iree_host_size_t positions[LOOM_SERVE_TEXT_ROW_CAPACITY];
  // Total output credit, including the optional second epoch.
  uint32_t output_limits[LOOM_SERVE_TEXT_ROW_CAPACITY];
  // Original input phase, independent of length-one span shape.
  bool prefill[LOOM_SERVE_TEXT_ROW_CAPACITY];
} loom_serve_text_epoch_t;

// Submission-only second-epoch storage. The native model copies these inputs
// during submit; unlike the cohort feedback, this may live on the caller stack.
// continuation borrows this object's arrays, so the object is not movable.
typedef struct loom_serve_text_epoch_continuation_t {
  // Borrowed view over the following arrays; zero spans means absent.
  loom_serve_text_continuation_t continuation;
  // Known input or device-fed verifier spans in the next shape.
  loom_serve_text_span_t spans[LOOM_SERVE_TEXT_ROW_CAPACITY];
  // Per-span output limits across both epochs.
  uint32_t output_limits[LOOM_SERVE_TEXT_ROW_CAPACITY];
} loom_serve_text_epoch_continuation_t;

// Converts an already credited, nonempty packed partition into model inputs.
// The partition, source capacities and row snapshots are trusted owner state.
// Caller supplies start identity/time after planning and before submission.
void loom_serve_text_epoch_prepare(
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_row_t* rows, iree_host_size_t shape_index,
    iree_host_size_t count, const loom_serve_packed_span_t* scheduled,
    loom_serve_text_epoch_t* epoch,
    loom_serve_text_epoch_continuation_t* continuation);

// Submits a packed cohort or the current isolated span without waiting.
// continuation is required for packed execution and unused for isolated modes.
// The owner consumes successful readiness before advancing the isolated cursor
// or publishing results. Failed submission retains accepted model ownership.
iree_status_t loom_serve_text_epoch_submit(
    loom_serve_text_model_t* model, loom_serve_text_schedule_mode_t mode,
    const loom_serve_text_epoch_t* epoch,
    const loom_serve_text_epoch_continuation_t* continuation,
    loom_serve_execution_frontier_t* out_frontier);

// Consumes successful ALL readiness and publishes authoritative model progress
// into results. Isolated execution completes exactly the current span and
// advances cursor; it never synthesizes progress for unsubmitted spans. The
// owner still decides whether to issue another span or stop on cancellation.
iree_status_t loom_serve_text_epoch_complete(
    loom_serve_text_model_t* model, loom_serve_text_schedule_mode_t mode,
    loom_serve_text_epoch_t* epoch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_EPOCH_H_
