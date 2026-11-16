// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_SERVICE_SCHEDULE_H_
#define LOOM_RUNTIME_SERVICE_SCHEDULE_H_

#include "loom/runtime/service_control.h"

#ifdef __cplusplus
extern "C" {
#endif

// Caller-owned decision buffer. Policy owns emission until it returns or its
// qualified asynchronous completion hands off the buffer. The serialized owner
// then applies a prefix and records outcomes. Counts are ordinary data, never
// concurrent readiness signals. Input control borrows may end after selection;
// decisions retain numeric keys instead of pointers into those inputs.
typedef struct loom_runtime_service_plan_t loom_runtime_service_plan_t;

// One source-owned legal choice in the composition's existing dispatch table.
// Output cardinality is independent of input cardinality; repeated service
// ordinals are valid and every attempt receives its own outcome.
typedef struct loom_runtime_service_decision_t {
  // Stable dispatch slot, checked before accessing its current occupant.
  uint64_t service_ordinal;
  // Occupant lifetime, checked before dispatching to the service.
  uint64_t service_incarnation;
  // Source opportunity whose revision established the legal choices.
  loom_runtime_service_offer_key_t offer;
  // Alternative within that opportunity, checked by the source owner.
  uint64_t choice_ordinal;
} loom_runtime_service_decision_t;

typedef uint64_t loom_runtime_service_selection_flags_t;
enum loom_runtime_service_selection_flag_bits_e {
  // An append reached the output bound; the producer may have more choices.
  LOOM_RUNTIME_SERVICE_SELECTION_CAPACITY = 1u << 0,
};

// Decoded 64-byte header, exclusively borrowed at the current handoff. All
// stored words are little-endian u64, with 8-byte-aligned array starts.
// Decision records occupy 40 bytes and outcomes 32 bytes. No native pointers,
// struct padding, status objects, or host-size-dependent fields enter the
// bytes.
typedef struct loom_runtime_service_plan_layout_t {
  // Complete initialized extent, including both capacity-sized arrays.
  uint64_t byte_length;
  // Independent construction-time output bound, including zero.
  uint64_t decision_capacity;
  // Published decisions in this exclusive handoff, at most capacity.
  uint64_t decision_count;
  // Exactly-once attempted prefix; stale and failed attempts advance it too.
  uint64_t applied_count;
  // Relative start of the decision array.
  uint64_t decisions_offset;
  // Relative start of the outcome array.
  uint64_t results_offset;
  // Producer-reported selection facts, independent of application progress.
  loom_runtime_service_selection_flags_t selection_flags;
  // Reserved word, initialized to zero.
  uint64_t reserved;
} loom_runtime_service_plan_layout_t;

// Native turn limits, independent of output capacity and source work size.
// Accepted obligations progress even when either new-work limit is zero.
typedef struct loom_runtime_service_schedule_budget_t {
  // Remaining policy invocations, including empty or changed observations.
  uint64_t evaluations;
  // Remaining application attempts, including stale choices and failures.
  uint64_t decisions;
} loom_runtime_service_schedule_budget_t;

// Native invocation values. A compiled policy uses the same two i64 values
// through its qualified binding instead of borrowing this native struct.
typedef struct loom_runtime_service_schedule_context_t {
  // Host-monotonic time sampled once by the owner for this policy invocation.
  int64_t now_ns;
  // Next requested evaluation in the same clock, initially infinite future.
  // Event-driven policy leaves it infinite; timed deferral supplies a deadline.
  int64_t wake_time_ns;
} loom_runtime_service_schedule_context_t;

// Cold checked sizing. Capacity zero requires only the 64-byte header.
iree_status_t loom_runtime_service_plan_size(iree_host_size_t decision_capacity,
                                             iree_host_size_t* out_byte_length);

// Initializes caller storage with 8-byte base alignment and sufficient extent.
// The caller joins all policy/application accesses before reinitialization or
// release. Initialization is allocation-free and returns no plan on failure.
iree_status_t loom_runtime_service_plan_initialize(
    iree_host_size_t decision_capacity, iree_byte_span_t storage,
    loom_runtime_service_plan_t** out_plan);

// Ordinary header copy under the current owner's exclusive borrow.
loom_runtime_service_plan_layout_t loom_runtime_service_plan_layout(
    const loom_runtime_service_plan_t* plan);

// Starts a fresh selection after the preceding handoff. Explicitly discards any
// unattempted suffix; accepted operations remain owned by their services. Prior
// outcomes cease to be readable. Storage, offsets, and capacity are unchanged.
void loom_runtime_service_plan_reset(loom_runtime_service_plan_t* plan);

// Appends during exclusive policy emission. False sets CAPACITY without writing
// beyond the array. A full buffer does not truncate an already accepted unit.
bool loom_runtime_service_plan_append(
    loom_runtime_service_plan_t* plan,
    const loom_runtime_service_decision_t* decision);

// Reads a producer-established ordinal below decision_count after selection.
loom_runtime_service_decision_t loom_runtime_service_plan_decision(
    const loom_runtime_service_plan_t* plan, uint64_t ordinal);

// Records the next attempted decision and advances applied_count once. The
// owner calls this exactly once per attempt, including stale/error outcomes,
// with applied_count below decision_count. A nonzero operation serial records
// acceptance independently of status. No borrowed native status enters storage.
void loom_runtime_service_plan_complete(
    loom_runtime_service_plan_t* plan, iree_status_code_t status_code,
    loom_runtime_service_operation_key_t operation);

// Reads an ordinal in the applied prefix after the application handoff.
loom_runtime_service_schedule_outcome_t loom_runtime_service_plan_outcome(
    const loom_runtime_service_plan_t* plan, uint64_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_SERVICE_SCHEDULE_H_
