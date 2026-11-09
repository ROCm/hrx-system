// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_FINITE_ACCUMULATION_H_
#define LOOM_SERVICE_FINITE_ACCUMULATION_H_

#include "loom/service/finite/result.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_service_finite_output_flag_bits_e {
  LOOM_SERVICE_FINITE_OUTPUT_FLAG_DEVICE = 1u << 0,
  LOOM_SERVICE_FINITE_OUTPUT_FLAG_HOST = 1u << 1,
} loom_service_finite_output_flag_bits_t;
typedef uint32_t loom_service_finite_output_flags_t;

// Logical request output independent of reusable numerical cohort storage.
// Zero initialization is empty. The first partial publication allocates one
// row table for the protocol result count. All subsequent publications fill
// its ordered prefix; the source boundary has checked the remaining capacity.
// Native buffers and host snapshots are retained only for interested consumer
// kinds. No numerical bytes are copied by this owner.
typedef struct loom_service_finite_accumulation_t {
  // Allocator owning the row table, selected on its first publication.
  iree_allocator_t host_allocator;
  // Number of captured rows in original logical-request order.
  iree_host_size_t count;
  // Owned allocation containing the selected row/owner arrays below.
  void* storage;
  // Owned device buffer references; NULL for host-only consumers.
  iree_hal_buffer_ref_t* device_rows;
  // Borrowed spans protected by corresponding snapshot owners.
  iree_const_byte_span_t* host_rows;
  // Owned snapshot references, one for each host row borrower.
  loom_service_finite_snapshot_t** host_owners;
  // Latest producer result, retaining the execution and dominating frontier.
  // All device rows were produced by preceding epochs on this same execution.
  loom_service_finite_result_t* latest;
} loom_service_finite_accumulation_t;

void loom_service_finite_accumulation_deinitialize(
    loom_service_finite_accumulation_t* accumulation);

// Captures a nonempty checked prefix after submission. Host spans are merely
// recorded here; their bytes are read only after successful cohort readiness.
// expected_count and flags remain constant across a logical request.
iree_status_t loom_service_finite_accumulation_append(
    loom_service_finite_accumulation_t* accumulation,
    iree_host_size_t expected_count, loom_service_finite_output_flags_t flags,
    loom_service_finite_result_t* result,
    loom_service_finite_snapshot_t* snapshot, iree_host_size_t request,
    iree_allocator_t host_allocator);

// Publishes immutable metadata over the retained native rows after the logical
// request has exhausted its protocol result count. The output is independently
// owned; releasing this accumulator cannot invalidate it or its queue captures.
iree_status_t loom_service_finite_accumulation_result(
    const loom_service_finite_accumulation_t* accumulation,
    uint64_t input_tokens, loom_service_finite_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_FINITE_ACCUMULATION_H_
