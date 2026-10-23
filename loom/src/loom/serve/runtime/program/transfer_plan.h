// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_TRANSFER_PLAN_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_TRANSFER_PLAN_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One independent byte placement. Binding ordinals refer to source and target
// tables supplied by the invoking service, never pointers, device addresses,
// or transport registrations. Each table has its own ordinal space.
typedef struct loom_serve_transfer_range_t {
  // Ordinal of the storage providing the bytes.
  uint64_t source_binding;
  // First byte in the source binding.
  uint64_t source_offset;
  // Ordinal of the storage receiving the bytes.
  uint64_t target_binding;
  // First byte in the target binding, independent of source_offset.
  uint64_t target_offset;
  // Exact number of bytes to place, excluding any unused capacity.
  uint64_t length;
} loom_serve_transfer_range_t;

enum { LOOM_SERVE_TRANSFER_RANGE_BYTES = 5 * sizeof(uint64_t) };

// Checked view of a source-authored byte-placement table. Each record encodes
// {source_binding, source_offset, target_binding, target_offset, length} as
// five LE u64 values. Nonempty ranges are ordered by target binding and target
// offset, with no overlap. Source ranges may repeat or overlap. Table order
// describes layout, not execution order: all placements are independent.
//
// The owner supplies distinct target storage and sources that remain unchanged
// by these writes. It resolves bindings, establishes producer readiness, and
// keeps payloads alive through actual transport retirement, including failure.
// Source return, target visibility, and target-consumer retirement are separate
// events; this table conveys none of them.
typedef struct loom_serve_transfer_plan_t {
  // Borrowed immutable records, valid until the consumer captures descriptors.
  const uint8_t* records;
  // Number of complete, checked records.
  iree_host_size_t count;
} loom_serve_transfer_plan_t;

// Validates an external model's complete table against bound byte extents
// before any placement is submitted. Empty tables and zero-length ranges are
// allowed; zero-length ranges still require valid bindings and bounded offsets
// but do not participate in target ordering. This borrows metadata, allocates
// nothing, and neither resolves storage nor moves bytes. Failure leaves
// out_plan empty.
//
// The binding tables and their extents are native-owner facts. Rebinding a
// checked plan requires the same extents or another validation at that
// boundary.
iree_status_t loom_serve_transfer_plan_initialize(
    iree_const_byte_span_t bytes, iree_host_size_t source_count,
    const uint64_t* source_lengths, iree_host_size_t target_count,
    const uint64_t* target_lengths, loom_serve_transfer_plan_t* out_plan);

// Reads one record from a checked plan. The caller supplies index < plan.count;
// internal consumers do not repeat external-input validation.
loom_serve_transfer_range_t loom_serve_transfer_plan_range(
    loom_serve_transfer_plan_t plan, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_TRANSFER_PLAN_H_
