// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_PACKET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_PACKET_LEGALIZATION_H_

#include "iree/base/api.h"
#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target packet widths and the source payload boundary where packetization
// replaces ordinary target or reference lowering.
typedef struct loom_vector_packet_policy_t {
  // Native packet widths in bits. Widths are byte-aligned, each byte width is
  // a power of two, and ordering is not significant.
  const uint16_t* native_bit_counts;
  // Number of entries in native_bit_counts.
  uint8_t native_bit_count_count;
  // Largest payload in bits that remains owned by ordinary lowering.
  uint16_t maximum_unpacketized_bit_count;
} loom_vector_packet_policy_t;

// Packetizes a rank-one table lookup over the common lane interval supported
// by both its index and result element types. The table remains one captured
// SSA value. Existing index values supply static slices while decomposable
// index producers stream packet by packet. Returns false through
// |out_rewritten| when neither carrier needs splitting or the packet plan
// cannot be materialized within the static expansion bound.
iree_status_t loom_vector_packet_legalize_table_lookup(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a dense vector load into target-native widths and concatenates
// the packets into the original logical vector. Returns false through
// |out_rewritten| when the access or policy does not admit an exact split.
iree_status_t loom_vector_packet_legalize_load(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a dense vector store into target-native widths. Decomposable
// producer graphs stream packets; other SSA values retain their snapshot and
// supply static slices. Returns false through |out_rewritten| when the access
// or policy does not admit an exact packetization.
iree_status_t loom_vector_packet_legalize_store(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a vector reduction's decomposable producer graph and carries the
// scalar accumulator across native-width packets. Returns false through
// |out_rewritten| when the graph or policy does not admit an exact
// packetization.
iree_status_t loom_vector_packet_legalize_reduce(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_PACKET_LEGALIZATION_H_
