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

typedef enum loom_vector_packet_reduce_result_e {
  // The reduction does not require packet legalization or exceeds its static
  // expansion bound.
  LOOM_VECTOR_PACKET_REDUCE_RESULT_NONE = 0,
  // Packet legalization rewrote the reduction or one of its source reads.
  LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN = 1,
  // The producer graph cannot be decomposed, but a bounded scalar fallback can
  // retain the input SSA value and consume terminal lane extracts from it.
  LOOM_VECTOR_PACKET_REDUCE_RESULT_CAPTURE_INPUT = 2,
} loom_vector_packet_reduce_result_t;

// Packetizes a static rank-one shape-preserving elementwise operation into
// target-native packets and concatenates its packet results. Each operand
// retains its authored SSA snapshot and supplies static packet slices. Shared
// snapshots whose direct packet fanout exceeds the static expansion bound
// are first reified in private storage for independent consumer packetization.
// Returns false through |out_rewritten| when the operation already fits one
// packet, lacks the decomposable elementwise contract, or has no bounded
// packet representation.
iree_status_t loom_vector_packet_legalize_elementwise(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes one static rank-one decomposable vector result and its compatible
// producer graph into target-native packets, then concatenates the packet
// results into the original logical value. Block arguments and producer values
// outside the selected graph supply static slices. Shared snapshots whose
// direct packet fanout exceeds the static expansion bound are first reified
// in private storage for independent consumer packetization. Returns false
// through |out_rewritten| when every root field already fits one packet or the
// graph has no bounded packet representation.
iree_status_t loom_vector_packet_legalize_decomposable_graph(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a static vector splat into target-native rank-one packets,
// concatenates them into a flat carrier, and restores the logical result
// shape. Returns false through |out_rewritten| when the result already fits
// one packet or the packet plan exceeds the static expansion bound.
iree_status_t loom_vector_packet_legalize_splat(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a rank-one table lookup over the common lane interval supported
// by both its index and result element types. The table remains one captured
// SSA value. Existing index values supply static slices while decomposable
// index producers stream packet by packet. Shared index snapshots whose
// direct packet fanout exceeds the static expansion bound are first reified
// in private storage for independent consumer packetization. Returns false
// through |out_rewritten| when neither carrier needs splitting or the packet
// plan has no bounded representation.
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
// producer graphs stream packets; other SSA values retain their authored
// snapshot. Shared snapshots are reified in private storage when retaining the
// packet tuple across consumers would exceed the static expansion bound.
// Returns false through |out_rewritten| when the access or policy does not
// admit an exact packetization.
iree_status_t loom_vector_packet_legalize_store(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a vector reduction's decomposable producer graph and carries the
// scalar accumulator across native-width packets. A bounded reduction whose
// producer graph cannot be decomposed requests a captured-input scalar
// fallback through |out_result|. Shared snapshots are reified in private
// storage when retaining the packet tuple across consumers would exceed the
// static expansion bound. Returns NONE when neither representation has a
// bounded lowering.
iree_status_t loom_vector_packet_legalize_reduce(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* policy,
    loom_vector_packet_reduce_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_PACKET_LEGALIZATION_H_
