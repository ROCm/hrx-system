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

// Packetizes a static rank-one scalar insertion whose result spans more than
// one target-native packet. Static insertion updates only the owning packet.
// Dynamic insertion builds one native candidate per packet and selects the
// candidate over that packet's lane interval before concatenating the result.
// Returns false through |out_rewritten| when the result already fits one
// packet, the operation is not a scalar insertion, or the expansion exceeds
// the static bound.
iree_status_t loom_vector_packet_legalize_insert(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

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
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a dense vector load into target-admitted native widths and
// concatenates the packets into the original logical vector. Returns false
// through |out_rewritten| when the access or policy does not admit an exact
// split.
iree_status_t loom_vector_packet_legalize_load(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a dense vector store into target-admitted native widths.
// Decomposable producer graphs stream packets; other SSA values retain their
// authored snapshot. Shared snapshots are reified in private storage when
// retaining the packet tuple across consumers would exceed the static
// expansion bound. Returns false through |out_rewritten| when the access or
// policy does not admit an exact packetization.
iree_status_t loom_vector_packet_legalize_store(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a vector reduction's decomposable producer graph and carries the
// scalar accumulator across native-width packets. A bounded reduction whose
// producer graph cannot be decomposed requests a captured-input scalar
// fallback through |out_result|. Shared snapshots are reified in private
// storage when retaining the packet tuple across consumers would exceed the
// static expansion bound. Returns NONE when neither representation has a
// bounded lowering.
iree_status_t loom_vector_packet_legalize_reduce(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    loom_vector_packet_reduce_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_PACKET_LEGALIZATION_H_
