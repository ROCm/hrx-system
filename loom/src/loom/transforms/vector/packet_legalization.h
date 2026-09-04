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
  // Native packet widths in bits. Ordering is not significant.
  const uint16_t* native_bit_counts;
  // Number of entries in native_bit_counts.
  uint8_t native_bit_count_count;
  // Largest payload in bits that remains owned by ordinary lowering.
  uint16_t maximum_unpacketized_bit_count;
} loom_vector_packet_policy_t;

// Packetizes a dense vector store and its decomposable producer graph into
// target-native widths. Returns false through |out_rewritten| when the graph or
// policy does not admit an exact packetization.
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
