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
  // The target accepts the authored reduction and packet legalization did not
  // select a component-consumption rewrite.
  LOOM_VECTOR_PACKET_REDUCE_RESULT_RETAIN_NATIVE = 3,
} loom_vector_packet_reduce_result_t;

typedef struct loom_vector_packet_consumer_materializer_t {
  // Opaque state passed to the materializer callbacks.
  void* user_data;
  // Number of IR operations emitted by one materialization callback.
  uint8_t operation_count;
  // Returns true when this materializer implements |op|'s semantics.
  bool (*matches)(void* user_data, const loom_module_t* module,
                  const loom_op_t* op);
  // Returns true when |op| has complete source and result carriers for the
  // proposed packet lane count.
  bool (*supports_packet)(void* user_data,
                          const loom_target_legalization_context_t* context,
                          const loom_op_t* op, uint32_t lane_count);
  // Materializes one packet result from |source| using |op|'s semantics.
  iree_status_t (*materialize)(void* user_data,
                               loom_target_legalization_context_t* context,
                               const loom_op_t* op, uint32_t lane_count,
                               loom_value_id_t source,
                               loom_value_id_t* out_result);
} loom_vector_packet_consumer_materializer_t;

// Returns whether a rejected dynamic window must retain its direct consumer
// chain so packet replay can fuse eligible consumers at runtime-selected
// origins. Invalid or target-legal windows do not claim the chain.
iree_status_t loom_vector_packet_dynamic_window_owns_consumer_chain(
    loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy, const loom_op_t* op,
    bool* out_owns_consumer_chain);

// Packetizes a rank-one dynamic slice by materializing its decomposable
// producer graph directly at each runtime-selected result packet. The optional
// |consumer_materializer| may fuse explicitly supported unary consumers into
// each packet before transient carriers accumulate. Ordinary reads may move to
// the selected packet origins when indexed motion analysis proves that no
// write is crossed. Returns false through |out_rewritten| when the slice is
// static, its producer graph cannot stream, or the expansion exceeds the
// static bound.
iree_status_t loom_vector_packet_legalize_dynamic_slice(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_packet_consumer_materializer_t* consumer_materializer,
    bool* out_rewritten);

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

// Packetizes a dense vector load using the widest target-admitted packet no
// larger than |maximum_lane_count|. This lets a component closure preserve one
// directly consumable packet width across a shared memory root. A zero maximum
// retains the ordinary target-selected width.
iree_status_t loom_vector_packet_legalize_load_with_maximum_lane_count(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    uint32_t maximum_lane_count, bool* out_rewritten);

// Returns whether a dense vector store can consume an exact packet of
// |lane_count| without regrouping it through the logical aggregate.
iree_status_t loom_vector_packet_store_supports_lane_count(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, uint32_t lane_count,
    bool* out_supported);

// Packetizes a dense vector store into target-admitted native widths. Exact
// partitions retained from an oversized producer graph are stored directly
// even when the logical result itself fits a target carrier. Other
// decomposable producer graphs stream packets, while opaque SSA values retain
// their authored snapshot. Shared snapshots are reified in private storage
// when retaining the packet tuple across consumers would exceed the static
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
