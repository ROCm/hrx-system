// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_CAST_PACKET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_CAST_PACKET_LEGALIZATION_H_

#include "iree/base/api.h"
#include "loom/target/legalization.h"
#include "loom/transforms/vector/packet_legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packetizes a static rank-one vector.bitcast operation whose carrier exceeds
// the target contract. Each packet preserves the source/result bit count and
// is admitted by a projected contract query. Equal-width bitcasts may also
// alias storage when both packet carriers have complete target mappings;
// explicit contract rejections remain authoritative. Returns false through
// |out_rewritten| when no exact, bounded packet schema is admitted.
iree_status_t loom_vector_packet_legalize_bitcast(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

// Packetizes a static rank-one i1 extension whose source or result exceeds a
// target carrier. Each packet lowers directly to the extension's exact select
// form so partial target contracts for native select lowering do not obscure
// the carrier boundary. Returns false through |out_rewritten| when no exact,
// bounded packet schema has complete target carriers.
iree_status_t loom_vector_packet_legalize_predicate_extension(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

// Returns packet materialization semantics for unary i1 extensions. Dynamic
// window legalization uses this to form the extension's exact select while
// each predicate packet is available.
loom_vector_packet_consumer_materializer_t
loom_vector_predicate_extension_packet_materializer(void);

// Packetizes a static rank-one vector.bitpack or vector.bitunpack operation
// whose logical-lane or packed-storage carrier exceeds the target contract.
// Contract queries project both sides of the rate-changing operation, then
// each accepted logical-lane interval consumes or produces its corresponding
// packed-storage interval. Returns false through |out_rewritten| when no exact,
// bounded packet schema is admitted by the target.
iree_status_t loom_vector_packet_legalize_bitstream(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_CAST_PACKET_LEGALIZATION_H_
