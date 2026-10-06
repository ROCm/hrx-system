// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_TABLE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_TABLE_H_

#include "loom/target/legalization.h"
#include "loom/transforms/vector/packet_legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packetizes wide indices and results before expanding register lookups into
// packed bit tests and selection trees. Double-word indices narrow to native
// word predicates after packetization because every defined table index fits.
// Predicate table packets widen once to bytes so every leaf can retain native
// indexed-broadcast selection before the selected bytes return to a predicate;
// other uniform leaves remain direct native broadcasts. Returns false through
// |out_rewritten| when direct target selection should handle the lookup or no
// supported vector rewrite exists.
iree_status_t loom_aie2p_table_lookup_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* packet_policy,
    bool* out_rewritten);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_TABLE_H_
