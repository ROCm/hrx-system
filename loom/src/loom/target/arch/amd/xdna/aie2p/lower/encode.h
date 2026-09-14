// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native block-floating operand encoding and payload representation.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_ENCODE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_ENCODE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns whether the retained value schema uses the native BFP16EBS8 carrier:
// 64 signed mantissa bytes followed by eight shared exponent bytes in EX.
bool loom_aie2p_value_has_bfp_storage(const loom_value_fact_table_t* fact_table,
                                      loom_value_id_t value);

// Selects the supported BF16-to-BFP conversion with nearest-even mantissas and
// flushed BF16 subnormals. The computed exponents remain in the result payload.
bool loom_aie2p_encode_matches(loom_low_lower_context_t* context,
                               const loom_op_t* source_op);

// Emits a selected BF16-to-BFP conversion into native accumulator/EX registers.
iree_status_t loom_aie2p_emit_encode(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_ENCODE_H_
