// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMD XDNA AIE2P target-op interpretation.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_TARGET_H_

#include "loom/ir/ir.h"
#include "loom/ops/target/facts.h"
#include "loom/target/resolved_target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Projects the exact device profile carried by a verified target record.
extern const loom_target_fact_projector_t loom_aie2p_target_fact_projector;

// Materializes exact resolved AIE2P facts as an aie2p.target definition.
iree_status_t loom_aie2p_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location);

// Verifies AIE2P-specific target-record semantics.
iree_status_t loom_aie2p_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_TARGET_H_
