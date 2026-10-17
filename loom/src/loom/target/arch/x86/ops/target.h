// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 target op interpretation.

#ifndef LOOM_TARGET_ARCH_X86_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_X86_OPS_TARGET_H_

#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/resolved_target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes an exact resolved x86 target as an x86.target definition.
iree_status_t loom_x86_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location);

iree_status_t loom_x86_target_record_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_OPS_TARGET_H_
