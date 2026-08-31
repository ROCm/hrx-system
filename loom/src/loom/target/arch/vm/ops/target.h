// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM target op construction.

#ifndef LOOM_TARGET_ARCH_VM_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_VM_OPS_TARGET_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

typedef struct loom_builder_t loom_builder_t;

#ifdef __cplusplus
extern "C" {
#endif

// Builds the canonical Core VM target definition for |symbol|.
iree_status_t loom_vm_target_build_core(loom_builder_t* builder,
                                        loom_symbol_ref_t symbol,
                                        loom_location_id_t location,
                                        loom_op_t** out_target_op);

// Builds a Core VM target definition with the common execution limits from
// |source_snapshot|. Device-specific identity, ABI, address-space, and
// representation fields are intentionally not projected.
iree_status_t loom_vm_target_build_core_with_execution_limits(
    loom_builder_t* builder, loom_symbol_ref_t symbol,
    const loom_target_snapshot_t* source_snapshot, loom_location_id_t location,
    loom_op_t** out_target_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_OPS_TARGET_H_
