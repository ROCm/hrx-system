// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared successor payload verification for branch-like ops.

#ifndef LOOM_OPS_SUCCESSOR_VERIFY_H_
#define LOOM_OPS_SUCCESSOR_VERIFY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies that |args| match |target| block arguments after remapping
// destination block arguments to edge payload values. This emits STRUCTURE/025
// for count mismatches and STRUCTURE/026 for type mismatches.
iree_status_t loom_ops_verify_successor_args(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t op_name, uint8_t successor_index,
    const loom_block_t* target, const loom_value_id_t* args,
    iree_host_size_t arg_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SUCCESSOR_VERIFY_H_
