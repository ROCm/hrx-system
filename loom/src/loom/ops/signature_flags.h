// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_SIGNATURE_FLAGS_H_
#define LOOM_OPS_SIGNATURE_FLAGS_H_

#include "loom/ir/ir.h"
#include "loom/ir/signature_flags.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns explicit flags in argument-then-result order, or an empty span when
// absent. Each byte is one loom_signature_flags_t. No inferred facts are stored
// here. The span expires when the signature is mutated.
iree_const_byte_span_t loom_func_like_signature_flags(
    loom_func_like_t function);

// Compacts an existing contract using the signature owner's original arities
// and removal masks. A NULL mask preserves that entire half. The caller owns
// the structural argument/result edits. One new flat array preserves shared
// clone payloads; removing every promise elides the attribute.
iree_status_t loom_func_like_prune_signature_flags(
    loom_module_t* module, loom_func_like_t function, uint16_t argument_count,
    uint16_t result_count, const bool* removed_arguments,
    const bool* removed_results);

// Projects an existing contract with the signature owner's old-to-new start
// indices and original arities. The source body may already have projected
// block arguments. Each old value maps to the range ending at the next start,
// or at the target arity. Storage access bounds follow buffer/view components.
// Scalar components do not inherit a storage contract. Source and target may
// share immutable attribute payloads before this call.
iree_status_t loom_func_like_project_signature_flags(
    loom_module_t* module, loom_func_like_t source, loom_func_like_t target,
    uint16_t source_argument_count, uint16_t source_result_count,
    const uint16_t* argument_indices, const uint16_t* result_indices);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SIGNATURE_FLAGS_H_
