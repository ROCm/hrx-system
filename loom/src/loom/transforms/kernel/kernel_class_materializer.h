// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independent kernel source products specialized to one live launch class.

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_MATERIALIZER_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/transforms/kernel/kernel_class_classifier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Materializes one independently owned kernel source module for
// |class_ordinal|.
//
// Accepted provider decisions become ordinary input assumptions followed by
// exact template calls. Decisions intentionally skipped during collection
// remain generic template applications. No classifier or collection state is
// retained by the returned module, which the caller must free with
// loom_module_free().
//
// The classifier, collection, and source module must describe the same
// immutable compiler snapshot. |class_ordinal| must name a live class in
// |collection|. These are trusted compiler invariants rather than user-input
// validation boundaries.
iree_status_t loom_kernel_class_materialize(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_collection_t* collection,
    loom_decision_class_ordinal_t class_ordinal,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_CLASS_MATERIALIZER_H_
