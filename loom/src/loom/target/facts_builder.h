// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Construction-only helpers for immutable typed target facts.

#ifndef LOOM_TARGET_FACTS_BUILDER_H_
#define LOOM_TARGET_FACTS_BUILDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the target-neutral base of a family-owned fact object from one
// complete structured target bundle with no explicit common fields.
void loom_target_facts_builder_initialize(
    const loom_target_fact_type_t* fact_type,
    const loom_target_bundle_t* bundle, loom_target_facts_t* out_facts);

// Clones |source| into mutable construction storage allocated from |arena|.
iree_status_t loom_target_facts_builder_clone(const loom_target_facts_t* source,
                                              iree_arena_allocator_t* arena,
                                              loom_target_facts_t** out_facts);

// Applies every explicit common field in |requirement| to |effective| and
// unions the explicit field sets.
//
// The caller must first prove that |effective| satisfies |requirement|.
void loom_target_facts_builder_apply_requirement(
    const loom_target_facts_t* requirement, loom_target_facts_t* effective);

// Replaces the common target bundle while preserving family-owned facts.
void loom_target_facts_builder_replace_bundle(
    const loom_target_bundle_t* bundle, loom_target_facts_t* facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FACTS_BUILDER_H_
