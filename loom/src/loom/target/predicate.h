// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-aware pass.where predicate provider.

#ifndef LOOM_TARGET_PREDICATE_H_
#define LOOM_TARGET_PREDICATE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/pass/predicate.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_pass_predicate_provider_storage_t {
  // Block pool used for transient symbol facts during predicate evaluation.
  iree_arena_block_pool_t* block_pool;
} loom_target_pass_predicate_provider_storage_t;

// Initializes storage for a target pass predicate provider.
void loom_target_pass_predicate_provider_storage_initialize(
    iree_arena_block_pool_t* block_pool,
    loom_target_pass_predicate_provider_storage_t* out_storage);

// Returns a pass.where provider implementing the `target` predicate.
loom_pass_predicate_provider_t loom_target_pass_predicate_provider(
    loom_target_pass_predicate_provider_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PREDICATE_H_
