// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact target-family facts paired with the provider that owns them.

#ifndef LOOM_TARGET_RESOLVED_TARGET_H_
#define LOOM_TARGET_RESOLVED_TARGET_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_facts_t loom_target_facts_t;
typedef struct loom_target_provider_t loom_target_provider_t;

// One complete target-family resolution inherited by semantic callees.
//
// Both pointers are borrowed, immutable, and non-NULL for a resolved target.
// Keeping them in one value makes family ownership explicit at every boundary
// that interprets or materializes the facts.
typedef struct loom_resolved_target_t {
  // Target-family provider owning and interpreting |facts|.
  const loom_target_provider_t* provider;

  // Exact target-family facts without function-local ABI/export overlays.
  const loom_target_facts_t* facts;
} loom_resolved_target_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_RESOLVED_TARGET_H_
