// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// gfx1250 A0 instruction errata classification.

#ifndef LOOM_TARGET_ARCH_AMDGPU_GFX1250_A0_ERRATA_H_
#define LOOM_TARGET_ARCH_AMDGPU_GFX1250_A0_ERRATA_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_gfx1250_a0_erratum_t {
  // Stable key identifying the architectural A0 restriction.
  iree_string_view_t erratum_key;
  // Stable key identifying the legalization required before native emission.
  iree_string_view_t legalization_key;
} loom_amdgpu_gfx1250_a0_erratum_t;

// Returns the gfx1250 A0 erratum affecting |descriptor|, or NULL when the
// descriptor is legal on A0 without stepping-specific legalization.
const loom_amdgpu_gfx1250_a0_erratum_t*
loom_amdgpu_gfx1250_a0_erratum_for_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_GFX1250_A0_ERRATA_H_
