// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native emission preflight over target-low tables.
//
// This layer validates the target-independent native fragment contract and
// derives AMDGPU register metadata facts before text, binary, metadata, or
// descriptor emission consume the function.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PREFLIGHT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PREFLIGHT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/error/emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_native_preflight_t {
  // Schedule table analyzed by this preflight result.
  const loom_low_schedule_table_t* schedule;
  // Allocation table analyzed by this preflight result.
  const loom_low_allocation_table_t* allocation;
  // Number of blocking target diagnostics emitted during preflight.
  iree_host_size_t error_count;
  // Highest SGPR index used by the function body plus one.
  uint32_t next_free_sgpr;
  // Highest VGPR index used by the function body plus one.
  uint32_t next_free_vgpr;
} loom_amdgpu_native_preflight_t;

typedef struct loom_amdgpu_native_preflight_options_t {
  // Optional structured diagnostic emitter for user-dependent target failures.
  iree_diagnostic_emitter_t emitter;
} loom_amdgpu_native_preflight_options_t;

// Validates one scheduled and allocated AMDGPU native fragment before emission.
//
// The resulting facts remain valid only while |schedule|, |allocation|, and
// their backing target descriptor tables remain live and immutable.
iree_status_t loom_amdgpu_native_preflight_analyze(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_native_preflight_options_t* options,
    loom_amdgpu_native_preflight_t* out_preflight);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PREFLIGHT_H_
