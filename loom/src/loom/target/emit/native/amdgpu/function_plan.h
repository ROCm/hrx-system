// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-prepared AMDGPU native function plan consumed by final emission.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_PLAN_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_PLAN_H_

#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-native decisions shared by kernel entries and callable helpers.
//
// Native preflight, storage placement, and packet insertion depend only on one
// accepted Low function frame. Kernel metadata and residency remain a separate
// entry-only contract layered over this plan.
typedef struct loom_amdgpu_function_plan_t {
  // Register and storage facts accepted before native planning.
  loom_amdgpu_native_preflight_t preflight;
  // Function-local storage placement consumed by instruction emission.
  loom_amdgpu_storage_layout_t storage_layout;
  // Native packet insertions and target scheduling decisions.
  loom_amdgpu_packet_plan_t packet_plan;
} loom_amdgpu_function_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_PLAN_H_
