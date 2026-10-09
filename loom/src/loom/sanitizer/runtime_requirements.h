// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime services required by executable sanitizer operations.

#ifndef LOOM_SANITIZER_RUNTIME_REQUIREMENTS_H_
#define LOOM_SANITIZER_RUNTIME_REQUIREMENTS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/sanitizer/options.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_sanitizer_runtime_requirement_bit_e {
  // No sanitizer runtime service is required.
  LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE = 0u,
  // Requires a kernel-to-runtime diagnostic feedback channel.
  LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK = 1u << 0,
  // Requires address-sanitizer shadow state.
  LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW = 1u << 1,
  // Requires race-sanitizer shadow state.
  LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW = 1u << 2,
};

// Bitset of loom_sanitizer_runtime_requirement_bit_e values.
typedef uint32_t loom_sanitizer_runtime_requirements_t;

// Returns runtime requirements implied by instrumentation |options|.
loom_sanitizer_runtime_requirements_t
loom_sanitizer_runtime_requirements_from_options(
    const loom_sanitizer_options_t* options);

// Returns runtime requirements for executable sanitizer operations in
// |module| and instrumentation requested by |options|.
//
// The query is the source-IR ownership boundary for runtime provisioning. It
// accounts for authored sanitizer operations that execute independently of
// insertion options, including materializable semantic assertions. The module
// is walked once and no result retains module storage.
iree_status_t loom_sanitizer_runtime_requirements_query(
    const loom_module_t* module, const loom_sanitizer_options_t* options,
    loom_sanitizer_runtime_requirements_t* out_requirements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_RUNTIME_REQUIREMENTS_H_
