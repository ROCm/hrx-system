// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_OWNERSHIP_H_
#define LOOM_CODEGEN_LOW_OWNERSHIP_H_

#include "loom/analysis/ownership.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_descriptor_registry_t loom_low_descriptor_registry_t;

// Binds verified Low instruction ownership to the shared lifetime analysis.
// The registry and its immutable descriptor tables outlive the analysis.
loom_ownership_effect_resolver_t loom_low_ownership_effect_resolver(
    const loom_low_descriptor_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_OWNERSHIP_H_
