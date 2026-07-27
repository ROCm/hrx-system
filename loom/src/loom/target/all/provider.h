// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Developer convenience provider set that links every current target provider.
// Core tools should depend on explicit provider sets instead of this package.

#ifndef LOOM_TARGET_ALL_PROVIDER_H_
#define LOOM_TARGET_ALL_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the static all-target provider set.
const loom_target_provider_set_t* loom_all_target_provider_set(void);

// Returns the lazily-composed all-target environment.
const loom_target_environment_t* loom_all_target_environment(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ALL_PROVIDER_H_
