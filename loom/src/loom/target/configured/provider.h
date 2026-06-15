// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Configured target provider set selected by //loom/config/target.

#ifndef LOOM_TARGET_CONFIGURED_PROVIDER_H_
#define LOOM_TARGET_CONFIGURED_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the static configured target provider set.
const loom_target_provider_set_t* loom_configured_target_provider_set(void);

// Returns the lazily-composed configured target environment.
const loom_target_environment_t* loom_configured_target_environment(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CONFIGURED_PROVIDER_H_
