// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable command IR plus build-configured target IR providers.

#ifndef LOOM_TARGET_CONFIGURED_IR_PROVIDER_SET_H_
#define LOOM_TARGET_CONFIGURED_IR_PROVIDER_SET_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the immutable configured IR provider set. Providers contribute only
// the dialects, Low descriptors, target facts, and source diagnostics required
// to parse, structurally verify, and print target IR.
const loom_target_provider_set_t* loom_configured_ir_provider_set(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CONFIGURED_IR_PROVIDER_SET_H_
