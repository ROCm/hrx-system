// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMD XDNA AIE2P target provider.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROVIDER_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// AIE2P target dialect, descriptor, and source-to-Low capabilities.
extern const loom_target_provider_t loom_aie2p_target_provider;

// Provider set containing only the AIE2P target provider.
extern const loom_target_provider_set_t loom_aie2p_target_provider_set;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROVIDER_H_
