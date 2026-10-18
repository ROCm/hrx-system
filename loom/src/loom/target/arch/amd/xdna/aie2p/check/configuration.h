// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CHECK_CONFIGURATION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CHECK_CONFIGURATION_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Check-only pass comparing worker analysis across configuration
// materialization.
extern const loom_target_provider_t loom_aie2p_configuration_check_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CHECK_CONFIGURATION_H_
