// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CHECK_HARDWARE_LOOPS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CHECK_HARDWARE_LOOPS_H_

#include "loom/tools/loom-check/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Test-only AIE2P hardware-loop artifact inspection.
extern const loom_check_provider_t loom_aie2p_hardware_loops_check_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_CHECK_HARDWARE_LOOPS_H_
