// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_CONFIGURATION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_CONFIGURATION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps specialized integer constants to physical configuration operands. All
// source calculations finish before lowering; configuration has no arithmetic
// execution or runtime argument ABI.
extern const loom_low_lower_policy_t loom_aie2p_configuration_low_lower_policy;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_CONFIGURATION_H_
