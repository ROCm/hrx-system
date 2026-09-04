// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// AIE2P source target-legalization provider.
extern const loom_target_legalizer_provider_t
    loom_aie2p_target_legalizer_provider_storage;

const loom_target_legalizer_provider_t* loom_aie2p_target_legalizer_provider(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_H_
