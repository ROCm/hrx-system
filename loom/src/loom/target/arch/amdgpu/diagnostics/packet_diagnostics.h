// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU packet diagnostics for authored target-low IR.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PACKET_DIAGNOSTICS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PACKET_DIAGNOSTICS_H_

#include "loom/target/low_packet_diagnostics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-owned provider for AMDGPU low packet selection diagnostics.
extern const loom_target_low_packet_diagnostic_provider_t
    loom_amdgpu_low_packet_diagnostic_provider_storage;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PACKET_DIAGNOSTICS_H_
