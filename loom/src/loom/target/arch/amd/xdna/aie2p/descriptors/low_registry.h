// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P target-low registry package.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_LOW_REGISTRY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_LOW_REGISTRY_H_

#include "loom/codegen/low/schedule/types.h"
#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the AIE2P-only Low descriptor registry.
void loom_aie2p_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

// Returns the AIE2P native schedule models for structural Low operations.
loom_low_schedule_structural_model_list_t
loom_aie2p_low_structural_schedule_models(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_LOW_REGISTRY_H_
