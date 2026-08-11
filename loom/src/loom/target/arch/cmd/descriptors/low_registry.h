// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable command-machine low descriptor registry.

#ifndef LOOM_TARGET_ARCH_CMD_DESCRIPTORS_LOW_REGISTRY_H_
#define LOOM_TARGET_ARCH_CMD_DESCRIPTORS_LOW_REGISTRY_H_

#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes |out_registry| with the portable command-machine descriptor set.
void loom_cmd_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_DESCRIPTORS_LOW_REGISTRY_H_
