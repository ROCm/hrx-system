// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_SPIRV_RECORDS_TARGET_RECORDS_H_

#include "loom/target/arch/spirv/features.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target bundle table consumed by the generated spirv.target op.
extern const loom_target_bundle_table_t loom_spirv_target_bundles;

// Vulkan 1.3 SPIR-V target bundle.
extern const loom_target_bundle_t loom_spirv_low_target_bundle_vulkan1_3;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_RECORDS_TARGET_RECORDS_H_
