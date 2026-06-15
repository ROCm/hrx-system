// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V target-local runtime profile payload.
//
// Core target selection passes this structure opaquely as target_data. Only the
// SPIR-V target package interprets it.

#ifndef LOOM_TARGET_ARCH_SPIRV_PROFILE_H_
#define LOOM_TARGET_ARCH_SPIRV_PROFILE_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/cooperative_properties.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_target_profile_t {
  // Cooperative operation rows selected for the active target, or NULL to use
  // the static modeled rows implied by the target feature set.
  const loom_spirv_cooperative_property_set_t* cooperative_properties;
} loom_spirv_target_profile_t;

// Returns the SPIR-V target profile stored in |target_data|, or NULL.
static inline const loom_spirv_target_profile_t*
loom_spirv_target_profile_from_data(const void* target_data) {
  return (const loom_spirv_target_profile_t*)target_data;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_PROFILE_H_
