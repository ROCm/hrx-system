// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable SPIR-V target facts.

#ifndef LOOM_TARGET_ARCH_SPIRV_FACTS_H_
#define LOOM_TARGET_ARCH_SPIRV_FACTS_H_

#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// SPIR-V target row selected by spirv.target and structured profiles.
typedef uint8_t loom_spirv_target_kind_t;
enum loom_spirv_target_kind_e {
  // Unknown or absent SPIR-V target row.
  LOOM_SPIRV_TARGET_KIND_UNKNOWN = 0,
  // Vulkan 1.3 logical SPIR-V module row.
  LOOM_SPIRV_TARGET_KIND_VULKAN1_3 = 1,
};

typedef struct loom_spirv_target_facts_t {
  // Target-neutral facts shared by all target families.
  loom_target_facts_t base;

  // Cooperative operation rows available to target-local lowering.
  loom_spirv_cooperative_property_set_t cooperative_properties;
} loom_spirv_target_facts_t;

// Static fact type used by SPIR-V target projection and structured profiles.
extern const loom_target_fact_type_t loom_spirv_target_fact_type;

// Returns |facts| as SPIR-V facts, or NULL for another target family.
static inline const loom_spirv_target_facts_t* loom_spirv_target_facts_cast(
    const loom_target_facts_t* facts) {
  return facts != NULL && facts->fact_type == &loom_spirv_target_fact_type
             ? (const loom_spirv_target_facts_t*)facts
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_FACTS_H_
