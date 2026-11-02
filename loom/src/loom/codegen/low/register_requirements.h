// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural register requirements shared by scheduling and allocation.

#ifndef LOOM_CODEGEN_LOW_REGISTER_REQUIREMENTS_H_
#define LOOM_CODEGEN_LOW_REGISTER_REQUIREMENTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reason a value must remain in register storage during spill planning.
enum loom_low_register_requirement_e {
  // The value can be assigned spill storage.
  LOOM_LOW_REGISTER_REQUIREMENT_NONE = 0,
  // The value is a reload result or spill source in already materialized
  // traffic and cannot recursively spill.
  LOOM_LOW_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC = 1,
  // The value is an argument of a nested region whose structured-control ABI
  // requires a register at the region boundary.
  LOOM_LOW_REGISTER_REQUIREMENT_NESTED_REGION_ARGUMENT = 2,
  // No operand use can be replaced with a reload, so spilling cannot shorten
  // the value's lifetime, including any dependent-type observations.
  LOOM_LOW_REGISTER_REQUIREMENT_NO_OPERAND_USES = 3,
};
typedef uint8_t loom_low_register_requirement_t;

// Classifies the current snapshot once, indexed by |value_domain| ordinals.
// Function-region arguments can spill their operand uses while preserving
// metadata references; nested arguments retain their structured-control ABI
// registers. Non-register values have no requirement. A NULL result represents
// a snapshot with no structural requirements and needs no arena storage.
iree_status_t loom_low_register_requirements_build(
    const loom_local_value_domain_t* value_domain,
    const loom_region_t* function_region, iree_arena_allocator_t* arena,
    loom_low_register_requirement_t** out_requirements);

// Returns a retained fact for a value in the snapshot's local domain.
static inline loom_low_register_requirement_t
loom_low_register_requirements_get(
    const loom_low_register_requirement_t* requirements,
    loom_value_ordinal_t value_ordinal) {
  return requirements ? requirements[value_ordinal]
                      : LOOM_LOW_REGISTER_REQUIREMENT_NONE;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REGISTER_REQUIREMENTS_H_
