// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation constraints for materialized spill traffic.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_TRAFFIC_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_TRAFFIC_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reason a value must remain in register storage during spill planning.
typedef enum loom_low_allocation_spill_register_requirement_e {
  // The value can be assigned spill storage.
  LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NONE = 0,
  // The value is a reload result or spill source in already materialized
  // traffic and cannot recursively spill.
  LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_MATERIALIZED_TRAFFIC = 1,
  // The value is an argument of a nested region whose structured-control ABI
  // requires a register at the region boundary.
  LOOM_LOW_ALLOCATION_SPILL_REGISTER_REQUIREMENT_NESTED_REGION_ARGUMENT = 2,
} loom_low_allocation_spill_register_requirement_t;

// Returns the register requirement for |value_id| under the spill materializer
// owned by |function_region|. Function-region block arguments are supported by
// CFG edge rewriting; nested-region block arguments are not.
loom_low_allocation_spill_register_requirement_t
loom_low_allocation_spill_register_requirement_for_value(
    const loom_module_t* module, const loom_region_t* function_region,
    loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_TRAFFIC_H_
