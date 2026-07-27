// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation constraints for materialized spill traffic.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_TRAFFIC_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_TRAFFIC_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |value_id| must remain in register storage because it is
// part of materialized spill/reload traffic.
bool loom_low_allocation_spill_traffic_value_requires_register_location(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true when |interval| must remain in register storage because its
// value is part of materialized spill/reload traffic.
bool loom_low_allocation_spill_traffic_interval_requires_register_location(
    const loom_module_t* module, const loom_liveness_interval_t* interval);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_TRAFFIC_H_
