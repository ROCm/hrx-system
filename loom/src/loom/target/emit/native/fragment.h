// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared native low-function fragment emission contracts.
//
// Native fragment emitters consume low functions after target ABI lowering,
// allocation, and spill materialization. This layer checks those API-boundary
// contracts once so target text/binary sinks do not grow their own copies of
// whole-function invariant logic.

#ifndef LOOM_TARGET_EMIT_NATIVE_FRAGMENT_H_
#define LOOM_TARGET_EMIT_NATIVE_FRAGMENT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/packet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates the target-independent native fragment emission contract.
//
// The schedule and allocation tables must describe the same low function, all
// low.return terminators must be ABI-lowered void returns, all allocated
// register values and edge-copy temporaries must have physical locations, and
// allocation must contain no unmaterialized spill plans or spill-slot
// assignments. Target-owned emitters remain responsible for their target
// identity and instruction descriptor contracts.
iree_status_t loom_native_fragment_validate_emission_inputs(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_FRAGMENT_H_
