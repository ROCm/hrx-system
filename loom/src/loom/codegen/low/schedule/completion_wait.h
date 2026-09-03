// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-declared completion-wait cost queries for Low scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_WAIT_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_WAIT_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries the target-provided issue cost for resolving a schedule class's
// wait-counter hazard. Returns false when the class is not counter tracked.
bool loom_low_schedule_class_query_completion_wait(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_class_t* schedule_class, uint16_t* out_wait_cycles);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_WAIT_H_
