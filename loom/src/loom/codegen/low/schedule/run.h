// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Schedule execution for target-low function bodies.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_RUN_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_RUN_H_

#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Schedules one target-low function body and writes an arena-owned table. The
// caller must keep |module| semantically immutable and |arena| alive for as
// long as |out_table| is used. This function performs descriptor target
// resolution and liveness analysis; malformed user IR is reported through
// |options->emitter| when provided and otherwise fails loud with status.
iree_status_t loom_low_schedule_function(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_schedule_options_t* options, iree_arena_allocator_t* arena,
    loom_low_schedule_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_RUN_H_
