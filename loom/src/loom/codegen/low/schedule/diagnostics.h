// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostic emission for target-low scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_DIAGNOSTICS_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_DIAGNOSTICS_H_

#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits a retained scheduling failure, or the requested feedback for a
// successful schedule. Construction must have requested the same diagnostic
// flags so optional pressure and candidate evidence is present. This consumes
// only the completed table; it neither schedules nor allocates scratch.
iree_status_t loom_low_schedule_diagnostics_emit(
    const loom_low_schedule_table_t* table,
    loom_low_schedule_diagnostic_flags_t flags,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_DIAGNOSTICS_H_
