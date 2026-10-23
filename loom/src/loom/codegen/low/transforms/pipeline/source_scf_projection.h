// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final source SCF projection owned by source-to-Low lowering.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_SCF_PROJECTION_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_SCF_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_scf_projection_result_t {
  // Number of keyed SSA tables projected to conditional selections.
  uint32_t lookups_projected;
  // Number of multi-way regions projected to nested conditional regions.
  uint32_t switches_projected;
} loom_low_source_scf_projection_result_t;

// Projects source SCF forms not admitted by the selected source-to-Low control
// representation. Keyed value lookups are projected in every mode. Structured
// lowering additionally projects multi-way control to nested conditions; CFG
// lowering consumes scf.switch directly before source-to-Low runs.
iree_status_t loom_low_source_scf_projection_run(
    loom_module_t* module, loom_func_like_t function,
    loom_low_control_flow_lowering_t control_flow_lowering,
    iree_arena_allocator_t* arena,
    loom_low_source_scf_projection_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_SCF_PROJECTION_H_
