// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Memory visibility obligations over an existing portable command schedule.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_DEPENDENCIES_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_DEPENDENCIES_H_

#include "loom/analysis/storage_access.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/target/arch/cmd/lower/transients.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Combines canonical argument effects and alias facts with completed physical
// placement. No source body is opened or interpreted here. Timestamp storage is
// validated through the same placement index, before any Low mutation. Unknown
// callees/accesses remain global; a storage-wide formal cannot be narrowed to
// its actual view. Temporary interval/writer state is retired at return.
//
// |kernel_accesses| is indexed by plan-wide requirement, and |requirements|
// maps root-local entry ordinals to those requirements. NULL summaries denote
// opaque entries. |binding_values| follows the root signature. Single-wave
// roots validate timestamps without requesting kernel summaries or a plan.
iree_status_t loom_cmd_dependency_plan_build(
    loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_value_id_t* binding_values, const loom_cmd_lower_plan_t* lower,
    const loom_storage_access_summary_t* const* kernel_accesses,
    const uint32_t* requirements, loom_cmd_transient_requirement_t* transient,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_valid, loom_cmd_dependency_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_DEPENDENCIES_H_
