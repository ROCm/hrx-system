// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P physical array-plan compile reporting.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARRAY_REPORT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARRAY_REPORT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Records one fully compiled AIE2P spatial realization.
iree_status_t loom_aie2p_array_report_record(
    const loom_module_t* module, iree_string_view_t root_name,
    const loom_aie2p_array_plan_t* plan, const loom_aie2p_xdna_tile_t* tiles,
    loom_target_compile_report_t* report,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARRAY_REPORT_H_
