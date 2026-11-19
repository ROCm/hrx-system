// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CODE_BANKS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CODE_BANKS_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_code_catalog_demand_t {
  // Memory tile providing the worker's code delivery.
  uint16_t column;
  // Maximum contiguous descriptor slots needed to compile this catalog.
  // Zero for workers executing entirely from resident program memory.
  iree_host_size_t descriptor_count;
} loom_aie2p_code_catalog_demand_t;

// Finds a placement order for catalogs competing for the two SRAM DMA banks.
// Ordinals remain indices into the original configuration. A NULL result means
// that no column benefits from a different order. Native placement still owns
// exact descriptor counts, queue admission, and physical routes.
iree_status_t loom_aie2p_code_banks_order(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_code_catalog_demand_t* demands, iree_host_size_t count,
    iree_arena_allocator_t* arena, iree_host_size_t** out_order);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CODE_BANKS_H_
