// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// External AIE2P array binding transfer planning.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_BINDING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_BINDING_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the physical shim DMA transfer that realizes one logical external
// binding view. |source_type| is the full binding-view tile type and
// |record_type| is the trailing tile record transferred through the channel.
iree_status_t loom_aie2p_array_plan_binding_transfer(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_xdna_array_family_t* family, loom_type_t source_type,
    loom_type_t record_type, uint64_t binding_view_byte_offset,
    bool partitioned, uint32_t partition_lane,
    uint32_t logical_record_byte_length, uint32_t logical_record_count,
    const loom_xdna_dma_facts_t* dma_facts,
    loom_aie2p_array_binding_plan_t* binding_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_BINDING_H_
