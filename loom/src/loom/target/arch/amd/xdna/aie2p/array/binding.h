// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P DMA transfer planning from endpoint storage geometry.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_BINDING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_BINDING_H_

#include "iree/base/api.h"
#include "loom/analysis/storage_geometry.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Admits a DMA transfer for retained endpoint storage geometry and tile facts.
// Logical topology validation has proved the trailing record rank and sequence
// count. This consumes exact coordinates; it does not resolve source IR or
// facts. Singleton axes consume no address dimension; their strides have no
// effect on the transfer. Returns false with a static reason when native DMA
// cannot express the layout.
bool loom_aie2p_array_resolve_binding_transfer(
    const loom_storage_geometry_t* geometry, uint8_t record_rank,
    const loom_xdna_array_family_t* family, uint64_t binding_view_byte_offset,
    bool partitioned, uint32_t partition_lane,
    uint32_t logical_record_byte_length, uint32_t logical_record_count,
    const loom_xdna_dma_facts_t* dma_facts,
    loom_aie2p_array_binding_plan_t* binding_plan,
    iree_string_view_t* out_reason);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_BINDING_H_
