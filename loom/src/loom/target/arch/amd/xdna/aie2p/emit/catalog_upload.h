// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CATALOG_UPLOAD_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CATALOG_UPLOAD_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/program_catalog.h"

#ifdef __cplusplus
extern "C" {
#endif

// Temporary initialization before application routes and DMA engines take over.
typedef struct loom_aie2p_catalog_upload_t {
  // Byte ranges and transfers for the loader; patch indices include its prefix.
  loom_aie2p_program_upload_t program;
  // True for each physical column installed by this loader.
  const bool* columns;
} loom_aie2p_catalog_upload_t;

// Groups placed catalogs by destination and builds an independently audited
// shim-to-SRAM circuit per selected column. All receivers start before senders.
// Positive arrival credits and both engines' retirement precede reuse by the
// application. NULL means the admitted catalogs use inline firmware writes.
iree_status_t loom_aie2p_catalog_upload_build(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, iree_host_size_t first_record,
    iree_arena_allocator_t* arena, loom_aie2p_catalog_upload_t** out_upload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_CATALOG_UPLOAD_H_
