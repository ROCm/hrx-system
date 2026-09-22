// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_TABLE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_TABLE_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Expands profitable register lookups into packed bit tests and selection
// trees. Uniform table-index leaves retain native indexed-broadcast selection.
// Returns false through |out_rewritten| when scalar lanes are cheaper or the
// source types do not fit the native packed comparison and selection carriers.
iree_status_t loom_aie2p_table_lookup_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_TABLE_H_
