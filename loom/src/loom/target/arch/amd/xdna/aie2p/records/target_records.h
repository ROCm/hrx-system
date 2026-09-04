// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMD XDNA AIE2P compute-tile target records.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_RECORDS_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target bundle rows selected by aie2p.target.
extern const loom_target_bundle_table_t loom_aie2p_target_bundles;

// Returns the AIE2P logical-array program target bundle.
const loom_target_bundle_t* loom_aie2p_array_target_bundle(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_RECORDS_TARGET_RECORDS_H_
