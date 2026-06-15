// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm target-family records.

#ifndef LOOM_TARGET_ARCH_WASM_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_WASM_RECORDS_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_bundle_t loom_wasm_low_target_bundle_core_simd128;

extern const loom_target_bundle_table_t loom_wasm_target_bundles;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_WASM_RECORDS_TARGET_RECORDS_H_
