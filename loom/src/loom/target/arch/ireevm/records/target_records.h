// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM target-family records.

#ifndef LOOM_TARGET_ARCH_IREEVM_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_IREEVM_RECORDS_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_bundle_t loom_ireevm_low_target_bundle_core;

extern const loom_target_bundle_table_t loom_ireevm_target_bundles;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_IREEVM_RECORDS_TARGET_RECORDS_H_
