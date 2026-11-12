// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Synthetic target records for compiler infrastructure tests.

#ifndef LOOM_TARGET_TEST_TARGET_RECORDS_H_
#define LOOM_TARGET_TEST_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Synthetic target rows, including a device whose workers use low_core.
typedef enum loom_test_target_kind_e {
  LOOM_TEST_TARGET_KIND_LOW_CORE = 1,
  LOOM_TEST_TARGET_KIND_QUIRKY = 2,
  LOOM_TEST_TARGET_KIND_DEVICE = 3,
  LOOM_TEST_TARGET_KIND_LOW_LEAF = 4,
  LOOM_TEST_TARGET_KIND_COUNT_ = 5,
} loom_test_target_kind_t;

// Typed fact identity shared by synthetic test target profiles and IR records.
extern const loom_target_fact_type_t loom_test_target_fact_type;

extern const loom_target_bundle_table_t loom_test_target_bundles;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_TARGET_RECORDS_H_
