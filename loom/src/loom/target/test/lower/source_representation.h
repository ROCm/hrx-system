// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical source-representation tables used by the synthetic test target.

#ifndef LOOM_TARGET_TEST_LOWER_SOURCE_REPRESENTATION_H_
#define LOOM_TARGET_TEST_LOWER_SOURCE_REPRESENTATION_H_

#include "loom/codegen/low/source_representation.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY \
  UINT64_C(0x38db07f2e90f71a1)
#define LOOM_TEST_LOW_SOURCE_REPRESENTATION_ALTERNATE_KEY \
  UINT64_C(0x60e8dd0c1081b2b7)

#define LOOM_TEST_LOW_SOURCE_REPRESENTATION_ADDI_GROUP_KEY \
  UINT64_C(0x2425afed010c4043)
#define LOOM_TEST_LOW_SOURCE_REPRESENTATION_CAST_GROUP_KEY \
  UINT64_C(0x73b2855b605c7201)

enum loom_test_low_source_representation_configuration_flag_bits_e {
  // Makes noncanonical candidates pass their target capability predicate.
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_ENABLE_ALTERNATE = (uint32_t)1u << 0,
  // Returns an explicitly empty i32 value domain for conflict coverage.
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_EMPTY_I32_DOMAIN = (uint32_t)1u << 1,
};
typedef uint32_t loom_test_low_source_representation_configuration_flags_t;

typedef struct loom_test_low_source_representation_configuration_t {
  // Synthetic target capability and domain-control flags.
  loom_test_low_source_representation_configuration_flags_t flags;
} loom_test_low_source_representation_configuration_t;

typedef enum loom_test_low_source_representation_realization_e {
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_CANONICAL = 0,
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_ALTERNATE = 1,
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_CAST_CANONICAL = 2,
  LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_CAST_ALTERNATE = 3,
} loom_test_low_source_representation_realization_t;

typedef struct loom_test_low_source_representation_target_data_t {
  // Synthetic target realization retained by the common planner.
  loom_test_low_source_representation_realization_t realization;
} loom_test_low_source_representation_target_data_t;

extern const loom_low_source_representation_provider_t
    loom_test_low_source_representation_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_LOWER_SOURCE_REPRESENTATION_H_
