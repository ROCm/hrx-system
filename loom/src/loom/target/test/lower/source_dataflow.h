// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical source-value classifications used by the synthetic test target.

#ifndef LOOM_TARGET_TEST_LOWER_SOURCE_DATAFLOW_H_
#define LOOM_TARGET_TEST_LOWER_SOURCE_DATAFLOW_H_

#include "loom/analysis/source_dataflow.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_test_low_source_dataflow_bit_e {
  // Value is derived from a source function entry argument or i32 constant.
  LOOM_TEST_LOW_SOURCE_DATAFLOW_ENTRY_DERIVED = (loom_source_dataflow_bits_t)1u
                                                << 0,
  // Every relevant producer input is entry-derived.
  LOOM_TEST_LOW_SOURCE_DATAFLOW_ALL_ENTRY_DERIVED =
      (loom_source_dataflow_bits_t)1u << 1,
  // Descending test feasibility has rejected the synthetic scalar candidate.
  LOOM_TEST_LOW_SOURCE_DATAFLOW_SCALAR_CANDIDATE_REJECTED =
      (loom_source_dataflow_bits_t)1u << 2,
  // A callable boundary transitively requires the value.
  LOOM_TEST_LOW_SOURCE_DATAFLOW_BOUNDARY_REQUIRED =
      (loom_source_dataflow_bits_t)1u << 3,
};

// Immutable test-target transfer provider.
extern const loom_source_dataflow_provider_t loom_test_low_source_dataflow;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_LOWER_SOURCE_DATAFLOW_H_
