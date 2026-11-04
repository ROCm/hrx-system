// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_ANALYSIS_PROVIDERS_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_ANALYSIS_PROVIDERS_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints shared transport proofs for channels backed by caller buffers.
// These proofs describe record flow, not physical storage or device admission.
extern const loom_check_emit_provider_t
    loom_check_test_pipeline_transport_provider;

// Prints passive liveness and storage-interference observations for authored
// IR without changing the program being analyzed.
extern const loom_check_emit_provider_t
    loom_check_test_analysis_observation_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_ANALYSIS_PROVIDERS_H_
