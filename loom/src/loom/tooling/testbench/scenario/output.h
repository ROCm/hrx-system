// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_TESTBENCH_SCENARIO_OUTPUT_H_
#define LOOM_TOOLING_TESTBENCH_SCENARIO_OUTPUT_H_

#include "loom/tooling/testbench/scenario/values.h"

#ifdef __cplusplus
extern "C" {
#endif

// Publishes selected completed observations, one value at a time. Paths include
// the scenario, configuration, trial domain, trial ordinal, and realization.
// No payload is copied or retained beyond the call. A failed output fails the
// trial; ON_FAILURE writes are omitted when |trial_failed| is false.
iree_status_t loom_testbench_scenario_write_files(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_scenario_trial_values_t* values, bool trial_failed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_SCENARIO_OUTPUT_H_
