// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime value realization for finite check.scenario domains.

#ifndef LOOM_TOOLING_TESTBENCH_SCENARIO_VALUES_H_
#define LOOM_TOOLING_TESTBENCH_SCENARIO_VALUES_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/invocation.h"
#include "loom/tooling/testbench/value_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_testbench_scenario_value_flags_t;
enum loom_testbench_scenario_value_flag_bits_e {
  // The value tables contain one complete realized coordinate.
  LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED = 1u << 0,
  // The trial owns an oracle table in addition to its target table.
  LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_HAS_ORACLE = 1u << 1,
};

typedef uint8_t loom_testbench_scenario_trial_realization_t;
enum loom_testbench_scenario_trial_realization_e {
  // Materializes only the target value graph.
  LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_ONLY = 0u,
  // Materializes independent target and oracle value graphs.
  LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_AND_ORACLE = 1u,
};

// Exact coordinate needed to replay one scenario trial.
typedef struct loom_testbench_scenario_trial_identity_t {
  // Campaign entropy root supplied by the runner.
  loom_testbench_entropy_t entropy_root;
  // Concrete scenario configuration ordinal.
  iree_host_size_t configuration_ordinal;
  // Source-order trial domain ordinal within the scenario.
  iree_host_size_t trial_index;
  // Concrete ordinal within the selected trial domain.
  iree_host_size_t trial_ordinal;
} loom_testbench_scenario_trial_identity_t;

// Values produced by one scenario configuration recipe.
//
// Configuration realization happens before subject preparation. The selected
// configuration values may therefore control declared specializations while
// trial-local values cannot reach either compiler.
typedef struct loom_testbench_scenario_configuration_values_t {
  // Scenario whose configuration recipe owns |values|.
  const loom_testbench_scenario_plan_t* scenario_plan;
  // Materialization state flags.
  loom_testbench_scenario_value_flags_t flags;
  // Campaign entropy root used to derive |scenario_entropy|.
  loom_testbench_entropy_t entropy_root;
  // Concrete configuration ordinal represented by |values|.
  iree_host_size_t configuration_ordinal;
  // Scenario-scoped entropy identity captured by configured scenarios.
  loom_testbench_entropy_t scenario_entropy;
  // Immutable configuration values shared into each trial realization.
  loom_testbench_value_table_t values;
} loom_testbench_scenario_configuration_values_t;

// Independent target and oracle values for one scenario trial.
typedef struct loom_testbench_scenario_trial_values_t {
  // Scenario that owns |trial_plan|.
  const loom_testbench_scenario_plan_t* scenario_plan;
  // Trial domain whose recipe owns the value tables.
  const loom_testbench_trial_plan_t* trial_plan;
  // Materialization and oracle-presence flags.
  loom_testbench_scenario_value_flags_t flags;
  // Exact identity of the currently materialized trial.
  loom_testbench_scenario_trial_identity_t identity;
  // Target-local realization consumed by the product under test.
  loom_testbench_value_table_t target;
  // Oracle-local realization consumed by a comparison oracle.
  loom_testbench_value_table_t oracle;
} loom_testbench_scenario_trial_values_t;

// Initializes reusable storage for one scenario's configuration recipe.
iree_status_t loom_testbench_scenario_configuration_values_initialize(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_t* scenario_plan,
    iree_allocator_t host_allocator,
    loom_testbench_scenario_configuration_values_t* out_values);

// Releases storage owned by |values|.
void loom_testbench_scenario_configuration_values_deinitialize(
    loom_testbench_scenario_configuration_values_t* values);

// Materializes one configuration before subject preparation begins.
iree_status_t loom_testbench_scenario_configuration_values_materialize(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_entropy_t entropy_root,
    iree_host_size_t configuration_ordinal,
    loom_testbench_scenario_configuration_values_t* values);

// Materializes one configuration in an explicit entropy domain.
//
// Independently named scenarios can share |entropy_domain| when they must
// realize identical generated values, such as candidates in one controlled
// benchmark comparison. The ordinary materializer uses the scenario name.
iree_status_t
loom_testbench_scenario_configuration_values_materialize_in_entropy_domain(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_entropy_t entropy_root, iree_string_view_t entropy_domain,
    iree_host_size_t configuration_ordinal,
    loom_testbench_scenario_configuration_values_t* values);

// Initializes reusable target and optional oracle storage for |trial_index|.
iree_status_t loom_testbench_scenario_trial_values_initialize(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_t* scenario_plan,
    iree_host_size_t trial_index,
    loom_testbench_scenario_trial_realization_t realization,
    iree_allocator_t host_allocator,
    loom_testbench_scenario_trial_values_t* out_values);

// Releases storage owned by |values|.
void loom_testbench_scenario_trial_values_deinitialize(
    loom_testbench_scenario_trial_values_t* values);

// Materializes one trial after subject preparation has completed.
//
// Target and oracle recipes start from the same immutable configuration and
// entropy identity but allocate all mutable shaped values independently.
iree_status_t loom_testbench_scenario_trial_values_materialize(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_invocation_executor_t* generator_executor,
    const loom_testbench_scenario_configuration_values_t* configuration,
    iree_host_size_t trial_ordinal,
    loom_testbench_scenario_trial_values_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_SCENARIO_VALUES_H_
