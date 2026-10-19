// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/values.h"

#include <string.h>

static iree_status_t loom_testbench_scenario_assign_ordinal(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    iree_host_size_t ordinal) {
  loom_testbench_value_t value = {
      .kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR,
      .scalar =
          {
              .kind = IREE_TOOLING_VALUE_KIND_I64,
              .storage.i64 = (int64_t)ordinal,
          },
  };
  return loom_testbench_value_table_assign_move(table, value_id, &value);
}

static iree_status_t loom_testbench_scenario_assign_entropy(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    loom_testbench_entropy_t entropy) {
  loom_testbench_value_t value = {
      .kind = LOOM_TESTBENCH_VALUE_KIND_ENTROPY,
      .entropy = entropy,
  };
  return loom_testbench_value_table_assign_move(table, value_id, &value);
}

iree_status_t loom_testbench_scenario_configuration_values_initialize(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_t* scenario_plan,
    iree_allocator_t host_allocator,
    loom_testbench_scenario_configuration_values_t* out_values) {
  memset(out_values, 0, sizeof(*out_values));
  out_values->scenario_plan = scenario_plan;
  iree_status_t status =
      loom_testbench_value_table_initialize_scenario_configuration(
          module, scenario_plan, host_allocator, &out_values->values);
  if (!iree_status_is_ok(status)) {
    loom_testbench_scenario_configuration_values_deinitialize(out_values);
  }
  return status;
}

void loom_testbench_scenario_configuration_values_deinitialize(
    loom_testbench_scenario_configuration_values_t* values) {
  if (!values) {
    return;
  }
  loom_testbench_value_table_deinitialize(&values->values);
  memset(values, 0, sizeof(*values));
}

iree_status_t
loom_testbench_scenario_configuration_values_materialize_in_entropy_domain(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_entropy_t entropy_root, iree_string_view_t entropy_domain,
    iree_host_size_t configuration_ordinal,
    loom_testbench_scenario_configuration_values_t* values) {
  const loom_testbench_scenario_plan_t* scenario = values->scenario_plan;
  if (configuration_ordinal >= scenario->configuration_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "configuration ordinal %zu exceeds scenario configuration count %zu",
        configuration_ordinal, scenario->configuration_count);
  }

  values->flags &= ~LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED;
  loom_testbench_value_table_reset(&values->values);
  values->entropy_root = entropy_root;
  values->configuration_ordinal = configuration_ordinal;
  values->scenario_entropy =
      loom_testbench_entropy_fork(entropy_root, entropy_domain);

  iree_status_t status = iree_ok_status();
  if (scenario->is_configured) {
    status = loom_testbench_scenario_assign_ordinal(
        &values->values, scenario->configuration_ordinal_value_id,
        configuration_ordinal);
    if (iree_status_is_ok(status)) {
      status = loom_testbench_scenario_assign_entropy(
          &values->values, scenario->configuration_entropy_value_id,
          values->scenario_entropy);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_materialize_value_sources(
        options, scenario->configuration_sources,
        scenario->configuration_source_count, &values->values);
  }
  if (iree_status_is_ok(status)) {
    values->flags |= LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED;
  } else {
    loom_testbench_value_table_reset(&values->values);
  }
  return status;
}

iree_status_t loom_testbench_scenario_configuration_values_materialize(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_entropy_t entropy_root,
    iree_host_size_t configuration_ordinal,
    loom_testbench_scenario_configuration_values_t* values) {
  return loom_testbench_scenario_configuration_values_materialize_in_entropy_domain(
      options, entropy_root, values->scenario_plan->name, configuration_ordinal,
      values);
}

iree_status_t loom_testbench_scenario_trial_values_initialize(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_t* scenario_plan,
    iree_host_size_t trial_index,
    loom_testbench_scenario_trial_realization_t realization,
    iree_allocator_t host_allocator,
    loom_testbench_scenario_trial_values_t* out_values) {
  memset(out_values, 0, sizeof(*out_values));
  if (trial_index >= scenario_plan->trial_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "trial index %zu exceeds scenario trial domain count %zu", trial_index,
        scenario_plan->trial_count);
  }
  if (realization != LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_ONLY &&
      realization !=
          LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_AND_ORACLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid scenario trial realization %u",
                            (unsigned)realization);
  }

  out_values->scenario_plan = scenario_plan;
  out_values->trial_plan = &scenario_plan->trials[trial_index];
  out_values->identity.trial_index = trial_index;
  iree_status_t status = loom_testbench_value_table_initialize_scenario_trial(
      module, scenario_plan, out_values->trial_plan, host_allocator,
      &out_values->target);
  if (iree_status_is_ok(status) &&
      realization ==
          LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_AND_ORACLE) {
    if (out_values->trial_plan->action.kind !=
        LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
      loom_testbench_scenario_trial_values_deinitialize(out_values);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "only check.compare trials admit target and oracle realization");
    }
    status = loom_testbench_value_table_initialize_scenario_trial(
        module, scenario_plan, out_values->trial_plan, host_allocator,
        &out_values->oracle);
    if (iree_status_is_ok(status)) {
      out_values->flags |= LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_HAS_ORACLE;
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_scenario_trial_values_deinitialize(out_values);
  }
  return status;
}

void loom_testbench_scenario_trial_values_deinitialize(
    loom_testbench_scenario_trial_values_t* values) {
  if (!values) {
    return;
  }
  loom_testbench_value_table_deinitialize(&values->oracle);
  loom_testbench_value_table_deinitialize(&values->target);
  memset(values, 0, sizeof(*values));
}

static iree_status_t loom_testbench_scenario_copy_configuration_values(
    const loom_testbench_value_table_t* source,
    loom_testbench_value_table_t* target) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t slot_index = 0;
       iree_status_is_ok(status) && slot_index < source->slot_count;
       ++slot_index) {
    const loom_testbench_value_slot_t* source_slot = &source->slots[slot_index];
    if (!iree_any_bit_set(source_slot->flags,
                          LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED)) {
      continue;
    }
    IREE_ASSERT(!loom_testbench_value_is_buffer(&source_slot->value),
                "scenario configuration values must be immutable");
    loom_testbench_value_t value = {0};
    loom_testbench_value_retain(&source_slot->value, &value);
    status = loom_testbench_value_table_assign_move(
        target, source_slot->value_id, &value);
    if (!iree_status_is_ok(status)) {
      loom_testbench_value_deinitialize(&value);
    }
  }
  return status;
}

static loom_testbench_entropy_t loom_testbench_scenario_trial_entropy(
    const loom_testbench_scenario_configuration_values_t* configuration,
    iree_host_size_t trial_index, iree_host_size_t trial_ordinal) {
  loom_testbench_entropy_t entropy = loom_testbench_entropy_fork(
      configuration->scenario_entropy, IREE_SV("loom.check.trial"));
  entropy = loom_testbench_entropy_at(
      entropy, (uint64_t)configuration->configuration_ordinal);
  entropy = loom_testbench_entropy_at(entropy, (uint64_t)trial_index);
  return loom_testbench_entropy_at(entropy, (uint64_t)trial_ordinal);
}

static iree_status_t loom_testbench_scenario_materialize_trial_table(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_invocation_executor_t* generator_executor,
    const loom_testbench_scenario_configuration_values_t* configuration,
    const loom_testbench_trial_plan_t* trial,
    loom_testbench_entropy_t trial_entropy, iree_host_size_t trial_ordinal,
    loom_testbench_value_table_t* table) {
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_copy_configuration_values(
      &configuration->values, table));
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_assign_ordinal(
      table, trial->ordinal_value_id, trial_ordinal));
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_assign_entropy(
      table, trial->entropy_value_id, trial_entropy));
  if (generator_executor != NULL) {
    loom_testbench_invocation_executor_reset_issues(generator_executor);
  }
  iree_host_size_t generator_index = 0;
  for (iree_host_size_t step_index = 0; step_index < trial->recipe_step_count;
       ++step_index) {
    const loom_testbench_trial_recipe_step_t* step =
        &trial->recipe_steps[step_index];
    if (step->kind == LOOM_TESTBENCH_TRIAL_RECIPE_STEP_VALUE_SOURCE) {
      IREE_RETURN_IF_ERROR(loom_testbench_materialize_value_sources(
          options, &step->value_source, 1, table));
      continue;
    }
    if (step->kind == LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR) {
      if (generator_executor == NULL || generator_executor->schedule == NULL ||
          generator_index >= generator_executor->schedule->invocation_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "scenario generator invocation was not prepared");
      }
      IREE_RETURN_IF_ERROR(loom_testbench_run_prepared_invocation(
          generator_executor,
          &generator_executor->schedule->invocations[generator_index++],
          table));
      if (generator_executor->issue_count != 0) {
        const loom_testbench_sample_issue_t* issue =
            &generator_executor->issues[0];
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "scenario generator failed in %.*s (%.*s): %.*s",
            (int)issue->stage.size, issue->stage.data, (int)issue->kind.size,
            issue->kind.data, (int)issue->message.size, issue->message.data);
      }
      continue;
    }
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "invalid scenario recipe step kind %u",
                            (unsigned)step->kind);
  }
  IREE_ASSERT(generator_executor == NULL ||
              generator_index ==
                  generator_executor->schedule->invocation_count);
  return iree_ok_status();
}

iree_status_t loom_testbench_scenario_trial_values_materialize(
    const loom_testbench_value_materializer_options_t* options,
    loom_testbench_invocation_executor_t* generator_executor,
    const loom_testbench_scenario_configuration_values_t* configuration,
    iree_host_size_t trial_ordinal,
    loom_testbench_scenario_trial_values_t* values) {
  const loom_testbench_trial_plan_t* trial = values->trial_plan;
  if (!iree_any_bit_set(configuration->flags,
                        LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "scenario configuration is not materialized");
  }
  if (configuration->scenario_plan != values->scenario_plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scenario configuration and trial belong to different scenarios");
  }
  if (trial_ordinal >= trial->trial_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trial ordinal %zu exceeds trial domain count %zu",
                            trial_ordinal, trial->trial_count);
  }

  values->flags &= ~LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED;
  loom_testbench_value_table_reset(&values->target);
  loom_testbench_value_table_reset(&values->oracle);
  values->identity.entropy_root = configuration->entropy_root;
  values->identity.configuration_ordinal = configuration->configuration_ordinal;
  values->identity.trial_ordinal = trial_ordinal;
  const loom_testbench_entropy_t trial_entropy =
      loom_testbench_scenario_trial_entropy(
          configuration, values->identity.trial_index, trial_ordinal);

  iree_status_t status = loom_testbench_scenario_materialize_trial_table(
      options, generator_executor, configuration, trial, trial_entropy,
      trial_ordinal, &values->target);
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(values->flags,
                       LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_HAS_ORACLE)) {
    status = loom_testbench_scenario_materialize_trial_table(
        options, generator_executor, configuration, trial, trial_entropy,
        trial_ordinal, &values->oracle);
  }
  if (iree_status_is_ok(status)) {
    values->flags |= LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED;
  } else {
    loom_testbench_value_table_reset(&values->target);
    loom_testbench_value_table_reset(&values->oracle);
  }
  return status;
}
