// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/pipeline/pass_environment.h"

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/facts.h"

static bool loom_low_pass_capability_satisfies_requirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  const loom_low_pass_capability_t* low_capability =
      (const loom_low_pass_capability_t*)capability;
  if (iree_string_view_equal(
          requirement,
          IREE_SV(LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY))) {
    return low_capability->descriptor_registry != NULL;
  }
  if (iree_string_view_equal(
          requirement,
          IREE_SV(
              LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_LOWER_POLICY_REGISTRY))) {
    return low_capability->lower_policy_registry != NULL;
  }
  if (iree_string_view_equal(
          requirement,
          IREE_SV(LOOM_LOW_PASS_REQUIREMENT_TARGET_LEGALIZER_REGISTRY))) {
    return low_capability->legalizer_registry != NULL;
  }
  return false;
}

const loom_pass_environment_capability_type_t loom_low_pass_capability_type = {
    .name = IREE_SVL("low"),
    .satisfies_requirement = loom_low_pass_capability_satisfies_requirement,
};

loom_low_pass_capability_t loom_low_pass_capability_make(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_lower_policy_registry_t* lower_policy_registry,
    const loom_target_low_legality_provider_list_t* legality_provider_list,
    const loom_target_legalizer_registry_t* legalizer_registry,
    loom_target_compile_report_t* compile_report) {
  return (loom_low_pass_capability_t){
      .base =
          {
              .type = &loom_low_pass_capability_type,
          },
      .descriptor_registry = descriptor_registry,
      .lower_policy_registry = lower_policy_registry,
      .legality_provider_list = legality_provider_list,
      .legalizer_registry = legalizer_registry,
      .compile_report = compile_report,
  };
}

const loom_low_pass_capability_t* loom_low_pass_capability_from_environment(
    const loom_pass_environment_t* environment) {
  return (const loom_low_pass_capability_t*)loom_pass_environment_lookup(
      environment, &loom_low_pass_capability_type);
}

const loom_low_pass_capability_t* loom_low_pass_capability_from_pass(
    const loom_pass_t* pass) {
  return pass && pass->environment
             ? loom_low_pass_capability_from_environment(pass->environment)
             : NULL;
}

const loom_low_descriptor_registry_t*
loom_low_pass_capability_descriptor_registry(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->descriptor_registry : NULL;
}

const loom_low_lower_policy_registry_t*
loom_low_pass_capability_lower_policy_registry(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->lower_policy_registry : NULL;
}

const loom_target_low_legality_provider_list_t*
loom_low_pass_capability_legality_provider_list(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->legality_provider_list : NULL;
}

const loom_target_legalizer_registry_t*
loom_low_pass_capability_legalizer_registry(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->legalizer_registry : NULL;
}

loom_target_compile_report_t* loom_low_pass_capability_compile_report(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->compile_report : NULL;
}

static const loom_low_lower_policy_t*
loom_low_pass_capability_lower_policy_for_target_facts(
    const loom_low_pass_capability_t* capability,
    const loom_target_facts_t* target_facts) {
  if (capability == NULL || capability->lower_policy_registry == NULL ||
      target_facts == NULL) {
    return NULL;
  }
  return loom_low_lower_policy_registry_lookup_for_bundle(
      capability->lower_policy_registry,
      loom_target_facts_bundle(target_facts));
}

const loom_target_vector_packet_policy_t*
loom_low_pass_capability_vector_packet_policy(
    const loom_low_pass_capability_t* capability,
    const loom_target_facts_t* target_facts) {
  const loom_low_lower_policy_t* policy =
      loom_low_pass_capability_lower_policy_for_target_facts(capability,
                                                             target_facts);
  return policy != NULL ? policy->vector_packet_policy : NULL;
}

bool loom_low_pass_capability_source_vector_carrier_supported(
    const loom_low_pass_capability_t* capability,
    const loom_target_facts_t* target_facts, const loom_module_t* module,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_type_t type) {
  const loom_low_lower_policy_t* policy =
      loom_low_pass_capability_lower_policy_for_target_facts(capability,
                                                             target_facts);
  if (policy == NULL) {
    return false;
  }
  const loom_target_vector_carrier_supported_callback_t callback =
      policy->source_vector_carrier_supported;
  return callback.fn == NULL || callback.fn(callback.user_data, target_facts,
                                            module, fact_table, value_id, type);
}

bool loom_low_pass_capability_source_function_vector_carrier_supported(
    const loom_low_pass_capability_t* capability,
    const loom_target_facts_t* target_facts, const loom_module_t* module,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_type_t type) {
  const loom_low_lower_policy_t* policy =
      loom_low_pass_capability_lower_policy_for_target_facts(capability,
                                                             target_facts);
  if (policy == NULL) {
    return false;
  }
  loom_target_vector_carrier_supported_callback_t callback =
      policy->source_function_vector_carrier_supported;
  if (callback.fn == NULL) {
    callback = policy->source_vector_carrier_supported;
  }
  return callback.fn == NULL || callback.fn(callback.user_data, target_facts,
                                            module, fact_table, value_id, type);
}
