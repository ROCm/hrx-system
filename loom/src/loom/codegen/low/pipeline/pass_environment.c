// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/pipeline/pass_environment.h"

#include "loom/codegen/low/pipeline/pass_requirements.h"

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
    const loom_target_legalizer_provider_list_t* legalizer_provider_list,
    loom_target_compile_report_t* compile_report) {
  return (loom_low_pass_capability_t){
      .base =
          {
              .type = &loom_low_pass_capability_type,
          },
      .descriptor_registry = descriptor_registry,
      .lower_policy_registry = lower_policy_registry,
      .legality_provider_list = legality_provider_list,
      .legalizer_provider_list = legalizer_provider_list,
      .compile_report = compile_report,
  };
}

static loom_pass_environment_t
loom_low_pass_environment_storage_initialize_with_target(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_lower_policy_registry_t* lower_policy_registry,
    const loom_target_low_legality_provider_list_t* legality_provider_list,
    const loom_target_legalizer_provider_list_t* legalizer_provider_list,
    const loom_target_math_policy_registry_t* math_policy_registry,
    loom_target_compile_report_t* compile_report,
    loom_target_pass_capability_t target_capability,
    loom_low_pass_environment_storage_t* out_storage) {
  out_storage->target_capability = target_capability;
  out_storage->low_capability = loom_low_pass_capability_make(
      descriptor_registry, lower_policy_registry, legality_provider_list,
      legalizer_provider_list, compile_report);
  out_storage->math_capability = loom_target_math_pass_capability_make(
      math_policy_registry, compile_report);
  out_storage->capabilities[0] = &out_storage->target_capability.base;
  out_storage->capabilities[1] = &out_storage->low_capability.base;
  out_storage->capabilities[2] = &out_storage->math_capability.base;
  out_storage->environment = loom_pass_environment_make(
      out_storage->capabilities, IREE_ARRAYSIZE(out_storage->capabilities));
  return out_storage->environment;
}

loom_pass_environment_t loom_low_pass_environment_storage_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_lower_policy_registry_t* lower_policy_registry,
    const loom_target_low_legality_provider_list_t* legality_provider_list,
    const loom_target_legalizer_provider_list_t* legalizer_provider_list,
    const loom_target_math_policy_registry_t* math_policy_registry,
    loom_target_compile_report_t* compile_report,
    const loom_target_environment_t* target_environment,
    const loom_function_version_list_t* function_versions,
    loom_low_pass_environment_storage_t* out_storage) {
  return loom_low_pass_environment_storage_initialize_with_target(
      descriptor_registry, lower_policy_registry, legality_provider_list,
      legalizer_provider_list, math_policy_registry, compile_report,
      loom_target_pass_capability_make(target_environment, function_versions),
      out_storage);
}

loom_pass_environment_t loom_low_pass_environment_storage_initialize_mutable(
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_lower_policy_registry_t* lower_policy_registry,
    const loom_target_low_legality_provider_list_t* legality_provider_list,
    const loom_target_legalizer_provider_list_t* legalizer_provider_list,
    const loom_target_math_policy_registry_t* math_policy_registry,
    loom_target_compile_report_t* compile_report,
    const loom_target_environment_t* target_environment,
    loom_function_version_owner_t* function_version_owner,
    loom_low_pass_environment_storage_t* out_storage) {
  return loom_low_pass_environment_storage_initialize_with_target(
      descriptor_registry, lower_policy_registry, legality_provider_list,
      legalizer_provider_list, math_policy_registry, compile_report,
      loom_target_pass_capability_make_mutable(target_environment,
                                               function_version_owner),
      out_storage);
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

const loom_target_legalizer_provider_list_t*
loom_low_pass_capability_legalizer_provider_list(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->legalizer_provider_list : NULL;
}

loom_target_compile_report_t* loom_low_pass_capability_compile_report(
    const loom_low_pass_capability_t* capability) {
  return capability ? capability->compile_report : NULL;
}
