// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/execution_provider.h"

static iree_status_t loom_run_execution_environment_append_target_provider(
    loom_run_execution_environment_t* environment,
    const loom_run_execution_provider_t* provider) {
  if (provider->target_provider == NULL) {
    return iree_ok_status();
  }
  if (environment->target_provider_count >=
      IREE_ARRAYSIZE(environment->target_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom execution target provider capacity exceeded");
  }
  environment->target_providers[environment->target_provider_count++] =
      provider->target_provider;
  return iree_ok_status();
}

static iree_status_t loom_run_execution_environment_append_device_provider(
    loom_run_execution_environment_t* environment,
    const loom_run_execution_provider_t* provider) {
  if (provider->device_provider == NULL) {
    return iree_ok_status();
  }
  if (environment->device_provider_count >=
      IREE_ARRAYSIZE(environment->device_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom HAL device provider capacity exceeded");
  }
  environment->device_providers[environment->device_provider_count++] =
      provider->device_provider;
  return iree_ok_status();
}

iree_status_t loom_run_execution_environment_initialize(
    const loom_run_execution_provider_set_t* provider_set,
    loom_run_execution_environment_t* out_environment) {
  *out_environment = (loom_run_execution_environment_t){
      .provider_set = provider_set,
  };

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_run_execution_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(loom_run_execution_environment_append_target_provider(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(loom_run_execution_environment_append_device_provider(
        out_environment, provider));
  }
  out_environment->target_provider_set =
      loom_target_provider_set_make(out_environment->target_providers,
                                    out_environment->target_provider_count);
  IREE_RETURN_IF_ERROR(
      loom_target_environment_initialize(&out_environment->target_provider_set,
                                         &out_environment->target_environment));
  loom_device_provider_registry_initialize_from_entries(
      out_environment->device_providers, out_environment->device_provider_count,
      &out_environment->device_provider_registry);
  return iree_ok_status();
}

void loom_run_execution_environment_deinitialize(
    loom_run_execution_environment_t* environment) {
  if (environment == NULL) {
    return;
  }
  *environment = (loom_run_execution_environment_t){0};
}

static iree_status_t loom_run_execution_environment_register_context(
    void* user_data, loom_context_t* context) {
  loom_run_execution_environment_t* environment =
      (loom_run_execution_environment_t*)user_data;
  return loom_target_environment_register_context(
      &environment->target_environment, context);
}

loom_run_register_context_callback_t
loom_run_execution_environment_register_context_callback(
    loom_run_execution_environment_t* environment) {
  return (loom_run_register_context_callback_t){
      .fn = loom_run_execution_environment_register_context,
      .user_data = environment,
  };
}

static iree_status_t
loom_run_execution_environment_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  loom_run_execution_environment_t* environment =
      (loom_run_execution_environment_t*)user_data;
  return loom_target_environment_initialize_low_descriptor_registry(
      &environment->target_environment, out_registry);
}

loom_run_initialize_low_descriptor_registry_callback_t
loom_run_execution_environment_low_descriptor_registry_callback(
    loom_run_execution_environment_t* environment) {
  return (loom_run_initialize_low_descriptor_registry_callback_t){
      .fn = loom_run_execution_environment_initialize_low_descriptor_registry,
      .user_data = environment,
  };
}

const loom_target_environment_t*
loom_run_execution_environment_target_environment(
    const loom_run_execution_environment_t* environment) {
  return &environment->target_environment;
}

const loom_device_provider_registry_t*
loom_run_execution_environment_device_provider_registry(
    const loom_run_execution_environment_t* environment) {
  return &environment->device_provider_registry;
}
