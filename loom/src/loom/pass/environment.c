// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/environment.h"

iree_status_t loom_pass_environment_verify(
    const loom_pass_environment_t* environment) {
  if (environment->capability_count != 0 && !environment->capabilities) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass environment capability list is required when count is non-zero");
  }
  for (iree_host_size_t i = 0; i < environment->capability_count; ++i) {
    const loom_pass_environment_capability_t* capability =
        environment->capabilities[i];
    if (!capability || !capability->type) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass environment capability %zu is invalid", i);
    }
    for (iree_host_size_t j = i + 1; j < environment->capability_count; ++j) {
      const loom_pass_environment_capability_t* other =
          environment->capabilities[j];
      if (other && other->type == capability->type) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass environment contains duplicate capability type '%.*s'",
            (int)capability->type->name.size, capability->type->name.data);
      }
    }
  }
  return iree_ok_status();
}

const loom_pass_environment_capability_t* loom_pass_environment_lookup(
    const loom_pass_environment_t* environment,
    const loom_pass_environment_capability_type_t* type) {
  if (!environment->capabilities) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < environment->capability_count; ++i) {
    const loom_pass_environment_capability_t* capability =
        environment->capabilities[i];
    if (capability && capability->type == type) {
      return capability;
    }
  }
  return NULL;
}

bool loom_pass_environment_capability_satisfies_requirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  if (!capability || !capability->type ||
      !capability->type->satisfies_requirement) {
    return false;
  }
  return capability->type->satisfies_requirement(capability, requirement);
}
