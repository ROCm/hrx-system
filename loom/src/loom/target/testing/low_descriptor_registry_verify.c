// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/testing/low_descriptor_registry_verify.h"

iree_status_t loom_target_low_descriptor_registry_verify(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low registry is required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor-set providers are required");
  }
  if (registry->registry.descriptor_set_count != 0 ||
      registry->registry.descriptor_sets != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor view must use provider tables");
  }
  if (registry->registry.descriptor_set_providers !=
          registry->descriptor_set_providers ||
      registry->registry.descriptor_set_provider_count !=
          registry->descriptor_set_provider_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low registry descriptor view does not match provider table");
  }

  if (requirements != 0) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_verify_requirements(
        &registry->registry, requirements));
  }
  return iree_ok_status();
}
