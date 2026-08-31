// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime/build_tools/bazel/test/hal_driver_registration_fixture.h"

static int iree_hal_driver_registration_fixture_sequence[3];
static iree_host_size_t iree_hal_driver_registration_fixture_sequence_count;
static int iree_hal_driver_registration_fixture_failing_module;

void iree_hal_driver_registration_fixture_reset(int failing_module) {
  iree_hal_driver_registration_fixture_sequence_count = 0;
  iree_hal_driver_registration_fixture_failing_module = failing_module;
}

iree_host_size_t iree_hal_driver_registration_fixture_count(void) {
  return iree_hal_driver_registration_fixture_sequence_count;
}

int iree_hal_driver_registration_fixture_at(iree_host_size_t index) {
  return iree_hal_driver_registration_fixture_sequence[index];
}

static iree_status_t iree_hal_driver_registration_fixture_record(
    iree_hal_driver_registry_t* registry, int module) {
  if (!registry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "registry must not be null");
  }
  iree_hal_driver_registration_fixture_sequence
      [iree_hal_driver_registration_fixture_sequence_count++] = module;
  if (module == iree_hal_driver_registration_fixture_failing_module) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "registration fixture module %d failed", module);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_driver_registration_fixture_alpha(
    iree_hal_driver_registry_t* registry) {
  return iree_hal_driver_registration_fixture_record(registry, 1);
}

iree_status_t iree_hal_driver_registration_fixture_beta(
    iree_hal_driver_registry_t* registry) {
  return iree_hal_driver_registration_fixture_record(registry, 2);
}

iree_status_t iree_hal_driver_registration_fixture_gamma(
    iree_hal_driver_registry_t* registry) {
  return iree_hal_driver_registration_fixture_record(registry, 3);
}
