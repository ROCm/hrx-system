// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#include "loom/tools/loom-serve/activity_test.h"
#include "loomc/target/amdgpu/base.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cpu/iree_hal.h"
#include "loomc/target/x86.h"

static loomc_status_t loom_serve_activity_test_create_environment(
    loomc_allocator_t allocator, loomc_target_environment_t** out_environment) {
  loomc_target_environment_t* environments[2] = {NULL, NULL};
  loomc_status_t status =
      loomc_target_environment_create_x86(allocator, &environments[0]);
  if (loomc_status_is_ok(status)) {
    status =
        loomc_target_environment_create_amdgpu(allocator, &environments[1]);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_environment_create(
        IREE_ARRAYSIZE(environments), environments, allocator, out_environment);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(environments); ++i) {
    loomc_target_environment_release(environments[i]);
  }
  return status;
}

int main(int argc, char** argv) {
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_cpu_iree_hal_target_provider(),
      loomc_amdgpu_iree_hal_target_provider(),
  };
  const loom_serve_compiler_options_t compiler = {
      .target_environment_create = loom_serve_activity_test_create_environment,
      .target_providers = providers,
      .target_provider_count = IREE_ARRAYSIZE(providers),
  };
  iree_hal_driver_registry_t* drivers = NULL;
  iree_status_t status =
      iree_hal_driver_registry_allocate(iree_allocator_system(), &drivers);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_driver_module_register(drivers);
  }
  int result = 1;
  if (iree_status_is_ok(status)) {
    result = loom_serve_activity_test_main(argc, argv, &compiler, drivers);
  } else {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
  iree_hal_driver_registry_free(drivers);
  return result;
}
