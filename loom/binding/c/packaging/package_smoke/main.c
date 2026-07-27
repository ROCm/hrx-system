// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <loomc/loomc.h>
#include <loomc/target/amdgpu.h>

int main(void) {
  loomc_target_environment_t* target_environment = NULL;
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment);
  if (!loomc_status_is_ok(status)) {
    loomc_status_free(status);
    return 1;
  }
  loomc_target_environment_release(target_environment);
  return 0;
}
