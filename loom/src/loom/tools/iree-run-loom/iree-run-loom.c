// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom binary with build-selected execution providers.

#include <stdio.h>

#include "loom/tooling/execution/configured.h"
#include "loom/tooling/execution/execution_provider.h"
#include "loom/tools/iree-run-loom/main.h"

int main(int argc, char** argv) {
  const loom_tooling_execution_providers_t* configured_providers =
      loom_tooling_configured_execution_providers();
  loom_run_execution_environment_t environment;
  iree_status_t status = loom_run_execution_environment_initialize(
      &configured_providers->execution_provider_set, &environment);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const iree_run_loom_configuration_t configuration = {
      .tool_name = "iree-run-loom",
      .register_context =
          loom_run_execution_environment_register_context_callback(
              &environment),
      .initialize_low_descriptor_registry =
          loom_run_execution_environment_low_descriptor_registry_callback(
              &environment),
      .target_environment =
          loom_run_execution_environment_target_environment(&environment),
      .execution_backend_registry =
          *loom_run_execution_environment_execution_backend_registry(
              &environment),
  };
  int exit_code = iree_run_loom_main(argc, argv, &configuration);
  loom_run_execution_environment_deinitialize(&environment);
  return exit_code;
}
