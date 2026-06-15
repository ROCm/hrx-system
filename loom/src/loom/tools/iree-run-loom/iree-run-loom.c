// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom binary with build-selected execution providers.

#include <stddef.h>
#include <stdio.h>

#include "loom/tooling/execution/execution_provider.h"
#include "loom/tools/iree-run-loom/main.h"

#ifndef IREE_RUN_LOOM_HAVE_AMDGPU
#define IREE_RUN_LOOM_HAVE_AMDGPU 0
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
#ifndef IREE_RUN_LOOM_HAVE_IREE_VM
#define IREE_RUN_LOOM_HAVE_IREE_VM 0
#endif  // IREE_RUN_LOOM_HAVE_IREE_VM
#ifndef IREE_RUN_LOOM_HAVE_SPIRV
#define IREE_RUN_LOOM_HAVE_SPIRV 0
#endif  // IREE_RUN_LOOM_HAVE_SPIRV

#define IREE_RUN_LOOM_HAVE_ANY_PROVIDER                       \
  (IREE_RUN_LOOM_HAVE_AMDGPU || IREE_RUN_LOOM_HAVE_IREE_VM || \
   IREE_RUN_LOOM_HAVE_SPIRV)

#if IREE_RUN_LOOM_HAVE_AMDGPU
#include "loom/tooling/target/amdgpu/execution/provider.h"
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
#if IREE_RUN_LOOM_HAVE_IREE_VM
#include "loom/tooling/execution/ireevm/provider.h"
#endif  // IREE_RUN_LOOM_HAVE_IREE_VM
#if IREE_RUN_LOOM_HAVE_SPIRV
#include "loom/tooling/target/spirv/execution/provider.h"
#endif  // IREE_RUN_LOOM_HAVE_SPIRV

#if IREE_RUN_LOOM_HAVE_ANY_PROVIDER
static const loom_run_execution_provider_t* const kIreeRunLoomProviders[] = {
#if IREE_RUN_LOOM_HAVE_AMDGPU
    &loom_amdgpu_hal_execution_provider,
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
#if IREE_RUN_LOOM_HAVE_IREE_VM
    &loom_ireevm_execution_provider,
#endif  // IREE_RUN_LOOM_HAVE_IREE_VM
#if IREE_RUN_LOOM_HAVE_SPIRV
    &loom_spirv_vulkan_hal_execution_provider,
#endif  // IREE_RUN_LOOM_HAVE_SPIRV
};
#endif  // IREE_RUN_LOOM_HAVE_ANY_PROVIDER

static const loom_run_execution_provider_set_t kIreeRunLoomProviderSet = {
#if IREE_RUN_LOOM_HAVE_ANY_PROVIDER
    .providers = kIreeRunLoomProviders,
    .provider_count = IREE_ARRAYSIZE(kIreeRunLoomProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // IREE_RUN_LOOM_HAVE_ANY_PROVIDER
};

int main(int argc, char** argv) {
  loom_run_execution_environment_t environment;
  iree_status_t status = loom_run_execution_environment_initialize(
      &kIreeRunLoomProviderSet, &environment);
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
