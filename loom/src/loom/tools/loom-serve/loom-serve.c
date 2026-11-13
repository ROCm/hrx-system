// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tracing.h"
#include "iree/tooling/device_util.h"
#include "loom/tools/loom-serve/build_identity.h"
#include "loom/tools/loom-serve/main.h"
#include "loomc/target/configured.h"

#ifndef LOOM_SERVE_CONFIG_HAVE_X86
#define LOOM_SERVE_CONFIG_HAVE_X86 0
#endif  // LOOM_SERVE_CONFIG_HAVE_X86
#ifndef LOOM_SERVE_CONFIG_HAVE_AMDGPU
#define LOOM_SERVE_CONFIG_HAVE_AMDGPU 0
#endif  // LOOM_SERVE_CONFIG_HAVE_AMDGPU
#ifndef LOOM_SERVE_CONFIG_HAVE_SPIRV
#define LOOM_SERVE_CONFIG_HAVE_SPIRV 0
#endif  // LOOM_SERVE_CONFIG_HAVE_SPIRV
#ifndef LOOM_SERVE_CONFIG_HAVE_XDNA
#define LOOM_SERVE_CONFIG_HAVE_XDNA 0
#endif  // LOOM_SERVE_CONFIG_HAVE_XDNA

#define LOOM_SERVE_CONFIG_HAVE_ANY_TARGET                         \
  (LOOM_SERVE_CONFIG_HAVE_X86 || LOOM_SERVE_CONFIG_HAVE_AMDGPU || \
   LOOM_SERVE_CONFIG_HAVE_SPIRV || LOOM_SERVE_CONFIG_HAVE_XDNA)

#if LOOM_SERVE_CONFIG_HAVE_X86
#include "loomc/target/cpu/iree_hal.h"
#endif  // LOOM_SERVE_CONFIG_HAVE_X86
#if LOOM_SERVE_CONFIG_HAVE_AMDGPU
#include "loomc/target/amdgpu/iree_hal.h"
#endif  // LOOM_SERVE_CONFIG_HAVE_AMDGPU
#if LOOM_SERVE_CONFIG_HAVE_SPIRV
#include "loomc/target/spirv/iree_hal.h"
#endif  // LOOM_SERVE_CONFIG_HAVE_SPIRV
#if LOOM_SERVE_CONFIG_HAVE_XDNA
#include "loomc/target/amd/xdna/iree_hal.h"
#endif  // LOOM_SERVE_CONFIG_HAVE_XDNA

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  const loom_runtime_compiler_options_t* compiler_options = NULL;
#if LOOM_SERVE_CONFIG_HAVE_ANY_TARGET
  const loomc_iree_hal_target_provider_t* target_providers[] = {
#if LOOM_SERVE_CONFIG_HAVE_X86
      loomc_cpu_iree_hal_target_provider(),
#endif  // LOOM_SERVE_CONFIG_HAVE_X86
#if LOOM_SERVE_CONFIG_HAVE_AMDGPU
      loomc_amdgpu_iree_hal_target_provider(),
#endif  // LOOM_SERVE_CONFIG_HAVE_AMDGPU
#if LOOM_SERVE_CONFIG_HAVE_SPIRV
      loomc_spirv_iree_hal_target_provider(),
#endif  // LOOM_SERVE_CONFIG_HAVE_SPIRV
#if LOOM_SERVE_CONFIG_HAVE_XDNA
      loomc_xdna_iree_hal_target_provider(),
#endif  // LOOM_SERVE_CONFIG_HAVE_XDNA
  };
  const loom_runtime_compiler_options_t configured_compiler_options = {
      .target_environment_create = loomc_target_environment_create_configured,
      .target_providers = target_providers,
      .target_provider_count = IREE_ARRAYSIZE(target_providers),
  };
  compiler_options = &configured_compiler_options;
#endif  // LOOM_SERVE_CONFIG_HAVE_ANY_TARGET
  const loom_serve_configuration_t configuration = {
      .build_identity = loom_serve_build_identity(),
      .compiler_options = compiler_options,
      .driver_registry = iree_hal_available_driver_registry(),
  };
  int exit_code = loom_serve_main(argc, argv, &configuration);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
