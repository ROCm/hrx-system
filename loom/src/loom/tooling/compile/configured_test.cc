// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured.h"

#include "iree/testing/gtest.h"

#ifndef LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS

namespace loom {
namespace {

TEST(ConfiguredCompileTest, ReturnsStableCompleteEnvironment) {
  const loom_tooling_compile_environment_t* environment =
      loom_tooling_configured_compile_environment();
  ASSERT_NE(environment, nullptr);
  EXPECT_EQ(environment, loom_tooling_configured_compile_environment());
  ASSERT_NE(environment->target_environment, nullptr);
  ASSERT_NE(environment->target_environment->provider_set, nullptr);
  EXPECT_GT(environment->target_environment->provider_set->provider_count, 0u);
  EXPECT_NE(environment->cleanup_pattern_provider_set, nullptr);

  const bool has_amdgpu_emitter =
      loom_target_environment_lookup_emitter(
          environment->target_environment, IREE_SV("amdgpu-hsaco")) != nullptr;
  const bool has_spirv_emitter =
      loom_target_environment_lookup_emitter(environment->target_environment,
                                             IREE_SV("spirv")) != nullptr;
  const bool has_vm_emitter =
      loom_target_environment_lookup_emitter(environment->target_environment,
                                             IREE_SV("vm")) != nullptr;
  const bool has_wasm_emitter =
      loom_target_environment_lookup_emitter(environment->target_environment,
                                             IREE_SV("wasm-binary")) != nullptr;
  const bool has_xdna_emitter =
      loom_target_environment_lookup_emitter(environment->target_environment,
                                             IREE_SV("xdna")) != nullptr;
  EXPECT_EQ(has_amdgpu_emitter,
            static_cast<bool>(LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS));
  EXPECT_EQ(has_spirv_emitter,
            static_cast<bool>(LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS));
  EXPECT_EQ(has_vm_emitter,
            static_cast<bool>(LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS));
  EXPECT_EQ(has_wasm_emitter,
            static_cast<bool>(LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS));
  EXPECT_EQ(has_xdna_emitter,
            static_cast<bool>(LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS));
}

}  // namespace
}  // namespace loom
