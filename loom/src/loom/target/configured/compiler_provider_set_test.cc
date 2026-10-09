// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/configured/compiler_provider_set.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/selection.h"

#ifndef LOOM_CONFIG_COMPILER_HAVE_X86
#define LOOM_CONFIG_COMPILER_HAVE_X86 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_X86
#ifndef LOOM_CONFIG_COMPILER_HAVE_AMDGPU
#define LOOM_CONFIG_COMPILER_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_AMDGPU
#ifndef LOOM_CONFIG_COMPILER_HAVE_SPIRV
#define LOOM_CONFIG_COMPILER_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_SPIRV
#ifndef LOOM_CONFIG_COMPILER_HAVE_VM
#define LOOM_CONFIG_COMPILER_HAVE_VM 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_VM
#ifndef LOOM_CONFIG_COMPILER_HAVE_WASM
#define LOOM_CONFIG_COMPILER_HAVE_WASM 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_WASM
#ifndef LOOM_CONFIG_COMPILER_HAVE_XDNA
#define LOOM_CONFIG_COMPILER_HAVE_XDNA 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_XDNA

namespace loom {
namespace {

TEST(ConfiguredCompilerProviderSetTest, ComposesSelectedTargetCompilers) {
  const loom_target_provider_set_t* emitter_provider_set =
      loom_configured_emitter_provider_set();
  const loom_target_provider_set_t* compiler_provider_set =
      loom_configured_compiler_provider_set();
  EXPECT_EQ(compiler_provider_set, loom_configured_compiler_provider_set());
  ASSERT_EQ(compiler_provider_set->provider_count,
            2 * emitter_provider_set->provider_count);
  for (iree_host_size_t i = 0; i < emitter_provider_set->provider_count; ++i) {
    bool found = false;
    for (iree_host_size_t j = 0; j < compiler_provider_set->provider_count;
         ++j) {
      found |= compiler_provider_set->providers[j] ==
               emitter_provider_set->providers[i];
    }
    EXPECT_TRUE(found);
  }

  loom_target_environment_t environment;
  IREE_ASSERT_OK(
      loom_target_environment_initialize(compiler_provider_set, &environment));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("x86-elf")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_X86));
  EXPECT_EQ(loom_target_environment_lookup_emitter(
                &environment, IREE_SV("amdgpu-hsaco")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_AMDGPU));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("spirv")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_SPIRV));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("vm")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_VM));
  EXPECT_EQ(loom_target_environment_lookup_emitter(
                &environment, IREE_SV("wasm-binary")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_WASM));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("xdna")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_XDNA));
  loom_target_environment_deinitialize(&environment);
}

#if LOOM_CONFIG_COMPILER_HAVE_X86
TEST(ConfiguredCompilerProviderSetTest, NativeCpuSelectionUsesDeviceFacts) {
  loom_target_environment_t environment;
  IREE_ASSERT_OK(loom_target_environment_initialize(
      loom_configured_compiler_provider_set(), &environment));
  const loom_target_specification_t scalar_specification = {
      .family = IREE_SVL("x86"),
      .selector = IREE_SVL("scalar"),
  };
  const loom_target_profile_t* scalar_profile = nullptr;
  IREE_ASSERT_OK(loom_target_environment_select_profile(
      &environment, &scalar_specification, &scalar_profile));

  // No optional ISA features are promised by this execution device.
  iree_cpu_data_t cpu_data =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_X86_64;
  loom_target_profile_selection_t selected = {};
  IREE_ASSERT_OK(loom_target_environment_select_cpu_profile(
      &environment, &cpu_data, nullptr, nullptr, &selected,
      iree_allocator_system()));
  ASSERT_NE(selected.profile, nullptr);
  EXPECT_NE(selected.profile, scalar_profile);
  EXPECT_NE(selected.destroy, nullptr);
  const loom_target_bundle_t* selected_bundle =
      loom_target_profile_bundle(selected.profile);
  ASSERT_NE(selected_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(selected_bundle->config->contract_set_key,
                                     IREE_SV("x86.scalar.core")));
  loom_target_profile_selection_release(&selected, iree_allocator_system());
  IREE_ASSERT_OK(loom_target_environment_select_cpu_profile(
      &environment, &cpu_data, nullptr, scalar_profile, &selected,
      iree_allocator_system()));
  EXPECT_EQ(selected.profile, scalar_profile);
  EXPECT_EQ(selected.destroy, nullptr);
  loom_target_profile_selection_release(&selected, iree_allocator_system());

  // A compiler running on x86 must still reject an x86 profile for ARM.
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_ARM_64;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_target_environment_select_cpu_profile(
                            &environment, &cpu_data, nullptr, scalar_profile,
                            &selected, iree_allocator_system()));
  EXPECT_EQ(selected.profile, nullptr);
  loom_target_environment_deinitialize(&environment);
}
#endif  // LOOM_CONFIG_COMPILER_HAVE_X86

}  // namespace
}  // namespace loom
