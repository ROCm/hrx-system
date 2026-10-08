// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/provider.h"

#include "iree/base/cpu_data.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/x86/ops/ops.h"

namespace loom {
namespace {

TEST(X86ProviderTest, PreservesLowCalls) {
  ASSERT_NE(loom_x86_target_provider.select_call_policy, nullptr);
  const loom_resolved_target_t resolved_target = {};
  EXPECT_EQ(loom_x86_target_provider.select_call_policy(
                &resolved_target, nullptr, LOOM_CALL_LIKE_KIND_LOW_INTERNAL,
                loom_call_like_t{}, loom_func_like_t{}),
            LOOM_TARGET_CALL_POLICY_DIRECT);
}

TEST(X86ProviderTest, SelectsStrongestExecutableCpuProfile) {
  ASSERT_NE(loom_x86_target_provider.select_profile, nullptr);
  ASSERT_NE(loom_x86_target_provider.select_cpu_profile, nullptr);
  const loom_target_profile_t* scalar_profile = nullptr;
  const loom_target_profile_t* simd128_profile = nullptr;
  const loom_target_profile_t* avx2_profile = nullptr;
  const loom_target_profile_t* avx512_profile = nullptr;
  IREE_ASSERT_OK(loom_x86_target_provider.select_profile(IREE_SV("scalar"),
                                                         &scalar_profile));
  IREE_ASSERT_OK(loom_x86_target_provider.select_profile(IREE_SV("simd128"),
                                                         &simd128_profile));
  IREE_ASSERT_OK(
      loom_x86_target_provider.select_profile(IREE_SV("avx2"), &avx2_profile));
  IREE_ASSERT_OK(loom_x86_target_provider.select_profile(IREE_SV("avx512"),
                                                         &avx512_profile));

  iree_cpu_data_t cpu_data = {};
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_X86_64;
  EXPECT_EQ(
      loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr, nullptr),
      scalar_profile);

  cpu_data.fields[0] = IREE_CPU_DATA0_X86_64_AVX;
  EXPECT_EQ(
      loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr, nullptr),
      simd128_profile);

  cpu_data.fields[0] = IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
                       IREE_CPU_DATA0_X86_64_AVX2;
  EXPECT_EQ(
      loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr, nullptr),
      avx2_profile);

  cpu_data.fields[0] |=
      IREE_CPU_DATA0_X86_64_AVX512F | IREE_CPU_DATA0_X86_64_AVX512VL |
      IREE_CPU_DATA0_X86_64_AVX512DQ | IREE_CPU_DATA0_X86_64_AVX512BW;
  EXPECT_EQ(
      loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr, nullptr),
      avx512_profile);
}

TEST(X86ProviderTest, RequiresCompleteCpuFeatureClosures) {
  ASSERT_NE(loom_x86_target_provider.select_profile, nullptr);
  ASSERT_NE(loom_x86_target_provider.select_cpu_profile, nullptr);
  struct ProfileCase {
    // Public profile selector spelling.
    const char* selector;
    // Complete CPU field-zero feature closure.
    uint64_t required_features;
  } cases[] = {
      {"simd128", IREE_CPU_DATA0_X86_64_AVX},
      {"avx2", IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
                   IREE_CPU_DATA0_X86_64_AVX2},
      {"avx512",
       IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
           IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_AVX512F |
           IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512DQ |
           IREE_CPU_DATA0_X86_64_AVX512BW},
  };
  iree_cpu_data_t cpu_data = {};
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_X86_64;
  for (const auto& profile_case : cases) {
    const loom_target_profile_t* profile = nullptr;
    IREE_ASSERT_OK(loom_x86_target_provider.select_profile(
        iree_make_cstring_view(profile_case.selector), &profile));
    cpu_data.fields[0] = profile_case.required_features;
    EXPECT_EQ(loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr,
                                                          profile),
              profile)
        << profile_case.selector;
    for (uint64_t feature = 1; feature != 0; feature <<= 1) {
      if (!iree_any_bit_set(profile_case.required_features, feature)) {
        continue;
      }
      cpu_data.fields[0] = profile_case.required_features & ~feature;
      EXPECT_EQ(loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr,
                                                            profile),
                nullptr)
          << profile_case.selector << " missing feature " << feature;
    }
  }
}

TEST(X86ProviderTest, PreservesRequirementIdentity) {
  const loom_target_profile_t* avx2_profile = nullptr;
  IREE_ASSERT_OK(
      loom_x86_target_provider.select_profile(IREE_SV("avx2"), &avx2_profile));
  const loom_target_facts_t requirement = {
      /*.fact_type=*/avx2_profile->type->fact_type,
      /*.selector=*/LOOM_X86_TARGET_KIND_AVX2,
  };
  iree_cpu_data_t cpu_data = {};
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_X86_64;
  cpu_data.fields[0] =
      IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
      IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_AVX512F |
      IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512DQ |
      IREE_CPU_DATA0_X86_64_AVX512BW;
  EXPECT_EQ(loom_x86_target_provider.select_cpu_profile(&cpu_data, &requirement,
                                                        nullptr),
            avx2_profile);
}

TEST(X86ProviderTest, RejectsNonExecutableAndForeignProfiles) {
  const loom_target_profile_t* packed_dot_profile = nullptr;
  const loom_target_profile_t* avx512_packed_dot_profile = nullptr;
  const loom_target_profile_t* scalar_profile = nullptr;
  IREE_ASSERT_OK(loom_x86_target_provider.select_profile(IREE_SV("packed_dot"),
                                                         &packed_dot_profile));
  IREE_ASSERT_OK(loom_x86_target_provider.select_profile(
      IREE_SV("avx512_packed_dot"), &avx512_packed_dot_profile));
  IREE_ASSERT_OK(loom_x86_target_provider.select_profile(IREE_SV("scalar"),
                                                         &scalar_profile));

  iree_cpu_data_t cpu_data = {};
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_X86_64;
  cpu_data.fields[0] = UINT64_MAX;
  EXPECT_EQ(loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr,
                                                        packed_dot_profile),
            nullptr);
  EXPECT_EQ(loom_x86_target_provider.select_cpu_profile(
                &cpu_data, nullptr, avx512_packed_dot_profile),
            nullptr);

  cpu_data.architecture = IREE_CPU_ARCHITECTURE_ARM_64;
  EXPECT_EQ(loom_x86_target_provider.select_cpu_profile(&cpu_data, nullptr,
                                                        scalar_profile),
            nullptr);
}

}  // namespace
}  // namespace loom
