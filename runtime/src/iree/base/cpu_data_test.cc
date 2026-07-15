// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/cpu_data.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(CpuDataTest, QueriesSyntheticFeatures) {
  iree_cpu_data_t cpu_data = {
      /*.architecture=*/IREE_CPU_ARCHITECTURE_X86_64,
      /*.fields=*/{IREE_CPU_DATA0_X86_64_AVX2},
  };
  EXPECT_EQ(IREE_CPU_FEATURE_AVAILABILITY_AVAILABLE,
            iree_cpu_data_query_feature(&cpu_data, IREE_SV("avx2")));
  EXPECT_EQ(IREE_CPU_FEATURE_AVAILABILITY_UNAVAILABLE,
            iree_cpu_data_query_feature(&cpu_data, IREE_SV("avx512f")));
  EXPECT_EQ(IREE_CPU_FEATURE_AVAILABILITY_UNKNOWN,
            iree_cpu_data_query_feature(&cpu_data, IREE_SV("sve")));

  const iree_host_size_t feature_count =
      iree_cpu_feature_count(IREE_CPU_ARCHITECTURE_X86_64);
  ASSERT_GT(feature_count, 0u);
  EXPECT_FALSE(iree_string_view_is_empty(
      iree_cpu_feature_name(IREE_CPU_ARCHITECTURE_X86_64, feature_count - 1)));
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_cpu_feature_name(IREE_CPU_ARCHITECTURE_X86_64, feature_count)));
}

TEST(CpuDataTest, ParsesAndFormatsCanonicalTargetKeys) {
  iree_cpu_data_t cpu_data = {};
  IREE_ASSERT_OK(
      iree_cpu_data_parse_target_key(IREE_SV("x86_64:+avx2:+avx"), &cpu_data));
  EXPECT_EQ(IREE_CPU_ARCHITECTURE_X86_64, cpu_data.architecture);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(iree_cpu_data_append_target_key(&cpu_data, &builder));
  EXPECT_TRUE(iree_string_view_equal(iree_string_builder_view(&builder),
                                     IREE_SV("x86_64:+avx:+avx2")));
  iree_string_builder_deinitialize(&builder);
}

TEST(CpuDataTest, RejectsMalformedTargetKeys) {
  iree_cpu_data_t cpu_data = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_cpu_data_parse_target_key(IREE_SV("x86_64:avx2"), &cpu_data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_cpu_data_parse_target_key(IREE_SV("x86_64:+avx2:+avx2"), &cpu_data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_cpu_data_parse_target_key(IREE_SV("arm_64:+avx2"), &cpu_data));
}

TEST(CpuDataTest, ChecksRequiredFeatureSubsets) {
  iree_cpu_data_t available = {
      /*.architecture=*/IREE_CPU_ARCHITECTURE_X86_64,
      /*.fields=*/{IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_AVX2},
  };
  iree_cpu_data_t required = {
      /*.architecture=*/IREE_CPU_ARCHITECTURE_X86_64,
      /*.fields=*/{IREE_CPU_DATA0_X86_64_AVX2},
  };
  EXPECT_TRUE(iree_cpu_data_satisfies_features(&available, &required));
  required.fields[0] |= IREE_CPU_DATA0_X86_64_AVX512F;
  EXPECT_FALSE(iree_cpu_data_satisfies_features(&available, &required));
  required = {};
  required.architecture = IREE_CPU_ARCHITECTURE_ARM_64;
  EXPECT_FALSE(iree_cpu_data_satisfies_features(&available, &required));

  required = {};
  required.architecture = IREE_CPU_ARCHITECTURE_X86_64;
  required.fields[1] = 1;
  EXPECT_FALSE(iree_cpu_data_satisfies_features(&available, &required));
}

}  // namespace
