// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/cpu_x86_64.h"

#include "iree/base/internal/cpu.h"
#include "iree/testing/gtest.h"

namespace {

constexpr uint64_t kSseFeatures =
    IREE_CPU_DATA0_X86_64_SSE3 | IREE_CPU_DATA0_X86_64_SSSE3 |
    IREE_CPU_DATA0_X86_64_SSE41 | IREE_CPU_DATA0_X86_64_SSE42 |
    IREE_CPU_DATA0_X86_64_SSE4A;
constexpr uint64_t kAvxFeatures =
    IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
    IREE_CPU_DATA0_X86_64_FMA4 | IREE_CPU_DATA0_X86_64_XOP |
    IREE_CPU_DATA0_X86_64_F16C | IREE_CPU_DATA0_X86_64_AVX2 |
    IREE_CPU_DATA0_X86_64_AVXVNNI | IREE_CPU_DATA0_X86_64_AVXVNNIINT8 |
    IREE_CPU_DATA0_X86_64_AVXVNNIINT16 | IREE_CPU_DATA0_X86_64_AVXNECONVERT;
constexpr uint64_t kAvx512Features =
    IREE_CPU_DATA0_X86_64_AVX512F | IREE_CPU_DATA0_X86_64_AVX512CD |
    IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512DQ |
    IREE_CPU_DATA0_X86_64_AVX512BW | IREE_CPU_DATA0_X86_64_AVX512IFMA |
    IREE_CPU_DATA0_X86_64_AVX512VBMI | IREE_CPU_DATA0_X86_64_AVX512VPOPCNTDQ |
    IREE_CPU_DATA0_X86_64_AVX512VNNI | IREE_CPU_DATA0_X86_64_AVX512VBMI2 |
    IREE_CPU_DATA0_X86_64_AVX512BITALG | IREE_CPU_DATA0_X86_64_AVX512BF16 |
    IREE_CPU_DATA0_X86_64_AVX512FP16;
constexpr uint64_t kAmxFeatures = IREE_CPU_DATA0_X86_64_AMXTILE |
                                  IREE_CPU_DATA0_X86_64_AMXINT8 |
                                  IREE_CPU_DATA0_X86_64_AMXBF16;

// All known instruction bits are set independently of enabled OS state.
iree_cpu_x86_64_capabilities_t InstructionCapabilities() {
  iree_cpu_x86_64_capabilities_t capabilities = {};
  capabilities.leaf1.ecx = (1u << 0) | (1u << 9) | (1u << 12) | (1u << 19) |
                           (1u << 20) | (1u << 26) | (1u << 27) | (1u << 28) |
                           (1u << 29);
  capabilities.leaf7_0.ebx = (1u << 5) | (1u << 16) | (1u << 17) | (1u << 21) |
                             (1u << 28) | (1u << 30) | (1u << 31);
  capabilities.leaf7_0.ecx =
      (1u << 1) | (1u << 6) | (1u << 11) | (1u << 12) | (1u << 14);
  capabilities.leaf7_0.edx = (1u << 22) | (1u << 23) | (1u << 24) | (1u << 25);
  capabilities.leaf7_1.eax = (1u << 4) | (1u << 5);
  capabilities.leaf7_1.edx = (1u << 4) | (1u << 5) | (1u << 10);
  capabilities.extended_leaf1.ecx = (1u << 6) | (1u << 11) | (1u << 16);
  return capabilities;
}

TEST(CpuX86_64Test, UnsupportedLeaves) {
  iree_cpu_x86_64_capabilities_t capabilities = {
      .enabled_xstate = UINT64_MAX, .permitted_xstate = UINT64_MAX};
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities), 0u);
}

TEST(CpuX86_64Test, InstructionSupportDoesNotEnableOsState) {
  auto capabilities = InstructionCapabilities();
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities), kSseFeatures);
}

TEST(CpuX86_64Test, XsaveAndOsxsaveAreRequired) {
  for (uint32_t missing_bit : {26u, 27u}) {
    auto capabilities = InstructionCapabilities();
    capabilities.leaf1.ecx &= ~(1u << missing_bit);
    capabilities.enabled_xstate = 0x600E7;
    capabilities.permitted_xstate = 0x600E7;
    EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities), kSseFeatures);
  }
}

TEST(CpuX86_64Test, AvxRequiresXmmAndYmmState) {
  for (uint64_t xstate : {0u, 1u, 3u, 5u}) {
    auto capabilities = InstructionCapabilities();
    capabilities.enabled_xstate = xstate;
    EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities), kSseFeatures);
  }
  auto capabilities = InstructionCapabilities();
  capabilities.enabled_xstate = 7;
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
            kSseFeatures | kAvxFeatures);
}

TEST(CpuX86_64Test, VectorStateDoesNotImplyAvxInstructions) {
  auto capabilities = InstructionCapabilities();
  capabilities.enabled_xstate = 0xE7;
  capabilities.leaf1.ecx &= ~(1u << 28);
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities), kSseFeatures);
}

TEST(CpuX86_64Test, Avx512RequiresAllZmmStateComponents) {
  for (uint32_t missing_bit : {5u, 6u, 7u}) {
    auto capabilities = InstructionCapabilities();
    capabilities.enabled_xstate = 0xE7 & ~(1u << missing_bit);
    EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
              kSseFeatures | kAvxFeatures);
  }
  auto capabilities = InstructionCapabilities();
  capabilities.enabled_xstate = 0xE7;
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
            kSseFeatures | kAvxFeatures | kAvx512Features);
  capabilities.leaf7_0.ebx &= ~(1u << 16);
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
            kSseFeatures | kAvxFeatures);
}

TEST(CpuX86_64Test, InstructionFeaturesRemainIndependent) {
  auto capabilities = InstructionCapabilities();
  capabilities.enabled_xstate = 0xE7;
  capabilities.leaf1.ecx &= ~((1u << 12) | (1u << 29));
  capabilities.leaf7_0.ebx &= ~(1u << 5);
  capabilities.extended_leaf1.ecx &= ~((1u << 11) | (1u << 16));
  capabilities.leaf7_1.eax = 0;
  capabilities.leaf7_1.edx = 0;
  capabilities.leaf7_0.edx &= ~(1u << 23);
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
            kSseFeatures | IREE_CPU_DATA0_X86_64_AVX |
                (kAvx512Features & ~(IREE_CPU_DATA0_X86_64_AVX512BF16 |
                                     IREE_CPU_DATA0_X86_64_AVX512FP16)));
}

TEST(CpuX86_64Test, AmxRequiresBothEnabledStateAndProcessPermission) {
  for (uint64_t xstate : {0u, 0x20000u, 0x40000u, 0x60000u}) {
    for (uint64_t permission : {0u, 0x20000u, 0x40000u, 0x60000u}) {
      auto capabilities = InstructionCapabilities();
      capabilities.enabled_xstate = 7 | xstate;
      capabilities.permitted_xstate = permission;
      const uint64_t expected_amx =
          xstate == 0x60000 && permission == 0x60000 ? kAmxFeatures : 0;
      EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
                kSseFeatures | kAvxFeatures | expected_amx);
    }
  }
}

TEST(CpuX86_64Test, AmxInstructionFamiliesRequireTileSupport) {
  auto capabilities = InstructionCapabilities();
  capabilities.enabled_xstate = 0x60007;
  capabilities.permitted_xstate = 0x60000;
  capabilities.leaf7_0.edx &= ~(1u << 24);
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
            kSseFeatures | kAvxFeatures);
  capabilities.leaf7_0.edx = 1u << 24;
  EXPECT_EQ(iree_cpu_x86_64_decode_features(&capabilities),
            kSseFeatures | kAvxFeatures | IREE_CPU_DATA0_X86_64_AMXTILE);
}

#if defined(IREE_ARCH_X86_64) && defined(IREE_PLATFORM_LINUX) && \
    defined(__GNUC__)
TEST(CpuX86_64Test, HostQueryMatchesCompilerVectorCapabilities) {
  // The Linux compiler runtime independently handles CPUID and OS vector state.
  // Compare through the public discovery path without executing vector code.
  __builtin_cpu_init();
  iree_cpu_data_t cpu_data;
  iree_cpu_query_data(iree_allocator_system(), &cpu_data);
  EXPECT_EQ(cpu_data.architecture, IREE_CPU_ARCHITECTURE_X86_64);
  EXPECT_EQ((cpu_data.fields[0] & IREE_CPU_DATA0_X86_64_AVX) != 0,
            __builtin_cpu_supports("avx") != 0);
  EXPECT_EQ((cpu_data.fields[0] & IREE_CPU_DATA0_X86_64_AVX2) != 0,
            __builtin_cpu_supports("avx2") != 0);
  EXPECT_EQ((cpu_data.fields[0] & IREE_CPU_DATA0_X86_64_AVX512F) != 0,
            __builtin_cpu_supports("avx512f") != 0);
}
#endif  // IREE_ARCH_X86_64 && IREE_PLATFORM_LINUX && __GNUC__

}  // namespace
