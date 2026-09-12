// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/context_cache.h"
#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/testing/gtest.h"

namespace {

constexpr iree_host_size_t kTestSharedCodeMemoryBytes = 64u * 1024u * 1024u;
constexpr iree_host_size_t kTestMissReserveBytes = 4u * 1024u * 1024u;

// The per-architecture budget sizes the cache, so each supported NPU generation
// must map to its characterized value and anything else to 0 (unknown).
TEST(HardwareContextBudgetTest, MapsKnownArchitectures) {
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Phoenix")),
      6u);
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Strix")),
            32u);
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Strix Halo")),
      32u);
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Krackan")),
      32u);
}

TEST(HardwareContextBudgetTest, UnknownArchitectureIsZero) {
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Navi")),
            0u);
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_arch(
                iree_string_view_empty()),
            0u);
}

TEST(HardwareContextBudgetTest, MapsKnownPciIds) {
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amdxdna_npu_arch_for_pci(0x1022u, 0x1502u, 0u),
      IREE_SV("Phoenix")));
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amdxdna_npu_arch_for_pci(0x1022u, 0x17f0u, 0x11u),
      IREE_SV("Strix")));
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_pci(0x1022u, 0x1502u, 0u),
      6u);
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_pci(0x1022u, 0x17f0u, 0x11u),
      32u);
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_hal_amdxdna_npu_arch_for_pci(0x1022u, 0xffffu, 0u)));
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_pci(0x8086u, 0x17f0u, 0u),
      0u);
}

// A nonzero device budget wins, and a zero budget (unknown architecture) falls
// back to the built-in default of 8.
TEST(ContextCacheCapacityTest, PrefersDeviceBudget) {
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(32), 32u);
}

TEST(ContextCacheCapacityTest, FallsBackToDefaultWhenBudgetUnknown) {
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(0), 8u);
}

// A fixed native allocation domain must account for live context images before
// retaining prepared command code, leaving room to construct the next miss.
TEST(SharedCodeMemoryBudgetTest,
     DefaultCommandCapPlusContextImagesExceedsDomain) {
  const iree_host_size_t old_command_cap = 32u * 1024u * 1024u;
  const iree_host_size_t live_context_image_bytes = 32u * 1024u * 1024u;
  const iree_host_size_t miss_bytes = 33u * 1024u;

  EXPECT_GE(old_command_cap + live_context_image_bytes + miss_bytes,
            kTestSharedCodeMemoryBytes);

  const iree_host_size_t budget =
      iree_hal_amdxdna_shared_code_memory_command_budget(
          kTestSharedCodeMemoryBytes, kTestMissReserveBytes,
          live_context_image_bytes, 0);
  EXPECT_EQ(budget, 28u * 1024u * 1024u);
  EXPECT_LT(budget, old_command_cap);
  EXPECT_LE(miss_bytes, budget);
  EXPECT_EQ(budget + live_context_image_bytes + kTestMissReserveBytes,
            kTestSharedCodeMemoryBytes);
}

TEST(SharedCodeMemoryBudgetTest, EmptyDomainLeavesMissReserve) {
  EXPECT_EQ(iree_hal_amdxdna_shared_code_memory_command_budget(
                kTestSharedCodeMemoryBytes, kTestMissReserveBytes, 0, 0),
            kTestSharedCodeMemoryBytes - kTestMissReserveBytes);
}

TEST(SharedCodeMemoryBudgetTest, ChargesOtherCommandCaches) {
  EXPECT_EQ(iree_hal_amdxdna_shared_code_memory_command_budget(
                kTestSharedCodeMemoryBytes, kTestMissReserveBytes,
                32u * 1024u * 1024u, 2u * 1024u * 1024u),
            26u * 1024u * 1024u);
}

TEST(SharedCodeMemoryBudgetTest, SaturatedDomainIsZero) {
  EXPECT_EQ(iree_hal_amdxdna_shared_code_memory_command_budget(
                kTestSharedCodeMemoryBytes, kTestMissReserveBytes,
                kTestSharedCodeMemoryBytes, 0),
            0u);
  EXPECT_EQ(iree_hal_amdxdna_shared_code_memory_command_budget(
                kTestSharedCodeMemoryBytes, kTestMissReserveBytes, 0,
                kTestSharedCodeMemoryBytes),
            0u);
}

}  // namespace
