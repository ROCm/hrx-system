// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "iree/hal/drivers/amdxdna/context_cache.h"
#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/testing/gtest.h"

namespace {

constexpr char kCapacityEnv[] = "IREE_HAL_AMDXDNA_CONTEXT_CACHE_CAPACITY";

void SetCapacityEnv(const char* value) {
#ifdef _WIN32
  ASSERT_EQ(_putenv_s(kCapacityEnv, value), 0);
#else
  ASSERT_EQ(setenv(kCapacityEnv, value, 1), 0);
#endif
}

void UnsetCapacityEnv() {
#ifdef _WIN32
  ASSERT_EQ(_putenv_s(kCapacityEnv, ""), 0);
#else
  ASSERT_EQ(unsetenv(kCapacityEnv), 0);
#endif
}

// The per-architecture budget sizes the cache, so each supported NPU generation
// must map to its characterized value and anything else to 0 (unknown).
TEST(HardwareContextBudgetTest, MapsKnownArchitectures) {
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Phoenix")), 6u);
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Strix")),
            32u);
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_arch(
                IREE_SV("Strix Halo")),
            32u);
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Krackan")),
      32u);
}

TEST(HardwareContextBudgetTest, UnknownArchitectureIsZero) {
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_arch(IREE_SV("Navi")),
            0u);
  EXPECT_EQ(
      iree_hal_amdxdna_hardware_context_budget_for_arch(iree_string_view_empty()),
      0u);
}

TEST(HardwareContextBudgetTest, MapsKnownPciIds) {
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amdxdna_npu_arch_for_pci(0x1022u, 0x1502u, 0u),
      IREE_SV("Phoenix")));
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amdxdna_npu_arch_for_pci(0x1022u, 0x17f0u, 0x11u),
      IREE_SV("Strix")));
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_pci(0x1022u, 0x1502u,
                                                             0u),
            6u);
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_pci(0x1022u, 0x17f0u,
                                                             0x11u),
            32u);
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_hal_amdxdna_npu_arch_for_pci(0x1022u, 0xffffu, 0u)));
  EXPECT_EQ(iree_hal_amdxdna_hardware_context_budget_for_pci(0x8086u, 0x17f0u,
                                                             0u),
            0u);
}

// Capacity precedence with no env override: a nonzero device budget wins, and a
// zero budget (unknown architecture) falls back to the built-in default of 8.
TEST(ContextCacheCapacityTest, PrefersDeviceBudget) {
  UnsetCapacityEnv();
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(32), 32u);
}

TEST(ContextCacheCapacityTest, FallsBackToDefaultWhenBudgetUnknown) {
  UnsetCapacityEnv();
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(0), 8u);
}

// A valid env value overrides the device budget; 0 there disables the bound.
TEST(ContextCacheCapacityTest, EnvironmentOverridesBudget) {
  SetCapacityEnv("4");
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(32), 4u);
  SetCapacityEnv("0");
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(32), 0u);
  UnsetCapacityEnv();
}

// A malformed env value is ignored in favor of the device budget.
TEST(ContextCacheCapacityTest, MalformedEnvIgnored) {
  SetCapacityEnv("notanumber");
  EXPECT_EQ(iree_hal_amdxdna_context_cache_resolve_capacity(32), 32u);
  UnsetCapacityEnv();
}

}  // namespace
