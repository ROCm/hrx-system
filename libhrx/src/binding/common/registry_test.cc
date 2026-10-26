// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/registry.h"

#include <array>

#include "iree/testing/status_matchers.h"

namespace {

TEST(SymbolRegistryTest, EmptyProcessRegistryRetainsNoHeapStorage) {
  auto* registry = iree_hal_streaming_global_symbol_registry();
  ASSERT_EQ(registry->module_count, 0u);
  EXPECT_EQ(registry->modules, nullptr);
  EXPECT_EQ(registry->module_capacity, 0u);
  const uint8_t image[] = {0};
  for (int epoch = 0; epoch < 2; ++epoch) {
    std::array<iree_hal_streaming_module_registration_t*, 32> modules = {};
    for (auto& module : modules) {
      IREE_ASSERT_OK(iree_hal_streaming_global_symbol_registry_register_module(
          registry, image, &module));
    }
    EXPECT_EQ(registry->module_count, modules.size());
    for (auto* module : modules) {
      IREE_ASSERT_OK(
          iree_hal_streaming_global_symbol_registry_unregister_module(registry,
                                                                      module));
    }
    EXPECT_EQ(iree_hal_streaming_global_symbol_registry(), registry);
    EXPECT_EQ(registry->module_count, 0u);
    EXPECT_EQ(registry->module_capacity, 0u);
    EXPECT_EQ(registry->modules, nullptr);
  }
}

}  // namespace
