// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Include the standalone ABI first to verify that its shared language types
// remain compatible when the full runtime headers are included afterward.
#include "iree/hal/local/executable_plugin.h"

#include <cstring>

#include "iree/hal/local/plugins/static_plugin.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

struct TestPluginState {
  // Allocator retained from the load environment until unload.
  iree_hal_executable_plugin_allocator_t host_allocator;
};

TestPluginState test_plugin_state;

iree_hal_executable_plugin_status_t TestPluginLoad(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  if (param_count != 1 || params[0].key.size != 3 ||
      std::memcmp(params[0].key.data, "key", 3) != 0 ||
      params[0].value.size != 5 ||
      std::memcmp(params[0].value.data, "value", 5) != 0) {
    return iree_hal_executable_plugin_status_from_code(
        IREE_HAL_EXECUTABLE_PLUGIN_STATUS_INVALID_ARGUMENT);
  }
  test_plugin_state.host_allocator = environment->host_allocator;
  *out_self = &test_plugin_state;
  return iree_hal_executable_plugin_ok_status();
}

void TestPluginUnload(void* self) {
  auto* state = static_cast<TestPluginState*>(self);
  state->host_allocator = {};
}

iree_hal_executable_plugin_status_t TestPluginResolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  auto* state = static_cast<TestPluginState*>(self);
  void* allocation = nullptr;
  iree_hal_executable_plugin_status_t status =
      iree_hal_executable_plugin_allocator_malloc(state->host_allocator, 64,
                                                  &allocation);
  if (iree_hal_executable_plugin_status_is_ok(status)) {
    status = iree_hal_executable_plugin_allocator_free(state->host_allocator,
                                                       allocation);
  }
  if (iree_hal_executable_plugin_status_is_ok(status) && params->count == 1) {
    *out_resolution = IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
  }
  return status;
}

const iree_hal_executable_plugin_header_t test_plugin_header = {
    /*.version=*/IREE_HAL_EXECUTABLE_PLUGIN_VERSION_0_1,
    /*.name=*/"type-safe-test-plugin",
    /*.description=*/"Exercises standalone plugin ABI adapters",
    /*.features=*/IREE_HAL_EXECUTABLE_PLUGIN_FEATURE_NONE,
    /*.sanitizer=*/IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_NONE,
    /*.reserved=*/{0},
};

const iree_hal_executable_plugin_v0_t test_plugin = {
    /*.header=*/&test_plugin_header,
    /*.load=*/TestPluginLoad,
    /*.unload=*/TestPluginUnload,
    /*.resolve=*/TestPluginResolve,
};

const iree_hal_executable_plugin_header_t* const* TestPluginQuery(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  (void)reserved;
  return max_version >= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_0_1
             ? &test_plugin.header
             : nullptr;
}

TEST(ExecutablePluginTest, PreservesStandaloneAbiTypesAndAllocatorLifetime) {
  const iree_string_pair_t params[] = {
      iree_make_cstring_pair("key", "value"),
  };
  iree_hal_executable_plugin_t* plugin = nullptr;
  IREE_ASSERT_OK(iree_hal_static_executable_plugin_create(
      TestPluginQuery, IREE_ARRAYSIZE(params), params, iree_allocator_system(),
      &plugin));

  const char* symbol_names[] = {"?missing"};
  void* function_pointers[] = {nullptr};
  void* function_contexts[] = {nullptr};
  iree_hal_executable_import_resolution_t resolution = 0;
  IREE_EXPECT_OK(iree_hal_executable_import_provider_try_resolve(
      iree_hal_executable_plugin_provider(plugin), IREE_ARRAYSIZE(symbol_names),
      symbol_names, function_pointers, function_contexts, &resolution));
  EXPECT_TRUE(iree_all_bits_set(
      resolution, IREE_HAL_EXECUTABLE_IMPORT_RESOLUTION_MISSING_OPTIONAL));

  iree_hal_executable_plugin_release(plugin);
  EXPECT_EQ(test_plugin_state.host_allocator.ctl, nullptr);
}

}  // namespace
}  // namespace iree::hal
