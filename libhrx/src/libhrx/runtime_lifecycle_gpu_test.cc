// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <memory>

#include "hrx_runtime.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T>
T Resolve(iree_dynamic_library_t* library, const char* name) {
  void* symbol = nullptr;
  IREE_EXPECT_OK(iree_dynamic_library_lookup_symbol(library, name, &symbol));
  return reinterpret_cast<T>(symbol);
}

TEST(HrxRuntimeLifecycleTest, ReopensSharedRuntimeAfterGpuShutdown) {
  const char* path = std::getenv("HRX_TEST_RUNTIME_LIBRARY");
  ASSERT_NE(path, nullptr);
  for (int generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    iree_dynamic_library_t* raw_library = nullptr;
    IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
        path, IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(),
        &raw_library));
    std::unique_ptr<iree_dynamic_library_t,
                    decltype(&iree_dynamic_library_release)>
        library(raw_library, iree_dynamic_library_release);
    const auto initialize = Resolve<decltype(&hrx_gpu_initialize)>(
        library.get(), "hrx_gpu_initialize");
    const auto shutdown =
        Resolve<decltype(&hrx_gpu_shutdown)>(library.get(), "hrx_gpu_shutdown");
    const auto device_count = Resolve<decltype(&hrx_gpu_device_count)>(
        library.get(), "hrx_gpu_device_count");
    const auto status_code =
        Resolve<decltype(&hrx_status_code)>(library.get(), "hrx_status_code");
    const auto status_ignore = Resolve<decltype(&hrx_status_ignore)>(
        library.get(), "hrx_status_ignore");
    ASSERT_NE(initialize, nullptr);
    ASSERT_NE(shutdown, nullptr);
    ASSERT_NE(device_count, nullptr);
    ASSERT_NE(status_code, nullptr);
    ASSERT_NE(status_ignore, nullptr);
    const auto consume_status = [&](hrx_status_t status) {
      const hrx_status_code_t code = status_code(status);
      status_ignore(status);
      return code;
    };
    ASSERT_EQ(HRX_STATUS_OK, consume_status(initialize(0)));
    int count = 0;
    EXPECT_EQ(HRX_STATUS_OK, consume_status(device_count(&count)));
    EXPECT_GT(count, 0);
    ASSERT_EQ(HRX_STATUS_OK, consume_status(shutdown()));
    EXPECT_EQ(HRX_STATUS_UNAVAILABLE, consume_status(device_count(&count)));
    // Drop the application's loader reference before reopening. HRX owns the
    // module reference required by its retained native connection.
    library.reset();
  }
}

}  // namespace
