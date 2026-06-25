// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/runtime.h"

#include <cstring>
#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  T* value_ = nullptr;
};

TEST(Id4ToolingRuntimeTest, CreatesEmbeddedKernelLibrary) {
  Ref<id4_pipeline_kernel_library_t, id4_pipeline_kernel_library_release>
      library;
  IREE_ASSERT_OK(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), library.out()));

  const id4_pipeline_kernel_module_t* module = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_library_lookup(
      library.get(), IREE_SV("qwen3_vl/rmsnorm"), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(module->module_path, IREE_SV("qwen3_vl/rmsnorm")));
}

TEST(Id4ToolingRuntimeTest, RequiresDeviceFlag) {
  id4_tooling_runtime_context_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.executable_cache_identifier = IREE_SV("id4.test");

  id4_tooling_runtime_context_t context;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_tooling_runtime_context_initialize_from_flags(
                            &options, iree_allocator_system(), &context));
}

TEST(Id4ToolingRuntimeTest, RequiresParameterScopeFromFlags) {
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      id4_tooling_create_parameter_provider_from_flags(
          iree_string_view_empty(), iree_allocator_system(), provider.out()));
}

}  // namespace
