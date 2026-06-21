// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/kernel_cache.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(KernelCacheTest, CreatesAmdgpuCompilerState) {
  id4_pipeline_kernel_cache_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.amdgpu_processor = IREE_SV("gfx1100");

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &options, iree_allocator_system(), &kernel_cache));
  EXPECT_TRUE(iree_string_view_equal(
      id4_pipeline_kernel_cache_amdgpu_processor(kernel_cache),
      IREE_SV("gfx1100")));
  id4_pipeline_kernel_cache_release(kernel_cache);
}

TEST(KernelCacheTest, CreateRequiresProcessor) {
  id4_pipeline_kernel_cache_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  iree_status_t status = id4_pipeline_kernel_cache_create(
      &options, iree_allocator_system(), &kernel_cache);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
  EXPECT_EQ(kernel_cache, nullptr);
}

TEST(KernelCacheTest, PrepareRequiresRealExecutableCache) {
  id4_pipeline_kernel_cache_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.amdgpu_processor = IREE_SV("gfx1100");

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &create_options, iree_allocator_system(), &kernel_cache));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  prepare_options.source_identifier = IREE_SV("kernel.loom");
  prepare_options.module_path = IREE_SV("test/kernel");
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_kernel_executable_t* executable = nullptr;
  iree_status_t status = id4_pipeline_kernel_cache_prepare_executable(
      kernel_cache, &prepare_options, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
  EXPECT_EQ(executable, nullptr);

  id4_pipeline_kernel_cache_release(kernel_cache);
}

}  // namespace
