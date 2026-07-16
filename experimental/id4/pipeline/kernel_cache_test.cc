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

static const iree_string_view_t kTestTargetProcessor = IREE_SV("gfx1100");

static iree_hal_device_spec_t* CreateDeviceSpec(
    iree_host_size_t target_count,
    const iree_hal_executable_target_t* targets) {
  const iree_hal_device_executable_spec_t executables = {
      /*.format_count=*/0,
      /*.formats=*/nullptr,
      /*.target_count=*/target_count,
      /*.targets=*/targets,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  const iree_hal_device_spec_params_t params = {
      /*.identity=*/nullptr,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/nullptr,
      /*.dispatch=*/nullptr,
      /*.timing=*/nullptr,
      /*.executables=*/&executables,
      /*.sanitizer=*/nullptr,
      /*.facet_count=*/0,
      /*.facets=*/nullptr,
  };
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_CHECK_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                            &device_spec));
  return device_spec;
}

TEST(KernelCacheTest, CreatesCompilerState) {
  id4_pipeline_kernel_cache_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.target_processor = kTestTargetProcessor;
  options.entry_limit = ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT;

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &options, iree_allocator_system(), &kernel_cache));
  EXPECT_TRUE(iree_string_view_equal(
      id4_pipeline_kernel_cache_target_processor(kernel_cache),
      kTestTargetProcessor));
  id4_pipeline_kernel_cache_release(kernel_cache);
}

TEST(KernelCacheTest, CreateRequiresProcessor) {
  id4_pipeline_kernel_cache_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.entry_limit = ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT;

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_kernel_cache_create(
                            &options, iree_allocator_system(), &kernel_cache));
  EXPECT_EQ(kernel_cache, nullptr);
}

TEST(KernelCacheTest, CreateAllowsNoExecutableRetention) {
  id4_pipeline_kernel_cache_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.target_processor = kTestTargetProcessor;
  options.entry_limit = 0;

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &options, iree_allocator_system(), &kernel_cache));
  id4_pipeline_kernel_cache_release(kernel_cache);
}

TEST(KernelCacheTest, PrepareRequiresRealExecutableCache) {
  id4_pipeline_kernel_cache_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.target_processor = kTestTargetProcessor;
  create_options.entry_limit =
      ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT;

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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_kernel_cache_prepare_executable(
                            kernel_cache, &prepare_options, &executable));
  EXPECT_EQ(executable, nullptr);

  id4_pipeline_kernel_cache_release(kernel_cache);
}

TEST(KernelCacheTest, SelectsExactAmdgpuDeviceTarget) {
  const iree_hal_executable_target_t targets[] = {
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.architecture=*/IREE_SV("gfxip"),
          /*.processor=*/IREE_SV("gfx942"),
          /*.features=*/iree_string_view_empty(),
          /*.artifact_format=*/IREE_SV("amdgpu-hsaco-fb"),
          /*.runtime_abi=*/IREE_SV("hsa"),
          /*.loader_namespace=*/IREE_SV("amdgpu"),
          /*.loader_target=*/IREE_SV("amdgpu-hsaco-fb-gfx942"),
          /*.metadata_schema=*/IREE_SV("amdgpu.hsaco.metadata"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          /*.priority=*/100,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.architecture=*/IREE_SV("gfxip"),
          /*.processor=*/IREE_SV("gfx9-generic"),
          /*.features=*/iree_string_view_empty(),
          /*.artifact_format=*/IREE_SV("amdgpu-hsaco-fb"),
          /*.runtime_abi=*/IREE_SV("hsa"),
          /*.loader_namespace=*/IREE_SV("amdgpu"),
          /*.loader_target=*/IREE_SV("amdgpu-hsaco-fb-gfx9-generic"),
          /*.metadata_schema=*/IREE_SV("amdgpu.hsaco.metadata"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
          /*.priority=*/50,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(IREE_ARRAYSIZE(targets), targets);

  iree_string_view_t target_processor = iree_string_view_empty();
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_select_amdgpu_target_processor(
      device_spec, &target_processor));
  EXPECT_TRUE(iree_string_view_equal(target_processor, IREE_SV("gfx942")));

  iree_hal_device_spec_release(device_spec);
}

TEST(KernelCacheTest, RejectsDeviceWithoutExactAmdgpuTarget) {
  iree_hal_device_spec_t* device_spec = CreateDeviceSpec(0, nullptr);

  iree_string_view_t target_processor = IREE_SV("unchanged");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      id4_pipeline_kernel_cache_select_amdgpu_target_processor(
          device_spec, &target_processor));
  EXPECT_TRUE(iree_string_view_is_empty(target_processor));

  iree_hal_device_spec_release(device_spec);
}

}  // namespace
