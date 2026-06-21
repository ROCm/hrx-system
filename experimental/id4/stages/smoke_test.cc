// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/smoke.h"

#include <cstring>
#include <string>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/smoke_test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SmokeStage, RunsConfiguredKernelThroughPreparedRegion) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4::test::SmokeExecutableCache* executable_cache = nullptr;
  IREE_ASSERT_OK(id4::test::CreateExecutableCache(iree_allocator_system(),
                                                  &executable_cache));

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(
      id4::test::CreateKernelCache(iree_allocator_system(), &kernel_cache));

  id4_pipeline_stage_t* stage = nullptr;
  IREE_ASSERT_OK(id4::test::CreateSmokeStage(device_group, executable_cache,
                                             kernel_cache,
                                             iree_allocator_system(), &stage));

  id4::test::SmokeDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  EXPECT_EQ(id4_pipeline_plan_memory_slab_count(plan), 1u);
  const id4_pipeline_memory_slab_plan_t* memory_slab =
      id4_pipeline_plan_memory_slab_at(plan, 0);
  ASSERT_NE(memory_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(memory_slab->name), "smoke.local");
  EXPECT_EQ(memory_slab->placement_id, 0u);
  EXPECT_EQ(memory_slab->binding_slot, 1u);
  EXPECT_EQ(memory_slab->byte_length, 16u);
  EXPECT_EQ(memory_slab->alignment, 16u);
  EXPECT_EQ(memory_slab->high_water_mark, 16u);

  EXPECT_EQ(id4_pipeline_plan_kernel_count(plan), 1u);
  const id4_pipeline_kernel_plan_t* kernel =
      id4_pipeline_plan_kernel_at(plan, 0);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(id4::test::ToString(kernel->specialization_key),
            "id4_smoke_configured:workgroups_x=1:workgroup_size_x=64");
  EXPECT_EQ(id4::test::ToString(kernel->function_name), "id4_smoke_configured");
  EXPECT_EQ(kernel->config_binding_count, 2u);
  ASSERT_NE(kernel->config_bindings, nullptr);
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[0].key),
            "id4.smoke.workgroups_x");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[0].value), "1");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[1].key),
            "id4.smoke.workgroup_size_x");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[1].value), "64");

  EXPECT_EQ(id4_pipeline_plan_region_count(plan), 1u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(id4::test::ToString(region->name), "smoke.region");
  EXPECT_EQ(region->binding_capacity, 3u);
  EXPECT_EQ(region->local_binding_slot, 1u);
  EXPECT_EQ(region->statistics.operation_count, 1u);
  EXPECT_EQ(region->statistics.dispatch_count, 1u);
  EXPECT_EQ(region->statistics.local_acquire_count, 1u);
  EXPECT_EQ(region->statistics.bound_import_count, 1u);

  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 1u);
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 0);
  ASSERT_NE(tap, nullptr);
  EXPECT_EQ(id4::test::ToString(tap->name), "smoke.output.after_dispatch");
  EXPECT_EQ(tap->region_id, 0u);
  EXPECT_EQ(tap->after_operation_ordinal, 0u);
  EXPECT_EQ(id4::test::ToString(tap->target_name), "smoke.output");

  iree_string_builder_t plan_json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &plan_json_builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &plan_json_builder));
  std::string plan_json =
      id4::test::ToString(iree_string_builder_view(&plan_json_builder));
  EXPECT_NE(plan_json.find("\"memory_slabs\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"kernels\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"regions\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"diagnostic_taps\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"smoke.output.after_dispatch\""),
            std::string::npos);
  iree_string_builder_deinitialize(&plan_json_builder);

  iree_io_parameter_provider_t* provider =
      id4::test::CreateSmokeParameterProvider();
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = id4::test::CreateSemaphore(device);
  uint64_t ready_value = 1;
  iree_hal_semaphore_list_t prepare_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&ready_semaphore,
      /*.payload_values=*/&ready_value,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = provider;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal_list;
  prepare_options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));

  iree_hal_semaphore_t* issue_semaphore = id4::test::CreateSemaphore(device);
  uint64_t issue_value = 1;
  iree_hal_semaphore_list_t issue_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&issue_semaphore,
      /*.payload_values=*/&issue_value,
  };

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = issue_signal_list;
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_issue(stage, bundle, &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(issue_semaphore, issue_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_t* output_buffer =
      id4_smoke_stage_bundle_output_buffer(bundle);
  ASSERT_NE(output_buffer, nullptr);
  uint32_t output_value = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer, 0, &output_value,
                                          sizeof(output_value)));
  EXPECT_EQ(output_value, 7u);

  EXPECT_EQ(executable_cache->infer_count, 1u);
  EXPECT_EQ(executable_cache->can_prepare_count, 1u);
  EXPECT_EQ(executable_cache->prepare_count, 1u);
  EXPECT_EQ(executable_cache->last_caching_mode,
            IREE_HAL_EXECUTABLE_CACHING_MODE_NONE);
  EXPECT_GT(diagnostics.kernel_event_count, 0u);
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.load"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "plan.create"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "parameter_slab.load"));
  EXPECT_TRUE(
      id4::test::ContainsKey(diagnostics.keys, "parameter_slab.gather"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "kernel_cache.prepare"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.prepare"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.issue"));

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(issue_semaphore);
  iree_hal_semaphore_release(ready_semaphore);
  iree_io_parameter_provider_release(provider);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  id4_pipeline_kernel_cache_release(kernel_cache);
  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache));
  iree_hal_device_group_release(device_group);
}

}  // namespace
