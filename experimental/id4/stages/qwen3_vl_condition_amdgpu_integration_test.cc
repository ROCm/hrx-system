// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl_condition.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string g_loom_source_path;
std::string g_device_uri;
std::string g_amdgpu_processor;

using BundleRef =
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>;
using ExecutableCacheRef =
    id4::test::OwningRef<iree_hal_executable_cache_t,
                         iree_hal_executable_cache_release>;
using FileContentsRef =
    id4::test::OwningRef<iree_io_file_contents_t, iree_io_file_contents_free>;
using KernelCacheRef = id4::test::OwningRef<id4_pipeline_kernel_cache_t,
                                            id4_pipeline_kernel_cache_release>;
using PlanRef =
    id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>;
using SemaphoreRef =
    id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using StageRef =
    id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>;

bool ParseIntegrationArguments(int argc, char** argv) {
  if (id4::test::ParseStringArgument(argc, argv,
                                     "--loom_source=", &g_loom_source_path) &&
      id4::test::ParseStringArgument(argc, argv,
                                     "--device_uri=", &g_device_uri) &&
      id4::test::ParseStringArgument(
          argc, argv, "--amdgpu_processor=", &g_amdgpu_processor)) {
    return true;
  }
  std::fprintf(stderr,
               "usage: qwen3_vl_condition_amdgpu_integration_test "
               "--loom_source=<path> --device_uri=<uri> "
               "--amdgpu_processor=<processor>\n");
  return false;
}

TEST(Qwen3VlConditionAmdgpuIntegrationTest, PrepareIssueConditionForward) {
  id4::test::LiveHalDevice live_device;
  IREE_ASSERT_OK(id4::test::CreateLiveHalDevice(
      id4::test::StringView(g_device_uri), &live_device));

  ExecutableCacheRef executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      live_device.device.get(), IREE_SV("id4-qwen3-vl-condition-integration"),
      executable_cache.out()));

  KernelCacheRef kernel_cache;
  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  std::memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.amdgpu_processor =
      id4::test::StringView(g_amdgpu_processor);
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &kernel_cache_options, iree_allocator_system(), kernel_cache.out()));

  FileContentsRef source_file;
  IREE_ASSERT_OK(
      iree_io_file_contents_read(id4::test::StringView(g_loom_source_path),
                                 iree_allocator_system(), source_file.out()));

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live_device.device_group.get();
  services.executable_cache = executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_condition_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = kernel_cache.get();
  create_options.source_identifier = IREE_SV("qwen3_vl_condition_f32.loom");
  create_options.source_contents = source_file.get()->const_buffer;
  create_options.module_name = IREE_SV("id4_qwen3_vl_condition_f32");
  create_options.executable_identifier =
      IREE_SV("id4_qwen3_vl_condition_f32.hsaco");
  create_options.apply_token_weights_function_name =
      IREE_SV("id4_qwen3_vl_condition_apply_token_weights_f32");
  create_options.normalize_token_weights_function_name =
      IREE_SV("id4_qwen3_vl_condition_normalize_token_weights_f32");
  create_options.hidden_row_count = 1;
  create_options.token_count = 4;
  create_options.workgroup_size_x = 256;

  StageRef stage;
  IREE_ASSERT_OK(id4_qwen3_vl_condition_stage_create(
      &create_options, iree_allocator_system(), stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  PlanRef plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;

  BundleRef bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  std::array<float, 4> selected_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 4> token_weights = {0.5f, 1.0f, 1.5f, 2.0f};
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      live_device.device.get(), selected_hidden_states.data(),
      id4_qwen3_vl_condition_stage_bundle_selected_hidden_states_buffer(
          bundle.get()),
      /*target_offset=*/0, sizeof(selected_hidden_states),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      live_device.device.get(), token_weights.data(),
      id4_qwen3_vl_condition_stage_bundle_token_weights_buffer(bundle.get()),
      /*target_offset=*/0, sizeof(token_weights),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  SemaphoreRef completion_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      live_device.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      completion_semaphore.out()));
  iree_hal_semaphore_t* signal_semaphores[] = {completion_semaphore.get()};
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      // Number of final signal semaphores.
      .count = IREE_ARRAYSIZE(signal_semaphores),
      // Final signal semaphore pointer array.
      .semaphores = signal_semaphores,
      // Final signal payload values.
      .payload_values = signal_payload_values,
  };

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      completion_semaphore.get(), signal_payload_values[0],
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::array<float, 4> actual_condition = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      live_device.device.get(),
      id4_qwen3_vl_condition_stage_bundle_condition_buffer(bundle.get()),
      /*source_offset=*/0, actual_condition.data(), sizeof(actual_condition),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  const double original_sum = 10.0;
  const double weighted_sum = 15.0;
  const double scale = original_sum / weighted_sum;
  std::array<float, 4> expected_condition = {
      static_cast<float>(selected_hidden_states[0] * token_weights[0] * scale),
      static_cast<float>(selected_hidden_states[1] * token_weights[1] * scale),
      static_cast<float>(selected_hidden_states[2] * token_weights[2] * scale),
      static_cast<float>(selected_hidden_states[3] * token_weights[3] * scale),
  };
  for (iree_host_size_t i = 0; i < expected_condition.size(); ++i) {
    EXPECT_NEAR(actual_condition[i], expected_condition[i], 1.0e-5f)
        << "condition element " << i;
  }
  EXPECT_GT(diagnostics.event_count, 0u);
  EXPECT_GT(diagnostics.kernel_event_count, 0u);
}

}  // namespace

int main(int argc, char** argv) {
  if (!ParseIntegrationArguments(argc, argv)) return 1;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
