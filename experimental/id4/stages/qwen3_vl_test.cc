// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/qwen3_vl_test_util.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/memory_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/numpy_io.h"

namespace {

static std::string JoinFixturePath(const char* root,
                                   const char* relative_path) {
  std::string path(root);
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  path.append(relative_path);
  return path;
}

static iree_status_t LoadNpyF32Matrix(const std::string& path,
                                      iree_hal_device_t* device,
                                      iree_hal_dim_t expected_row_count,
                                      iree_hal_dim_t expected_column_count,
                                      std::vector<float>* out_values) {
  out_values->clear();

  iree_io_file_contents_t* contents = nullptr;
  iree_io_stream_t* stream = nullptr;
  iree_hal_buffer_view_t* buffer_view = nullptr;

  iree_status_t status = iree_io_file_contents_read(
      iree_make_string_view(path.data(), path.size()), iree_allocator_system(),
      &contents);
  if (iree_status_is_ok(status)) {
    status = iree_io_memory_stream_wrap(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_SEEKABLE,
        contents->buffer, iree_io_stream_release_callback_null(),
        iree_allocator_system(), &stream);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t buffer_params;
    std::memset(&buffer_params, 0, sizeof(buffer_params));
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
    buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    status = iree_numpy_npy_load_ndarray(
        stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params, device,
        iree_hal_device_allocator(device), &buffer_view);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_buffer_view_shape_rank(buffer_view) != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "fixture %s must have rank 2", path.c_str());
  }
  if (iree_status_is_ok(status) &&
      (iree_hal_buffer_view_shape_dim(buffer_view, 0) != expected_row_count ||
       iree_hal_buffer_view_shape_dim(buffer_view, 1) !=
           expected_column_count)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "fixture %s has shape [%" PRIu64 ", %" PRIu64
                              "], expected [%" PRIu64 ", %" PRIu64 "]",
                              path.c_str(),
                              iree_hal_buffer_view_shape_dim(buffer_view, 0),
                              iree_hal_buffer_view_shape_dim(buffer_view, 1),
                              expected_row_count, expected_column_count);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_buffer_view_element_type(buffer_view) !=
          IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "fixture %s must contain f32 values", path.c_str());
  }
  if (iree_status_is_ok(status) &&
      iree_hal_buffer_view_encoding_type(buffer_view) !=
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "fixture %s must be dense row-major", path.c_str());
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
    const iree_device_size_t byte_length = iree_hal_buffer_byte_length(buffer);
    if ((byte_length % sizeof(float)) != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "fixture %s byte length is not f32-aligned",
                                path.c_str());
    } else {
      out_values->resize(byte_length / sizeof(float));
      status =
          iree_hal_buffer_map_read(buffer, 0, out_values->data(), byte_length);
    }
  }

  iree_hal_buffer_view_release(buffer_view);
  iree_io_stream_release(stream);
  iree_io_file_contents_free(contents);
  return status;
}

TEST(Qwen3VlStage, RunsConditionForwardThroughPreparedRegion) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4::test::Qwen3VlExecutableCache* executable_cache = nullptr;
  IREE_ASSERT_OK(id4::test::CreateQwen3VlExecutableCache(
      iree_allocator_system(), &executable_cache));

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(
      id4::test::CreateKernelCache(iree_allocator_system(), &kernel_cache));

  id4_pipeline_stage_t* stage = nullptr;
  IREE_ASSERT_OK(id4::test::CreateQwen3VlStage(
      device_group, executable_cache, kernel_cache, iree_allocator_system(),
      &stage));

  id4::test::StageDiagnostics diagnostics = {};
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

  EXPECT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_EQ(id4_pipeline_plan_memory_slab_count(plan), 2u);
  const id4_pipeline_memory_slab_plan_t* selected_slab =
      id4_pipeline_plan_memory_slab_at(plan, 0);
  ASSERT_NE(selected_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(selected_slab->name),
            "qwen3_vl.selected_hidden_states");
  EXPECT_EQ(selected_slab->binding_slot, 0u);
  EXPECT_EQ(selected_slab->byte_length, 8192u);
  EXPECT_EQ(selected_slab->high_water_mark, 8192u);
  const id4_pipeline_memory_slab_plan_t* condition_slab =
      id4_pipeline_plan_memory_slab_at(plan, 1);
  ASSERT_NE(condition_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(condition_slab->name), "qwen3_vl.condition");
  EXPECT_EQ(condition_slab->binding_slot, 1u);
  EXPECT_EQ(condition_slab->byte_length, 8192u);
  EXPECT_EQ(condition_slab->high_water_mark, 8192u);

  EXPECT_EQ(id4_pipeline_plan_kernel_count(plan), 1u);
  const id4_pipeline_kernel_plan_t* kernel =
      id4_pipeline_plan_kernel_at(plan, 0);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(id4::test::ToString(kernel->specialization_key),
            "id4_qwen3_vl_condition_forward_f32:element_count=2048:"
            "workgroup_size_x=256");
  EXPECT_EQ(id4::test::ToString(kernel->function_name),
            "id4_qwen3_vl_condition_forward_f32");
  EXPECT_EQ(kernel->config_binding_count, 2u);
  ASSERT_NE(kernel->config_bindings, nullptr);
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[0].key),
            "id4.qwen3_vl.condition.element_count");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[0].value), "2048");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[1].key),
            "id4.qwen3_vl.condition.workgroup_size_x");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[1].value), "256");

  EXPECT_EQ(id4_pipeline_plan_region_count(plan), 1u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(id4::test::ToString(region->name), "qwen3_vl.condition_forward");
  EXPECT_EQ(region->binding_capacity, 3u);
  EXPECT_EQ(region->local_binding_slot, 2u);
  EXPECT_EQ(region->statistics.operation_count, 1u);
  EXPECT_EQ(region->statistics.dispatch_count, 1u);
  EXPECT_EQ(region->statistics.local_acquire_count, 0u);
  EXPECT_EQ(region->statistics.bound_import_count, 2u);
  EXPECT_EQ(region->statistics.local_slab_byte_length, 0u);
  EXPECT_EQ(region->statistics.local_slab_high_water_mark, 0u);

  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 2u);
  const id4_pipeline_diagnostic_tap_plan_t* selected_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 0);
  ASSERT_NE(selected_tap, nullptr);
  EXPECT_EQ(id4::test::ToString(selected_tap->name),
            "qwen3_vl.selected_hidden_states.before_forward");
  EXPECT_EQ(id4::test::ToString(selected_tap->target_name),
            "qwen3_vl.encoder.selected_hidden_states");
  const id4_pipeline_diagnostic_tap_plan_t* condition_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 1);
  ASSERT_NE(condition_tap, nullptr);
  EXPECT_EQ(id4::test::ToString(condition_tap->name),
            "qwen3_vl.condition.after_forward");
  EXPECT_EQ(id4::test::ToString(condition_tap->target_name),
            "qwen3_vl.encoder.condition");

  iree_string_builder_t plan_json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &plan_json_builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &plan_json_builder));
  std::string plan_json =
      id4::test::ToString(iree_string_builder_view(&plan_json_builder));
  EXPECT_NE(plan_json.find("\"qwen3_vl\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"memory_slabs\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"kernels\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"regions\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"diagnostic_taps\""), std::string::npos);
  iree_string_builder_deinitialize(&plan_json_builder);

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));

  iree_hal_buffer_t* selected_buffer =
      id4_qwen3_vl_stage_bundle_selected_hidden_states_buffer(bundle);
  ASSERT_NE(selected_buffer, nullptr);
  iree_hal_buffer_t* condition_buffer =
      id4_qwen3_vl_stage_bundle_condition_buffer(bundle);
  ASSERT_NE(condition_buffer, nullptr);
  const iree_device_size_t condition_byte_length =
      id4_qwen3_vl_stage_bundle_condition_byte_length(bundle);
  EXPECT_EQ(condition_byte_length, 8192u);

  constexpr iree_host_size_t kElementCount = 2048;
  std::vector<float> selected_values(kElementCount);
  for (iree_host_size_t i = 0; i < selected_values.size(); ++i) {
    selected_values[i] = static_cast<float>(i) * 0.125f - 17.0f;
  }
  IREE_ASSERT_OK(iree_hal_buffer_map_write(selected_buffer, /*target_offset=*/0,
                                           selected_values.data(),
                                           condition_byte_length));

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
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

  std::vector<float> condition_values(kElementCount, 0.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(condition_buffer, /*source_offset=*/0,
                                          condition_values.data(),
                                          condition_byte_length));
  EXPECT_EQ(condition_values, selected_values);

  EXPECT_EQ(executable_cache->infer_count, 1u);
  EXPECT_EQ(executable_cache->can_prepare_count, 1u);
  EXPECT_EQ(executable_cache->prepare_count, 1u);
  EXPECT_EQ(executable_cache->last_caching_mode,
            IREE_HAL_EXECUTABLE_CACHING_MODE_NONE);
  EXPECT_GT(diagnostics.kernel_event_count, 0u);
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.load"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "plan.create"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "kernel_cache.prepare"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.prepare"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.issue"));

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(issue_semaphore);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  id4_pipeline_kernel_cache_release(kernel_cache);
  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache));
  iree_hal_device_group_release(device_group);
}

TEST(Qwen3VlStage, MatchesReferenceConditionFixtureWhenAvailable) {
  const char* fixture_root = std::getenv("ID4_REFERENCE_FIXTURE_ROOT");
  if (!fixture_root || fixture_root[0] == 0) {
    GTEST_SKIP()
        << "ID4_REFERENCE_FIXTURE_ROOT is not set; skipping Qwen fixture test";
  }

  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);

  std::vector<float> selected_values;
  IREE_ASSERT_OK(LoadNpyF32Matrix(
      JoinFixturePath(fixture_root,
                      "qwen/selected_hidden_states_0_32x0_64.npy"),
      device, /*expected_row_count=*/32, /*expected_column_count=*/64,
      &selected_values));
  std::vector<float> expected_condition_values;
  IREE_ASSERT_OK(LoadNpyF32Matrix(
      JoinFixturePath(fixture_root, "qwen/condition_0_32x0_64.npy"), device,
      /*expected_row_count=*/32, /*expected_column_count=*/64,
      &expected_condition_values));
  ASSERT_EQ(selected_values.size(), expected_condition_values.size());

  id4::test::Qwen3VlExecutableCache* executable_cache = nullptr;
  IREE_ASSERT_OK(id4::test::CreateQwen3VlExecutableCache(
      iree_allocator_system(), &executable_cache));

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(
      id4::test::CreateKernelCache(iree_allocator_system(), &kernel_cache));

  id4_pipeline_stage_t* stage = nullptr;
  IREE_ASSERT_OK(id4::test::CreateQwen3VlStage(
      device_group, executable_cache, kernel_cache, iree_allocator_system(),
      &stage));

  id4::test::StageDiagnostics diagnostics = {};
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

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(
      id4_pipeline_stage_prepare(stage, plan, &prepare_options, &bundle));

  iree_hal_buffer_t* selected_buffer =
      id4_qwen3_vl_stage_bundle_selected_hidden_states_buffer(bundle);
  ASSERT_NE(selected_buffer, nullptr);
  const iree_device_size_t condition_byte_length =
      id4_qwen3_vl_stage_bundle_condition_byte_length(bundle);
  ASSERT_EQ(condition_byte_length, selected_values.size() * sizeof(float));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(selected_buffer, /*target_offset=*/0,
                                           selected_values.data(),
                                           condition_byte_length));

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

  iree_hal_buffer_t* condition_buffer =
      id4_qwen3_vl_stage_bundle_condition_buffer(bundle);
  ASSERT_NE(condition_buffer, nullptr);
  std::vector<float> actual_condition_values(selected_values.size(), 0.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(condition_buffer, /*source_offset=*/0,
                                          actual_condition_values.data(),
                                          condition_byte_length));
  EXPECT_EQ(actual_condition_values, expected_condition_values);

  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(issue_semaphore);
  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  id4_pipeline_kernel_cache_release(kernel_cache);
  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache));
  iree_hal_device_group_release(device_group);
}

}  // namespace
