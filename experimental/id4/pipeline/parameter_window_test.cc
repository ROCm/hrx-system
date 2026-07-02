// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_window.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct PlanDeleter {
  void operator()(id4_pipeline_plan_t* plan) const {
    id4_pipeline_plan_release(plan);
  }
};

struct ParameterWindowDeleter {
  void operator()(id4_pipeline_parameter_window_t* window) const {
    id4_pipeline_parameter_window_release(window);
  }
};

using DeviceGroupPtr =
    std::unique_ptr<iree_hal_device_group_t,
                    decltype(&iree_hal_device_group_release)>;
using PlanPtr = std::unique_ptr<id4_pipeline_plan_t, PlanDeleter>;
using ParameterWindowPtr =
    std::unique_ptr<id4_pipeline_parameter_window_t, ParameterWindowDeleter>;

static iree_hal_buffer_params_t MakeStorageParams() {
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  params.min_alignment = 16;
  return params;
}

static id4_pipeline_region_plan_t MakeParameterRegion(
    iree_string_view_t name, const iree_host_size_t* parameter_load_groups,
    iree_host_size_t parameter_load_group_count) {
  id4_pipeline_region_plan_t region;
  std::memset(&region, 0, sizeof(region));
  region.name = name;
  region.placement_id = 0;
  region.binding_capacity = 2;
  region.local_binding_slot = 1;
  region.parameter_load_group_count = parameter_load_group_count;
  region.parameter_load_groups = parameter_load_groups;
  return region;
}

static PlanPtr MakeWindowPlan() {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);

  id4_pipeline_device_placement_t placement = {
      /*.role=*/IREE_SV("default"),
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  const iree_hal_buffer_params_t storage_params = MakeStorageParams();
  const id4_pipeline_parameter_request_t parameter_requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("layers.0.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0,
                                      /*length=*/100)),
      id4_pipeline_parameter_request(
          IREE_SV("layers.1.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/4,
                                      /*buffer_offset=*/100,
                                      /*length=*/200)),
      id4_pipeline_parameter_request(
          IREE_SV("layers.2.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/8,
                                      /*buffer_offset=*/300,
                                      /*length=*/300)),
  };
  const id4_pipeline_parameter_slab_plan_t parameter_slab =
      id4_pipeline_make_parameter_slab_plan(
          IREE_SV("model"), /*placement_id=*/0, /*binding_slot=*/0,
          storage_params, /*byte_length=*/600, /*alignment=*/16,
          IREE_ARRAYSIZE(parameter_requests), parameter_requests);
  const id4_pipeline_parameter_load_step_t parameter_load_steps[] = {
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.layers.0.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/1),
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.layers.1.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/1,
          /*request_count=*/1),
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.layers.2.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/2,
          /*request_count=*/1),
  };
  const iree_host_size_t region0_load_groups[] = {0};
  const iree_host_size_t region1_load_groups[] = {1};
  const iree_host_size_t region2_load_groups[] = {1, 2};
  const id4_pipeline_region_plan_t regions[] = {
      MakeParameterRegion(IREE_SV("region0"), region0_load_groups,
                          IREE_ARRAYSIZE(region0_load_groups)),
      MakeParameterRegion(IREE_SV("region1"), region1_load_groups,
                          IREE_ARRAYSIZE(region1_load_groups)),
      MakeParameterRegion(IREE_SV("region2"), region2_load_groups,
                          IREE_ARRAYSIZE(region2_load_groups)),
  };

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_plan_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.stage_name = IREE_SV("test.plan");
  options.device_group = device_group.get();
  options.placement_count = 1;
  options.placements = &placement;
  options.parameter_slab_count = 1;
  options.parameter_slabs = &parameter_slab;
  options.parameter_load_step_count = IREE_ARRAYSIZE(parameter_load_steps);
  options.parameter_load_steps = parameter_load_steps;
  options.region_count = IREE_ARRAYSIZE(regions);
  options.regions = regions;
  options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* raw_plan = nullptr;
  IREE_CHECK_OK(
      id4_pipeline_plan_create(&options, iree_allocator_system(), &raw_plan));
  return PlanPtr(raw_plan);
}

static ParameterWindowPtr MakeParameterWindow(const id4_pipeline_plan_t* plan,
                                              iree_host_size_t region_offset,
                                              iree_host_size_t region_count) {
  id4_pipeline_parameter_window_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.plan = plan;
  options.region_offset = region_offset;
  options.region_count = region_count;
  id4_pipeline_parameter_window_t* raw_window = nullptr;
  IREE_CHECK_OK(id4_pipeline_parameter_window_create(
      &options, iree_allocator_system(), &raw_window));
  return ParameterWindowPtr(raw_window);
}

TEST(ParameterWindowTest, PacksOnlyRequestsUsedByWindow) {
  PlanPtr plan = MakeWindowPlan();
  ParameterWindowPtr window =
      MakeParameterWindow(plan.get(), /*region_offset=*/1, /*region_count=*/2);

  EXPECT_EQ(id4_pipeline_parameter_window_region_offset(window.get()), 1u);
  EXPECT_EQ(id4_pipeline_parameter_window_region_count(window.get()), 2u);
  ASSERT_EQ(id4_pipeline_parameter_window_slab_count(window.get()), 1u);
  const id4_pipeline_parameter_window_slab_t* slab =
      id4_pipeline_parameter_window_slab_at(window.get(), 0);
  ASSERT_NE(slab, nullptr);
  EXPECT_EQ(slab->original_slab_index, 0u);
  EXPECT_EQ(slab->binding_slot, 0u);
  EXPECT_EQ(slab->byte_length, 508u);
  EXPECT_EQ(slab->alignment, 16u);
  EXPECT_EQ(slab->request_offset, 0u);
  EXPECT_EQ(slab->request_count, 2u);

  ASSERT_EQ(id4_pipeline_parameter_window_request_count(window.get()), 2u);
  const id4_pipeline_parameter_window_request_t* first_request =
      id4_pipeline_parameter_window_request_at(window.get(), 0);
  ASSERT_NE(first_request, nullptr);
  EXPECT_EQ(first_request->original_slab_index, 0u);
  EXPECT_EQ(first_request->original_request_index, 1u);
  EXPECT_EQ(first_request->global_request_index, 1u);
  EXPECT_EQ(first_request->span.parameter_offset, 4u);
  EXPECT_EQ(first_request->span.buffer_offset, 0u);
  EXPECT_EQ(first_request->span.length, 200u);

  const id4_pipeline_parameter_window_request_t* second_request =
      id4_pipeline_parameter_window_request_at(window.get(), 1);
  ASSERT_NE(second_request, nullptr);
  EXPECT_EQ(second_request->original_slab_index, 0u);
  EXPECT_EQ(second_request->original_request_index, 2u);
  EXPECT_EQ(second_request->global_request_index, 2u);
  EXPECT_EQ(second_request->span.parameter_offset, 8u);
  EXPECT_EQ(second_request->span.buffer_offset, 208u);
  EXPECT_EQ(second_request->span.length, 300u);

  EXPECT_EQ(id4_pipeline_parameter_window_load_group_count(window.get()), 2u);
  EXPECT_EQ(id4_pipeline_parameter_window_load_group_at(window.get(), 0), 1u);
  EXPECT_EQ(id4_pipeline_parameter_window_load_group_at(window.get(), 1), 2u);
  EXPECT_EQ(id4_pipeline_parameter_window_load_group_at(window.get(), 2),
            IREE_HOST_SIZE_MAX);

  EXPECT_EQ(id4_pipeline_parameter_window_resolve_request(window.get(), 0),
            nullptr);
  EXPECT_EQ(id4_pipeline_parameter_window_resolve_request(window.get(), 1),
            first_request);
  EXPECT_EQ(id4_pipeline_parameter_window_resolve_request(window.get(), 2),
            second_request);
  EXPECT_EQ(id4_pipeline_parameter_window_resolve_request(window.get(), 3),
            nullptr);
}

TEST(ParameterWindowTest, CoalescesDuplicateGroupsInsideWindow) {
  PlanPtr plan = MakeWindowPlan();
  ParameterWindowPtr window =
      MakeParameterWindow(plan.get(), /*region_offset=*/0, /*region_count=*/3);

  EXPECT_EQ(id4_pipeline_parameter_window_load_group_count(window.get()), 3u);
  EXPECT_EQ(id4_pipeline_parameter_window_request_count(window.get()), 3u);
  ASSERT_EQ(id4_pipeline_parameter_window_slab_count(window.get()), 1u);
  const id4_pipeline_parameter_window_slab_t* slab =
      id4_pipeline_parameter_window_slab_at(window.get(), 0);
  ASSERT_NE(slab, nullptr);
  EXPECT_EQ(slab->byte_length, 620u);
  EXPECT_EQ(id4_pipeline_parameter_window_load_group_at(window.get(), 0), 0u);
  EXPECT_EQ(id4_pipeline_parameter_window_load_group_at(window.get(), 1), 1u);
  EXPECT_EQ(id4_pipeline_parameter_window_load_group_at(window.get(), 2), 2u);
}

TEST(ParameterWindowTest, RejectsInvalidRegionRange) {
  PlanPtr plan = MakeWindowPlan();
  id4_pipeline_parameter_window_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.plan = plan.get();
  options.region_offset = 3;
  options.region_count = 1;
  id4_pipeline_parameter_window_t* window = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        id4_pipeline_parameter_window_create(
                            &options, iree_allocator_system(), &window));
  EXPECT_EQ(window, nullptr);
}

}  // namespace
