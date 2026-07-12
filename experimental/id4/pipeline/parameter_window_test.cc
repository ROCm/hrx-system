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

struct ParameterWindowScheduleDeleter {
  void operator()(id4_pipeline_parameter_window_schedule_t* schedule) const {
    id4_pipeline_parameter_window_schedule_release(schedule);
  }
};

using DeviceGroupPtr =
    std::unique_ptr<iree_hal_device_group_t,
                    decltype(&iree_hal_device_group_release)>;
using PlanPtr = std::unique_ptr<id4_pipeline_plan_t, PlanDeleter>;
using ParameterWindowPtr =
    std::unique_ptr<id4_pipeline_parameter_window_t, ParameterWindowDeleter>;
using ParameterWindowSchedulePtr =
    std::unique_ptr<id4_pipeline_parameter_window_schedule_t,
                    ParameterWindowScheduleDeleter>;

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

static id4_pipeline_tensor_shape_t MakeShape1(uint64_t dim0) {
  id4_pipeline_tensor_shape_t shape;
  std::memset(&shape, 0, sizeof(shape));
  shape.rank = 1;
  shape.dims[0] = dim0;
  return shape;
}

static id4_pipeline_tensor_shape_t MakeShape2(uint64_t dim0, uint64_t dim1) {
  id4_pipeline_tensor_shape_t shape;
  std::memset(&shape, 0, sizeof(shape));
  shape.rank = 2;
  shape.dims[0] = dim0;
  shape.dims[1] = dim1;
  return shape;
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
          /*placement_id=*/0, /*binding_slot=*/0, storage_params,
          /*byte_length=*/600, /*alignment=*/16);
  const id4_pipeline_parameter_request_table_t parameter_request_table =
      id4_pipeline_make_parameter_request_table(
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
  options.parameter_request_tables = &parameter_request_table;
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

static PlanPtr MakeDenseTensorWindowPlan() {
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
          IREE_SV("embedding.table"),
          id4_pipeline_parameter_span(/*parameter_offset=*/16,
                                      /*buffer_offset=*/0,
                                      /*length=*/8)),
      id4_pipeline_parameter_request(
          IREE_SV("embedding.table"),
          id4_pipeline_parameter_span(/*parameter_offset=*/40,
                                      /*buffer_offset=*/8,
                                      /*length=*/8)),
      id4_pipeline_parameter_request(
          IREE_SV("next.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/16,
                                      /*length=*/32)),
  };
  const id4_pipeline_parameter_slab_plan_t parameter_slab =
      id4_pipeline_make_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, storage_params,
          /*byte_length=*/48, /*alignment=*/16);
  const id4_pipeline_parameter_request_table_t parameter_request_table =
      id4_pipeline_make_parameter_request_table(
          IREE_ARRAYSIZE(parameter_requests), parameter_requests);
  const id4_pipeline_parameter_tensor_plan_t parameter_tensors[] = {
      {
          /*.layout=*/
          {
              /*.name=*/IREE_SV("embedding.rows"),
              /*.dtype=*/ID4_PIPELINE_TENSOR_DTYPE_BF16,
              /*.shape=*/MakeShape2(2, 4),
              /*.byte_length=*/16,
              /*.alignment=*/16,
          },
          /*.program_tensor_ordinal=*/0,
          /*.parameter_slab_index=*/0,
          /*.request_offset=*/0,
          /*.request_count=*/2,
          /*.global_request_offset=*/0,
          /*.offset=*/0,
      },
      {
          /*.layout=*/
          {
              /*.name=*/IREE_SV("next.weight"),
              /*.dtype=*/ID4_PIPELINE_TENSOR_DTYPE_BF16,
              /*.shape=*/MakeShape2(4, 4),
              /*.byte_length=*/32,
              /*.alignment=*/16,
          },
          /*.program_tensor_ordinal=*/1,
          /*.parameter_slab_index=*/0,
          /*.request_offset=*/2,
          /*.request_count=*/1,
          /*.global_request_offset=*/2,
          /*.offset=*/16,
      },
  };
  const id4_pipeline_parameter_load_step_t parameter_load_steps[] = {
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.embedding.rows"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/2),
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.next.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/2,
          /*request_count=*/1),
  };
  const iree_host_size_t region0_load_groups[] = {0};
  const id4_pipeline_region_plan_t regions[] = {
      MakeParameterRegion(IREE_SV("embedding.region"), region0_load_groups,
                          IREE_ARRAYSIZE(region0_load_groups)),
  };

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_plan_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.stage_name = IREE_SV("test.dense_tensor_window");
  options.device_group = device_group.get();
  options.placement_count = 1;
  options.placements = &placement;
  options.parameter_slab_count = 1;
  options.parameter_slabs = &parameter_slab;
  options.parameter_request_tables = &parameter_request_table;
  options.parameter_tensor_count = IREE_ARRAYSIZE(parameter_tensors);
  options.parameter_tensors = parameter_tensors;
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

static PlanPtr MakeSchedulePlan() {
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
                                      /*length=*/64)),
      id4_pipeline_parameter_request(
          IREE_SV("layers.1.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/64,
                                      /*length=*/64)),
      id4_pipeline_parameter_request(
          IREE_SV("layers.2.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/128,
                                      /*length=*/64)),
      id4_pipeline_parameter_request(
          IREE_SV("layers.3.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/192,
                                      /*length=*/64)),
  };
  const id4_pipeline_parameter_slab_plan_t parameter_slab =
      id4_pipeline_make_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, storage_params,
          /*byte_length=*/256, /*alignment=*/16);
  const id4_pipeline_parameter_request_table_t parameter_request_table =
      id4_pipeline_make_parameter_request_table(
          IREE_ARRAYSIZE(parameter_requests), parameter_requests);

  const id4_pipeline_parameter_load_source_t encode1_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("fp8"), IREE_SV("layers.1.weight"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, MakeShape2(32, 1),
          /*byte_length=*/32),
      id4_pipeline_parameter_load_source(
          IREE_SV("fp8"), IREE_SV("layers.1.scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, MakeShape1(32),
          /*byte_length=*/128),
  };
  const id4_pipeline_parameter_load_source_t encode2_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("fp8"), IREE_SV("layers.2.weight"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, MakeShape2(32, 1),
          /*byte_length=*/32),
      id4_pipeline_parameter_load_source(
          IREE_SV("fp8"), IREE_SV("layers.2.scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, MakeShape1(32),
          /*byte_length=*/128),
  };
  const iree_host_size_t indexed_requests[] = {3, 1};
  id4_pipeline_parameter_load_step_t parameter_load_steps[] = {
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.layers.0.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/1),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("encode.layers.1.weight"), IREE_ARRAYSIZE(encode1_sources),
          encode1_sources,
          /*target_slab_index=*/0, /*request_offset=*/1),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("encode.layers.2.weight"), IREE_ARRAYSIZE(encode2_sources),
          encode2_sources,
          /*target_slab_index=*/0, /*request_offset=*/2),
      id4_pipeline_parameter_indexed_gather_load_step(
          IREE_SV("gather.layers.3.and.1.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, IREE_ARRAYSIZE(indexed_requests),
          indexed_requests),
  };
  parameter_load_steps[1].readiness_group_key = 7;
  parameter_load_steps[2].readiness_group_key = 7;

  const iree_host_size_t region0_load_groups[] = {0};
  const iree_host_size_t region1_load_groups[] = {1};
  const iree_host_size_t region2_load_groups[] = {2};
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
  options.stage_name = IREE_SV("test.schedule");
  options.device_group = device_group.get();
  options.placement_count = 1;
  options.placements = &placement;
  options.parameter_slab_count = 1;
  options.parameter_slabs = &parameter_slab;
  options.parameter_request_tables = &parameter_request_table;
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

static ParameterWindowSchedulePtr MakeParameterWindowSchedule(
    id4_pipeline_plan_t* plan, const id4_pipeline_parameter_window_t* window) {
  id4_pipeline_parameter_window_schedule_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.plan = plan;
  options.window = window;
  id4_pipeline_parameter_window_schedule_t* raw_schedule = nullptr;
  IREE_CHECK_OK(id4_pipeline_parameter_window_schedule_create(
      &options, iree_allocator_system(), &raw_schedule));
  return ParameterWindowSchedulePtr(raw_schedule);
}

static id4_pipeline_parameter_window_statistics_t QueryWindowStatistics(
    const id4_pipeline_plan_t* plan, iree_host_size_t concurrent_window_count) {
  id4_pipeline_parameter_window_statistics_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.plan = plan;
  options.concurrent_window_count = concurrent_window_count;
  options.encoder_staging_chunk_byte_capacity =
      ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
  id4_pipeline_parameter_window_statistics_t statistics;
  IREE_CHECK_OK(id4_pipeline_parameter_window_query_statistics(
      &options, iree_allocator_system(), &statistics));
  return statistics;
}

TEST(ParameterWindowTest, ReportsIndependentlyAllocatedSlidingWindows) {
  PlanPtr plan = MakeWindowPlan();

  id4_pipeline_parameter_window_statistics_t statistics =
      QueryWindowStatistics(plan.get(), /*concurrent_window_count=*/1);
  EXPECT_EQ(statistics.concurrent_window_count, 1u);
  EXPECT_EQ(statistics.window_count, 3u);
  EXPECT_EQ(statistics.full_slab_target_byte_length, 600u);
  EXPECT_EQ(statistics.peak_target_byte_length, 508u);
  EXPECT_EQ(statistics.encoder_staging_byte_length, 0u);
  EXPECT_EQ(statistics.peak_live_byte_length, 508u);
  EXPECT_EQ(statistics.peak_source_transfer_byte_length, 500u);
  EXPECT_EQ(statistics.total_target_byte_length, 808u);
  EXPECT_EQ(statistics.total_source_transfer_byte_length, 800u);
  EXPECT_EQ(statistics.peak_load_group_count, 2u);
  EXPECT_EQ(statistics.total_load_group_count, 4u);
  EXPECT_EQ(statistics.largest_load_group_target_byte_length, 300u);
  EXPECT_EQ(statistics.largest_load_group_index, 2u);
  EXPECT_EQ(statistics.largest_request_target_byte_length, 300u);
  EXPECT_EQ(statistics.largest_request_index, 2u);
  EXPECT_EQ(statistics.largest_request_load_group_index, 2u);

  statistics = QueryWindowStatistics(plan.get(), /*concurrent_window_count=*/2);
  EXPECT_EQ(statistics.window_count, 3u);
  EXPECT_EQ(statistics.peak_target_byte_length, 708u);
  EXPECT_EQ(statistics.peak_source_transfer_byte_length, 700u);
  EXPECT_EQ(statistics.peak_load_group_count, 3u);
  EXPECT_EQ(statistics.total_target_byte_length, 808u);
  EXPECT_EQ(statistics.total_load_group_count, 4u);

  statistics = QueryWindowStatistics(plan.get(), /*concurrent_window_count=*/3);
  EXPECT_EQ(statistics.window_count, 3u);
  EXPECT_EQ(statistics.peak_target_byte_length, 808u);
  EXPECT_EQ(statistics.peak_source_transfer_byte_length, 800u);
  EXPECT_EQ(statistics.peak_load_group_count, 4u);
  EXPECT_EQ(statistics.total_target_byte_length, 808u);
  EXPECT_EQ(statistics.total_load_group_count, 4u);
}

TEST(ParameterWindowTest, IncludesEncoderStagingInLivePeak) {
  PlanPtr plan = MakeSchedulePlan();

  id4_pipeline_parameter_window_statistics_t statistics =
      QueryWindowStatistics(plan.get(), /*concurrent_window_count=*/1);
  EXPECT_EQ(statistics.full_slab_target_byte_length, 256u);
  EXPECT_EQ(statistics.peak_target_byte_length, 128u);
  EXPECT_EQ(statistics.encoder_staging_byte_length, 320u);
  EXPECT_EQ(statistics.peak_live_byte_length, 448u);
  EXPECT_EQ(statistics.peak_source_transfer_byte_length, 320u);
  EXPECT_EQ(statistics.total_target_byte_length, 320u);
  EXPECT_EQ(statistics.total_source_transfer_byte_length, 512u);
  EXPECT_EQ(statistics.peak_encode_load_step_count, 2u);
  EXPECT_EQ(statistics.total_encode_load_step_count, 2u);

  statistics = QueryWindowStatistics(plan.get(), /*concurrent_window_count=*/2);
  EXPECT_EQ(statistics.peak_target_byte_length, 256u);
  EXPECT_EQ(statistics.encoder_staging_byte_length, 320u);
  EXPECT_EQ(statistics.peak_live_byte_length, 576u);
  EXPECT_EQ(statistics.peak_source_transfer_byte_length, 448u);
  EXPECT_EQ(statistics.peak_load_group_count, 2u);
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

TEST(ParameterWindowTest, PreservesDenseTensorLayoutAcrossSourceSpans) {
  PlanPtr plan = MakeDenseTensorWindowPlan();
  ParameterWindowPtr window =
      MakeParameterWindow(plan.get(), /*region_offset=*/0, /*region_count=*/1);

  ASSERT_EQ(id4_pipeline_parameter_window_slab_count(window.get()), 1u);
  const id4_pipeline_parameter_window_slab_t* slab =
      id4_pipeline_parameter_window_slab_at(window.get(), 0);
  ASSERT_NE(slab, nullptr);
  EXPECT_EQ(slab->byte_length, 16u);
  EXPECT_EQ(slab->request_count, 2u);

  ASSERT_EQ(id4_pipeline_parameter_window_request_count(window.get()), 2u);
  const id4_pipeline_parameter_window_request_t* first_request =
      id4_pipeline_parameter_window_request_at(window.get(), 0);
  ASSERT_NE(first_request, nullptr);
  EXPECT_EQ(first_request->original_request_index, 0u);
  EXPECT_EQ(first_request->span.parameter_offset, 16u);
  EXPECT_EQ(first_request->span.buffer_offset, 0u);
  EXPECT_EQ(first_request->span.length, 8u);

  const id4_pipeline_parameter_window_request_t* second_request =
      id4_pipeline_parameter_window_request_at(window.get(), 1);
  ASSERT_NE(second_request, nullptr);
  EXPECT_EQ(second_request->original_request_index, 1u);
  EXPECT_EQ(second_request->span.parameter_offset, 40u);
  EXPECT_EQ(second_request->span.buffer_offset, 8u);
  EXPECT_EQ(second_request->span.length, 8u);
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

TEST(ParameterWindowScheduleTest, RewritesLoadsAndStepsForWindow) {
  PlanPtr plan = MakeSchedulePlan();
  ParameterWindowPtr window =
      MakeParameterWindow(plan.get(), /*region_offset=*/1, /*region_count=*/2);
  ParameterWindowSchedulePtr schedule =
      MakeParameterWindowSchedule(plan.get(), window.get());

  ASSERT_EQ(id4_pipeline_parameter_window_schedule_load_count(schedule.get()),
            1u);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule.get());
  ASSERT_NE(loads, nullptr);
  EXPECT_EQ(loads[0].slab_index, 0u);
  ASSERT_NE(loads[0].slab, nullptr);
  ASSERT_NE(loads[0].request_table, nullptr);
  EXPECT_EQ(loads[0].slab->binding_slot, 0u);
  EXPECT_EQ(loads[0].slab->byte_length, 192u);
  ASSERT_EQ(loads[0].request_table->count, 3u);
  EXPECT_TRUE(iree_string_view_equal(loads[0].request_table->values[0].key,
                                     IREE_SV("layers.1.weight")));
  EXPECT_EQ(loads[0].request_table->values[0].span.buffer_offset, 0u);
  EXPECT_TRUE(iree_string_view_equal(loads[0].request_table->values[1].key,
                                     IREE_SV("layers.2.weight")));
  EXPECT_EQ(loads[0].request_table->values[1].span.buffer_offset, 64u);
  EXPECT_TRUE(iree_string_view_equal(loads[0].request_table->values[2].key,
                                     IREE_SV("layers.3.weight")));
  EXPECT_EQ(loads[0].request_table->values[2].span.buffer_offset, 128u);

  ASSERT_EQ(
      id4_pipeline_parameter_window_schedule_load_step_count(schedule.get()),
      3u);
  const id4_pipeline_parameter_load_step_t* load_steps =
      id4_pipeline_parameter_window_schedule_load_steps(schedule.get());
  ASSERT_NE(load_steps, nullptr);
  EXPECT_EQ(
      load_steps[0].kind,
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16);
  EXPECT_EQ(load_steps[0].target_slab_index, 0u);
  EXPECT_EQ(load_steps[0].request_offset, 0u);
  EXPECT_EQ(load_steps[0].readiness_group_key, 1u);
  ASSERT_EQ(load_steps[0].source_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(load_steps[0].sources[0].key,
                                     IREE_SV("layers.1.weight")));

  EXPECT_EQ(
      load_steps[1].kind,
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16);
  EXPECT_EQ(load_steps[1].target_slab_index, 0u);
  EXPECT_EQ(load_steps[1].request_offset, 1u);
  EXPECT_EQ(load_steps[1].readiness_group_key, 1u);

  EXPECT_EQ(load_steps[2].kind, ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER);
  EXPECT_EQ(load_steps[2].target_slab_index, 0u);
  ASSERT_NE(load_steps[2].request_indices, nullptr);
  ASSERT_EQ(load_steps[2].request_count, 2u);
  EXPECT_EQ(load_steps[2].request_indices[0], 2u);
  EXPECT_EQ(load_steps[2].request_indices[1], 0u);

  ASSERT_EQ(
      id4_pipeline_parameter_window_schedule_load_group_count(schedule.get()),
      2u);
  EXPECT_EQ(id4_pipeline_parameter_window_schedule_original_load_group_at(
                schedule.get(), 0),
            1u);
  EXPECT_EQ(id4_pipeline_parameter_window_schedule_original_load_group_at(
                schedule.get(), 1),
            2u);
  EXPECT_EQ(id4_pipeline_parameter_window_schedule_compact_load_group(
                schedule.get(), 0),
            IREE_HOST_SIZE_MAX);
  EXPECT_EQ(id4_pipeline_parameter_window_schedule_compact_load_group(
                schedule.get(), 1),
            0u);
  EXPECT_EQ(id4_pipeline_parameter_window_schedule_compact_load_group(
                schedule.get(), 2),
            1u);

  iree_host_size_t compact_group_count = 0;
  IREE_ASSERT_OK(id4_pipeline_parameter_load_group_count(
      id4_pipeline_parameter_window_schedule_load_step_count(schedule.get()),
      load_steps, &compact_group_count));
  EXPECT_EQ(compact_group_count, 2u);
}

}  // namespace
