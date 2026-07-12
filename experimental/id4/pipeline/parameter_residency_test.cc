// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_residency.h"

#include <cstring>
#include <memory>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct PlanDeleter {
  void operator()(id4_pipeline_plan_t* plan) const {
    id4_pipeline_plan_release(plan);
  }
};

struct ResidencyPlanDeleter {
  void operator()(id4_pipeline_parameter_residency_plan_t* plan) const {
    id4_pipeline_parameter_residency_plan_release(plan);
  }
};

using DeviceGroupPtr =
    std::unique_ptr<iree_hal_device_group_t,
                    decltype(&iree_hal_device_group_release)>;
using PlanPtr = std::unique_ptr<id4_pipeline_plan_t, PlanDeleter>;
using ResidencyPlanPtr =
    std::unique_ptr<id4_pipeline_parameter_residency_plan_t,
                    ResidencyPlanDeleter>;

static id4_pipeline_program_tensor_t AddDirectParameter(
    id4_pipeline_program_builder_t* builder, iree_string_view_t name,
    uint64_t element_count) {
  const id4_pipeline_program_parameter_source_t source = {
      /*.source_scope=*/IREE_SV("model"),
      /*.key=*/name,
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(element_count),
  };
  id4_pipeline_program_parameter_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/1,
      /*.sources=*/&source,
      /*.key=*/name,
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(element_count),
  };
  id4_pipeline_program_tensor_t tensor = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(id4_pipeline_program_parameter(builder, &options, &tensor));
  return tensor;
}

static void AddReadDispatch(id4_pipeline_program_builder_t* builder,
                            iree_string_view_t name,
                            id4_pipeline_program_tensor_t tensor) {
  const id4_pipeline_program_dispatch_binding_t binding =
      id4_pipeline_program_read(tensor);
  id4_pipeline_program_dispatch_loom_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.name=*/name,
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/read"), IREE_SV("read")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/1,
      /*.bindings=*/&binding,
  };
  IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder, &options));
}

static void AddBarrier(id4_pipeline_program_builder_t* builder,
                       iree_string_view_t name) {
  id4_pipeline_program_barrier_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.name=*/name,
  };
  IREE_CHECK_OK(id4_pipeline_program_barrier(builder, &options));
}

static PlanPtr MakeResidencyPlan() {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                   iree_allocator_system(), &block_pool);
  id4_pipeline_program_builder_create_options_t builder_options = {
      /*.structure_size=*/sizeof(builder_options),
      /*.next=*/nullptr,
      /*.program_name=*/IREE_SV("test.residency"),
      /*.block_pool=*/&block_pool,
  };
  id4_pipeline_program_builder_t* builder = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_create(
      &builder_options, iree_allocator_system(), &builder));
  const id4_pipeline_program_tensor_t parameter_a =
      AddDirectParameter(builder, IREE_SV("parameter.a"), 32);
  const id4_pipeline_program_tensor_t parameter_b =
      AddDirectParameter(builder, IREE_SV("parameter.b"), 40);
  const id4_pipeline_program_tensor_t parameter_c =
      AddDirectParameter(builder, IREE_SV("parameter.c"), 48);
  AddReadDispatch(builder, IREE_SV("read.a.0"), parameter_a);
  AddReadDispatch(builder, IREE_SV("read.b"), parameter_b);
  AddBarrier(builder, IREE_SV("after.a.b"));
  AddReadDispatch(builder, IREE_SV("read.c"), parameter_c);
  AddBarrier(builder, IREE_SV("after.c"));
  AddReadDispatch(builder, IREE_SV("read.a.1"), parameter_a);

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  id4_pipeline_program_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);

  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);
  const id4_pipeline_device_placement_t placement = {
      /*.role=*/IREE_SV("default"),
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  iree_hal_buffer_params_t buffer_params;
  std::memset(&buffer_params, 0, sizeof(buffer_params));
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  buffer_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  buffer_params.min_alignment = 16;
  const id4_pipeline_parameter_slab_plan_t parameter_slab =
      id4_pipeline_make_parameter_slab_plan(
          /*placement_id=*/0, /*binding_slot=*/0, buffer_params,
          /*byte_length=*/240, /*alignment=*/16);
  const id4_pipeline_parameter_request_t requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("parameter.a"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0, /*length=*/64)),
      id4_pipeline_parameter_request(
          IREE_SV("parameter.b"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/64, /*length=*/80)),
      id4_pipeline_parameter_request(
          IREE_SV("parameter.c"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/144, /*length=*/96)),
  };
  const id4_pipeline_parameter_request_table_t request_table =
      id4_pipeline_make_parameter_request_table(IREE_ARRAYSIZE(requests),
                                                requests);
  const id4_pipeline_parameter_tensor_plan_t parameter_tensors[] = {
      {
          /*.layout=*/{IREE_SV("parameter.a"),
                       ID4_PIPELINE_TENSOR_DTYPE_BF16,
                       {/*.rank=*/1, /*.dims=*/{32}},
                       64,
                       16},
          /*.program_tensor_ordinal=*/parameter_a.ordinal,
          /*.parameter_slab_index=*/0,
          /*.request_offset=*/0,
          /*.request_count=*/1,
          /*.global_request_offset=*/0,
          /*.offset=*/0,
      },
      {
          /*.layout=*/{IREE_SV("parameter.b"),
                       ID4_PIPELINE_TENSOR_DTYPE_BF16,
                       {/*.rank=*/1, /*.dims=*/{40}},
                       80,
                       16},
          /*.program_tensor_ordinal=*/parameter_b.ordinal,
          /*.parameter_slab_index=*/0,
          /*.request_offset=*/1,
          /*.request_count=*/1,
          /*.global_request_offset=*/1,
          /*.offset=*/64,
      },
      {
          /*.layout=*/{IREE_SV("parameter.c"),
                       ID4_PIPELINE_TENSOR_DTYPE_BF16,
                       {/*.rank=*/1, /*.dims=*/{48}},
                       96,
                       16},
          /*.program_tensor_ordinal=*/parameter_c.ordinal,
          /*.parameter_slab_index=*/0,
          /*.request_offset=*/2,
          /*.request_count=*/1,
          /*.global_request_offset=*/2,
          /*.offset=*/144,
      },
  };
  const id4_pipeline_parameter_load_step_t load_steps[] = {
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.parameter.a"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/0, /*request_count=*/1),
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.parameter.b"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/1, /*request_count=*/1),
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.parameter.c"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/2, /*request_count=*/1),
  };
  const iree_host_size_t region_load_groups[] = {0, 1, 2};
  id4_pipeline_region_plan_t region;
  std::memset(&region, 0, sizeof(region));
  region.name = IREE_SV("forward");
  region.source_operation_offset = 3;
  region.source_operation_count = 6;
  region.placement_id = 0;
  region.binding_capacity = 2;
  region.local_binding_slot = 1;
  region.local_tensor_alignment = 16;
  region.parameter_load_group_count = IREE_ARRAYSIZE(region_load_groups);
  region.parameter_load_groups = region_load_groups;

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_plan_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.stage_name = IREE_SV("test.residency");
  options.device_group = device_group.get();
  options.source_program = program;
  options.placement_count = 1;
  options.placements = &placement;
  options.parameter_slab_count = 1;
  options.parameter_slabs = &parameter_slab;
  options.parameter_request_tables = &request_table;
  options.parameter_tensor_count = IREE_ARRAYSIZE(parameter_tensors);
  options.parameter_tensors = parameter_tensors;
  options.parameter_load_step_count = IREE_ARRAYSIZE(load_steps);
  options.parameter_load_steps = load_steps;
  options.region_count = 1;
  options.regions = &region;
  options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_plan_t* plan = nullptr;
  IREE_CHECK_OK(
      id4_pipeline_plan_create(&options, iree_allocator_system(), &plan));
  id4_pipeline_program_release(program);
  return PlanPtr(plan);
}

static iree_status_t CreateResidencyPlan(
    id4_pipeline_plan_t* plan, iree_device_size_t maximum_target_byte_length,
    id4_pipeline_parameter_window_source_kind_t source_kind,
    id4_pipeline_parameter_residency_plan_t** out_residency_plan) {
  id4_pipeline_parameter_residency_plan_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.plan = plan;
  options.source_kind = source_kind;
  options.maximum_target_byte_length = maximum_target_byte_length;
  options.encoder_staging_chunk_byte_capacity =
      ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
  return id4_pipeline_parameter_residency_plan_create(
      &options, iree_allocator_system(), out_residency_plan);
}

TEST(ParameterResidencyTest, CutsOnlyBetweenAuthoredBarrierEpochs) {
  PlanPtr source_plan = MakeResidencyPlan();
  id4_pipeline_parameter_residency_plan_t* raw_residency_plan = nullptr;
  IREE_ASSERT_OK(
      CreateResidencyPlan(source_plan.get(),
                          /*maximum_target_byte_length=*/160,
                          ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_CHECKPOINT,
                          &raw_residency_plan));
  ResidencyPlanPtr residency_plan(raw_residency_plan);

  const id4_pipeline_parameter_residency_statistics_t statistics =
      id4_pipeline_parameter_residency_plan_statistics(residency_plan.get());
  EXPECT_EQ(statistics.semantic_region_count, 1u);
  EXPECT_EQ(statistics.segment_count, 2u);
  EXPECT_EQ(statistics.maximum_target_byte_length, 160u);
  EXPECT_EQ(statistics.resident_target_byte_length, 240u);
  EXPECT_EQ(statistics.peak_segment_target_byte_length, 160u);
  EXPECT_EQ(statistics.peak_segment_live_byte_length, 160u);
  EXPECT_EQ(statistics.unique_target_byte_length, 240u);
  EXPECT_EQ(statistics.unique_source_transfer_byte_length, 240u);
  EXPECT_EQ(statistics.total_target_byte_length, 304u);
  EXPECT_EQ(statistics.total_source_transfer_byte_length, 304u);
  EXPECT_EQ(statistics.duplicated_target_byte_length, 64u);
  EXPECT_EQ(statistics.duplicated_source_transfer_byte_length, 64u);
  EXPECT_EQ(statistics.total_load_group_count, 4u);

  id4_pipeline_parameter_window_statistics_t issue_statistics;
  IREE_ASSERT_OK(id4_pipeline_parameter_residency_plan_query_live_statistics(
      residency_plan.get(),
      /*parameter_load_prefetch_segment_distance=*/0, &issue_statistics));
  EXPECT_EQ(issue_statistics.concurrent_window_count, 1u);
  EXPECT_EQ(issue_statistics.window_count, 2u);
  EXPECT_EQ(issue_statistics.peak_target_byte_length, 160u);
  EXPECT_EQ(issue_statistics.peak_live_byte_length, 160u);
  EXPECT_EQ(issue_statistics.total_target_byte_length, 304u);

  IREE_ASSERT_OK(id4_pipeline_parameter_residency_plan_query_live_statistics(
      residency_plan.get(),
      /*parameter_load_prefetch_segment_distance=*/1, &issue_statistics));
  EXPECT_EQ(issue_statistics.concurrent_window_count, 2u);
  EXPECT_EQ(issue_statistics.peak_target_byte_length, 304u);
  EXPECT_EQ(issue_statistics.peak_live_byte_length, 304u);

  const id4_pipeline_parameter_residency_segment_t* first =
      id4_pipeline_parameter_residency_plan_segment_at(residency_plan.get(), 0);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->semantic_region_id, 0u);
  EXPECT_EQ(first->source_operation_offset, 3u);
  EXPECT_EQ(first->source_operation_count, 3u);
  EXPECT_EQ(first->dispatch_count, 2u);
  EXPECT_EQ(first->barrier_count, 1u);
  ASSERT_EQ(first->parameter_tensor_count, 2u);
  EXPECT_EQ(first->parameter_tensor_ordinals[0], 0u);
  EXPECT_EQ(first->parameter_tensor_ordinals[1], 1u);
  EXPECT_EQ(first->resource_statistics.target_byte_length, 144u);
  EXPECT_EQ(first->resource_statistics.source_transfer_byte_length, 144u);

  const id4_pipeline_parameter_residency_segment_t* second =
      id4_pipeline_parameter_residency_plan_segment_at(residency_plan.get(), 1);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->semantic_region_id, 0u);
  EXPECT_EQ(second->source_operation_offset, 6u);
  EXPECT_EQ(second->source_operation_count, 3u);
  EXPECT_EQ(second->dispatch_count, 2u);
  EXPECT_EQ(second->barrier_count, 1u);
  ASSERT_EQ(second->parameter_tensor_count, 2u);
  EXPECT_EQ(second->parameter_tensor_ordinals[0], 0u);
  EXPECT_EQ(second->parameter_tensor_ordinals[1], 2u);
  EXPECT_EQ(second->resource_statistics.target_byte_length, 160u);
  EXPECT_EQ(second->resource_statistics.source_transfer_byte_length, 160u);
  EXPECT_EQ(
      id4_pipeline_parameter_residency_plan_segment_at(residency_plan.get(), 2),
      nullptr);

  iree_string_builder_t json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &json_builder);
  IREE_ASSERT_OK(id4_pipeline_parameter_residency_plan_format_json(
      residency_plan.get(), &json_builder));
  const iree_string_view_t json = iree_string_builder_view(&json_builder);
  EXPECT_NE(iree_string_view_find(json, IREE_SV("\"segments\":["), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(json, IREE_SV("\"parameter.a\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                json, IREE_SV("\"duplicated_target_byte_length\":64"), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&json_builder);
}

TEST(ParameterResidencyTest, ExecutionLayoutStatisticsExcludeCheckpointWork) {
  PlanPtr source_plan = MakeResidencyPlan();
  id4_pipeline_parameter_residency_plan_t* raw_residency_plan = nullptr;
  IREE_ASSERT_OK(CreateResidencyPlan(
      source_plan.get(), /*maximum_target_byte_length=*/160,
      ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_EXECUTION_LAYOUT,
      &raw_residency_plan));
  ResidencyPlanPtr residency_plan(raw_residency_plan);

  const id4_pipeline_parameter_residency_statistics_t statistics =
      id4_pipeline_parameter_residency_plan_statistics(residency_plan.get());
  EXPECT_EQ(statistics.segment_count, 2u);
  EXPECT_EQ(statistics.peak_encoder_staging_byte_length, 0u);
  EXPECT_EQ(statistics.peak_segment_live_byte_length,
            statistics.peak_segment_target_byte_length);
  EXPECT_EQ(statistics.unique_source_transfer_byte_length,
            statistics.unique_target_byte_length);
  EXPECT_EQ(statistics.total_source_transfer_byte_length,
            statistics.total_target_byte_length);
  EXPECT_EQ(statistics.total_encode_load_step_count, 0u);
  EXPECT_EQ(statistics.total_load_group_count, statistics.segment_count);
}

TEST(ParameterResidencyTest, RejectsAnIndivisibleEpochOverBudget) {
  PlanPtr source_plan = MakeResidencyPlan();
  id4_pipeline_parameter_residency_plan_t* residency_plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      CreateResidencyPlan(source_plan.get(), /*maximum_target_byte_length=*/143,
                          ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_CHECKPOINT,
                          &residency_plan));
  EXPECT_EQ(residency_plan, nullptr);
}

}  // namespace
