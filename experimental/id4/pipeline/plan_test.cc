// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/plan.h"

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

using DeviceGroupPtr =
    std::unique_ptr<iree_hal_device_group_t,
                    decltype(&iree_hal_device_group_release)>;
using PlanPtr = std::unique_ptr<id4_pipeline_plan_t, PlanDeleter>;

TEST(PlanTest, ReportsAggregateStatistics) {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);

  id4_pipeline_device_placement_t placement = {
      // Human-readable placement role.
      /*.role=*/IREE_SV("default"),
      // Local-sync device index.
      /*.device_index=*/0,
      // Queue affinity selected by the test plan.
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  iree_hal_buffer_params_t storage_params;
  std::memset(&storage_params, 0, sizeof(storage_params));
  storage_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  storage_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  storage_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  storage_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  storage_params.min_alignment = 16;

  const id4_pipeline_parameter_request_t parameter_requests[] = {
      id4_pipeline_parameter_request(
          IREE_SV("layers.0.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0,
                                      /*length=*/128)),
      id4_pipeline_parameter_request(
          IREE_SV("layers.1.weight"),
          id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                      /*buffer_offset=*/0,
                                      /*length=*/512)),
  };
  const id4_pipeline_parameter_slab_plan_t parameter_slabs[] = {
      id4_pipeline_make_parameter_slab_plan(
          IREE_SV("model"), /*placement_id=*/0, /*binding_slot=*/0,
          storage_params, /*byte_length=*/128, /*alignment=*/16,
          /*request_count=*/1, /*requests=*/&parameter_requests[0]),
      id4_pipeline_make_parameter_slab_plan(
          IREE_SV("model"), /*placement_id=*/0, /*binding_slot=*/1,
          storage_params, /*byte_length=*/512, /*alignment=*/16,
          /*request_count=*/1, /*requests=*/&parameter_requests[1]),
  };
  const id4_pipeline_parameter_load_source_t encoded_sources[] = {
      id4_pipeline_parameter_load_source(
          IREE_SV("model.fp8"), IREE_SV("layers.1.weight"),
          ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3, {/*.rank=*/2, /*.dims=*/{16, 16}},
          /*byte_length=*/256),
      id4_pipeline_parameter_load_source(
          IREE_SV("model.fp8"), IREE_SV("layers.1.weight_scale"),
          ID4_PIPELINE_TENSOR_DTYPE_F32, {/*.rank=*/1, /*.dims=*/{16}},
          /*byte_length=*/64),
  };
  const id4_pipeline_parameter_load_step_t parameter_load_steps[] = {
      id4_pipeline_parameter_gather_load_step(
          IREE_SV("gather.layers.0.weight"), IREE_SV("model"),
          /*target_slab_index=*/0, /*request_offset=*/0,
          /*request_count=*/1),
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
          IREE_SV("encode.layers.1.weight"),
          /*source_count=*/IREE_ARRAYSIZE(encoded_sources), encoded_sources,
          /*target_slab_index=*/1, /*request_offset=*/0),
  };
  const id4_pipeline_constant_slab_plan_t constant_slab = {
      // Human-readable slab name.
      /*.name=*/IREE_SV("constants"),
      // Placement containing the constant slab.
      /*.placement_id=*/0,
      // Binding-table slot used by the constant slab.
      /*.binding_slot=*/3,
      // HAL buffer parameters for slab allocation.
      /*.target_params=*/storage_params,
      // Total slab byte length.
      /*.byte_length=*/64,
      // Required slab base alignment.
      /*.alignment=*/16,
      // Empty constant request table.
      /*.request_count=*/0,
      // No constant requests in this synthetic plan.
      /*.requests=*/nullptr,
  };
  const id4_pipeline_memory_slab_plan_t memory_slab = {
      // Human-readable slab name.
      /*.name=*/IREE_SV("locals"),
      // Local slab is visible only to region 0.
      /*.scope=*/ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL,
      // Region containing the local slab.
      /*.region_id=*/0,
      // Placement containing the local slab.
      /*.placement_id=*/0,
      // Binding-table slot used by the local slab.
      /*.binding_slot=*/2,
      // HAL buffer parameters for slab allocation.
      /*.params=*/storage_params,
      // Reserved local slab byte length.
      /*.byte_length=*/1024,
      // Required slab base alignment.
      /*.alignment=*/16,
      // Peak live byte count in the local slab.
      /*.high_water_mark=*/768,
  };
  const id4_pipeline_region_plan_t region = {
      // Human-readable region name.
      /*.name=*/IREE_SV("test.region"),
      // No source program backs this synthetic plan.
      /*.source_operation_offset=*/0,
      // No source program backs this synthetic plan.
      /*.source_operation_count=*/0,
      // Placement containing this region.
      /*.placement_id=*/0,
      // Binding capacity covering storage, boundary, and tap tensors.
      /*.binding_capacity=*/6,
      // Binding-table slot reserved for the local slab.
      /*.local_binding_slot=*/2,
      // Required local tensor alignment.
      /*.local_tensor_alignment=*/16,
      // Dry-run region statistics.
      /*.statistics=*/
      {
          // Number of region operations.
          /*.operation_count=*/7,
          // Number of region dispatches.
          /*.dispatch_count=*/3,
      },
  };
  const id4_pipeline_boundary_tensor_plan_t boundary = {
      // Boundary tensor layout.
      /*.layout=*/
      {
          // Stable boundary tensor name.
          /*.name=*/IREE_SV("output"),
          // Scalar element type.
          /*.dtype=*/ID4_PIPELINE_TENSOR_DTYPE_F32,
          // Dense rank-2 tensor shape.
          /*.shape=*/{/*.rank=*/2, /*.dims=*/{2, 4}},
          // Dense tensor byte length.
          /*.byte_length=*/8 * sizeof(float),
          // Base alignment required by the tensor.
          /*.alignment=*/alignof(float),
      },
      // Caller-provided storage exported by the stage.
      /*.flags=*/ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
          ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      // Region containing this boundary tensor.
      /*.region_id=*/0,
      // Placement containing this boundary tensor.
      /*.placement_id=*/0,
      // Binding-table slot used by this boundary tensor.
      /*.binding_slot=*/4,
  };
  const id4_pipeline_diagnostic_tap_plan_t tap = {
      // Human-readable tap name.
      /*.name=*/IREE_SV("tap"),
      // Region containing this tap.
      /*.region_id=*/0,
      // Placement containing this tap.
      /*.placement_id=*/0,
      // Binding-table slot receiving the tap copy.
      /*.binding_slot=*/5,
      // Operation ordinal after which the tap can be captured.
      /*.after_operation_ordinal=*/2,
      // Tensor or value name exposed by the tap.
      /*.target_name=*/IREE_SV("hidden"),
      // Captured tensor layout.
      /*.layout=*/
      {
          // Stable tap tensor name.
          /*.name=*/IREE_SV("hidden.tap"),
          // Scalar element type.
          /*.dtype=*/ID4_PIPELINE_TENSOR_DTYPE_F32,
          // Dense rank-1 tensor shape.
          /*.shape=*/{/*.rank=*/1, /*.dims=*/{4}},
          // Dense tensor byte length.
          /*.byte_length=*/4 * sizeof(float),
          // Base alignment required by the tensor.
          /*.alignment=*/alignof(float),
      },
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
  options.parameter_slab_count = IREE_ARRAYSIZE(parameter_slabs);
  options.parameter_slabs = parameter_slabs;
  options.parameter_load_step_count = IREE_ARRAYSIZE(parameter_load_steps);
  options.parameter_load_steps = parameter_load_steps;
  options.constant_slab_count = 1;
  options.constant_slabs = &constant_slab;
  options.memory_slab_count = 1;
  options.memory_slabs = &memory_slab;
  options.boundary_tensor_count = 1;
  options.boundary_tensors = &boundary;
  options.region_count = 1;
  options.regions = &region;
  options.diagnostic_tap_count = 1;
  options.diagnostic_taps = &tap;
  options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(
      id4_pipeline_plan_create(&options, iree_allocator_system(), &plan));
  PlanPtr plan_owner(plan);

  id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan_owner.get());
  EXPECT_EQ(statistics.parameter_slab_byte_length, 640u);
  EXPECT_EQ(statistics.largest_parameter_slab_byte_length, 512u);
  EXPECT_EQ(statistics.parameter_source_byte_length, 448u);
  EXPECT_EQ(statistics.parameter_direct_source_byte_length, 128u);
  EXPECT_EQ(statistics.parameter_encoded_source_byte_length, 320u);
  EXPECT_EQ(statistics.parameter_gather_load_step_count, 1u);
  EXPECT_EQ(statistics.parameter_encode_load_step_count, 1u);
  EXPECT_EQ(statistics.parameter_load_group_count, 2u);
  EXPECT_EQ(statistics.parameter_gather_load_group_count, 1u);
  EXPECT_EQ(statistics.parameter_encode_load_group_count, 1u);
  EXPECT_EQ(statistics.constant_slab_byte_length, 64u);
  EXPECT_EQ(statistics.memory_slab_byte_length, 1024u);
  EXPECT_EQ(statistics.memory_slab_high_water_mark, 768u);
  EXPECT_EQ(statistics.boundary_tensor_byte_length, 32u);
  EXPECT_EQ(statistics.diagnostic_tap_byte_length, 16u);
  EXPECT_EQ(statistics.kernel_count, 0u);
  EXPECT_EQ(statistics.region_count, 1u);
  EXPECT_EQ(statistics.operation_count, 7u);
  EXPECT_EQ(statistics.dispatch_count, 3u);
}

TEST(PlanTest, ScopesMemorySlabBindingSlotsPerRegion) {
  DeviceGroupPtr device_group(id4::test::CreateLocalSyncDeviceGroup(),
                              iree_hal_device_group_release);

  id4_pipeline_device_placement_t placement = {
      // Human-readable placement role.
      /*.role=*/IREE_SV("default"),
      // Local-sync device index.
      /*.device_index=*/0,
      // Queue affinity selected by the test plan.
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  iree_hal_buffer_params_t storage_params;
  std::memset(&storage_params, 0, sizeof(storage_params));
  storage_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  storage_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  storage_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  storage_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  storage_params.min_alignment = 16;

  const id4_pipeline_region_plan_t regions[] = {
      {
          // Human-readable region name.
          /*.name=*/IREE_SV("test.region0"),
          // No source program backs this synthetic plan.
          /*.source_operation_offset=*/0,
          // No source program backs this synthetic plan.
          /*.source_operation_count=*/0,
          // Placement containing this region.
          /*.placement_id=*/0,
          // Binding capacity covering local and shared slabs.
          /*.binding_capacity=*/4,
          // Binding-table slot reserved for this region's local slab.
          /*.local_binding_slot=*/2,
          // Required local tensor alignment.
          /*.local_tensor_alignment=*/16,
      },
      {
          // Human-readable region name.
          /*.name=*/IREE_SV("test.region1"),
          // No source program backs this synthetic plan.
          /*.source_operation_offset=*/0,
          // No source program backs this synthetic plan.
          /*.source_operation_count=*/0,
          // Placement containing this region.
          /*.placement_id=*/0,
          // Binding capacity covering local and shared slabs.
          /*.binding_capacity=*/4,
          // Binding-table slot reserved for this region's local slab.
          /*.local_binding_slot=*/2,
          // Required local tensor alignment.
          /*.local_tensor_alignment=*/16,
      },
  };
  const id4_pipeline_memory_slab_plan_t memory_slabs[] = {
      {
          // Human-readable slab name.
          /*.name=*/IREE_SV("region0.local"),
          // Local slab is visible only to region 0.
          /*.scope=*/ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL,
          // Region containing this local slab.
          /*.region_id=*/0,
          // Placement containing this local slab.
          /*.placement_id=*/0,
          // Local slab binding slot for region 0.
          /*.binding_slot=*/2,
          // HAL buffer parameters for slab allocation.
          /*.params=*/storage_params,
          // Reserved local slab byte length.
          /*.byte_length=*/64,
          // Required slab base alignment.
          /*.alignment=*/16,
          // Peak live byte count in the local slab.
          /*.high_water_mark=*/64,
      },
      {
          // Human-readable slab name.
          /*.name=*/IREE_SV("region1.local"),
          // Local slab is visible only to region 1.
          /*.scope=*/ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL,
          // Region containing this local slab.
          /*.region_id=*/1,
          // Placement containing this local slab.
          /*.placement_id=*/0,
          // Local slab binding slot for region 1.
          /*.binding_slot=*/2,
          // HAL buffer parameters for slab allocation.
          /*.params=*/storage_params,
          // Reserved local slab byte length.
          /*.byte_length=*/128,
          // Required slab base alignment.
          /*.alignment=*/16,
          // Peak live byte count in the local slab.
          /*.high_water_mark=*/96,
      },
      {
          // Human-readable slab name.
          /*.name=*/IREE_SV("stage.shared"),
          // Shared slab is visible to both regions.
          /*.scope=*/ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED,
          // Plan-shared slabs do not have one owning region.
          /*.region_id=*/0,
          // Placement containing this shared slab.
          /*.placement_id=*/0,
          // Shared slab binding slot in every region.
          /*.binding_slot=*/3,
          // HAL buffer parameters for slab allocation.
          /*.params=*/storage_params,
          // Reserved shared slab byte length.
          /*.byte_length=*/256,
          // Required slab base alignment.
          /*.alignment=*/16,
          // Peak live byte count in the shared slab.
          /*.high_water_mark=*/192,
      },
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
  options.memory_slab_count = IREE_ARRAYSIZE(memory_slabs);
  options.memory_slabs = memory_slabs;
  options.region_count = IREE_ARRAYSIZE(regions);
  options.regions = regions;
  options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* raw_plan = nullptr;
  IREE_ASSERT_OK(
      id4_pipeline_plan_create(&options, iree_allocator_system(), &raw_plan));
  PlanPtr plan(raw_plan);

  ASSERT_EQ(id4_pipeline_plan_memory_slab_count(plan.get()), 3u);
  const id4_pipeline_memory_slab_plan_t* local0 =
      id4_pipeline_plan_memory_slab_at(plan.get(), 0);
  ASSERT_NE(local0, nullptr);
  EXPECT_EQ(local0->scope, ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL);
  EXPECT_EQ(local0->region_id, 0u);
  const id4_pipeline_memory_slab_plan_t* local1 =
      id4_pipeline_plan_memory_slab_at(plan.get(), 1);
  ASSERT_NE(local1, nullptr);
  EXPECT_EQ(local1->scope, ID4_PIPELINE_MEMORY_SLAB_SCOPE_REGION_LOCAL);
  EXPECT_EQ(local1->region_id, 1u);
  const id4_pipeline_memory_slab_plan_t* shared =
      id4_pipeline_plan_memory_slab_at(plan.get(), 2);
  ASSERT_NE(shared, nullptr);
  EXPECT_EQ(shared->scope, ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED);
  EXPECT_EQ(shared->region_id, 0u);

  iree_string_builder_t json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &json_builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan.get(), &json_builder));
  iree_string_view_t json = iree_string_builder_view(&json_builder);
  EXPECT_NE(
      iree_string_view_find(json, IREE_SV("\"scope\":\"region_local\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(json, IREE_SV("\"scope\":\"plan_shared\""), 0),
      IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&json_builder);
}

}  // namespace
