// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_plan.h"

#include <string>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void ExpectStringViewEqual(iree_string_view_t actual,
                                  iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(actual, expected))
      << "actual: " << std::string(actual.data, actual.size)
      << ", expected: " << std::string(expected.data, expected.size);
}

class ProgramBuilderScope {
 public:
  ProgramBuilderScope() {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_program_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.program_name=*/IREE_SV("test.forward"),
        /*.block_pool=*/&block_pool_,
    };
    IREE_CHECK_OK(id4_pipeline_program_builder_create(
        &options, iree_allocator_system(), &builder_));
  }

  ~ProgramBuilderScope() {
    DestroyBuilder();
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  id4_pipeline_program_builder_t* builder() { return builder_; }

  void DestroyBuilder() {
    id4_pipeline_program_builder_destroy(builder_);
    builder_ = nullptr;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

static iree_hal_dispatch_config_t MakeTestDispatchConfig() {
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 1;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  return config;
}

static iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("id4-program-plan-local-sync"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      IREE_SV("id4-program-plan-local-sync"), &sync_params, &create_params,
      /*loader_count=*/0, /*loaders=*/nullptr, device_allocator,
      iree_allocator_system(), &device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      tracker_options, iree_allocator_system(), &frontier_tracker));

  iree_hal_device_group_t* device_group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group));

  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  return device_group;
}

static id4_pipeline_program_t* CreateLinearProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("hidden_states.input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_parameter_options_t weight_options = {
      /*.structure_size=*/sizeof(weight_options),
      /*.next=*/nullptr,
      /*.key=*/IREE_SV("model.layers.0.linear.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_parameter(builder, &weight_options, &weight));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("hidden_states.linear"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(IREE_SV("@batch"), IREE_SV("1")),
      id4_pipeline_make_kernel_config_binding(IREE_SV("@hidden_size"),
                                              IREE_SV("4")),
  };
  id4_pipeline_program_dispatch_binding_t first_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t first_dispatch_options = {
      /*.structure_size=*/sizeof(first_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.linear.first"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/linear"), IREE_SV("linear")),
      /*.dispatch_config=*/MakeTestDispatchConfig(),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.binding_count=*/IREE_ARRAYSIZE(first_bindings),
      /*.bindings=*/first_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &first_dispatch_options));

  id4_pipeline_program_barrier_options_t barrier_options = {
      /*.structure_size=*/sizeof(barrier_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.linear.after_first"),
  };
  IREE_CHECK_OK(id4_pipeline_program_barrier(builder, &barrier_options));

  id4_pipeline_program_dispatch_binding_t second_bindings[] = {
      id4_pipeline_program_read_write(output),
      id4_pipeline_program_read(weight),
  };
  id4_pipeline_program_dispatch_loom_options_t second_dispatch_options = {
      /*.structure_size=*/sizeof(second_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.linear.second"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/linear"), IREE_SV("linear")),
      /*.dispatch_config=*/MakeTestDispatchConfig(),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.binding_count=*/IREE_ARRAYSIZE(second_bindings),
      /*.bindings=*/second_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &second_dispatch_options));

  id4_pipeline_program_tap_options_t tap_options = {
      /*.structure_size=*/sizeof(tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.linear.output"),
      /*.tensor=*/output,
  };
  IREE_CHECK_OK(id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.output"),
      /*.tensor=*/output,
  };
  IREE_CHECK_OK(id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static id4_pipeline_program_plan_options_t MakePlanOptions(
    const id4_pipeline_program_t* program,
    iree_hal_device_group_t* device_group,
    const id4_pipeline_device_placement_t* placement,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_buffer_params_t parameter_params =
      id4_pipeline_parameter_slab_device_local_params(
          IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          /*min_alignment=*/16);
  iree_hal_buffer_params_t local_params = {};
  local_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  local_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  local_params.min_alignment = 16;
  id4_pipeline_program_plan_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS,
      /*.program=*/program,
      /*.device_group=*/device_group,
      /*.placement_count=*/1,
      /*.placements=*/placement,
      /*.parameter_scope=*/IREE_SV(""),
      /*.parameter_slab_placement_id=*/0,
      /*.parameter_slab_binding_slot=*/0,
      /*.parameter_slab_target_params=*/parameter_params,
      /*.parameter_slab_alignment=*/16,
      /*.parameter_request_alignment=*/16,
      /*.kernel_placement_id=*/0,
      /*.region_placement_id=*/0,
      /*.region_local_slab_params=*/local_params,
      /*.region_local_slab_alignment=*/16,
      /*.region_local_tensor_alignment=*/16,
      /*.region_binding_capacity=*/5,
      /*.region_local_binding_slot=*/4,
      /*.region_boundary_binding_slot_base=*/1,
      /*.diagnostic_tap_binding_slot_base=*/3,
      /*.diagnostics_sink=*/diagnostics_sink,
  };
  return options;
}

TEST(PipelineProgramPlan, DerivesParameterKernelRegionAndTapPlans) {
  id4_pipeline_program_t* program = CreateLinearProgram();
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_device_placement_t placement = {
      /*.role=*/IREE_SV("default"),
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_program_plan_options_t options =
      MakePlanOptions(program, device_group, &placement, &diagnostics_sink);

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_create_plan(
      &options, iree_allocator_system(), &plan));

  ExpectStringViewEqual(id4_pipeline_plan_stage_name(plan),
                        IREE_SV("test.forward"));
  ASSERT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 1u);
  const id4_pipeline_parameter_slab_plan_t* parameter_slab =
      id4_pipeline_plan_parameter_slab_at(plan, 0);
  ASSERT_NE(parameter_slab, nullptr);
  EXPECT_EQ(parameter_slab->byte_length, 32u);
  EXPECT_EQ(parameter_slab->binding_slot, 0u);
  ASSERT_EQ(parameter_slab->request_count, 1u);
  ExpectStringViewEqual(parameter_slab->requests[0].key,
                        IREE_SV("model.layers.0.linear.weight"));
  EXPECT_EQ(parameter_slab->requests[0].span.buffer_offset, 0u);
  EXPECT_EQ(parameter_slab->requests[0].span.length, 32u);
  EXPECT_EQ(id4_pipeline_plan_memory_slab_count(plan), 0u);

  ASSERT_EQ(id4_pipeline_plan_boundary_tensor_count(plan), 2u);
  const id4_pipeline_boundary_tensor_plan_t* input_boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, 0);
  ASSERT_NE(input_boundary, nullptr);
  ExpectStringViewEqual(input_boundary->layout.name,
                        IREE_SV("hidden_states.input"));
  EXPECT_EQ(input_boundary->binding_slot, 1u);
  EXPECT_TRUE(
      iree_all_bits_set(input_boundary->flags,
                        ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
                            ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED));
  const id4_pipeline_boundary_tensor_plan_t* output_boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, 1);
  ASSERT_NE(output_boundary, nullptr);
  ExpectStringViewEqual(output_boundary->layout.name,
                        IREE_SV("hidden_states.linear"));
  EXPECT_EQ(output_boundary->binding_slot, 2u);
  EXPECT_TRUE(iree_all_bits_set(
      output_boundary->flags, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
                                  ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED));

  ASSERT_EQ(id4_pipeline_plan_kernel_count(plan), 1u);
  const id4_pipeline_kernel_plan_t* kernel =
      id4_pipeline_plan_kernel_at(plan, 0);
  ASSERT_NE(kernel, nullptr);
  ExpectStringViewEqual(
      kernel->specialization_key,
      IREE_SV("test/linear::linear[@batch=1,@hidden_size=4]"));
  ExpectStringViewEqual(kernel->module_path, IREE_SV("test/linear"));
  ExpectStringViewEqual(kernel->function_name, IREE_SV("linear"));
  ASSERT_EQ(kernel->config_binding_count, 2u);

  ASSERT_EQ(id4_pipeline_plan_region_count(plan), 1u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  ExpectStringViewEqual(region->name, IREE_SV("test.forward"));
  EXPECT_EQ(region->statistics.operation_count, 6u);
  EXPECT_EQ(region->statistics.dispatch_count, 2u);
  EXPECT_EQ(region->statistics.copy_count, 1u);
  EXPECT_EQ(region->statistics.barrier_count, 3u);
  EXPECT_EQ(region->statistics.current_epoch, 3u);

  ASSERT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 1u);
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 0);
  ASSERT_NE(tap, nullptr);
  ExpectStringViewEqual(tap->name, IREE_SV("block0.linear.output"));
  EXPECT_EQ(tap->region_id, 0u);
  EXPECT_EQ(tap->placement_id, 0u);
  EXPECT_EQ(tap->binding_slot, 3u);
  EXPECT_EQ(tap->after_operation_ordinal, 2u);
  ExpectStringViewEqual(tap->target_name, IREE_SV("hidden_states.linear"));
  ExpectStringViewEqual(tap->layout.name, IREE_SV("block0.linear.output"));
  EXPECT_EQ(tap->layout.dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  EXPECT_EQ(tap->layout.byte_length, 16u);

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(PipelineProgramPlan, RejectsTapBeforeExecutableRegionOperation) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();
  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("hidden_states.input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));
  id4_pipeline_program_tap_options_t tap_options = {
      /*.structure_size=*/sizeof(tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("input.tap"),
      /*.tensor=*/input,
  };
  IREE_ASSERT_OK(id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();

  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_device_placement_t placement = {
      /*.role=*/IREE_SV("default"),
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_program_plan_options_t options =
      MakePlanOptions(program, device_group, &placement, &diagnostics_sink);
  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_pipeline_program_create_plan(
                            &options, iree_allocator_system(), &plan));

  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

}  // namespace
