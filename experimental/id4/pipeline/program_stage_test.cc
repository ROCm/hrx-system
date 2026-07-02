// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_stage.h"

#include <cstring>
#include <string>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ProgramBuilderScope {
 public:
  ProgramBuilderScope() {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_program_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.program_name=*/IREE_SV("test.sampler"),
        /*.block_pool=*/&block_pool_,
    };
    IREE_CHECK_OK(id4_pipeline_program_builder_create(
        &options, iree_allocator_system(), &builder_));
  }

  ~ProgramBuilderScope() {
    id4_pipeline_program_builder_destroy(builder_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  id4_pipeline_program_builder_t* builder() { return builder_; }

  void ResetBuilder() {
    id4_pipeline_program_builder_destroy(builder_);
    builder_ = nullptr;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

static iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("id4-program-stage-local-sync"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      IREE_SV("id4-program-stage-local-sync"), &sync_params, &create_params,
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

static id4_pipeline_program_t* CreateSamplerProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();
  const id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);

  id4_pipeline_program_tensor_t x_t = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t x_t_options = {
      /*.structure_size=*/sizeof(x_t_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("sampler.x_t"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/latent_shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &x_t_options, &x_t));

  id4_pipeline_program_tensor_t scalings =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t scalings_options = {
      /*.structure_size=*/sizeof(scalings_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("sampler.scalings"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(3),
  };
  IREE_CHECK_OK(id4_pipeline_program_import_tensor(builder, &scalings_options,
                                                   &scalings));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("sampler.denoised"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/latent_shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(
          IREE_SV("id4.sampler.element_count"), IREE_SV("8")),
  };
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(x_t),
      id4_pipeline_program_read(scalings),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      /*.structure_size=*/sizeof(dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("sampler.test"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("sampler/test"),
                                   IREE_SV("sampler_test")),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.binding_count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("sampler.denoised"),
      /*.tensor=*/output,
  };
  IREE_CHECK_OK(id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.ResetBuilder();
  return program;
}

static id4_pipeline_program_t* CreateCrossDispatchTransientProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();
  const id4_pipeline_program_shape_t vector_shape =
      id4_pipeline_program_make_shape_rank1(4);

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("cross_dispatch.input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/vector_shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("cross_dispatch.output"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/vector_shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_program_tensor_t hidden = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t hidden_options = {
      /*.structure_size=*/sizeof(hidden_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("cross_dispatch.hidden"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/vector_shape,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &hidden_options, &hidden));

  id4_pipeline_program_dispatch_binding_t producer_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(hidden),
  };
  id4_pipeline_program_dispatch_loom_options_t producer_options = {
      /*.structure_size=*/sizeof(producer_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("cross_dispatch.producer"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/cross_dispatch"),
                                   IREE_SV("producer")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(producer_bindings),
      /*.bindings=*/producer_bindings,
  };
  IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder, &producer_options));

  id4_pipeline_program_tap_options_t hidden_tap_options = {
      /*.structure_size=*/sizeof(hidden_tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("cross_dispatch.hidden"),
      /*.tensor=*/hidden,
  };
  IREE_CHECK_OK(id4_pipeline_program_tap(builder, &hidden_tap_options));

  id4_pipeline_program_dispatch_binding_t consumer_bindings[] = {
      id4_pipeline_program_read(hidden),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t consumer_options = {
      /*.structure_size=*/sizeof(consumer_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("cross_dispatch.consumer"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/cross_dispatch"),
                                   IREE_SV("consumer")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(consumer_bindings),
      /*.bindings=*/consumer_bindings,
  };
  IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder, &consumer_options));

  id4_pipeline_program_tap_options_t output_tap_options = {
      /*.structure_size=*/sizeof(output_tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("cross_dispatch.output"),
      /*.tensor=*/output,
  };
  IREE_CHECK_OK(id4_pipeline_program_tap(builder, &output_tap_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("cross_dispatch.output"),
      /*.tensor=*/output,
  };
  IREE_CHECK_OK(id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.ResetBuilder();
  return program;
}

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static id4_pipeline_stage_plan_options_t MakeStagePlanOptions(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_stage_plan_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.device_index = 0;
  options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  options.diagnostics_sink = diagnostics_sink;
  return options;
}

static id4_pipeline_program_stage_plan_options_t MakeProgramStagePlanOptions(
    id4_pipeline_stage_plan_options_t* stage_options,
    id4_pipeline_program_t* program, iree_hal_device_group_t* device_group) {
  id4_pipeline_program_stage_plan_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.stage_name = IREE_SV("sampler.loop");
  options.stage_options = stage_options;
  options.program = program;
  options.device_group = device_group;
  options.parameter_scope = iree_string_view_empty();
  options.alignment = 16;
  return options;
}

TEST(ProgramStage, PlansParameterFreeProgramStage) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_program_t* program = CreateSamplerProgram();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_plan_options_t stage_options =
      MakeStagePlanOptions(&diagnostics_sink);
  id4_pipeline_program_stage_plan_options_t options =
      MakeProgramStagePlanOptions(&stage_options, program, device_group);

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_stage_create_plan(
      &options, iree_allocator_system(), &plan));

  EXPECT_EQ(ToString(id4_pipeline_plan_stage_name(plan)), "sampler.loop");
  EXPECT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_boundary_tensor_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);
  EXPECT_EQ(id4_pipeline_plan_region_count(plan), 1u);

  id4_pipeline_plan_release(plan);
  id4_pipeline_program_release(program);
  iree_hal_device_group_release(device_group);
}

TEST(ProgramStage, PacksParameterFreeBindingsDensely) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_program_t* program = CreateSamplerProgram();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_plan_options_t stage_options =
      MakeStagePlanOptions(&diagnostics_sink);
  id4_pipeline_program_stage_plan_options_t options =
      MakeProgramStagePlanOptions(&stage_options, program, device_group);

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_stage_create_plan(
      &options, iree_allocator_system(), &plan));

  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(region->binding_capacity,
            id4_pipeline_plan_boundary_tensor_count(plan) + 1);
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    ASSERT_NE(boundary, nullptr);
    EXPECT_EQ(boundary->binding_slot, i);
  }

  id4_pipeline_plan_release(plan);
  id4_pipeline_program_release(program);
  iree_hal_device_group_release(device_group);
}

TEST(ProgramStage, PromotesImplicitCrossDispatchTransientsToSharedSlab) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_program_t* program = CreateCrossDispatchTransientProgram();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_plan_options_t stage_options =
      MakeStagePlanOptions(&diagnostics_sink);
  stage_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH;
  id4_pipeline_program_stage_plan_options_t options =
      MakeProgramStagePlanOptions(&stage_options, program, device_group);

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_stage_create_plan(
      &options, iree_allocator_system(), &plan));

  EXPECT_EQ(id4_pipeline_plan_region_count(plan), 2u);
  EXPECT_EQ(id4_pipeline_plan_shared_tensor_count(plan), 1u);
  const id4_pipeline_shared_tensor_plan_t* shared_tensor =
      id4_pipeline_plan_shared_tensor_at(plan, 0);
  ASSERT_NE(shared_tensor, nullptr);
  EXPECT_EQ(shared_tensor->acquire_region_id, 0u);
  EXPECT_EQ(shared_tensor->last_use_region_id, 1u);

  const id4_pipeline_memory_slab_plan_t* shared_slab =
      id4_pipeline_plan_memory_slab_at(plan, shared_tensor->memory_slab_index);
  ASSERT_NE(shared_slab, nullptr);
  EXPECT_EQ(shared_slab->scope, ID4_PIPELINE_MEMORY_SLAB_SCOPE_PLAN_SHARED);

  const id4_pipeline_region_plan_t* first_region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(first_region, nullptr);
  const id4_pipeline_region_plan_t* second_region =
      id4_pipeline_plan_region_at(plan, 1);
  ASSERT_NE(second_region, nullptr);
  EXPECT_EQ(first_region->binding_capacity, second_region->binding_capacity);
  EXPECT_GT(first_region->binding_capacity,
            id4_pipeline_plan_boundary_tensor_count(plan) + 1);

  id4_pipeline_plan_release(plan);
  id4_pipeline_program_release(program);
  iree_hal_device_group_release(device_group);
}

TEST(ProgramStage, DiagnosticTapsDoNotRenumberProductionBindings) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_program_t* program = CreateCrossDispatchTransientProgram();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_plan_options_t base_stage_options =
      MakeStagePlanOptions(&diagnostics_sink);
  base_stage_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH;
  id4_pipeline_program_stage_plan_options_t base_options =
      MakeProgramStagePlanOptions(&base_stage_options, program, device_group);

  id4_pipeline_plan_t* base_plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_stage_create_plan(
      &base_options, iree_allocator_system(), &base_plan));

  const id4_pipeline_shared_tensor_plan_t* base_shared_tensor =
      id4_pipeline_plan_shared_tensor_at(base_plan, 0);
  ASSERT_NE(base_shared_tensor, nullptr);
  const id4_pipeline_memory_slab_plan_t* base_shared_slab =
      id4_pipeline_plan_memory_slab_at(base_plan,
                                       base_shared_tensor->memory_slab_index);
  ASSERT_NE(base_shared_slab, nullptr);
  const id4_pipeline_region_plan_t* base_region =
      id4_pipeline_plan_region_at(base_plan, 0);
  ASSERT_NE(base_region, nullptr);

  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("cross_dispatch.hidden"),
      IREE_SV("cross_dispatch.output"),
  };
  id4_pipeline_stage_plan_options_t tap_stage_options =
      MakeStagePlanOptions(&diagnostics_sink);
  tap_stage_options.flags =
      ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH |
      ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  tap_stage_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  id4_pipeline_program_stage_plan_options_t tap_options =
      MakeProgramStagePlanOptions(&tap_stage_options, program, device_group);

  id4_pipeline_plan_t* tap_plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_stage_create_plan(
      &tap_options, iree_allocator_system(), &tap_plan));

  const id4_pipeline_shared_tensor_plan_t* tap_shared_tensor =
      id4_pipeline_plan_shared_tensor_at(tap_plan, 0);
  ASSERT_NE(tap_shared_tensor, nullptr);
  const id4_pipeline_memory_slab_plan_t* tap_shared_slab =
      id4_pipeline_plan_memory_slab_at(tap_plan,
                                       tap_shared_tensor->memory_slab_index);
  ASSERT_NE(tap_shared_slab, nullptr);
  const id4_pipeline_region_plan_t* tap_region =
      id4_pipeline_plan_region_at(tap_plan, 0);
  ASSERT_NE(tap_region, nullptr);
  ASSERT_EQ(id4_pipeline_plan_diagnostic_tap_count(tap_plan), 2u);

  EXPECT_EQ(tap_shared_slab->binding_slot, base_shared_slab->binding_slot);
  EXPECT_EQ(tap_region->local_binding_slot, base_region->local_binding_slot);
  EXPECT_EQ(tap_region->binding_capacity,
            base_region->binding_capacity +
                id4_pipeline_plan_diagnostic_tap_count(tap_plan));
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_diagnostic_tap_count(tap_plan); ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(tap_plan, i);
    ASSERT_NE(tap, nullptr);
    EXPECT_GE(tap->binding_slot, base_region->binding_capacity);
  }

  id4_pipeline_plan_release(tap_plan);
  id4_pipeline_plan_release(base_plan);
  id4_pipeline_program_release(program);
  iree_hal_device_group_release(device_group);
}

}  // namespace
