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

static iree_hal_dispatch_config_t MakeDispatchConfig(uint32_t element_count) {
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config((element_count + 255) / 256, 1, 1);
  config.workgroup_size[0] = 256;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  return config;
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
      /*.dispatch_config=*/MakeDispatchConfig(/*element_count=*/8),
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
  options.parameter_slab_binding_slot = 0;
  options.boundary_binding_slot_base = 0;
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

TEST(ProgramStage, RejectsSparseParameterFreeBindingLayout) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  id4_pipeline_program_t* program = CreateSamplerProgram();

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_plan_options_t stage_options =
      MakeStagePlanOptions(&diagnostics_sink);
  id4_pipeline_program_stage_plan_options_t options =
      MakeProgramStagePlanOptions(&stage_options, program, device_group);
  options.boundary_binding_slot_base = 1;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_stage_create_plan(
                            &options, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);

  id4_pipeline_program_release(program);
  iree_hal_device_group_release(device_group);
}

}  // namespace
