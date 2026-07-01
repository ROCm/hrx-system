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

static void ExpectFinds(iree_string_view_t value, iree_string_view_t needle) {
  EXPECT_NE(iree_string_view_find(value, needle, 0), IREE_STRING_VIEW_NPOS)
      << "expected to find: " << std::string(needle.data, needle.size);
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
  const id4_pipeline_program_parameter_source_t weight_sources[] = {
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.key=*/IREE_SV("model.layers.0.linear.weight"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
      },
  };
  id4_pipeline_program_parameter_options_t weight_options = {
      /*.structure_size=*/sizeof(weight_options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/IREE_ARRAYSIZE(weight_sources),
      /*.sources=*/weight_sources,
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

static id4_pipeline_program_t* CreateRegionCutProgram() {
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

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("hidden_states.output"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      /*.structure_size=*/sizeof(dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("entry.copy"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/copy"), IREE_SV("copy")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_region_cut_options_t cut_options = {
      /*.structure_size=*/sizeof(cut_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("entry.after_copy"),
  };
  IREE_CHECK_OK(id4_pipeline_program_region_cut(builder, &cut_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static id4_pipeline_program_t* CreateLocalReuseProgram() {
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

  id4_pipeline_program_tensor_t first = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t first_options = {
      /*.structure_size=*/sizeof(first_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.first"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &first_options, &first));

  id4_pipeline_program_dispatch_binding_t first_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(first),
  };
  id4_pipeline_program_dispatch_loom_options_t first_dispatch_options = {
      /*.structure_size=*/sizeof(first_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("reuse.first"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/reuse"), IREE_SV("reuse")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(first_bindings),
      /*.bindings=*/first_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &first_dispatch_options));

  id4_pipeline_program_barrier_options_t first_barrier_options = {
      /*.structure_size=*/sizeof(first_barrier_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("reuse.after_first"),
  };
  IREE_CHECK_OK(id4_pipeline_program_barrier(builder, &first_barrier_options));

  id4_pipeline_program_tensor_t second = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t second_options = {
      /*.structure_size=*/sizeof(second_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.second"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &second_options, &second));

  id4_pipeline_program_dispatch_binding_t second_bindings[] = {
      id4_pipeline_program_read(first),
      id4_pipeline_program_write(second),
  };
  id4_pipeline_program_dispatch_loom_options_t second_dispatch_options = {
      /*.structure_size=*/sizeof(second_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("reuse.second"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/reuse"), IREE_SV("reuse")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(second_bindings),
      /*.bindings=*/second_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &second_dispatch_options));

  id4_pipeline_program_barrier_options_t second_barrier_options = {
      /*.structure_size=*/sizeof(second_barrier_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("reuse.after_second"),
  };
  IREE_CHECK_OK(id4_pipeline_program_barrier(builder, &second_barrier_options));

  id4_pipeline_program_tensor_t third = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t third_options = {
      /*.structure_size=*/sizeof(third_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.third"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &third_options, &third));

  id4_pipeline_program_dispatch_binding_t third_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(third),
  };
  id4_pipeline_program_dispatch_loom_options_t third_dispatch_options = {
      /*.structure_size=*/sizeof(third_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("reuse.third"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/reuse"), IREE_SV("reuse")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(third_bindings),
      /*.bindings=*/third_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &third_dispatch_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static id4_pipeline_program_t* CreateEncodedParameterProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  const id4_pipeline_program_parameter_source_t sources[] = {
      {
          /*.source_scope=*/IREE_SV("fp8"),
          /*.key=*/IREE_SV("model.layers.0.linear.weight"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
      },
      {
          /*.source_scope=*/IREE_SV("fp8"),
          /*.key=*/IREE_SV("model.layers.0.linear.weight_scale"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
          /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
      },
  };
  id4_pipeline_program_parameter_options_t weight_options = {
      /*.structure_size=*/sizeof(weight_options),
      /*.next=*/nullptr,
      /*.encoding=*/
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16,
      /*.source_count=*/IREE_ARRAYSIZE(sources),
      /*.sources=*/sources,
      /*.key=*/IREE_SV("model.layers.0.linear.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(
      id4_pipeline_program_parameter(builder, &weight_options, &weight));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static id4_pipeline_program_t* CreateInterleavedParameterSourceProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  const id4_pipeline_program_parameter_source_t direct_first_sources[] = {
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.key=*/IREE_SV("model.layers.0.attn.q.weight"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
      },
  };
  id4_pipeline_program_parameter_options_t direct_first_options = {
      /*.structure_size=*/sizeof(direct_first_options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/IREE_ARRAYSIZE(direct_first_sources),
      /*.sources=*/direct_first_sources,
      /*.key=*/IREE_SV("model.layers.0.attn.q.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  id4_pipeline_program_tensor_t direct_first =
      id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(id4_pipeline_program_parameter(builder, &direct_first_options,
                                               &direct_first));

  const id4_pipeline_program_parameter_source_t encoded_sources[] = {
      {
          /*.source_scope=*/IREE_SV("fp8"),
          /*.key=*/IREE_SV("model.layers.0.mlp.w1.weight"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
      },
      {
          /*.source_scope=*/IREE_SV("fp8"),
          /*.key=*/IREE_SV("model.layers.0.mlp.w1.weight_scale"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
          /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
      },
  };
  id4_pipeline_program_parameter_options_t encoded_options = {
      /*.structure_size=*/sizeof(encoded_options),
      /*.next=*/nullptr,
      /*.encoding=*/
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16,
      /*.source_count=*/IREE_ARRAYSIZE(encoded_sources),
      /*.sources=*/encoded_sources,
      /*.key=*/IREE_SV("model.layers.0.mlp.w1.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  id4_pipeline_program_tensor_t encoded = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(
      id4_pipeline_program_parameter(builder, &encoded_options, &encoded));

  const id4_pipeline_program_parameter_source_t direct_second_sources[] = {
      {
          /*.source_scope=*/IREE_SV("model"),
          /*.key=*/IREE_SV("model.layers.0.attn.o.weight"),
          /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
          /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
      },
  };
  id4_pipeline_program_parameter_options_t direct_second_options = {
      /*.structure_size=*/sizeof(direct_second_options),
      /*.next=*/nullptr,
      /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      /*.source_count=*/IREE_ARRAYSIZE(direct_second_sources),
      /*.sources=*/direct_second_sources,
      /*.key=*/IREE_SV("model.layers.0.attn.o.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  id4_pipeline_program_tensor_t direct_second =
      id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(id4_pipeline_program_parameter(builder, &direct_second_options,
                                               &direct_second));

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
      /*.flags=*/0,
      /*.stage_name=*/IREE_SV("test.stage"),
      /*.program=*/program,
      /*.device_group=*/device_group,
      /*.placement_count=*/1,
      /*.placements=*/placement,
      /*.parameter_scope=*/IREE_SV("model"),
      /*.parameter_slab_placement_id=*/0,
      /*.parameter_slab_binding_slot=*/0,
      /*.parameter_slab_target_params=*/parameter_params,
      /*.parameter_slab_alignment=*/16,
      /*.parameter_request_alignment=*/16,
      /*.constant_slab_placement_id=*/0,
      /*.constant_slab_binding_slot=*/3,
      /*.constant_slab_target_params=*/parameter_params,
      /*.constant_slab_alignment=*/16,
      /*.constant_request_alignment=*/16,
      /*.kernel_placement_id=*/0,
      /*.region_placement_id=*/0,
      /*.region_local_slab_params=*/local_params,
      /*.region_local_slab_alignment=*/16,
      /*.region_local_tensor_alignment=*/16,
      /*.region_binding_capacity=*/5,
      /*.region_local_binding_slot=*/4,
      /*.region_boundary_binding_slot_base=*/1,
      /*.diagnostic_tap_names=*/iree_string_view_list_empty(),
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
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("block0.linear.output"),
  };
  options.flags = ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_create_plan(
      &options, iree_allocator_system(), &plan));

  ExpectStringViewEqual(id4_pipeline_plan_stage_name(plan),
                        IREE_SV("test.stage"));
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
  ASSERT_EQ(id4_pipeline_plan_parameter_load_step_count(plan), 1u);
  const id4_pipeline_parameter_load_step_t* load_step =
      id4_pipeline_plan_parameter_load_step_at(plan, 0);
  ASSERT_NE(load_step, nullptr);
  ExpectStringViewEqual(load_step->name, IREE_SV("parameters.gather"));
  EXPECT_EQ(load_step->kind, ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER);
  ExpectStringViewEqual(load_step->source_scope, IREE_SV("model"));
  EXPECT_EQ(load_step->target_slab_index, 0u);
  EXPECT_EQ(load_step->request_offset, 0u);
  EXPECT_EQ(load_step->request_count, 1u);
  EXPECT_EQ(load_step->request_indices, nullptr);
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

  iree_string_builder_t json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &json_builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &json_builder));
  iree_string_view_t json = iree_string_builder_view(&json_builder);
  ExpectFinds(json, IREE_SV("\"program\":{\"name\":\"test.forward\""));
  ExpectFinds(json, IREE_SV("\"dispatch_count\":2"));
  ExpectFinds(json, IREE_SV("\"dispatch_ordinal\":0"));
  ExpectFinds(json, IREE_SV("\"region_operation_ordinal\":0"));
  ExpectFinds(json, IREE_SV("\"dispatch_ordinal\":1"));
  ExpectFinds(json, IREE_SV("\"region_operation_ordinal\":2"));
  ExpectFinds(json, IREE_SV("\"module_path\":\"test/linear\""));
  ExpectFinds(json, IREE_SV("\"function_name\":\"linear\""));
  ExpectFinds(json, IREE_SV("\"config_bindings\":[{\"key\":\"@batch\""));
  ExpectFinds(json, IREE_SV("\"bindings\":[{\"index\":0"));
  ExpectFinds(json, IREE_SV("\"name\":\"hidden_states.input\""));
  ExpectFinds(json, IREE_SV("\"access\":\"read\""));
  iree_string_builder_deinitialize(&json_builder);

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(PipelineProgramPlan, RejectsRegionCutsUntilMultiRegionPlanningExists) {
  id4_pipeline_program_t* program = CreateRegionCutProgram();
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_pipeline_program_create_plan(
                            &options, iree_allocator_system(), &plan));
  EXPECT_EQ(plan, nullptr);

  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(PipelineProgramPlan, PlansFp8ScaledParameterLoadStep) {
  id4_pipeline_program_t* program = CreateEncodedParameterProgram();
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

  ASSERT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 1u);
  const id4_pipeline_parameter_slab_plan_t* parameter_slab =
      id4_pipeline_plan_parameter_slab_at(plan, 0);
  ASSERT_NE(parameter_slab, nullptr);
  ASSERT_EQ(parameter_slab->request_count, 1u);
  EXPECT_EQ(parameter_slab->byte_length, 32u);
  ExpectStringViewEqual(parameter_slab->requests[0].key,
                        IREE_SV("model.layers.0.linear.weight"));
  EXPECT_EQ(parameter_slab->requests[0].span.length, 32u);

  ASSERT_EQ(id4_pipeline_plan_parameter_load_step_count(plan), 1u);
  const id4_pipeline_parameter_load_step_t* load_step =
      id4_pipeline_plan_parameter_load_step_at(plan, 0);
  ASSERT_NE(load_step, nullptr);
  EXPECT_EQ(
      load_step->kind,
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16);
  EXPECT_EQ(load_step->request_offset, 0u);
  EXPECT_EQ(load_step->request_count, 1u);
  ASSERT_EQ(load_step->source_count, 2u);
  ExpectStringViewEqual(load_step->sources[0].source_scope, IREE_SV("fp8"));
  ExpectStringViewEqual(load_step->sources[0].key,
                        IREE_SV("model.layers.0.linear.weight"));
  EXPECT_EQ(load_step->sources[0].dtype, ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3);
  EXPECT_EQ(load_step->sources[0].byte_length, 16u);
  ExpectStringViewEqual(load_step->sources[1].key,
                        IREE_SV("model.layers.0.linear.weight_scale"));
  EXPECT_EQ(load_step->sources[1].dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  EXPECT_EQ(load_step->sources[1].byte_length, 16u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &builder));
  iree_string_view_t json = iree_string_builder_view(&builder);
  ExpectFinds(json, IREE_SV("encode_fp8_e4m3_scaled_to_bf16"));
  ExpectFinds(json, IREE_SV("model.layers.0.linear.weight_scale"));
  iree_string_builder_deinitialize(&builder);

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(PipelineProgramPlan, GroupsDirectParameterLoadsBySourceScope) {
  id4_pipeline_program_t* program = CreateInterleavedParameterSourceProgram();
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

  ASSERT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 1u);
  const id4_pipeline_parameter_slab_plan_t* parameter_slab =
      id4_pipeline_plan_parameter_slab_at(plan, 0);
  ASSERT_NE(parameter_slab, nullptr);
  EXPECT_EQ(parameter_slab->byte_length, 96u);
  ASSERT_EQ(parameter_slab->request_count, 3u);
  ExpectStringViewEqual(parameter_slab->requests[0].key,
                        IREE_SV("model.layers.0.attn.q.weight"));
  ExpectStringViewEqual(parameter_slab->requests[1].key,
                        IREE_SV("model.layers.0.mlp.w1.weight"));
  ExpectStringViewEqual(parameter_slab->requests[2].key,
                        IREE_SV("model.layers.0.attn.o.weight"));

  ASSERT_EQ(id4_pipeline_plan_parameter_load_step_count(plan), 2u);
  const id4_pipeline_parameter_load_step_t* encode_step =
      id4_pipeline_plan_parameter_load_step_at(plan, 0);
  ASSERT_NE(encode_step, nullptr);
  EXPECT_EQ(
      encode_step->kind,
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16);
  EXPECT_EQ(encode_step->request_offset, 1u);
  EXPECT_EQ(encode_step->request_count, 1u);

  const id4_pipeline_parameter_load_step_t* gather_step =
      id4_pipeline_plan_parameter_load_step_at(plan, 1);
  ASSERT_NE(gather_step, nullptr);
  EXPECT_EQ(gather_step->kind, ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER);
  ExpectStringViewEqual(gather_step->source_scope, IREE_SV("model"));
  EXPECT_EQ(gather_step->target_slab_index, 0u);
  EXPECT_EQ(gather_step->request_count, 2u);
  ASSERT_NE(gather_step->request_indices, nullptr);
  EXPECT_EQ(gather_step->request_indices[0], 0u);
  EXPECT_EQ(gather_step->request_indices[1], 2u);

  id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan);
  EXPECT_EQ(statistics.parameter_direct_source_byte_length, 64u);
  EXPECT_EQ(statistics.parameter_encoded_source_byte_length, 32u);
  EXPECT_EQ(statistics.parameter_gather_load_step_count, 1u);
  EXPECT_EQ(statistics.parameter_encode_load_step_count, 1u);
  EXPECT_EQ(statistics.parameter_load_group_count, 2u);
  EXPECT_EQ(statistics.parameter_gather_load_group_count, 1u);
  EXPECT_EQ(statistics.parameter_encode_load_group_count, 1u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &builder));
  iree_string_view_t json = iree_string_builder_view(&builder);
  ExpectFinds(json, IREE_SV("\"request_indices\":[0,2]"));
  iree_string_builder_deinitialize(&builder);

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(PipelineProgramPlan, DerivesConstantSlabPlan) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  const float scale_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  id4_pipeline_program_tensor_t scale = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_constant_options_t scale_options = {
      /*.structure_size=*/sizeof(scale_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("scale"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
      /*.data=*/iree_make_const_byte_span(scale_values, sizeof(scale_values)),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_constant(builder, &scale_options, &scale));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("output"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(scale),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      /*.structure_size=*/sizeof(dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("scale.dispatch"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/scale"), IREE_SV("scale")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("output"),
      /*.tensor=*/output,
  };
  IREE_ASSERT_OK(id4_pipeline_program_export(builder, &export_options));

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
  options.parameter_slab_binding_slot = 0;
  options.constant_slab_binding_slot = 1;
  options.region_boundary_binding_slot_base = 2;
  options.region_local_binding_slot = 4;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_create_plan(
      &options, iree_allocator_system(), &plan));

  EXPECT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  ASSERT_EQ(id4_pipeline_plan_constant_slab_count(plan), 1u);
  const id4_pipeline_constant_slab_plan_t* constant_slab =
      id4_pipeline_plan_constant_slab_at(plan, 0);
  ASSERT_NE(constant_slab, nullptr);
  EXPECT_EQ(constant_slab->binding_slot, 1u);
  EXPECT_EQ(constant_slab->byte_length, sizeof(scale_values));
  ASSERT_EQ(constant_slab->request_count, 1u);
  ExpectStringViewEqual(constant_slab->requests[0].name, IREE_SV("scale"));
  EXPECT_EQ(constant_slab->requests[0].span.buffer_offset, 0u);
  EXPECT_EQ(constant_slab->requests[0].span.length, sizeof(scale_values));

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(PipelineProgramPlan, ReleasesLocalTensorsAtLastUse) {
  id4_pipeline_program_t* program = CreateLocalReuseProgram();
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

  ASSERT_EQ(id4_pipeline_plan_memory_slab_count(plan), 1u);
  const id4_pipeline_memory_slab_plan_t* slab =
      id4_pipeline_plan_memory_slab_at(plan, 0);
  ASSERT_NE(slab, nullptr);
  EXPECT_EQ(slab->byte_length, 32u);
  EXPECT_EQ(slab->high_water_mark, 32u);

  ASSERT_EQ(id4_pipeline_plan_region_count(plan), 1u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(region->statistics.local_acquire_count, 3u);
  EXPECT_EQ(region->statistics.local_release_count, 3u);
  EXPECT_EQ(region->statistics.local_reuse_count, 1u);
  EXPECT_EQ(region->statistics.local_slab_byte_length, 32u);
  EXPECT_EQ(region->statistics.local_slab_high_water_mark, 32u);
  ASSERT_EQ(region->local_lifetime_count, 3u);
  const id4_pipeline_region_local_lifetime_t* first =
      &region->local_lifetimes[0];
  ExpectStringViewEqual(first->name, IREE_SV("hidden_states.first"));
  EXPECT_EQ(first->offset, 0u);
  EXPECT_EQ(first->byte_length, 16u);
  EXPECT_EQ(first->acquire_operation_ordinal, 0u);
  EXPECT_EQ(first->release_operation_ordinal, 3u);
  const id4_pipeline_region_local_lifetime_t* second =
      &region->local_lifetimes[1];
  ExpectStringViewEqual(second->name, IREE_SV("hidden_states.second"));
  EXPECT_EQ(second->offset, 16u);
  EXPECT_EQ(second->release_operation_ordinal, 3u);
  const id4_pipeline_region_local_lifetime_t* third =
      &region->local_lifetimes[2];
  ExpectStringViewEqual(third->name, IREE_SV("hidden_states.third"));
  EXPECT_EQ(third->offset, 0u);
  EXPECT_TRUE(iree_all_bits_set(
      third->flags, ID4_PIPELINE_REGION_LOCAL_LIFETIME_FLAG_REUSED));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &builder));
  std::string json = std::string(iree_string_builder_view(&builder).data,
                                 iree_string_builder_view(&builder).size);
  EXPECT_NE(json.find("\"local_lifetimes\""), std::string::npos);
  EXPECT_NE(json.find("\"hidden_states.first\""), std::string::npos);
  EXPECT_NE(json.find("\"release_operation_ordinal\":3"), std::string::npos);
  iree_string_builder_deinitialize(&builder);

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
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("input.tap"),
  };
  options.flags = ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_pipeline_program_create_plan(
                            &options, iree_allocator_system(), &plan));

  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

}  // namespace
