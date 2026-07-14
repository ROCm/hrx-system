// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/lora_bake_plan.h"

#include <cstdio>
#include <string>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_plan.h"
#include "experimental/id4/stages/test_util.h"
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
        /*.program_name=*/IREE_SV("test.lora_bake"),
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

  id4_pipeline_program_t* Seal() {
    id4_pipeline_program_t* program = nullptr;
    IREE_CHECK_OK(id4_pipeline_program_builder_seal(
        builder_, iree_allocator_system(), &program));
    id4_pipeline_program_builder_destroy(builder_);
    builder_ = nullptr;
    return program;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

static id4_ideogram4_dit_model_config_t MakeModelConfig() {
  id4_ideogram4_dit_model_config_t model = {};
  model.layer_count = 1;
  model.hidden_size = 32;
  model.intermediate_size = 48;
  model.adaln_size = 16;
  return model;
}

static iree_status_t AddLoraTensor(iree_io_parameter_index_t* index,
                                   iree_string_view_t key, uint32_t rows,
                                   uint32_t columns) {
  char metadata_buffer[128];
  const int metadata_length =
      std::snprintf(metadata_buffer, sizeof(metadata_buffer),
                    "{\"dtype\":\"BF16\",\"shape\":[%" PRIu32 ",%" PRIu32
                    "],\"data_offsets\":[0,%" PRIu64 "]}",
                    rows, columns, (uint64_t)rows * columns * sizeof(uint16_t));
  if (metadata_length < 0 ||
      (iree_host_size_t)metadata_length >= sizeof(metadata_buffer)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "test LoRA metadata overflow");
  }
  iree_io_parameter_index_entry_t entry = {};
  entry.key = key;
  entry.metadata = iree_make_const_byte_span(metadata_buffer, metadata_length);
  entry.length = (uint64_t)rows * columns * sizeof(uint16_t);
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length = 1;
  return iree_io_parameter_index_add(index, &entry);
}

static id4_ideogram4_lora_topology_t* CreateTopology() {
  iree_io_parameter_index_t* index = nullptr;
  IREE_CHECK_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));
  IREE_CHECK_OK(AddLoraTensor(
      index, IREE_SV("diffusion_model.layers.0.attention.qkv.lora_A.weight"),
      /*rows=*/2, /*columns=*/32));
  IREE_CHECK_OK(AddLoraTensor(
      index, IREE_SV("diffusion_model.layers.0.attention.qkv.lora_B.weight"),
      /*rows=*/96, /*columns=*/2));

  id4_ideogram4_lora_import_options_t import_options = {};
  import_options.structure_size = sizeof(import_options);
  import_options.model = MakeModelConfig();
  import_options.parameter_index = index;
  import_options.source_scope = IREE_SV("adapter");
  id4_ideogram4_lora_t* lora = nullptr;
  IREE_CHECK_OK(id4_ideogram4_lora_import(&import_options,
                                          iree_allocator_system(), &lora));
  iree_io_parameter_index_release(index);

  id4_ideogram4_lora_t* loras[] = {lora};
  id4_ideogram4_lora_topology_create_options_t topology_options = {};
  topology_options.structure_size = sizeof(topology_options);
  topology_options.lora_count = IREE_ARRAYSIZE(loras);
  topology_options.loras = loras;
  id4_ideogram4_lora_topology_t* topology = nullptr;
  IREE_CHECK_OK(id4_ideogram4_lora_topology_create(
      &topology_options, iree_allocator_system(), &topology));
  id4_ideogram4_lora_release(lora);
  return topology;
}

typedef uint32_t BaseProgramFlags;
typedef enum BaseProgramFlagBits {
  BASE_PROGRAM_FLAG_NONE = 0u,
  BASE_PROGRAM_FLAG_INCLUDE_SCALE = 1u << 0,
  BASE_PROGRAM_FLAG_INCLUDE_DYNAMIC_STRENGTHS = 1u << 1,
} BaseProgramFlagBits;

typedef struct BaseProgramOptions {
  // Parameter encoding assigned to the target weight.
  id4_pipeline_program_parameter_encoding_t weight_encoding;
  // Optional base-program features used by negative tests.
  BaseProgramFlags flags;
} BaseProgramOptions;

static id4_pipeline_program_t* CreateBaseProgram(BaseProgramOptions options) {
  ProgramBuilderScope builder_scope;
  const id4_pipeline_program_parameter_source_t weight_source = {
      /*.source_scope=*/IREE_SV("model"),
      /*.key=*/IREE_SV("layers.0.attention.qkv.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(96, 32),
  };
  id4_pipeline_program_parameter_options_t weight_options = {
      /*.structure_size=*/sizeof(weight_options),
      /*.next=*/nullptr,
      /*.encoding=*/options.weight_encoding,
      /*.source_count=*/1,
      /*.sources=*/&weight_source,
      /*.key=*/weight_source.key,
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
      /*.shape=*/weight_source.shape,
      /*.source_span_count=*/0,
      /*.source_spans=*/nullptr,
      /*.domain=*/IREE_SV("lora_patchable"),
  };
  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  IREE_CHECK_OK(id4_pipeline_program_parameter(builder_scope.builder(),
                                               &weight_options, &weight));

  if (iree_any_bit_set(options.flags, BASE_PROGRAM_FLAG_INCLUDE_SCALE)) {
    const id4_pipeline_program_parameter_source_t scale_source = {
        /*.source_scope=*/IREE_SV("model"),
        /*.key=*/IREE_SV("layers.0.attention.qkv.weight_scale"),
        /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
        /*.shape=*/id4_pipeline_program_make_shape_rank1(96),
    };
    id4_pipeline_program_parameter_options_t scale_options = {
        /*.structure_size=*/sizeof(scale_options),
        /*.next=*/nullptr,
        /*.encoding=*/ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
        /*.source_count=*/1,
        /*.sources=*/&scale_source,
        /*.key=*/scale_source.key,
        /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
        /*.shape=*/scale_source.shape,
        /*.source_span_count=*/0,
        /*.source_spans=*/nullptr,
        /*.domain=*/IREE_SV("lora_patchable"),
    };
    id4_pipeline_program_tensor_t scale = id4_pipeline_program_tensor_invalid();
    IREE_CHECK_OK(id4_pipeline_program_parameter(builder_scope.builder(),
                                                 &scale_options, &scale));
  }

  if (iree_any_bit_set(options.flags,
                       BASE_PROGRAM_FLAG_INCLUDE_DYNAMIC_STRENGTHS)) {
    id4_pipeline_program_import_tensor_options_t import_options = {
        /*.structure_size=*/sizeof(import_options),
        /*.next=*/nullptr,
        /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
        /*.name=*/IREE_SV("lora.strengths"),
        /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
        /*.shape=*/id4_pipeline_program_make_shape_rank1(1),
    };
    id4_pipeline_program_tensor_t strengths =
        id4_pipeline_program_tensor_invalid();
    IREE_CHECK_OK(id4_pipeline_program_import_tensor(
        builder_scope.builder(), &import_options, &strengths));
    const id4_pipeline_program_dispatch_binding_t bindings[] = {
        id4_pipeline_program_read(strengths),
    };
    id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
        /*.structure_size=*/sizeof(dispatch_options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("test.consume_lora_strengths"),
        /*.kernel=*/
        id4_pipeline_make_kernel_ref(IREE_SV("test/consume_lora_strengths"),
                                     IREE_SV("consume_lora_strengths")),
        /*.config_binding_count=*/0,
        /*.config_bindings=*/nullptr,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.bindings=*/bindings,
    };
    IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder_scope.builder(),
                                                     &dispatch_options));
  }

  return builder_scope.Seal();
}

static id4_pipeline_plan_t* CreatePlan(BaseProgramOptions base_options) {
  id4_pipeline_program_t* program = CreateBaseProgram(base_options);
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  const id4_pipeline_device_placement_t placement = {
      /*.role=*/IREE_SV("default"),
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  iree_hal_buffer_params_t parameter_params =
      id4_pipeline_parameter_slab_device_local_params(
          IREE_HAL_QUEUE_AFFINITY_ANY,
          IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
              IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          /*min_alignment=*/16);
  iree_hal_buffer_params_t local_params = {};
  local_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  local_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  local_params.min_alignment = 16;
  id4_pipeline_program_plan_options_t plan_options = {
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.stage_name=*/IREE_SV("ideogram4.dit.conditioned"),
      /*.program=*/program,
      /*.device_group=*/device_group,
      /*.placement_count=*/1,
      /*.placements=*/&placement,
      /*.parameter_scope=*/IREE_SV("model"),
      /*.parameter_slab_placement_id=*/0,
      /*.parameter_slab_binding_slot_base=*/0,
      /*.parameter_slab_target_params=*/parameter_params,
      /*.parameter_slab_alignment=*/16,
      /*.parameter_request_alignment=*/16,
      /*.constant_slab_placement_id=*/0,
      /*.constant_slab_binding_slot=*/2,
      /*.constant_slab_target_params=*/parameter_params,
      /*.constant_slab_alignment=*/16,
      /*.constant_request_alignment=*/16,
      /*.kernel_placement_id=*/0,
      /*.region_placement_id=*/0,
      /*.region_local_slab_params=*/local_params,
      /*.region_local_slab_alignment=*/16,
      /*.region_local_tensor_alignment=*/16,
      /*.region_binding_capacity=*/5,
      /*.region_local_binding_slot=*/3,
      /*.region_shared_binding_slot=*/4,
      /*.region_boundary_binding_slot_base=*/1,
      /*.diagnostic_tap_names=*/iree_string_view_list_empty(),
      /*.diagnostic_tap_binding_slot_base=*/2,
      /*.diagnostics_sink=*/&diagnostics_sink,
  };
  id4_pipeline_plan_t* plan = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_create_plan(
      &plan_options, iree_allocator_system(), &plan));
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
  return plan;
}

static iree_status_t CreateBakePlan(
    const id4_pipeline_plan_t* base_plan,
    id4_ideogram4_lora_topology_t* topology,
    iree_device_size_t working_set_byte_capacity,
    id4_ideogram4_lora_bake_plan_t** out_plan) {
  id4_ideogram4_lora_bake_plan_create_options_t options = {};
  options.structure_size = sizeof(options);
  options.base_plan = base_plan;
  options.topology = topology;
  options.working_set_byte_capacity = working_set_byte_capacity;
  return id4_ideogram4_lora_bake_plan_create(&options, iree_allocator_system(),
                                             out_plan);
}

TEST(Ideogram4LoraBakePlan, PlansBoundedCompactTargetWindows) {
  id4_pipeline_plan_t* base_plan = CreatePlan({
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      BASE_PROGRAM_FLAG_INCLUDE_SCALE,
  });
  id4_ideogram4_lora_topology_t* topology = CreateTopology();

  const iree_device_size_t working_set_byte_capacity = 4096;
  id4_ideogram4_lora_bake_plan_t* bake_plan = nullptr;
  IREE_ASSERT_OK(CreateBakePlan(base_plan, topology, working_set_byte_capacity,
                                &bake_plan));
  id4_pipeline_plan_release(base_plan);
  id4_ideogram4_lora_topology_release(topology);

  EXPECT_NE(id4_ideogram4_lora_bake_plan_base_plan(bake_plan), nullptr);
  EXPECT_NE(id4_ideogram4_lora_bake_plan_topology(bake_plan), nullptr);
  EXPECT_EQ(id4_ideogram4_lora_bake_plan_target_count(bake_plan), 1u);
  EXPECT_EQ(id4_ideogram4_lora_bake_plan_adapter_byte_length(bake_plan),
            (2u * 32u + 96u * 2u) * sizeof(uint16_t));
  EXPECT_LE(id4_ideogram4_lora_bake_plan_working_set_high_water_mark(bake_plan),
            working_set_byte_capacity);

  const id4_ideogram4_lora_bake_target_t* target =
      id4_ideogram4_lora_bake_plan_target_at(bake_plan, 0);
  ASSERT_NE(target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target->base_parameter_key,
                                     IREE_SV("layers.0.attention.qkv.weight")));
  EXPECT_EQ(target->input_size, 32u);
  EXPECT_EQ(target->output_size, 96u);
  EXPECT_EQ(target->total_rank, 2u);
  EXPECT_EQ(target->maximum_segment_rank, 2u);
  EXPECT_GT(target->output_rows_per_window, 0u);
  EXPECT_EQ(target->output_rows_per_window % 16u, 0u);
  EXPECT_LE(target->output_rows_per_window, target->output_size);
  EXPECT_EQ(target->window_count,
            (target->output_size + target->output_rows_per_window - 1) /
                target->output_rows_per_window);
  EXPECT_LE(target->working_set.byte_length, working_set_byte_capacity);
  EXPECT_EQ(target->working_set.down_source.length,
            2u * 32u * sizeof(uint16_t));
  EXPECT_EQ(target->working_set.down.length, 16u * 32u * sizeof(uint16_t));
  EXPECT_EQ(target->working_set.up.length,
            target->output_rows_per_window * 2u * sizeof(uint16_t));
  EXPECT_EQ(target->working_set.effective_weight.length,
            target->output_rows_per_window * 32u * sizeof(uint16_t));
  EXPECT_EQ(target->working_set.strengths.length, sizeof(float));
  EXPECT_EQ(target->working_set.strengths.offset, 0u);
  EXPECT_LE(target->working_set.strengths.offset +
                target->working_set.strengths.length,
            target->working_set.down_source.offset);
  EXPECT_LE(target->working_set.down_source.offset +
                target->working_set.down_source.length,
            target->working_set.down.offset);
  EXPECT_LE(target->working_set.down.offset + target->working_set.down.length,
            target->working_set.up.offset);
  EXPECT_LE(target->working_set.up.offset + target->working_set.up.length,
            target->working_set.effective_weight.offset);
  EXPECT_LE(target->working_set.effective_weight.offset +
                target->working_set.effective_weight.length,
            target->working_set.byte_length);
  EXPECT_EQ(target->weight_range.length, 96u * 32u);
  EXPECT_EQ(target->scale_range.length, 96u * sizeof(float));
  EXPECT_LE(target->weight_range.offset + target->weight_range.length,
            id4_ideogram4_lora_bake_plan_patchable_slab_byte_length(bake_plan));
  EXPECT_LE(target->scale_range.offset + target->scale_range.length,
            id4_ideogram4_lora_bake_plan_patchable_slab_byte_length(bake_plan));
  EXPECT_EQ(id4_ideogram4_lora_bake_plan_target_at(bake_plan, 1), nullptr);

  id4_ideogram4_lora_bake_plan_release(bake_plan);
}

TEST(Ideogram4LoraBakePlan, WorkingBudgetControlsWindowShape) {
  id4_pipeline_plan_t* base_plan = CreatePlan({
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      BASE_PROGRAM_FLAG_INCLUDE_SCALE,
  });
  id4_ideogram4_lora_topology_t* topology = CreateTopology();
  id4_ideogram4_lora_bake_plan_t* large_plan = nullptr;
  id4_ideogram4_lora_bake_plan_t* small_plan = nullptr;
  IREE_ASSERT_OK(CreateBakePlan(base_plan, topology,
                                /*working_set_byte_capacity=*/16384,
                                &large_plan));
  IREE_ASSERT_OK(CreateBakePlan(base_plan, topology,
                                /*working_set_byte_capacity=*/4096,
                                &small_plan));

  const id4_ideogram4_lora_bake_target_t* large_target =
      id4_ideogram4_lora_bake_plan_target_at(large_plan, 0);
  const id4_ideogram4_lora_bake_target_t* small_target =
      id4_ideogram4_lora_bake_plan_target_at(small_plan, 0);
  ASSERT_NE(large_target, nullptr);
  ASSERT_NE(small_target, nullptr);
  EXPECT_GE(large_target->output_rows_per_window,
            small_target->output_rows_per_window);
  EXPECT_LE(large_target->window_count, small_target->window_count);

  id4_ideogram4_lora_bake_plan_release(small_plan);
  id4_ideogram4_lora_bake_plan_release(large_plan);
  id4_ideogram4_lora_topology_release(topology);
  id4_pipeline_plan_release(base_plan);
}

TEST(Ideogram4LoraBakePlan, RejectsInsufficientWorkingBudget) {
  id4_pipeline_plan_t* base_plan = CreatePlan({
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      BASE_PROGRAM_FLAG_INCLUDE_SCALE,
  });
  id4_ideogram4_lora_topology_t* topology = CreateTopology();
  id4_ideogram4_lora_bake_plan_t* bake_plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      CreateBakePlan(base_plan, topology, /*working_set_byte_capacity=*/128,
                     &bake_plan));
  EXPECT_EQ(bake_plan, nullptr);
  id4_ideogram4_lora_topology_release(topology);
  id4_pipeline_plan_release(base_plan);
}

TEST(Ideogram4LoraBakePlan, RejectsNonCompactBaseWeight) {
  id4_pipeline_plan_t* base_plan = CreatePlan({
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT,
      BASE_PROGRAM_FLAG_INCLUDE_SCALE,
  });
  id4_ideogram4_lora_topology_t* topology = CreateTopology();
  id4_ideogram4_lora_bake_plan_t* bake_plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateBakePlan(base_plan, topology, /*working_set_byte_capacity=*/4096,
                     &bake_plan));
  EXPECT_EQ(bake_plan, nullptr);
  id4_ideogram4_lora_topology_release(topology);
  id4_pipeline_plan_release(base_plan);
}

TEST(Ideogram4LoraBakePlan, RejectsMissingOutputRowScale) {
  id4_pipeline_plan_t* base_plan = CreatePlan({
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      BASE_PROGRAM_FLAG_NONE,
  });
  id4_ideogram4_lora_topology_t* topology = CreateTopology();
  id4_ideogram4_lora_bake_plan_t* bake_plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      CreateBakePlan(base_plan, topology, /*working_set_byte_capacity=*/4096,
                     &bake_plan));
  EXPECT_EQ(bake_plan, nullptr);
  id4_ideogram4_lora_topology_release(topology);
  id4_pipeline_plan_release(base_plan);
}

TEST(Ideogram4LoraBakePlan, RejectsDynamicAdapterProgram) {
  id4_pipeline_plan_t* base_plan = CreatePlan({
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE,
      BASE_PROGRAM_FLAG_INCLUDE_SCALE |
          BASE_PROGRAM_FLAG_INCLUDE_DYNAMIC_STRENGTHS,
  });
  id4_ideogram4_lora_topology_t* topology = CreateTopology();
  id4_ideogram4_lora_bake_plan_t* bake_plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateBakePlan(base_plan, topology, /*working_set_byte_capacity=*/4096,
                     &bake_plan));
  EXPECT_EQ(bake_plan, nullptr);
  id4_ideogram4_lora_topology_release(topology);
  id4_pipeline_plan_release(base_plan);
}

}  // namespace
