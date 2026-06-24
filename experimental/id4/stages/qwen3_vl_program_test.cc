// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl_program.h"

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_plan.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_host_size_t kProgramBuilderBlockSize = 64 * 1024;
static constexpr uint32_t kSelectedLayerOrdinals[] = {0, 1};

static id4_qwen3_vl_program_options_t MakeProgramOptions(uint32_t layer_count) {
  id4_qwen3_vl_program_options_t options = {
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Static model dimensions.
      /*.model=*/
      {
          // Number of decoder layers.
          /*.layer_count=*/layer_count,
          // Vocabulary row count.
          /*.vocab_size=*/32,
          // Hidden-state channel count.
          /*.hidden_size=*/32,
          // MLP intermediate channel count.
          /*.intermediate_size=*/64,
          // Number of query attention heads.
          /*.attention_head_count=*/2,
          // Number of key/value attention heads.
          /*.key_value_head_count=*/2,
          // Channel count per attention head.
          /*.head_size=*/16,
          // Number of selected layer outputs.
          /*.selected_layer_count=*/IREE_ARRAYSIZE(kSelectedLayerOrdinals),
          // Selected layer output ordinals.
          /*.selected_layer_ordinals=*/kSelectedLayerOrdinals,
      },
      // Dynamic request dimensions.
      /*.request=*/
      {
          // Number of token positions.
          /*.token_count=*/19,
      },
  };
  return options;
}

class ProgramBuilderScope {
 public:
  ProgramBuilderScope() {
    iree_arena_block_pool_initialize(kProgramBuilderBlockSize,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_program_builder_create_options_t options = {
        // Size of this structure for versioning.
        /*.structure_size=*/sizeof(options),
        // Extension structure chain.
        /*.next=*/nullptr,
        // Program name copied by the builder.
        /*.program_name=*/IREE_SV("test.forward"),
        // Arena block pool used by the builder.
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

static id4_pipeline_program_t* CreateForwardProgram(
    const id4_qwen3_vl_program_options_t* options) {
  ProgramBuilderScope builder_scope;
  IREE_CHECK_OK(
      id4_qwen3_vl_program_author_forward(options, builder_scope.builder()));
  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder_scope.builder(), iree_allocator_system(), &program));
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
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Planning behavior flags.
      /*.flags=*/0,
      // Stage boundary name copied into the plan.
      /*.stage_name=*/IREE_SV("qwen.encoder"),
      // Semantic program to lower.
      /*.program=*/program,
      // Retained HAL device group.
      /*.device_group=*/device_group,
      // Number of placements.
      /*.placement_count=*/1,
      // Explicit placement table.
      /*.placements=*/placement,
      // Provider parameter scope.
      /*.parameter_scope=*/IREE_SV(""),
      // Placement for the packed parameter slab.
      /*.parameter_slab_placement_id=*/0,
      // Binding table slot for the packed parameter slab.
      /*.parameter_slab_binding_slot=*/0,
      // HAL buffer parameters for the parameter slab.
      /*.parameter_slab_target_params=*/parameter_params,
      // Base alignment for the packed parameter slab.
      /*.parameter_slab_alignment=*/16,
      // Per-request alignment for packed parameters.
      /*.parameter_request_alignment=*/16,
      // Placement for the packed constant slab.
      /*.constant_slab_placement_id=*/0,
      // Binding table slot for the packed constant slab.
      /*.constant_slab_binding_slot=*/2,
      // HAL buffer parameters for the constant slab.
      /*.constant_slab_target_params=*/parameter_params,
      // Base alignment for the packed constant slab.
      /*.constant_slab_alignment=*/16,
      // Per-request alignment for packed constants.
      /*.constant_request_alignment=*/16,
      // Placement for kernels.
      /*.kernel_placement_id=*/0,
      // Placement for the executable region.
      /*.region_placement_id=*/0,
      // HAL buffer parameters for the local transient slab.
      /*.region_local_slab_params=*/local_params,
      // Base alignment for the local transient slab.
      /*.region_local_slab_alignment=*/16,
      // Per-tensor alignment inside the local transient slab.
      /*.region_local_tensor_alignment=*/16,
      // Binding table capacity for the derived region.
      /*.region_binding_capacity=*/8,
      // Local transient slab binding slot.
      /*.region_local_binding_slot=*/7,
      // First external boundary tensor binding slot.
      /*.region_boundary_binding_slot_base=*/1,
      // Selected diagnostic tap names.
      /*.diagnostic_tap_names=*/iree_string_view_list_empty(),
      // First diagnostic tap binding slot.
      /*.diagnostic_tap_binding_slot_base=*/0,
      // Diagnostics sink used during plan creation.
      /*.diagnostics_sink=*/diagnostics_sink,
  };
  return options;
}

static bool ShapeEquals(id4_pipeline_program_shape_t actual,
                        id4_pipeline_program_shape_t expected) {
  if (actual.rank != expected.rank) return false;
  for (uint32_t i = 0; i < actual.rank; ++i) {
    if (actual.dims[i] != expected.dims[i]) return false;
  }
  return true;
}

static bool ProgramExportsTensorWithShape(const id4_pipeline_program_t* program,
                                          id4_pipeline_program_dtype_t dtype,
                                          id4_pipeline_program_shape_t shape) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT) continue;
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(
            program, operation->payload.export_value.tensor.ordinal);
    if (!tensor) continue;
    if (tensor->dtype != dtype) continue;
    if (!ShapeEquals(tensor->shape, shape)) continue;
    return true;
  }
  return false;
}

TEST(Qwen3VlProgram, AuthorsForwardContractAndDerivedPlan) {
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/2);
  id4_pipeline_program_t* program = CreateForwardProgram(&options);

  EXPECT_TRUE(ProgramExportsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(
          options.model.selected_layer_count * options.model.hidden_size,
          options.request.token_count)));

  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_pipeline_device_placement_t placement = {
      // Human-readable role for diagnostics.
      /*.role=*/IREE_SV("default"),
      // Device index within the group.
      /*.device_index=*/0,
      // Queue affinity for planned work.
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_program_plan_options_t plan_options =
      MakePlanOptions(program, device_group, &placement, &diagnostics_sink);

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_create_plan(
      &plan_options, iree_allocator_system(), &plan));

  EXPECT_GT(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_memory_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_boundary_tensor_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_region_count(plan), 0u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_GT(region->statistics.local_acquire_count, 0u);
  EXPECT_GT(region->statistics.local_slab_byte_length, 0u);

  id4_pipeline_plan_release(plan);
  iree_hal_device_group_release(device_group);
  id4_pipeline_program_release(program);
}

TEST(Qwen3VlProgram, RejectsInvalidModelDimensions) {
  ProgramBuilderScope builder_scope;
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/1);
  options.model.head_size = 3;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_qwen3_vl_program_author_forward(&options, builder_scope.builder()));
}

TEST(Qwen3VlProgram, RejectsMissingTokenCount) {
  ProgramBuilderScope builder_scope;
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/1);
  options.request.token_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_qwen3_vl_program_author_forward(&options, builder_scope.builder()));
}

TEST(Qwen3VlProgram, RejectsMissingLayerCount) {
  ProgramBuilderScope builder_scope;
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_qwen3_vl_program_author_forward(&options, builder_scope.builder()));
}

TEST(Qwen3VlProgram, RejectsMissingSelectedLayers) {
  ProgramBuilderScope builder_scope;
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/1);
  options.model.selected_layer_count = 0;
  options.model.selected_layer_ordinals = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_qwen3_vl_program_author_forward(&options, builder_scope.builder()));
}

TEST(Qwen3VlProgram, RejectsInvalidSelectedLayerOrdinal) {
  ProgramBuilderScope builder_scope;
  static constexpr uint32_t kInvalidSelectedLayerOrdinals[] = {0, 2};
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/2);
  options.model.selected_layer_ordinals = kInvalidSelectedLayerOrdinals;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_qwen3_vl_program_author_forward(&options, builder_scope.builder()));
}

TEST(Qwen3VlProgram, RejectsUnsortedSelectedLayerOrdinals) {
  ProgramBuilderScope builder_scope;
  static constexpr uint32_t kUnsortedSelectedLayerOrdinals[] = {1, 0};
  id4_qwen3_vl_program_options_t options = MakeProgramOptions(
      /*layer_count=*/2);
  options.model.selected_layer_ordinals = kUnsortedSelectedLayerOrdinals;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_qwen3_vl_program_author_forward(&options, builder_scope.builder()));
}

}  // namespace
