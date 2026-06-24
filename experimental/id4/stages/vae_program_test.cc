// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/vae_program.h"

#include "experimental/id4/pipeline/program.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_host_size_t kProgramBuilderBlockSize = 16 * 1024;

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
        /*.program_name=*/IREE_SV("test.vae"),
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
  // Arena block pool backing the program builder.
  iree_arena_block_pool_t block_pool_;
  // Owned builder under test.
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

static id4_vae_model_config_t MakeSmallModelConfig() {
  return id4_vae_model_config_t{
      // Latent-to-image scale factor along the width axis.
      /*.scale_x=*/2,
      // Latent-to-image scale factor along the height axis.
      /*.scale_y=*/2,
      // Latent-to-media scale factor along the temporal axis.
      /*.scale_t=*/1,
      // Channel count in latent tensors.
      /*.latent_channel_count=*/1,
      // Channel count in decoded tensors.
      /*.decoded_channel_count=*/1,
      // Minimum latent tile width.
      /*.min_tile_size_x=*/1,
      // Minimum latent tile height.
      /*.min_tile_size_y=*/1,
      // Default latent tile width.
      /*.default_tile_size_x=*/2,
      // Default latent tile height.
      /*.default_tile_size_y=*/2,
      // Maximum legal overlap.
      /*.max_overlap=*/0.5f,
      // Supported implementation capabilities.
      /*.capabilities=*/ID4_VAE_CAPABILITY_DECODE |
          ID4_VAE_CAPABILITY_SPATIAL_TILING,
  };
}

static id4_vae_decode_request_config_t MakeExplicitRequest(
    id4_pipeline_program_shape_t latent_shape, uint32_t tile_size_x,
    uint32_t tile_size_y, float overlap) {
  id4_vae_decode_request_config_t request = {
      // Latent tensor shape.
      /*.latent_shape=*/latent_shape,
      // Tiling policy.
      /*.tiling=*/
      {
          // Tiling policy selected for this request.
          /*.mode=*/ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE,
          // Requested latent tile width.
          /*.tile_size_x=*/tile_size_x,
          // Requested latent tile height.
          /*.tile_size_y=*/tile_size_y,
          // Relative latent width factor.
          /*.relative_size_x=*/0.0f,
          // Relative latent height factor.
          /*.relative_size_y=*/0.0f,
          // Requested overlap.
          /*.overlap=*/overlap,
          // Memory budget.
          /*.memory_budget=*/0,
      },
  };
  return request;
}

static id4_vae_program_options_t MakeProgramOptions(
    id4_vae_model_config_t model, id4_vae_decode_request_config_t request) {
  id4_vae_program_options_t options = {
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Static VAE implementation capabilities.
      /*.model=*/model,
      // Dynamic decode request.
      /*.request=*/request,
  };
  return options;
}

static id4_pipeline_program_t* CreateVaeProgram(
    const id4_vae_program_options_t* options) {
  ProgramBuilderScope builder_scope;
  IREE_CHECK_OK(
      id4_vae_program_author_decode(options, builder_scope.builder()));
  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder_scope.builder(), iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static bool ShapeEquals(id4_pipeline_program_shape_t actual,
                        id4_pipeline_program_shape_t expected) {
  if (actual.rank != expected.rank) return false;
  for (uint32_t i = 0; i < actual.rank; ++i) {
    if (actual.dims[i] != expected.dims[i]) return false;
  }
  return true;
}

static bool ProgramExportsTensorWithShape(
    const id4_pipeline_program_t* program, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t expected_shape) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation || operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT) {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(
            program, operation->payload.export_value.tensor.ordinal);
    if (!tensor) continue;
    if (tensor->dtype != dtype) continue;
    if (!ShapeEquals(tensor->shape, expected_shape)) continue;
    return true;
  }
  return false;
}

TEST(VaeProgram, ResolvesStableDiffusionStyleFlux2Tiling) {
  const id4_vae_model_config_t* model = id4_vae_program_flux2_model_config();
  id4_vae_decode_request_config_t request = MakeExplicitRequest(
      id4_pipeline_program_make_shape_rank4(64, 64, 128, 1), 32, 32, 0.5f);

  id4_vae_decode_tiling_plan_t tiling_plan;
  IREE_ASSERT_OK(
      id4_vae_program_resolve_decode_tiling(*model, request, &tiling_plan));

  EXPECT_EQ(tiling_plan.decoded_height, 1024u);
  EXPECT_EQ(tiling_plan.decoded_width, 1024u);
  EXPECT_EQ(tiling_plan.tile_size_x, 32u);
  EXPECT_EQ(tiling_plan.tile_size_y, 32u);
  EXPECT_EQ(tiling_plan.tile_count_x, 3u);
  EXPECT_EQ(tiling_plan.tile_count_y, 3u);
  EXPECT_EQ(tiling_plan.overlap_pixels_x, 16u);
  EXPECT_EQ(tiling_plan.overlap_pixels_y, 16u);
  EXPECT_EQ(tiling_plan.tile_step_x, 16u);
  EXPECT_EQ(tiling_plan.tile_step_y, 16u);
  EXPECT_EQ(tiling_plan.overlap_milli, 500u);
  EXPECT_EQ(tiling_plan.tile_element_count, 1u * 512u * 512u * 3u);
}

TEST(VaeProgram, ResolvesAdjustedOverlapForNonDivisibleTileGrid) {
  const id4_vae_model_config_t* model = id4_vae_program_flux2_model_config();
  id4_vae_decode_request_config_t request = MakeExplicitRequest(
      id4_pipeline_program_make_shape_rank4(65, 65, 128, 1), 32, 32, 0.5f);

  id4_vae_decode_tiling_plan_t tiling_plan;
  IREE_ASSERT_OK(
      id4_vae_program_resolve_decode_tiling(*model, request, &tiling_plan));

  EXPECT_EQ(tiling_plan.tile_count_x, 3u);
  EXPECT_EQ(tiling_plan.tile_count_y, 3u);
  EXPECT_EQ(tiling_plan.overlap_pixels_x, 15u);
  EXPECT_EQ(tiling_plan.overlap_pixels_y, 15u);
  EXPECT_EQ(tiling_plan.tile_step_x, 17u);
  EXPECT_EQ(tiling_plan.tile_step_y, 17u);
  EXPECT_EQ(tiling_plan.overlap_milli, 484u);
}

TEST(VaeProgram, ResolvesMemoryBudgetTiling) {
  id4_vae_model_config_t model = MakeSmallModelConfig();
  id4_vae_decode_request_config_t request = {};
  request.latent_shape = id4_pipeline_program_make_shape_rank4(8, 8, 1, 1);
  request.tiling.mode = ID4_VAE_TILING_MODE_MEMORY_BUDGET;
  request.tiling.overlap = 0.0f;
  request.tiling.memory_budget = 4 * 4 * sizeof(float);

  id4_vae_decode_tiling_plan_t tiling_plan;
  IREE_ASSERT_OK(
      id4_vae_program_resolve_decode_tiling(model, request, &tiling_plan));

  EXPECT_LE(tiling_plan.estimated_transient_peak, request.tiling.memory_budget);
  EXPECT_GE(tiling_plan.tile_size_x, model.min_tile_size_x);
  EXPECT_GE(tiling_plan.tile_size_y, model.min_tile_size_y);
}

TEST(VaeProgram, AuthorsDecodeBoundaryContract) {
  id4_vae_model_config_t model = MakeSmallModelConfig();
  id4_vae_decode_request_config_t request = MakeExplicitRequest(
      id4_pipeline_program_make_shape_rank4(2, 2, 1, 1), 2, 2, 0.0f);
  id4_vae_program_options_t options = MakeProgramOptions(model, request);
  id4_pipeline_program_t* program = CreateVaeProgram(&options);

  EXPECT_TRUE(ProgramExportsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank4(4, 4, 1, 1)));

  id4_pipeline_program_release(program);
}

TEST(VaeProgram, RejectsLatentShapeWithWrongChannelCount) {
  ProgramBuilderScope builder_scope;
  id4_vae_model_config_t model = MakeSmallModelConfig();
  id4_vae_decode_request_config_t request = MakeExplicitRequest(
      id4_pipeline_program_make_shape_rank4(2, 2, 2, 1), 2, 2, 0.0f);
  id4_vae_program_options_t options = MakeProgramOptions(model, request);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_vae_program_author_decode(&options, builder_scope.builder()));
}

TEST(VaeProgram, RejectsTileSizeOutsideLatentShape) {
  ProgramBuilderScope builder_scope;
  id4_vae_model_config_t model = MakeSmallModelConfig();
  id4_vae_decode_request_config_t request = MakeExplicitRequest(
      id4_pipeline_program_make_shape_rank4(2, 2, 1, 1), 3, 2, 0.0f);
  id4_vae_program_options_t options = MakeProgramOptions(model, request);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      id4_vae_program_author_decode(&options, builder_scope.builder()));
}

}  // namespace
