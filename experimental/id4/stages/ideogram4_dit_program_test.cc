// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program.h"

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
        /*.program_name=*/IREE_SV("test.ideogram4.dit"),
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

static id4_ideogram4_dit_model_config_t MakeModelConfig() {
  return id4_ideogram4_dit_model_config_t{
      // Number of transformer blocks in the DiT.
      /*.layer_count=*/1,
      // Channel count of each VAE latent image token.
      /*.input_channel_count=*/4,
      // Transformer hidden-state channel count.
      /*.hidden_size=*/32,
      // Feed-forward intermediate channel count.
      /*.intermediate_size=*/64,
      // Transformer attention head count.
      /*.attention_head_count=*/2,
      // AdaLN conditioning vector channel count.
      /*.adaln_size=*/4,
      // Qwen condition feature channel count.
      /*.llm_feature_count=*/208,
      // Number of image-indicator embedding rows.
      /*.image_indicator_count=*/2,
  };
}

static id4_ideogram4_dit_program_options_t MakeProgramOptions(
    id4_pipeline_program_shape_t latent_shape) {
  id4_ideogram4_dit_program_options_t options = {
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Static model dimensions.
      /*.model=*/MakeModelConfig(),
      // Dynamic request dimensions.
      /*.request=*/
      {
          // Latent tensor shape.
          /*.latent_shape=*/latent_shape,
          // Conditioning path.
          /*.conditioning_mode=*/
          ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED,
          // Number of imported Qwen condition token positions.
          /*.text_token_count=*/0,
      },
      // Activation storage format.
      /*.activation_format=*/
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL,
      // Diagnostic tap names requested by the stage plan.
      /*.diagnostic_tap_names=*/iree_string_view_list_empty(),
  };
  return options;
}

static id4_pipeline_program_t* CreateForwardProgram(
    const id4_ideogram4_dit_program_options_t* options) {
  ProgramBuilderScope builder_scope;
  IREE_CHECK_OK(id4_ideogram4_dit_program_author_forward(
      options, builder_scope.builder()));
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

static bool ProgramHasTensorWithShape(
    const id4_pipeline_program_t* program, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t expected_shape) {
  for (iree_host_size_t i = 0; i < id4_pipeline_program_tensor_count(program);
       ++i) {
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program, i);
    if (!tensor) continue;
    if (tensor->dtype != dtype) continue;
    if (!ShapeEquals(tensor->shape, expected_shape)) continue;
    return true;
  }
  return false;
}

static bool ProgramTapsTensorWithShape(
    const id4_pipeline_program_t* program, id4_pipeline_program_dtype_t dtype,
    id4_pipeline_program_shape_t expected_shape) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation || operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_TAP) {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program,
                                       operation->payload.tap.tensor.ordinal);
    if (!tensor) continue;
    if (tensor->dtype != dtype) continue;
    if (!ShapeEquals(tensor->shape, expected_shape)) continue;
    return true;
  }
  return false;
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

TEST(Ideogram4DitProgram, AuthorsForwardPreludeSliceContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t head_size =
      options.model.hidden_size / options.model.attention_head_count;

  EXPECT_TRUE(ProgramHasTensorWithShape(program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
                                        latent_shape));
  EXPECT_TRUE(
      ProgramHasTensorWithShape(program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
                                id4_pipeline_program_make_shape_rank1(1)));
  EXPECT_TRUE(
      ProgramHasTensorWithShape(program, ID4_PIPELINE_PROGRAM_DTYPE_I32,
                                id4_pipeline_program_make_shape_rank2(2, 1)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank4(2, 2, head_size / 2, 2)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(
          options.model.hidden_size, options.model.input_channel_count)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options.model.image_indicator_count,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options.model.adaln_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramTapsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size, 2)));
  EXPECT_TRUE(ProgramTapsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options.model.adaln_size)));
  EXPECT_TRUE(ProgramExportsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32, latent_shape));
  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsConditionedPreludeSliceContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  options.request.text_token_count = 3;
  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t head_size =
      options.model.hidden_size / options.model.attention_head_count;

  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(options.model.llm_feature_count,
                                            options.request.text_token_count)));
  EXPECT_TRUE(
      ProgramHasTensorWithShape(program, ID4_PIPELINE_PROGRAM_DTYPE_I32,
                                id4_pipeline_program_make_shape_rank2(5, 1)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank4(2, 2, head_size / 2, 5)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank1(options.model.llm_feature_count)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.llm_feature_count)));
  EXPECT_TRUE(ProgramTapsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size, 5)));
  EXPECT_TRUE(ProgramTapsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options.model.adaln_size)));
  EXPECT_TRUE(ProgramExportsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32, latent_shape));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, RejectsInvalidLatentRank) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank1(2));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsInvalidActivationFormat) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.activation_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_INVALID;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsLatentChannelMismatch) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 8, 1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsUnsupportedImageIndicatorCount) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.model.image_indicator_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsInvalidAdalnSize) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.model.adaln_size = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsInvalidAttentionHeadCount) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.model.attention_head_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsUnevenAttentionHeadSize) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.model.attention_head_count = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsConditionedRequestWithoutTextTokens) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  options.request.text_token_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsUnconditionedRequestWithTextTokens) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.request.text_token_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

}  // namespace
