// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_decode_program.h"

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
        /*.program_name=*/IREE_SV("test.ideogram4.decode"),
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

static id4_vae_model_config_t MakeSmallVaeConfig() {
  return id4_vae_model_config_t{
      // Latent-to-image scale factor along the width axis.
      /*.scale_x=*/2,
      // Latent-to-image scale factor along the height axis.
      /*.scale_y=*/2,
      // Latent-to-media scale factor along the temporal axis.
      /*.scale_t=*/1,
      // Channel count in latent tensors.
      /*.latent_channel_count=*/4,
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

static id4_ideogram4_decode_model_config_t MakeSmallModelConfig() {
  return id4_ideogram4_decode_model_config_t{
      // Reusable VAE decode model configuration.
      /*.vae=*/MakeSmallVaeConfig(),
  };
}

static id4_ideogram4_decode_request_config_t MakeRequest(
    id4_pipeline_program_shape_t diffusion_shape) {
  id4_ideogram4_decode_request_config_t request = {
      // Diffusion latent shape.
      /*.diffusion_latent_shape=*/diffusion_shape,
      // VAE tiling policy.
      /*.vae_tiling=*/
      {
          // Tiling policy selected for this request.
          /*.mode=*/ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE,
          // Requested latent tile width.
          /*.tile_size_x=*/2,
          // Requested latent tile height.
          /*.tile_size_y=*/2,
          // Relative latent width factor.
          /*.relative_size_x=*/0.0f,
          // Relative latent height factor.
          /*.relative_size_y=*/0.0f,
          // Requested overlap.
          /*.overlap=*/0.0f,
          // Memory budget.
          /*.memory_budget=*/0,
      },
  };
  return request;
}

static id4_ideogram4_decode_program_options_t MakeProgramOptions(
    id4_ideogram4_decode_model_config_t model,
    id4_ideogram4_decode_request_config_t request) {
  id4_ideogram4_decode_program_options_t options = {
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Static decode model contract.
      /*.model=*/model,
      // Dynamic decode request.
      /*.request=*/request,
      // Activation storage format for internal VAE intermediates.
      /*.vae_activation_format=*/ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL,
  };
  return options;
}

static id4_pipeline_program_t* CreateDecodeProgram(
    const id4_ideogram4_decode_program_options_t* options) {
  ProgramBuilderScope builder_scope;
  IREE_CHECK_OK(id4_ideogram4_decode_program_author_decode(
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

static const id4_pipeline_program_tensor_record_t* FindBoundaryTensor(
    const id4_pipeline_program_t* program, id4_pipeline_program_op_kind_t kind,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation || operation->kind != kind) continue;
    id4_pipeline_program_tensor_t tensor =
        id4_pipeline_program_tensor_invalid();
    if (kind == ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT) {
      tensor = operation->payload.import_value.tensor;
    } else if (kind == ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT) {
      tensor = operation->payload.export_value.tensor;
    } else {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* record =
        id4_pipeline_program_tensor_at(program, tensor.ordinal);
    if (record && iree_string_view_equal(record->name, name)) return record;
  }
  return nullptr;
}

TEST(Ideogram4DecodeProgram, ValidatesDiffusionLatentShape) {
  IREE_ASSERT_OK(id4_ideogram4_decode_program_validate_diffusion_latent_shape(
      MakeSmallModelConfig(),
      id4_pipeline_program_make_shape_rank4(2, 3, 4, 1)));
}

TEST(Ideogram4DecodeProgram, AuthorsDecodeBoundaryContract) {
  id4_ideogram4_decode_model_config_t model = MakeSmallModelConfig();
  id4_ideogram4_decode_request_config_t request =
      MakeRequest(id4_pipeline_program_make_shape_rank4(2, 3, 4, 1));
  id4_ideogram4_decode_program_options_t options =
      MakeProgramOptions(model, request);
  id4_pipeline_program_t* program = CreateDecodeProgram(&options);

  const id4_pipeline_program_tensor_record_t* diffusion_latent =
      FindBoundaryTensor(program, ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT,
                         IREE_SV("media.latent.diffusion"));
  ASSERT_NE(diffusion_latent, nullptr);
  EXPECT_TRUE(
      ShapeEquals(diffusion_latent->shape, request.diffusion_latent_shape));

  const id4_pipeline_program_tensor_record_t* decoded =
      FindBoundaryTensor(program, ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT,
                         IREE_SV("media.image.decoded"));
  ASSERT_NE(decoded, nullptr);
  EXPECT_TRUE(ShapeEquals(decoded->shape,
                          id4_pipeline_program_make_shape_rank4(4, 6, 1, 1)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DecodeProgram, RejectsWrongDiffusionChannelCount) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_decode_model_config_t model = MakeSmallModelConfig();
  id4_ideogram4_decode_request_config_t request =
      MakeRequest(id4_pipeline_program_make_shape_rank4(2, 3, 8, 1));
  id4_ideogram4_decode_program_options_t options =
      MakeProgramOptions(model, request);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_decode_program_author_decode(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DecodeProgram, RejectsNonSingletonDiffusionBatchDimension) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_decode_model_config_t model = MakeSmallModelConfig();
  id4_ideogram4_decode_request_config_t request =
      MakeRequest(id4_pipeline_program_make_shape_rank4(2, 3, 4, 2));
  id4_ideogram4_decode_program_options_t options =
      MakeProgramOptions(model, request);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_decode_program_author_decode(
                            &options, builder_scope.builder()));
}

}  // namespace
