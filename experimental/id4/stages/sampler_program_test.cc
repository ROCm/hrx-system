// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/sampler_program.h"

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
        /*.program_name=*/IREE_SV("test.sampler"),
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

static id4_sampler_program_options_t MakeProgramOptions(
    id4_pipeline_program_shape_t latent_shape) {
  id4_sampler_program_options_t options = {
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Dynamic request dimensions.
      /*.request=*/
      {
          // Latent tensor shape.
          /*.latent_shape=*/latent_shape,
      },
  };
  return options;
}

static id4_pipeline_program_t* CreateSamplerProgram(
    const id4_sampler_program_options_t* options) {
  ProgramBuilderScope builder_scope;
  IREE_CHECK_OK(id4_sampler_program_author_denoise_step(
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

TEST(SamplerProgram, AuthorsDenoiseStepBoundaryContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 128, 1);
  id4_sampler_program_options_t options = MakeProgramOptions(latent_shape);
  id4_pipeline_program_t* program = CreateSamplerProgram(&options);

  EXPECT_TRUE(ProgramExportsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32, latent_shape));
  EXPECT_TRUE(ProgramTapsTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32, latent_shape));

  id4_pipeline_program_release(program);
}

TEST(SamplerProgram, RejectsScalarLatentShape) {
  ProgramBuilderScope builder_scope;
  id4_sampler_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank0());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_sampler_program_author_denoise_step(
                            &options, builder_scope.builder()));
}

TEST(SamplerProgram, RejectsElementCountAboveKernelRange) {
  ProgramBuilderScope builder_scope;
  id4_sampler_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank1(
          ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT + 1ull));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        id4_sampler_program_author_denoise_step(
                            &options, builder_scope.builder()));
}

}  // namespace
