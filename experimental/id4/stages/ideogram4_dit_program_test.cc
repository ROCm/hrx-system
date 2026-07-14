// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_program.h"

#include <string>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_host_size_t kProgramBuilderBlockSize = 16 * 1024;
static constexpr uint32_t kSmallLatentBf16TokenCapacity = 128;

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

TEST(Ideogram4DitProgramTest, CalculatesImageTokenCount) {
  uint32_t token_count = 0;
  IREE_ASSERT_OK(id4_ideogram4_dit_program_image_token_count(
      MakeModelConfig(), id4_pipeline_program_make_shape_rank4(8, 4, 4, 1),
      &token_count));
  EXPECT_EQ(token_count, 32u);
}

TEST(Ideogram4DitProgramTest, RejectsInvalidImageTokenCountInputs) {
  uint32_t token_count = 99;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_dit_program_image_token_count(
          MakeModelConfig(), id4_pipeline_program_make_shape_rank4(8, 4, 5, 1),
          &token_count));
  EXPECT_EQ(token_count, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_dit_program_image_token_count(
          MakeModelConfig(), id4_pipeline_program_make_shape_rank4(8, 4, 4, 1),
          nullptr));
}

TEST(Ideogram4DitProgramTest, CalculatesBf16TokenCapacity) {
  uint32_t token_capacity = 0;
  IREE_ASSERT_OK(id4_ideogram4_dit_program_calculate_bf16_token_capacity(
      1, &token_capacity));
  EXPECT_EQ(token_capacity, 128u);
  IREE_ASSERT_OK(id4_ideogram4_dit_program_calculate_bf16_token_capacity(
      128, &token_capacity));
  EXPECT_EQ(token_capacity, 128u);
  IREE_ASSERT_OK(id4_ideogram4_dit_program_calculate_bf16_token_capacity(
      129, &token_capacity));
  EXPECT_EQ(token_capacity, 256u);
}

TEST(Ideogram4DitProgramTest, RejectsInvalidBf16TokenCapacityInputs) {
  uint32_t token_capacity = 99;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_calculate_bf16_token_capacity(
                            0, &token_capacity));
  EXPECT_EQ(token_capacity, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_dit_program_calculate_bf16_token_capacity(1, nullptr));
}

static id4_ideogram4_dit_program_options_t MakeProgramOptions(
    id4_pipeline_program_shape_t latent_shape) {
  id4_ideogram4_dit_program_options_t options = {
      // Size of this structure for versioning.
      /*.structure_size=*/sizeof(options),
      // Extension structure chain.
      /*.next=*/nullptr,
      // Parameter source policy.
      /*.parameter_sources=*/
      {
          // Default provider source scope.
          /*.default_scope=*/IREE_SV("model"),
          // Exact source rule count.
          /*.rule_count=*/0,
          // Exact source rules.
          /*.rules=*/nullptr,
      },
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
      // Execution storage strategy selected for linear weights.
      /*.weight_execution_format=*/
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT,
      // Attention implementation.
      /*.attention_implementation=*/
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
      // Feed-forward implementation.
      /*.feed_forward_implementation=*/
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT,
      // Diagnostic tap names requested by the stage plan.
      /*.diagnostic_tap_names=*/iree_string_view_list_empty(),
      // Static conditioned-DiT LoRA topology.
      /*.lora_topology=*/{},
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

static bool ProgramHasBoundedAttentionScratch(
    const id4_pipeline_program_t* program, id4_pipeline_program_dtype_t dtype,
    uint32_t attention_head_count, uint32_t padded_token_count) {
  for (iree_host_size_t i = 0; i < id4_pipeline_program_tensor_count(program);
       ++i) {
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program, i);
    if (!tensor) continue;
    if (tensor->dtype != dtype) continue;
    if (tensor->shape.rank != 3) continue;
    if (tensor->shape.dims[0] != attention_head_count) continue;
    if (tensor->shape.dims[2] != padded_token_count) continue;
    if (tensor->shape.dims[1] >= padded_token_count) continue;
    return true;
  }
  return false;
}

static bool ProgramHasTensor(const id4_pipeline_program_t* program,
                             iree_string_view_t name,
                             id4_pipeline_program_dtype_t dtype,
                             id4_pipeline_program_shape_t expected_shape) {
  for (iree_host_size_t i = 0; i < id4_pipeline_program_tensor_count(program);
       ++i) {
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(program, i);
    if (!tensor) continue;
    if (!iree_string_view_equal(tensor->name, name)) continue;
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

static bool ProgramTapsTensor(const id4_pipeline_program_t* program,
                              iree_string_view_t name,
                              id4_pipeline_program_dtype_t dtype,
                              id4_pipeline_program_shape_t expected_shape) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation || operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_TAP) {
      continue;
    }
    if (!iree_string_view_equal(operation->payload.tap.name, name)) continue;
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

static bool ProgramHasParameter(const id4_pipeline_program_t* program,
                                iree_string_view_t key,
                                iree_string_view_t source_scope,
                                id4_pipeline_program_dtype_t dtype,
                                id4_pipeline_program_shape_t expected_shape) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) {
      continue;
    }
    if (operation->payload.parameter.source_count == 0 ||
        !iree_string_view_equal(
            operation->payload.parameter.sources[0].source_scope,
            source_scope)) {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(
            program, operation->payload.parameter.tensor.ordinal);
    if (!tensor) continue;
    if (!iree_string_view_equal(tensor->name, key)) continue;
    if (tensor->dtype != dtype) continue;
    if (!ShapeEquals(tensor->shape, expected_shape)) continue;
    return true;
  }
  return false;
}

static const id4_pipeline_program_parameter_op_t* FindProgramParameter(
    const id4_pipeline_program_t* program, iree_string_view_t key) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(
            program, operation->payload.parameter.tensor.ordinal);
    if (tensor && iree_string_view_equal(tensor->name, key)) {
      return &operation->payload.parameter;
    }
  }
  return nullptr;
}

static const id4_pipeline_program_parameter_op_t*
FindProgramParameterByFirstSourceKey(const id4_pipeline_program_t* program,
                                     iree_string_view_t source_key) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) {
      continue;
    }
    const id4_pipeline_program_parameter_op_t* parameter =
        &operation->payload.parameter;
    if (parameter->source_count == 0) continue;
    if (iree_string_view_equal(parameter->sources[0].key, source_key)) {
      return parameter;
    }
  }
  return nullptr;
}

static bool ProgramHasFp8ScaledBf16ExecutionParameter(
    const id4_pipeline_program_t* program, iree_string_view_t key,
    iree_string_view_t scale_key, iree_string_view_t source_scope,
    id4_pipeline_program_shape_t weight_shape,
    id4_pipeline_program_shape_t scale_shape) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(
            program, operation->payload.parameter.tensor.ordinal);
    if (!tensor) continue;
    if (!iree_string_view_equal(tensor->name, key)) continue;
    if (tensor->dtype != ID4_PIPELINE_PROGRAM_DTYPE_BF16) continue;
    if (!ShapeEquals(tensor->shape, weight_shape)) continue;
    if (operation->payload.parameter.encoding !=
        ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16) {
      continue;
    }
    if (operation->payload.parameter.source_count != 2) continue;
    const id4_pipeline_program_parameter_source_t* weight_source =
        &operation->payload.parameter.sources[0];
    const id4_pipeline_program_parameter_source_t* scale_source =
        &operation->payload.parameter.sources[1];
    if (!iree_string_view_equal(weight_source->source_scope, source_scope)) {
      continue;
    }
    if (!iree_string_view_equal(weight_source->key, key)) continue;
    if (weight_source->dtype != ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3) continue;
    if (!ShapeEquals(weight_source->shape, weight_shape)) continue;
    if (!iree_string_view_equal(scale_source->source_scope, source_scope)) {
      continue;
    }
    if (!iree_string_view_equal(scale_source->key, scale_key)) continue;
    if (scale_source->dtype != ID4_PIPELINE_PROGRAM_DTYPE_F32) continue;
    if (!ShapeEquals(scale_source->shape, scale_shape)) continue;
    return true;
  }
  return false;
}

static bool ProgramHasFp8ExecutionParameterEncoding(
    const id4_pipeline_program_t* program, iree_string_view_t key,
    iree_string_view_t source_scope,
    id4_pipeline_program_shape_t expected_shape,
    id4_pipeline_program_parameter_encoding_t expected_encoding) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER) {
      continue;
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(
            program, operation->payload.parameter.tensor.ordinal);
    if (!tensor) continue;
    if (!iree_string_view_equal(tensor->name, key)) continue;
    if (tensor->dtype != ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3) continue;
    if (!ShapeEquals(tensor->shape, expected_shape)) continue;
    if (operation->payload.parameter.encoding != expected_encoding) continue;
    if (operation->payload.parameter.source_count != 1) continue;
    const id4_pipeline_program_parameter_source_t* source =
        &operation->payload.parameter.sources[0];
    if (!iree_string_view_equal(source->source_scope, source_scope)) continue;
    if (!iree_string_view_equal(source->key, key)) continue;
    if (source->dtype != ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3) continue;
    if (!ShapeEquals(source->shape, expected_shape)) continue;
    return true;
  }
  return false;
}

static bool ProgramHasFp8CompactExecutionParameter(
    const id4_pipeline_program_t* program, iree_string_view_t key,
    iree_string_view_t source_scope,
    id4_pipeline_program_shape_t expected_shape) {
  return ProgramHasFp8ExecutionParameterEncoding(
      program, key, source_scope, expected_shape,
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE);
}

static bool ProgramHasLoomDispatch(const id4_pipeline_program_t* program,
                                   iree_string_view_t module_path,
                                   iree_string_view_t function_name) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(program, i);
    if (!operation ||
        operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      continue;
    }
    const id4_pipeline_program_dispatch_loom_op_t* dispatch =
        &operation->payload.dispatch_loom;
    if (!iree_string_view_equal(dispatch->kernel.module_path, module_path)) {
      continue;
    }
    if (!iree_string_view_equal(dispatch->kernel.function_name,
                                function_name)) {
      continue;
    }
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

TEST(Ideogram4DitProgram, TapsBf16ConditionPreludeWithoutChangingStorage) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  options.request.text_token_count = 3;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
  };
  options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("ideogram4.cond.prelude.llm_cond_norm_packed"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(16,
                                            options.model.llm_feature_count)));
  EXPECT_TRUE(ProgramTapsTensor(
      program, IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(options.model.llm_feature_count,
                                            options.request.text_token_count)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsScaledFp8ProjectionParameterContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  const id4_ideogram4_dit_parameter_source_rule_t rules[] = {
      {
          // Logical parameter key.
          /*.key=*/IREE_SV("layers.0.attention.qkv.weight"),
          // Provider source scope.
          /*.source_scope=*/IREE_SV("fp8"),
          // Physical storage format.
          /*.storage=*/ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
      },
      {
          // Logical parameter key.
          /*.key=*/IREE_SV("layers.0.attention.o.weight"),
          // Provider source scope.
          /*.source_scope=*/IREE_SV("fp8"),
          // Physical storage format.
          /*.storage=*/ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
      },
      {
          // Logical parameter key.
          /*.key=*/IREE_SV("layers.0.feed_forward.w1.weight"),
          // Provider source scope.
          /*.source_scope=*/IREE_SV("fp8"),
          // Physical storage format.
          /*.storage=*/ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
      },
      {
          // Logical parameter key.
          /*.key=*/IREE_SV("layers.0.feed_forward.w3.weight"),
          // Provider source scope.
          /*.source_scope=*/IREE_SV("fp8"),
          // Physical storage format.
          /*.storage=*/ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
      },
      {
          // Logical parameter key.
          /*.key=*/IREE_SV("layers.0.feed_forward.w2.weight"),
          // Provider source scope.
          /*.source_scope=*/IREE_SV("fp8"),
          // Physical storage format.
          /*.storage=*/ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED,
      },
  };
  options.parameter_sources.rule_count = IREE_ARRAYSIZE(rules);
  options.parameter_sources.rules = rules;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t qkv_size = options.model.hidden_size * 3;
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.attention.qkv.weight"),
      IREE_SV("layers.0.attention.qkv.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(qkv_size,
                                            options.model.hidden_size),
      id4_pipeline_program_make_shape_rank1(qkv_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.attention.o.weight"),
      IREE_SV("layers.0.attention.o.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.hidden_size),
      id4_pipeline_program_make_shape_rank1(options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w1.weight"),
      IREE_SV("layers.0.feed_forward.w1.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size),
      id4_pipeline_program_make_shape_rank1(options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w3.weight"),
      IREE_SV("layers.0.feed_forward.w3.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size),
      id4_pipeline_program_make_shape_rank1(options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w2.weight"),
      IREE_SV("layers.0.feed_forward.w2.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.intermediate_size),
      id4_pipeline_program_make_shape_rank1(options.model.hidden_size)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsCompactFp8ProjectionParameterContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS;
  options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
  options.model.hidden_size = 128;
  options.model.intermediate_size = 128;
  options.model.attention_head_count = 2;
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3, options.model,
      IREE_SV("fp8"), iree_allocator_system(), &rules));
  options.parameter_sources.rule_count = rules.count;
  options.parameter_sources.rules = rules.values;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t qkv_size = options.model.hidden_size * 3;
  EXPECT_TRUE(ProgramHasFp8CompactExecutionParameter(
      program, IREE_SV("layers.0.attention.qkv.weight"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(qkv_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasParameter(
      program, IREE_SV("layers.0.attention.qkv.weight_scale"), IREE_SV("fp8"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(qkv_size)));
  EXPECT_TRUE(ProgramHasFp8CompactExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w1.weight"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasParameter(
      program, IREE_SV("layers.0.feed_forward.w1.weight_scale"), IREE_SV("fp8"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasFp8CompactExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w2.weight"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasParameter(
      program, IREE_SV("layers.0.feed_forward.w2.weight_scale"), IREE_SV("fp8"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(options.model.hidden_size)));

  id4_pipeline_program_release(program);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitProgram, IsolatesLoraPatchableWeightsAndScales) {
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS;
  options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
  options.model.hidden_size = 128;
  options.model.intermediate_size = 128;
  options.model.attention_head_count = 2;
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3, options.model,
      IREE_SV("fp8"), iree_allocator_system(), &rules));
  options.parameter_sources.rule_count = rules.count;
  options.parameter_sources.rules = rules.values;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const iree_string_view_t patchable_parameters[] = {
      IREE_SV("layers.0.adaln_modulation.weight"),
      IREE_SV("layers.0.adaln_modulation.weight_scale"),
      IREE_SV("layers.0.attention.qkv.weight"),
      IREE_SV("layers.0.attention.qkv.weight_scale"),
      IREE_SV("layers.0.attention.o.weight"),
      IREE_SV("layers.0.attention.o.weight_scale"),
      IREE_SV("layers.0.feed_forward.w1.weight"),
      IREE_SV("layers.0.feed_forward.w1.weight_scale"),
      IREE_SV("layers.0.feed_forward.w3.weight"),
      IREE_SV("layers.0.feed_forward.w3.weight_scale"),
      IREE_SV("layers.0.feed_forward.w2.weight"),
      IREE_SV("layers.0.feed_forward.w2.weight_scale"),
  };
  for (iree_string_view_t key : patchable_parameters) {
    const id4_pipeline_program_parameter_op_t* parameter =
        FindProgramParameter(program, key);
    ASSERT_NE(parameter, nullptr) << std::string(key.data, key.size);
    EXPECT_TRUE(
        iree_string_view_equal(parameter->domain, IREE_SV("lora_patchable")))
        << std::string(key.data, key.size);
  }

  const iree_string_view_t shared_parameters[] = {
      IREE_SV("input_proj.weight"),
      IREE_SV("layers.0.adaln_modulation.bias"),
      IREE_SV("layers.0.attention.norm_q.weight"),
  };
  for (iree_string_view_t key : shared_parameters) {
    const id4_pipeline_program_parameter_op_t* parameter =
        FindProgramParameter(program, key);
    ASSERT_NE(parameter, nullptr) << std::string(key.data, key.size);
    EXPECT_TRUE(iree_string_view_is_empty(parameter->domain))
        << std::string(key.data, key.size);
  }

  id4_pipeline_program_release(program);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitProgram, AuthorsCompactFp8FusedFeedForwardCompactParameters) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS;
  options.model.hidden_size = 128;
  options.model.intermediate_size = 128;
  options.model.attention_head_count = 2;
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3, options.model,
      IREE_SV("fp8"), iree_allocator_system(), &rules));
  options.parameter_sources.rule_count = rules.count;
  options.parameter_sources.rules = rules.values;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  EXPECT_TRUE(ProgramHasFp8CompactExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w1.weight"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasFp8CompactExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w3.weight"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasFp8CompactExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w2.weight"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.intermediate_size)));

  id4_pipeline_program_release(program);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitProgram,
     AuthorsCompactFp8WithFeedForwardBf16ExecutionParameterContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS_FEED_FORWARD_BF16_RESIDENT;
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3, options.model,
      IREE_SV("fp8"), iree_allocator_system(), &rules));
  options.parameter_sources.rule_count = rules.count;
  options.parameter_sources.rules = rules.values;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t qkv_size = options.model.hidden_size * 3;
  EXPECT_TRUE(
      ProgramHasParameter(program, IREE_SV("layers.0.attention.qkv.weight"),
                          IREE_SV("fp8"), ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
                          id4_pipeline_program_make_shape_rank2(
                              qkv_size, options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasParameter(
      program, IREE_SV("layers.0.attention.qkv.weight_scale"), IREE_SV("fp8"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(qkv_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w1.weight"),
      IREE_SV("layers.0.feed_forward.w1.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size),
      id4_pipeline_program_make_shape_rank1(options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w3.weight"),
      IREE_SV("layers.0.feed_forward.w3.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.intermediate_size,
                                            options.model.hidden_size),
      id4_pipeline_program_make_shape_rank1(options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasFp8ScaledBf16ExecutionParameter(
      program, IREE_SV("layers.0.feed_forward.w2.weight"),
      IREE_SV("layers.0.feed_forward.w2.weight_scale"), IREE_SV("fp8"),
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.intermediate_size),
      id4_pipeline_program_make_shape_rank1(options.model.hidden_size)));

  id4_pipeline_program_release(program);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitProgram, AuthorsStreamingCompactRhsParameterContract) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_STREAMING_COMPACT_RHS;
  options.model.hidden_size = 128;
  options.model.intermediate_size = 128;
  options.model.attention_head_count = 2;
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3, options.model,
      IREE_SV("fp8"), iree_allocator_system(), &rules));
  options.parameter_sources.rule_count = rules.count;
  options.parameter_sources.rules = rules.values;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t qkv_size = options.model.hidden_size * 3;
  EXPECT_TRUE(
      ProgramHasParameter(program, IREE_SV("layers.0.attention.qkv.weight"),
                          IREE_SV("fp8"), ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
                          id4_pipeline_program_make_shape_rank2(
                              qkv_size, options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasParameter(
      program, IREE_SV("layers.0.attention.qkv.weight_scale"), IREE_SV("fp8"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank1(qkv_size)));
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("layers.0.attention.qkv.weight.compact_rhs_bf16"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(qkv_size,
                                            options.model.hidden_size)));
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("layers.0.feed_forward.w2.weight.compact_rhs_bf16"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            options.model.intermediate_size)));
  EXPECT_TRUE(ProgramHasLoomDispatch(
      program, IREE_SV("parameter/fp8_e4m3_scaled_to_bf16_linear_rhs_tile"),
      IREE_SV("id4_parameter_fp8_e4m3_scaled_to_bf16_linear_rhs_tile")));

  id4_pipeline_program_release(program);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitProgram, AuthorsMaterializedWmmaAttentionIntermediates) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank3(options.model.attention_head_count,
                                            kSmallLatentBf16TokenCapacity,
                                            kSmallLatentBf16TokenCapacity)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank3(options.model.attention_head_count,
                                            kSmallLatentBf16TokenCapacity,
                                            kSmallLatentBf16TokenCapacity)));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(kSmallLatentBf16TokenCapacity,
                                            options.model.hidden_size)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsStreamingBf16AttentionContextAsLinearInput) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.attention.context"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(kSmallLatentBf16TokenCapacity,
                                            options.model.hidden_size)));
  EXPECT_FALSE(ProgramHasTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.attention.context"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size,
                                            kSmallLatentBf16TokenCapacity)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsBlockedWmmaAttentionScratchBlocks) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(16, 64, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t padded_token_count = 1024;
  EXPECT_TRUE(ProgramHasBoundedAttentionScratch(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      options.model.attention_head_count, padded_token_count));
  EXPECT_TRUE(ProgramHasBoundedAttentionScratch(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      options.model.attention_head_count, padded_token_count));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(padded_token_count,
                                            options.model.hidden_size)));
  EXPECT_FALSE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank3(options.model.attention_head_count,
                                            padded_token_count,
                                            padded_token_count)));
  EXPECT_FALSE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank3(options.model.attention_head_count,
                                            padded_token_count,
                                            padded_token_count)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsOnlineWmmaAttentionWithoutMatrixScratch) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(16, 64, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.model.hidden_size = 512;
  options.model.attention_head_count = 2;
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const uint32_t padded_token_count = 1024;
  EXPECT_FALSE(ProgramHasBoundedAttentionScratch(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      options.model.attention_head_count, padded_token_count));
  EXPECT_FALSE(ProgramHasBoundedAttentionScratch(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      options.model.attention_head_count, padded_token_count));
  EXPECT_TRUE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(padded_token_count,
                                            options.model.hidden_size)));
  EXPECT_FALSE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_F32,
      id4_pipeline_program_make_shape_rank3(options.model.attention_head_count,
                                            padded_token_count,
                                            padded_token_count)));
  EXPECT_FALSE(ProgramHasTensorWithShape(
      program, ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank3(options.model.attention_head_count,
                                            padded_token_count,
                                            padded_token_count)));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, AuthorsPyTorchParityFeedForwardIntermediates) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const id4_pipeline_program_shape_t projection_shape =
      id4_pipeline_program_make_shape_rank2(kSmallLatentBf16TokenCapacity,
                                            options.model.intermediate_size);
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.ffn.w1_projection.output"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16, projection_shape));
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.ffn.w3_projection.output"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16, projection_shape));
  EXPECT_TRUE(
      ProgramHasTensor(program, IREE_SV("ideogram4.uncond.layers.0.ffn.hidden"),
                       ID4_PIPELINE_PROGRAM_DTYPE_BF16, projection_shape));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, TapsBf16BlockBoundariesWithoutChangingStorage) {
  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(latent_shape);
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.uncond.layers.0.attention_input"),
      IREE_SV("ideogram4.uncond.layers.0.attention.output"),
      IREE_SV("ideogram4.uncond.layers.0.ffn.output"),
  };
  options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  const id4_pipeline_program_shape_t internal_shape =
      id4_pipeline_program_make_shape_rank2(kSmallLatentBf16TokenCapacity,
                                            options.model.hidden_size);
  const id4_pipeline_program_shape_t tap_shape =
      id4_pipeline_program_make_shape_rank2(options.model.hidden_size, 2);
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.attention_input"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16, internal_shape));
  EXPECT_TRUE(ProgramHasTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.attention.output"),
      ID4_PIPELINE_PROGRAM_DTYPE_BF16, internal_shape));
  EXPECT_TRUE(
      ProgramHasTensor(program, IREE_SV("ideogram4.uncond.layers.0.ffn.output"),
                       ID4_PIPELINE_PROGRAM_DTYPE_BF16, internal_shape));
  EXPECT_TRUE(ProgramTapsTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.attention_input"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32, tap_shape));
  EXPECT_TRUE(ProgramTapsTensor(
      program, IREE_SV("ideogram4.uncond.layers.0.attention.output"),
      ID4_PIPELINE_PROGRAM_DTYPE_F32, tap_shape));
  EXPECT_TRUE(ProgramTapsTensor(program,
                                IREE_SV("ideogram4.uncond.layers.0.ffn.output"),
                                ID4_PIPELINE_PROGRAM_DTYPE_F32, tap_shape));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, RejectsMaterializedWmmaAttentionWithCanonicalF32) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsBlockedWmmaAttentionWithCanonicalF32) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsOnlineWmmaAttentionWithCanonicalF32) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(16, 64, 4, 1));
  options.model.hidden_size = 512;
  options.model.attention_head_count = 2;
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsOnlineWmmaAttentionWithUnsupportedHeadSize) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsPyTorchParityFeedForwardWithCanonicalF32) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsProjectionTapWithoutPyTorchParityFeedForward) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.uncond.layers.0.ffn.w1_projection.output"),
  };
  options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, ComposesLoraParameterSourcesForConditionedForward) {
  const id4_ideogram4_dit_lora_segment_t segments[] = {
      {
          /*.source_scope=*/IREE_SV("portrait"),
          /*.adapter_ordinal=*/0,
          /*.rank_offset=*/0,
          /*.rank=*/2,
          /*.down_parameter_key=*/IREE_SV("portrait.qkv.lora_A.weight"),
          /*.up_parameter_key=*/IREE_SV("portrait.qkv.lora_B.weight"),
      },
      {
          /*.source_scope=*/IREE_SV("lighting"),
          /*.adapter_ordinal=*/1,
          /*.rank_offset=*/2,
          /*.rank=*/3,
          /*.down_parameter_key=*/IREE_SV("lighting.qkv.lora_A.weight"),
          /*.up_parameter_key=*/IREE_SV("lighting.qkv.lora_B.weight"),
      },
  };
  const id4_ideogram4_dit_lora_target_t targets[] = {
      {
          /*.base_parameter_key=*/IREE_SV("layers.0.attention.qkv.weight"),
          /*.input_size=*/32,
          /*.output_size=*/96,
          /*.total_rank=*/5,
          /*.segment_count=*/IREE_ARRAYSIZE(segments),
          /*.segments=*/segments,
      },
  };
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  options.request.text_token_count = 2;
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.lora_topology = {
      /*.adapter_count=*/2,
      /*.target_count=*/IREE_ARRAYSIZE(targets),
      /*.targets=*/targets,
  };

  id4_pipeline_program_t* program = CreateForwardProgram(&options);
  ASSERT_TRUE(ProgramHasTensor(program, IREE_SV("lora.strengths"),
                               ID4_PIPELINE_PROGRAM_DTYPE_F32,
                               id4_pipeline_program_make_shape_rank1(2)));

  const id4_pipeline_program_parameter_op_t* down =
      FindProgramParameterByFirstSourceKey(
          program, IREE_SV("portrait.qkv.lora_A.weight"));
  ASSERT_NE(down, nullptr);
  EXPECT_EQ(down->encoding, ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT);
  ASSERT_EQ(down->source_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(down->domain, IREE_SV("lora_dynamic")));
  EXPECT_TRUE(iree_string_view_equal(down->sources[0].source_scope,
                                     IREE_SV("portrait")));
  EXPECT_TRUE(iree_string_view_equal(down->sources[1].source_scope,
                                     IREE_SV("lighting")));
  EXPECT_TRUE(iree_string_view_equal(down->sources[1].key,
                                     IREE_SV("lighting.qkv.lora_A.weight")));
  ASSERT_EQ(down->source_span_count, 2u);
  EXPECT_EQ(down->source_spans[0].source_index, 0u);
  EXPECT_EQ(down->source_spans[0].target_offset, 0u);
  EXPECT_EQ(down->source_spans[0].length, 2u * 32u * sizeof(uint16_t));
  EXPECT_EQ(down->source_spans[1].source_index, 1u);
  EXPECT_EQ(down->source_spans[1].target_offset, 2u * 32u * sizeof(uint16_t));
  EXPECT_EQ(down->source_spans[1].length, 3u * 32u * sizeof(uint16_t));

  id4_pipeline_program_release(program);
}

TEST(Ideogram4DitProgram, RejectsLoraTargetOutsideAuthoredModel) {
  ProgramBuilderScope builder_scope;
  const id4_ideogram4_dit_lora_segment_t segment = {
      /*.source_scope=*/IREE_SV("adapter"),
      /*.adapter_ordinal=*/0,
      /*.rank_offset=*/0,
      /*.rank=*/2,
      /*.down_parameter_key=*/IREE_SV("missing.lora_A.weight"),
      /*.up_parameter_key=*/IREE_SV("missing.lora_B.weight"),
  };
  const id4_ideogram4_dit_lora_target_t target = {
      /*.base_parameter_key=*/IREE_SV("layers.99.attention.qkv.weight"),
      /*.input_size=*/32,
      /*.output_size=*/96,
      /*.total_rank=*/2,
      /*.segment_count=*/1,
      /*.segments=*/&segment,
  };
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  options.request.text_token_count = 2;
  options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  options.lora_topology = {
      /*.adapter_count=*/1,
      /*.target_count=*/1,
      /*.targets=*/&target,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsLoraOnUnconditionedForward) {
  ProgramBuilderScope builder_scope;
  const id4_ideogram4_dit_lora_segment_t segment = {
      /*.source_scope=*/IREE_SV("adapter"),
      /*.adapter_ordinal=*/0,
      /*.rank_offset=*/0,
      /*.rank=*/2,
      /*.down_parameter_key=*/IREE_SV("qkv.lora_A.weight"),
      /*.up_parameter_key=*/IREE_SV("qkv.lora_B.weight"),
  };
  const id4_ideogram4_dit_lora_target_t target = {
      /*.base_parameter_key=*/IREE_SV("layers.0.attention.qkv.weight"),
      /*.input_size=*/32,
      /*.output_size=*/96,
      /*.total_rank=*/2,
      /*.segment_count=*/1,
      /*.segments=*/&segment,
  };
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.lora_topology = {
      /*.adapter_count=*/1,
      /*.target_count=*/1,
      /*.targets=*/&target,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
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

TEST(Ideogram4DitProgram, RejectsInvalidWeightExecutionFormat) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_INVALID;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsInvalidAttentionImplementation) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_INVALID;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_program_author_forward(
                            &options, builder_scope.builder()));
}

TEST(Ideogram4DitProgram, RejectsInvalidFeedForwardImplementation) {
  ProgramBuilderScope builder_scope;
  id4_ideogram4_dit_program_options_t options =
      MakeProgramOptions(id4_pipeline_program_make_shape_rank4(1, 2, 4, 1));
  options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_INVALID;
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
