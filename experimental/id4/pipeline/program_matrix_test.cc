// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_matrix.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ProgramMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_program_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.program_name=*/IREE_SV("matrix.test"),
        /*.block_pool=*/&block_pool_,
    };
    IREE_CHECK_OK(id4_pipeline_program_builder_create(
        &options, iree_allocator_system(), &builder_));
  }

  void TearDown() override {
    id4_pipeline_program_builder_destroy(builder_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  id4_pipeline_program_tensor_t Import(iree_string_view_t name,
                                       id4_pipeline_program_dtype_t dtype,
                                       id4_pipeline_program_shape_t shape) {
    id4_pipeline_program_import_tensor_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
        /*.name=*/name,
        /*.dtype=*/dtype,
        /*.shape=*/shape,
    };
    id4_pipeline_program_tensor_t tensor =
        id4_pipeline_program_tensor_invalid();
    IREE_CHECK_OK(
        id4_pipeline_program_import_tensor(builder_, &options, &tensor));
    return tensor;
  }

  id4_pipeline_program_tensor_t Acquire(iree_string_view_t name,
                                        id4_pipeline_program_dtype_t dtype,
                                        id4_pipeline_program_shape_t shape) {
    id4_pipeline_program_acquire_tensor_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.name=*/name,
        /*.dtype=*/dtype,
        /*.shape=*/shape,
    };
    id4_pipeline_program_tensor_t tensor =
        id4_pipeline_program_tensor_invalid();
    IREE_CHECK_OK(
        id4_pipeline_program_acquire_tensor(builder_, &options, &tensor));
    return tensor;
  }

  id4_pipeline_program_matrix_options_t MakeBlockScaledOptions() {
    id4_pipeline_program_matrix_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("layer.linear"),
        /*.request=*/
        {
            /*.valid_m=*/451,
            /*.m_capacity=*/512,
            /*.n=*/1024,
            /*.k=*/4096,
            /*.input_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.input_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            /*.weight_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.weight_layout=*/
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
            /*.scale_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.scale_layout=*/
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(512, 4096));
    options.operands.weight =
        Import(IREE_SV("weight"), ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
               id4_pipeline_program_make_shape_rank2(1024, 4096));
    options.operands.scale =
        Import(IREE_SV("scale"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
               id4_pipeline_program_make_shape_rank2(8, 32));
    options.operands.output =
        Acquire(IREE_SV("output"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                id4_pipeline_program_make_shape_rank2(512, 1024));
    options.operands.addend = id4_pipeline_program_tensor_invalid();
    return options;
  }

  id4_pipeline_program_matrix_options_t MakeRowScaledOptions() {
    id4_pipeline_program_matrix_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("transformer.linear"),
        /*.request=*/
        {
            /*.valid_m=*/113,
            /*.m_capacity=*/128,
            /*.n=*/4608,
            /*.k=*/4608,
            /*.input_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.input_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            /*.weight_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.weight_layout=*/
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
            /*.scale_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.scale_layout=*/
            ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("transformer_input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(128, 4608));
    options.operands.weight = Import(
        IREE_SV("transformer_weight"), ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
        id4_pipeline_program_make_shape_rank2(4608, 4608));
    options.operands.scale =
        Import(IREE_SV("transformer_scale"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
               id4_pipeline_program_make_shape_rank1(4608));
    options.operands.output =
        Acquire(IREE_SV("transformer_output"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                id4_pipeline_program_make_shape_rank2(128, 4608));
    options.operands.addend = id4_pipeline_program_tensor_invalid();
    return options;
  }

  id4_pipeline_program_swiglu_options_t MakeSwiGLUOptions(
      id4_pipeline_program_matrix_scale_layout_t scale_layout,
      id4_pipeline_program_shape_t scale_shape) {
    id4_pipeline_program_swiglu_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("feed_forward.swiglu"),
        /*.projection=*/
        {
            /*.valid_m=*/451,
            /*.m_capacity=*/512,
            /*.n=*/12288,
            /*.k=*/4096,
            /*.input_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.input_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            /*.weight_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.weight_layout=*/
            ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TILE_16X16,
            /*.scale_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.scale_layout=*/scale_layout,
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("swiglu_input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(512, 4096));
    options.operands.gate_weight =
        Import(IREE_SV("gate_weight"), ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
               id4_pipeline_program_make_shape_rank2(12288, 4096));
    options.operands.gate_scale = Import(
        IREE_SV("gate_scale"), ID4_PIPELINE_PROGRAM_DTYPE_F32, scale_shape);
    options.operands.up_weight =
        Import(IREE_SV("up_weight"), ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
               id4_pipeline_program_make_shape_rank2(12288, 4096));
    options.operands.up_scale = Import(
        IREE_SV("up_scale"), ID4_PIPELINE_PROGRAM_DTYPE_F32, scale_shape);
    options.operands.output =
        Acquire(IREE_SV("swiglu_output"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                id4_pipeline_program_make_shape_rank2(512, 12288));
    return options;
  }

  iree_arena_block_pool_t block_pool_;
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

TEST_F(ProgramMatrixTest, AuthorsSemanticBlockScaledContraction) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));

  const id4_pipeline_program_op_t* dispatch = id4_pipeline_program_operation_at(
      program, id4_pipeline_program_operation_count(program) - 1);
  ASSERT_NE(dispatch, nullptr);
  ASSERT_EQ(dispatch->kind, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM);
  EXPECT_TRUE(iree_string_view_equal(dispatch->payload.dispatch_loom.name,
                                     options.name));
  EXPECT_FALSE(iree_string_view_is_empty(
      dispatch->payload.dispatch_loom.kernel.module_path));
  EXPECT_FALSE(iree_string_view_is_empty(
      dispatch->payload.dispatch_loom.kernel.function_name));
  EXPECT_EQ(dispatch->payload.dispatch_loom.binding_count, 4u);
  EXPECT_GE(dispatch->payload.dispatch_loom.config_binding_count, 3u);
  for (iree_host_size_t i = 0;
       i < dispatch->payload.dispatch_loom.config_binding_count; ++i) {
    const id4_pipeline_kernel_config_binding_t binding =
        dispatch->payload.dispatch_loom.config_bindings[i];
    EXPECT_EQ(iree_string_view_find(binding.key, options.name, 0),
              IREE_STRING_VIEW_NPOS);
  }

  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, AuthorsSemanticRowScaledContraction) {
  id4_pipeline_program_matrix_options_t options = MakeRowScaledOptions();
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));

  const id4_pipeline_program_op_t* dispatch = id4_pipeline_program_operation_at(
      program, id4_pipeline_program_operation_count(program) - 1);
  ASSERT_NE(dispatch, nullptr);
  ASSERT_EQ(dispatch->kind, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM);
  EXPECT_TRUE(iree_string_view_equal(dispatch->payload.dispatch_loom.name,
                                     options.name));
  EXPECT_FALSE(iree_string_view_is_empty(
      dispatch->payload.dispatch_loom.kernel.module_path));
  EXPECT_FALSE(iree_string_view_is_empty(
      dispatch->payload.dispatch_loom.kernel.function_name));
  EXPECT_EQ(dispatch->payload.dispatch_loom.binding_count, 4u);
  EXPECT_GE(dispatch->payload.dispatch_loom.config_binding_count, 3u);

  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, AuthorsSemanticResidualContraction) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.request.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
  options.operands.addend =
      Import(IREE_SV("residual"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
             id4_pipeline_program_make_shape_rank2(512, 1024));
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));

  const id4_pipeline_program_op_t* dispatch = id4_pipeline_program_operation_at(
      program, id4_pipeline_program_operation_count(program) - 1);
  ASSERT_NE(dispatch, nullptr);
  ASSERT_EQ(dispatch->kind, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM);
  EXPECT_TRUE(iree_string_view_equal(dispatch->payload.dispatch_loom.name,
                                     options.name));
  EXPECT_EQ(dispatch->payload.dispatch_loom.binding_count, 5u);

  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, AuthorsSemanticSwiGLUProjectionPair) {
  id4_pipeline_program_swiglu_options_t options = MakeSwiGLUOptions(
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
      id4_pipeline_program_make_shape_rank2(96, 32));
  IREE_ASSERT_OK(id4_pipeline_program_swiglu(builder_, &options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));

  const id4_pipeline_program_op_t* dispatch = id4_pipeline_program_operation_at(
      program, id4_pipeline_program_operation_count(program) - 1);
  ASSERT_NE(dispatch, nullptr);
  ASSERT_EQ(dispatch->kind, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM);
  EXPECT_TRUE(iree_string_view_equal(dispatch->payload.dispatch_loom.name,
                                     options.name));
  EXPECT_FALSE(iree_string_view_is_empty(
      dispatch->payload.dispatch_loom.kernel.module_path));
  EXPECT_FALSE(iree_string_view_is_empty(
      dispatch->payload.dispatch_loom.kernel.function_name));
  EXPECT_EQ(dispatch->payload.dispatch_loom.binding_count, 6u);

  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, AuthorsSemanticRowScaledSwiGLUProjectionPair) {
  id4_pipeline_program_swiglu_options_t options =
      MakeSwiGLUOptions(ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
                        id4_pipeline_program_make_shape_rank1(12288));
  IREE_ASSERT_OK(id4_pipeline_program_swiglu(builder_, &options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));

  const id4_pipeline_program_op_t* dispatch = id4_pipeline_program_operation_at(
      program, id4_pipeline_program_operation_count(program) - 1);
  ASSERT_NE(dispatch, nullptr);
  ASSERT_EQ(dispatch->kind, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM);
  EXPECT_TRUE(iree_string_view_equal(dispatch->payload.dispatch_loom.name,
                                     options.name));
  EXPECT_EQ(dispatch->payload.dispatch_loom.binding_count, 6u);

  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, RejectsValidRowsBeyondCapacity) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.request.valid_m = options.request.m_capacity + 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

TEST_F(ProgramMatrixTest, RequiresScaleTensorForScaledWeights) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.operands.scale = id4_pipeline_program_tensor_invalid();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

TEST_F(ProgramMatrixTest, RequiresAddendForAddEpilogue) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.request.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

}  // namespace
