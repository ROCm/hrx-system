// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_matrix.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static const id4_pipeline_program_op_t* FindDispatchByName(
    const id4_pipeline_program_t* program, iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (op && op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM &&
        iree_string_view_equal(op->payload.dispatch_loom.name, name)) {
      return op;
    }
  }
  return nullptr;
}

static iree_string_view_t FindSemanticAttribute(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch,
    iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < dispatch->semantic_attributes.count; ++i) {
    const iree_string_pair_t* attribute =
        &dispatch->semantic_attributes.pairs[i];
    if (iree_string_view_equal(attribute->key, key)) return attribute->value;
  }
  return iree_string_view_empty();
}

static void ExpectSemanticAttribute(
    const id4_pipeline_program_dispatch_loom_op_t* dispatch,
    iree_string_view_t key, iree_string_view_t expected_value) {
  const iree_string_view_t actual_value = FindSemanticAttribute(dispatch, key);
  EXPECT_TRUE(iree_string_view_equal(actual_value, expected_value));
}

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

  id4_pipeline_program_tensor_t AcquireInitializedPrefix(
      iree_string_view_t name, id4_pipeline_program_dtype_t dtype,
      id4_pipeline_program_shape_t shape,
      iree_device_size_t initialized_byte_length) {
    id4_pipeline_program_tensor_t tensor = Acquire(name, dtype, shape);
    const id4_pipeline_program_dispatch_binding_t bindings[] = {
        id4_pipeline_program_write_range(tensor, 0, initialized_byte_length),
    };
    const id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
        /*.structure_size=*/sizeof(dispatch_options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("initialize.prefix"),
        /*.kernel=*/
        {
            /*.module_path=*/IREE_SV("test/initialize_prefix"),
            /*.function_name=*/IREE_SV("initialize_prefix"),
        },
        /*.config_binding_count=*/0,
        /*.config_bindings=*/nullptr,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.bindings=*/bindings,
    };
    IREE_CHECK_OK(
        id4_pipeline_program_dispatch_loom(builder_, &dispatch_options));
    const id4_pipeline_program_barrier_options_t barrier_options = {
        /*.structure_size=*/sizeof(barrier_options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("initialize.prefix.ready"),
    };
    IREE_CHECK_OK(id4_pipeline_program_barrier(builder_, &barrier_options));
    return tensor;
  }

  void SealAndRelease() {
    id4_pipeline_program_t* program = nullptr;
    IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
        builder_, iree_allocator_system(), &program));
    id4_pipeline_program_release(program);
  }

  id4_pipeline_program_matrix_options_t MakeBlockScaledOptions() {
    id4_pipeline_program_matrix_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("layer.linear"),
        /*.problem=*/
        {
            /*.valid_m=*/451,
            /*.m_capacity=*/512,
            /*.n=*/1024,
            /*.k=*/4096,
            /*.input_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.input_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(512, 4096));
    options.operands.parameter = {
        /*.weight=*/
        {
            /*.source_scope=*/IREE_SV("model"),
            /*.key=*/IREE_SV("weight"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.shape=*/id4_pipeline_program_make_shape_rank2(1024, 4096),
        },
        /*.weight_layout=*/
        ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        /*.scale=*/
        {
            /*.source_scope=*/IREE_SV("model"),
            /*.key=*/IREE_SV("scale"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.shape=*/id4_pipeline_program_make_shape_rank2(8, 32),
        },
        /*.scale_layout=*/
        ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
    };
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
        /*.problem=*/
        {
            /*.valid_m=*/113,
            /*.m_capacity=*/128,
            /*.n=*/4608,
            /*.k=*/4608,
            /*.input_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.input_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("transformer_input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(128, 4608));
    options.operands.parameter = {
        /*.weight=*/
        {
            /*.source_scope=*/IREE_SV("transformer"),
            /*.key=*/IREE_SV("transformer_weight"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.shape=*/id4_pipeline_program_make_shape_rank2(4608, 4608),
        },
        /*.weight_layout=*/
        ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        /*.scale=*/
        {
            /*.source_scope=*/IREE_SV("transformer"),
            /*.key=*/IREE_SV("transformer_scale"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.shape=*/id4_pipeline_program_make_shape_rank1(4608),
        },
        /*.scale_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
    };
    options.operands.output =
        Acquire(IREE_SV("transformer_output"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                id4_pipeline_program_make_shape_rank2(128, 4608));
    options.operands.addend = id4_pipeline_program_tensor_invalid();
    return options;
  }

  id4_pipeline_program_matrix_options_t MakeBf16Options() {
    id4_pipeline_program_matrix_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.name=*/IREE_SV("vae.conv"),
        /*.problem=*/
        {
            /*.valid_m=*/256,
            /*.m_capacity=*/256,
            /*.n=*/512,
            /*.k=*/4608,
            /*.input_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.input_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("im2col"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(256, 4608));
    options.operands.parameter = {
        /*.weight=*/
        {
            /*.source_scope=*/IREE_SV("vae"),
            /*.key=*/IREE_SV("conv.weight.packed_oc_ky_kx_ic.bf16"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.shape=*/id4_pipeline_program_make_shape_rank2(512, 4608),
        },
        /*.weight_layout=*/
        ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        /*.scale=*/{},
        /*.scale_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_NONE,
    };
    options.operands.output =
        Acquire(IREE_SV("contraction"), ID4_PIPELINE_PROGRAM_DTYPE_F32,
                id4_pipeline_program_make_shape_rank2(256, 512));
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
            /*.accumulator_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.epilogue=*/ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_NONE,
            /*.output_dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
            /*.output_layout=*/ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_ROW_MAJOR,
        },
    };
    options.operands.input =
        Import(IREE_SV("swiglu_input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
               id4_pipeline_program_make_shape_rank2(512, 4096));
    options.operands.gate_parameter = {
        /*.weight=*/
        {
            /*.source_scope=*/IREE_SV("model"),
            /*.key=*/IREE_SV("gate_weight"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.shape=*/id4_pipeline_program_make_shape_rank2(12288, 4096),
        },
        /*.weight_layout=*/
        ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        /*.scale=*/
        {
            /*.source_scope=*/IREE_SV("model"),
            /*.key=*/IREE_SV("gate_scale"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.shape=*/scale_shape,
        },
        /*.scale_layout=*/scale_layout,
    };
    options.operands.up_parameter = {
        /*.weight=*/
        {
            /*.source_scope=*/IREE_SV("model"),
            /*.key=*/IREE_SV("up_weight"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3,
            /*.shape=*/id4_pipeline_program_make_shape_rank2(12288, 4096),
        },
        /*.weight_layout=*/
        ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_RHS_TRANSPOSED_ROW_MAJOR,
        /*.scale=*/
        {
            /*.source_scope=*/IREE_SV("model"),
            /*.key=*/IREE_SV("up_scale"),
            /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
            /*.shape=*/scale_shape,
        },
        /*.scale_layout=*/scale_layout,
    };
    options.operands.output =
        Acquire(IREE_SV("swiglu_output"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
                id4_pipeline_program_make_shape_rank2(512, 12288));
    return options;
  }

  iree_arena_block_pool_t block_pool_;
  id4_pipeline_program_builder_t* builder_ = nullptr;
};

TEST_F(ProgramMatrixTest, AcceptsBlockScaledParameterContraction) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, PreservesSemanticContractionDiagnostics) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));
  const id4_pipeline_program_op_t* op =
      FindDispatchByName(program, options.name);
  ASSERT_NE(op, nullptr);
  const id4_pipeline_program_dispatch_loom_op_t* dispatch =
      &op->payload.dispatch_loom;
  ExpectSemanticAttribute(dispatch, IREE_SV("semantic.family"),
                          IREE_SV("matrix"));
  ExpectSemanticAttribute(dispatch, IREE_SV("semantic.operation"),
                          IREE_SV("contraction"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.valid_m"), IREE_SV("451"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.m_capacity"),
                          IREE_SV("512"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.n"), IREE_SV("1024"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.k"), IREE_SV("4096"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.input.dtype"),
                          IREE_SV("bf16"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.accumulator.dtype"),
                          IREE_SV("f32"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.output.dtype"),
                          IREE_SV("bf16"));
  ExpectSemanticAttribute(dispatch, IREE_SV("matrix.execution.weight.dtype"),
                          IREE_SV("f8e4m3"));
  ExpectSemanticAttribute(dispatch,
                          IREE_SV("matrix.execution.input_row_coverage"),
                          IREE_SV("valid_m"));
  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, DeclaresMaskedCandidateValidInputRows) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  const iree_device_size_t valid_input_byte_length =
      (iree_device_size_t)options.problem.valid_m * options.problem.k *
      sizeof(uint16_t);
  options.operands.input = AcquireInitializedPrefix(
      IREE_SV("partial_input"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(options.problem.m_capacity,
                                            options.problem.k),
      valid_input_byte_length);
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));
  const id4_pipeline_program_dispatch_binding_t* input_binding = nullptr;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (op && op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM &&
        iree_string_view_equal(op->payload.dispatch_loom.name, options.name)) {
      input_binding = &op->payload.dispatch_loom.bindings[0];
      break;
    }
  }
  ASSERT_NE(input_binding, nullptr);
  EXPECT_TRUE(
      iree_all_bits_set(input_binding->flags,
                        ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_READ_RANGE));
  EXPECT_EQ(input_binding->read_range.offset, 0u);
  EXPECT_EQ(input_binding->read_range.length, valid_input_byte_length);
  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, AcceptsRowScaledParameterContraction) {
  id4_pipeline_program_matrix_options_t options = MakeRowScaledOptions();
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, AcceptsBf16ParameterContractionToF32) {
  id4_pipeline_program_matrix_options_t options = MakeBf16Options();
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, AcceptsResidualEpilogue) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
  options.operands.addend =
      Import(IREE_SV("residual"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
             id4_pipeline_program_make_shape_rank2(512, 1024));
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, AcceptsBroadcastBiasEpilogue) {
  id4_pipeline_program_matrix_options_t options = MakeRowScaledOptions();
  options.problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_BIAS;
  options.operands.addend =
      Import(IREE_SV("bias"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
             id4_pipeline_program_make_shape_rank1(options.problem.n));
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, AcceptsColumnMajorBroadcastBiasEpilogue) {
  id4_pipeline_program_matrix_options_t options = MakeRowScaledOptions();
  options.problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_BIAS;
  options.problem.output_layout =
      ID4_PIPELINE_PROGRAM_MATRIX_LAYOUT_COLUMN_MAJOR;
  options.operands.addend =
      Import(IREE_SV("bias"), ID4_PIPELINE_PROGRAM_DTYPE_BF16,
             id4_pipeline_program_make_shape_rank1(options.problem.n));
  IREE_ASSERT_OK(id4_pipeline_program_matrix(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, AcceptsBlockScaledSwiGLUProjectionPair) {
  id4_pipeline_program_swiglu_options_t options = MakeSwiGLUOptions(
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
      id4_pipeline_program_make_shape_rank2(96, 32));
  IREE_ASSERT_OK(id4_pipeline_program_swiglu(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, SwiGLUDispatchesRetainOneSemanticRequest) {
  id4_pipeline_program_swiglu_options_t options = MakeSwiGLUOptions(
      ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_INPUT_BLOCK_128X128,
      id4_pipeline_program_make_shape_rank2(96, 32));
  IREE_ASSERT_OK(id4_pipeline_program_swiglu(builder_, &options));
  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder_, iree_allocator_system(), &program));
  iree_host_size_t semantic_dispatch_count = 0;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_program_operation_count(program); ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op || op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) continue;
    const id4_pipeline_program_dispatch_loom_op_t* dispatch =
        &op->payload.dispatch_loom;
    if (!iree_string_view_equal(
            FindSemanticAttribute(dispatch, IREE_SV("semantic.family")),
            IREE_SV("matrix"))) {
      continue;
    }
    ++semantic_dispatch_count;
    ExpectSemanticAttribute(dispatch, IREE_SV("semantic.operation"),
                            IREE_SV("swiglu"));
    EXPECT_FALSE(iree_string_view_is_empty(
        FindSemanticAttribute(dispatch, IREE_SV("semantic.role"))));
    EXPECT_FALSE(iree_string_view_is_empty(
        FindSemanticAttribute(dispatch, IREE_SV("semantic.schedule"))));
    ExpectSemanticAttribute(dispatch, IREE_SV("matrix.valid_m"),
                            IREE_SV("451"));
    ExpectSemanticAttribute(dispatch, IREE_SV("matrix.m_capacity"),
                            IREE_SV("512"));
    ExpectSemanticAttribute(dispatch, IREE_SV("matrix.n"), IREE_SV("12288"));
    ExpectSemanticAttribute(dispatch, IREE_SV("matrix.k"), IREE_SV("4096"));
  }
  EXPECT_GT(semantic_dispatch_count, 0u);
  id4_pipeline_program_release(program);
}

TEST_F(ProgramMatrixTest, AcceptsRowScaledSwiGLUProjectionPair) {
  id4_pipeline_program_swiglu_options_t options =
      MakeSwiGLUOptions(ID4_PIPELINE_PROGRAM_MATRIX_SCALE_LAYOUT_OUTPUT_ROW,
                        id4_pipeline_program_make_shape_rank1(12288));
  IREE_ASSERT_OK(id4_pipeline_program_swiglu(builder_, &options));
  SealAndRelease();
}

TEST_F(ProgramMatrixTest, RejectsValidRowsBeyondCapacity) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.problem.valid_m = options.problem.m_capacity + 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

TEST_F(ProgramMatrixTest, RequiresScaleTensorForScaledWeights) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.operands.parameter.scale = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

TEST_F(ProgramMatrixTest, RequiresAddendForAddEpilogue) {
  id4_pipeline_program_matrix_options_t options = MakeBlockScaledOptions();
  options.problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_ADD;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

TEST_F(ProgramMatrixTest, RequiresAddendForBiasEpilogue) {
  id4_pipeline_program_matrix_options_t options = MakeRowScaledOptions();
  options.problem.epilogue = ID4_PIPELINE_PROGRAM_MATRIX_EPILOGUE_BIAS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_program_matrix(builder_, &options));
}

}  // namespace
