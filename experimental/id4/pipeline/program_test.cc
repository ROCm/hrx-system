// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void ExpectStringViewEqual(iree_string_view_t actual,
                                  iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(actual, expected))
      << "actual: " << std::string(actual.data, actual.size)
      << ", expected: " << std::string(expected.data, expected.size);
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

TEST(PipelineProgram, ComputesDenseTensorByteLength) {
  iree_device_size_t byte_length = 0;
  IREE_ASSERT_OK(id4_pipeline_program_tensor_byte_length(
      ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      id4_pipeline_program_make_shape_rank2(3, 4), &byte_length));
  EXPECT_EQ(byte_length, 24u);

  EXPECT_EQ(
      id4_pipeline_program_dtype_byte_length(ID4_PIPELINE_PROGRAM_DTYPE_F32),
      4u);
  EXPECT_EQ(id4_pipeline_program_dtype_byte_length(
                ID4_PIPELINE_PROGRAM_DTYPE_INVALID),
            0u);

  id4_pipeline_program_shape_t invalid_shape = {};
  invalid_shape.rank = 1;
  invalid_shape.dims[0] = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_program_tensor_byte_length(ID4_PIPELINE_PROGRAM_DTYPE_F32,
                                              invalid_shape, &byte_length));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_program_tensor_byte_length(
          ID4_PIPELINE_PROGRAM_DTYPE_INVALID,
          id4_pipeline_program_make_shape_rank1(1), &byte_length));
}

TEST(PipelineProgram, AuthorsProgramAndSealsImmutableCopies) {
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

  id4_pipeline_program_tensor_t weight = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_parameter_options_t weight_options = {
      /*.structure_size=*/sizeof(weight_options),
      /*.next=*/nullptr,
      /*.key=*/IREE_SV("model.layers.0.linear.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_parameter(builder, &weight_options, &weight));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.linear"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(1, 4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_acquire_tensor(builder, &output_options, &output));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      id4_pipeline_make_kernel_config_binding(IREE_SV("@batch"), IREE_SV("1")),
      id4_pipeline_make_kernel_config_binding(IREE_SV("@hidden_size"),
                                              IREE_SV("4")),
  };
  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_read(weight),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      /*.structure_size=*/sizeof(dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.linear"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/linear"), IREE_SV("linear")),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.binding_count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_tap_options_t tap_options = {
      /*.structure_size=*/sizeof(tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.linear.output"),
      /*.tensor=*/output,
  };
  IREE_ASSERT_OK(id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_barrier_options_t barrier_options = {
      /*.structure_size=*/sizeof(barrier_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("block0.after_linear"),
  };
  IREE_ASSERT_OK(id4_pipeline_program_barrier(builder, &barrier_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.output"),
      /*.tensor=*/output,
  };
  IREE_ASSERT_OK(id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();

  ExpectStringViewEqual(id4_pipeline_program_name(program),
                        IREE_SV("test.forward"));
  EXPECT_EQ(id4_pipeline_program_tensor_count(program), 3u);
  EXPECT_EQ(id4_pipeline_program_operation_count(program), 7u);

  const id4_pipeline_program_tensor_record_t* input_record =
      id4_pipeline_program_tensor_at(program, input.ordinal);
  ASSERT_NE(input_record, nullptr);
  ExpectStringViewEqual(input_record->name, IREE_SV("hidden_states.input"));
  EXPECT_EQ(input_record->dtype, ID4_PIPELINE_PROGRAM_DTYPE_F32);
  EXPECT_EQ(input_record->byte_length, 16u);
  EXPECT_EQ(input_record->producer_operation_ordinal, 0u);

  const id4_pipeline_program_tensor_record_t* weight_record =
      id4_pipeline_program_tensor_at(program, weight.ordinal);
  ASSERT_NE(weight_record, nullptr);
  ExpectStringViewEqual(weight_record->name,
                        IREE_SV("model.layers.0.linear.weight"));
  EXPECT_EQ(weight_record->dtype, ID4_PIPELINE_PROGRAM_DTYPE_BF16);
  EXPECT_EQ(weight_record->byte_length, 32u);
  EXPECT_EQ(weight_record->producer_operation_ordinal, 1u);

  const id4_pipeline_program_op_t* dispatch =
      id4_pipeline_program_operation_at(program, 3);
  ASSERT_NE(dispatch, nullptr);
  ASSERT_EQ(dispatch->kind, ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM);
  ExpectStringViewEqual(dispatch->payload.dispatch_loom.name,
                        IREE_SV("block0.linear"));
  ExpectStringViewEqual(dispatch->payload.dispatch_loom.kernel.module_path,
                        IREE_SV("test/linear"));
  ExpectStringViewEqual(dispatch->payload.dispatch_loom.kernel.function_name,
                        IREE_SV("linear"));
  ASSERT_EQ(dispatch->payload.dispatch_loom.config_binding_count, 2u);
  ExpectStringViewEqual(dispatch->payload.dispatch_loom.config_bindings[0].key,
                        IREE_SV("@batch"));
  ExpectStringViewEqual(dispatch->payload.dispatch_loom.config_bindings[1].key,
                        IREE_SV("@hidden_size"));
  ASSERT_EQ(dispatch->payload.dispatch_loom.binding_count, 3u);
  EXPECT_EQ(dispatch->payload.dispatch_loom.bindings[0].tensor.ordinal,
            input.ordinal);
  EXPECT_EQ(dispatch->payload.dispatch_loom.bindings[1].tensor.ordinal,
            weight.ordinal);
  EXPECT_EQ(dispatch->payload.dispatch_loom.bindings[2].access,
            ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE);

  const id4_pipeline_program_op_t* exported =
      id4_pipeline_program_operation_at(program, 6);
  ASSERT_NE(exported, nullptr);
  ASSERT_EQ(exported->kind, ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT);
  EXPECT_EQ(exported->payload.export_value.tensor.ordinal, output.ordinal);

  id4_pipeline_program_release(program);
}

TEST(PipelineProgram, InternsRepeatedParameterDeclarations) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_parameter_options_t weight_options = {
      /*.structure_size=*/sizeof(weight_options),
      /*.next=*/nullptr,
      /*.key=*/IREE_SV("model.layers.0.linear.weight"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_BF16,
      /*.shape=*/id4_pipeline_program_make_shape_rank2(4, 4),
  };
  id4_pipeline_program_tensor_t first_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_ASSERT_OK(
      id4_pipeline_program_parameter(builder, &weight_options, &first_weight));

  id4_pipeline_program_tensor_t second_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_ASSERT_OK(
      id4_pipeline_program_parameter(builder, &weight_options, &second_weight));
  EXPECT_EQ(second_weight.ordinal, first_weight.ordinal);

  id4_pipeline_program_parameter_options_t incompatible_weight_options =
      weight_options;
  incompatible_weight_options.shape =
      id4_pipeline_program_make_shape_rank2(8, 4);
  id4_pipeline_program_tensor_t incompatible_weight =
      id4_pipeline_program_tensor_invalid();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_program_parameter(builder, &incompatible_weight_options,
                                     &incompatible_weight));

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();

  EXPECT_EQ(id4_pipeline_program_tensor_count(program), 1u);
  EXPECT_EQ(id4_pipeline_program_operation_count(program), 1u);
  const id4_pipeline_program_op_t* op =
      id4_pipeline_program_operation_at(program, 0);
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->kind, ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER);

  id4_pipeline_program_release(program);
}

TEST(PipelineProgram, AuthorsConstantTensorWithOwnedData) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  float values[] = {1.0f, 2.0f};
  id4_pipeline_program_tensor_t constant =
      id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_constant_options_t constant_options = {
      /*.structure_size=*/sizeof(constant_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("test.constant"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(2),
      /*.data=*/iree_make_const_byte_span(values, sizeof(values)),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_constant(builder, &constant_options, &constant));
  values[0] = 9.0f;

  id4_pipeline_program_t* program = nullptr;
  IREE_ASSERT_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();

  ASSERT_EQ(id4_pipeline_program_tensor_count(program), 1u);
  const id4_pipeline_program_tensor_record_t* record =
      id4_pipeline_program_tensor_at(program, constant.ordinal);
  ASSERT_NE(record, nullptr);
  ExpectStringViewEqual(record->name, IREE_SV("test.constant"));
  EXPECT_EQ(record->byte_length, sizeof(values));

  ASSERT_EQ(id4_pipeline_program_operation_count(program), 1u);
  const id4_pipeline_program_op_t* op =
      id4_pipeline_program_operation_at(program, 0);
  ASSERT_NE(op, nullptr);
  ASSERT_EQ(op->kind, ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT);
  ASSERT_EQ(op->payload.constant.data_length, sizeof(values));
  const float* copied_values =
      reinterpret_cast<const float*>(op->payload.constant.data);
  EXPECT_EQ(copied_values[0], 1.0f);
  EXPECT_EQ(copied_values[1], 2.0f);

  id4_pipeline_program_release(program);
}

TEST(PipelineProgram, RejectsUninitializedReadsAndCaptures) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t scratch = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t scratch_options = {
      /*.structure_size=*/sizeof(scratch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("scratch"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_acquire_tensor(builder, &scratch_options, &scratch));

  id4_pipeline_program_dispatch_binding_t read_binding =
      id4_pipeline_program_read(scratch);
  id4_pipeline_program_dispatch_loom_options_t read_dispatch_options = {
      /*.structure_size=*/sizeof(read_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("read_scratch"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/read"), IREE_SV("read")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/1,
      /*.bindings=*/&read_binding,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_pipeline_program_dispatch_loom(builder, &read_dispatch_options));

  id4_pipeline_program_tap_options_t tap_options = {
      /*.structure_size=*/sizeof(tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("scratch.before_write"),
      /*.tensor=*/scratch,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("scratch.output"),
      /*.tensor=*/scratch,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_dispatch_binding_t write_binding =
      id4_pipeline_program_write(scratch);
  id4_pipeline_program_dispatch_loom_options_t write_dispatch_options = {
      /*.structure_size=*/sizeof(write_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("write_scratch"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/write"), IREE_SV("write")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/1,
      /*.bindings=*/&write_binding,
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_dispatch_loom(builder, &write_dispatch_options));
  IREE_ASSERT_OK(id4_pipeline_program_tap(builder, &tap_options));
  IREE_ASSERT_OK(id4_pipeline_program_export(builder, &export_options));

  builder_scope.DestroyBuilder();
}

TEST(PipelineProgram, ExternalOutputImportRequiresWriteBeforeExport) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("stage.output"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_program_dispatch_binding_t read_binding =
      id4_pipeline_program_read(output);
  id4_pipeline_program_dispatch_loom_options_t read_dispatch_options = {
      /*.structure_size=*/sizeof(read_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("read_output"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/read"), IREE_SV("read")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/1,
      /*.bindings=*/&read_binding,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_pipeline_program_dispatch_loom(builder, &read_dispatch_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("stage.output"),
      /*.tensor=*/output,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_pipeline_program_export(builder, &export_options));

  id4_pipeline_program_dispatch_binding_t write_binding =
      id4_pipeline_program_write(output);
  id4_pipeline_program_dispatch_loom_options_t write_dispatch_options = {
      /*.structure_size=*/sizeof(write_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("write_output"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/write"), IREE_SV("write")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/1,
      /*.bindings=*/&write_binding,
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_dispatch_loom(builder, &write_dispatch_options));
  IREE_ASSERT_OK(id4_pipeline_program_export(builder, &export_options));

  builder_scope.DestroyBuilder();
}

TEST(PipelineProgram, RejectsAmbiguousIdentitiesAndInvalidHandles) {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("hidden_states"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_ASSERT_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  id4_pipeline_program_parameter_options_t duplicate_parameter_options = {
      /*.structure_size=*/sizeof(duplicate_parameter_options),
      /*.next=*/nullptr,
      /*.key=*/IREE_SV("hidden_states"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  id4_pipeline_program_tensor_t duplicate =
      id4_pipeline_program_tensor_invalid();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        id4_pipeline_program_parameter(
                            builder, &duplicate_parameter_options, &duplicate));

  id4_pipeline_program_tap_options_t invalid_tap_options = {
      /*.structure_size=*/sizeof(invalid_tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("invalid"),
      /*.tensor=*/id4_pipeline_program_tensor_invalid(),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_program_tap(builder, &invalid_tap_options));

  id4_pipeline_program_export_options_t export_options = {
      /*.structure_size=*/sizeof(export_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("hidden_states.output"),
      /*.tensor=*/input,
  };
  IREE_ASSERT_OK(id4_pipeline_program_export(builder, &export_options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        id4_pipeline_program_export(builder, &export_options));

  builder_scope.DestroyBuilder();
}

}  // namespace
