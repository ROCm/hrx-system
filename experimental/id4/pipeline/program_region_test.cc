// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_region.h"

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
        /*.program_name=*/IREE_SV("test.region"),
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

class RegionBuilderScope {
 public:
  RegionBuilderScope() {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    id4_pipeline_region_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.region_name=*/IREE_SV("test.region"),
        /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN,
        /*.flags=*/0,
        /*.block_pool=*/&block_pool_,
        /*.command_buffer=*/nullptr,
        /*.binding_capacity=*/4,
        /*.local_binding_slot=*/0,
    };
    IREE_CHECK_OK(id4_pipeline_region_builder_create(
        &options, iree_allocator_system(), &builder_));
  }

  ~RegionBuilderScope() {
    id4_pipeline_region_builder_destroy(builder_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  id4_pipeline_region_builder_t* builder() { return builder_; }

 private:
  iree_arena_block_pool_t block_pool_;
  id4_pipeline_region_builder_t* builder_ = nullptr;
};

static id4_pipeline_tensor_shape_t ConvertShape(
    id4_pipeline_program_shape_t source) {
  id4_pipeline_tensor_shape_t target = {};
  target.rank = source.rank;
  for (uint32_t i = 0; i < source.rank; ++i) {
    target.dims[i] = source.dims[i];
  }
  return target;
}

static id4_pipeline_tensor_layout_t MakeTensorLayout(
    const id4_pipeline_program_tensor_record_t* tensor) {
  return id4_pipeline_tensor_layout_t{
      /*.name=*/tensor->name,
      /*.dtype=*/id4_pipeline_program_region_convert_dtype(tensor->dtype),
      /*.shape=*/ConvertShape(tensor->shape),
      /*.byte_length=*/tensor->byte_length,
      /*.alignment=*/0,
  };
}

static iree_status_t ResolveImport(
    void* user_data, const id4_pipeline_program_import_op_t* import_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t import_ordinal, id4_pipeline_tensor_import_t* out_import) {
  (void)user_data;
  id4_pipeline_tensor_import_flags_t flags = 0;
  if (iree_all_bits_set(import_op->flags,
                        ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED)) {
    flags |= ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED;
  }
  *out_import = id4_pipeline_tensor_import_t{
      /*.layout=*/MakeTensorLayout(tensor),
      /*.binding_slot=*/(uint32_t)(1 + import_ordinal),
      /*.offset=*/0,
      /*.flags=*/flags,
  };
  return iree_ok_status();
}

static iree_status_t ResolveSharedTensor(
    void* user_data, const id4_pipeline_program_acquire_op_t* acquire_op,
    const id4_pipeline_program_tensor_record_t* tensor, bool* out_is_shared,
    id4_pipeline_tensor_import_t* out_import) {
  (void)user_data;
  (void)acquire_op;
  *out_is_shared = true;
  *out_import = id4_pipeline_tensor_import_t{
      /*.layout=*/MakeTensorLayout(tensor),
      /*.binding_slot=*/1,
      /*.offset=*/0,
      /*.flags=*/0,
  };
  return iree_ok_status();
}

static iree_status_t ResolveTap(
    void* user_data, const id4_pipeline_program_tap_op_t* tap_op,
    const id4_pipeline_program_tensor_record_t* tensor,
    iree_host_size_t tap_ordinal, id4_pipeline_tensor_import_t* out_import) {
  (void)user_data;
  (void)tap_op;
  *out_import = id4_pipeline_tensor_import_t{
      /*.layout=*/MakeTensorLayout(tensor),
      /*.binding_slot=*/(uint32_t)(2 + tap_ordinal),
      /*.offset=*/0,
      /*.flags=*/0,
  };
  return iree_ok_status();
}

static id4_pipeline_program_t* CreateBoundaryDispatchProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  id4_pipeline_program_tensor_t output = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t output_options = {
      /*.structure_size=*/sizeof(output_options),
      /*.next=*/nullptr,
      /*.flags=*/0,
      /*.name=*/IREE_SV("output"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &output_options, &output));

  id4_pipeline_program_dispatch_binding_t bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(output),
  };
  id4_pipeline_program_dispatch_loom_options_t dispatch_options = {
      /*.structure_size=*/sizeof(dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("copy"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/copy"), IREE_SV("copy")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  IREE_CHECK_OK(id4_pipeline_program_dispatch_loom(builder, &dispatch_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static id4_pipeline_program_t* CreateSharedPartialWriteProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t shared = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t shared_options = {
      /*.structure_size=*/sizeof(shared_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("shared"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_U32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &shared_options, &shared));

  id4_pipeline_program_dispatch_binding_t first_bindings[] = {
      id4_pipeline_program_write_range(shared, 0, 8),
  };
  id4_pipeline_program_dispatch_loom_options_t first_dispatch_options = {
      /*.structure_size=*/sizeof(first_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("write_first"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/write"),
                                   IREE_SV("write_first")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(first_bindings),
      /*.bindings=*/first_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &first_dispatch_options));

  id4_pipeline_program_region_cut_options_t cut_options = {
      /*.structure_size=*/sizeof(cut_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("after_first"),
  };
  IREE_CHECK_OK(id4_pipeline_program_region_cut(builder, &cut_options));

  id4_pipeline_program_dispatch_binding_t second_bindings[] = {
      id4_pipeline_program_write_range(shared, 8, 8),
  };
  id4_pipeline_program_dispatch_loom_options_t second_dispatch_options = {
      /*.structure_size=*/sizeof(second_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("write_second"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/write"),
                                   IREE_SV("write_second")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(second_bindings),
      /*.bindings=*/second_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &second_dispatch_options));

  id4_pipeline_program_barrier_options_t barrier_options = {
      /*.structure_size=*/sizeof(barrier_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("after_second"),
  };
  IREE_CHECK_OK(id4_pipeline_program_barrier(builder, &barrier_options));

  id4_pipeline_program_tap_options_t tap_options = {
      /*.structure_size=*/sizeof(tap_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("shared"),
      /*.tensor=*/shared,
  };
  IREE_CHECK_OK(id4_pipeline_program_tap(builder, &tap_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

static id4_pipeline_program_t* CreateLocalCrossingProgram() {
  ProgramBuilderScope builder_scope;
  id4_pipeline_program_builder_t* builder = builder_scope.builder();

  id4_pipeline_program_tensor_t input = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_import_tensor_options_t input_options = {
      /*.structure_size=*/sizeof(input_options),
      /*.next=*/nullptr,
      /*.flags=*/ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED,
      /*.name=*/IREE_SV("input"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_import_tensor(builder, &input_options, &input));

  id4_pipeline_program_tensor_t scratch = id4_pipeline_program_tensor_invalid();
  id4_pipeline_program_acquire_tensor_options_t scratch_options = {
      /*.structure_size=*/sizeof(scratch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("scratch"),
      /*.dtype=*/ID4_PIPELINE_PROGRAM_DTYPE_F32,
      /*.shape=*/id4_pipeline_program_make_shape_rank1(4),
  };
  IREE_CHECK_OK(
      id4_pipeline_program_acquire_tensor(builder, &scratch_options, &scratch));

  id4_pipeline_program_dispatch_binding_t first_bindings[] = {
      id4_pipeline_program_read(input),
      id4_pipeline_program_write(scratch),
  };
  id4_pipeline_program_dispatch_loom_options_t first_dispatch_options = {
      /*.structure_size=*/sizeof(first_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("write_scratch"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/copy"), IREE_SV("copy")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(first_bindings),
      /*.bindings=*/first_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &first_dispatch_options));

  id4_pipeline_program_barrier_options_t barrier_options = {
      /*.structure_size=*/sizeof(barrier_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("after_write"),
  };
  IREE_CHECK_OK(id4_pipeline_program_barrier(builder, &barrier_options));

  id4_pipeline_program_dispatch_binding_t second_bindings[] = {
      id4_pipeline_program_read(scratch),
  };
  id4_pipeline_program_dispatch_loom_options_t second_dispatch_options = {
      /*.structure_size=*/sizeof(second_dispatch_options),
      /*.next=*/nullptr,
      /*.name=*/IREE_SV("read_scratch"),
      /*.kernel=*/
      id4_pipeline_make_kernel_ref(IREE_SV("test/read"), IREE_SV("read")),
      /*.config_binding_count=*/0,
      /*.config_bindings=*/nullptr,
      /*.binding_count=*/IREE_ARRAYSIZE(second_bindings),
      /*.bindings=*/second_bindings,
  };
  IREE_CHECK_OK(
      id4_pipeline_program_dispatch_loom(builder, &second_dispatch_options));

  id4_pipeline_program_t* program = nullptr;
  IREE_CHECK_OK(id4_pipeline_program_builder_seal(
      builder, iree_allocator_system(), &program));
  builder_scope.DestroyBuilder();
  return program;
}

TEST(PipelineProgramRegion, LowersSourceRangeWithPriorBoundaryDeclarations) {
  id4_pipeline_program_t* program = CreateBoundaryDispatchProgram();
  RegionBuilderScope region_scope;

  id4_pipeline_program_region_lower_options_t lower_options = {
      /*.structure_size=*/sizeof(lower_options),
      /*.next=*/nullptr,
      /*.program=*/program,
      /*.source_operation_offset=*/2,
      /*.source_operation_count=*/1,
      /*.builder=*/region_scope.builder(),
      /*.tap_mode=*/ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE,
      /*.captured_tap_names=*/iree_string_view_list_empty(),
      /*.local_tensor_alignment=*/16,
      /*.user_data=*/nullptr,
      /*.resolve_import=*/ResolveImport,
      /*.resolve_parameter=*/nullptr,
      /*.resolve_constant=*/nullptr,
      /*.resolve_kernel=*/nullptr,
      /*.resolve_tap=*/nullptr,
  };
  IREE_ASSERT_OK(id4_pipeline_program_region_lower(&lower_options,
                                                   iree_allocator_system()));

  id4_pipeline_region_statistics_t statistics = {};
  id4_pipeline_region_builder_statistics(region_scope.builder(), &statistics);
  EXPECT_EQ(statistics.bound_import_count, 2u);
  EXPECT_EQ(statistics.dispatch_count, 1u);
  EXPECT_EQ(statistics.operation_count, 1u);
  EXPECT_EQ(statistics.local_acquire_count, 0u);

  id4_pipeline_program_release(program);
}

TEST(PipelineProgramRegion, RejectsLocalTensorCrossingSourceRange) {
  id4_pipeline_program_t* program = CreateLocalCrossingProgram();
  RegionBuilderScope region_scope;

  id4_pipeline_program_region_lower_options_t lower_options = {
      /*.structure_size=*/sizeof(lower_options),
      /*.next=*/nullptr,
      /*.program=*/program,
      /*.source_operation_offset=*/1,
      /*.source_operation_count=*/2,
      /*.builder=*/region_scope.builder(),
      /*.tap_mode=*/ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_IGNORE,
      /*.captured_tap_names=*/iree_string_view_list_empty(),
      /*.local_tensor_alignment=*/16,
      /*.user_data=*/nullptr,
      /*.resolve_import=*/ResolveImport,
      /*.resolve_parameter=*/nullptr,
      /*.resolve_constant=*/nullptr,
      /*.resolve_kernel=*/nullptr,
      /*.resolve_tap=*/nullptr,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        id4_pipeline_program_region_lower(
                            &lower_options, iree_allocator_system()));

  id4_pipeline_program_release(program);
}

TEST(PipelineProgramRegion,
     SharedTensorImportsPriorWriteRangesIntoSourceRange) {
  id4_pipeline_program_t* program = CreateSharedPartialWriteProgram();
  RegionBuilderScope region_scope;
  iree_string_view_t captured_tap_names[] = {IREE_SV("shared")};

  id4_pipeline_program_region_lower_options_t lower_options = {
      /*.structure_size=*/sizeof(lower_options),
      /*.next=*/nullptr,
      /*.program=*/program,
      /*.source_operation_offset=*/3,
      /*.source_operation_count=*/3,
      /*.builder=*/region_scope.builder(),
      /*.tap_mode=*/ID4_PIPELINE_PROGRAM_REGION_TAP_MODE_CAPTURE,
      /*.captured_tap_names=*/
      iree_string_view_list_t{
          /*.count=*/IREE_ARRAYSIZE(captured_tap_names),
          /*.values=*/captured_tap_names,
      },
      /*.local_tensor_alignment=*/16,
      /*.user_data=*/nullptr,
      /*.resolve_import=*/ResolveImport,
      /*.resolve_parameter=*/nullptr,
      /*.resolve_constant=*/nullptr,
      /*.resolve_shared_tensor=*/ResolveSharedTensor,
      /*.resolve_kernel=*/nullptr,
      /*.resolve_tap=*/ResolveTap,
  };
  IREE_ASSERT_OK(id4_pipeline_program_region_lower(&lower_options,
                                                   iree_allocator_system()));

  id4_pipeline_region_statistics_t statistics = {};
  id4_pipeline_region_builder_statistics(region_scope.builder(), &statistics);
  EXPECT_EQ(statistics.bound_import_count, 2u);
  EXPECT_EQ(statistics.dispatch_count, 1u);
  EXPECT_EQ(statistics.copy_count, 1u);

  id4_pipeline_program_release(program);
}

}  // namespace
