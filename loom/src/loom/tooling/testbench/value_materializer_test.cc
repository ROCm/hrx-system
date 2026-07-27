// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/value_materializer.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/io/memory_stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/numpy_io.h"
#include "iree/tooling/testdata/npy/npy_files.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"

namespace loom {
namespace {

class ValueMaterializerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(IREE_SV("testbench"), host_allocator_,
                                       host_allocator_, &device_allocator_));
  }

  void TearDown() override {
    iree_io_stream_release(written_stream_);
    iree_hal_allocator_release(device_allocator_);
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_module_t* ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("value_materializer_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModule(loom_module_t* module) {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(
        loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));
    return plan;
  }

  loom_testbench_value_materializer_options_t MaterializerOptions() {
    loom_testbench_value_materializer_options_t options = {};
    loom_testbench_value_materializer_options_initialize(&options);
    options.device_allocator = device_allocator_;
    options.open_read_file.fn = ValueMaterializerTest::OpenReadFile;
    options.open_read_file.user_data = this;
    options.open_write_file.fn = ValueMaterializerTest::OpenWriteFile;
    options.open_write_file.user_data = this;
    return options;
  }

  static iree_status_t OpenReadFile(void* user_data, iree_string_view_t path,
                                    iree_io_stream_t** out_stream) {
    (void)user_data;
    const iree_file_toc_t* file_toc = iree_numpy_npy_files_create();
    for (iree_host_size_t file_index = 0;
         file_index < iree_numpy_npy_files_size(); ++file_index) {
      if (iree_string_view_equal(
              path, iree_make_cstring_view(file_toc[file_index].name))) {
        return iree_io_memory_stream_wrap(
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_SEEKABLE,
            iree_make_byte_span((void*)file_toc[file_index].data,
                                file_toc[file_index].size),
            iree_io_stream_release_callback_null(), iree_allocator_system(),
            out_stream);
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "test NPY fixture was not found");
  }

  static iree_status_t OpenWriteFile(void* user_data, iree_string_view_t path,
                                     iree_io_stream_t** out_stream) {
    ValueMaterializerTest* test =
        static_cast<ValueMaterializerTest*>(user_data);
    iree_io_stream_release(test->written_stream_);
    test->written_stream_ = nullptr;
    test->written_path_.assign(path.data, path.size);

    iree_io_stream_t* stream = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE,
        /*block_size=*/64, test->host_allocator_, &stream));
    iree_io_stream_retain(stream);
    test->written_stream_ = stream;
    *out_stream = stream;
    return iree_ok_status();
  }

  iree_hal_buffer_view_t* LookupBufferView(
      const loom_testbench_value_table_t* table, loom_value_id_t value_id,
      loom_testbench_value_t* out_value) {
    IREE_EXPECT_OK(
        loom_testbench_value_table_lookup_retain(table, value_id, out_value));
    iree_hal_buffer_view_t* buffer_view =
        loom_testbench_value_buffer_view(out_value);
    EXPECT_NE(buffer_view, nullptr);
    return buffer_view;
  }

  template <typename T>
  void ExpectBufferViewContents(iree_hal_buffer_view_t* buffer_view,
                                std::vector<iree_hal_dim_t> shape,
                                iree_hal_element_type_t element_type,
                                std::vector<T> expected_contents) {
    ASSERT_EQ(iree_hal_buffer_view_shape_rank(buffer_view), shape.size());
    for (iree_host_size_t dim_index = 0; dim_index < shape.size();
         ++dim_index) {
      EXPECT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, dim_index),
                shape[dim_index]);
    }
    EXPECT_EQ(iree_hal_buffer_view_element_type(buffer_view), element_type);
    EXPECT_EQ(iree_hal_buffer_view_encoding_type(buffer_view),
              IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR);

    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
    ASSERT_EQ(iree_hal_buffer_byte_length(buffer),
              expected_contents.size() * sizeof(T));
    std::vector<T> actual_contents(expected_contents.size());
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        buffer, /*source_offset=*/0, actual_contents.data(),
        actual_contents.size() * sizeof(T)));
    EXPECT_EQ(actual_contents, expected_contents);
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
  iree_hal_allocator_t* device_allocator_ = nullptr;
  iree_io_stream_t* written_stream_ = nullptr;
  std::string written_path_;
};

TEST_F(ValueMaterializerTest, MaterializesGeneratedValues) {
  loom_module_t* module = ParseModule(R"(
check.case @generated {
  %m = check.param.range linear bounds(2 to 4) step(1) : index
  %seed = check.param.seed base(7) count(2) : i64
  %scalar = check.literal value(42) : i32
  %i8 = check.generate.iota offset(-2) step(1) : tensor<4xi8>
  %i8_down = check.generate.iota offset(2) step(-1) : tensor<4xi8>
  %iota = check.generate.iota offset(0) step(1) : tensor<[%m]xi32>
  %f16 = check.generate.fill value(0.5) : tensor<2xf16>
  %fill = check.generate.fill value(1.5) : tensor<3xf32>
  %bf16 = check.generate.fill value(0.25) : tensor<2xbf16>
  %uniform = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<4xf32>
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t options = MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
      &options, &case_plan, /*sample_ordinal=*/1, &table));

  ASSERT_EQ(case_plan.parameter_count, 2u);
  ASSERT_EQ(case_plan.value_source_count, 8u);
  loom_testbench_value_t scalar = {};
  IREE_ASSERT_OK(loom_testbench_value_table_lookup_retain(
      &table, case_plan.value_sources[0].value_id, &scalar));
  ASSERT_TRUE(loom_testbench_value_is_scalar(&scalar));
  EXPECT_EQ(scalar.scalar.kind, IREE_TOOLING_VALUE_KIND_I32);
  EXPECT_EQ(scalar.scalar.storage.i32, 42);
  loom_testbench_value_deinitialize(&scalar);

  loom_testbench_value_t i8 = {};
  iree_hal_buffer_view_t* i8_view =
      LookupBufferView(&table, case_plan.value_sources[1].value_id, &i8);
  ExpectBufferViewContents<int8_t>(i8_view, {4}, IREE_HAL_ELEMENT_TYPE_SINT_8,
                                   {-2, -1, 0, 1});
  loom_testbench_value_deinitialize(&i8);

  loom_testbench_value_t i8_down = {};
  iree_hal_buffer_view_t* i8_down_view =
      LookupBufferView(&table, case_plan.value_sources[2].value_id, &i8_down);
  ExpectBufferViewContents<int8_t>(i8_down_view, {4},
                                   IREE_HAL_ELEMENT_TYPE_SINT_8, {2, 1, 0, -1});
  loom_testbench_value_deinitialize(&i8_down);

  loom_testbench_value_t iota = {};
  iree_hal_buffer_view_t* iota_view =
      LookupBufferView(&table, case_plan.value_sources[3].value_id, &iota);
  ExpectBufferViewContents<int32_t>(iota_view, {3},
                                    IREE_HAL_ELEMENT_TYPE_SINT_32, {0, 1, 2});
  loom_testbench_value_deinitialize(&iota);

  loom_testbench_value_t f16 = {};
  iree_hal_buffer_view_t* f16_view =
      LookupBufferView(&table, case_plan.value_sources[4].value_id, &f16);
  ExpectBufferViewContents<uint16_t>(
      f16_view, {2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
      {iree_math_f32_to_f16(0.5f), iree_math_f32_to_f16(0.5f)});
  loom_testbench_value_deinitialize(&f16);

  loom_testbench_value_t fill = {};
  iree_hal_buffer_view_t* fill_view =
      LookupBufferView(&table, case_plan.value_sources[5].value_id, &fill);
  ExpectBufferViewContents<float>(
      fill_view, {3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32, {1.5f, 1.5f, 1.5f});
  loom_testbench_value_deinitialize(&fill);

  loom_testbench_value_t bf16 = {};
  iree_hal_buffer_view_t* bf16_view =
      LookupBufferView(&table, case_plan.value_sources[6].value_id, &bf16);
  ExpectBufferViewContents<uint16_t>(
      bf16_view, {2}, IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
      {iree_math_f32_to_bf16(0.25f), iree_math_f32_to_bf16(0.25f)});
  loom_testbench_value_deinitialize(&bf16);

  loom_testbench_value_t uniform = {};
  iree_hal_buffer_view_t* uniform_view =
      LookupBufferView(&table, case_plan.value_sources[7].value_id, &uniform);
  ASSERT_EQ(iree_hal_buffer_view_shape_rank(uniform_view), 1u);
  EXPECT_EQ(iree_hal_buffer_view_shape_dim(uniform_view, 0), 4);
  EXPECT_EQ(iree_hal_buffer_view_element_type(uniform_view),
            IREE_HAL_ELEMENT_TYPE_FLOAT_32);
  std::vector<float> uniform_contents(4);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      iree_hal_buffer_view_buffer(uniform_view), /*source_offset=*/0,
      uniform_contents.data(), uniform_contents.size() * sizeof(float)));
  for (float value : uniform_contents) {
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
  }
  loom_testbench_value_deinitialize(&uniform);

  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ValueMaterializerTest,
       MaterializesExplicitCartesianSamplePastRetainedPrefix) {
  loom_module_t* module = ParseModule(R"(
check.case @sweep {
  %m = check.param.range linear bounds(1 to 3) step(1) : i32
  %n = check.param.choice values([10, 20]) : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_plan_options_t plan_options = {};
  loom_testbench_plan_options_initialize(&plan_options);
  plan_options.max_samples_per_case = 3;
  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, &plan_options, &plan_arena_, &plan));

  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  EXPECT_EQ(case_plan.cartesian_sample_count, 6u);
  EXPECT_EQ(case_plan.sample_count, 3u);
  EXPECT_TRUE(case_plan.sample_count_truncated);

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t options = MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
      &options, &case_plan, /*sample_ordinal=*/5, &table));

  loom_testbench_value_t m = {};
  IREE_ASSERT_OK(loom_testbench_value_table_lookup_retain(
      &table, case_plan.parameters[0].value_id, &m));
  ASSERT_TRUE(loom_testbench_value_is_scalar(&m));
  EXPECT_EQ(m.scalar.kind, IREE_TOOLING_VALUE_KIND_I32);
  EXPECT_EQ(m.scalar.storage.i32, 3);
  loom_testbench_value_deinitialize(&m);

  loom_testbench_value_t n = {};
  IREE_ASSERT_OK(loom_testbench_value_table_lookup_retain(
      &table, case_plan.parameters[1].value_id, &n));
  ASSERT_TRUE(loom_testbench_value_is_scalar(&n));
  EXPECT_EQ(n.scalar.kind, IREE_TOOLING_VALUE_KIND_I32);
  EXPECT_EQ(n.scalar.storage.i32, 20);
  loom_testbench_value_deinitialize(&n);

  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ValueMaterializerTest, ReadsAndWritesNpyFiles) {
  loom_module_t* module = ParseModule(R"(
check.case @file_io {
  %file = check.file.read.npy path("single.npy") : tensor<3xf32>
  check.file.write.npy value(%file) path("actual.npy") mode(always) : tensor<3xf32>
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t options = MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
      &options, &case_plan, /*sample_ordinal=*/0, &table));

  ASSERT_EQ(case_plan.value_source_count, 1u);
  loom_testbench_value_t input = {};
  iree_hal_buffer_view_t* input_view =
      LookupBufferView(&table, case_plan.value_sources[0].value_id, &input);
  ExpectBufferViewContents<float>(
      input_view, {3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32, {1.1f, 2.2f, 3.3f});

  IREE_ASSERT_OK(loom_testbench_write_case_files(&options, &case_plan, &table,
                                                 /*case_failed=*/false));
  EXPECT_EQ(written_path_, "actual.npy");
  ASSERT_NE(written_stream_, nullptr);

  IREE_ASSERT_OK(
      iree_io_stream_seek(written_stream_, IREE_IO_STREAM_SEEK_SET, 0));
  iree_hal_buffer_view_t* written_buffer_view = nullptr;
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  IREE_ASSERT_OK(iree_numpy_npy_load_ndarray(
      written_stream_, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
      /*device=*/nullptr, device_allocator_, &written_buffer_view));
  ExpectBufferViewContents<float>(written_buffer_view, {3},
                                  IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                  {1.1f, 2.2f, 3.3f});
  iree_hal_buffer_view_release(written_buffer_view);

  loom_testbench_value_deinitialize(&input);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
