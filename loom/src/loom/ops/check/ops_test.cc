// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/check/ops.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class CheckOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_check_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_CHECK,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("checks"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    case_ref_ = Symbol("case");
    loom_op_t* case_op = nullptr;
    IREE_ASSERT_OK(loom_check_case_build(&builder_, 0, 0, case_ref_,
                                         LOOM_LOCATION_UNKNOWN, &case_op));
    auto saved = loom_builder_enter_region(&builder_, case_op,
                                           loom_check_case_body(case_op));
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(
        loom_check_return_build(&builder_, LOOM_LOCATION_UNKNOWN, &return_op));
    loom_builder_restore(&builder_, saved);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {.module_id = 0, .symbol_id = symbol_id};
  }

  loom_op_t* Benchmark(loom_symbol_ref_t name) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_check_benchmark_build(&builder_, 0, case_ref_, name, {},
                                             LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  uint32_t Verify(const loom_module_t* module) {
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, nullptr, &result));
    return result.error_count;
  }

  std::vector<uint8_t> WriteBytecode() {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module_, stream, nullptr, &pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  // Module and bytecode materialization storage.
  iree_arena_block_pool_t pool_ = {};
  // Production check dialect metadata.
  loom_context_t context_ = {};
  // Owned native-builder module.
  loom_module_t* module_ = nullptr;
  // Builder inserting benchmark records at module scope.
  loom_builder_t builder_ = {};
  // Named case shared by the benchmark records.
  loom_symbol_ref_t case_ref_ = loom_symbol_ref_null();
};

TEST_F(CheckOpsTest, RejectsAbsentBenchmarkSymbol) {
  auto* benchmark = Benchmark(loom_symbol_ref_null());
  // A native producer may omit a required attribute.
  IREE_ASSERT_OK(loom_check_benchmark_set_benchmark(module_, benchmark,
                                                    loom_attr_absent()));
  EXPECT_EQ(Verify(module_), 1u);
}

TEST_F(CheckOpsTest, RejectsNullBenchmarkSymbol) {
  Benchmark(loom_symbol_ref_null());
  EXPECT_EQ(Verify(module_), 1u);
}

TEST_F(CheckOpsTest, NamedBenchmarkBytecodeRoundTrip) {
  Benchmark(Symbol("latency"));
  EXPECT_EQ(Verify(module_), 0u);
  auto bytes = WriteBytecode();
  loom_bytecode_read_result_t result = {};
  loom_module_t* loaded = nullptr;
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("checks.loombc"), &context_, &pool_, nullptr, &result, &loaded,
      iree_allocator_system()));
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(Verify(loaded), 0u);
  EXPECT_EQ(loaded->symbols.count, 2u);
  loom_op_t* benchmark = loom_module_block(loaded)->last_op;
  ASSERT_TRUE(loom_check_benchmark_isa(benchmark));
  auto ref = loom_check_benchmark_benchmark(benchmark);
  EXPECT_EQ(loaded->symbols.entries[ref.symbol_id].defining_op, benchmark);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&loaded->strings,
                            loaded->symbols.entries[ref.symbol_id].name_id),
      IREE_SV("latency")));
  auto case_ref = loom_check_benchmark_case_ref(benchmark);
  EXPECT_EQ(loaded->symbols.entries[case_ref.symbol_id].defining_op,
            loom_module_block(loaded)->first_op);
  loom_module_free(loaded);
}

}  // namespace
}  // namespace loom
