// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

namespace loom {
namespace {

class CfgRoundtripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    iree_host_size_t count = 0;
    const auto* vtables = loom_cfg_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_CFG,
                                                 vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("cfg"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(decoded_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Shared allocation pool for source, wire encoding and decoded module.
  iree_arena_block_pool_t pool_;
  // Operation vocabulary used by the source and bytecode reader.
  loom_context_t context_;
  // Source module, whose physical block order must remain unchanged.
  loom_module_t* module_ = nullptr;
  // Independently decoded module owned by this fixture.
  loom_module_t* decoded_ = nullptr;
};

TEST_F(CfgRoundtripTest, LoopDefinitionsPrecedeUsesWithoutReorderingSource) {
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("iterate"), &name));
  loom_symbol_id_t symbol;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  const loom_type_t condition_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t value_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* function = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, {0, symbol}, &condition_type, 1, &value_type, 1,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function));
  loom_region_t* body = loom_test_func_body(function);
  loom_block_t* entry = loom_region_entry_block(body);
  loom_block_t* exit = nullptr;
  loom_block_t* backedge = nullptr;
  loom_block_t* header = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &exit));
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &backedge));
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &header));
  const std::array<loom_block_t*, 4> original = {entry, exit, backedge, header};
  loom_value_id_t carried;
  IREE_ASSERT_OK(loom_module_define_value(module_, value_type, &carried));
  IREE_ASSERT_OK(loom_block_add_arg(module_, header, carried));

  loom_builder_enter_region(&builder, function, body);
  loom_op_t* seed = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(1), value_type, LOOM_LOCATION_UNKNOWN, &seed));
  const loom_value_id_t initial = loom_test_constant_result(seed);
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder, header, &initial, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder, header);
  loom_op_t* increment = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder, carried, initial, value_type,
                                      LOOM_LOCATION_UNKNOWN, &increment));
  const loom_value_id_t next = loom_test_addi_result(increment);
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder, entry->arg_ids[0], backedge,
                                        exit, LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder, backedge);
  IREE_ASSERT_OK(loom_cfg_br_build(&builder, header, &next, 1,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder, exit);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &next, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, iree_allocator_system(), &stream));
  IREE_ASSERT_OK(loom_bytecode_write_module(module_, stream, nullptr, &pool_));
  std::vector<uint8_t> bytes(iree_io_stream_length(stream));
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  iree_io_stream_release(stream);
  for (uint16_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(body->blocks[i], original[i]);
    EXPECT_EQ(original[i]->region_index, i);
  }
  loom_bytecode_read_result_t result = {};
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("cfg.loombc"), &context_, &pool_, nullptr,
      iree_allocator_system(), &result, &decoded_));
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_NE(decoded_, nullptr);
  loom_op_t* decoded_function = loom_module_block(decoded_)->first_op;
  loom_block_t* decoded_entry =
      loom_region_entry_block(loom_test_func_body(decoded_function));
  loom_block_t* decoded_header = loom_cfg_br_dest(decoded_entry->last_op);
  loom_op_t* decoded_increment = decoded_header->first_op;
  ASSERT_TRUE(loom_test_addi_isa(decoded_increment));
  const loom_value_id_t decoded_next = loom_test_addi_result(decoded_increment);
  EXPECT_EQ(loom_test_addi_lhs(decoded_increment), decoded_header->arg_ids[0]);
  EXPECT_EQ(loom_test_addi_rhs(decoded_increment),
            loom_test_constant_result(decoded_entry->first_op));
  loom_block_t* decoded_backedge =
      loom_op_successors(decoded_header->last_op)[0];
  loom_block_t* decoded_exit = loom_op_successors(decoded_header->last_op)[1];
  EXPECT_EQ(loom_cfg_br_dest(decoded_backedge->last_op), decoded_header);
  EXPECT_EQ(loom_op_operands(decoded_backedge->last_op)[0], decoded_next);
  EXPECT_TRUE(loom_test_yield_isa(decoded_exit->last_op));
  EXPECT_EQ(loom_op_operands(decoded_exit->last_op)[0], decoded_next);
}

TEST_F(CfgRoundtripTest, PreservesBlockOperationCountsAbove16Bits) {
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("chain"), &name));
  loom_symbol_id_t symbol;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* function = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&builder, 0, 0, 0, {0, symbol}, &type, 1,
                                      &type, 1, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &function));
  loom_region_t* body = loom_test_func_body(function);
  loom_block_t* entry = loom_region_entry_block(body);
  loom_builder_enter_region(&builder, function, body);
  const loom_value_id_t argument = entry->arg_ids[0];
  loom_value_id_t value = argument;
  for (uint32_t i = 0; i < UINT16_MAX; ++i) {
    loom_op_t* add = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(&builder, value, argument, type,
                                        LOOM_LOCATION_UNKNOWN, &add));
    value = loom_test_addi_result(add);
  }
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &value, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield));
  ASSERT_EQ(entry->op_count, 65536u);

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, iree_allocator_system(), &stream));
  IREE_ASSERT_OK(loom_bytecode_write_module(module_, stream, nullptr, &pool_));
  std::vector<uint8_t> bytes(iree_io_stream_length(stream));
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  iree_io_stream_release(stream);
  loom_bytecode_read_result_t result = {};
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("chain.loombc"), &context_, &pool_, nullptr,
      iree_allocator_system(), &result, &decoded_));
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_NE(decoded_, nullptr);
  loom_op_t* decoded_function = loom_module_block(decoded_)->first_op;
  loom_block_t* decoded_entry =
      loom_region_entry_block(loom_test_func_body(decoded_function));
  ASSERT_EQ(decoded_entry->op_count, entry->op_count);
  value = decoded_entry->arg_ids[0];
  for (loom_op_t* op = decoded_entry->first_op; op != decoded_entry->last_op;
       op = op->next_op) {
    ASSERT_TRUE(loom_test_addi_isa(op));
    EXPECT_EQ(loom_test_addi_lhs(op), value);
    EXPECT_EQ(loom_test_addi_rhs(op), decoded_entry->arg_ids[0]);
    value = loom_test_addi_result(op);
  }
  ASSERT_TRUE(loom_test_yield_isa(decoded_entry->last_op));
  EXPECT_EQ(loom_op_operands(decoded_entry->last_op)[0], value);
}

}  // namespace
}  // namespace loom
