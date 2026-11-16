// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/worker.h"

#include <bitset>
#include <tuple>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"

namespace {

class ControlWriteTest
    : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &registry_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&registry_));
    iree_host_size_t count;
    const auto* tables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&registry_, LOOM_DIALECT_LOW,
                                                 tables, count));
    IREE_ASSERT_OK(loom_context_finalize(&registry_));
    IREE_ASSERT_OK(loom_module_allocate(&registry_, IREE_SV("control_write"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &code_));
    realization_.worker_versions = versions_;
    code_.realization = &realization_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&registry_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  uint32_t Integer(loom_value_id_t value) {
    const auto* op = loom_value_def_op(loom_module_value(module_, value));
    const auto attrs = loom_low_const_attrs(op);
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (attrs.entries[i].name_id == code_.integer_name) {
        return static_cast<uint32_t>(attrs.entries[i].value.i64);
      }
    }
    ADD_FAILURE() << "missing integer attribute";
    return 0;
  }

  bool Is(const loom_op_t* op, uint32_t descriptor) {
    return loom_low_op_isa(op) &&
           loom_low_op_descriptor(op) ==
               loom_low_descriptor_set_descriptor_ordinal(
                   code_.descriptors,
                   &code_.descriptors->descriptors[descriptor]);
  }

  // Backing storage for the emitted module and transient native state.
  iree_arena_block_pool_t blocks_;
  // Transient native emission allocations.
  iree_arena_allocator_t arena_;
  // Ordinary Low operation registry.
  loom_context_t registry_;
  // Owns the emitted callable and values.
  loom_module_t* module_ = nullptr;
  // Production native worker emitter.
  loom_aie2p_worker_builder_t code_ = {};
  // Standalone helper emission has no optional function version.
  loom_pipeline_realization_t realization_ = {};
  // One absent function-version owner.
  loom_target_function_version_t* versions_[1] = {};
};

TEST_P(ControlWriteTest, ConsecutiveRegistersPreservePacketBoundaries) {
  const auto [first_word, count] = GetParam();
  const uint32_t address = 0x1d000u + first_word * 4;
  loom_builder_t builder;
  loom_symbol_ref_t symbol;
  loom_op_t* function;
  IREE_ASSERT_OK(loom_aie2p_worker_helper(&code_, nullptr, 0, nullptr, 0,
                                          &builder, &symbol, &function));
  loom_value_id_t values[4];
  for (uint32_t i = 0; i < count; ++i) {
    IREE_ASSERT_OK(
        loom_aie2p_worker_constant(&code_, &builder, 101 + i, &values[i]));
  }
  IREE_ASSERT_OK(loom_aie2p_worker_control_write(&code_, &builder, 17, address,
                                                 values, count));
  IREE_ASSERT_OK(loom_aie2p_worker_return(&builder, nullptr, 0));

  std::vector<std::pair<uint32_t, bool>> stream;
  for (const auto* op =
           loom_region_entry_block(loom_low_func_def_body(function))->first_op;
       op; op = op->next_op) {
    const bool last = Is(op, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32);
    if (last || Is(op, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32)) {
      stream.emplace_back(Integer(loom_op_operands(op)[0]), last);
    }
  }
  uint32_t consumed = 0;
  for (size_t cursor = 0; cursor < stream.size();) {
    ASSERT_LE(cursor + 2, stream.size());
    const auto [route, route_last] = stream[cursor++];
    const auto [command, command_last] = stream[cursor++];
    EXPECT_EQ(route & 0x7fffffffu, 17u);
    EXPECT_EQ(std::bitset<32>(route).count() % 2, 1u);
    EXPECT_FALSE(route_last);
    EXPECT_FALSE(command_last);
    EXPECT_EQ(std::bitset<32>(command).count() % 2, 1u);
    EXPECT_EQ((command >> 22) & 0x1ffu, 0u);
    const uint32_t packet_address = command & 0xfffffu;
    const uint32_t packet_count = ((command >> 20) & 3) + 1;
    EXPECT_EQ(packet_address, address + consumed * 4);
    EXPECT_LE((packet_address & 15) + packet_count * 4, 16u);
    ASSERT_LE(cursor + packet_count, stream.size());
    for (uint32_t i = 0; i < packet_count; ++i) {
      const auto [value, last] = stream[cursor++];
      EXPECT_EQ(value, 101u + consumed++);
      EXPECT_EQ(last, i + 1 == packet_count);
    }
  }
  EXPECT_EQ(consumed, count);
}

INSTANTIATE_TEST_SUITE_P(
    WordOffsetAndLength, ControlWriteTest,
    ::testing::Combine(::testing::Values<uint32_t>(0, 1, 2, 3),
                       ::testing::Values<uint32_t>(1, 2, 3, 4)));

}  // namespace
