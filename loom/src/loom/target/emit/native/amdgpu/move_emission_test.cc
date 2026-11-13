// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/encoding.h"

namespace loom {
namespace {

class AmdgpuMoveEmissionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &module_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const auto* vtables = loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("move_emission"),
                                        &module_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_amdgpu_low_descriptor_registry_initialize(&registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &registry_.registry, IREE_SV("amdgpu.rdna4.gfx125x.core"));
    ASSERT_NE(descriptor_set_, nullptr);

    op_.kind = LOOM_OP_LOW_CONCAT;
    node_.op = &op_;
    node_.kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL;
    block_.block = &source_block_;
    block_.node_count = 1;
    block_.scheduled_node_count = 1;
    schedule_.module = module_;
    schedule_.target.descriptor_set = descriptor_set_;
    schedule_.blocks = &block_;
    schedule_.block_count = 1;
    schedule_.nodes = &node_;
    schedule_.node_count = 1;
    schedule_.scheduled_node_indices = &scheduled_node_index_;
    schedule_.scheduled_node_count = 1;

    move_.kind = LOOM_LOW_MOVE_KIND_EXCHANGE;
    move_.destination = {
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        .location = 255,
    };
    move_.source = {
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        .location = 768,
    };
    move_group_.source_ordinal = 0;
    move_group_.move_group.moves.count = 1;
    allocation_.target.descriptor_set = descriptor_set_;
    allocation_.packet_move_groups = &move_group_;
    allocation_.packet_move_group_count = 1;
    allocation_.moves = &move_;
    allocation_.move_count = 1;
    allocation_.packet_move_count = 1;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&module_pool_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  iree_arena_block_pool_t module_pool_;
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t source_block_ = {};
  loom_op_t op_ = {};
  loom_low_schedule_node_t node_ = {};
  loom_low_schedule_block_t block_ = {};
  uint32_t scheduled_node_index_ = 0;
  loom_low_schedule_table_t schedule_ = {};
  loom_low_move_t move_ = {};
  loom_low_allocation_packet_move_group_t move_group_ = {};
  loom_low_allocation_table_t allocation_ = {};
};

TEST_F(AmdgpuMoveEmissionTest, HighBankExchangeMatchesAssemblyAndEncoding) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  bool assembly_accepted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
      &schedule_, &allocation_, /*options=*/nullptr, &builder, &arena_,
      /*emitter=*/{}, &assembly_accepted));
  ASSERT_TRUE(assembly_accepted);
  const iree_string_view_t assembly = iree_string_builder_view(&builder);
  EXPECT_EQ(std::string(assembly.data, assembly.size),
            "  s_set_vgpr_msb 3\n"
            "  v_swap_b32 v255, v0\n"
            "  s_set_vgpr_msb 768\n");
  iree_string_builder_deinitialize(&builder);

  loom_amdgpu_encoded_instruction_stream_t stream = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &schedule_, &allocation_, /*options=*/nullptr, &arena_,
      /*emitter=*/{}, &accepted, &stream));
  ASSERT_TRUE(accepted);
  const iree_const_byte_span_t text = stream.text;
  ASSERT_EQ(text.data_length, 3 * sizeof(uint32_t));
  EXPECT_EQ(iree_unaligned_load_le_u32(text.data), UINT32_C(0xBF860003));
  EXPECT_EQ(iree_unaligned_load_le_u32(text.data + 4), UINT32_C(0x7FFECB00));
  EXPECT_EQ(iree_unaligned_load_le_u32(text.data + 8), UINT32_C(0xBF860300));
}

}  // namespace
}  // namespace loom
