// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/spill.h"

#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/register_parts.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/stack_frame.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class Aie2pSpillTest : public ::testing::TestWithParam<uint64_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("spill"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  struct Piece {
    // Byte offset from the start of the complete slot.
    uint32_t offset;
    // Parts read/written by this transfer; zero means the complete carrier.
    uint32_t mask;
    // Number of bytes transferred by the memory instruction.
    uint32_t bytes;
    // Logical chunk produced by this transfer.
    uint32_t chunk = 0;
  };

  void CheckTransfers(uint16_t class_id, uint32_t units, uint32_t bytes,
                      uint32_t alignment, uint32_t defined_parts,
                      std::initializer_list<Piece> expected,
                      uint32_t narrow_offset = 0, uint32_t narrow_units = 0) {
    const auto* descriptors = loom_aie2p_core_descriptor_set();
    const auto type =
        loom_low_register_type(descriptors->stable_id, class_id, units);
    // The transfer API consumes the selected value's type and retained parts;
    // it does not inspect its defining packet. Native reloads below provide
    // real partial values to the store callback.
    loom_value_id_t value;
    IREE_ASSERT_OK(loom_module_define_value(module_, type, &value));
    loom_low_allocation_table_t table = {
        .module = module_,
        .target = {.descriptor_set = descriptors},
    };
    loom_low_storage_layout_builder_t storage;
    loom_low_storage_layout_builder_initialize(&storage);
    loom_op_t* anchor = nullptr;
    IREE_ASSERT_OK(loom_low_storage_reserve_build(
        &builder_, GetParam() + 4, 4,
        loom_type_storage(LOOM_STORAGE_SPACE_STACK), LOOM_LOCATION_UNKNOWN,
        &anchor));
    IREE_ASSERT_OK(loom_low_storage_layout_builder_append(module_, anchor,
                                                          &arena_, &storage));
    IREE_ASSERT_OK(loom_low_storage_layout_builder_finish(
        &storage, &arena_, &table.storage_layout));
    const loom_low_allocation_spill_plan_t spill = {
        .value_id = value,
        .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK,
        .byte_size = bytes,
        .byte_alignment = alignment,
        .defined_parts = defined_parts,
    };
    loom_low_spill_materialization_action_t actions[] = {
        {.kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD},
        {.kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE},
    };
    if (narrow_units) {
      const auto narrow_type = loom_low_register_type(descriptors->stable_id,
                                                      class_id, narrow_units);
      IREE_ASSERT_OK(loom_low_slice_build(&builder_, value, narrow_offset,
                                          narrow_type, LOOM_LOCATION_UNKNOWN,
                                          &anchor));
      actions[0] = {
          .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD,
          .value =
              {
                  .narrow_slice_reload =
                      {
                          .slice_op = anchor,
                          .unit_offset = narrow_offset,
                      },
              },
      };
    }
    const loom_low_spill_materialization_instruction_t instructions[] = {
        {.action_index_and_after = 0}, {.action_index_and_after = 1}};
    const loom_low_spill_materialization_record_t record = {
        .store_count = 1,
        .store_bytes = bytes,
        .reload_count = 1,
        .reload_bytes = bytes,
    };
    const loom_low_spill_materialization_plan_t materialization = {
        .actions = actions,
        .action_count = 2,
        .instructions = instructions,
        .instruction_count = 2,
        .records = &record,
        .record_count = 1,
    };
    loom_aie2p_stack_layout_t layout = {.arena = &arena_};
    loom_low_spill_materialization_traffic_plan_t plan = {};
    IREE_ASSERT_OK(loom_aie2p_spill_plan(&layout, &table, &spill, 1,
                                         &materialization, &arena_, &plan));
    const uint64_t depth =
        (GetParam() + 4 + bytes + alignment - 1) & ~(uint64_t)(alignment - 1);
    EXPECT_EQ(layout.byte_length, depth);
    for (uint32_t action = 0; action < (narrow_units ? 1u : 2u); ++action) {
      SCOPED_TRACE(action);
      anchor = loom_module_block(module_)->last_op;
      loom_low_spill_materialized_action_t emitted = {};
      IREE_ASSERT_OK(plan.emit(plan.user_data, action, &builder_,
                               LOOM_VALUE_ID_INVALID, value,
                               LOOM_LOCATION_UNKNOWN, &arena_, &emitted));
      auto expected_piece = expected.begin();
      loom_value_id_t previous = LOOM_VALUE_ID_INVALID;
      uint32_t previous_chunk = UINT32_MAX;
      int64_t far_displacement = 0;
      for (auto* op = anchor->next_op; op != nullptr; op = op->next_op) {
        if (loom_low_const_isa(op)) {
          far_displacement =
              loom_attr_as_i64(loom_low_const_attrs(op).entries[0].value);
          continue;
        }
        if (!loom_low_op_isa(op)) {
          continue;
        }
        const auto* descriptor =
            &descriptors->descriptors[loom_low_op_descriptor(op)];
        const auto memory =
            loom_low_descriptor_memory_effect_summary(descriptors, descriptor);
        if (!memory.read_byte_count && !memory.write_byte_count) {
          continue;
        }
        ASSERT_NE(expected_piece, expected.end());
        const auto& part = *expected_piece++;
        EXPECT_EQ(memory.read_byte_count, action == 0 ? part.bytes : 0);
        EXPECT_EQ(memory.write_byte_count, action == 1 ? part.bytes : 0);
        const auto immediate =
            loom_attr_as_i64(loom_low_op_attrs(op).entries[0].value);
        EXPECT_EQ(immediate == 0 ? far_displacement : immediate,
                  -(int64_t)depth + part.offset);
        if (GetParam() == 0) {
          EXPECT_NE(immediate, 0);
        }
        if (GetParam() == 32768) {
          EXPECT_EQ(immediate, 0);
        }
        const auto* operand =
            loom_low_descriptor_operand(descriptors, descriptor, 0);
        const auto actual_type = loom_module_value_type(
            module_, action == 0 ? loom_low_op_results(op).values[0]
                                 : loom_low_op_operands(op).values[0]);
        const auto actual_class = loom_low_register_type_class_id(actual_type);
        EXPECT_EQ(loom_low_register_parts_operand_mask(descriptors, operand,
                                                       actual_class),
                  part.mask ? part.mask
                            : descriptors->reg_classes[actual_class]
                                  .full_register_part_mask);
        if (action == 0) {
          const bool continuation = part.chunk == previous_chunk;
          ASSERT_EQ(op->tied_result_count, continuation ? 1 : 0);
          if (continuation) {
            const auto source_index = loom_op_tied_results(op)[0].operand_index;
            EXPECT_EQ(loom_low_op_operands(op).values[source_index], previous);
          }
          previous = loom_low_op_results(op).values[0];
          previous_chunk = part.chunk;
        }
      }
      EXPECT_EQ(expected_piece, expected.end());
      if (action == 0) {
        value = emitted.value_id;
        EXPECT_EQ(loom_low_register_type_class_id(
                      loom_module_value_type(module_, value)),
                  class_id);
        EXPECT_EQ(loom_low_register_type_unit_count(
                      loom_module_value_type(module_, value)),
                  narrow_units ? narrow_units : units);
      }
    }
  }

  // Pool backing the IR and transient plan arenas.
  iree_arena_block_pool_t pool_;
  // Callback and stack-layout lifetime for one materialization batch.
  iree_arena_allocator_t arena_;
  // Low operation registry used by native traffic construction.
  loom_context_t context_;
  // Module owning the instruction fragments and register types.
  loom_module_t* module_ = nullptr;
  // Insertion point for the planned traffic.
  loom_builder_t builder_;
};

TEST_P(Aie2pSpillTest, PreservesTupleWordsAndNarrowOffsetsAtEncodingBoundary) {
  const auto* descriptors = loom_aie2p_core_descriptor_set();
  loom_type_t word_type;
  loom_type_t pair_type;
  IREE_ASSERT_OK(loom_low_build_register_type(
      descriptors, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, &word_type));
  IREE_ASSERT_OK(loom_low_build_register_type(
      descriptors, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 2, &pair_type));
  loom_value_id_t pair;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_), pair_type, &pair));
  loom_op_t* slice = nullptr;
  IREE_ASSERT_OK(loom_low_slice_build(&builder_, pair, 1, word_type,
                                      LOOM_LOCATION_UNKNOWN, &slice));
  loom_low_allocation_table_t table = {
      .module = module_,
      .target = {.descriptor_set = descriptors},
  };
  loom_low_storage_layout_builder_t storage_builder;
  loom_low_storage_layout_builder_initialize(&storage_builder);
  if (GetParam() != 0) {
    loom_op_t* prefix = nullptr;
    IREE_ASSERT_OK(loom_low_storage_reserve_build(
        &builder_, GetParam(), 4, loom_type_storage(LOOM_STORAGE_SPACE_STACK),
        LOOM_LOCATION_UNKNOWN, &prefix));
    IREE_ASSERT_OK(loom_low_storage_layout_builder_append(
        module_, prefix, &arena_, &storage_builder));
  }
  IREE_ASSERT_OK(loom_low_storage_layout_builder_finish(
      &storage_builder, &arena_, &table.storage_layout));
  const loom_low_allocation_spill_plan_t spill = {
      .value_id = pair,
      .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK,
      .byte_size = 8,
      .byte_alignment = 4,
  };
  const loom_low_spill_materialization_action_t actions[] = {
      {.kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE},
      {.kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD},
      {
          .kind = LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD,
          .value =
              {
                  .narrow_slice_reload =
                      {
                          .slice_op = slice,
                          .unit_offset = 1,
                      },
              },
      },
  };
  const loom_low_spill_materialization_instruction_t instructions[] = {
      {.action_index_and_after = 0},
      {.action_index_and_after = 1},
      {.action_index_and_after = 2},
  };
  const loom_low_spill_materialization_record_t record = {
      .value_is_block_argument = true,
      .store_count = 1,
      .store_bytes = 8,
      .reload_count = 2,
      .reload_bytes = 12,
  };
  const loom_low_spill_materialization_plan_t materialization = {
      .actions = actions,
      .action_count = IREE_ARRAYSIZE(actions),
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .records = &record,
      .record_count = 1,
  };
  loom_aie2p_stack_layout_t layout = {.arena = &arena_};
  loom_low_spill_materialization_traffic_plan_t plan = {};
  IREE_ASSERT_OK(loom_aie2p_spill_plan(&layout, &table, &spill, 1,
                                       &materialization, &arena_, &plan));
  EXPECT_EQ(layout.byte_length, GetParam() + 8);
  for (size_t action = 0; action < 3; ++action) {
    SCOPED_TRACE(action);
    loom_op_t* anchor = loom_module_block(module_)->last_op;
    loom_low_spill_materialized_action_t emitted = {};
    IREE_ASSERT_OK(plan.emit(plan.user_data, action, &builder_,
                             LOOM_VALUE_ID_INVALID, pair, LOOM_LOCATION_UNKNOWN,
                             &arena_, &emitted));
    loom_op_t* op = anchor->next_op;
    const bool store = action == 0;
    const uint32_t word_count = action == 2 ? 1 : 2;
    const int64_t start = -(int64_t)(GetParam() + 8) + (action == 2 ? 4 : 0);
    uint32_t far_count = 0;
    loom_value_id_t words[2] = {};
    for (uint32_t word = 0; word < word_count; ++word) {
      SCOPED_TRACE(word);
      loom_value_id_t source = LOOM_VALUE_ID_INVALID;
      if (store) {
        ASSERT_NE(op, nullptr);
        ASSERT_TRUE(loom_low_slice_isa(op));
        EXPECT_EQ(loom_low_slice_source(op), pair);
        EXPECT_EQ(loom_low_slice_offset(op), word);
        source = loom_low_slice_result(op);
        op = op->next_op;
      }
      const bool uses_far_address = start + word * 4 < -2048;
      far_count += uses_far_address;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      if (uses_far_address) {
        ASSERT_NE(op, nullptr);
        ASSERT_TRUE(loom_low_op_isa(op));
        ASSERT_EQ(loom_low_op_descriptor(op),
                  AIE2P_CORE_DESCRIPTOR_REF_STACK_ADDRESS);
        const auto base = loom_low_op_results(op).values[0];
        op = op->next_op;
        ASSERT_NE(op, nullptr);
        ASSERT_TRUE(loom_low_const_isa(op));
        ASSERT_EQ(loom_low_const_descriptor(op),
                  AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_MODIFIER_MOVXM);
        EXPECT_EQ(loom_attr_as_i64(loom_low_const_attrs(op).entries[0].value),
                  start + word * 4);
        const auto modifier = loom_low_const_result(op);
        op = op->next_op;
        ASSERT_NE(op, nullptr);
        EXPECT_EQ(loom_low_op_descriptor(op),
                  AIE2P_CORE_DESCRIPTOR_REF_ADDRESS_ADD_A_REGISTER);
        EXPECT_EQ(loom_low_op_operands(op).values[0], base);
        EXPECT_EQ(loom_low_op_operands(op).values[1], modifier);
        ASSERT_EQ(op->tied_result_count, 1u);
        EXPECT_EQ(loom_op_tied_results(op)[0].operand_index, 0u);
        address = loom_low_op_results(op).values[0];
        op = op->next_op;
      }
      ASSERT_NE(op, nullptr);
      EXPECT_EQ(
          loom_low_op_descriptor(op),
          uses_far_address
              ? (store
                     ? AIE2P_CORE_DESCRIPTOR_REF_STORE_SCALAR_I32_INDEXED_IMMEDIATE
                     : AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_I32_INDEXED_IMMEDIATE)
              : (store ? AIE2P_CORE_DESCRIPTOR_REF_STACK_STORE_ER
                       : AIE2P_CORE_DESCRIPTOR_REF_STACK_LOAD_ER));
      EXPECT_EQ(loom_attr_as_i64(loom_low_op_attrs(op).entries[0].value),
                uses_far_address ? 0 : start + word * 4);
      if (store) {
        EXPECT_EQ(loom_low_op_operands(op).values[0], source);
      }
      if (uses_far_address) {
        EXPECT_EQ(loom_low_op_operands(op).values[store ? 1 : 0], address);
      }
      if (!store) {
        words[word] = loom_low_op_results(op).values[0];
      }
      op = op->next_op;
    }
    if (action == 1) {
      ASSERT_NE(op, nullptr);
      ASSERT_TRUE(loom_low_concat_isa(op));
      EXPECT_EQ(loom_op_operands(op)[0], words[0]);
      EXPECT_EQ(loom_op_operands(op)[1], words[1]);
      EXPECT_EQ(emitted.value_id, loom_low_concat_result(op));
      op = op->next_op;
    } else {
      EXPECT_EQ(emitted.value_id, store ? LOOM_VALUE_ID_INVALID : words[0]);
    }
    EXPECT_EQ(op, nullptr);
    EXPECT_EQ(emitted.emitted_value_count,
              word_count + far_count * 3 + (action == 1 ? 1 : 0));
  }
}

TEST_P(Aie2pSpillTest, NativeCarrierFamilies) {
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ES, 1, 4, 4, 0, {{0, 0, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_EDJ, 1, 4, 4, 0, {{0, 0, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 4, 128, 64, 0,
                 {{0, 0, 64}, {64, 0, 64, 1}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE, 1, 8, 4, 0,
                 {{0, 1, 4}, {4, 2, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ML8M, 1, 8, 4, 0,
                 {{0, 1, 4}, {4, 2, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ED, 1, 16, 4, 0,
                 {{0, 1, 4}, {4, 2, 4}, {8, 4, 4}, {12, 8, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_EDS, 1, 32, 4, 0,
                 {{0, 1, 4},
                  {4, 2, 4},
                  {8, 4, 4},
                  {12, 8, 4},
                  {16, 16, 4},
                  {20, 32, 4},
                  {24, 64, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ELDFIFOREG, 1, 128, 64, 0,
                 {{0, 1, 64}, {64, 2, 64}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_MSTFIFO, 1, 128, 64, 0,
                 {{0, 1, 64}, {64, 2, 64}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_EDM, 1, 256, 64, 0,
                 {{0, 1, 64}, {64, 2, 64}, {128, 4, 64}, {192, 8, 64}});
}

TEST_P(Aie2pSpillTest, PreservesOnlyDefinedPartsIncludingNonPrefixSubsets) {
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ELDFIFOREG, 1, 128, 64, 1,
                 {{0, 1, 64}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_ELDFIFOREG, 1, 128, 64, 2,
                 {{64, 2, 64}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_EDS, 1, 32, 4, 0x51,
                 {{0, 1, 4}, {16, 16, 4}, {24, 64, 4}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_EDM, 1, 256, 64, 0xA,
                 {{64, 2, 64}, {192, 8, 64}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_MEXA, 1, 72, 64, 0x5,
                 {{0, 1, 64}, {68, 4, 4}});
}

TEST_P(Aie2pSpillTest, PackedPlanesAndNarrowTupleSlices) {
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_MEXA, 2, 144, 64, 0,
                 {{0, 1, 64},
                  {128, 2, 4},
                  {132, 4, 4},
                  {64, 1, 64, 1},
                  {136, 2, 4, 1},
                  {140, 4, 4, 1}});
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_MEXA, 2, 144, 64, 0,
                 {{64, 1, 64}, {136, 2, 4}, {140, 4, 4}}, 1, 1);
  CheckTransfers(AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 4, 128, 64, 0,
                 {{32, 0, 32}, {64, 0, 32, 1}}, 1, 2);
}

INSTANTIATE_TEST_SUITE_P(StackDepths, Aie2pSpillTest,
                         ::testing::Values(0, 2040, 2044, 2048, 32768));

}  // namespace
}  // namespace loom
