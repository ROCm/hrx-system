// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/function.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/emit/native/x86/encoding.h"
#include "loom/target/emit/native/x86/transport.h"

namespace {

loom_x86_instruction_t Instruction(loom_x86_encoding_form_t form,
                                   uint16_t encoding_id = 0,
                                   loom_x86_encoding_operands_t operands = {},
                                   uint32_t reference = UINT32_MAX) {
  return {operands, reference, static_cast<uint16_t>(form), encoding_id};
}

// Encoding has independent bitfield tests. These tests isolate the writer's
// composition and layout contract from individual instruction encodings.
std::string Encode(const loom_x86_instruction_t& instruction) {
  loom_x86_encoded_instruction_t encoded;
  loom_x86_encode_instruction(instruction.encoding_format_id,
                              instruction.encoding_id, &instruction.operands,
                              &encoded);
  return std::string(reinterpret_cast<const char*>(encoded.bytes),
                     encoded.length);
}

class FunctionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        1024, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    iree_io_stream_release(stream_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  std::string Read() {
    std::string bytes(static_cast<size_t>(iree_io_stream_length(stream_)),
                      '\0');
    IREE_CHECK_OK(iree_io_stream_seek(stream_, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream_, bytes.size(), bytes.data(), nullptr));
    return bytes;
  }

  // Backing storage for the writer's temporary layout records.
  iree_arena_block_pool_t pool_;
  // Per-test writer scratch.
  iree_arena_allocator_t arena_;
  // Owned output with the seek contract required for branch fixups.
  iree_io_stream_t* stream_ = nullptr;
};

TEST(X86LoopPlacementCostTest, AdmitsOnlySingleScalarInstruction) {
  loom_low_loop_placement_cost_t cost = {};
  const loom_low_descriptor_layout_t layouts[] = {
      {.encoding_format_id = LOOM_X86_ENCODING_FORM_UNARY_R_RM},
      {.encoding_format_id = LOOM_X86_ENCODING_FORM_SELECT},
      {.encoding_format_id = LOOM_X86_ENCODING_FORMAT_VECTOR},
      {},
  };
  loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = layouts,
      .descriptor_layout_count = IREE_ARRAYSIZE(layouts),
  };
  const loom_low_descriptor_t scalar = {
      .encoding_id = 0x89,
  };
  EXPECT_TRUE(
      loom_x86_loop_placement_cost_query(&descriptor_set, &scalar, &cost));
  EXPECT_EQ(cost.maximum_execution_multiplier, 8u);
  EXPECT_EQ(cost.maximum_schedule_cycle_count, 2u);

  const loom_low_descriptor_t expanded = {
      .encoding_id = 0x44,
      .layout_index = 1,
  };
  EXPECT_FALSE(
      loom_x86_loop_placement_cost_query(&descriptor_set, &expanded, &cost));

  const loom_low_descriptor_t vector = {
      .layout_index = 2,
  };
  EXPECT_FALSE(
      loom_x86_loop_placement_cost_query(&descriptor_set, &vector, &cost));

  const loom_low_descriptor_t unavailable = {.layout_index = 3};
  EXPECT_FALSE(
      loom_x86_loop_placement_cost_query(&descriptor_set, &unavailable, &cost));
}

TEST_F(FunctionTest, EmptyLeafNeedsOnlyReturn) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {
      .block_starts = block_starts,
      .block_count = 1,
  };

  uint64_t instruction_count = 0;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, &instruction_count));
  EXPECT_EQ(instruction_count, 1u);
  EXPECT_EQ(Read(), Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST(FunctionConstructionTest, PreservesOnlyWrittenCalleeSavedGprs) {
  constexpr uint8_t registers[] = {0, 1,  2,  3,  5,  6,  7, 8,
                                   9, 10, 11, 12, 13, 14, 15};
  for (uint16_t width : {uint16_t{0}, uint16_t{LOOM_X86_ENCODING_REX_W}}) {
    for (uint8_t reg : registers) {
      SCOPED_TRACE(::testing::Message()
                   << "register=" << +reg << " width=" << width);
      loom_x86_instruction_t instructions[1] = {};
      loom_x86_function_t function = {.instructions = instructions};
      loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_CONSTANT,
                               0xb8 | width, {.immediate = 42, .result = reg},
                               UINT32_MAX);
      EXPECT_EQ(function.instruction_count, 1u);
      const bool preserved = reg == 3 || reg == 5 || reg >= 12;
      EXPECT_EQ(function.saved_registers, preserved ? 1u << reg : 0);
      EXPECT_FALSE(function.may_dirty_upper_vector_state);
      EXPECT_EQ(instructions[0].operands.result, reg);
    }
  }
}

TEST(FunctionConstructionTest, AccumulatesWritesButNotReadsOrVectorOrdinals) {
  loom_x86_instruction_t instructions[4] = {};
  loom_x86_function_t function = {.instructions = instructions};
  loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_COMPARE, 0x39,
                           {.inputs = {3, 12}}, UINT32_MAX);
  EXPECT_EQ(function.saved_registers, 0);
  // A packed vector result numbered 15 does not name the GPR R15.
  loom_x86_function_append(&function, 0xc1c0, 0x096f,
                           {.result = 15, .inputs = {7}}, UINT32_MAX);
  EXPECT_EQ(function.saved_registers, 0);
  loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_CONSTANT, 0xb8,
                           {.immediate = 42, .result = 3}, UINT32_MAX);
  loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_CONSTANT, 0xb8,
                           {.immediate = 43, .result = 12}, UINT32_MAX);
  EXPECT_EQ(function.instruction_count, 4u);
  EXPECT_EQ(function.saved_registers, (1u << 3) | (1u << 12));
}

TEST(FunctionConstructionTest, RetainsSelectedTransportEffects) {
  struct TransferCase {
    // Register class of the destination.
    uint16_t destination_class;
    // Physical allocation ordinal within the destination class.
    uint32_t destination;
    // Register class of the source.
    uint16_t source_class;
    // Physical allocation ordinal within the source class.
    uint32_t source;
    // Callee-preserved architectural GPR writes.
    uint16_t saved_registers;
    // Whether the transfer can dirty the low sixteen YMM/ZMM registers.
    bool dirty_upper_state;
  };
  const TransferCase cases[] = {
      {LOOM_X86_REGISTER_CLASS_GPR64, 12, LOOM_X86_REGISTER_CLASS_GPR64, 7,
       1u << 12, false},
      {LOOM_X86_REGISTER_CLASS_GPR32, 3, LOOM_X86_REGISTER_CLASS_GPR64, 7,
       1u << 3, false},
      {LOOM_X86_REGISTER_CLASS_GPR64, 12, LOOM_X86_REGISTER_CLASS_GPR64, 12, 0,
       false},
      {LOOM_X86_REGISTER_CLASS_XMM, 15, LOOM_X86_REGISTER_CLASS_XMM, 9, 0,
       false},
      {LOOM_X86_REGISTER_CLASS_YMM, 12, LOOM_X86_REGISTER_CLASS_YMM, 11, 0,
       true},
      {LOOM_X86_REGISTER_CLASS_ZMM, 18, LOOM_X86_REGISTER_CLASS_ZMM, 17, 0,
       false},
      {LOOM_X86_REGISTER_CLASS_K, 0, LOOM_X86_REGISTER_CLASS_K, 7, 0, false},
      {LOOM_X86_REGISTER_CLASS_GPR64, 12, LOOM_X86_REGISTER_CLASS_ZMM, 17,
       1u << 12, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "destination class=" << test_case.destination_class
                 << " source class=" << test_case.source_class);
    loom_x86_transport_instruction_t transfer;
    ASSERT_TRUE(loom_x86_transport_select_register(
        test_case.destination_class, test_case.destination,
        test_case.source_class, test_case.source, &transfer));
    loom_x86_instruction_t instructions[2] = {};
    loom_x86_function_t function = {.instructions = instructions};
    loom_x86_function_append_transport(&function, &transfer);
    EXPECT_EQ(function.instruction_count,
              transfer.encoding_format_id == 0 ? 0u : 1u);
    EXPECT_EQ(function.saved_registers, test_case.saved_registers);
    EXPECT_EQ(function.may_dirty_upper_vector_state,
              test_case.dirty_upper_state);
    // Appending a read-only scalar instruction preserves accumulated effects.
    loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_COMPARE, 0x39,
                             {.inputs = {3, 12}}, UINT32_MAX);
    EXPECT_EQ(function.saved_registers, test_case.saved_registers);
    EXPECT_EQ(function.may_dirty_upper_vector_state,
              test_case.dirty_upper_state);
  }
}

TEST(FunctionStackLayoutTest, PaddingAndRestoration) {
  struct LayoutCase {
    // Admitted storage before ABI padding.
    uint32_t storage_byte_length;
    // Strongest storage or call alignment requirement.
    uint32_t storage_alignment;
    // Architectural callee-preserved GPRs written by the function.
    uint16_t saved_registers;
    // Concrete stack adjustments consumed by the native writer.
    loom_x86_stack_frame_t expected;
  };
  const LayoutCase cases[] = {
      {0, 0, 0, {}},
      {0, 0, 1u << 3, {}},
      {0, 8, 0, {0, 8, {}}},
      {16, 1, 0, {16, 8, {}}},
      {8, 16, 0, {8, 16, {}}},
      {8, 16, 1u << 3, {16, 16, {}}},
      {8, 16, (1u << 3) | (1u << 12), {8, 16, {}}},
      {0, 16, 0, {8, 16, {}}},
      {0, 16, 1u << 3, {0, 16, {}}},
      {8, 64, 1u << 3, {64, 64, {-64, 8, 11}}},
      {17, 32, 0, {32, 32, {-32, 24, 11}}},
      {32, 32, 0, {64, 32, {-32, 32, 11}}},
      {0, 64, 0, {64, 64, {-64, 0, 11}}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "storage=" << test_case.storage_byte_length
                 << " alignment=" << test_case.storage_alignment
                 << " saved=" << test_case.saved_registers);
    loom_x86_stack_frame_t stack = {};
    ASSERT_TRUE(loom_x86_function_layout_stack(
        test_case.storage_byte_length, test_case.storage_alignment,
        test_case.saved_registers, &stack));
    EXPECT_EQ(stack.allocation_size, test_case.expected.allocation_size);
    EXPECT_EQ(stack.alignment, test_case.expected.alignment);
    EXPECT_EQ(stack.realignment.mask, test_case.expected.realignment.mask);
    EXPECT_EQ(stack.realignment.saved_pointer_offset,
              test_case.expected.realignment.saved_pointer_offset);
    EXPECT_EQ(stack.realignment.scratch_register,
              test_case.expected.realignment.scratch_register);
  }
}

TEST(FunctionStackLayoutTest, AdmitsOnlyEncodableAdjustments) {
  loom_x86_stack_frame_t stack = {};
  ASSERT_TRUE(loom_x86_function_layout_stack(INT32_MAX - 7u, 8, 0, &stack));
  EXPECT_EQ(stack.allocation_size, INT32_MAX - 7u);
  EXPECT_FALSE(loom_x86_function_layout_stack(INT32_MAX, 8, 0, &stack));

  // The local extent fits, but alignment plus the restoration slot may not.
  ASSERT_TRUE(loom_x86_function_layout_stack(INT32_MAX - 127u, 64, 0, &stack));
  EXPECT_EQ(stack.allocation_size, INT32_MAX - 63u);
  EXPECT_FALSE(loom_x86_function_layout_stack(INT32_MAX - 63u, 64, 0, &stack));
}

TEST(FunctionStackLayoutTest, IncomingAddressesFollowFinalPreservation) {
  struct IncomingCase {
    // Local extent before ABI padding and any restoration slot.
    uint32_t storage_byte_length;
    // Required body stack alignment.
    uint32_t alignment;
    // Architectural callee-saved GPRs written by the body.
    uint16_t saved_registers;
    // Displacement added to each argument's caller-stack offset.
    uint32_t argument_adjustment;
    // Restoration slot displacement, or zero for static alignment.
    uint32_t saved_pointer_offset;
  };
  const IncomingCase cases[] = {
      {0, 8, 1u << 5, 16, 0},
      {32, 8, 0, 40, 0},
      {32, 16, 0, 48, 0},
      {32, 16, 1u << 5, 48, 0},
      {32, 16, (1u << 3) | (1u << 12), 64, 0},
      {32, 32, 0, 8, 32},
      {40, 64, 1u << 5, 16, 40},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "storage=" << test_case.storage_byte_length
                 << " alignment=" << test_case.alignment
                 << " saved=" << test_case.saved_registers);
    loom_x86_instruction_t instructions[5] = {};
    loom_x86_function_t function = {
        .instructions = instructions,
        .saved_registers = test_case.saved_registers,
    };
    if (test_case.storage_byte_length != 0) {
      // Local storage is already relative to the final RSP, not caller RSP.
      loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_STORE,
                               0x89 | LOOM_X86_ENCODING_REX_W,
                               {.immediate = 24, .inputs = {11, 4}},
                               UINT32_MAX);
    }
    const bool realigned = test_case.alignment > 16;
    uint32_t vector_base = UINT32_MAX;
    if (realigned) {
      vector_base = static_cast<uint32_t>(function.instruction_count);
      loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_LOAD,
                               0x8b | LOOM_X86_ENCODING_REX_W,
                               {.result = 11, .inputs = {4}}, UINT32_MAX);
    }
    loom_x86_incoming_fixup_t fixups[2] = {
        {UINT32_MAX, static_cast<uint32_t>(function.instruction_count)},
        {UINT32_MAX, 0},
    };
    loom_x86_transport_instruction_t vector_load;
    ASSERT_TRUE(loom_x86_transport_select_abi_storage(
        LOOM_X86_STORAGE_TRANSFER_LOAD, LOOM_X86_REGISTER_CLASS_YMM, 32, 0,
        realigned ? 11 : 4, 0, &vector_load));
    loom_x86_function_append_transport(&function, &vector_load);
    if (realigned) {
      fixups[1].pointer_instruction =
          static_cast<uint32_t>(function.instruction_count);
      loom_x86_function_append(&function, LOOM_X86_ENCODING_FORM_LOAD,
                               0x8b | LOOM_X86_ENCODING_REX_W,
                               {.result = 0, .inputs = {4}}, UINT32_MAX);
    }
    fixups[1].value_instruction =
        static_cast<uint32_t>(function.instruction_count);
    loom_x86_function_append(
        &function, LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
        {
            .immediate = 32,
            .result = 0,
            .inputs = {static_cast<uint8_t>(realigned ? 0 : 4)},
        },
        UINT32_MAX);
    ASSERT_TRUE(loom_x86_function_layout_stack(
        test_case.storage_byte_length, test_case.alignment,
        function.saved_registers, &function.stack));
    loom_x86_function_fixup_incoming(&function, fixups, IREE_ARRAYSIZE(fixups),
                                     vector_base);
    if (test_case.storage_byte_length != 0) {
      EXPECT_EQ(instructions[0].operands.immediate, 24u);
    }
    EXPECT_EQ(instructions[fixups[0].value_instruction].operands.immediate,
              test_case.argument_adjustment);
    EXPECT_EQ(instructions[fixups[1].value_instruction].operands.immediate,
              32u + test_case.argument_adjustment);
    if (realigned) {
      EXPECT_EQ(instructions[vector_base].operands.immediate,
                test_case.saved_pointer_offset);
      EXPECT_EQ(instructions[fixups[1].pointer_instruction].operands.immediate,
                test_case.saved_pointer_offset);
    }
  }
}

TEST(FunctionConstructionTest, PreservesBorrowedVectorStackBase) {
  for (uint64_t spill_offset : {UINT64_MAX, uint64_t{0}, uint64_t{32}}) {
    SCOPED_TRACE(::testing::Message() << "spill offset=" << spill_offset);
    const bool preserve = spill_offset != UINT64_MAX;
    loom_x86_instruction_t instructions[5] = {};
    loom_x86_function_t function = {.instructions = instructions};
    const loom_x86_instruction_t entry_move = Instruction(
        LOOM_X86_ENCODING_FORM_UNARY_R_RM, 0x8b | LOOM_X86_ENCODING_REX_W,
        {.result = 11, .inputs = {7}});
    loom_x86_function_append(&function, entry_move.encoding_format_id,
                             entry_move.encoding_id, entry_move.operands,
                             entry_move.reference);
    const uint32_t reload =
        loom_x86_function_begin_vector_stack_inputs(&function, spill_offset);
    EXPECT_EQ(reload, preserve ? 2u : 1u);
    const loom_x86_incoming_fixup_t fixup = {
        UINT32_MAX, static_cast<uint32_t>(function.instruction_count)};
    loom_x86_transport_instruction_t vector_load;
    ASSERT_TRUE(loom_x86_transport_select_abi_storage(
        LOOM_X86_STORAGE_TRANSFER_LOAD, LOOM_X86_REGISTER_CLASS_YMM, 32, 0, 11,
        0, &vector_load));
    loom_x86_function_append_transport(&function, &vector_load);
    loom_x86_function_end_vector_stack_inputs(&function, spill_offset);
    const uint32_t storage_bytes =
        preserve ? static_cast<uint32_t>(spill_offset + 8) : 32;
    ASSERT_TRUE(
        loom_x86_function_layout_stack(storage_bytes, 32, 0, &function.stack));
    loom_x86_function_fixup_incoming(&function, &fixup, 1, reload);
    EXPECT_EQ(function.instruction_count, preserve ? 5u : 3u);
    EXPECT_EQ(function.saved_registers, 0u);

    std::string expected = Encode(entry_move);
    if (preserve) {
      expected += Encode(Instruction(
          LOOM_X86_ENCODING_FORM_STORE, 0x89 | LOOM_X86_ENCODING_REX_W,
          {
              .immediate = static_cast<int64_t>(spill_offset),
              .inputs = {11, 4},
          }));
    }
    // The borrowed pointer comes from the restoration slot, not the local
    // slot used to preserve the allocated R11 value.
    expected += Encode(
        Instruction(LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
                    {.immediate = storage_bytes, .result = 11, .inputs = {4}}));
    vector_load.operands.immediate = 8;
    expected +=
        Encode({vector_load.operands, UINT32_MAX,
                vector_load.encoding_format_id, vector_load.encoding_id});
    if (preserve) {
      expected += Encode(Instruction(
          LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
          {
              .immediate = static_cast<int64_t>(spill_offset),
              .result = 11,
              .inputs = {4},
          }));
    }
    std::string actual;
    for (iree_host_size_t i = 0; i < function.instruction_count; ++i) {
      actual += Encode(instructions[i]);
    }
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(FunctionTest, DirtyUpperVectorStateCleansBeforeReturn) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {
      .block_starts = block_starts,
      .block_count = 1,
      .may_dirty_upper_vector_state = true,
  };

  uint64_t instruction_count = 0;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, &instruction_count));
  EXPECT_EQ(instruction_count, 2u);
  EXPECT_EQ(Read(), std::string("\xc5\xf8\x77\xc3", 4));
}

TEST_F(FunctionTest, DirtyUpperVectorStatePreservesWideResult) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {
      .block_starts = block_starts,
      .block_count = 1,
      .may_dirty_upper_vector_state = true,
      .has_upper_vector_result = true,
  };

  uint64_t instruction_count = 0;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, &instruction_count));
  EXPECT_EQ(instruction_count, 1u);
  EXPECT_EQ(Read(), Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, DirtyUpperVectorStateCleansBeforeEligibleCall) {
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_CALL),
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const uint32_t cleanup_indices[] = {0};
  loom_x86_function_t function = {
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .block_starts = block_starts,
      .block_count = 1,
      .may_dirty_upper_vector_state = true,
      .upper_vector_call_cleanup_indices = cleanup_indices,
      .upper_vector_call_cleanup_count = IREE_ARRAYSIZE(cleanup_indices),
  };

  uint64_t instruction_count = 0;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, &instruction_count));
  EXPECT_EQ(instruction_count, 4u);
  EXPECT_EQ(Read(), std::string("\xc5\xf8\x77", 3) + Encode(instructions[0]) +
                        std::string("\xc5\xf8\x77\xc3", 4));
}

TEST_F(FunctionTest, PackedVectorRecipeUsesOrdinaryWriterPath) {
  loom_x86_encoding_operands_t operands = {.result = 17};
  operands.inputs[0] = 18;
  operands.inputs[1] = 19;
  loom_x86_instruction_t instructions[] = {
      {operands, UINT32_MAX, 0x8210, 0xa5fe},
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const loom_x86_function_t function = {
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .symbol_fixup_count = 0,
      .block_starts = block_starts,
      .block_count = 1,
  };

  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, nullptr));
  EXPECT_EQ(Read(), std::string("\x62\xa1\x6d\x40\xfe\xcb\xc3", 7));
}

TEST_F(FunctionTest, ReadOnlyDataUsesGenericObjectFixup) {
  loom_x86_encoding_operands_t operands = {.result = 1};
  loom_x86_instruction_t instructions[] = {
      {operands, 7, 0xe440, 0x096f},
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const loom_x86_function_t function = {
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .symbol_fixup_count = 1,
      .block_starts = block_starts,
      .block_count = 1,
  };
  uint32_t symbol_indices[8] = {};
  symbol_indices[7] = 3;
  loom_native_object_fixup_t fixup = {};

  IREE_ASSERT_OK(loom_x86_function_write(&function, symbol_indices, 5, &fixup,
                                         stream_, &arena_, nullptr));

  EXPECT_EQ(Read(), std::string("\xc5\xfa\x6f\x0d\0\0\0\0\xc3", 9));
  EXPECT_EQ(fixup.section_contribution_index, 5u);
  EXPECT_EQ(fixup.section_offset, 4u);
  EXPECT_EQ(fixup.relocation_kind, LOOM_X86_RELOCATION_ADDRESS);
  EXPECT_EQ(fixup.target_symbol_index, 3u);
  EXPECT_EQ(fixup.addend, -4);
}

TEST_F(FunctionTest, RestoreStackAndRegistersAfterResultTransport) {
  loom_x86_encoding_operands_t left = {
      .immediate = 42,
      .result = 3,  // RBX.
  };
  loom_x86_encoding_operands_t right = {
      .immediate = 43,
      .result = 12,  // R12.
  };
  loom_x86_encoding_operands_t result = {
      .result = 0,    // RAX.
      .inputs = {3},  // RBX.
  };
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_CONSTANT,
                  0xb8 | LOOM_X86_ENCODING_REX_W, left),
      Instruction(LOOM_X86_ENCODING_FORM_CONSTANT,
                  0xb8 | LOOM_X86_ENCODING_REX_W, right),
      Instruction(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                  0x8b | LOOM_X86_ENCODING_REX_W, result),
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const loom_x86_function_t function = {
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .symbol_fixup_count = 0,
      .block_starts = block_starts,
      .block_count = 1,
      .saved_registers = (1u << 3) | (1u << 12),
      .may_dirty_upper_vector_state = false,
      .has_upper_vector_result = false,
      .upper_vector_call_cleanup_indices = nullptr,
      .upper_vector_call_cleanup_count = 0,
      .stack = {24, 16, {}},
  };
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, nullptr));

  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t r12 = {};
  r12.inputs[0] = 12;
  loom_x86_encoding_operands_t stack = {
      .immediate = 24,
      .result = 4,
  };
  EXPECT_EQ(Read(),
            Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, r12)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
                                   stack)) +
                Encode(instructions[0]) + Encode(instructions[1]) +
                Encode(instructions[2]) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | LOOM_X86_ENCODING_REX_W, stack)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, r12)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, BranchesSkipEntryTransportAndPreservation) {
  loom_x86_encoding_operands_t entry = {
      .result = 3,    // RBX.
      .inputs = {2},  // RDX.
  };
  loom_x86_encoding_operands_t condition = {};
  condition.inputs[0] = 7;  // EDI.
  loom_x86_encoding_operands_t value = {
      .immediate = 42,
      .result = 3,  // RBX.
  };
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                  0x8b | LOOM_X86_ENCODING_REX_W, entry),
      Instruction(LOOM_X86_ENCODING_FORM_BRANCH_ZERO, 0, condition, 3),
      Instruction(LOOM_X86_ENCODING_FORM_CONSTANT,
                  0xb8 | LOOM_X86_ENCODING_REX_W, value),
      Instruction(LOOM_X86_ENCODING_FORM_JUMP, 0, {}, 1),
  };
  // Empty entry aliases the condition block. Empty exit aliases the epilogue.
  // The back edge skips preservation, stack allocation, and entry transport;
  // the exit reaches stack restoration before POP and RET.
  iree_host_size_t block_starts[] = {1, 1, 2, 4, 4};
  const loom_x86_function_t function = {
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .symbol_fixup_count = 0,
      .block_starts = block_starts,
      .block_count = 4,
      .saved_registers = 1u << 3,
      .may_dirty_upper_vector_state = false,
      .has_upper_vector_result = false,
      .upper_vector_call_cleanup_indices = nullptr,
      .upper_vector_call_cleanup_count = 0,
      .stack = {16, 16, {}},
  };

  // Layout also works when a previous function has already used the stream.
  const std::string prefix = Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN));
  IREE_ASSERT_OK(iree_io_stream_write(stream_, prefix.size(), prefix.data()));
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, nullptr));
  EXPECT_EQ(iree_io_stream_offset(stream_), iree_io_stream_length(stream_));

  const size_t branch_length = Encode(instructions[1]).size();
  const size_t body_length = Encode(instructions[2]).size();
  const size_t jump_length = Encode(instructions[3]).size();
  loom_x86_instruction_t forward = instructions[1];
  forward.operands.immediate = static_cast<int64_t>(body_length + jump_length);
  loom_x86_instruction_t backward = instructions[3];
  backward.operands.immediate =
      -static_cast<int64_t>(branch_length + body_length + jump_length);
  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t stack = {
      .immediate = 16,
      .result = 4,
  };
  EXPECT_EQ(Read(),
            prefix + Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
                                   stack)) +
                Encode(instructions[0]) + Encode(forward) +
                Encode(instructions[2]) + Encode(backward) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | LOOM_X86_ENCODING_REX_W, stack)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, RealignmentRestoresTheSavedStackPointerBeforePops) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {
      .block_starts = block_starts,
      .block_count = 1,
      .saved_registers = 1u << 3,
  };
  ASSERT_TRUE(loom_x86_function_layout_stack(8, 64, function.saved_registers,
                                             &function.stack));
  uint64_t instruction_count = 0;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_, &instruction_count));
  EXPECT_EQ(instruction_count, 8u);

  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t capture = {.result = 11};
  capture.inputs[0] = 4;
  loom_x86_encoding_operands_t align = {
      .immediate = -64,
      .result = 4,
  };
  loom_x86_encoding_operands_t allocate = {
      .immediate = 64,
      .result = 4,
  };
  loom_x86_encoding_operands_t save = {};
  save.inputs[0] = 11;
  save.inputs[1] = 4;
  save.immediate = 8;
  loom_x86_encoding_operands_t restore = {
      .result = 4,
  };
  restore.inputs[0] = 4;
  restore.immediate = 8;
  EXPECT_EQ(Read(),
            Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                                   0x8b | LOOM_X86_ENCODING_REX_W, capture)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | (4u << 9) | LOOM_X86_ENCODING_REX_W,
                                   align)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
                                   allocate)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_STORE,
                                   0x89 | LOOM_X86_ENCODING_REX_W, save)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_LOAD,
                                   0x8b | LOOM_X86_ENCODING_REX_W, restore)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, SymbolFixupsKeepSectionOffsetsAndTheirOwnNamespace) {
  const std::string prefix = "preceding section bytes";
  IREE_ASSERT_OK(iree_io_stream_write(stream_, prefix.size(), prefix.data()));
  loom_x86_encoding_operands_t address = {.result = 0};
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_ADDRESS_PC_RELATIVE,
                  0x8d | LOOM_X86_ENCODING_REX_W, address, 2),
      Instruction(LOOM_X86_ENCODING_FORM_CALL, 0, {}, 0),
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  loom_x86_function_t function = {
      .instructions = instructions,
      .instruction_count = IREE_ARRAYSIZE(instructions),
      .symbol_fixup_count = 2,
      .block_starts = block_starts,
      .block_count = 1,
  };
  function.stack.allocation_size = 8;
  function.stack.alignment = 16;
  const uint32_t symbol_indices[] = {4, 1, 3};
  loom_native_object_fixup_t fixups[2];
  IREE_ASSERT_OK(loom_x86_function_write(&function, symbol_indices, 5, fixups,
                                         stream_, &arena_, nullptr));
  // SUB rsp,8 occupies four bytes. The first LEA's disp32 begins three bytes
  // later; the CALL's displacement follows its one-byte opcode.
  EXPECT_EQ(fixups[0].section_contribution_index, 5u);
  EXPECT_EQ(fixups[0].section_offset, 7u);
  EXPECT_EQ(fixups[0].target_symbol_index, 3u);
  EXPECT_EQ(fixups[0].relocation_kind, LOOM_X86_RELOCATION_ADDRESS);
  EXPECT_EQ(fixups[0].addend, -4);
  EXPECT_EQ(fixups[1].section_contribution_index, 5u);
  EXPECT_EQ(fixups[1].section_offset, 12u);
  EXPECT_EQ(fixups[1].target_symbol_index, 4u);
  EXPECT_EQ(fixups[1].relocation_kind, LOOM_X86_RELOCATION_CALL);
  EXPECT_EQ(fixups[1].addend, -4);
}

}  // namespace
