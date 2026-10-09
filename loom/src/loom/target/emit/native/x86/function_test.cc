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
#include "loom/target/emit/native/x86/encoding.h"

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

TEST_F(FunctionTest, EmptyLeafNeedsOnlyReturn) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {.block_starts = block_starts,
                                  .block_count = 1};

  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));
  EXPECT_EQ(Read(), Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, DirtyUpperVectorStateCleansBeforeReturn) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {.block_starts = block_starts,
                                  .block_count = 1,
                                  .may_dirty_upper_vector_state = true};

  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));
  EXPECT_EQ(Read(), std::string("\xc5\xf8\x77\xc3", 4));
}

TEST_F(FunctionTest, DirtyUpperVectorStatePreservesWideResult) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {.block_starts = block_starts,
                                  .block_count = 1,
                                  .may_dirty_upper_vector_state = true,
                                  .has_upper_vector_result = true};

  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));
  EXPECT_EQ(Read(), Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, DirtyUpperVectorStateCleansBeforeEligibleCall) {
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_CALL),
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const uint32_t cleanup_indices[] = {0};
  loom_x86_function_t function = {};
  function.instructions = instructions;
  function.instruction_count = IREE_ARRAYSIZE(instructions);
  function.block_starts = block_starts;
  function.block_count = 1;
  function.may_dirty_upper_vector_state = true;
  function.upper_vector_call_cleanup_indices = cleanup_indices;
  function.upper_vector_call_cleanup_count = IREE_ARRAYSIZE(cleanup_indices);

  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));
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
      /*.instructions=*/instructions,
      /*.instruction_count=*/IREE_ARRAYSIZE(instructions),
      /*.symbol_fixup_count=*/0,
      /*.block_starts=*/block_starts,
      /*.block_count=*/1,
  };

  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));
  EXPECT_EQ(Read(), std::string("\x62\xa1\x6d\x40\xfe\xcb\xc3", 7));
}

TEST_F(FunctionTest, ReadOnlyDataUsesGenericObjectFixup) {
  loom_x86_encoding_operands_t operands = {.result = 1};
  loom_x86_instruction_t instructions[] = {
      {operands, 7, 0xe440, 0x096f},
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const loom_x86_function_t function = {
      /*.instructions=*/instructions,
      /*.instruction_count=*/IREE_ARRAYSIZE(instructions),
      /*.symbol_fixup_count=*/1,
      /*.block_starts=*/block_starts,
      /*.block_count=*/1,
  };
  uint32_t symbol_indices[8] = {};
  symbol_indices[7] = 3;
  loom_native_object_fixup_t fixup = {};

  IREE_ASSERT_OK(loom_x86_function_write(&function, symbol_indices, 5, &fixup,
                                         stream_, &arena_));

  EXPECT_EQ(Read(), std::string("\xc5\xfa\x6f\x0d\0\0\0\0\xc3", 9));
  EXPECT_EQ(fixup.section_contribution_index, 5u);
  EXPECT_EQ(fixup.section_offset, 4u);
  EXPECT_EQ(fixup.relocation_kind, LOOM_X86_RELOCATION_ADDRESS);
  EXPECT_EQ(fixup.target_symbol_index, 3u);
  EXPECT_EQ(fixup.addend, -4);
}

TEST_F(FunctionTest, RestoreStackAndRegistersAfterResultTransport) {
  loom_x86_encoding_operands_t left = {};
  left.result = 3;  // RBX.
  left.immediate = 42;
  loom_x86_encoding_operands_t right = {};
  right.result = 12;  // R12.
  right.immediate = 43;
  loom_x86_encoding_operands_t result = {};
  result.result = 0;     // RAX.
  result.inputs[0] = 3;  // RBX.
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_CONSTANT,
                  0xb8 | LOOM_X86_ENCODING_REX_W, left),
      Instruction(LOOM_X86_ENCODING_FORM_CONSTANT,
                  0xb8 | LOOM_X86_ENCODING_REX_W, right),
      Instruction(LOOM_X86_ENCODING_FORM_MOVE, 0x8b | LOOM_X86_ENCODING_REX_W,
                  result),
  };
  iree_host_size_t block_starts[] = {0, IREE_ARRAYSIZE(instructions)};
  const loom_x86_function_t function = {
      /*.instructions=*/instructions,
      /*.instruction_count=*/IREE_ARRAYSIZE(instructions),
      /*.symbol_fixup_count=*/0,
      /*.block_starts=*/block_starts,
      /*.block_count=*/1,
      /*.saved_registers=*/(1u << 3) | (1u << 12),
      /*.may_dirty_upper_vector_state=*/false,
      /*.has_upper_vector_result=*/false,
      /*.upper_vector_call_cleanup_indices=*/nullptr,
      /*.upper_vector_call_cleanup_count=*/0,
      /*.stack=*/{24, 16, {}},
  };
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));

  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t r12 = {};
  r12.inputs[0] = 12;
  loom_x86_encoding_operands_t stack = {};
  stack.result = 4;
  stack.immediate = 24;
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
  loom_x86_encoding_operands_t entry = {};
  entry.result = 3;     // RBX.
  entry.inputs[0] = 2;  // RDX.
  loom_x86_encoding_operands_t condition = {};
  condition.inputs[0] = 7;  // EDI.
  loom_x86_encoding_operands_t value = {};
  value.result = 3;  // RBX.
  value.immediate = 42;
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_MOVE, 0x8b | LOOM_X86_ENCODING_REX_W,
                  entry),
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
      /*.instructions=*/instructions,
      /*.instruction_count=*/IREE_ARRAYSIZE(instructions),
      /*.symbol_fixup_count=*/0,
      /*.block_starts=*/block_starts,
      /*.block_count=*/4,
      /*.saved_registers=*/1u << 3,
      /*.may_dirty_upper_vector_state=*/false,
      /*.has_upper_vector_result=*/false,
      /*.upper_vector_call_cleanup_indices=*/nullptr,
      /*.upper_vector_call_cleanup_count=*/0,
      /*.stack=*/{16, 16, {}},
  };

  // Layout also works when a previous function has already used the stream.
  const std::string prefix = Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN));
  IREE_ASSERT_OK(iree_io_stream_write(stream_, prefix.size(), prefix.data()));
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));
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
  loom_x86_encoding_operands_t stack = {};
  stack.result = 4;
  stack.immediate = 16;
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
  loom_x86_function_t function = {.block_starts = block_starts,
                                  .block_count = 1,
                                  .saved_registers = 1u << 3};
  function.stack.allocation_size = 64;
  function.stack.alignment = 64;
  function.stack.realignment.mask = -64;
  function.stack.realignment.saved_pointer_offset = 8;
  function.stack.realignment.scratch_register = 11;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));

  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t capture = {.result = 11};
  capture.inputs[0] = 4;
  loom_x86_encoding_operands_t align = {};
  align.result = 4;
  align.immediate = -64;
  loom_x86_encoding_operands_t allocate = {};
  allocate.result = 4;
  allocate.immediate = 64;
  loom_x86_encoding_operands_t save = {};
  save.inputs[0] = 11;
  save.inputs[1] = 4;
  save.immediate = 8;
  loom_x86_encoding_operands_t restore = {};
  restore.result = 4;
  restore.inputs[0] = 4;
  restore.immediate = 8;
  EXPECT_EQ(Read(),
            Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_MOVE,
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
  loom_x86_function_t function = {};
  function.instructions = instructions;
  function.instruction_count = IREE_ARRAYSIZE(instructions);
  function.symbol_fixup_count = 2;
  function.block_starts = block_starts;
  function.block_count = 1;
  function.stack.allocation_size = 8;
  function.stack.alignment = 16;
  const uint32_t symbol_indices[] = {4, 1, 3};
  loom_native_object_fixup_t fixups[2];
  IREE_ASSERT_OK(loom_x86_function_write(&function, symbol_indices, 5, fixups,
                                         stream_, &arena_));
  // SUB rsp,8 occupies seven bytes. The first LEA's disp32 begins three bytes
  // later; the CALL's displacement follows its one-byte opcode.
  EXPECT_EQ(fixups[0].section_contribution_index, 5u);
  EXPECT_EQ(fixups[0].section_offset, 10u);
  EXPECT_EQ(fixups[0].target_symbol_index, 3u);
  EXPECT_EQ(fixups[0].relocation_kind, LOOM_X86_RELOCATION_ADDRESS);
  EXPECT_EQ(fixups[0].addend, -4);
  EXPECT_EQ(fixups[1].section_contribution_index, 5u);
  EXPECT_EQ(fixups[1].section_offset, 15u);
  EXPECT_EQ(fixups[1].target_symbol_index, 4u);
  EXPECT_EQ(fixups[1].relocation_kind, LOOM_X86_RELOCATION_CALL);
  EXPECT_EQ(fixups[1].addend, -4);
}

TEST_F(FunctionTest, FramePointerDoesNotSuppressOutgoingStackRealignment) {
  iree_host_size_t block_starts[] = {0, 0};
  loom_x86_function_t function = {.block_starts = block_starts,
                                  .block_count = 1,
                                  .saved_registers = (1u << 3) | (1u << 5)};
  function.stack.allocation_size = 64;
  function.stack.alignment = 32;
  function.stack.has_frame_pointer = true;
  function.stack.realignment.mask = -32;
  IREE_ASSERT_OK(loom_x86_function_write(&function, nullptr, 0, nullptr,
                                         stream_, &arena_));

  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t rbp = {};
  rbp.inputs[0] = 5;
  loom_x86_encoding_operands_t establish = {.result = 5};
  establish.inputs[0] = 4;
  loom_x86_encoding_operands_t align = {};
  align.result = 4;
  align.immediate = -32;
  loom_x86_encoding_operands_t allocate = {};
  allocate.result = 4;
  allocate.immediate = 64;
  loom_x86_encoding_operands_t restore = {.result = 4};
  restore.inputs[0] = 5;
  EXPECT_EQ(Read(),
            Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbp)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_MOVE,
                                   0x8b | LOOM_X86_ENCODING_REX_W, establish)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | (4u << 9) | LOOM_X86_ENCODING_REX_W,
                                   align)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                                   0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
                                   allocate)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_MOVE,
                                   0x8b | LOOM_X86_ENCODING_REX_W, restore)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbp)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

}  // namespace
