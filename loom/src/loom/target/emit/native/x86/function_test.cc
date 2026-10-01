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
                                   uint32_t branch_target = UINT32_MAX) {
  return {operands, branch_target, static_cast<uint16_t>(form), encoding_id};
}

// Encoding has independent bitfield tests. These tests isolate the writer's
// composition and layout contract from individual instruction encodings.
std::string Encode(const loom_x86_instruction_t& instruction) {
  loom_x86_encoded_instruction_t encoded;
  loom_x86_encode_instruction(
      static_cast<loom_x86_encoding_form_t>(instruction.form),
      instruction.encoding_id, &instruction.operands, &encoded);
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
  loom_x86_function_t function = {};
  function.block_starts = block_starts;
  function.block_count = 1;

  IREE_ASSERT_OK(loom_x86_function_write(&function, stream_, &arena_));
  EXPECT_EQ(Read(), Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, RestoreRegistersInReverseOrderAfterResultTransport) {
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
      /*.block_starts=*/block_starts,
      /*.block_count=*/1,
      /*.saved_registers=*/(1u << 3) | (1u << 12),
  };
  IREE_ASSERT_OK(loom_x86_function_write(&function, stream_, &arena_));

  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  loom_x86_encoding_operands_t r12 = {};
  r12.inputs[0] = 12;
  EXPECT_EQ(Read(),
            Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, r12)) +
                Encode(instructions[0]) + Encode(instructions[1]) +
                Encode(instructions[2]) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, r12)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

TEST_F(FunctionTest, BranchesResolveToBlocksInsidePreservationEnvelope) {
  loom_x86_encoding_operands_t condition = {};
  condition.inputs[0] = 7;  // EDI.
  loom_x86_encoding_operands_t value = {};
  value.result = 3;  // RBX.
  value.immediate = 42;
  loom_x86_instruction_t instructions[] = {
      Instruction(LOOM_X86_ENCODING_FORM_BRANCH_ZERO, 0, condition, 3),
      Instruction(LOOM_X86_ENCODING_FORM_CONSTANT,
                  0xb8 | LOOM_X86_ENCODING_REX_W, value),
      Instruction(LOOM_X86_ENCODING_FORM_JUMP, 0, {}, 1),
  };
  // Empty entry aliases the condition block. Empty exit aliases the epilogue.
  // The back edge must skip PUSH; the forward edge must reach POP, not RET.
  iree_host_size_t block_starts[] = {0, 0, 1, 3, 3};
  const loom_x86_function_t function = {
      /*.instructions=*/instructions,
      /*.instruction_count=*/IREE_ARRAYSIZE(instructions),
      /*.block_starts=*/block_starts,
      /*.block_count=*/4,
      /*.saved_registers=*/1u << 3,
  };

  // Layout also works when a previous function has already used the stream.
  const std::string prefix = Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN));
  IREE_ASSERT_OK(iree_io_stream_write(stream_, prefix.size(), prefix.data()));
  IREE_ASSERT_OK(loom_x86_function_write(&function, stream_, &arena_));
  EXPECT_EQ(iree_io_stream_offset(stream_), iree_io_stream_length(stream_));

  const size_t branch_length = Encode(instructions[0]).size();
  const size_t body_length = Encode(instructions[1]).size();
  const size_t jump_length = Encode(instructions[2]).size();
  loom_x86_instruction_t forward = instructions[0];
  forward.operands.immediate = static_cast<int64_t>(body_length + jump_length);
  loom_x86_instruction_t backward = instructions[2];
  backward.operands.immediate =
      -static_cast<int64_t>(branch_length + body_length + jump_length);
  loom_x86_encoding_operands_t rbx = {};
  rbx.inputs[0] = 3;
  EXPECT_EQ(Read(),
            prefix + Encode(Instruction(LOOM_X86_ENCODING_FORM_PUSH, 0, rbx)) +
                Encode(forward) + Encode(instructions[1]) + Encode(backward) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_POP, 0, rbx)) +
                Encode(Instruction(LOOM_X86_ENCODING_FORM_RETURN)));
}

}  // namespace
