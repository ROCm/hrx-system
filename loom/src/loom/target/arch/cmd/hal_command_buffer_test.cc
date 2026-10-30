// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/hal_command_buffer.h"

#include <algorithm>
#include <array>

#include "iree/base/alignment.h"
#include "iree/hal/cts/util/test_base.h"
#include "loom/target/arch/cmd/format.h"

namespace iree::hal::cts {
namespace {

// Consumer fixture containing valid canonical tables. The artifact reader's
// malformed-input coverage belongs to program_test; this exercises recording
// and execution against the real HAL with fixed and late-bound buffer ranges.
struct TransferProgram {
  // Validated program view borrowed for recording.
  loom_cmd_program_t program = {};
  // One source range and one destination range, relative to bound roots.
  std::array<uint8_t, 2 * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE> references = {};
  // Mixed buffer and global obligations on the same boundary.
  std::array<uint8_t, 2 * LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SIZE>
      dependencies = {};
  // Fill, publish, and copy commands.
  std::array<uint8_t, 3 * LOOM_CMD_PROGRAM_COMMAND_SIZE> commands = {};

  explicit TransferProgram(RecordingMode mode) {
    const bool indirect = mode == RecordingMode::kIndirect;
    program.requirements.fixed_buffer_count = indirect ? 0 : 2;
    program.requirements.rebindable_binding_count = indirect ? 2 : 0;
    program.requirements.transient.binding_index = UINT32_MAX;
    program.requirements.launch_counts.binding_index = UINT32_MAX;
    program.buffer_refs = {references.data(), 2};
    program.memory_dependencies = {dependencies.data(), 2};
    program.maximum_barrier_dependency_count = 2;
    program.commands = {commands.data(), 3};
    for (uint32_t i = 0; i < 2; ++i) {
      uint8_t* row = references.data() + i * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
      iree_unaligned_store_le_u32(row + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET,
                                  indirect
                                      ? LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE
                                      : LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED);
      iree_unaligned_store_le_u32(
          row + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET, i);
      iree_unaligned_store_le_u64(
          row + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET,
          i == 0 ? 16 : 32);
      iree_unaligned_store_le_u64(
          row + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET, 32);
      row = dependencies.data() + i * LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SIZE;
      iree_unaligned_store_le_u32(
          row + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SOURCE_SCOPE_OFFSET,
          LOOM_CMD_PROGRAM_ACCESS_TRANSFER_WRITE);
      iree_unaligned_store_le_u32(
          row + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_TARGET_SCOPE_OFFSET,
          LOOM_CMD_PROGRAM_ACCESS_TRANSFER_READ);
      iree_unaligned_store_le_u32(
          row + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_BUFFER_REF_OFFSET,
          i == 0 ? 0 : LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_GLOBAL);
    }
    StoreCommand(0, LOOM_CMD_PROGRAM_COMMAND_KIND_FILL_BARRIER, 0, 0x3a, 1);
    StoreCommand(1, LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER,
                 LOOM_CMD_PROGRAM_STAGE_TRANSFER,
                 LOOM_CMD_PROGRAM_STAGE_TRANSFER, 0, 0, 2);
    StoreCommand(2, LOOM_CMD_PROGRAM_COMMAND_KIND_COPY, 0, 1);
  }

  void StoreCommand(uint32_t index, loom_cmd_program_command_kind_t kind,
                    uint32_t operand0, uint32_t operand1, uint32_t operand2 = 0,
                    uint32_t operand3 = 0, uint32_t operand4 = 0) {
    uint8_t* row = commands.data() + index * LOOM_CMD_PROGRAM_COMMAND_SIZE;
    const uint32_t words[] = {static_cast<uint32_t>(kind),
                              0,
                              0,
                              operand0,
                              operand1,
                              operand2,
                              operand3,
                              operand4};
    for (size_t i = 0; i < IREE_ARRAYSIZE(words); ++i) {
      iree_unaligned_store_le_u32(row + i * sizeof(uint32_t), words[i]);
    }
  }
};

class CmdHalCommandBufferTest : public CtsTestBase<> {};

TEST_P(CmdHalCommandBufferTest, RangedDependenciesAcrossReusedRecordings) {
  TransferProgram fixture(recording_mode());
  Ref<iree_hal_buffer_t> source;
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(128, source.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(128, target.out()));
  const iree_hal_buffer_binding_t bindings[] = {
      {source.get(), 8, 96},
      {target.get(), 8, 96},
  };
  const bool indirect = recording_mode() == RecordingMode::kIndirect;
  auto* queue = QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH |
                                          IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  ASSERT_NE(queue, nullptr);
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(loom_cmd_program_hal_command_buffer_create(
      iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      &fixture.program, nullptr, indirect ? 0 : 2,
      indirect ? nullptr : bindings, 0, nullptr, iree_allocator_system(),
      command_buffer.out()));
  for (int iteration = 0; iteration < 2; ++iteration) {
    IREE_ASSERT_OK(SubmitCommandBufferAndWait(
        command_buffer, indirect ? iree_hal_buffer_binding_table_t{2, bindings}
                                 : iree_hal_buffer_binding_table_empty()));
    std::vector<uint8_t> expected(128, 0);
    std::fill(expected.begin() + 40, expected.begin() + 72, 0x3a);
    EXPECT_THAT(ReadBufferBytes(target, 0, 128),
                ::testing::ContainerEq(expected));
  }
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CmdHalCommandBufferTest);

}  // namespace
}  // namespace iree::hal::cts
