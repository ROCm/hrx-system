// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/command/composition.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(CommandCompositionTest, AnswersCompleteFamilyQueries) {
  static const struct {
    loom_op_kind_t op_kind;
    loom_command_composition_mode_t schedule_mode;
    loom_command_composition_mode_t composition_mode;
    bool is_kernel_launch_schedule;
    bool is_serial_program_schedule;
  } kCases[] = {
      {LOOM_OP_COMMAND_DEF, LOOM_COMMAND_COMPOSITION_MODE_NONE,
       LOOM_COMMAND_COMPOSITION_MODE_SERIAL, false, false},
      {LOOM_OP_COMMAND_SERIAL, LOOM_COMMAND_COMPOSITION_MODE_SERIAL,
       LOOM_COMMAND_COMPOSITION_MODE_SERIAL, false, true},
      {LOOM_OP_COMMAND_CONCURRENT, LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT,
       LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT, false, false},
      {LOOM_OP_KERNEL_LAUNCH_SERIAL, LOOM_COMMAND_COMPOSITION_MODE_SERIAL,
       LOOM_COMMAND_COMPOSITION_MODE_SERIAL, true, false},
      {LOOM_OP_KERNEL_LAUNCH_CONCURRENT,
       LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT,
       LOOM_COMMAND_COMPOSITION_MODE_CONCURRENT, true, false},
  };

  for (const auto& test_case : kCases) {
    loom_op_t op = {.kind = test_case.op_kind};
    EXPECT_EQ(loom_command_composition_schedule_mode(&op),
              test_case.schedule_mode);
    EXPECT_EQ(loom_command_composition_mode(&op), test_case.composition_mode);
    EXPECT_EQ(loom_command_composition_is_kernel_launch_schedule(&op),
              test_case.is_kernel_launch_schedule);
    EXPECT_EQ(
        loom_command_composition_is_serial(&op),
        test_case.composition_mode == LOOM_COMMAND_COMPOSITION_MODE_SERIAL);
    EXPECT_EQ(loom_command_composition_is_serial_program_schedule(&op),
              test_case.is_serial_program_schedule);
  }

  loom_op_t non_composition = {.kind = LOOM_OP_KERNEL_LAUNCH};
  EXPECT_EQ(loom_command_composition_schedule_mode(&non_composition),
            LOOM_COMMAND_COMPOSITION_MODE_NONE);
  EXPECT_FALSE(
      loom_command_composition_is_kernel_launch_schedule(&non_composition));
  EXPECT_FALSE(loom_command_composition_is_serial(&non_composition));
  EXPECT_EQ(loom_command_composition_mode(&non_composition),
            LOOM_COMMAND_COMPOSITION_MODE_NONE);
  EXPECT_FALSE(
      loom_command_composition_is_serial_program_schedule(&non_composition));

  EXPECT_EQ(loom_command_composition_schedule_mode(nullptr),
            LOOM_COMMAND_COMPOSITION_MODE_NONE);
  EXPECT_FALSE(loom_command_composition_is_kernel_launch_schedule(nullptr));
  EXPECT_FALSE(loom_command_composition_is_serial(nullptr));
  EXPECT_EQ(loom_command_composition_mode(nullptr),
            LOOM_COMMAND_COMPOSITION_MODE_NONE);
  EXPECT_FALSE(loom_command_composition_is_serial_program_schedule(nullptr));
}

}  // namespace
}  // namespace loom
