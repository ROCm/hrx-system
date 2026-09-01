// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/testing/gtest.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

namespace {

TEST(InstructionFuzzTest, ExercisesEveryOpcodeSelector) {
  for (uint16_t selector = 0; selector <= UINT8_MAX; ++selector) {
    SCOPED_TRACE(selector);
    const uint8_t input = static_cast<uint8_t>(selector);
    EXPECT_EQ(LLVMFuzzerTestOneInput(&input, sizeof(input)), 0);
  }
}

}  // namespace
