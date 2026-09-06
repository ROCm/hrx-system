// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_ELEMENTARY_TEST_UTIL_H_
#define IREE_MATH_ELEMENTARY_TEST_UTIL_H_

#include <cstdint>
#include <cstring>
#include <iomanip>

#include "iree/base/internal/fpu_state.h"
#include "iree/testing/gtest.h"

namespace iree::math::testing {

using F32Function = float (*)(float);

inline float F32FromBits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint32_t F32ToBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline void ExpectF32Mapping(F32Function function, uint32_t input_bits,
                             uint32_t expected_bits) {
  SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << input_bits);
  EXPECT_EQ(expected_bits, F32ToBits(function(F32FromBits(input_bits))));
}

class ElementaryTest : public ::testing::Test {
 protected:
  void SetUp() override { fpu_state_ = iree_fpu_state_push(kRequiredFpuState); }

  void TearDown() override { iree_fpu_state_pop(fpu_state_); }

  void EnableFlushDenormalsToZero() {
    iree_fpu_state_pop(fpu_state_);
    fpu_state_ = iree_fpu_state_push(
        kRequiredFpuState | IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  }

 private:
  static constexpr iree_fpu_state_flags_t kRequiredFpuState =
      IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS |
      IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST;

  // Caller floating-point control state restored after each test.
  iree_fpu_state_t fpu_state_ = {};
};

}  // namespace iree::math::testing

#endif  // IREE_MATH_ELEMENTARY_TEST_UTIL_H_
