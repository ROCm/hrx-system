// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/fpu_state.h"

#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
#include <xmmintrin.h>
#endif  // IREE_ARCH_X86_*

#include "iree/testing/gtest.h"

namespace {

// NOTE: depending on compiler options or architecture denormals may always be
// flushed to zero. Here we just test that they are flushed when we request them
// to be.
TEST(FPUStateTest, FlushDenormalsToZero) {
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);

  float f = 1.0f;
  volatile float* fp = &f;
  *fp = *fp * 1e-39f;
  EXPECT_EQ(0.0f, f);

  iree_fpu_state_pop(fpu_state);
}

#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
TEST(FPUStateTest, MasksExceptionsAndRoundsToNearest) {
  constexpr uint32_t kExceptionFlags = UINT32_C(0x0000003F);
  constexpr uint32_t kExceptionMasks = UINT32_C(0x00001F80);
  constexpr uint32_t kRoundingMode = UINT32_C(0x00006000);
  constexpr uint32_t kFlushModes = UINT32_C(0x00008040);
  const uint32_t original_state = _mm_getcsr();
  const uint32_t hostile_state =
      (original_state | kRoundingMode | kFlushModes) &
      ~(kExceptionFlags | kExceptionMasks);
  _mm_setcsr(hostile_state);

  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS |
                          IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST);
  const uint32_t canonical_state = _mm_getcsr();
  EXPECT_EQ(canonical_state & kExceptionMasks, kExceptionMasks);
  EXPECT_EQ(canonical_state & kRoundingMode, 0u);
  EXPECT_EQ(canonical_state & kFlushModes, 0u);

  iree_fpu_state_pop(fpu_state);
  EXPECT_EQ(_mm_getcsr(), hostile_state);
  _mm_setcsr(original_state);
}
#endif  // IREE_ARCH_X86_*

}  // namespace
