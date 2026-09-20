// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

int64_t external_add_one(int64_t value) { return value + 1; }

int64_t external_sum_eight(int64_t a, int64_t b, int64_t c, int64_t d,
                           int64_t e, int64_t f, int64_t g, int64_t h) {
  return a + b + c + d + e + f + g + h;
}

extern int64_t loom_direct_call_test(int64_t value);
extern int64_t loom_direct_call_stress(int64_t a, int64_t b, int64_t c,
                                       int64_t d, int64_t e, int64_t f,
                                       int64_t g, int64_t h);
extern int32_t loom_view_call(const int32_t* input);
extern int32_t loom_selected_view_call(const int32_t* first,
                                       const int32_t* second,
                                       int32_t condition);
extern uint32_t loom_unsigned_divide_by_three(uint32_t value);
extern uint32_t loom_unsigned_remainder_high_bit(uint32_t value);
extern uint64_t loom_bitwise_immediates(uint64_t value);

int main(void) {
  if (loom_direct_call_test(21) != 43) {
    return 1;
  }
  if (loom_direct_call_stress(1, 2, 3, 4, 5, 6, 7, 8) != 108) {
    return 1;
  }
  const int32_t first[] = {10, 11, 12};
  const int32_t second[] = {20, 21, 22};
  if (loom_view_call(first) != 11) {
    return 1;
  }
  if (loom_selected_view_call(first, second, 1) != 11 ||
      loom_selected_view_call(first, second, 0) != 22) {
    return 1;
  }
  if (loom_unsigned_divide_by_three(UINT32_MAX) != UINT32_C(0x55555555)) {
    return 1;
  }
  if (loom_unsigned_remainder_high_bit(UINT32_MAX) != UINT32_C(0x7FFFFFFE)) {
    return 1;
  }
  const uint64_t bitwise_input = UINT64_C(0x0123456789ABCDEF);
  uint64_t bitwise_expected = bitwise_input;
  bitwise_expected &= UINT64_C(0xFFFFFFFF80000000);
  bitwise_expected |= UINT64_C(0x000000007FFFFFFF);
  bitwise_expected ^= UINT64_C(0xFFFFFFFFFFFFFF7F);
  if (loom_bitwise_immediates(bitwise_input) != bitwise_expected) {
    return 1;
  }
  return 0;
}
