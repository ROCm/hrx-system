#pragma once
#include <array>
#include <cmath>
#pragma clang fp contract(off)
using WideValues = std::array<float, 8>;
static inline WideValues transform_block_axis(WideValues input) {
  float sum = input[0] + input[4];
  float difference = input[0] - input[4];
  float rotation = (input[2] + input[6]) * 0x1.1517a80000000p-1f;
  float two = std::fma(input[6], -0x1.d906bc0000000p+0f, rotation);
  float six = std::fma(input[2], 0x1.87de2a0000000p-1f, rotation);
  float even0 = sum + six;
  float even1 = difference + two;
  float even2 = difference - two;
  float even3 = sum - six;
  float odd_rotation =
      (input[1] + input[3] + input[5] + input[7]) * 0x1.2d062e0000000p+0f;
  float seven_three =
      std::fma(input[7] + input[3], -0x1.f6297c0000000p+0f, odd_rotation);
  float five_one =
      std::fma(input[5] + input[1], -0x1.8f8b840000000p-2f, odd_rotation);
  float seven_one = (input[7] + input[1]) * -0x1.ccc9ae0000000p-1f;
  float five_three = (input[5] + input[3]) * -0x1.480d9e0000000p+1f;
  float odd0 = std::fma(input[1], 0x1.8056940000000p+0f, seven_one) + five_one;
  float odd1 =
      std::fma(input[3], 0x1.894e980000000p+1f, five_three) + seven_three;
  float odd2 = std::fma(input[5], 0x1.06cca20000000p+1f, five_three) + five_one;
  float odd3 =
      std::fma(input[7], 0x1.31cc6a0000000p-2f, seven_one) + seven_three;
  return {even0 + odd0, even1 + odd1, even2 + odd2, even3 + odd3,
          even3 - odd3, even2 - odd2, even1 - odd1, even0 - odd0};
}
