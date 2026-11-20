// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "frame_transpose.h"

#include <loomcxx/low.h>
#include <loomcxx/target.h>
namespace fft_experiment {
struct [[loom::representation("x86.avx512.core")]] TransposeContract {};
template <unsigned Control>
[[loom::force_inline]] static Packet shuffle_pair(Packet first, Packet second) {
  if constexpr (Control == 0x44) {
    return loom::low::assembly<TransposeContract, Packet>(R"(
        (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
          %result = vshufps.xmm %first, %second, 68
          return %result
        })",
                                                          first, second);
  } else if constexpr (Control == 0xee) {
    return loom::low::assembly<TransposeContract, Packet>(R"(
        (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
          %result = vshufps.xmm %first, %second, 238
          return %result
        })",
                                                          first, second);
  } else if constexpr (Control == 0x88) {
    return loom::low::assembly<TransposeContract, Packet>(R"(
        (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
          %result = vshufps.xmm %first, %second, 136
          return %result
        })",
                                                          first, second);
  } else {
    static_assert(Control == 0xdd);
    return loom::low::assembly<TransposeContract, Packet>(R"(
        (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
          %result = vshufps.xmm %first, %second, 221
          return %result
        })",
                                                          first, second);
  }
}
struct [[loom::target("x86.target")]] TransposeTarget {
  // Authored native CPU profile.
  const char* kind;
};
constexpr TransposeTarget transpose_target{.kind = "avx512"};

LOOM_TEMPLATE_DEF(transpose_block)
[[loom::priority(1)]] Block transpose_semantic(Packet first, Packet second,
                                               Packet third, Packet fourth) {
  return {Packet{first[0], second[0], third[0], fourth[0]},
          Packet{first[1], second[1], third[1], fourth[1]},
          Packet{first[2], second[2], third[2], fourth[2]},
          Packet{first[3], second[3], third[3], fourth[3]}};
}

LOOM_TEMPLATE_DEF(transpose_block)
[[loom::target(transpose_target), loom::priority(20)]] Block transpose_native(
    Packet first, Packet second, Packet third, Packet fourth) {
  const auto first_low = shuffle_pair<0x44>(first, second);
  const auto first_high = shuffle_pair<0xee>(first, second);
  const auto second_low = shuffle_pair<0x44>(third, fourth);
  const auto second_high = shuffle_pair<0xee>(third, fourth);
  return {shuffle_pair<0x88>(first_low, second_low),
          shuffle_pair<0xdd>(first_low, second_low),
          shuffle_pair<0x88>(first_high, second_high),
          shuffle_pair<0xdd>(first_high, second_high)};
}
LOOM_TEMPLATE_DEF(interleave_pair)
[[loom::priority(1)]] TwoRows interleave_pair_semantic(Packet first,
                                                       Packet second) {
  return {Packet{first[0], second[0], first[1], second[1]},
          Packet{first[2], second[2], first[3], second[3]}};
}
LOOM_TEMPLATE_DEF(interleave_pair)
[[loom::target(transpose_target), loom::priority(20)]]
TwoRows interleave_pair_native(Packet first, Packet second) {
  return {loom::low::assembly<TransposeContract, Packet>(R"(
        (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
          %result = vpunpckldq.xmm %first, %second
          return %result
        })",
                                                         first, second),
          loom::low::assembly<TransposeContract, Packet>(R"(
        (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
          %joined = vshufps.xmm %first, %second, 238
          %result = vshufps.xmm %joined, %joined, 216
          return %result
        })",
                                                         first, second)};
}
LOOM_TEMPLATE_DEF(deinterleave_pair)
[[loom::priority(1)]] TwoRows deinterleave_pair_semantic(Packet first,
                                                         Packet second) {
  return {Packet{first[0], first[2], second[0], second[2]},
          Packet{first[1], first[3], second[1], second[3]}};
}
LOOM_TEMPLATE_DEF(deinterleave_pair)
[[loom::target(transpose_target), loom::priority(20)]]
TwoRows deinterleave_pair_native(Packet first, Packet second) {
  return {shuffle_pair<0x88>(first, second), shuffle_pair<0xdd>(first, second)};
}

}  // namespace fft_experiment
