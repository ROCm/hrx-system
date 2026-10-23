// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "deinterleave.h"

#include <loomcxx/low.h>
#include <loomcxx/target.h>
namespace fft_experiment {
struct [[loom::representation("x86.avx512.core")]] StagingContract {};
struct [[loom::target("x86.target")]] StagingTarget {
  // Native CPU profile used by the packet shuffle.
  const char* kind;
};
constexpr StagingTarget staging_target{.kind = "avx512"};
LOOM_TEMPLATE_DEF(deinterleave)
[[loom::priority(1)]] Complex<Packet> deinterleave_semantic(Packet first,
                                                            Packet second) {
  return {Packet{first[0], first[2], second[0], second[2]},
          Packet{first[1], first[3], second[1], second[3]}};
}
LOOM_TEMPLATE_DEF(deinterleave)
[[loom::target(staging_target), loom::priority(20)]]
Complex<Packet> deinterleave_native(Packet first, Packet second) {
  return {loom::low::assembly<StagingContract, Packet>(R"(
      (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
        %result = vshufps.xmm %first, %second, 136
        return %result
      })",
                                                       first, second),
          loom::low::assembly<StagingContract, Packet>(R"(
      (%first: reg<x86.xmm>, %second: reg<x86.xmm>) -> (reg<x86.xmm>) {
        %result = vshufps.xmm %first, %second, 221
        return %result
      })",
                                                       first, second)};
}
}  // namespace fft_experiment
