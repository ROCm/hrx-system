// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

using Byte4 = unsigned char __attribute__((ext_vector_type(4)));
using Float8x4 =
    loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(4)));

constexpr Byte4 kPartial{255, 128};
constexpr Byte4 kZero{};
constexpr Byte4 kFirst{7};
constexpr Byte4 kSplat = 9;
constexpr Float8x4 kFloat8{loom::type::float8_e4m3fn_t(0x1.1000000000001p0),
                           loom::type::float8_e4m3fn_t(-0.0)};

static_assert(kPartial[1] == 128 && kPartial[3] == 0);
static_assert(kZero[3] == 0);
static_assert(kFirst[0] == 7 && kFirst[3] == 0);
static_assert(kSplat[0] == 9 && kSplat[3] == 9);
static_assert(kFloat8[0] == 1.125 && kFloat8[3] == 0.0);

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void constant_vectors(Byte4* partial, Byte4* zero, Byte4* first, Byte4* splat,
                      Float8x4* floating) {
  unsigned lane = loom::kernel::workitem::id.x;
  partial[lane] = kPartial;
  zero[lane] = kZero;
  first[lane] = kFirst;
  splat[lane] = kSplat;
  floating[lane] = kFloat8;
}
