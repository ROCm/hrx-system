// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_HIP_VECTOR_TYPES_H_
#define LOOMCXX_HIP_VECTOR_TYPES_H_

// HIP's two- and four-component values carry native vector alignment and
// transport semantics. One- and three-component values remain unpadded records
// so their object layout is exactly one and three scalar elements.
#define LOOMCXX_HIP_VECTOR_FAMILY(Name, Scalar, MakeScalar)            \
  struct Name##1 { Scalar x; };                                        \
  using Name##2 = Scalar __attribute__((ext_vector_type(2)));          \
  struct Name##3 {                                                     \
    Scalar x;                                                          \
    Scalar y;                                                          \
    Scalar z;                                                          \
  };                                                                   \
  using Name##4 = Scalar __attribute__((ext_vector_type(4)));          \
  [[loom::device, loom::inline]] static inline Name##1 make_##Name##1( \
      MakeScalar x) {                                                  \
    return Name##1 {x};                                                \
  }                                                                    \
  [[loom::device, loom::inline]] static inline Name##2 make_##Name##2( \
      MakeScalar x, MakeScalar y) {                                    \
    return Name##2 {x, y};                                             \
  }                                                                    \
  [[loom::device, loom::inline]] static inline Name##3 make_##Name##3( \
      MakeScalar x, MakeScalar y, MakeScalar z) {                      \
    return Name##3 {x, y, z};                                          \
  }                                                                    \
  [[loom::device, loom::inline]] static inline Name##4 make_##Name##4( \
      MakeScalar x, MakeScalar y, MakeScalar z, MakeScalar w) {        \
    return Name##4 {x, y, z, w};                                       \
  }

LOOMCXX_HIP_VECTOR_FAMILY(char, char, signed char)
LOOMCXX_HIP_VECTOR_FAMILY(uchar, unsigned char, unsigned char)
LOOMCXX_HIP_VECTOR_FAMILY(short, short, short)
LOOMCXX_HIP_VECTOR_FAMILY(ushort, unsigned short, unsigned short)
LOOMCXX_HIP_VECTOR_FAMILY(int, int, int)
LOOMCXX_HIP_VECTOR_FAMILY(uint, unsigned int, unsigned int)
LOOMCXX_HIP_VECTOR_FAMILY(long, long, long)
LOOMCXX_HIP_VECTOR_FAMILY(ulong, unsigned long, unsigned long)
LOOMCXX_HIP_VECTOR_FAMILY(longlong, long long, long long)
LOOMCXX_HIP_VECTOR_FAMILY(ulonglong, unsigned long long, unsigned long long)
LOOMCXX_HIP_VECTOR_FAMILY(float, float, float)
LOOMCXX_HIP_VECTOR_FAMILY(double, double, double)

#undef LOOMCXX_HIP_VECTOR_FAMILY

#endif  // LOOMCXX_HIP_VECTOR_TYPES_H_
