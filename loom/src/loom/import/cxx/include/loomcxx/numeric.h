// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_NUMERIC_H_
#define LOOMCXX_NUMERIC_H_

namespace loom::type {
// One-byte E4M3 with signed finite saturation and NaN preservation.
using float8_e4m3fn_t = __float8_e4m3fn;
// One-byte IEEE E5M2 with infinities and NaNs.
using float8_e5m2_t = __float8_e5m2;

}  // namespace loom::type

namespace loom {

// Converts one scalar or each element of a native or semantic vector. Scalar
// and native-vector specializations may be used in constant expressions.
template <class To, class From>
[[loom::op("numeric.convert")]] constexpr To convert(From value);

}  // namespace loom

#endif  // LOOMCXX_NUMERIC_H_
