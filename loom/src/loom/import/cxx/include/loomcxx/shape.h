// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_SHAPE_H_
#define LOOMCXX_SHAPE_H_

namespace loom::type {

using size_type = __SIZE_TYPE__;

// Marks an extent whose value is supplied with a shaped source value.
inline constexpr size_type dynamic = ~size_type{0};

// A rank-generic logical shape. Static extents and dynamic-axis positions are
// part of its C++ type in the same order Loom prints shaped IR types.
template <size_type... Extents>
struct [[loom::type("shape")]] shape {
  // Number of logical axes.
  static constexpr size_type rank = sizeof...(Extents);
  // Number of explicit runtime extents, in logical axis order.
  static constexpr size_type dynamic_count =
      (size_type{0} + ... + (Extents == dynamic ? size_type{1} : size_type{0}));
};

}  // namespace loom::type

#endif  // LOOMCXX_SHAPE_H_
