// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_TARGET_H_
#define LOOMCXX_TARGET_H_

namespace loom::target {

// Link-time target identity whose concrete record is supplied by another
// module. Namespace-scope extern const objects of this type become
// target.decl symbols and can be named by loom::target(object).
struct [[loom::target("target.decl")]] declaration {};

// Presence-bearing source value for an optional target field. Direct
// assignment sets the field; value initialization leaves it absent so the
// selected target row supplies its default.
template <class T>
class optional {
 public:
  constexpr optional() = default;
  constexpr optional(T value) : value_(value), present_(true) {}

 private:
  // Explicit field value when present_ is true.
  T value_{};
  // Whether the source definition explicitly sets this field.
  bool present_ = false;
};

// Source polarity for a target feature assertion.
enum class feature : unsigned char {
  // Leaves support and polarity to the selected target row.
  unspecified,
  // Requires the selected target feature to be enabled.
  enabled,
  // Requires the selected target feature to be disabled.
  disabled,
};

namespace subgroup {

// Reads the selected target's subgroup width in launch configuration code or
// a trailing loom::where template applicability contract. This is a
// compilation input, unlike loom::kernel::subgroup::size(), which queries the
// executing kernel topology.
[[loom::op("target.subgroup.size")]] unsigned size();

}  // namespace subgroup

}  // namespace loom::target

#endif  // LOOMCXX_TARGET_H_
