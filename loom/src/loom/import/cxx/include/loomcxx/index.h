// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_INDEX_H_
#define LOOMCXX_INDEX_H_

namespace loom::type {

// Native target-independent signed coordinate supplied by structured Loom
// regions or constructed from a compile-time integer. The carrier width is
// selected when the program is lowered for a target.
class [[loom::type("index")]] index {
 public:
  explicit constexpr index(long long value) : value_(value) {}

  explicit constexpr operator long long() const { return value_; }

 private:
  long long value_;
};

// Native target-independent unsigned byte quantity. Offsets remain distinct
// from logical coordinates so layouts cannot accidentally multiply or divide
// physical byte positions.
class [[loom::type("offset")]] offset {
 public:
  explicit constexpr offset(unsigned long long value) : value_(value) {}

  explicit constexpr operator unsigned long long() const { return value_; }

 private:
  unsigned long long value_;
};

// Quotient and remainder produced together from one logical division.
struct divrem_result {
  // Nonnegative quotient.
  index quotient;
  // Nonnegative remainder.
  index remainder;
};

// Ordinary operators retain logical coordinates in the signed index domain.
[[loom::op("index.add")]] index operator+(index lhs, index rhs);
[[loom::op("index.sub")]] index operator-(index lhs, index rhs);
[[loom::op("index.mul")]] index operator*(index lhs, index rhs);
[[loom::op("index.div")]] index operator/(index lhs, index rhs);
[[loom::op("index.rem")]] index operator%(index lhs, index rhs);
[[loom::op("index.andi")]] index operator&(index lhs, index rhs);
[[loom::op("index.ori")]] index operator|(index lhs, index rhs);
[[loom::op("index.xori")]] index operator^(index lhs, index rhs);
[[loom::op("index.shli")]] index operator<<(index lhs, index rhs);
[[loom::op("index.shrsi")]] index operator>>(index lhs, index rhs);
[[loom::op("index.cmp", "eq")]] bool operator==(index lhs, index rhs);
[[loom::op("index.cmp", "ne")]] bool operator!=(index lhs, index rhs);
[[loom::op("index.cmp", "slt")]] bool operator<(index lhs, index rhs);
[[loom::op("index.cmp", "sle")]] bool operator<=(index lhs, index rhs);
[[loom::op("index.cmp", "sgt")]] bool operator>(index lhs, index rhs);
[[loom::op("index.cmp", "sge")]] bool operator>=(index lhs, index rhs);

// Offset arithmetic wraps in the unsigned physical carrier selected by the
// target. Multiplication, division, bitwise operations, and shifts are absent
// because physical byte quantities do not define those operations.
[[loom::op("index.add")]] offset operator+(offset lhs, offset rhs);
[[loom::op("index.sub")]] offset operator-(offset lhs, offset rhs);
[[loom::op("index.cmp", "eq")]] bool operator==(offset lhs, offset rhs);
[[loom::op("index.cmp", "ne")]] bool operator!=(offset lhs, offset rhs);
[[loom::op("index.cmp", "ult")]] bool operator<(offset lhs, offset rhs);
[[loom::op("index.cmp", "ule")]] bool operator<=(offset lhs, offset rhs);
[[loom::op("index.cmp", "ugt")]] bool operator>(offset lhs, offset rhs);
[[loom::op("index.cmp", "uge")]] bool operator>=(offset lhs, offset rhs);

// A fixed count of native index values. Class template argument deduction lets
// source spell runtime shape and coordinate records as indices{a, b, ...}.
template <__SIZE_TYPE__ Count>
struct indices {
  // Values in logical axis order.
  index values[Count];
};

template <>
struct indices<0> {};

template <class... Values>
indices(Values...) -> indices<sizeof...(Values)>;

}  // namespace loom::type

namespace loom::detail {

// Explicit dimensions for a shaped source carrier's dependent result axes.
template <class Carrier>
using shape_dimensions =
    loom::type::indices<Carrier::shape_type::dynamic_count>;

}  // namespace loom::detail

namespace loom::index {

// Converts a non-Boolean C++ integer to Loom's target-independent index type.
// Unsigned narrow integers are widened before conversion so their high bit
// retains ordinary C++ value semantics.
template <class Integer>
  requires(__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("index.cast")]] loom::type::index cast(Integer value);

// Converts a logical coordinate to a fixed-width C++ integer. The explicit
// result type records the source program's intended width and signedness.
template <class Integer>
  requires(__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("index.cast")]] Integer cast(loom::type::index value);

// Converts a physical byte quantity to a logical coordinate. Target lowering
// requires its numeric value to fit the selected index carrier.
[[loom::op("index.cast")]] loom::type::index cast(loom::type::offset value);

// Scales a logical coordinate by a physical byte stride.
[[loom::op("index.scale")]] loom::type::offset scale(
    loom::type::index value, loom::type::offset byte_stride);

// Computes a nonnegative quotient and remainder together.
[[loom::op("index.divrem")]] loom::type::divrem_result divrem(
    loom::type::index lhs, loom::type::index rhs);

[[loom::op("index.min")]] loom::type::index min(loom::type::index lhs,
                                                loom::type::index rhs);
[[loom::op("index.max")]] loom::type::index max(loom::type::index lhs,
                                                loom::type::index rhs);
[[loom::op("index.madd")]] loom::type::index madd(loom::type::index a,
                                                  loom::type::index b,
                                                  loom::type::index c);
[[loom::op("index.shrui")]] loom::type::index shrui(loom::type::index lhs,
                                                    loom::type::index rhs);
[[loom::op("index.rotli")]] loom::type::index rotli(loom::type::index value,
                                                    loom::type::index amount);
[[loom::op("index.rotri")]] loom::type::index rotri(loom::type::index value,
                                                    loom::type::index amount);
[[loom::op("index.ctlzi")]] loom::type::index ctlzi(loom::type::index value);
[[loom::op("index.cttzi")]] loom::type::index cttzi(loom::type::index value);
[[loom::op("index.ctpopi")]] loom::type::index ctpopi(loom::type::index value);

// Explicit unsigned comparisons preserve uncommon address interpretations.
[[loom::op("index.cmp", "ult")]] bool ult(loom::type::index lhs,
                                          loom::type::index rhs);
[[loom::op("index.cmp", "ule")]] bool ule(loom::type::index lhs,
                                          loom::type::index rhs);
[[loom::op("index.cmp", "ugt")]] bool ugt(loom::type::index lhs,
                                          loom::type::index rhs);
[[loom::op("index.cmp", "uge")]] bool uge(loom::type::index lhs,
                                          loom::type::index rhs);

}  // namespace loom::index

namespace loom::offset {

// Converts a non-Boolean C++ integer to Loom's target-independent unsigned
// byte-offset type.
template <class Integer>
  requires(__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("index.cast")]] loom::type::offset cast(Integer value);

// Converts a physical byte quantity to a fixed-width C++ integer. The
// explicit result type records the source program's intended width and
// signedness.
template <class Integer>
  requires(__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("index.cast")]] Integer cast(loom::type::offset value);

// Converts a logical coordinate to a physical byte quantity. Target lowering
// requires the coordinate to be nonnegative and representable by the selected
// offset carrier.
[[loom::op("index.cast")]] loom::type::offset cast(loom::type::index value);

// Explicit signed comparisons preserve uncommon physical interpretations.
[[loom::op("index.cmp", "slt")]] bool slt(loom::type::offset lhs,
                                          loom::type::offset rhs);
[[loom::op("index.cmp", "sle")]] bool sle(loom::type::offset lhs,
                                          loom::type::offset rhs);
[[loom::op("index.cmp", "sgt")]] bool sgt(loom::type::offset lhs,
                                          loom::type::offset rhs);
[[loom::op("index.cmp", "sge")]] bool sge(loom::type::offset lhs,
                                          loom::type::offset rhs);

}  // namespace loom::offset

#endif  // LOOMCXX_INDEX_H_
