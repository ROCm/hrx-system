// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_VIEW_H_
#define LOOMCXX_VIEW_H_

#include <loomcxx/encoding_type.h>

namespace loom::type {

// Marks an extent whose value is supplied when a view is constructed.
inline constexpr size_type dynamic = ~size_type{0};

// A rank-generic logical shape. Static extents and dynamic-axis positions are
// part of its C++ type in the same order Loom prints shaped IR types.
template <size_type... Extents>
struct [[loom::type("shape")]] shape {
  // Number of logical axes.
  static constexpr size_type rank = sizeof...(Extents);
};

// A typed, borrowed view. Static extents are part of Shape; each dynamic extent
// is captured when the view is constructed. Role records whether the attached
// encoding is an address layout or composed physical storage. The private
// object representation gives source copies and sizeof real C++ semantics.
template <class Shape, class T,
          loom::encoding::role Role = loom::encoding::role::layout>
class [[loom::type("view")]] view {
  // Borrowed base pointer retained by ordinary source-language copies.
  T* data_;
  // Source object storage preserving all logical extents.
  size_type extents_[Shape::rank ? Shape::rank : 1];
  // Source object storage preserving all element strides.
  size_type strides_[Shape::rank ? Shape::rank : 1];
};

// A logical coordinate or origin in source-axis order.
template <size_type Rank>
struct coordinates {
  // One coordinate for each logical axis.
  size_type values[Rank];
};

}  // namespace loom::type

namespace loom::detail {

// Runtime extents in source-axis order. Static axes have no value slot.
template <loom::type::size_type Count>
struct dimensions {
  // One extent for each dynamic source axis.
  loom::type::size_type values[Count];
};

template <>
struct dimensions<0> {};

template <loom::type::size_type... Extents>
using view_dimensions =
    dimensions<(loom::type::size_type{0} + ... +
                (Extents == loom::type::dynamic ? loom::type::size_type{1}
                                                : loom::type::size_type{0}))>;

}  // namespace loom::detail

namespace loom::encoding::layout {

// Constructs a row-major dense address layout for Rank axes.
template <loom::type::size_type Rank = 2>
[[loom::op("encoding.layout.dense")]]
loom::type::encoding<role::layout, Rank> dense();

// Constructs an address layout from one element stride per logical axis.
template <class... Strides>
[[loom::op("encoding.layout.strided")]]
loom::type::encoding<role::layout, sizeof...(Strides)> strided(
    Strides... strides);

}  // namespace loom::encoding::layout

namespace loom::buffer {

// Forms a borrowed view over data. Dimensions supplies one runtime extent for
// each dynamic axis and encoding captures its address or storage mapping.
template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T, Role> view(
    T* data, loom::detail::view_dimensions<Extents...> dimensions,
    loom::type::encoding<Role, sizeof...(Extents)> encoding);

}  // namespace loom::buffer

namespace loom::view {

// Forms a subview with an explicitly selected result extent pattern.
template <loom::type::size_type... Extents, class T,
          loom::type::size_type... SourceExtents, loom::encoding::role Role>
[[loom::op("view.subview")]]
loom::type::view<loom::type::shape<Extents...>, T, Role> subview(
    loom::type::view<loom::type::shape<SourceExtents...>, T, Role> source,
    loom::type::coordinates<sizeof...(SourceExtents)> origin,
    loom::detail::view_dimensions<Extents...> dimensions);

// Forms a subview while preserving the source's static/dynamic extent pattern.
template <class T, loom::type::size_type... Extents, loom::encoding::role Role>
[[loom::op("view.subview")]]
loom::type::view<loom::type::shape<Extents...>, T, Role> subview(
    loom::type::view<loom::type::shape<Extents...>, T, Role> source,
    loom::type::coordinates<sizeof...(Extents)> origin,
    loom::detail::view_dimensions<Extents...> dimensions);

// Loads one scalar at a logical coordinate.
template <class Shape, class T, loom::encoding::role Role, class... Indices>
[[loom::op("view.load")]]
T load(loom::type::view<Shape, T, Role> source, Indices... indices);

// Observes one volatile element and returns its ordinary scalar value.
template <class Shape, class T, loom::encoding::role Role, class... Indices>
[[loom::op("view.load")]]
T load(loom::type::view<Shape, volatile T, Role> source, Indices... indices);

// Stores one scalar at a logical coordinate. Template deduction
// rejects destinations whose element type is const.
template <class Shape, class T, loom::encoding::role Role, class... Indices>
[[loom::op("view.store")]]
void store(T value, loom::type::view<Shape, T, Role> destination,
           Indices... indices);

// Observes a store through a volatile view without qualifying the scalar value.
template <class Shape, class T, loom::encoding::role Role, class... Indices>
[[loom::op("view.store")]]
void store(T value, loom::type::view<Shape, volatile T, Role> destination,
           Indices... indices);

}  // namespace loom::view

#endif  // LOOMCXX_VIEW_H_
