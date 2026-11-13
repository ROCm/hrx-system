// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_VIEW_H_
#define LOOMCXX_VIEW_H_

#include <loomcxx/buffer.h>
#include <loomcxx/encoding_type.h>
#include <loomcxx/index.h>

namespace loom::type {

// A typed, borrowed view. Static extents are part of Shape; each dynamic extent
// is captured when the view is constructed. Encoding names a transported
// address layout or physical storage encoding; void represents an ordinary
// unencoded Loom view. The private object representation gives source copies
// and sizeof real C++ semantics.
template <class Shape, class T, class Encoding = void>
class [[loom::type("view")]] view {
 public:
  using shape_type = Shape;

 private:
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
    dimensions<loom::type::shape<Extents...>::dynamic_count>;

template <loom::type::size_type... Extents>
using view_index_dimensions =
    loom::type::indices<loom::type::shape<Extents...>::dynamic_count>;

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

// Forms an ordinary unencoded view over pointer storage.
template <loom::type::size_type... Extents, class T>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T> view(
    T* data, loom::detail::view_dimensions<Extents...> dimensions);

template <loom::type::size_type... Extents, class T>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T> view(
    T* data, loom::detail::view_index_dimensions<Extents...> dimensions);

// Forms a borrowed view over data. Dimensions supplies one runtime extent for
// each dynamic axis and encoding captures its address or storage mapping.
template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T,
                 loom::type::encoding<Role, sizeof...(Extents)>>
view(T* data, loom::detail::view_dimensions<Extents...> dimensions,
     loom::type::encoding<Role, sizeof...(Extents)> encoding);

// Forms a borrowed view while retaining dynamic extents already in Loom's
// index domain. At least one dynamic axis distinguishes this from static {}.
template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T,
                 loom::type::encoding<Role, sizeof...(Extents)>>
view(T* data, loom::detail::view_index_dimensions<Extents...> dimensions,
     loom::type::encoding<Role, sizeof...(Extents)> encoding);

// Forms a typed view at the root of an existing Loom buffer value.
template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T,
                 loom::type::encoding<Role, sizeof...(Extents)>>
view(loom::type::buffer<T> data,
     loom::detail::view_dimensions<Extents...> dimensions,
     loom::type::encoding<Role, sizeof...(Extents)> encoding);

// Forms an indexed dynamic view at the root of a Loom buffer value.
template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T,
                 loom::type::encoding<Role, sizeof...(Extents)>>
view(loom::type::buffer<T> data,
     loom::detail::view_index_dimensions<Extents...> dimensions,
     loom::type::encoding<Role, sizeof...(Extents)> encoding);

// Forms an ordinary unencoded view at a typed buffer root.
template <loom::type::size_type... Extents, class T>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T> view(
    loom::type::buffer<T> data,
    loom::detail::view_dimensions<Extents...> dimensions);

template <loom::type::size_type... Extents, class T>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T> view(
    loom::type::buffer<T> data,
    loom::detail::view_index_dimensions<Extents...> dimensions);

// Forms a view at a byte offset within a typed buffer. The offset remains in
// Loom's physical address domain and does not become pointer arithmetic.
template <loom::type::size_type... Extents, class T>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T> view(
    loom::type::buffer<T> data, loom::type::offset byte_offset,
    loom::detail::view_dimensions<Extents...> dimensions);

template <loom::type::size_type... Extents, class T>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T> view(
    loom::type::buffer<T> data, loom::type::offset byte_offset,
    loom::detail::view_index_dimensions<Extents...> dimensions);

template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T,
                 loom::type::encoding<Role, sizeof...(Extents)>>
view(loom::type::buffer<T> data, loom::type::offset byte_offset,
     loom::detail::view_dimensions<Extents...> dimensions,
     loom::type::encoding<Role, sizeof...(Extents)> encoding);

template <loom::type::size_type... Extents, class T, loom::encoding::role Role>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("buffer.view")]]
loom::type::view<loom::type::shape<Extents...>, T,
                 loom::type::encoding<Role, sizeof...(Extents)>>
view(loom::type::buffer<T> data, loom::type::offset byte_offset,
     loom::detail::view_index_dimensions<Extents...> dimensions,
     loom::type::encoding<Role, sizeof...(Extents)> encoding);

}  // namespace loom::buffer

namespace loom::view {

// Asserts Result's shape while preserving borrowed storage and access rights.
// An encoded Result retains the source encoding unless a replacement is
// supplied. An unencoded Result asserts ordinary unencoded addressing. No copy,
// repack, or runtime check occurs; incorrect assertions make the program
// invalid.
template <class Result, class Source>
[[loom::op("view.refine")]] Result refine(Source source);

template <class Result, class Source>
[[loom::op("view.refine")]] Result refine(
    Source source, loom::detail::shape_dimensions<Result> dimensions);

template <class Result, class Source, loom::encoding::role Role,
          loom::type::size_type Rank>
[[loom::op("view.refine")]] Result refine(
    Source source, loom::type::encoding<Role, Rank> encoding);

template <class Result, class Source, loom::encoding::role Role,
          loom::type::size_type Rank>
[[loom::op("view.refine")]] Result refine(
    Source source, loom::detail::shape_dimensions<Result> dimensions,
    loom::type::encoding<Role, Rank> encoding);

// Forms a subview with an explicitly selected result extent pattern.
template <loom::type::size_type... Extents, class T,
          loom::type::size_type... SourceExtents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(SourceExtents)>>
[[loom::op("view.subview")]]
loom::type::view<loom::type::shape<Extents...>, T, Encoding> subview(
    loom::type::view<loom::type::shape<SourceExtents...>, T, Encoding> source,
    Origin origin, loom::detail::view_dimensions<Extents...> dimensions);

// Forms a subview retaining result extents already in Loom's index domain.
template <loom::type::size_type... Extents, class T,
          loom::type::size_type... SourceExtents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(SourceExtents)>>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("view.subview")]]
loom::type::view<loom::type::shape<Extents...>, T, Encoding> subview(
    loom::type::view<loom::type::shape<SourceExtents...>, T, Encoding> source,
    Origin origin, loom::detail::view_index_dimensions<Extents...> dimensions);

// Forms a subview while preserving the source's static/dynamic extent pattern.
template <class T, loom::type::size_type... Extents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("view.subview")]]
loom::type::view<loom::type::shape<Extents...>, T, Encoding> subview(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, loom::detail::view_dimensions<Extents...> dimensions);

// Preserves a source's dynamic extent pattern and index-domain identities.
template <class T, loom::type::size_type... Extents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
  requires((Extents == loom::type::dynamic) || ...)
[[loom::op("view.subview")]]
loom::type::view<loom::type::shape<Extents...>, T, Encoding> subview(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, loom::detail::view_index_dimensions<Extents...> dimensions);

// Reinterprets one all-static plain view as another equal-sized dense shape.
// Storage, byte origin, access rights, and borrowed lifetime are preserved.
template <class Result, class Shape, class T, class Encoding>
[[loom::op("view.bitcast")]] Result bitcast(
    loom::type::view<Shape, T, Encoding> source);

// Loads one scalar at a logical coordinate.
template <loom::memory_access Access = loom::memory_access::none, class Shape,
          class T, class Encoding, class... Indices>
[[loom::op("view.load")]]
T load(loom::type::view<Shape, T, Encoding> source, Indices... indices);

// Observes one volatile element and returns its ordinary scalar value.
template <loom::memory_access Access = loom::memory_access::none, class Shape,
          class T, class Encoding, class... Indices>
[[loom::op("view.load")]]
T load(loom::type::view<Shape, volatile T, Encoding> source,
       Indices... indices);

// Stores one scalar at a logical coordinate. Template deduction
// rejects destinations whose element type is const.
template <loom::memory_access Access = loom::memory_access::none, class Shape,
          class T, class Encoding, class... Indices>
[[loom::op("view.store")]]
void store(T value, loom::type::view<Shape, T, Encoding> destination,
           Indices... indices);

// Observes a store through a volatile view without qualifying the scalar value.
template <loom::memory_access Access = loom::memory_access::none, class Shape,
          class T, class Encoding, class... Indices>
[[loom::op("view.store")]]
void store(T value, loom::type::view<Shape, volatile T, Encoding> destination,
           Indices... indices);

}  // namespace loom::view

#endif  // LOOMCXX_VIEW_H_
