// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_FRAGMENT_H_
#define LOOMCXX_FRAGMENT_H_

#include <loomcxx/vector.h>

namespace loom::vector {

namespace fragment {

// Matrix operand interpretation attached to a physical vector. Names and
// ordinals match vector.fragment's role parameter.
enum class role : unsigned char { lhs = 0, rhs = 1, init = 2, result = 3 };

// Logical two-dimensional matrix fragment shape. Values remain ordinary SSA
// dimensions and may come from configs, target queries, or source arithmetic.
template <class Rows, class Columns>
struct shape {
  Rows rows;
  Columns columns;
};

// Independent matrix blocks followed by each block's two-dimensional shape.
template <class Blocks, class Rows, class Columns>
struct batched_shape {
  Blocks blocks;
  Rows rows;
  Columns columns;
};

// Attaches a logical role and shape to an ordinary physical vector. Parameters
// is a flat record whose field names identify explicit values such as schema,
// scale, codebook, or sparsity.
template <role Role, class Vector, class Shape>
[[loom::op("vector.fragment")]] Vector attach(Vector data, Shape logical_shape);

template <role Role, class Vector, class Shape, class Parameters>
[[loom::op("vector.fragment")]] Vector attach(Vector data, Shape logical_shape,
                                              Parameters parameters);

// Reinterprets or converts one native fragment carrier for another role. The
// result carrier is explicit because target providers own its physical shape.
template <role Role, class Result, class Source, class Shape>
[[loom::op("vector.fragment.repack")]] Result repack(Source source,
                                                     Shape logical_shape);

// Loads one target-shaped carrier at a full-rank logical origin. Auxiliary is
// a flat record of named runtime values required by the view's storage schema.
template <role Role, class Result,
          loom::memory_access Access = loom::memory_access::none, class T,
          loom::type::size_type... Extents, class Encoding, class Shape,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.fragment.load", "memory_access")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, Shape logical_shape);

template <role Role, class Result,
          loom::memory_access Access = loom::memory_access::none, class T,
          loom::type::size_type... Extents, class Encoding, class Shape,
          class Auxiliary,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.fragment.load", "memory_access")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, Shape logical_shape, Auxiliary auxiliary);

// Preserves a mixed static/dynamic logical origin in the source
// specialization. Each loom::type::dynamic entry consumes one value from the
// runtime origin record; all-static origins omit that empty record. Static
// origins spell Access before the pattern and use memory_access::none when the
// element type supplies the observation semantics.
template <role Role, class Result, loom::memory_access Access,
          loom::type::size_type... OriginPattern, class T,
          loom::type::size_type... Extents, class Encoding, class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern != loom::type::dynamic) && ...))
[[loom::op("vector.fragment.load", "memory_access")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Shape logical_shape);

template <role Role, class Result, loom::memory_access Access,
          loom::type::size_type... OriginPattern, class T,
          loom::type::size_type... Extents, class Encoding, class Shape,
          class Auxiliary>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern != loom::type::dynamic) && ...))
[[loom::op("vector.fragment.load", "memory_access")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Shape logical_shape, Auxiliary auxiliary);

template <role Role, class Result, loom::memory_access Access,
          loom::type::size_type... OriginPattern, class T,
          loom::type::size_type... Extents, class Encoding, class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern == loom::type::dynamic) || ...))
[[loom::op("vector.fragment.load", "memory_access")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    loom::type::indices<(loom::type::size_type{0} + ... +
                         (OriginPattern == loom::type::dynamic
                              ? loom::type::size_type{1}
                              : loom::type::size_type{0}))>
        dynamic_origin,
    Shape logical_shape);

template <role Role, class Result, loom::memory_access Access,
          loom::type::size_type... OriginPattern, class T,
          loom::type::size_type... Extents, class Encoding, class Shape,
          class Auxiliary>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern == loom::type::dynamic) || ...))
[[loom::op("vector.fragment.load", "memory_access")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    loom::type::indices<(loom::type::size_type{0} + ... +
                         (OriginPattern == loom::type::dynamic
                              ? loom::type::size_type{1}
                              : loom::type::size_type{0}))>
        dynamic_origin,
    Shape logical_shape, Auxiliary auxiliary);

// Stores one target-shaped carrier at a full-rank logical origin.
template <role Role, loom::memory_access Access = loom::memory_access::none,
          class Vector, class T, loom::type::size_type... Extents,
          class Encoding, class Shape,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.fragment.store", "memory_access")]] void store(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Origin origin, Shape logical_shape);

template <role Role, loom::memory_access Access,
          loom::type::size_type... OriginPattern, class Vector, class T,
          loom::type::size_type... Extents, class Encoding, class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern != loom::type::dynamic) && ...))
[[loom::op("vector.fragment.store", "memory_access")]] void store(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Shape logical_shape);

template <role Role, loom::memory_access Access,
          loom::type::size_type... OriginPattern, class Vector, class T,
          loom::type::size_type... Extents, class Encoding, class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern == loom::type::dynamic) || ...))
[[loom::op("vector.fragment.store", "memory_access")]] void store(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    loom::type::indices<(loom::type::size_type{0} + ... +
                         (OriginPattern == loom::type::dynamic
                              ? loom::type::size_type{1}
                              : loom::type::size_type{0}))>
        dynamic_origin,
    Shape logical_shape);

}  // namespace fragment

// Semantic permissions for matrix multiply-accumulate.
enum class mma_flags : unsigned char { none = 0, saturate = 1 };

// Multiplies logical lhs/rhs fragments and accumulates into init. Fragment
// role, shape, schema, and dynamic parameters come from the operand facts.
template <mma_flags Flags = mma_flags::none, class Left, class Right,
          class Accumulator>
[[loom::op("vector.mma")]] Accumulator mma(Left lhs, Right rhs,
                                           Accumulator init);

}  // namespace loom::vector

#endif  // LOOMCXX_FRAGMENT_H_
