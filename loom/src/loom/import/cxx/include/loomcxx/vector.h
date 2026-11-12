// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_VECTOR_H_
#define LOOMCXX_VECTOR_H_

#include <loomcxx/view.h>

namespace loom::detail {

template <class T>
concept vector_arithmetic_element =
    __is_floating_point(T) || (__is_integral(T) && !__is_same(T, bool));

template <class T>
concept vector_integer_element = __is_integral(T) && !__is_same(T, bool);

template <class T>
concept vector_bitwise_element = __is_integral(T);

}  // namespace loom::detail

namespace loom::type {

// A rank-generic register value whose logical dimensions precede its element
// type, matching Loom vector syntax. Native C++ vectors remain the concise
// spelling for static rank-one values; this type retains dependent dimensions
// through functions, records, templates, and structured control flow.
template <class Shape, class T>
class [[loom::type("vector")]] vector {};

// Lanewise arithmetic preserves Shape and T. For dependent shapes, binary
// results retain the left value's dynamic dimensions and assert that the right
// value describes the same logical lane grid.
template <class Shape, class T>
  requires(loom::detail::vector_arithmetic_element<T>)
[[loom::op("vector.add")]] vector<Shape, T> operator+(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_arithmetic_element<T>)
[[loom::op("vector.sub")]] vector<Shape, T> operator-(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_arithmetic_element<T>)
[[loom::op("vector.mul")]] vector<Shape, T> operator*(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_arithmetic_element<T>)
[[loom::op("vector.div")]] vector<Shape, T> operator/(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_arithmetic_element<T>)
[[loom::op("vector.rem")]] vector<Shape, T> operator%(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_arithmetic_element<T>)
[[loom::op("vector.neg")]] vector<Shape, T> operator-(vector<Shape, T> input);
template <class Shape, class T>
  requires(loom::detail::vector_bitwise_element<T>)
[[loom::op("vector.and")]] vector<Shape, T> operator&(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_bitwise_element<T>)
[[loom::op("vector.or")]] vector<Shape, T> operator|(vector<Shape, T> lhs,
                                                     vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_bitwise_element<T>)
[[loom::op("vector.xor")]] vector<Shape, T> operator^(vector<Shape, T> lhs,
                                                      vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_bitwise_element<T>)
[[loom::op("vector.complement")]] vector<Shape, T> operator~(
    vector<Shape, T> input);
template <class Shape, class T>
  requires(loom::detail::vector_integer_element<T>)
[[loom::op("vector.shl")]] vector<Shape, T> operator<<(vector<Shape, T> lhs,
                                                       vector<Shape, T> rhs);
template <class Shape, class T>
  requires(loom::detail::vector_integer_element<T>)
[[loom::op("vector.shr")]] vector<Shape, T> operator>>(vector<Shape, T> lhs,
                                                       vector<Shape, T> rhs);

}  // namespace loom::type

namespace loom::detail {

template <class Vector>
struct vector_traits {
  // Native C++ vectors have one axis; source admission rejects other types.
  static constexpr loom::type::size_type rank = 1;
};

template <loom::type::size_type... Extents, class T>
struct vector_traits<loom::type::vector<loom::type::shape<Extents...>, T>> {
  // Number of logical axes, including those with dependent extents.
  static constexpr loom::type::size_type rank = sizeof...(Extents);
  using element_type = T;
  using dimensions_type = loom::type::indices<(
      loom::type::size_type{0} + ... +
      (Extents == loom::type::dynamic ? loom::type::size_type{1}
                                      : loom::type::size_type{0}))>;
};

template <class Vector>
using vector_element = typename vector_traits<Vector>::element_type;

template <class Vector>
using vector_dimensions = typename vector_traits<Vector>::dimensions_type;

}  // namespace loom::detail

namespace loom::vector {

// Asserts that source describes Result's lane grid without a conversion, copy,
// or runtime check. Rank and lane type stay exact. An incorrect dynamic shape
// assertion makes the program invalid; contradictory static extents diagnose.
template <class Result, class Source>
[[loom::op("vector.refine")]] Result refine(Source source);

// Supplies one native index per dynamic result axis, in source-axis order.
template <class Result, class Source>
[[loom::op("vector.refine")]] Result refine(
    Source source, loom::detail::vector_dimensions<Result> dimensions);

// Replicates one scalar across a semantic vector. Dimensions supplies one
// index for each dynamic result axis in source-axis order.
template <class Result>
[[loom::op("vector.splat")]] Result splat(
    loom::detail::vector_dimensions<Result> dimensions,
    loom::detail::vector_element<Result> value);

// Loads one contiguous native C++ vector at a full-rank logical origin. Result
// selects its static register shape while the view supplies element type,
// layout, extents, alignment, and observation semantics.
template <class Result, loom::memory_access Access = loom::memory_access::none,
          class T, loom::type::size_type... Extents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin);

// Loads one semantic vector at a full-rank logical origin. Dimensions supplies
// one index for each dynamic result axis in source-axis order.
template <class Result, loom::memory_access Access = loom::memory_access::none,
          class T, loom::type::size_type... Extents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, loom::detail::vector_dimensions<Result> dimensions);

// Stores one contiguous vector at a full-rank logical origin. A semantic value
// carries its dynamic dimensions. Const destination element types are rejected
// during source admission.
template <loom::memory_access Access = loom::memory_access::none, class Vector,
          class T, loom::type::size_type... Extents, class Encoding,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.store")]] void store(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Origin origin);

// Loads active lanes from their corresponding footprint positions and takes
// inactive lanes from passthrough. Nonzero integer mask lanes are active.
template <class Result, class T, loom::type::size_type... Extents,
          class Encoding, class Mask,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.load.mask")]] Result load_mask(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, Mask mask, Result passthrough);

// Stores value lanes whose corresponding integer mask lane is nonzero.
template <class Vector, class T, loom::type::size_type... Extents,
          class Encoding, class Mask,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.store.mask")]] void store_mask(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Origin origin, Mask mask);

// Densely consumes consecutive source elements for active lanes and takes
// inactive lanes from passthrough. Nonzero integer mask lanes are active.
template <class Result, class T, loom::type::size_type... Extents,
          class Encoding, class Mask,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.load.expand")]] Result load_expand(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, Mask mask, Result passthrough);

// Densely writes active value lanes to consecutive destination elements.
template <class Vector, class T, loom::type::size_type... Extents,
          class Encoding, class Mask,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.store.compress")]] void store_compress(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Origin origin, Mask mask);

// Extracts a rank-preserving register subvector. Offsets supplies one index or
// integer per source axis; rank-one vectors also accept a scalar offset.
// Result keeps the source element type and selects the retained static shape.
template <class Result, class Source,
          class Offsets = loom::type::coordinates<
              loom::detail::vector_traits<Source>::rank>>
[[loom::op("vector.slice")]] Result slice(Source source, Offsets offsets);

// Supplies one logical index for each dynamic result axis in source-axis
// order. Source dimensions remain attached to source; they do not determine
// how much of each axis the result retains.
template <class Result, class Source,
          class Offsets = loom::type::coordinates<
              loom::detail::vector_traits<Source>::rank>>
[[loom::op("vector.slice")]] Result slice(
    Source source, Offsets offsets,
    loom::detail::vector_dimensions<Result> dimensions);

// Constructs row-major lane coordinates from scalar base and step values.
// Static native or semantic result shapes are selected explicitly.
template <class Result, class Scalar>
[[loom::op("vector.iota")]] Result iota(Scalar base, Scalar step);

// Concatenates same-rank vectors along Axis. Result makes the widened shape
// explicit while input carriers may be native or semantic vectors.
template <loom::type::size_type Axis, class Result, class... Inputs>
[[loom::op("vector.concat")]] Result concat(Inputs... inputs);

// Permutes register axes. Each selector maps its result axis to one source
// axis, matching vector.transpose's result-to-source ordering.
template <class Result, loom::type::size_type... Permutation, class Source>
[[loom::op("vector.transpose")]] Result transpose(Source source);

// Interleaves even and odd coordinates along Axis into a doubled result axis.
template <loom::type::size_type Axis, class Result, class Vector>
[[loom::op("vector.interleave")]] Result interleave(Vector even, Vector odd);

// Reinterprets an all-static vector register value without numeric conversion.
template <class Result, class Source>
[[loom::op("vector.bitcast")]] Result bitcast(Source source);

// Extracts a scalar or tail vector at compile-time leading coordinates.
template <class Result, loom::type::size_type... StaticIndices, class Source>
  requires(sizeof...(StaticIndices) > 0)
[[loom::op("vector.extract")]] Result extract(Source source);

// Extracts a scalar or tail vector at runtime leading coordinates.
template <class Result, class Source, class... Indices>
  requires(sizeof...(Indices) > 0)
[[loom::op("vector.extract")]] Result extract(Source source,
                                              Indices... indices);

// Selects each result lane from true_value or false_value using an i1 mask.
template <class Condition, class Vector>
[[loom::op("vector.select")]] Vector select(Condition condition,
                                            Vector true_value,
                                            Vector false_value);

// Exact integer comparison predicates. Result is explicit because native C++
// vector comparisons produce integer masks while Loom comparisons produce i1
// masks. Signed and unsigned relational names match vector.cmpi.
namespace cmpi {
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "eq")]] Result eq(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "ne")]] Result ne(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "slt")]] Result slt(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "sle")]] Result sle(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "sgt")]] Result sgt(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "sge")]] Result sge(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "ult")]] Result ult(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "ule")]] Result ule(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "ugt")]] Result ugt(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpi", "uge")]] Result uge(Left lhs, Right rhs);
}  // namespace cmpi

// IEEE 754 ordered and unordered floating-point comparison predicates. Names
// and NaN behavior match vector.cmpf exactly.
namespace cmpf {
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "oeq")]] Result oeq(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ogt")]] Result ogt(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "oge")]] Result oge(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "olt")]] Result olt(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ole")]] Result ole(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "one")]] Result one(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ord")]] Result ord(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ueq")]] Result ueq(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ugt")]] Result ugt(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "uge")]] Result uge(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ult")]] Result ult(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "ule")]] Result ule(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "une")]] Result une(Left lhs, Right rhs);
template <class Result, class Left, class Right>
[[loom::op("vector.cmpf", "uno")]] Result uno(Left lhs, Right rhs);
}  // namespace cmpf

// Computes the absolute value of every floating-point lane.
template <class Vector>
[[loom::op("vector.absf")]] Vector absf(Vector input);

// Computes the signed absolute value of every integer lane.
template <class Vector>
[[loom::op("vector.absi")]] Vector absi(Vector input);

// Computes IEEE 754 minimum lane-wise, propagating NaNs.
template <class Vector>
[[loom::op("vector.minimumf")]] Vector minimumf(Vector lhs, Vector rhs);

// Computes IEEE 754 maximum lane-wise, propagating NaNs.
template <class Vector>
[[loom::op("vector.maximumf")]] Vector maximumf(Vector lhs, Vector rhs);

// Computes C99 fmin-style minimum lane-wise, selecting a non-NaN operand.
template <class Vector>
[[loom::op("vector.minnumf")]] Vector minnumf(Vector lhs, Vector rhs);

// Computes C99 fmax-style maximum lane-wise, selecting a non-NaN operand.
template <class Vector>
[[loom::op("vector.maxnumf")]] Vector maxnumf(Vector lhs, Vector rhs);

// Computes signed integer minimum lane-wise.
template <class Vector>
[[loom::op("vector.minsi")]] Vector minsi(Vector lhs, Vector rhs);

// Computes signed integer maximum lane-wise.
template <class Vector>
[[loom::op("vector.maxsi")]] Vector maxsi(Vector lhs, Vector rhs);

// Computes unsigned integer minimum lane-wise.
template <class Vector>
[[loom::op("vector.minui")]] Vector minui(Vector lhs, Vector rhs);

// Computes unsigned integer maximum lane-wise.
template <class Vector>
[[loom::op("vector.maxui")]] Vector maxui(Vector lhs, Vector rhs);

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
template <role Role, class Result, class T, loom::type::size_type... Extents,
          class Encoding, class Shape,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.fragment.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, Shape logical_shape);

template <role Role, class Result, class T, loom::type::size_type... Extents,
          class Encoding, class Shape, class Auxiliary,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.fragment.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Origin origin, Shape logical_shape, Auxiliary auxiliary);

// Preserves a mixed static/dynamic logical origin in the source
// specialization. Each loom::type::dynamic entry consumes one value from the
// runtime origin record; all-static origins omit that empty record.
template <role Role, class Result, loom::type::size_type... OriginPattern,
          class T, loom::type::size_type... Extents, class Encoding,
          class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern != loom::type::dynamic) && ...))
[[loom::op("vector.fragment.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Shape logical_shape);

template <role Role, class Result, loom::type::size_type... OriginPattern,
          class T, loom::type::size_type... Extents, class Encoding,
          class Shape, class Auxiliary>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern != loom::type::dynamic) && ...))
[[loom::op("vector.fragment.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    Shape logical_shape, Auxiliary auxiliary);

template <role Role, class Result, loom::type::size_type... OriginPattern,
          class T, loom::type::size_type... Extents, class Encoding,
          class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern == loom::type::dynamic) || ...))
[[loom::op("vector.fragment.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    loom::type::indices<(loom::type::size_type{0} + ... +
                         (OriginPattern == loom::type::dynamic
                              ? loom::type::size_type{1}
                              : loom::type::size_type{0}))>
        dynamic_origin,
    Shape logical_shape);

template <role Role, class Result, loom::type::size_type... OriginPattern,
          class T, loom::type::size_type... Extents, class Encoding,
          class Shape, class Auxiliary>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern == loom::type::dynamic) || ...))
[[loom::op("vector.fragment.load")]] Result load(
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> source,
    loom::type::indices<(loom::type::size_type{0} + ... +
                         (OriginPattern == loom::type::dynamic
                              ? loom::type::size_type{1}
                              : loom::type::size_type{0}))>
        dynamic_origin,
    Shape logical_shape, Auxiliary auxiliary);

// Stores one target-shaped carrier at a full-rank logical origin.
template <role Role, class Vector, class T, loom::type::size_type... Extents,
          class Encoding, class Shape,
          class Origin = loom::type::coordinates<sizeof...(Extents)>>
[[loom::op("vector.fragment.store")]] void store(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Origin origin, Shape logical_shape);

template <role Role, loom::type::size_type... OriginPattern, class Vector,
          class T, loom::type::size_type... Extents, class Encoding,
          class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern != loom::type::dynamic) && ...))
[[loom::op("vector.fragment.store")]] void store(
    Vector value,
    loom::type::view<loom::type::shape<Extents...>, T, Encoding> destination,
    Shape logical_shape);

template <role Role, loom::type::size_type... OriginPattern, class Vector,
          class T, loom::type::size_type... Extents, class Encoding,
          class Shape>
  requires(sizeof...(OriginPattern) == sizeof...(Extents) &&
           ((OriginPattern == loom::type::dynamic) || ...))
[[loom::op("vector.fragment.store")]] void store(
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

// Groups adjacent four-lane byte products into signed i32 accumulator lanes.
// Each input lane's C++ signedness selects its interpretation, covering the
// s8s8, u8s8, s8u8, and u8u8 vector.dot4i variants without a separate source
// tag. Inputs have the same shape and four times the accumulator's lane count.
template <class Left, class Right, class Accumulator>
[[loom::op("vector.dot4i")]] Accumulator dot4i(Left lhs, Right rhs,
                                               Accumulator accumulator);

// Decodes a physical vector into Result using an encoding<schema>. Auxiliary
// is an ordinary aggregate whose vector fields name schema operands such as
// scale, zero_point, or codebook. Values remain explicit SSA operands; a schema
// does not capture them. Payload, Schema and Auxiliary are deduced from values.
template <class Result, class Payload, class Schema, class Auxiliary>
[[loom::op("vector.decode")]] Result decode(Payload payload, Schema schema,
                                            Auxiliary auxiliary);

// Decodes a schema that requires no auxiliary values.
template <class Result, class Payload, class Schema>
[[loom::op("vector.decode")]] Result decode(Payload payload, Schema schema);

// Accumulates adjacent FP16 or BF16 pairs into F32 lanes using target-native
// grouped arithmetic. Both inputs have the same type and twice the
// accumulator's lane count. Intermediate precision and rounding can differ from
// two ordered F32 FMAs; folding and scalar expansion use that ordered reference
// evaluation. Input and accumulator types are deduced independently from the
// arguments.
template <class Input, class Accumulator>
[[loom::op("vector.dot2f")]] Accumulator dot2f(Input lhs, Input rhs,
                                               Accumulator accumulator);

// Accumulates fused products in logical lane order starting from init. Inputs
// have the same vector type; the scalar seed and result match its element type.
// Separately rounded products use vector multiplication followed by reduction.
template <class Vector, class Scalar>
[[loom::op("vector.dotf")]] Scalar dotf(Vector lhs, Vector rhs, Scalar init);

// Seeded reductions preserve the vector's element type. The importer checks
// concrete shapes and element interpretations when a specialization is used.
// Floating reductions retain lane order; a custom loom::op declaration can
// explicitly grant permissions such as reassoc or contract.
namespace reduce {

// Integer addition with wrapping at the element width.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "addi")]] Scalar addi(Vector input, Scalar init);

// Floating-point addition.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "addf")]] Scalar addf(Vector input, Scalar init);

// Integer multiplication with wrapping at the element width.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "muli")]] Scalar muli(Vector input, Scalar init);

// Floating-point multiplication.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "mulf")]] Scalar mulf(Vector input, Scalar init);

// Signed integer minimum.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "minsi")]] Scalar minsi(Vector input, Scalar init);

// Signed integer maximum.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "maxsi")]] Scalar maxsi(Vector input, Scalar init);

// Unsigned integer minimum.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "minui")]] Scalar minui(Vector input, Scalar init);

// Unsigned integer maximum.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "maxui")]] Scalar maxui(Vector input, Scalar init);

// Bitwise AND.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "andi")]] Scalar andi(Vector input, Scalar init);

// Bitwise OR.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "ori")]] Scalar ori(Vector input, Scalar init);

// Bitwise XOR.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "xori")]] Scalar xori(Vector input, Scalar init);

// IEEE 754 minimum, propagating NaNs.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "minimumf")]] Scalar minimumf(Vector input,
                                                          Scalar init);

// IEEE 754 maximum, propagating NaNs.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "maximumf")]] Scalar maximumf(Vector input,
                                                          Scalar init);

// C99 fmin-style minimum, selecting the number when only one input is NaN.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "minnumf")]] Scalar minnumf(Vector input,
                                                        Scalar init);

// C99 fmax-style maximum, selecting the number when only one input is NaN.
template <class Vector, class Scalar>
[[loom::op("vector.reduce", "maxnumf")]] Scalar maxnumf(Vector input,
                                                        Scalar init);

}  // namespace reduce
}  // namespace loom::vector

#endif  // LOOMCXX_VECTOR_H_
