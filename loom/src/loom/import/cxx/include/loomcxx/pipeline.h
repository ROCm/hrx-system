// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_PIPELINE_H_
#define LOOMCXX_PIPELINE_H_

#include <loomcxx/view.h>

namespace loom::channel {

// Delivery order for records admitted to a channel.
enum class discipline : unsigned char {
  // Reservations and acceptances observe first-in, first-out order.
  fifo = 1,
};

// Permission carried by an owned consuming access.
enum class mode : unsigned char {
  // Immutable payload access. This is the unqualified read type in Loom IR.
  immutable = 0,
  // Exclusive initialized payload access permitting mutation.
  mutable_ = 1,
};

}  // namespace loom::channel

namespace loom::type {

// A tile-level aggregate whose dimensions precede its element type, matching
// Loom IR. Each dynamic dimension and optional encoding travel with the value
// through functions, records, and structured control flow.
template <class Shape, class T, class Encoding = void>
class [[loom::type("tile")]] tile {
 public:
  using shape_type = Shape;
};

// A borrowed allocation resource selected for one enclosing invocation.
class [[loom::type("pool")]] pool {};

// A communication identity carrying records of Payload. Storage, capacity,
// and transport belong to its binding rather than this handle type.
template <class Payload>
class [[loom::type("channel")]] channel {};

// Owned consumption of one channel record. Immutable access is the default;
// mutable access remains exclusive until released or transferred.
template <class Payload,
          loom::channel::mode Mode = loom::channel::mode::immutable>
class [[loom::type("read")]] read {};

// Owned producer reservation for one channel record. Publication is explicit.
template <class Payload>
class [[loom::type("write")]] write {};

}  // namespace loom::type

namespace loom::tile {

// Asserts Result's shape or encoding for the same logical tile. An omitted
// encoding on an encoded Result retains the source encoding. Explicit
// replacements assert identity; they do not convert, repack, or insert a
// runtime check. Incorrect assertions make the program invalid.
template <class Result, class Source>
[[loom::op("tile.refine")]] Result refine(Source source);

template <class Result, class Source>
[[loom::op("tile.refine")]] Result refine(
    Source source, loom::detail::shape_dimensions<Result> dimensions);

template <class Result, class Source, loom::encoding::role Role,
          loom::type::size_type Rank>
[[loom::op("tile.refine")]] Result refine(
    Source source, loom::type::encoding<Role, Rank> encoding);

template <class Result, class Source, loom::encoding::role Role,
          loom::type::size_type Rank>
[[loom::op("tile.refine")]] Result refine(
    Source source, loom::detail::shape_dimensions<Result> dimensions,
    loom::type::encoding<Role, Rank> encoding);

}  // namespace loom::tile

namespace loom::detail {

// Channel payload encodings that also describe memory addressing survive on
// borrowed views. Numeric transforms and schemas remain payload semantics.
template <class Encoding>
struct channel_view_encoding {
  using type = void;
};

template <loom::type::size_type Rank>
struct channel_view_encoding<
    loom::type::encoding<loom::encoding::role::layout, Rank>> {
  using type = loom::type::encoding<loom::encoding::role::layout, Rank>;
};

template <loom::type::size_type Rank>
struct channel_view_encoding<
    loom::type::encoding<loom::encoding::role::storage, Rank>> {
  using type = loom::type::encoding<loom::encoding::role::storage, Rank>;
};

template <class Payload>
struct channel_write_view;

template <class Shape, class T, class Encoding>
struct channel_write_view<loom::type::tile<Shape, T, Encoding>> {
  using type = loom::type::view<Shape, T,
                                typename channel_view_encoding<Encoding>::type>;
};

template <class Payload, loom::channel::mode Mode>
struct channel_read_view;

template <class Shape, class T, class Encoding>
struct channel_read_view<loom::type::tile<Shape, T, Encoding>,
                         loom::channel::mode::immutable> {
  using type = loom::type::view<Shape, const T,
                                typename channel_view_encoding<Encoding>::type>;
};

template <class Shape, class T, class Encoding>
struct channel_read_view<loom::type::tile<Shape, T, Encoding>,
                         loom::channel::mode::mutable_> {
  using type = loom::type::view<Shape, T,
                                typename channel_view_encoding<Encoding>::type>;
};

template <class Payload>
using channel_write_view_t = typename channel_write_view<Payload>::type;

template <class Payload, loom::channel::mode Mode>
using channel_read_view_t = typename channel_read_view<Payload, Mode>::type;

}  // namespace loom::detail

namespace loom::channel {

// Producer ownership and its borrowed writable record storage.
template <class Payload>
struct reservation {
  loom::type::write<Payload> write;
  loom::detail::channel_write_view_t<Payload> payload;
};

// Consumer ownership and the borrowed initialized record storage it protects.
template <class Payload, mode Mode = mode::immutable>
struct acquisition {
  loom::type::read<Payload, Mode> read;
  loom::detail::channel_read_view_t<Payload, Mode> payload;
};

// Producer identity selected from a relation-bound channel and its record.
template <class Payload>
struct selection {
  loom::type::index source_rank;
  loom::type::read<Payload> read;
};

// Independently retiring immutable reads produced by one ownership transfer.
template <loom::type::size_type Count, class Payload>
struct fanout_result {
  loom::type::read<Payload> first;
  fanout_result<Count - 1, Payload> remaining;
};

template <class Payload>
struct fanout_result<2, Payload> {
  loom::type::read<Payload> first;
  loom::type::read<Payload> second;
};

template <class Payload>
struct fanout_result<1, Payload> {
  loom::type::read<Payload> first;
};

template <class Payload>
struct fanout_result<0, Payload> {};

// Binds caller-owned slot storage to a fresh FIFO communication identity. The
// leading storage dimension counts slots; remaining dimensions form Payload.
template <discipline Discipline, loom::type::size_type Slots,
          loom::type::size_type... Extents, class T, class StorageEncoding>
[[loom::op("channel.bind")]]
loom::type::channel<loom::type::tile<loom::type::shape<Extents...>, T>> bind(
    loom::type::view<loom::type::shape<Slots, Extents...>, T, StorageEncoding>
        storage,
    loom::type::index capacity);

// Carries an encoded payload independently of the rank-one-larger slot
// storage layout. The encoding is an ordinary SSA dependency of the channel.
template <discipline Discipline, loom::type::size_type Slots,
          loom::type::size_type... Extents, class T, class StorageEncoding,
          class PayloadEncoding>
[[loom::op("channel.bind")]]
loom::type::channel<
    loom::type::tile<loom::type::shape<Extents...>, T, PayloadEncoding>>
bind(loom::type::view<loom::type::shape<Slots, Extents...>, T, StorageEncoding>
         storage,
     loom::type::index capacity, PayloadEncoding payload_encoding);

// Reserves one uninitialized producer record and borrows its writable storage.
template <class Payload>
[[loom::op("channel.reserve")]] reservation<Payload> reserve(
    loom::type::channel<Payload> channel);

// Accepts the next record without waiting for payload readiness.
template <mode Mode = mode::immutable, class Payload>
[[loom::op("channel.accept")]] loom::type::read<Payload, Mode> accept(
    loom::type::channel<Payload> channel);

// Waits for one accepted record and borrows its initialized payload.
template <class Payload, mode Mode>
[[loom::op("channel.wait")]]
loom::detail::channel_read_view_t<Payload, Mode> wait(
    loom::type::read<Payload, Mode> read);

// Accepts and waits for the next record in one operation.
template <mode Mode = mode::immutable, class Payload>
[[loom::op("channel.acquire")]] acquisition<Payload, Mode> acquire(
    loom::type::channel<Payload> channel);

// Publishes an initialized producer record and consumes its reservation.
template <class Payload>
[[loom::op("channel.publish")]] void publish(loom::type::write<Payload> write);

// Retires a consuming obligation after its last payload use.
template <class Payload, mode Mode>
[[loom::op("channel.release")]] void release(
    loom::type::read<Payload, Mode> read);

// Transfers one immutable read into Count independently retiring reads.
template <loom::type::size_type Count, class Payload>
[[loom::op("channel.fanout")]] fanout_result<Count, Payload> fanout(
    loom::type::read<Payload> read);

// Transfers source and destination ownership to asynchronous transport.
template <class Payload, mode Mode>
[[loom::op("channel.copy")]] void copy(loom::type::read<Payload, Mode> source,
                                       loom::type::write<Payload> destination);

// Waits for any ready endpoint of a relation-bound channel.
template <class Payload>
[[loom::op("channel.select")]] selection<Payload> select(
    loom::type::channel<Payload> channel);

}  // namespace loom::channel

namespace loom::pipeline {

// Selects the pool at one worker coordinate tuple in the enclosing invocation.
template <loom::memory_space Space, loom::type::size_type Rank>
[[loom::op("pipeline.memory")]] loom::type::pool memory(
    loom::type::indices<Rank> coordinates);

// Constructs one fresh child graph and substitutes its complete staged
// signature. Within [[loom::serial]], the child's strands and transfers finish
// before the next child starts. Outside that region, children run concurrently.
// All children remain part of their enclosing resident invocation.
template <auto Pipeline, class... Args>
[[loom::op("pipeline.compose")]] void compose(Args... arguments);

}  // namespace loom::pipeline

namespace loom::buffer {

// Selects an explicit backing resource for a fresh element allocation.
template <class T, loom::memory_space Space,
          loom::type::size_type Alignment = alignof(T)>
[[loom::op("buffer.alloca")]] T* alloca(loom::type::size_type element_count,
                                        loom::type::pool pool);

// Retains a native logical count while selecting the backing resource.
template <class T, loom::memory_space Space,
          loom::type::size_type Alignment = alignof(T)>
[[loom::op("buffer.alloca")]] T* alloca(loom::type::index element_count,
                                        loom::type::pool pool);

// Allocates an exact physical byte extent from the selected backing resource.
template <loom::memory_space Space, loom::type::size_type Alignment = 1>
[[loom::op("buffer.alloca")]] unsigned char* alloca(
    loom::type::offset byte_length, loom::type::pool pool);

}  // namespace loom::buffer

#endif  // LOOMCXX_PIPELINE_H_
