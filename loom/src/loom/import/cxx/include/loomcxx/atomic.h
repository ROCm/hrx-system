// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_ATOMIC_H_
#define LOOMCXX_ATOMIC_H_

#include <loomcxx/view.h>

namespace loom::atomic {

// Read-modify-write operation kind supported by view and vector atomics.
enum class kind {
  // Integer exchange.
  xchgi = 0,
  // Floating-point exchange.
  xchgf = 1,
  // Integer addition.
  addi = 2,
  // Floating-point addition.
  addf = 3,
  // Integer subtraction.
  subi = 4,
  // Bitwise AND.
  andi = 5,
  // Bitwise OR.
  ori = 6,
  // Bitwise XOR.
  xori = 7,
  // Signed integer minimum.
  minsi = 8,
  // Signed integer maximum.
  maxsi = 9,
  // Unsigned integer minimum.
  minui = 10,
  // Unsigned integer maximum.
  maxui = 11,
  // IEEE 754 floating-point minimum.
  minimumf = 12,
  // IEEE 754 floating-point maximum.
  maximumf = 13,
  // C99 fmin-style floating-point minimum.
  minnumf = 14,
  // C99 fmax-style floating-point maximum.
  maxnumf = 15,
};

}  // namespace loom::atomic

// Scalar integer pointer and typed-view projections. A pointer identifies one
// live, naturally aligned integer object; its buffer and byte origin are
// retained. A view retains its complete rank, encoding, and logical indices.
// Loads accept const storage; updates require mutable storage. Boolean payloads
// are rejected. Volatile storage is accepted: each operation is already an
// observable atomic memory effect.
// Ordering and scope are explicit. The selected target must implement the
// requested width, memory space, and synchronization contract.
namespace loom::view::atomic {

// Observes one object without modifying it. Ordering is relaxed, acquire, or
// seq_cst. Every call is a distinct observation, including discarded results.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope,
          loom::memory_access Access = loom::memory_access::none, class T>
[[loom::op("view.atomic.load")]] T load(const volatile T* source);

// Observes one logical view element without rebuilding its address as a
// pointer. One index is required for every source axis.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope,
          loom::memory_access Access = loom::memory_access::none, class Shape,
          class T, class Encoding, class... Indices>
[[loom::op("view.atomic.load")]] T load(
    loom::type::view<Shape, T, Encoding> source, Indices... indices);

// Publishes one pointer object or logical view element without reading its old
// value. Indices is empty for a pointer and has one value per view axis.
// Ordering is relaxed, release, or seq_cst.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope,
          loom::memory_access Access = loom::memory_access::none, class T,
          class Storage, class... Indices>
[[loom::op("view.atomic.store")]] void store(T value, Storage destination,
                                             Indices... indices);

// Atomically combines value with a pointer object or logical view element and
// returns the old memory value. Signed and unsigned minimum/maximum kinds match
// the integer source type.
template <loom::atomic::kind Kind, loom::atomic::ordering Ordering,
          loom::atomic::scope Scope,
          loom::memory_access Access = loom::memory_access::none, class T,
          class Storage, class... Indices>
[[loom::op("view.atomic.rmw")]] T rmw(T value, Storage destination,
                                      Indices... indices);

// Performs the same pointer or logical-view update without returning its old
// value.
template <loom::atomic::kind Kind, loom::atomic::ordering Ordering,
          loom::atomic::scope Scope,
          loom::memory_access Access = loom::memory_access::none, class T,
          class Storage, class... Indices>
[[loom::op("view.atomic.reduce")]] void reduce(T value, Storage destination,
                                               Indices... indices);

// Stores replacement exactly when a pointer object or logical view element
// equals expected. Returns the old value on both success and failure, without
// modifying expected. Failure cannot release or be stronger than success;
// weak/spurious failure is not part of this operation.
template <loom::atomic::ordering Success, loom::atomic::ordering Failure,
          loom::atomic::scope Scope,
          loom::memory_access Access = loom::memory_access::none, class T,
          class Storage, class... Indices>
[[loom::op("view.atomic.cmpxchg")]] T cmpxchg(T expected, T replacement,
                                              Storage destination,
                                              Indices... indices);

}  // namespace loom::view::atomic

#endif  // LOOMCXX_ATOMIC_H_
