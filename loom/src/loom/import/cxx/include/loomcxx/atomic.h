// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_ATOMIC_H_
#define LOOMCXX_ATOMIC_H_

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

// Atomic memory ordering between memory accesses.
enum class ordering {
  // Atomicity without inter-address synchronization.
  relaxed = 0,
  // Acquire ordering.
  acquire = 1,
  // Release ordering.
  release = 2,
  // Acquire and release ordering.
  acq_rel = 3,
  // Sequentially consistent ordering.
  seq_cst = 4,
};

// Synchronization scope for atomic memory effects.
enum class scope {
  // Current invocation or thread.
  thread = 0,
  // Current SIMD subgroup or wave.
  subgroup = 1,
  // Current workgroup or block.
  workgroup = 2,
  // Current device.
  device = 3,
  // Whole system.
  system = 4,
};

}  // namespace loom::atomic

// Scalar integer pointer projections. The pointer identifies one live,
// naturally aligned integer object; its buffer and byte origin are retained.
// Loads accept const storage; updates require mutable storage. Boolean payloads
// are rejected. Volatile pointers are accepted: each operation is already an
// observable atomic memory effect.
// Ordering and scope are explicit. The selected target must implement the
// requested width, memory space, and synchronization contract.
namespace loom::view::atomic {

// Observes one object without modifying it. Ordering is relaxed, acquire, or
// seq_cst. Every call is a distinct observation, including discarded results.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope, class T>
[[loom::op("view.atomic.load")]] T load(const volatile T* source);

// Publishes one object without reading its old value. Ordering is relaxed,
// release, or seq_cst.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope, class T>
[[loom::op("view.atomic.store")]] void store(T value, volatile T* destination);

// Atomically combines value with memory and returns the old memory value.
// Signed and unsigned minimum/maximum kinds match the integer source type.
template <loom::atomic::kind Kind, loom::atomic::ordering Ordering,
          loom::atomic::scope Scope, class T>
[[loom::op("view.atomic.rmw")]] T rmw(T value, volatile T* destination);

// Performs the same atomic update without returning the old value.
template <loom::atomic::kind Kind, loom::atomic::ordering Ordering,
          loom::atomic::scope Scope, class T>
[[loom::op("view.atomic.reduce")]] void reduce(T value,
                                               volatile T* destination);

// Stores replacement exactly when memory equals expected. Returns the old
// value on both success and failure, without modifying expected. Failure
// cannot release or be stronger than success; weak/spurious failure is
// not part of this operation.
template <loom::atomic::ordering Success, loom::atomic::ordering Failure,
          loom::atomic::scope Scope, class T>
[[loom::op("view.atomic.cmpxchg")]] T cmpxchg(T expected, T replacement,
                                              volatile T* destination);

}  // namespace loom::view::atomic

namespace loom::buffer {

// Orders the executing thread's memory accesses across storage objects.
// Ordering is acquire, release, acq_rel, or seq_cst. Matching atomic
// observations and publications establish synchronization. The fence does not
// rendezvous with other threads or complete independent asynchronous transfers.
// It is valid in ordinary functions as well as kernels.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope>
[[loom::op("buffer.fence")]] void fence();

}  // namespace loom::buffer

#endif  // LOOMCXX_ATOMIC_H_
