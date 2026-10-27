// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_BUFFER_H_
#define LOOMCXX_BUFFER_H_

namespace loom {

namespace type {

class index;
class offset;

// A borrowed, zero-origin buffer root with source-visible element type T.
// Unlike T*, this is one Loom buffer value and has no independent byte offset.
// Form a typed view before accessing its storage.
template <class T>
class [[loom::type("buffer")]] buffer {
  // Source object representation preserving ordinary trivial-copy semantics.
  T* data_;
};

}  // namespace type

namespace atomic {

// Memory ordering between synchronized accesses.
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

// Synchronization scope for memory effects.
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

}  // namespace atomic

// Storage domain attached to an allocated or externally supplied buffer root.
// Target lowering decides which domains are legal for each program kind.
enum class memory_space {
  // Device-visible global storage.
  global = 1,
  // Storage shared by invocations within a workgroup.
  workgroup = 2,
  // Storage private to one function or kernel invocation.
  private_ = 3,
  // Immutable target storage.
  constant = 4,
  // Host-visible storage.
  host = 5,
  // Descriptor storage interpreted by a target interface.
  descriptor = 6,
  // Storage whose concrete domain is selected during lowering.
  generic = 7,
};

namespace buffer {

// Allocates element_count uninitialized T objects in Space. Each execution
// produces a distinct buffer root. Alignment is a minimum byte alignment and
// cannot weaken T's natural alignment.
template <class T, loom::memory_space Space,
          __SIZE_TYPE__ Alignment = alignof(T)>
[[loom::op("buffer.alloca")]] T* alloca(__SIZE_TYPE__ element_count);

// Retains a native logical element count without exposing its target-selected
// carrier through source code.
template <class T, loom::memory_space Space,
          __SIZE_TYPE__ Alignment = alignof(T)>
[[loom::op("buffer.alloca")]] T* alloca(loom::type::index element_count);

// Allocates an exact physical byte extent. This form preserves a byte count
// already computed in Loom's offset domain instead of reinterpreting it as a
// logical element count.
template <loom::memory_space Space, __SIZE_TYPE__ Alignment = 1>
[[loom::op("buffer.alloca")]] unsigned char* alloca(
    loom::type::offset byte_length);

namespace assume {

// Promises that source identifies an independently comparable storage root in
// the current scope while preserving its exact pointer origin.
template <class T>
[[loom::op("buffer.assume.noalias")]] T* noalias(T* source);

// Refines a borrowed zero-origin buffer with the same independent-root
// promise.
template <class T>
[[loom::op("buffer.assume.noalias")]] loom::type::buffer<T> noalias(
    loom::type::buffer<T> source);

// Promises that source names the zero-origin buffer root with at least
// Alignment-byte alignment in the current scope.
template <__SIZE_TYPE__ Alignment, class T>
[[loom::op("buffer.assume.alignment")]] T* alignment(T* source);

// Refines a borrowed zero-origin buffer with the same alignment promise.
template <__SIZE_TYPE__ Alignment, class T>
[[loom::op("buffer.assume.alignment")]] loom::type::buffer<T> alignment(
    loom::type::buffer<T> source);

// Promises that source's existing storage belongs to Space in the current
// scope. Preserves the element type, pointer origin, and storage identity;
// neither allocates nor copies storage. Constant storage must remain immutable
// for all accesses using the refined root, including through other aliases.
template <loom::memory_space Space, class T>
[[loom::op("buffer.assume.memory_space")]] T* memory_space(T* source);

// Refines a borrowed zero-origin buffer with the same storage-domain promise.
template <loom::memory_space Space, class T>
[[loom::op("buffer.assume.memory_space")]] loom::type::buffer<T> memory_space(
    loom::type::buffer<T> source);

}  // namespace assume

// Orders the executing thread's memory accesses across storage objects.
// Ordering is acquire, release, acq_rel, or seq_cst. Matching atomic
// observations and publications establish synchronization. The fence does not
// rendezvous with other threads or complete independent asynchronous transfers.
// It is valid in ordinary functions as well as kernels.
template <loom::atomic::ordering Ordering, loom::atomic::scope Scope>
[[loom::op("buffer.fence")]] void fence();

}  // namespace buffer
}  // namespace loom

#endif  // LOOMCXX_BUFFER_H_
