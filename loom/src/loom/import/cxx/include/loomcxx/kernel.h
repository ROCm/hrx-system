// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_KERNEL_H_
#define LOOMCXX_KERNEL_H_

#include <loomcxx/atomic.h>

// Launch geometry belongs on the entry with loom::workgroup_size(x, y, z) and
// loom::workgroup_count(x, y, z). Unspecified dimensions remain Loom configs.
// The corresponding *_range attributes take xmin, xmax, ymin, ymax, zmin, zmax
// and constrain those required config values with inclusive positive bounds.
// Counted unsigned for loops accept loom::unroll(factor),
// loom::pipeline(depth), and
// loom::schedule("linear"|"interleaved"|"recurrence"). Factors and depths are
// pure integer expressions read after the for initializer. They become SSA
// values and must resolve exactly when Loom applies the schedule. Calls,
// mutation, volatile reads, and overloaded operations inside an annotation
// are rejected. Unroll factors 0/1 and pipeline depth 1 are serial. Bare
// loom::unroll requests full unrolling; pipeline depths must be in [1, 65535].
// Scheduling is an explicit compiler contract; unsupported policies diagnose.
// Kernel pointer parameters accept [[loom::assume_aligned(N)]] before the
// parameter type. N is a positive power-of-two byte alignment of the incoming
// pointer address. The caller supplies the guarantee; no runtime check is
// added.
#define LOOM_KERNEL [[loom::kernel]]
#define LOOM_DEVICE [[loom::device]]
#define LOOM_WORKGROUP [[loom::workgroup]]
#define LOOM_FORCE_INLINE [[loom::force_inline]] inline

namespace loom {

struct uint3 {
  // Coordinate along the x axis.
  unsigned x;
  // Coordinate along the y axis.
  unsigned y;
  // Coordinate along the z axis.
  unsigned z;
};

// Topology queries and subgroup intrinsics require a kernel body or a
// force-inline helper. Other helpers receive topology values as arguments.
[[loom::workitem_id]] extern const uint3 workitem_id;
[[loom::workgroup_id]] extern const uint3 workgroup_id;
[[loom::workgroup_size]] extern const uint3 workgroup_size;
[[loom::workgroup_count]] extern const uint3 workgroup_count;

// Synchronizes workgroup invocations and their global/workgroup-memory
// accesses.
[[loom::barrier]] void workgroup_barrier();

// Declares unsigned binding < bound contracts, optionally joined by &&. Bounds
// are pure integer constant expressions in [1, INT32_MAX]. Conditions are not
// evaluated at runtime; calls, mutation, and unsupported predicates diagnose.
[[loom::assume]] void assume(bool condition);

// Reads this subgroup's zero-based coordinate within the workgroup.
[[loom::op("kernel.subgroup.id")]] unsigned subgroup_id();

// Counts subgroups in the workgroup, including a partially occupied subgroup.
[[loom::op("kernel.subgroup.count")]] unsigned subgroup_count();

// Reads this invocation's physical lane, without compacting inactive lanes.
[[loom::op("kernel.subgroup.lane.id")]] unsigned subgroup_lane_id();

// Reads the target-selected execution width, including inactive lanes. It may
// exceed the workgroup size; no fixed wave size is implied.
[[loom::op("kernel.subgroup.size")]] unsigned subgroup_size();

// Votes over the active invocations at this convergent call. These operations
// do not synchronize memory or rendezvous with other subgroups.
[[loom::op("kernel.subgroup.vote.any")]] bool subgroup_any(bool predicate);
[[loom::op("kernel.subgroup.vote.all")]] bool subgroup_all(bool predicate);

// Bit i describes physical lane i. Mask must be a 32- or 64-bit integer whose
// width covers the target subgroup; the default also covers 64-lane waves.
template <class Mask = unsigned long long>
[[loom::op("kernel.subgroup.vote.ballot")]] Mask subgroup_ballot(
    bool predicate);

// Returns the participating lanes with the same explicit mask-width contract.
template <class Mask = unsigned long long>
[[loom::op("kernel.subgroup.active.mask")]] Mask subgroup_active_mask();

// Broadcasts a scalar or explicit vector from the named active lane. Native
// lane-range and uniformity requirements follow the selected High target.
template <class T>
[[loom::op("kernel.subgroup.broadcast")]] T subgroup_broadcast(T value,
                                                               unsigned lane);

// Broadcasts from the first active lane, which need not be lane zero.
template <class T>
[[loom::op("kernel.subgroup.broadcast.first")]] T subgroup_broadcast_first(
    T value);

// Memory spaces ordered by an execution barrier.
enum class memory_space {
  // Device-visible global storage.
  global = 1,
  // Storage shared by invocations within a workgroup.
  workgroup = 2,
};

// Rendezvous of all invocations in Scope (subgroup or workgroup) with memory
// ordering in Space. Global memory accepts acquire, release, or acq_rel;
// workgroup memory requires acq_rel. This does not complete asynchronous DMA.
// A system publication still needs its matching system acquire/release; this
// barrier distributes that ordering to cooperating invocations. Callable
// helpers may contain barriers without requiring inlining.
template <memory_space Space, atomic::scope Scope, atomic::ordering Ordering>
[[loom::op("kernel.barrier")]] void barrier();

// Exchanges values across lanes selected by XOR within the given width.
[[loom::shuffle_xor]] float shuffle_xor(float value, int mask, int width);

}  // namespace loom

#endif  // LOOMCXX_KERNEL_H_
