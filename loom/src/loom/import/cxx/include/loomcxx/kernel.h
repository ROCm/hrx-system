// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_KERNEL_H_
#define LOOMCXX_KERNEL_H_

#include <loomcxx/buffer.h>
#include <loomcxx/cache.h>
#include <loomcxx/view.h>

// Fixed launch geometry belongs on the entry with
// loom::workgroup_size(x, y, z) and loom::workgroup_count(x, y, z).
// Unspecified dimensions remain Loom configs. The corresponding *_range
// attributes take xmin, xmax, ymin, ymax, zmin, zmax and constrain those
// required config values with inclusive positive bounds. A
// [[loom::kernel(configuration)]] entry instead computes complete geometry from
// explicit scalar or storage workload arguments and immutable target
// properties. Configuration bodies may read workload storage but cannot write
// it or use loaded data to select workgroup or cluster sizes.
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
// Declares uninitialized scalar, vector, or fixed scalar-array storage shared
// by every invocation in the enclosing kernel workgroup.
#define LOOM_WORKGROUP [[loom::workgroup]]
#define LOOM_FORCE_INLINE [[loom::force_inline]] inline
// Declares an ordinary C++ function whose calls apply the named link-selected
// Loom template family. The declaration has no C++ definition; its parameters
// and result define the semantic family signature at each call site.
#define LOOM_TEMPLATE_DECL(FAMILY) [[loom::op("template.apply", FAMILY)]]
// Defines one bodyful implementation of a LOOM_TEMPLATE_DECL family. Linking
// selects among providers using optional loom::target(object) and
// loom::priority(constant) attributes. Providers inherit the family's calling
// convention and are retained without becoming ordinary callable C++ symbols.
#define LOOM_TEMPLATE_DEF(FAMILY) [[loom::template_def(FAMILY)]]

namespace loom {

namespace type {

// One initiated asynchronous transfer. A token must be committed to exactly
// one kernel.async.group before its destination is consumed.
class [[loom::type("kernel.async.token")]] async_token {};

// One committed position in a kernel's ordered asynchronous-transfer stream.
// Waiting for a group also completes every older group in that stream.
class [[loom::type("kernel.async.group")]] async_group {};

}  // namespace type

// Declares integer truth, comparison, and conjunction contracts. Conditions
// are retained as Loom facts without runtime evaluation. Calls, mutation,
// volatile reads, and expressions without a retained scalar identity diagnose
// at import.
[[loom::assume]] void assume(bool condition);

namespace kernel {

namespace async {

// Required memory-space direction for a byte-for-byte asynchronous copy.
enum class direction : unsigned char {
  // Read global-like storage and write workgroup storage.
  global_to_workgroup = 0,
  // Read workgroup storage and write global-like storage.
  workgroup_to_global = 1,
};

// Initiates a byte-for-byte transfer between views with equal logical payload
// sizes. Direction and advisory cache policy remain explicit source contracts.
template <direction Direction, cache::scope CacheScope,
          cache::temporal CacheTemporal, class SourceShape, class SourceType,
          encoding::role SourceRole, class DestinationShape,
          class DestinationType, encoding::role DestinationRole>
[[loom::op("kernel.async.copy")]] type::async_token copy(
    type::view<SourceShape, SourceType, SourceRole> source,
    type::view<DestinationShape, DestinationType, DestinationRole> destination);

// Predicated copy. A false predicate performs no memory access and returns an
// already-complete token so every invocation can retain the same group shape.
template <direction Direction, cache::scope CacheScope,
          cache::temporal CacheTemporal, class SourceShape, class SourceType,
          encoding::role SourceRole, class DestinationShape,
          class DestinationType, encoding::role DestinationRole>
[[loom::op("kernel.async.copy.mask")]] type::async_token copy_mask(
    type::view<SourceShape, SourceType, SourceRole> source,
    type::view<DestinationShape, DestinationType, DestinationRole> destination,
    bool predicate);

// Collectively gathers each subgroup invocation's source view into the
// corresponding lane of one subgroup-uniform workgroup destination tile.
template <cache::scope CacheScope, cache::temporal CacheTemporal,
          class SourceShape, class SourceType, encoding::role SourceRole,
          class DestinationShape, class DestinationType,
          encoding::role DestinationRole>
[[loom::op("kernel.async.gather")]] type::async_token gather(
    type::view<SourceShape, SourceType, SourceRole> source,
    type::view<DestinationShape, DestinationType, DestinationRole> destination);

// Predicated subgroup gather preserving a uniform async group shape.
template <cache::scope CacheScope, cache::temporal CacheTemporal,
          class SourceShape, class SourceType, encoding::role SourceRole,
          class DestinationShape, class DestinationType,
          encoding::role DestinationRole>
[[loom::op("kernel.async.gather.mask")]] type::async_token gather_mask(
    type::view<SourceShape, SourceType, SourceRole> source,
    type::view<DestinationShape, DestinationType, DestinationRole> destination,
    bool predicate);

namespace cluster {

// Collectively gathers corresponding values for the workgroup-cluster ranks
// named by participant_mask. Bit N names flat cluster rank N.
template <cache::scope CacheScope, cache::temporal CacheTemporal,
          class SourceShape, class SourceType, encoding::role SourceRole,
          class DestinationShape, class DestinationType,
          encoding::role DestinationRole>
[[loom::op("kernel.async.cluster.gather")]] type::async_token gather(
    type::view<SourceShape, SourceType, SourceRole> source,
    type::view<DestinationShape, DestinationType, DestinationRole> destination,
    unsigned participant_mask);

// Predicated cluster gather preserving a uniform async group shape.
template <cache::scope CacheScope, cache::temporal CacheTemporal,
          class SourceShape, class SourceType, encoding::role SourceRole,
          class DestinationShape, class DestinationType,
          encoding::role DestinationRole>
[[loom::op("kernel.async.cluster.gather.mask")]] type::async_token gather_mask(
    type::view<SourceShape, SourceType, SourceRole> source,
    type::view<DestinationShape, DestinationType, DestinationRole> destination,
    unsigned participant_mask, bool predicate);

}  // namespace cluster

// Commits zero or more transfers at one position in the ordered async stream.
// Empty groups are valid pipeline markers.
template <class... Tokens>
[[loom::op("kernel.async.group")]] type::async_group group(Tokens... tokens);

// Completes group and every older group while allowing exactly NewerGroups
// younger groups to remain outstanding. This does not synchronize invocations;
// use a separate barrier before another invocation consumes staged storage.
template <unsigned long long NewerGroups>
[[loom::op("kernel.async.wait")]] void wait(type::async_group group);

}  // namespace async

// One target-independent kernel topology coordinate. Auto-deduced values keep
// Loom's index identity through templated APIs. Ordinary C++ and HIP-style
// expressions convert implicitly to their conventional unsigned carrier;
// address-domain arithmetic requires an explicit conversion to type::index.
class [[loom::type("index")]] coordinate {
 public:
  operator unsigned() const;
  explicit operator type::index() const;
};

// Three-dimensional target-independent kernel topology coordinates.
struct coordinates {
  // Coordinate along the x axis.
  coordinate x;
  // Coordinate along the y axis.
  coordinate y;
  // Coordinate along the z axis.
  coordinate z;
};

// Three-dimensional unsigned launch geometry.
struct uint3 {
  // Coordinate along the x axis.
  unsigned x;
  // Coordinate along the y axis.
  unsigned y;
  // Coordinate along the z axis.
  unsigned z;
};

// Complete launch geometry computed from explicit workload values and target
// properties. A function returning this aggregate may be referenced by
// [[loom::kernel(function)]]. Its body becomes the kernel's deterministic
// launch configuration region rather than an independently callable function.
struct [[loom::launch_config]] configuration {
  // Number of workgroups to launch in each dimension.
  uint3 workgroup_count;
  // Required workgroup size in each dimension.
  uint3 workgroup_size;
};

// Launch geometry with an additional workgroup-cluster size. The selected
// target must support the requested cluster dimensions.
struct [[loom::clustered_launch_config]] clustered_configuration {
  // Number of workgroups to launch in each dimension.
  uint3 workgroup_count;
  // Required workgroup size in each dimension.
  uint3 workgroup_size;
  // Number of cooperating workgroups in each cluster dimension.
  uint3 workgroup_cluster_size;
};

// Floating sums and products use native target arithmetic by default. Preserve
// requires source-format combines, including rounding, subnormals, signed
// zeros, and nonfinite values. Both permit parallel trees; neither prescribes
// serial order or reproducibility. Unsupported requirements fail compilation.
// Extrema always retain the selected kind's NaN and signed-zero semantics. This
// explicit requirement is independent of the caller's lexical fast-math
// permissions.
enum class collective_math : unsigned char {
  native = 0,
  preserve = 1,
};

// Reduces a scalar or explicit vector across the current subgroup. A nonzero
// ClusterSize selects independently reduced lane clusters; ClusterStride
// selects the lane spacing within a cluster and requires ClusterSize. Both are
// High attributes and therefore compile-time constants rather than SSA values.
// The result is uniform across the subgroup only when no cluster is selected.
namespace subgroup::reduce {

template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "addi")]] T addi(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.reduce", "addf")]] T addf(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "muli")]] T muli(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.reduce", "mulf")]] T mulf(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "minsi")]] T minsi(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "maxsi")]] T maxsi(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "minui")]] T minui(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "maxui")]] T maxui(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "andi")]] T andi(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "ori")]] T ori(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.reduce", "xori")]] T xori(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.reduce", "minimumf")]] T minimumf(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.reduce", "maximumf")]] T maximumf(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.reduce", "minnumf")]] T minnumf(T value);
template <unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.reduce", "maxnumf")]] T maxnumf(T value);

}  // namespace subgroup::reduce

// Prefix operations preserve each invocation's scalar or rank-one vector type.
// Exclusive scans return an empty-prefix seed at the first participating
// position, not an additional contribution to nonempty prefixes. Direction
// describes logical invocation order, not arithmetic evaluation order. These
// collectives do not imply memory synchronization.
enum class scan_mode : unsigned char {
  inclusive = 0,
  exclusive = 1,
};

enum class scan_direction : unsigned char {
  forward = 0,
  reverse = 1,
};

namespace subgroup::scan {

// ClusterSize and ClusterStride have the same contract as subgroup reductions.
// The defaults scan all participating lanes in increasing lane order.
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "addi")]] T addi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.scan", "addf")]] T addf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "muli")]] T muli(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.scan", "mulf")]] T mulf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "minsi")]] T minsi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "maxsi")]] T maxsi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "minui")]] T minui(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "maxui")]] T maxui(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "andi")]] T andi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "ori")]] T ori(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0, class T>
[[loom::op("kernel.subgroup.scan", "xori")]] T xori(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.scan", "minimumf")]] T minimumf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.scan", "maximumf")]] T maximumf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.scan", "minnumf")]] T minnumf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          unsigned long long ClusterSize = 0,
          unsigned long long ClusterStride = 0,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.subgroup.scan", "maxnumf")]] T maxnumf(T value);

}  // namespace subgroup::scan

namespace workgroup::reduce {

template <class T>
[[loom::op("kernel.workgroup.reduce", "addi")]] T addi(T value);
template <collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.reduce", "addf")]] T addf(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "muli")]] T muli(T value);
template <collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.reduce", "mulf")]] T mulf(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "minsi")]] T minsi(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "maxsi")]] T maxsi(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "minui")]] T minui(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "maxui")]] T maxui(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "andi")]] T andi(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "ori")]] T ori(T value);
template <class T>
[[loom::op("kernel.workgroup.reduce", "xori")]] T xori(T value);
template <collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.reduce", "minimumf")]] T minimumf(T value);
template <collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.reduce", "maximumf")]] T maximumf(T value);
template <collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.reduce", "minnumf")]] T minnumf(T value);
template <collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.reduce", "maxnumf")]] T maxnumf(T value);

}  // namespace workgroup::reduce

namespace workgroup::scan {

// All workitems participate, in linear workitem order, including a partial
// final subgroup. Clusters belong only to subgroup collectives.
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "addi")]] T addi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.scan", "addf")]] T addf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "muli")]] T muli(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.scan", "mulf")]] T mulf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "minsi")]] T minsi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "maxsi")]] T maxsi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "minui")]] T minui(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "maxui")]] T maxui(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "andi")]] T andi(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "ori")]] T ori(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward, class T>
[[loom::op("kernel.workgroup.scan", "xori")]] T xori(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.scan", "minimumf")]] T minimumf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.scan", "maximumf")]] T maximumf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.scan", "minnumf")]] T minnumf(T value);
template <scan_mode Mode = scan_mode::inclusive,
          scan_direction Direction = scan_direction::forward,
          collective_math Math = collective_math::native, class T>
[[loom::op("kernel.workgroup.scan", "maxnumf")]] T maxnumf(T value);

}  // namespace workgroup::scan

// Topology queries and collective intrinsics require a kernel body or a
// force-inline helper. Other helpers receive topology values as arguments.
namespace workitem {

// Zero-based invocation coordinate within the current workgroup.
[[loom::workitem_id]] extern const coordinates id;

}  // namespace workitem

namespace workgroup {

// Zero-based workgroup coordinate within the current launch.
[[loom::workgroup_id]] extern const coordinates id;
// Number of invocations in the current workgroup.
[[loom::workgroup_size]] extern const coordinates size;
// Number of workgroups in the current launch.
[[loom::workgroup_count]] extern const coordinates count;

// Synchronizes workgroup invocations and their global/workgroup-memory
// accesses.
[[loom::barrier]] void barrier();

}  // namespace workgroup

namespace subgroup {

// Reads this subgroup's zero-based coordinate within the workgroup.
[[loom::op("kernel.subgroup.id")]] unsigned id();

// Counts subgroups in the workgroup, including a partially occupied subgroup.
[[loom::op("kernel.subgroup.count")]] unsigned count();

// Reads this invocation's physical lane, without compacting inactive lanes.
[[loom::op("kernel.subgroup.lane.id")]] unsigned lane_id();

// Reads the target-selected execution width, including inactive lanes. It may
// exceed the workgroup size; no fixed wave size is implied.
[[loom::op("kernel.subgroup.size")]] unsigned size();

// Votes over the active invocations at this convergent call. These operations
// do not synchronize memory or rendezvous with other subgroups.
namespace vote {

[[loom::op("kernel.subgroup.vote.any")]] bool any(bool predicate);
[[loom::op("kernel.subgroup.vote.all")]] bool all(bool predicate);

// Bit i describes physical lane i. Mask must be a 32- or 64-bit integer whose
// width covers the target subgroup; the default also covers 64-lane waves.
template <class Mask = unsigned long long>
[[loom::op("kernel.subgroup.vote.ballot")]] Mask ballot(bool predicate);

}  // namespace vote

// Returns the participating lanes with the same explicit mask-width contract.
template <class Mask = unsigned long long>
[[loom::op("kernel.subgroup.active.mask")]] Mask active_mask();

// Broadcasts a scalar or explicit vector from the named active lane. Native
// lane-range and uniformity requirements follow the selected High target.
template <class T>
[[loom::op("kernel.subgroup.broadcast")]] T broadcast(T value, unsigned lane);

// Broadcasts from the first active lane, which need not be lane zero.
template <class T>
[[loom::op("kernel.subgroup.broadcast.first")]] T broadcast_first(T value);

// Result of an exchange within a logical subgroup. The payload is usable only
// when the selected source lane participates in the exchange.
template <class T>
struct shuffle_result {
  // Value read from the selected source lane when valid is true.
  T value;
  // Whether the source lies within the logical width and participates.
  bool valid;
};

// Exchanges a scalar or explicit rank-one vector with lane id XOR mask within
// the logical width. Physical subgroup size and participation remain separate.
template <class T>
[[loom::op("kernel.subgroup.shuffle", "xor")]]
shuffle_result<T> shuffle_xor(T value, unsigned mask, unsigned width);

// Reads a lower-numbered lane in the logical subgroup, or returns valid=false
// when the offset crosses its boundary or the source does not participate.
template <class T>
[[loom::op("kernel.subgroup.shuffle", "up")]]
shuffle_result<T> shuffle_up(T value, unsigned offset, unsigned width);

// Reads a higher-numbered lane with the same range and participation contract.
template <class T>
[[loom::op("kernel.subgroup.shuffle", "down")]]
shuffle_result<T> shuffle_down(T value, unsigned offset, unsigned width);

// Reads the named lane relative to this invocation's logical subgroup.
template <class T>
[[loom::op("kernel.subgroup.shuffle", "index")]]
shuffle_result<T> shuffle_index(T value, unsigned lane, unsigned width);

}  // namespace subgroup

// Rendezvous of all invocations in Scope (subgroup or workgroup) with memory
// ordering in Space. Global memory accepts acquire, release, or acq_rel;
// workgroup memory requires acq_rel. This does not complete asynchronous DMA.
// A system publication still needs its matching system acquire/release; this
// barrier distributes that ordering to cooperating invocations. Callable
// helpers may contain barriers without requiring inlining.
template <memory_space Space, atomic::scope Scope, atomic::ordering Ordering>
[[loom::op("kernel.barrier")]] void barrier();

}  // namespace kernel

}  // namespace loom

#endif  // LOOMCXX_KERNEL_H_
