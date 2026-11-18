// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Processor/NUMA affinity description for thread placement.
//
// This is a lightweight value type that can be used without pulling in the
// full threading library (iree_thread_t, pthreads, etc.). Separated from
// thread.h so that subsystems needing only topology descriptions (e.g., async
// proactor pools, NUMA-aware allocators) can depend on :threading without
// requiring :thread and its platform threading dependencies.

#ifndef IREE_BASE_THREADING_AFFINITY_H_
#define IREE_BASE_THREADING_AFFINITY_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/bitmap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Capacity of native processor sets, independent of Task worker counts.
// Bit N denotes native CPU N on Linux/Android or group * 64 + processor on
// Windows. The default covers the upstream Linux architecture limits supported
// by IREE and occupies 1 KiB. Override consistently across the build for a
// different deployment envelope. A value must be a positive multiple of 64;
// Linux/Android native masks exceeding the capacity are rejected, never
// truncated.
#ifndef IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT
#define IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT 8192
#endif
#if IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT <= 0 || \
    IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT % 64 != 0
#error IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT must be a positive multiple of 64
#endif

// Caller-owned storage for a native processor set. This value has no allocator
// or lifetime beyond its containing object. Consumers borrow a const pointer;
// mutable bitmap views are available for constructing or editing the set.
typedef struct iree_thread_affinity_set_t {
  // Native processor membership, with CPU N at bit N.
  uint64_t words[IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT / 64];
} iree_thread_affinity_set_t;

// Returns a mutable bitmap borrowing |set| for construction or editing.
// Its storage must outlive every use.
static inline iree_bitmap_t iree_thread_affinity_set_bitmap(
    iree_thread_affinity_set_t* set) {
  iree_bitmap_t bitmap = {IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT, set->words};
  return bitmap;
}

// Returns whether |processor| belongs to |set|. Native CPU IDs beyond the set's
// capacity are not members.
static inline bool iree_thread_affinity_set_test(
    const iree_thread_affinity_set_t* set, iree_host_size_t processor) {
  return processor < IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT &&
         (set->words[processor / 64] & (UINT64_C(1) << (processor % 64))) != 0;
}

// Finds the first member at or after |processor_offset|, or returns
// IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT if there is none.
IREE_API_EXPORT iree_host_size_t iree_thread_affinity_set_find_first_set(
    const iree_thread_affinity_set_t* set, iree_host_size_t processor_offset);

// Returns whether |set| contains no CPUs.
static inline bool iree_thread_affinity_set_is_empty(
    const iree_thread_affinity_set_t* set) {
  return iree_thread_affinity_set_find_first_set(set, 0) ==
         IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT;
}

// Specifies the processor affinity for a particular thread.
// Each platform handles this differently (if at all).
//
// macOS/iOS:
//   Only affinity tags are supported; the ID will be used by the kernel to
//   group threads that having matching values together and (hopefully) schedule
//   them on cores that may share some level of the cache hierarchy. The API is
//   effectively just asking nicely and hoping the kernel is on the same
//   wavelength. It rarely is. As a workaround for the lack of specific pinning
//   we use the smt bit of 1 to indicate that _only_ efficiency cores should be
//   used by effectively changing the QoS of the thread to one in the range
//   where only efficiency cores will be used.
//
//   Mapping:
//    group: (unused)
//       id: used for THREAD_AFFINITY_POLICY to request exclusive cores.
//      smt: 1 if only efficiency cores should be used (QOS_CLASS_BACKGROUND).
//
// Linux/Android:
//   sched_setaffinity is used to pin the thread to the core with the given ID.
//   There are, naturally, issues on Android where if the governor has turned
//   off some cores (such as powering down big cores in an ARM big.LITTLE
//   configuration) the affinity request will be dropped on the floor even if
//   the cores are later enabled. This is one of the reasons why we note in
//   iree_thread_request_affinity that requests may need to be made at
//   ¯\_(ツ)_/¯ intervals. In the future we can try to hook into power
//   management infra to see if we can tell when we need to do this.
//
//   Mapping:
//    group: NUMA node when group_any is set; otherwise unused.
//       id: CPU_SET bit indicating which CPU to run on.
//      smt: whether to CPU_SET both the base ID and the subsequent ID.
//
// Windows:
//   Mapping:
//    group: NUMA node when group_any is set; otherwise processor group.
//       id: GROUP_AFFINITY::Mask bit/PROCESSOR_NUMBER::Number.
//      smt: whether to set both the base ID and the subsequent ID in Mask.
typedef struct iree_thread_affinity_t {
  // When 1 the processor ID will be ignored and the platform will choose any
  // processor associated with the specified NUMA node. On Windows this node
  // is resolved to its primary processor group's node-local CPU mask.
  uint32_t group_any : 1;
  // NUMA node when group_any is set. Otherwise the native processor group for
  // a specific CPU (Windows), or unused (Linux/Darwin). Memory placement is
  // independent of thread affinity.
  uint32_t group;

  // When group_any is 0, indicates whether id specifies a processor. When
  // both flags are 0 the affinity is unspecified.
  uint32_t id_assigned : 1;
  // Processor ID the thread should be scheduled on. The interpretation and
  // efficacy of this request varies per platform.
  uint32_t id : 30;
  // When 1 and the specified processor ID is part of an SMT set all logical
  // cores in the set should be reserved for the thread to avoid contention.
  uint32_t smt : 1;
} iree_thread_affinity_t;

// Sets |out_thread_affinity| to match with any processor in the system.
IREE_API_EXPORT void iree_thread_affinity_set_any(
    iree_thread_affinity_t* out_thread_affinity);

// Returns true if |thread_affinity| does not specify any particular processor.
static inline bool iree_thread_affinity_is_unspecified(
    iree_thread_affinity_t thread_affinity) {
  return !thread_affinity.group_any && !thread_affinity.id_assigned;
}

// Sets |out_thread_affinity| to match all processors associated with the given
// NUMA node ID. Any eligible processor within the node may be
// selected by the platform.
IREE_API_EXPORT void iree_thread_affinity_set_group_any(
    uint32_t group, iree_thread_affinity_t* out_thread_affinity);

// Captures the calling thread's effective Linux/Android processor affinity into
// caller-owned storage. Native CPU IDs are preserved, including sparse IDs.
// Returns OUT_OF_RANGE if the kernel mask exceeds the configured capacity.
//
// Capture this at the application ownership boundary: querying a later pinned
// worker does not recover the application's original CPU budget. This observes
// thread affinity, not the enclosing cgroup's complete CPU allowance.
// Platforms without this effective-affinity query return UNAVAILABLE.
IREE_API_EXPORT iree_status_t iree_thread_affinity_query_current(
    iree_thread_affinity_set_t* out_processor_set);

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
// Resolves Linux/Android affinity into caller-owned native cpu_set_t storage.
// The native mask has |native_size| bytes, enough to represent at least
// IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT CPUs, and is intersected with the
// optional application processor budget without changing any thread's affinity.
// An unspecified affinity copies the budget, or yields an empty mask if none
// is given. Native consumers decide whether an empty mask is meaningful.
iree_status_t iree_thread_affinity_resolve_linux(
    iree_thread_affinity_t affinity,
    const iree_thread_affinity_set_t* processor_set,
    iree_host_size_t native_size, void* out_native_set);
#endif  // IREE_PLATFORM_ANDROID || IREE_PLATFORM_LINUX

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_AFFINITY_H_
