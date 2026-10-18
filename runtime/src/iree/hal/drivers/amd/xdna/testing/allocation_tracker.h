// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_TESTING_ALLOCATION_TRACKER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_TESTING_ALLOCATION_TRACKER_H_

#include <atomic>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree::hal::amd::xdna::testing {

struct HostAllocationCounters {
  // Allocation requests observed across all progress threads.
  std::atomic<uint64_t> allocations{0};
  // Reallocation requests, including in-place growth.
  std::atomic<uint64_t> reallocations{0};
  // Free requests, including null-pointer frees.
  std::atomic<uint64_t> frees{0};
  // Successful allocations not yet returned to the system allocator.
  std::atomic<int64_t> live_allocations{0};
};

struct HostAllocationSnapshot {
  // Allocation requests at the observation boundary.
  uint64_t allocations;
  // Reallocation requests at the observation boundary.
  uint64_t reallocations;
  // Free requests at the observation boundary.
  uint64_t frees;
};

inline iree_status_t TrackingAllocatorControl(void* self,
                                              iree_allocator_command_t command,
                                              const void* params,
                                              void** inout_pointer) {
  auto* counters = static_cast<HostAllocationCounters*>(self);
  const bool had_pointer = inout_pointer && *inout_pointer;
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC) {
    ++counters->allocations;
  } else if (command == IREE_ALLOCATOR_COMMAND_REALLOC) {
    ++counters->reallocations;
  } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    ++counters->frees;
  }
  iree_allocator_t system_allocator = iree_allocator_system();
  iree_status_t status = system_allocator.ctl(system_allocator.self, command,
                                              params, inout_pointer);
  if (iree_status_is_ok(status)) {
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        (command == IREE_ALLOCATOR_COMMAND_REALLOC && !had_pointer)) {
      ++counters->live_allocations;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
      --counters->live_allocations;
    }
  }
  return status;
}

inline iree_allocator_t TrackingAllocator(HostAllocationCounters* counters) {
  return {counters, TrackingAllocatorControl};
}

inline HostAllocationSnapshot SnapshotHostAllocations(
    const HostAllocationCounters& counters) {
  return {
      counters.allocations.load(),
      counters.reallocations.load(),
      counters.frees.load(),
  };
}

inline void ExpectHostAllocationsUnchanged(
    const HostAllocationSnapshot& before,
    const HostAllocationCounters& counters) {
  const HostAllocationSnapshot after = SnapshotHostAllocations(counters);
  EXPECT_EQ(after.allocations, before.allocations);
  EXPECT_EQ(after.reallocations, before.reallocations);
  EXPECT_EQ(after.frees, before.frees);
}

}  // namespace iree::hal::amd::xdna::testing

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_TESTING_ALLOCATION_TRACKER_H_
