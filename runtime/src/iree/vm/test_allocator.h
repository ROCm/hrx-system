// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_TEST_ALLOCATOR_H_
#define IREE_VM_TEST_ALLOCATOR_H_

#include "iree/base/api.h"

namespace iree::vm::testing {

// Forwards allocation commands while counting allocation and free attempts.
// An optional one-based allocation ordinal injects RESOURCE_EXHAUSTED before
// forwarding that command to the delegate.
class CountingAllocator {
 public:
  explicit CountingAllocator(iree_host_size_t fail_at_allocation = 0)
      : fail_at_allocation_(fail_at_allocation) {}

  iree_allocator_t allocator() { return iree_allocator_t{this, Control}; }

  iree_host_size_t allocation_count() const { return allocation_count_; }
  iree_host_size_t free_count() const { return free_count_; }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<CountingAllocator*>(self);
    switch (command) {
      case IREE_ALLOCATOR_COMMAND_MALLOC:
      case IREE_ALLOCATOR_COMMAND_CALLOC:
      case IREE_ALLOCATOR_COMMAND_REALLOC:
        if (++allocator->allocation_count_ == allocator->fail_at_allocation_) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "injected allocation failure");
        }
        break;
      case IREE_ALLOCATOR_COMMAND_FREE:
        ++allocator->free_count_;
        break;
      default:
        break;
    }
    return allocator->delegate_.ctl(allocator->delegate_.self, command, params,
                                    inout_ptr);
  }

  // Allocator performing the actual memory operations.
  iree_allocator_t delegate_ = iree_allocator_system();
  // Number of allocation-like commands forwarded or failed.
  iree_host_size_t allocation_count_ = 0;
  // Number of free commands forwarded.
  iree_host_size_t free_count_ = 0;
  // One-based allocation command to fail, or zero to never fail.
  iree_host_size_t fail_at_allocation_ = 0;
};

}  // namespace iree::vm::testing

#endif  // IREE_VM_TEST_ALLOCATOR_H_
