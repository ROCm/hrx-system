// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/semaphore.h"

#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct Allocator {
  // Rejects non-free allocation requests when exercising failure ownership.
  bool deny = false;
  // Number of allocation requests, including rejected ones.
  int requests = 0;
  // Number of live allocations, observed through the real allocator boundary.
  int live = 0;

  iree_allocator_t get() { return {this, Control}; }
  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& self = *static_cast<Allocator*>(user_data);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++self.requests;
      if (self.deny) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "allocation denied");
      }
    }
    const bool freeing = command == IREE_ALLOCATOR_COMMAND_FREE && *pointer;
    const auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      self.live += freeing ? -1 : *pointer ? 1 : 0;
    }
    return status;
  }
};

class SemaphoreListTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
  }
  void TearDown() override {
    ReleaseSemaphores();
    EXPECT_EQ(semaphore_allocator_.live, 0);
    iree_async_proactor_release(proactor_);
  }
  iree_hal_semaphore_t* Semaphore(uint64_t initial_value = 0) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_host_semaphore_create(
        proactor_, initial_value, semaphore_allocator_.get(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }
  void ReleaseSemaphores() {
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    semaphores_.clear();
  }
  void ExpectEmpty(iree_hal_semaphore_list_t list) {
    EXPECT_EQ(list.count, 0u);
    EXPECT_EQ(list.semaphores, nullptr);
    EXPECT_EQ(list.payload_values, nullptr);
  }
  // Platform facility borrowed by the real host semaphores.
  iree_async_proactor_t* proactor_ = nullptr;
  // Observes the semaphores' actual final resource releases.
  Allocator semaphore_allocator_;
  // Test-owned references, independent of any cloned list.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_F(SemaphoreListTest, EmptyCloneDoesNotAllocate) {
  Allocator allocator;
  allocator.deny = true;
  const auto source = iree_hal_semaphore_list_empty();
  iree_hal_semaphore_t* sentinel = nullptr;
  uint64_t value = 0;
  iree_hal_semaphore_list_t result = {1, &sentinel, &value};
  IREE_ASSERT_OK(
      iree_hal_semaphore_list_clone(&source, allocator.get(), &result));
  ExpectEmpty(result);
  EXPECT_EQ(allocator.requests, 0);
  iree_hal_semaphore_list_free(result, allocator.get());
  EXPECT_EQ(allocator.live, 0);
}

TEST_F(SemaphoreListTest, CloneOwnsAlignedArraysAndIndependentReferences) {
  auto* first = Semaphore(9);
  auto* second = Semaphore(5);
  iree_hal_semaphore_t* semaphores[] = {first, second, first};
  uint64_t values[] = {3, 5, 9};
  const iree_hal_semaphore_list_t source = {3, semaphores, values};
  iree_hal_semaphore_list_t result;
  Allocator allocator;
  IREE_ASSERT_OK(
      iree_hal_semaphore_list_clone(&source, allocator.get(), &result));
  EXPECT_EQ(allocator.requests, 1);
  EXPECT_EQ(allocator.live, 1);
  ASSERT_EQ(result.count, 3u);
  EXPECT_NE(result.semaphores, semaphores);
  EXPECT_NE(result.payload_values, values);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(result.semaphores) %
                alignof(iree_hal_semaphore_t*),
            0u);
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(result.payload_values) % alignof(uint64_t),
      0u);
  semaphores[0] = nullptr;
  values[0] = 100;
  ReleaseSemaphores();
  EXPECT_EQ(semaphore_allocator_.live, 2);
  EXPECT_EQ(result.semaphores[0], first);
  EXPECT_EQ(result.semaphores[1], second);
  EXPECT_EQ(result.semaphores[2], first);
  EXPECT_EQ(result.payload_values[0], 3u);
  EXPECT_EQ(result.payload_values[1], 5u);
  EXPECT_EQ(result.payload_values[2], 9u);
  EXPECT_TRUE(iree_hal_semaphore_list_poll(result));
  iree_hal_semaphore_list_free(result, allocator.get());
  EXPECT_EQ(allocator.live, 0);
  EXPECT_EQ(semaphore_allocator_.live, 0);
}

TEST_F(SemaphoreListTest, SourceAndOutputDescriptorsMayAlias) {
  auto* semaphore = Semaphore(7);
  uint64_t value = 7;
  iree_hal_semaphore_list_t list = {1, &semaphore, &value};
  IREE_ASSERT_OK(
      iree_hal_semaphore_list_clone(&list, iree_allocator_system(), &list));
  EXPECT_EQ(list.count, 1u);
  EXPECT_NE(list.semaphores, &semaphore);
  EXPECT_NE(list.payload_values, &value);
  ReleaseSemaphores();
  EXPECT_TRUE(iree_hal_semaphore_list_poll(list));
  iree_hal_semaphore_list_free(list, iree_allocator_system());
}

TEST_F(SemaphoreListTest, AllocationFailureRetainsNothing) {
  auto* semaphore = Semaphore();
  uint64_t value = 7;
  const iree_hal_semaphore_list_t source = {1, &semaphore, &value};
  iree_hal_semaphore_list_t result = source;
  Allocator allocator;
  allocator.deny = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_semaphore_list_clone(&source, allocator.get(), &result));
  ExpectEmpty(result);
  EXPECT_EQ(allocator.requests, 1);
  EXPECT_EQ(allocator.live, 0);
  ReleaseSemaphores();
  EXPECT_EQ(semaphore_allocator_.live, 0);
}

TEST_F(SemaphoreListTest, RejectsUnrepresentableLayoutBeforeAllocation) {
  auto* semaphore = Semaphore();
  uint64_t value = 1;
  const iree_host_size_t counts[] = {
      IREE_HOST_SIZE_MAX / sizeof(iree_hal_semaphore_t*) + 1,
      IREE_HOST_SIZE_MAX / sizeof(uint64_t) + 1,
      IREE_HOST_SIZE_MAX / (sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t)) +
          1,
  };
  Allocator allocator;
  allocator.deny = true;
  for (iree_host_size_t count : counts) {
    const iree_hal_semaphore_list_t source = {count, &semaphore, &value};
    iree_hal_semaphore_list_t result = source;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_semaphore_list_clone(&source, allocator.get(), &result));
    ExpectEmpty(result);
  }
  EXPECT_EQ(allocator.requests, 0);
  ReleaseSemaphores();
  EXPECT_EQ(semaphore_allocator_.live, 0);
}

TEST_F(SemaphoreListTest, PollRequiresAllValuesAndRejectsFailure) {
  EXPECT_TRUE(iree_hal_semaphore_list_poll(iree_hal_semaphore_list_empty()));
  iree_hal_semaphore_t* semaphores[] = {Semaphore(), Semaphore()};
  uint64_t values[] = {1, 2};
  const iree_hal_semaphore_list_t list = {2, semaphores, values};
  EXPECT_FALSE(iree_hal_semaphore_list_poll(list));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], 1, nullptr));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(list));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[1], 2, nullptr));
  EXPECT_TRUE(iree_hal_semaphore_list_poll(list));
  iree_hal_semaphore_fail(semaphores[0], iree_make_status(IREE_STATUS_DATA_LOSS,
                                                          "producer failed"));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(list));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_query(semaphores[0], &value));
  values[0] = 0;
  EXPECT_FALSE(iree_hal_semaphore_list_poll(list));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_query(semaphores[0], &value));
}

}  // namespace
