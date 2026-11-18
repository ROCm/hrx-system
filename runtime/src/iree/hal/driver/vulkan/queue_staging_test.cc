// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/queue_staging.h"

#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/vulkan/queue.h"

namespace iree::hal::cts {
namespace {

// Owns real queue staging loans through each test's completion callbacks.
class StagingLoans {
 public:
  explicit StagingLoans(iree_hal_vulkan_queue_staging_ring_t* ring)
      : ring_(ring) {}
  ~StagingLoans() {
    while (!slots_.empty()) {
      ReleaseOne();
    }
  }

  bool Acquire() {
    iree_hal_vulkan_queue_staging_slot_t* slot = nullptr;
    if (!iree_hal_vulkan_queue_staging_ring_try_acquire(ring_, &slot)) {
      return false;
    }
    slots_.push_back(slot);
    return true;
  }

  void ReleaseOne() {
    auto* slot = slots_.back();
    slots_.pop_back();
    iree_hal_vulkan_queue_staging_ring_release(ring_, slot);
  }

 private:
  // Borrowed queue storage, live until all loans return.
  iree_hal_vulkan_queue_staging_ring_t* ring_;
  // Exclusively owned slots, returned after the test's accesses.
  std::vector<iree_hal_vulkan_queue_staging_slot_t*> slots_;
};

struct OwnedWaiter {
  // Callback owner containing the placed waiter.
  iree_hal_resource_t resource;
  // Registration whose linked reference may be the final resource reference.
  iree_hal_vulkan_queue_staging_waiter_t waiter;
  // Borrowed destruction observation owned by the test.
  int* destroy_count;
};

static void DestroyWaiter(iree_hal_resource_t* base_resource) {
  auto* owner = reinterpret_cast<OwnedWaiter*>(base_resource);
  ++*owner->destroy_count;
  delete owner;
}

static const iree_hal_resource_vtable_t kWaiterVTable = {
    .destroy = DestroyWaiter,
};

class VulkanQueueStagingTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_);
    IREE_ASSERT_OK(iree_hal_vulkan_queue_activate(queue));
    ring_ = queue->upload_staging_ring;
  }

  // Borrowed from the device's exact transfer queue.
  iree_hal_vulkan_queue_staging_ring_t* ring_ = nullptr;
};

TEST_P(VulkanQueueStagingTest, CancellationCanDropTheFinalPlacedOwner) {
  StagingLoans loans(ring_);
  while (loans.Acquire()) {
  }
  int destroy_count = 0;
  auto* owner = new OwnedWaiter{};
  iree_hal_resource_initialize(&kWaiterVTable, &owner->resource);
  owner->destroy_count = &destroy_count;
  EXPECT_TRUE(iree_hal_vulkan_queue_staging_ring_queue_waiter(
      ring_, &owner->waiter,
      [](void*) { ADD_FAILURE() << "cancelled waiter was invoked"; }, nullptr,
      &owner->resource));
  iree_hal_resource_release(&owner->resource);
  iree_hal_vulkan_queue_staging_ring_cancel_waiter(ring_, &owner->waiter);
  EXPECT_EQ(destroy_count, 1);
}

TEST_P(VulkanQueueStagingTest, RequeueDuringWakeupPreservesThePendingOwner) {
  StagingLoans loans(ring_);
  while (loans.Acquire()) {
  }
  int destroy_count = 0;
  auto* owner = new OwnedWaiter{};
  iree_hal_resource_initialize(&kWaiterVTable, &owner->resource);
  owner->destroy_count = &destroy_count;
  struct State {
    // Borrowed physical storage used by both callbacks.
    iree_hal_vulkan_queue_staging_ring_t* ring;
    // Collector retaining acquired slots until explicitly returned.
    StagingLoans* loans;
    // Pending placed waiter whose registration races with another wakeup.
    OwnedWaiter* owner;
    // Number of invocations of the pending waiter.
    int calls = 0;
  } state = {ring_, &loans, owner};
  const auto pending_callback = +[](void* user_data) {
    auto* state = static_cast<State*>(user_data);
    ++state->calls;
    EXPECT_TRUE(state->loans->Acquire());
  };
  struct FirstState {
    // Shared transfer observations and storage.
    State* state;
    // Callback reused when the pending transfer races to register itself.
    iree_hal_vulkan_queue_staging_waiter_fn_t pending_callback;
  } first_state = {&state, pending_callback};
  iree_hal_vulkan_queue_staging_waiter_t first = {};
  EXPECT_TRUE(iree_hal_vulkan_queue_staging_ring_queue_waiter(
      ring_, &first,
      [](void* user_data) {
        auto* first = static_cast<FirstState*>(user_data);
        auto* state = first->state;
        EXPECT_TRUE(state->loans->Acquire());
        // A concurrent producer can retry admission while the first callback
        // owns the returned slot. The pending link and its resource stay
        // intact.
        EXPECT_TRUE(iree_hal_vulkan_queue_staging_ring_queue_waiter(
            state->ring, &state->owner->waiter, first->pending_callback, state,
            &state->owner->resource));
      },
      &first_state, nullptr));
  EXPECT_TRUE(iree_hal_vulkan_queue_staging_ring_queue_waiter(
      ring_, &owner->waiter, pending_callback, &state, &owner->resource));
  iree_hal_resource_release(&owner->resource);

  loans.ReleaseOne();
  EXPECT_EQ(state.calls, 0);
  EXPECT_EQ(destroy_count, 0);
  loans.ReleaseOne();
  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(destroy_count, 1);
}

CTS_REGISTER_TEST_SUITE(VulkanQueueStagingTest);

}  // namespace
}  // namespace iree::hal::cts
