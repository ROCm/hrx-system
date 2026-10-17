// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/queues.h"

#include <algorithm>
#include <thread>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/testing/device.h"

IREE_FLAG(string, execution_device_uri, "",
          "Device URI for the provisioned execution queue tests.");

namespace {

TEST(ExecutionQueuesTest, NullLifetimeOperations) {
  loom_serve_execution_queues_retain(nullptr);
  loom_serve_execution_queues_release(nullptr);
}

class ExecutionQueuesTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_host_create(2, nullptr, iree_allocator_system(), &host_));
    IREE_ASSERT_OK(loom_serve_test_driver_create(host_, iree_allocator_system(),
                                                 &driver_));
    const loom_serve_device_options_t options = {
        .driver = driver_,
        .uri = FLAG_execution_device_uri[0]
                   ? iree_make_cstring_view(FLAG_execution_device_uri)
                   : loom_serve_test_device()->device_uri,
    };
    IREE_ASSERT_OK(loom_serve_device_group_allocate(
        host_, 1, &options, iree_allocator_system(), &group_));
    auto* owner = loom_serve_device_group_at(group_, 0);
    device_ = loom_serve_device_handle(owner);
    dispatch_family_ =
        iree_hal_queue_family(loom_serve_device_dispatch_queue(owner));
    transfer_family_ = dispatch_family_;
    if (GetParam()) {
      transfer_family_ = nullptr;
      const auto* spec =
          iree_hal_device_spec_queues(iree_hal_device_spec(device_));
      for (iree_host_size_t i = 0; i < spec->family_count; ++i) {
        const auto* family = iree_hal_device_queue_family(device_, i);
        if (family != dispatch_family_ &&
            spec->families[i].provisioned_queue_count &&
            iree_all_bits_set(spec->families[i].role_flags,
                              IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
          transfer_family_ = family;
          break;
        }
      }
      if (!transfer_family_) {
        GTEST_SKIP() << "Device has no separate provisioned transfer family";
      }
    }
    dispatch_count_ =
        iree_hal_queue_family_spec(dispatch_family_)->provisioned_queue_count;
    transfer_count_ =
        iree_hal_queue_family_spec(transfer_family_)->provisioned_queue_count;
    IREE_ASSERT_OK(loom_serve_execution_queues_create(
        dispatch_family_, transfer_family_, iree_allocator_system(), &queues_));
    const iree_hal_pool_family_access_t families[] = {
        {
            .family = dispatch_family_,
            .usage =
                IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        },
        {.family = transfer_family_, .usage = IREE_HAL_BUFFER_USAGE_TRANSFER},
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = dispatch_family_ == transfer_family_ ? 1u : 2u,
        .families = families,
    };
    const iree_hal_slab_pool_options_t pool_options = {};
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        loom_serve_device_group(owner), scope, &pool_options,
        iree_allocator_system(), &storage_pool_));
    const iree_hal_asan_pool_options_t asan_options = {};
    IREE_ASSERT_OK(loom_serve_workspace_create(
        loom_serve_device_group(owner), storage_pool_, &asan_options,
        {this, WorkspaceAllocate}, &workspace_));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion_));
  }

  void TearDown() override {
    for (auto binding : bindings_) {
      loom_serve_execution_binding_release(binding);
    }
    loom_serve_execution_release(execution_);
    loom_serve_execution_queues_release(queues_);
    loom_serve_workspace_release(workspace_);
    iree_hal_pool_release(storage_pool_);
    iree_hal_semaphore_release(completion_);
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    iree_hal_driver_release(driver_);
    loom_serve_host_release(host_);
  }

  static iree_status_t WorkspaceAllocate(void* self,
                                         iree_allocator_command_t command,
                                         const void* parameters,
                                         void** inout_pointer) {
    auto* test = static_cast<ExecutionQueuesTest*>(self);
    if (test->fail_scope_ && command != IREE_ALLOCATOR_COMMAND_FREE) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, inout_pointer);
  }

  loom_serve_execution_binding_t Acquire() {
    auto binding = loom_serve_execution_queues_acquire(queues_);
    bindings_.push_back(binding);
    return binding;
  }

  void ReleaseLast() {
    loom_serve_execution_binding_release(bindings_.back());
    bindings_.pop_back();
  }

  void Check(loom_serve_execution_binding_t binding, uint32_t dispatch,
             uint32_t transfer) {
    EXPECT_EQ(binding.dispatch,
              iree_hal_device_queue(
                  device_, iree_hal_queue_family_ordinal(dispatch_family_),
                  dispatch));
    EXPECT_EQ(binding.transfer,
              iree_hal_device_queue(
                  device_, iree_hal_queue_family_ordinal(transfer_family_),
                  transfer));
    EXPECT_EQ(binding.dispatch_ordinal, dispatch);
    EXPECT_EQ(binding.transfer_ordinal, transfer);
    for (auto* queue : {binding.dispatch, binding.transfer}) {
      ++completion_value_;
      const iree_hal_semaphore_list_t signals = {1, &completion_,
                                                 &completion_value_};
      IREE_ASSERT_OK(iree_hal_queue_barrier(
          queue, iree_hal_semaphore_list_empty(), signals, nullptr,
          IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }

  // Workers and I/O services enclosing every device and queue lifetime.
  loom_serve_host_t* host_ = nullptr;
  // Physical driver selected by the linked target test leaf.
  iree_hal_driver_t* driver_ = nullptr;
  // Sealed device group enclosing all submissions and workspace captures.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed physical device shared by both selected families.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed provisioned dispatch family.
  const iree_hal_queue_family_t* dispatch_family_ = nullptr;
  // Borrowed transfer family, optionally distinct from dispatch.
  const iree_hal_queue_family_t* transfer_family_ = nullptr;
  // Provisioned dispatch queue count from the immutable device specification.
  uint32_t dispatch_count_ = 0;
  // Provisioned transfer queue count from the immutable device specification.
  uint32_t transfer_count_ = 0;
  // Shared queue inventory whose returned bindings are the test subject.
  loom_serve_execution_queues_t* queues_ = nullptr;
  // Owned bindings, including any left live by a failed assertion.
  std::vector<loom_serve_execution_binding_t> bindings_;
  // Owned model execution used to exercise final-reference and failure paths.
  loom_serve_execution_t* execution_ = nullptr;
  // Real backing admitted on both explicitly selected families.
  iree_hal_pool_t* storage_pool_ = nullptr;
  // Shared workspace with a controllable allocation dependency.
  loom_serve_workspace_t* workspace_ = nullptr;
  // Owner-thread failure injection for execution's workspace-scope allocation.
  bool fail_scope_ = false;
  // Completion proof for real submissions on each selected queue.
  iree_hal_semaphore_t* completion_ = nullptr;
  // Monotonic value, joined before another queue signals the next value.
  uint64_t completion_value_ = 0;
};

TEST_P(ExecutionQueuesTest, BalancesLiveBindingsAcrossFixedInventory) {
  const uint32_t count = std::max(dispatch_count_, transfer_count_) + 1;
  for (uint32_t i = 0; i < count; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        Check(Acquire(), i % dispatch_count_, i % transfer_count_));
  }
}

TEST_P(ExecutionQueuesTest, ReloadsReuseQueuesBesideLivePeer) {
  ASSERT_NO_FATAL_FAILURE(Check(Acquire(), 0, 0));
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        Check(Acquire(), 1 % dispatch_count_, 1 % transfer_count_));
    ReleaseLast();
  }
}

TEST_P(ExecutionQueuesTest, BindingKeepsInventoryAlive) {
  const auto binding = Acquire();
  loom_serve_execution_queues_release(queues_);
  queues_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(Check(binding, 0, 0));
}

TEST_P(ExecutionQueuesTest, FinalExecutionReferenceReturnsBinding) {
  IREE_ASSERT_OK(loom_serve_execution_create(
      queues_, workspace_, iree_allocator_system(), &execution_));
  loom_serve_execution_retain(execution_);
  loom_serve_execution_release(execution_);
  ASSERT_NO_FATAL_FAILURE(
      Check(Acquire(), 1 % dispatch_count_, 1 % transfer_count_));
  // Final release may run in a result's completion callback while the model
  // owner binds another execution. It publishes no device-completion signal.
  std::thread release(
      [execution = execution_] { loom_serve_execution_release(execution); });
  execution_ = nullptr;
  Acquire();
  ReleaseLast();
  release.join();
  ASSERT_NO_FATAL_FAILURE(Check(Acquire(), 0, 0));
}

TEST_P(ExecutionQueuesTest, ConstructionFailureReturnsBinding) {
  ASSERT_NO_FATAL_FAILURE(Check(Acquire(), 0, 0));
  fail_scope_ = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_execution_create(queues_, workspace_, iree_allocator_system(),
                                  &execution_));
  fail_scope_ = false;
  EXPECT_EQ(execution_, nullptr);
  ASSERT_NO_FATAL_FAILURE(
      Check(Acquire(), 1 % dispatch_count_, 1 % transfer_count_));
  ReleaseLast();
  IREE_ASSERT_OK(loom_serve_execution_create(
      queues_, workspace_, iree_allocator_system(), &execution_));
  IREE_ASSERT_OK(loom_serve_execution_barrier(execution_,
                                              iree_hal_semaphore_list_empty()));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
}

INSTANTIATE_TEST_SUITE_P(SeparateTransferFamily, ExecutionQueuesTest,
                         ::testing::Bool());

}  // namespace
