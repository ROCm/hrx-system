// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/event.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/xdna/device.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/hal/drivers/amd/xdna/memory.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"
#include "iree/hal/drivers/amd/xdna/testing/queue_test_util.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::xdna::testing::DirectBufferBindingLayout;
using iree::hal::amd::xdna::testing::ExpectHostAllocationsUnchanged;
using iree::hal::amd::xdna::testing::HostAllocationCounters;
using iree::hal::amd::xdna::testing::HostAllocationSnapshot;
using iree::hal::amd::xdna::testing::IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT;
using iree::hal::amd::xdna::testing::IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST;
using iree::hal::amd::xdna::testing::IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA;
using iree::hal::amd::xdna::testing::NativeProvider;
using iree::hal::amd::xdna::testing::Outcome;
using iree::hal::amd::xdna::testing::QueueHarness;
using iree::hal::amd::xdna::testing::SnapshotHostAllocations;
using iree::hal::amd::xdna::testing::TrackingAllocator;

// Contract-backed caller pages with a stable native table that can be
// published after queue capture. This models the buffer surface produced by a
// queue-ordered shared slab without replacing the queue or executable under
// test.
class PreparedContractBuffer {
 public:
  ~PreparedContractBuffer() {
    iree_hal_buffer_release(view_);
    iree_hal_buffer_release(root_);
    iree_hal_memory_contract_release(contract_);
  }

  void Initialize(QueueHarness& harness) {
    const iree_hal_queue_family_t* family =
        iree_hal_queue_family(harness.queue);
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        iree_hal_device_group_memory_domain(harness.group),
        iree_hal_device_group_memory_scope_count(harness.group),
        DirectBufferBindingLayout(), iree_allocator_system(), &contract_));
    for (iree_hal_memory_site_kind_t kind :
         {IREE_HAL_MEMORY_SITE_QUEUE, IREE_HAL_MEMORY_SITE_PROGRAM}) {
      iree_hal_memory_scope_t scope;
      IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
          harness.group, {kind, family}, &scope));
      iree_hal_memory_scope_access_t* access = &contract_->scopes[scope.id];
      access->usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ;
      access->interfaces = UINT32_C(1)
                           << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
      access->bindings[IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA] =
          IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA;
    }
    bindings_[IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST].host_pointer =
        bytes_.data();
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
            IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
            IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_USAGE_STORAGE_READ, bytes_.size(),
        iree_make_byte_span(bytes_.data(), bytes_.size()),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &root_));
    root_->memory.contract = contract_;
    root_->memory.bindings = bindings_.data();
    root_->host_binding_index = IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root_, kViewOffset, kViewLength,
                                           iree_allocator_system(), &view_));
  }

  void Publish(uint64_t device_address) {
    bindings_[IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA].device_address =
        device_address;
  }

  iree_hal_buffer_ref_t ref(iree_device_size_t offset,
                            iree_device_size_t length) const {
    return iree_hal_make_buffer_ref(view_, offset, length);
  }

  static constexpr iree_device_size_t kViewOffset = 16;
  static constexpr iree_device_size_t kViewLength = 64;

 private:
  // Caller-owned bytes represented by the prepared native table.
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) std::array<uint8_t, 128> bytes_ = {};
  // Stable slot storage published under semaphore ordering.
  std::array<iree_hal_buffer_native_binding_t,
             IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT>
      bindings_ = {};
  // Owned immutable contract borrowed by |root_| and |view_|.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // Owned allocation root wrapping caller pages.
  iree_hal_buffer_t* root_ = nullptr;
  // Owned nonzero-offset logical view used by dispatch.
  iree_hal_buffer_t* view_ = nullptr;
};

TEST(XdnaDeviceTest, PublishesMemoryProgressBackend) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());

  const iree_hal_memory_backend_t* base =
      iree_hal_device_memory_backend(harness.device);
  ASSERT_NE(base, nullptr);
  EXPECT_EQ(base->type, IREE_HAL_MEMORY_BACKEND_AMDF);
  const auto* backend =
      reinterpret_cast<const iree_hal_amd_fabric_memory_backend_t*>(base);
  EXPECT_EQ(backend->device, harness.device);
  EXPECT_EQ(backend->native.instance, harness.context->fabric->instance);
  EXPECT_EQ(backend->native.device, harness.context->device);
  ASSERT_EQ(backend->native.storage_count, 1u);
  EXPECT_EQ(backend->native.storage[0].scope,
            harness.context->data_source.scope);
  EXPECT_EQ(backend->native.storage[0].physical_device_affinity, 0u);
  EXPECT_NE(backend->notification, nullptr);
  EXPECT_NE(backend->allocation_queue, nullptr);
  ASSERT_NE(backend->epoch_query.fn, nullptr);
  EXPECT_FALSE(backend->epoch_query.fn(backend->epoch_query.user_data, 0, 0));

  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  const iree_async_axis_t axis =
      iree_hal_device_topology_info(harness.device)->frontier.base_axis;
  EXPECT_TRUE(backend->epoch_query.fn(backend->epoch_query.user_data, axis, 0));
  EXPECT_FALSE(
      backend->epoch_query.fn(backend->epoch_query.user_data, axis, 1));
}

TEST(XdnaQueueTest, DeviceCreatesOwnedDeviceCompatibleSemaphores) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  EXPECT_TRUE(iree_hal_amd_xdna_semaphore_isa(harness.done));
  EXPECT_TRUE(
      iree_hal_amd_xdna_semaphore_is_local(harness.done, harness.device));
  EXPECT_EQ(iree_hal_device_query_semaphore_compatibility(harness.device,
                                                          harness.done),
            IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL);
}

TEST(XdnaQueueTest, UnsupportedNativeQueueReportsDriverRequirementAndUnwinds) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.xdna.kernel_queue_create =
      +[](amdf_xdna_context_t*, const amdf_xdna_kernel_queue_create_info_t*,
          amdf_kernel_queue_t**) -> amdf_status_t {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  };
  HostAllocationCounters counters;
  iree_hal_queue_t* queue = nullptr;
  iree::Status status = iree_hal_amd_xdna_queue_create(
      iree_hal_queue_family(harness.queue), harness.context, harness.proactor,
      TrackingAllocator(&counters), &queue);
  EXPECT_EQ(status.code(), iree::StatusCode::kUnimplemented);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("update the NPU driver and matching firmware"));
  EXPECT_EQ(queue, nullptr);
  EXPECT_EQ(counters.live_allocations.load(), 0);
  EXPECT_EQ(harness.native.submission_count, 0u);
}

TEST(XdnaQueueTest, QueueBarrierPreservesGlobalBoundaryOrdering) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &ready));

  const iree_hal_barrier_t global = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
               IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      .effects = {IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
                  IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM},
  };
  const iree_hal_barrier_list_t global_list = {1, &global};
  const iree_hal_queue_barriers_t barriers = {&global_list, &global_list};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(harness.queue, {1, &ready, &value},
                                        {1, &harness.done, &value}, &barriers,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));

  uint64_t completed_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &completed_value));
  EXPECT_EQ(completed_value, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready, value, nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  EXPECT_EQ(harness.native.submission_count, 0u);

  iree_hal_semaphore_release(ready);
}

TEST(XdnaQueueTest, QueueBarrierRejectsRangedQueueOperation) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));

  const iree_hal_memory_transition_recipe_info_t operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      .range_granularity = 64,
  };
  const iree_hal_memory_transition_recipe_t recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      .operation_count = 1,
      .operations = &operation,
  };
  const iree_hal_buffer_barrier_t range = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
      .buffer_ref = iree_hal_make_buffer_ref(buffer, 0, 64),
      .recipe = &recipe,
  };
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_NONE,
      .effects = recipe.effects,
      .memory_barrier_count = 0,
      .memory_barriers = nullptr,
      .buffer_barrier_count = 1,
      .buffer_barriers = &range,
  };
  const iree_hal_barrier_list_t list = {1, &barrier};
  const iree_hal_queue_barriers_t barriers = {&list, nullptr};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_queue_barrier(harness.queue, {}, {}, &barriers,
                             IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  EXPECT_EQ(harness.native.submission_count, 0u);

  iree_hal_buffer_release(buffer);
}

TEST(XdnaQueueTest, DispatchAcceptsGlobalVisibilityBoundary) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());

  const iree_hal_barrier_t global = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      .effects = {IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM},
  };
  const iree_hal_barrier_list_t global_list = {1, &global};
  const iree_hal_queue_barriers_t barriers = {nullptr, &global_list};
  ASSERT_NO_FATAL_FAILURE(
      harness.SubmitWithWaits({}, harness.done, 1, &barriers));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_EQ(harness.native.submission_count, 1u);
}

TEST(XdnaQueueTest, DispatchValidatesImageBindingContractAtCapture) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  auto dispatch = [&](iree_hal_buffer_ref_list_t bindings) {
    return iree_hal_queue_dispatch(
        harness.queue, {}, {}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, bindings,
        /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, dispatch({}));
  iree_hal_buffer_ref_t binding = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, dispatch({1, &binding}));
  binding = iree_hal_make_indirect_buffer_ref(1, 0, 16);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, dispatch({1, &binding}));

  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  binding = iree_hal_make_buffer_ref(buffer, 56, 8);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, dispatch({1, &binding}));
  iree_hal_buffer_release(buffer);

  ASSERT_NO_FATAL_FAILURE(
      harness.MakeBuffer(&buffer, IREE_HAL_MEMORY_ACCESS_WRITE,
                         IREE_HAL_BUFFER_USAGE_STORAGE_READ));
  binding = iree_hal_make_buffer_ref(buffer, 0, 16);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, dispatch({1, &binding}));
  iree_hal_buffer_release(buffer);

  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(
      &buffer, IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_USAGE_TRANSFER));
  binding = iree_hal_make_buffer_ref(buffer, 0, 16);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, dispatch({1, &binding}));
  iree_hal_buffer_release(buffer);

  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) std::array<uint8_t, 64> host_bytes =
      {};
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_USAGE_STORAGE_READ,
      host_bytes.size(),
      iree_make_byte_span(host_bytes.data(), host_bytes.size()),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));
  binding = iree_hal_make_buffer_ref(buffer, 0, 16);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, dispatch({1, &binding}));
  iree_hal_buffer_release(buffer);
  iree_hal_executable_release(executable);
}

TEST(XdnaQueueTest, QueueWaitPublishesContractBindingBeforePreparation) {
  constexpr uint64_t kPublishedAddress = UINT64_C(0x123456780000);
  constexpr iree_device_size_t kBindingOffset = 4;
  constexpr iree_device_size_t kBindingLength = 32;

  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  PreparedContractBuffer prepared_buffer;
  ASSERT_NO_FATAL_FAILURE(prepared_buffer.Initialize(harness));

  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  iree_hal_semaphore_t* allocation_ready = nullptr;
  iree_hal_semaphore_t* completion = nullptr;
  for (auto** semaphore : {&allocation_ready, &completion}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }

  iree_hal_buffer_ref_t binding =
      prepared_buffer.ref(kBindingOffset, kBindingLength);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {1, &allocation_ready, &value}, {1, &completion, &value},
      executable, function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  EXPECT_EQ(harness.native.submission_count, 0u);

  prepared_buffer.Publish(kPublishedAddress);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(allocation_ready, value, nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(1));
  // Native submission precedes its proactor acknowledgement, which registers
  // the notification destination used by the explicit completion wake below.
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(1));
  std::vector<uint8_t> command_bytes;
  {
    std::lock_guard<std::mutex> lock(harness.native.mutex);
    ASSERT_EQ(harness.native.pending_commands.size(), 1u);
    command_bytes = harness.native.pending_commands[0].bytes;
  }
  const uint64_t patched_address = kPublishedAddress +
                                   PreparedContractBuffer::kViewOffset +
                                   kBindingOffset + 4;
  ASSERT_GE(command_bytes.size(), 16u);
  EXPECT_EQ(iree_unaligned_load_le_u32(command_bytes.data() + 8),
            (uint32_t)patched_address | 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(command_bytes.data() + 12),
            UINT32_C(0xA5A50000) | (uint32_t)(patched_address >> 32));

  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, completion));
  iree_hal_semaphore_release(completion);
  iree_hal_semaphore_release(allocation_ready);
  iree_hal_executable_release(executable);
}

TEST(XdnaQueueTest, ReadyDispatchCommitsBeforeProactorProgress) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  EXPECT_EQ(harness.native.submission_count, 1u);
  EXPECT_EQ(harness.native.notification_count, 1u);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
}

TEST(XdnaQueueTest, ExactWaitWithoutTopologyRetiresBeforeProactorProgress) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  EXPECT_EQ(harness.native.submission_count, 1u);
  EXPECT_EQ(harness.native.notification_count, 1u);
  EXPECT_EQ(harness.native.wait_count, 0u);
  EXPECT_EQ(harness.native.refresh_count, 0u);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      harness.done, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(harness.native.wait_count, 1u);
  EXPECT_EQ(harness.native.refresh_count, 1u);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 1u);
  EXPECT_EQ(harness.native.live_memories, 1u);

  // A topology-free queue cannot publish a complete global frontier. That
  // loss of precision must not route a later dispatch with no waits through
  // the proactor.
  iree_hal_semaphore_t* second = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &second));
  ASSERT_NO_FATAL_FAILURE(harness.Submit(second));
  EXPECT_EQ(harness.native.submission_count, 2u);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(second, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(harness.native.wait_count, 2u);
  iree_hal_semaphore_release(second);
}

TEST(XdnaQueueTest, ExactWaitOwnsRetirementBeforeNativeNotification) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  harness.native.BlockWait();
  ASSERT_NO_FATAL_FAILURE(harness.Submit());

  std::atomic<iree_status_code_t> wait_status{IREE_STATUS_UNKNOWN};
  std::thread waiter([&]() {
    iree_status_t status = iree_hal_semaphore_wait(
        harness.done, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    wait_status.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
  });
  struct WaitReleaseGuard {
    NativeProvider* provider;
    std::thread* waiter;
    ~WaitReleaseGuard() {
      if (provider) {
        provider->ReleaseWait();
      }
      if (waiter->joinable()) {
        waiter->join();
      }
    }
  } wait_release_guard = {&harness.native, &waiter};
  harness.native.AwaitBlockedWait();

  harness.native.hold_retirement = false;
  harness.native.Wake();
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  EXPECT_EQ(harness.native.refresh_count, 0u);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  harness.native.ReleaseWait();
  waiter.join();
  wait_release_guard.provider = nullptr;
  EXPECT_EQ(wait_status.load(std::memory_order_acquire), IREE_STATUS_OK);
  EXPECT_EQ(harness.native.wait_count, 1u);
  EXPECT_EQ(harness.native.refresh_count, 1u);
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 1u);
  EXPECT_EQ(harness.native.live_memories, 1u);
}

TEST(XdnaQueueTest, InexactReachedFrontierPublishesDirectly) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  iree_async_single_frontier_t host_frontier;
  iree_async_single_frontier_initialize(
      &host_frontier, iree_async_axis_make_queue(1, 1, 0, 0, 0), 1);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(
      gate, 1, iree_async_single_frontier_as_const_frontier(&host_frontier)));

  ASSERT_NO_FATAL_FAILURE(harness.SubmitAfter(gate, 1));
  EXPECT_EQ(harness.native.submission_count, 1u);
  iree_hal_submitted_signal_flags_t signal_flags =
      IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
  iree_async_axis_t producer_axis = 0;
  uint64_t producer_epoch = 0;
  uint64_t producer_value = 0;
  ASSERT_TRUE(iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(harness.done), &signal_flags,
      &producer_axis, &producer_epoch, &producer_value));
  (void)producer_axis;
  EXPECT_EQ(producer_epoch, 1u);
  EXPECT_EQ(producer_value, 1u);
  EXPECT_FALSE(iree_all_bits_set(
      signal_flags, IREE_HAL_SUBMITTED_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      harness.done, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(harness.native.wait_count, 1u);
  iree_hal_semaphore_release(gate);
}

TEST(XdnaQueueTest, AcceptedSignalChainUsesNativeFifoBeforeRetirement) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_queue_assign_frontier(harness.queue, tracker, axis));

  iree_hal_semaphore_t* edge = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER, &edge));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  ASSERT_NO_FATAL_FAILURE(harness.Submit(edge));
  EXPECT_EQ(harness.native.submission_count, 1u);
  ASSERT_NO_FATAL_FAILURE(harness.SubmitAfter(edge, 1));
  EXPECT_EQ(harness.native.submission_count, 2u);

  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(edge, &value));
  EXPECT_EQ(value, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_FALSE(iree_async_semaphore_is_value_tainted(
      reinterpret_cast<iree_async_semaphore_t*>(harness.done), 1));
  iree_hal_amd_xdna_frontier_t frontier;
  EXPECT_EQ(iree_async_semaphore_query_frontier(
                reinterpret_cast<iree_async_semaphore_t*>(harness.done),
                iree_async_fixed_frontier_as_frontier(&frontier),
                IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY),
            1u);
  EXPECT_EQ(frontier.entries[0].axis, axis);
  EXPECT_EQ(frontier.entries[0].epoch, 2u);

  iree_hal_semaphore_release(edge);
  iree_async_frontier_tracker_release(tracker);
}

TEST(XdnaQueueTest, DeferredConsumersTargetLowestPendingSignal) {
  QueueHarness harness;
  harness.native.pending_capacity = 4;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  IREE_ASSERT_OK(iree_hal_amd_xdna_queue_assign_frontier(
      harness.queue, tracker, iree_async_axis_make_queue(1, 0, 0, 0, 0)));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_semaphore_t* anchor_done = nullptr;
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* edge = nullptr;
  iree_hal_semaphore_t* reached = nullptr;
  std::array<iree_hal_semaphore_t*, 2> consumer_done = {};
  for (auto** semaphore : {&anchor_done, &gate, &edge, &reached,
                           &consumer_done[0], &consumer_done[1]}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }
  iree_hal_executable_t* anchor_executable = nullptr;
  iree_hal_executable_function_t anchor_function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(iree::hal::amd::xdna::testing::ImageFixture(),
                             &anchor_executable, &anchor_function));
  iree_hal_buffer_t* anchor_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&anchor_buffer));
  const iree_hal_buffer_ref_t anchor_binding =
      iree_hal_make_buffer_ref(anchor_buffer, 0, 64);

  // Hold the direct publisher so all following operations are admitted by the
  // proactor before any one of them becomes native-visible. The value-2
  // producer is submitted first but cannot run until |gate| is signaled. The
  // consumer must link to the later value-1 producer instead of selecting by
  // queue-list order or retrying behind every unrelated acceptance.
  harness.native.BlockSubmission();
  uint64_t anchor_value = 1;
  std::thread anchor_submitter([&]() {
    IREE_EXPECT_OK(iree_hal_queue_dispatch(
        harness.queue, {}, {1, &anchor_done, &anchor_value}, anchor_executable,
        anchor_function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
        {1, &anchor_binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  });
  harness.native.AwaitBlockedSubmission();
  harness.SubmitAfter(gate, 1, edge, 2);
  harness.Submit(edge);
  uint64_t wait_values[] = {1, 1};
  iree_hal_semaphore_t* wait_semaphores[] = {edge, reached};
  IREE_ASSERT_OK(iree_hal_semaphore_signal(reached, 1, nullptr));
  for (iree_hal_semaphore_t* completion : consumer_done) {
    harness.SubmitWithWaits(
        {IREE_ARRAYSIZE(wait_semaphores), wait_semaphores, wait_values},
        completion, 1);
  }
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));

  harness.native.ReleaseSubmission();
  anchor_submitter.join();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(4));
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(gate, &value));
  EXPECT_EQ(value, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_query(edge, &value));
  EXPECT_EQ(value, 0u);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, anchor_done));
  for (iree_hal_semaphore_t* completion : consumer_done) {
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, completion));
    iree_hal_semaphore_release(completion);
  }
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(edge, 2));

  iree_hal_semaphore_release(reached);
  iree_hal_semaphore_release(edge);
  iree_hal_semaphore_release(gate);
  iree_hal_semaphore_release(anchor_done);
  iree_hal_buffer_release(anchor_buffer);
  iree_hal_executable_release(anchor_executable);
  iree_async_frontier_tracker_release(tracker);
}

TEST(XdnaQueueTest, InexactTimelineChainDrainsAcrossNativeCapacity) {
  constexpr uint64_t kChainLength = 31;

  QueueHarness harness;
  harness.native.pending_capacity = 4;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_semaphore_t* completion = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &completion));
  for (uint64_t value = 1; value <= kChainLength; ++value) {
    uint64_t wait_value = value - 1;
    const iree_hal_semaphore_list_t waits =
        wait_value ? iree_hal_semaphore_list_t{1, &completion, &wait_value}
                   : iree_hal_semaphore_list_t{};
    harness.SubmitWithWaits(waits, completion, value);
  }
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(completion, kChainLength));
  EXPECT_EQ(harness.native.submission_count, kChainLength);

  iree_hal_semaphore_release(completion);
}

TEST(XdnaQueueTest, LargeSignalSetUsesPooledProducerChunks) {
  constexpr iree_host_size_t kSignalCount = 52;

  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  std::vector<iree_hal_semaphore_t*> signals(kSignalCount);
  std::vector<uint64_t> signal_values(kSignalCount, 1);
  for (iree_hal_semaphore_t*& signal : signals) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &signal));
  }

  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  const iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(buffer, 0, 64);
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {1, &gate, &gate_value},
      {signals.size(), signals.data(), signal_values.data()}, executable,
      function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  EXPECT_EQ(harness.native.submission_count, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value, nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(signals.back(), 1));
  for (iree_hal_semaphore_t* signal : signals) {
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(signal, &value));
    EXPECT_EQ(value, 1u);
    iree_hal_semaphore_release(signal);
  }

  iree_hal_buffer_release(buffer);
  iree_hal_executable_release(executable);
  iree_hal_semaphore_release(gate);
}

TEST(XdnaQueueTest, ConsumerSubmittedBeforeProducerDefersUntilRetirement) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_queue_assign_frontier(harness.queue, tracker, axis));

  iree_hal_semaphore_t* edge = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER, &edge));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  ASSERT_NO_FATAL_FAILURE(harness.SubmitAfter(edge, 1));
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  EXPECT_EQ(harness.native.submission_count, 0u);

  ASSERT_NO_FATAL_FAILURE(harness.Submit(edge));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(1));
  EXPECT_EQ(harness.native.submission_count, 1u);

  harness.native.retirement_limit = 1;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  harness.native.retirement_limit = 2;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(tracker, axis, 2));

  iree_hal_semaphore_release(edge);
  iree_async_frontier_tracker_release(tracker);
}

TEST(XdnaQueueTest, FullNativeCapacityUsesQueuedPublication) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_semaphore_t* following = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &following));

  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  EXPECT_EQ(harness.native.submission_count, 1u);
  ASSERT_NO_FATAL_FAILURE(harness.Submit(following));
  EXPECT_EQ(harness.native.submission_count, 1u);
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  EXPECT_EQ(harness.native.submission_count, 1u);

  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, following));
  EXPECT_EQ(harness.native.submission_count, 2u);
  iree_hal_semaphore_release(following);
}

TEST(XdnaQueueTest, NativeBusyWithoutPendingWorkFailsQueueInvariant) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  harness.native.ReturnBusyOnNextSubmission();

  ASSERT_NO_FATAL_FAILURE(harness.Submit(
      /*completion=*/nullptr, QueueHarness::CapturedResourceState::kReleased));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_INTERNAL));
  EXPECT_EQ(harness.native.submission_attempt_count, 1u);
  EXPECT_EQ(harness.native.submission_count, 0u);
  EXPECT_GE(harness.diagnostic_count, 1u);
}

TEST(XdnaQueueTest, NativeBusyRetainsDirectPublicationUntilCheckedProgress) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  IREE_ASSERT_OK(iree_hal_amd_xdna_queue_assign_frontier(
      harness.queue, tracker, iree_async_axis_make_queue(1, 0, 0, 0, 0)));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_semaphore_t* first = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &first));

  ASSERT_NO_FATAL_FAILURE(harness.Submit(first));
  EXPECT_EQ(harness.native.submission_count, 1u);
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(1));

  harness.native.ReturnBusyOnNextSubmission();
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  EXPECT_EQ(harness.native.submission_attempt_count, 2u);
  EXPECT_EQ(harness.native.submission_count, 1u);
  const size_t flush_count_after_busy = harness.native.FlushCount();
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  harness.native.retirement_limit =
      harness.native.last_accepted_submission.load(std::memory_order_acquire);
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  EXPECT_EQ(harness.native.submission_attempt_count, 3u);
  EXPECT_EQ(harness.native.FlushCount(), flush_count_after_busy);
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(2));

  harness.native.retirement_limit = UINT64_MAX;
  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, first));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_EQ(harness.diagnostic_count, 0u);
  iree_hal_semaphore_release(first);
  iree_async_frontier_tracker_release(tracker);
}

TEST(XdnaQueueTest, NativeBusyRetainsQueuedPublicationUntilCheckedProgress) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_semaphore_t* first = nullptr;
  iree_hal_semaphore_t* gate = nullptr;
  for (auto** semaphore : {&first, &gate}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }

  ASSERT_NO_FATAL_FAILURE(harness.Submit(first));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(1));
  iree_async_single_frontier_t host_frontier;
  iree_async_single_frontier_initialize(
      &host_frontier, iree_async_axis_make_queue(1, 1, 0, 0, 0), 1);

  harness.native.ReturnBusyOnNextSubmission();
  harness.native.BlockSubmission();
  ASSERT_NO_FATAL_FAILURE(harness.SubmitAfter(gate, 1));
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(
      gate, 1, iree_async_single_frontier_as_const_frontier(&host_frontier)));
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  harness.native.AwaitBlockedSubmission();
  EXPECT_EQ(harness.native.submission_attempt_count, 2u);
  harness.native.retirement_limit =
      harness.native.last_accepted_submission.load(std::memory_order_acquire);
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, first));
  EXPECT_EQ(harness.native.submission_count, 1u);
  const size_t flush_count_after_prepare = harness.native.FlushCount();

  harness.native.ReleaseSubmission();
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  EXPECT_EQ(harness.native.submission_attempt_count, 3u);
  EXPECT_EQ(harness.native.FlushCount(), flush_count_after_prepare);
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(2));

  harness.native.retirement_limit = UINT64_MAX;
  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_EQ(harness.diagnostic_count, 0u);
  iree_hal_semaphore_release(gate);
  iree_hal_semaphore_release(first);
}

TEST(XdnaQueueTest, PublicationClaimContentionUsesQueuedPublication) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  const auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
  std::array<iree_hal_semaphore_t*, 2> completions = {};
  for (auto& completion : completions) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &completion));
  }

  harness.native.BlockSubmission();
  uint64_t value = 1;
  std::atomic<bool> first_submission_returned{false};
  std::thread first_submitter([&]() {
    IREE_EXPECT_OK(iree_hal_queue_dispatch(
        harness.queue, {}, {1, &completions[0], &value}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
        /*barriers=*/NULL, 0));
    first_submission_returned.store(true, std::memory_order_release);
  });
  harness.native.AwaitBlockedSubmission();
  EXPECT_EQ(harness.native.submission_attempt_count, 1u);
  iree_status_t second_status = iree_hal_queue_dispatch(
      harness.queue, {}, {1, &completions[1], &value}, executable, function,
      iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
      /*barriers=*/NULL, 0);
  IREE_EXPECT_OK(second_status);
  EXPECT_EQ(harness.native.submission_attempt_count, 1u);
  IREE_EXPECT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  EXPECT_EQ(harness.native.submission_attempt_count, 1u);

  harness.native.ReleaseSubmission();
  first_submitter.join();
  EXPECT_TRUE(first_submission_returned.load(std::memory_order_acquire));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  harness.native.hold_retirement = false;
  harness.native.Wake();
  for (auto* completion : completions) {
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, completion));
    iree_hal_semaphore_release(completion);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_executable_release(executable);
}

TEST(XdnaQueueTest, PendingInvocationsKeepPrivateBindingsAndReuseBacking) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  // Place the instruction aperture at the target floor but below the image's
  // stronger alignment so this exercises arena-internal alignment padding.
  harness.native.next_address = 0x1008000;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree::hal::amd::xdna::testing::ImageFixture fixture;
  fixture.allocations[0].alignment = 65536;
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(fixture, &executable, &function));
  std::array<iree_hal_buffer_t*, 3> buffers = {};
  std::array<iree_hal_semaphore_t*, 2> completions = {};
  for (auto*& buffer : buffers) {
    ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  }
  for (auto*& completion : completions) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &completion));
  }
  // One instruction aperture, one static DMA catalog, and three caller buffers.
  EXPECT_EQ(harness.native.memory_create_count, 5u);
  for (uint64_t iteration = 1; iteration <= 4; ++iteration) {
    harness.native.hold_retirement = true;
    harness.native.flushes.clear();
    for (size_t i = 0; i < completions.size(); ++i) {
      auto binding =
          iree_hal_make_buffer_ref(buffers[(i + iteration) % 2], 0, 64);
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          harness.queue, {}, {1, &completions[i], &iteration}, executable,
          function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
          {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(iteration * 2));
    ASSERT_EQ(harness.native.pending_commands.size(), 2u);
    const auto& first = harness.native.pending_commands[0];
    const auto& second = harness.native.pending_commands[1];
    EXPECT_EQ(first.memory, second.memory);
    EXPECT_NE(first.offset, second.offset);
    EXPECT_EQ((first.memory->address + first.offset) % 65536, 0u);
    EXPECT_EQ((second.memory->address + second.offset) % 65536, 0u);
    EXPECT_FALSE(std::equal(first.bytes.begin() + 8, first.bytes.end(),
                            second.bytes.begin() + 8));
    // Both commands are ranges in the persistent instruction aperture.
    EXPECT_EQ(harness.native.memory_create_count, 5u);
    if (iteration > 1) {
      EXPECT_TRUE(harness.native.flushes.empty());
    }
    harness.native.hold_retirement = false;
    harness.native.Wake();
    for (auto* completion : completions) {
      uint64_t value = 0;
      IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
      while (value < iteration) {
        IREE_ASSERT_OK(iree_async_proactor_poll(
            harness.proactor, iree_infinite_timeout(), nullptr));
        IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
      }
    }
  }
  // A new binding tuple patches one retired manifestation exactly once. Its
  // next use finds the prepared match even when it is not the available head.
  for (uint64_t iteration = 5; iteration <= 7; ++iteration) {
    harness.native.flushes.clear();
    iree_hal_buffer_t* buffer = iteration == 6 ? buffers[0] : buffers[2];
    auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue, {}, {1, &completions[0], &iteration}, executable,
        function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
        {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(completions[0], iteration));
    if (iteration == 5) {
      ASSERT_EQ(harness.native.flushes.size(), 1u);
      EXPECT_EQ(harness.native.flushes[0].length, 8u);
    } else {
      EXPECT_TRUE(harness.native.flushes.empty());
    }
  }
  for (auto* completion : completions) {
    iree_hal_semaphore_release(completion);
  }
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_executable_release(executable);
  EXPECT_EQ(harness.native.live_memories, 1u);
}

TEST(XdnaQueueTest, GrowableCommandArenaReusesPersistentSlabs) {
  QueueHarness harness;
  harness.native.enforce_single_command_memory = false;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree::hal::amd::xdna::testing::ImageFixture fixture;
  fixture.allocations[0].byte_length = 5 * 1024 * 1024;

  std::array<iree_hal_executable_t*, 2> executables = {};
  iree_hal_executable_function_t function;
  for (auto& executable : executables) {
    ASSERT_NO_FATAL_FAILURE(
        harness.LoadExecutable(fixture, &executable, &function));
  }
  EXPECT_EQ(harness.native.command_memory_create_count, 2u);

  iree_hal_executable_release(executables[0]);
  executables[0] = nullptr;
  iree_hal_executable_t* replacement = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(fixture, &replacement, &function));
  EXPECT_EQ(harness.native.command_memory_create_count, 2u);

  iree_hal_executable_release(replacement);
  iree_hal_executable_release(executables[1]);
}

TEST(XdnaQueueTest, PartialRetirementReturnsCapacityWithOpaquePoints) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.point_stride = 7;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  std::array<iree_hal_semaphore_t*, 2> following = {};
  for (auto& completion : following) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &completion));
  }
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  for (auto* completion : following) {
    ASSERT_NO_FATAL_FAILURE(harness.Submit(completion));
  }
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  EXPECT_EQ(harness.native.submission_count, 2u);
  harness.native.retirement_limit = 7;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(3));
  EXPECT_EQ(harness.native.live_memories, 5u);
  EXPECT_EQ(harness.native.pending_commands.size(), 2u);
  for (auto* completion : following) {
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(completion, &value));
    EXPECT_EQ(value, 0u);
  }
  for (size_t i = 0; i < following.size(); ++i) {
    harness.native.retirement_limit = (i + 2) * 7;
    harness.native.Wake();
    ASSERT_NO_FATAL_FAILURE(
        harness.PollUntilDone(IREE_STATUS_OK, following[i]));
    iree_hal_semaphore_release(following[i]);
  }
  EXPECT_EQ(harness.native.live_memories, 1u);
}

TEST(XdnaQueueTest, PrivateAddressesPropagateThroughImmutableClosure) {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree::hal::amd::xdna::testing::ImageFixture fixture;
  // Command -> immutable A -> immutable B -> device-written scratch. The
  // destination-sorted relocations require more than one closure iteration.
  for (uint32_t i = 2; i < 4; ++i) {
    auto allocation = fixture.allocations[1];
    allocation.first_load = i + 2;
    fixture.allocations.push_back(allocation);
    auto load = fixture.loads[2];
    load.physical_address = i;
    load.file_range.offset = 4120 + (i - 2) * 8;
    fixture.loads.push_back(load);
    fixture.payloads.push_back(std::vector<uint8_t>(8, 0));
    fixture.uses.push_back(i);
  }
  fixture.allocations[1].flags = IREE_XDNA_ELF_ALLOCATION_FLAG_DEVICE_WRITE;
  auto dynamic = fixture.relocations[1];
  fixture.relocations.resize(4);
  fixture.relocations[0].source_ordinal = 2;
  fixture.relocations[1] = fixture.relocations[0];
  fixture.relocations[1].destination_use = 2;
  fixture.relocations[1].source_ordinal = 3;
  fixture.relocations[2] = fixture.relocations[0];
  fixture.relocations[2].destination_use = 3;
  fixture.relocations[2].source_ordinal = 1;
  fixture.relocations[3] = dynamic;
  fixture.entries[0].allocation_use_count = 4;
  fixture.entries[0].static_relocation_count = 3;
  fixture.entries[0].first_dynamic_relocation = 3;
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(fixture, &executable, &function));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  std::array<iree_hal_semaphore_t*, 2> completions = {};
  auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
  uint64_t value = 1;
  for (auto& completion : completions) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &completion));
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue, {}, {1, &completion, &value}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
        /*barriers=*/NULL, 0));
  }
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  EXPECT_EQ(harness.native.memory_create_count, 8u);
  ASSERT_EQ(harness.native.pending_commands.size(), 2u);
  const auto& first = harness.native.pending_commands[0].bytes;
  const auto& second = harness.native.pending_commands[1].bytes;
  EXPECT_FALSE(std::equal(first.begin(), first.begin() + 8, second.begin()));
  harness.native.hold_retirement = false;
  harness.native.Wake();
  for (auto* completion : completions) {
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, completion));
    iree_hal_semaphore_release(completion);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_executable_release(executable);
  EXPECT_EQ(harness.native.live_memories, 1u);
}

TEST(XdnaQueueTest, HostProducerProgressesWithFullNativeQueue) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  const auto axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_queue_assign_frontier(harness.queue, tracker, axis));
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(1));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_semaphore_t* produced = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &produced));
  const uint32_t payload = 0x12345678;
  iree_hal_transfer_operation_t transfer = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
  };
  transfer.update.target_buffer = buffer;
  transfer.update.source_buffer = &payload;
  transfer.update.length = sizeof(payload);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_transfer(harness.queue, {},
                                         {1, &produced, &value}, 1, &transfer,
                                         /*barriers=*/NULL));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK, produced));
  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, &actual, sizeof(actual)));
  EXPECT_EQ(actual, payload);
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(tracker, axis, 1));
  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(tracker, axis, 1));
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(tracker, axis, 2));
  iree_async_frontier_tracker_release(tracker);
  iree_hal_semaphore_release(produced);
  iree_hal_buffer_release(buffer);
}

TEST(XdnaQueueTest, BlockedNativePublicationDoesNotBlockHostOrProactor) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.BlockSubmission();
  struct SubmissionReleaseGuard {
    // Provider whose explicit submission gate must be released on every exit.
    NativeProvider* provider;
    ~SubmissionReleaseGuard() {
      if (provider) {
        provider->ReleaseSubmission();
      }
    }
  } release_guard = {&harness.native};

  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  harness.native.AwaitBlockedSubmission();

  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_semaphore_t* host_completion = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &host_completion));
  const uint32_t payload = 0xA5C33C5A;
  iree_hal_transfer_operation_t transfer = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
  };
  transfer.update.target_buffer = buffer;
  transfer.update.source_buffer = &payload;
  transfer.update.length = sizeof(payload);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_transfer(harness.queue, {},
                                         {1, &host_completion, &value}, 1,
                                         &transfer, /*barriers=*/NULL));

  std::atomic<bool> sibling_completed{false};
  iree_async_operation_t sibling_operation = {};
  iree_async_operation_initialize(
      &sibling_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      +[](void* user_data, iree_async_operation_t* operation,
          iree_status_t status, iree_async_completion_flags_t flags) {
        IREE_EXPECT_OK(status);
        static_cast<std::atomic<bool>*>(user_data)->store(
            true, std::memory_order_release);
      },
      &sibling_completed);
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(harness.proactor, &sibling_operation));
  while (!sibling_completed.load(std::memory_order_acquire)) {
    IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                            iree_infinite_timeout(), nullptr));
  }
  ASSERT_NO_FATAL_FAILURE(
      harness.PollUntilDone(IREE_STATUS_OK, host_completion));

  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, &actual, sizeof(actual)));
  EXPECT_EQ(actual, payload);
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  harness.native.ReleaseSubmission();
  release_guard.provider = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  iree_hal_semaphore_release(host_completion);
  iree_hal_buffer_release(buffer);
}

TEST(XdnaQueueTest, WarmQueueOperationsAllocateNoHostStorage) {
  constexpr uint64_t kWarmupIterations = 4;
  constexpr uint64_t kMeasuredIterations = 64;
  constexpr uint64_t kTotalIterations = kWarmupIterations + kMeasuredIterations;

  HostAllocationCounters allocation_counters;
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  ASSERT_NO_FATAL_FAILURE(
      harness.Initialize(TrackingAllocator(&allocation_counters)));

  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  iree_hal_executable_t* switching_executable = nullptr;
  iree_hal_executable_function_t switching_function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(iree::hal::amd::xdna::testing::ImageFixture(),
                             &switching_executable, &switching_function));
  iree_hal_buffer_t* dispatch_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&dispatch_buffer));
  const iree_hal_buffer_ref_t binding =
      iree_hal_make_buffer_ref(dispatch_buffer, 0, 64);

  iree_hal_semaphore_t* ready_completion = nullptr;
  iree_hal_semaphore_t* reached_gate = nullptr;
  iree_hal_semaphore_t* wait_gate = nullptr;
  iree_hal_semaphore_t* wait_completion = nullptr;
  iree_hal_semaphore_t* switching_completion = nullptr;
  std::array<iree_hal_semaphore_t*, 3> capacity_completions = {};
  iree_hal_semaphore_t* small_update_completion = nullptr;
  iree_hal_semaphore_t* large_update_completion = nullptr;
  auto create_semaphore = [&](iree_hal_semaphore_t** out_semaphore) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore));
  };
  create_semaphore(&ready_completion);
  create_semaphore(&reached_gate);
  create_semaphore(&wait_gate);
  create_semaphore(&wait_completion);
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER, &switching_completion));
  for (iree_hal_semaphore_t** semaphore :
       {&capacity_completions[0], &capacity_completions[1],
        &capacity_completions[2]}) {
    create_semaphore(semaphore);
  }
  create_semaphore(&small_update_completion);
  create_semaphore(&large_update_completion);

  auto check_steady_state = [&](const char* name,
                                const HostAllocationSnapshot& before,
                                size_t native_allocations_before) {
    SCOPED_TRACE(name);
    ExpectHostAllocationsUnchanged(before, allocation_counters);
    EXPECT_EQ(harness.native.memory_create_count, native_allocations_before);
  };

  HostAllocationSnapshot before = {};
  size_t native_allocations_before = 0;
  for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
    if (iteration == kWarmupIterations + 1) {
      before = SnapshotHostAllocations(allocation_counters);
      native_allocations_before = harness.native.memory_create_count;
    }
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue, {}, {1, &ready_completion, &iteration}, executable,
        function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
        {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    harness.PollUntilValue(ready_completion, iteration);
  }
  check_steady_state("ready dispatch", before, native_allocations_before);

  iree_async_frontier_tracker_t* switching_tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &switching_tracker));
  IREE_ASSERT_OK(iree_hal_amd_xdna_queue_assign_frontier(
      harness.queue, switching_tracker,
      iree_async_axis_make_queue(1, 0, 0, 0, 0)));
  iree_async_frontier_tracker_release(switching_tracker);

  uint64_t switching_value = 0;
  for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
    if (iteration == kWarmupIterations + 1) {
      before = SnapshotHostAllocations(allocation_counters);
      native_allocations_before = harness.native.memory_create_count;
    }
    harness.native.hold_retirement = true;
    const size_t submission_count = harness.native.submission_count;
    uint64_t preceding_value = switching_value;
    uint64_t first_value = ++switching_value;
    uint64_t second_value = ++switching_value;
    const iree_hal_semaphore_list_t preceding_wait =
        preceding_value ? iree_hal_semaphore_list_t{1, &switching_completion,
                                                    &preceding_value}
                        : iree_hal_semaphore_list_t{};
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue, preceding_wait, {1, &switching_completion, &first_value},
        executable, function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
        {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue, {1, &switching_completion, &first_value},
        {1, &switching_completion, &second_value}, switching_executable,
        switching_function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
        {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(submission_count + 2));
    harness.native.hold_retirement = false;
    harness.native.Wake();
    harness.PollUntilValue(switching_completion, second_value);
  }
  check_steady_state("executable switching", before, native_allocations_before);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(reached_gate, 1, nullptr));
  for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
    if (iteration == kWarmupIterations + 1) {
      before = SnapshotHostAllocations(allocation_counters);
      native_allocations_before = harness.native.memory_create_count;
    }
    iree_hal_semaphore_t* wait_semaphores[] = {reached_gate, wait_gate};
    uint64_t wait_values[] = {1, iteration};
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue,
        {IREE_ARRAYSIZE(wait_semaphores), wait_semaphores, wait_values},
        {1, &wait_completion, &iteration}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
        /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                            iree_infinite_timeout(), nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_gate, iteration, nullptr));
    harness.PollUntilValue(wait_completion, iteration);
  }
  check_steady_state("unsatisfied wait", before, native_allocations_before);

  // The preceding host-signaled waits deliberately made the accepted frontier
  // inexact. Restore the production topology assignment at this quiescent point
  // so the capacity and BUSY rows exercise direct publication.
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  IREE_ASSERT_OK(iree_hal_amd_xdna_queue_assign_frontier(
      harness.queue, tracker, iree_async_axis_make_queue(1, 0, 0, 0, 0)));
  iree_async_frontier_tracker_release(tracker);

  for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
    if (iteration == kWarmupIterations + 1) {
      before = SnapshotHostAllocations(allocation_counters);
      native_allocations_before = harness.native.memory_create_count;
    }
    harness.native.hold_retirement = true;
    const size_t submission_count = harness.native.submission_count;
    for (iree_hal_semaphore_t* completion : capacity_completions) {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          harness.queue, {}, {1, &completion, &iteration}, executable, function,
          iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
          /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(submission_count + 2));
    harness.native.hold_retirement = false;
    harness.native.Wake();
    for (iree_hal_semaphore_t* completion : capacity_completions) {
      harness.PollUntilValue(completion, iteration);
    }
  }
  check_steady_state("native capacity", before, native_allocations_before);

  before = SnapshotHostAllocations(allocation_counters);
  native_allocations_before = harness.native.memory_create_count;
  harness.native.hold_retirement = true;
  uint64_t retry_value = kTotalIterations + 1;
  const uint64_t submission_count = harness.native.submission_count;
  const size_t notification_count = harness.native.notification_count;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {}, {1, &capacity_completions[0], &retry_value},
      executable, function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(submission_count + 1));
  ASSERT_NO_FATAL_FAILURE(
      harness.PollUntilNotificationCount(notification_count + 1));
  harness.native.ReturnBusyOnNextSubmission();
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {}, {1, &capacity_completions[1], &retry_value},
      executable, function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  EXPECT_EQ(harness.native.submission_count, submission_count + 1);
  const size_t flush_count_after_busy = harness.native.FlushCount();
  harness.native.retirement_limit =
      harness.native.last_accepted_submission.load(std::memory_order_acquire);
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(submission_count + 2));
  EXPECT_EQ(harness.native.FlushCount(), flush_count_after_busy);
  ASSERT_NO_FATAL_FAILURE(
      harness.PollUntilNotificationCount(notification_count + 2));
  harness.native.retirement_limit = UINT64_MAX;
  harness.native.hold_retirement = false;
  harness.native.Wake();
  harness.PollUntilValue(capacity_completions[0], retry_value);
  harness.PollUntilValue(capacity_completions[1], retry_value);
  check_steady_state("native BUSY retry", before, native_allocations_before);

  constexpr iree_host_size_t kLargeUpdateLength = 128 * 1024;
  iree_hal_buffer_t* transfer_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(
      &transfer_buffer, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      kLargeUpdateLength));
  std::array<uint8_t, 64> small_update;
  small_update.fill(0x5A);
  std::vector<uint8_t> large_update(kLargeUpdateLength, 0xA5);

  auto run_updates = [&](iree_const_byte_span_t source,
                         iree_hal_semaphore_t* completion) {
    iree_hal_transfer_operation_t transfer = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
    };
    transfer.update.source_buffer = source.data;
    transfer.update.target_buffer = transfer_buffer;
    transfer.update.length = source.data_length;
    for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
      if (iteration == kWarmupIterations + 1) {
        before = SnapshotHostAllocations(allocation_counters);
        native_allocations_before = harness.native.memory_create_count;
      }
      IREE_ASSERT_OK(iree_hal_queue_transfer(harness.queue, {},
                                             {1, &completion, &iteration}, 1,
                                             &transfer, /*barriers=*/NULL));
      harness.PollUntilValue(completion, iteration);
    }
  };

  run_updates(
      iree_make_const_byte_span(small_update.data(), small_update.size()),
      small_update_completion);
  check_steady_state("64-byte update", before, native_allocations_before);
  run_updates(
      iree_make_const_byte_span(large_update.data(), large_update.size()),
      large_update_completion);
  check_steady_state("128-KiB update", before, native_allocations_before);

  std::vector<uint8_t> actual(large_update.size());
  IREE_ASSERT_OK(iree_hal_buffer_map_read(transfer_buffer, 0, actual.data(),
                                          actual.size()));
  EXPECT_EQ(actual, large_update);

  iree_hal_buffer_release(transfer_buffer);
  iree_hal_semaphore_release(large_update_completion);
  iree_hal_semaphore_release(small_update_completion);
  for (iree_hal_semaphore_t* completion : capacity_completions) {
    iree_hal_semaphore_release(completion);
  }
  iree_hal_semaphore_release(wait_completion);
  iree_hal_semaphore_release(switching_completion);
  iree_hal_semaphore_release(wait_gate);
  iree_hal_semaphore_release(reached_gate);
  iree_hal_semaphore_release(ready_completion);
  iree_hal_buffer_release(dispatch_buffer);
  iree_hal_executable_release(switching_executable);
  iree_hal_executable_release(executable);
}

TEST(XdnaQueueTest, FailedDependencyDoesNotWaitForNativeCapacity) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(1));
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_semaphore_t *dependency = nullptr, *failed = nullptr;
  for (auto** semaphore : {&dependency, &failed}) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
  }
  uint64_t value = 1;
  auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {1, &dependency, &value}, {1, &failed, &value}, executable,
      function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, /*barriers=*/NULL, 0));
  iree_hal_buffer_release(buffer);
  iree_hal_executable_release(executable);
  iree_hal_semaphore_fail(dependency,
                          iree_status_from_code(IREE_STATUS_ABORTED));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_ABORTED, failed));
  EXPECT_EQ(harness.native.submission_count, 1u);
  EXPECT_EQ(harness.native.live_memories, 3u);
  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  iree_hal_semaphore_release(failed);
  iree_hal_semaphore_release(dependency);
}

TEST(XdnaQueueTest, RejectedSubmissionReleasesCapture) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.outcome = Outcome::kRejectSubmission;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(
      harness.PollUntilDone(IREE_STATUS_RESOURCE_EXHAUSTED));
  EXPECT_EQ(harness.live_memories_at_completion, 1u);
  EXPECT_EQ(harness.native.live_memories, 1u);
  EXPECT_EQ(harness.native.notification_count, 0u);
}

TEST(XdnaQueueTest, RetiredFailureReleasesCaptureBeforePublishingFailure) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.outcome = Outcome::kRetiredFailure;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_DATA_LOSS));
  EXPECT_EQ(harness.live_memories_at_completion, 1u);
  EXPECT_EQ(harness.native.live_memories, 1u);
  EXPECT_EQ(harness.native.refresh_count, 1u);
}

TEST(XdnaQueueTest, EarlyWakeRearmsUntilCheckedRetirement) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.outcome = Outcome::kEarlyWake;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_EQ(harness.live_memories_at_completion, 1u);
  EXPECT_EQ(harness.native.live_memories, 1u);
  EXPECT_EQ(harness.native.notification_count, 2u);
  EXPECT_EQ(harness.native.refresh_count, 2u);
}

TEST(XdnaQueueTest, PublishesRetirementBeforeFollowingNativeSubmit) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_hal_semaphore_t* following_completion = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &following_completion));
  harness.native.preceding_completion = harness.done;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.Submit(following_completion));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  ASSERT_NO_FATAL_FAILURE(
      harness.PollUntilDone(IREE_STATUS_OK, following_completion));
  EXPECT_EQ(harness.native.submission_count, 2u);
  EXPECT_EQ(harness.live_memories_at_completion, 3u);
  EXPECT_EQ(harness.native.live_memories, 1u);
  iree_hal_semaphore_release(following_completion);
}

TEST(XdnaQueueTest, FinalPublicationAllowsImmediateDeviceRelease) {
  HostAllocationCounters allocation_counters;
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  ASSERT_NO_FATAL_FAILURE(
      harness.Initialize(TrackingAllocator(&allocation_counters)));
  iree_hal_semaphore_t* first = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &first));
  ASSERT_NO_FATAL_FAILURE(harness.Submit(first));
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  iree_async_semaphore_timepoint_t release_timepoint = {
      .callback =
          +[](void* user_data, iree_async_semaphore_timepoint_t* timepoint,
              iree_status_t status) {
            IREE_EXPECT_OK(status);
            static_cast<QueueHarness*>(user_data)->ReleaseDevice();
          },
      .user_data = &harness,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(harness.done), 1,
      &release_timepoint));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDeviceDestroyed());
  EXPECT_EQ(harness.completion_status, IREE_STATUS_OK);
  EXPECT_EQ(harness.live_memories_at_completion, 1u);
  EXPECT_EQ(harness.native.live_memories, 0u);
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 1u);
  EXPECT_EQ(harness.native.instance_destroy_count, 1u);
  EXPECT_TRUE(harness.native.context_destroy_saw_queue);
  EXPECT_EQ(harness.diagnostic_count, 0u);
  iree_hal_semaphore_release(first);
  harness.ReleasePool();
  EXPECT_EQ(allocation_counters.live_allocations.load(), 0);
}

// These paths deliberately preserve live native ownership until process exit.
// Child processes verify the retained resources and use _Exit so deliberate
// retention does not become an accidental LeakSanitizer failure at exit.
static void RunCleanupFailurePreservesParentOwnershipChild() {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.destroy_status =
      amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  harness.ReleaseDevice();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilQueueDestroyAttempted());
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 0u);
  EXPECT_EQ(harness.diagnostic_count, 1u);
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

static void RunObserverFailurePreservesAcceptedOwnershipChild(Outcome outcome) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.outcome = outcome;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  const auto expected = outcome == Outcome::kUnretiredFailure
                            ? IREE_STATUS_DATA_LOSS
                            : IREE_STATUS_INTERNAL;
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(expected));
  EXPECT_EQ(harness.live_memories_at_completion, 3u);
  EXPECT_EQ(harness.native.live_memories, 3u);
  harness.ReleaseDevice();
  EXPECT_EQ(harness.native.queue_destroy_count, 0u);
  EXPECT_EQ(harness.native.context_destroy_count, 0u);
  EXPECT_GE(harness.diagnostic_count, 1u);
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

static void RunObserverFailurePreservesEveryPendingInvocationChild() {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_hal_semaphore_t* second = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &second));
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.Submit(second));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  harness.native.outcome = Outcome::kRefreshFailure;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_INTERNAL));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_INTERNAL, second));
  EXPECT_EQ(harness.native.live_memories, 5u);
  harness.ReleaseDevice();
  iree_hal_semaphore_release(second);
  EXPECT_EQ(harness.native.queue_destroy_count, 0u);
  EXPECT_EQ(harness.native.context_destroy_count, 0u);
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

static void RunBusyNativeDestroyPreservesParentChild() {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.destroy_status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  harness.ReleaseDevice();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilQueueDestroyAttempted());
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 0u);
  EXPECT_EQ(harness.diagnostic_count, 1u);
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

TEST(XdnaQueueDeathTest, CleanupFailurePreservesParentOwnership) {
  ASSERT_EXIT(RunCleanupFailurePreservesParentOwnershipChild(),
              ::testing::ExitedWithCode(0), "");
}

TEST(XdnaQueueDeathTest, ObserverFailuresPreserveAcceptedOwnership) {
  for (Outcome outcome :
       {Outcome::kNotificationFailure, Outcome::kRefreshFailure,
        Outcome::kUnretiredFailure, Outcome::kInactiveWithoutFailure}) {
    SCOPED_TRACE(static_cast<int>(outcome));
    EXPECT_EXIT(RunObserverFailurePreservesAcceptedOwnershipChild(outcome),
                ::testing::ExitedWithCode(0), "");
  }
}

TEST(XdnaQueueDeathTest, ObserverFailurePreservesEveryPendingInvocation) {
  EXPECT_EXIT(RunObserverFailurePreservesEveryPendingInvocationChild(),
              ::testing::ExitedWithCode(0), "");
}

TEST(XdnaQueueDeathTest, BusyNativeDestroyPreservesParent) {
  EXPECT_EXIT(RunBusyNativeDestroyPreservesParentChild(),
              ::testing::ExitedWithCode(0), "");
}

}  // namespace
