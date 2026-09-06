// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <thread>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

// Status code injected as the queue failure. Distinct from every code the queue
// synthesizes on its own so an assertion cannot pass on a substituted status.
constexpr iree_status_code_t kInjectedFailureCode = IREE_STATUS_DATA_LOSS;

// Status code of a second, losing failure injection.
constexpr iree_status_code_t kLateFailureCode = IREE_STATUS_UNKNOWN;

// Upper bound on submissions issued while waiting for one to run out of queue
// capacity. Generous against the configured ring sizes so the bound is only
// ever reached if capacity accounting stopped working.
constexpr uint64_t kMaxCapacityFillAttempts = 4096;

class HostQueueFailureTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueFailureTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueFailureTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueFailureTest::topology_;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_queue_t* queue() const {
    return iree_hal_device_queue(base_device_, /*family_ordinal=*/0,
                                 /*queue_ordinal=*/0);
  }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(base_device_);
  }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device_;
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
    if (logical_device->physical_device_count == 0) return NULL;
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[0];
    if (physical_device->host_queue_count == 0) return NULL;
    return &physical_device->host_queues[0];
  }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device_ = NULL;

  // Device group that owns the topology assigned to |base_device_|.
  iree_hal_device_group_t* device_group_ = NULL;
};

static iree_status_t CreateHostVisibleTransferBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL |
                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                     iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_hal_semaphore_list_t MakeSemaphoreList(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return iree_hal_semaphore_list_t{
      /*count=*/1,
      /*semaphores=*/semaphore,
      /*payload_values=*/payload_value,
  };
}

static bool HostQueueHasPendingOps(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool has_pending_ops = queue->pending_head != NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return has_pending_ops;
}

static bool HostQueueHasPostDrainAction(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  const bool has_action = queue->post_drain.head != NULL;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
  return has_action;
}

static iree_status_code_t HostQueueErrorStatusCode(
    iree_hal_amdgpu_host_queue_t* queue) {
  return iree_status_code((iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire));
}

// Waits until hardware has retired every epoch the queue submitted.
//
// A recorded failure short-circuits every HAL wait, so a test can get its
// assertions back while the packets it submitted are still executing. The
// queue's own epoch signal is the only thing that orders that GPU work before
// the buffers it writes and the queue it runs on are destroyed.
static void WaitForSubmittedEpoch(const iree_hal_amdgpu_libhsa_t* libhsa,
                                  iree_hal_amdgpu_host_queue_t* queue) {
  const uint64_t submitted_epoch =
      queue->notification_ring.epoch.next_submission;
  if (submitted_epoch == 0) return;
  const hsa_signal_value_t compare_value =
      (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                           submitted_epoch) +
      1;
  (void)iree_hsa_signal_wait_scacquire(
      IREE_LIBHSA(libhsa),
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring),
      HSA_SIGNAL_CONDITION_LT, compare_value, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED);
}

// Enqueues a raw AQL barrier that blocks the hardware queue until
// |blocker_signal| reaches zero, holding every packet behind it in flight.
static void EnqueueRawBlockingBarrier(iree_hal_amdgpu_host_queue_t* queue,
                                      hsa_signal_t blocker_signal) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dep_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dep_signals, IREE_ARRAYSIZE(dep_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(),
      iree_hsa_signal_null());
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
}

// Deferred operations must report the failure the queue actually recorded, not
// a status synthesized by the cancellation path.
TEST_F(HostQueueFailureTest, DeferredOperationReportsRecordedQueueFailure) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  // Never signalled, so the fill below can only ever remain deferred.
  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  const iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1100u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      test_device.queue(), wait_list, signal_list, target_buffer,
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));

  // The terminal transition runs on the completion thread; this wait is the
  // only readiness contract the queue exposes for its completion.
  IREE_EXPECT_STATUS_IS(kInjectedFailureCode,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));

  IREE_EXPECT_OK(
      iree_hal_semaphore_signal(wait_semaphore, wait_value, /*frontier=*/NULL));
}

// Submitted notification entries report the same recorded failure the deferred
// operations above do.
TEST_F(HostQueueFailureTest, SubmittedEntryReportsRecordedQueueFailure) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  // Holds the submitted fill in flight so the failure drain observes it as an
  // outstanding notification entry rather than a completed one.
  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  EnqueueRawBlockingBarrier(queue, blocker_signal);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1101u;
  IREE_ASSERT_OK(
      iree_hal_queue_fill(test_device.queue(), iree_hal_semaphore_list_empty(),
                          signal_list, target_buffer,
                          /*target_offset=*/0, sizeof(pattern), &pattern,
                          sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  EXPECT_FALSE(HostQueueHasPendingOps(queue));

  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));

  // Release the hardware queue only after the failure is recorded so the fill
  // cannot have signalled its semaphore before the terminal transition began.
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  IREE_EXPECT_STATUS_IS(kInjectedFailureCode,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

  WaitForSubmittedEpoch(&libhsa_, queue);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
}

// The queue records only its first failure; later failures are dropped and
// their statuses freed.
TEST_F(HostQueueFailureTest, LaterQueueFailureDoesNotReplaceTheFirst) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  EXPECT_EQ(HostQueueErrorStatusCode(queue), IREE_STATUS_OK);
  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));
  EXPECT_EQ(HostQueueErrorStatusCode(queue), kInjectedFailureCode);
  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kLateFailureCode, "late queue failure"));
  EXPECT_EQ(HostQueueErrorStatusCode(queue), kInjectedFailureCode);
}

// A queue that has finished its terminal transition rejects further work
// instead of admitting it to a queue whose completion thread has exited.
TEST_F(HostQueueFailureTest, SubmissionAfterQueueFailureIsRejected) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  const iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1102u;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      test_device.queue(), wait_list, signal_list, target_buffer,
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE));
  ASSERT_TRUE(HostQueueHasPendingOps(queue));

  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));

  // Ordering the submission attempt behind this wait is what makes the
  // rejection deterministic: the transition is complete once a cancelled
  // deferred operation has failed its signal semaphore.
  IREE_EXPECT_NOT_OK(iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_semaphore_t> late_signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), late_signal_semaphore.out()));
  uint64_t late_signal_value = 1;
  iree_hal_semaphore_t* late_signal_semaphore_ptr = late_signal_semaphore.get();
  const iree_hal_semaphore_list_t late_signal_list =
      MakeSemaphoreList(&late_signal_semaphore_ptr, &late_signal_value);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_queue_fill(test_device.queue(), iree_hal_semaphore_list_empty(),
                          late_signal_list, target_buffer,
                          /*target_offset=*/0, sizeof(pattern), &pattern,
                          sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  IREE_EXPECT_OK(
      iree_hal_semaphore_signal(wait_semaphore, wait_value, /*frontier=*/NULL));
}

// The teardown wait reaches the HSA runtime only after deciding whether to
// block at all: it returns without waiting on a queue that has already failed.
// Its call into HSA is therefore the first instant at which that decision is
// behind it, and installing a thunk over that call in a private copy of the
// symbol table is what lets a test act there. That table only exists in the
// default dynamic build.
#if !IREE_HAL_AMDGPU_LIBHSA_STATIC

// Observation point for the teardown wait. The queue's completion thread blocks
// in the same entry point throughout, so only the armed thread's next wait is
// observed and every other call passes straight through.
struct TeardownWaitHook {
  // Real entry point every call delegates to.
  decltype(iree_hal_amdgpu_libhsa_t::hsa_amd_signal_wait_any) next;
  // Thread whose armed wait is observed. Written before the device built on the
  // hooked table exists and never again, so no HSA waiter races it.
  std::thread::id thread;
  // Set while the next wait on |thread| is the teardown wait.
  iree_atomic_int32_t armed;
  // Set once that wait has been reached.
  iree_atomic_int32_t reached;
  // Posted with |reached|.
  iree_notification_t reached_notification;
};

TeardownWaitHook g_teardown_wait_hook;

static uint32_t HSA_API TeardownWaitHookEntry(
    uint32_t signal_count, hsa_signal_t* signals, hsa_signal_condition_t* conds,
    hsa_signal_value_t* values, uint64_t timeout_hint,
    hsa_wait_state_t wait_hint, hsa_signal_value_t* satisfying_value) {
  if (std::this_thread::get_id() == g_teardown_wait_hook.thread &&
      iree_atomic_exchange(&g_teardown_wait_hook.armed, 0,
                           iree_memory_order_acq_rel) != 0) {
    iree_atomic_store(&g_teardown_wait_hook.reached, 1,
                      iree_memory_order_release);
    iree_notification_post(&g_teardown_wait_hook.reached_notification,
                           IREE_ALL_WAITERS);
  }
  return g_teardown_wait_hook.next(signal_count, signals, conds, values,
                                   timeout_hint, wait_hint, satisfying_value);
}

static bool TeardownWaitWasReached(void* user_data) {
  TeardownWaitHook* hook = (TeardownWaitHook*)user_data;
  return iree_atomic_load(&hook->reached, iree_memory_order_acquire) != 0;
}

// The teardown wait blocks until hardware retires the last submitted epoch, and
// a GPU that faulted never will. A failure delivered while teardown is already
// blocked there is the only thing that can release it, so failure delivery has
// to stay live from the moment admission closes until the queues are destroyed.
//
// The failure arrives from another thread and cannot arrive earlier: the thread
// that records it is released from inside the wait's own call into HSA. Nothing
// here depends on how the two threads are scheduled, which matters because
// recording the failure before the wait starts would skip the wait entirely -
// the call returns immediately on an already-failed queue - and leave the
// release path this covers unexercised while still passing.
TEST_F(HostQueueFailureTest, FailureDuringTeardownWaitReleasesIt) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  // Symbol table the device below is built on: this suite's table with the
  // blocking wait replaced. The device copies it during creation and holds its
  // own references to HSA and to the library, so this copy is released at the
  // end of the test while the device is still using those symbols.
  iree_hal_amdgpu_libhsa_t hooked_libhsa;
  IREE_ASSERT_OK(iree_hal_amdgpu_libhsa_copy(&libhsa_, &hooked_libhsa));
  g_teardown_wait_hook.next = hooked_libhsa.hsa_amd_signal_wait_any;
  g_teardown_wait_hook.thread = std::this_thread::get_id();
  iree_atomic_store(&g_teardown_wait_hook.armed, 0, iree_memory_order_release);
  iree_atomic_store(&g_teardown_wait_hook.reached, 0,
                    iree_memory_order_release);
  iree_notification_initialize(&g_teardown_wait_hook.reached_notification);
  hooked_libhsa.hsa_amd_signal_wait_any = TeardownWaitHookEntry;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, &hooked_libhsa, &topology_,
                                        host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  // Stands in for the faulted GPU: the epoch submitted below cannot retire
  // while this barrier holds the hardware queue.
  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  EnqueueRawBlockingBarrier(queue, blocker_signal);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1103u;
  IREE_ASSERT_OK(
      iree_hal_queue_fill(test_device.queue(), iree_hal_semaphore_list_empty(),
                          signal_list, target_buffer,
                          /*target_offset=*/0, sizeof(pattern), &pattern,
                          sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Teardown order: admission closes first, then the wait, with the queues
  // still able to receive a failure throughout.
  iree_hal_amdgpu_host_queue_begin_deinitialize(queue);

  // Stands in for the delivery path: a thread outside teardown recording the
  // failure, held until the wait it has to release has reached HSA.
  std::thread failing_thread([&]() {
    iree_notification_await(&g_teardown_wait_hook.reached_notification,
                            TeardownWaitWasReached, &g_teardown_wait_hook,
                            iree_infinite_timeout());
    iree_hal_amdgpu_host_queue_record_failure(
        queue,
        iree_make_status(kInjectedFailureCode, "injected queue failure"));
  });

  // Nothing has failed the queue yet, so the call below has no early exit and
  // blocks on an epoch the parked hardware will never publish.
  EXPECT_EQ(HostQueueErrorStatusCode(queue), IREE_STATUS_OK);
  iree_atomic_store(&g_teardown_wait_hook.armed, 1, iree_memory_order_release);

  // Returns only because the failure recorded on the other thread raised the
  // stop signal this wait is armed on: the epoch signal it is also armed on
  // cannot advance, and the thread that records the failure is released by this
  // call and by nothing else.
  iree_hal_amdgpu_host_queue_wait_idle_before_deinitialize(queue);

  // The hook consumes the arming when it runs, so an arming still set means the
  // wait returned without reaching HSA - the one shape in which the recorded
  // failure is not what ended it. Clearing it also keeps the hook away from a
  // notification this test tears down before the device is destroyed, since
  // destruction runs another teardown wait on this thread.
  EXPECT_EQ(iree_atomic_exchange(&g_teardown_wait_hook.armed, 0,
                                 iree_memory_order_acq_rel),
            0);

  // Releases the failing thread on the paths where the wait did not, so a run
  // that has already failed an assertion still terminates.
  iree_atomic_store(&g_teardown_wait_hook.reached, 1,
                    iree_memory_order_release);
  iree_notification_post(&g_teardown_wait_hook.reached_notification,
                         IREE_ALL_WAITERS);
  failing_thread.join();
  EXPECT_EQ(HostQueueErrorStatusCode(queue), kInjectedFailureCode);
  IREE_EXPECT_STATUS_IS(kInjectedFailureCode,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  WaitForSubmittedEpoch(&libhsa_, queue);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
  iree_notification_deinitialize(&g_teardown_wait_hook.reached_notification);
  iree_hal_amdgpu_libhsa_deinitialize(&hooked_libhsa);
}

#else

// A static HSA link has no function-pointer table to install a thunk into, so
// the wait cannot be observed from inside and the coverage above is not
// buildable. Report that as a skip rather than leaving it looking as though it
// ran.
TEST_F(HostQueueFailureTest, TeardownWaitCoverageRequiresDynamicLibhsa) {
  GTEST_SKIP() << "teardown wait coverage requires the dynamic libhsa function "
                  "table";
}

#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

// An operation that ran out of submission capacity parks itself as a post-drain
// action, and the terminal transition closes admission before it flushes those
// actions. The operation the flush resumes must still report the failure the
// queue recorded, because every other operation the same transition settles
// reports it.
TEST_F(HostQueueFailureTest, CapacityParkedOperationReportsRecordedFailure) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  // Small enough that a handful of submissions exhaust the ring rather than a
  // thousand of them.
  options.host_queues.notification_capacity = 16;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  // Nothing retires while this barrier holds the hardware queue, so submissions
  // consume ring capacity that is never reclaimed.
  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  EnqueueRawBlockingBarrier(queue, blocker_signal);

  // Submitted operations share this one, so its failure comes from the drain
  // that fails the notification ring rather than from the parked retry.
  Ref<iree_hal_semaphore_t> bulk_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), bulk_semaphore.out()));
  iree_hal_semaphore_t* bulk_semaphore_ptr = bulk_semaphore.get();

  // Submit with no waits so an operation that defers can only have deferred for
  // capacity, then stop as soon as one has.
  const uint32_t pattern = 0xCACE1104u;
  bool parked = false;
  for (uint64_t i = 0; i < kMaxCapacityFillAttempts && !parked; ++i) {
    uint64_t bulk_value = i + 1;
    const iree_hal_semaphore_list_t bulk_list =
        MakeSemaphoreList(&bulk_semaphore_ptr, &bulk_value);
    IREE_ASSERT_OK(iree_hal_queue_fill(
        test_device.queue(), iree_hal_semaphore_list_empty(), bulk_list,
        target_buffer,
        /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
        IREE_HAL_FILL_FLAG_NONE));
    parked = HostQueueHasPostDrainAction(queue);
  }
  ASSERT_TRUE(parked) << "no submission ran out of capacity";

  // Capacity is still exhausted, so this one parks too. Its own semaphore is
  // written by nothing else, which is what makes the assertion below name the
  // status the parked retry produced.
  Ref<iree_hal_semaphore_t> parked_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), parked_semaphore.out()));
  uint64_t parked_value = 1;
  iree_hal_semaphore_t* parked_semaphore_ptr = parked_semaphore.get();
  const iree_hal_semaphore_list_t parked_list =
      MakeSemaphoreList(&parked_semaphore_ptr, &parked_value);
  IREE_ASSERT_OK(
      iree_hal_queue_fill(test_device.queue(), iree_hal_semaphore_list_empty(),
                          parked_list, target_buffer,
                          /*target_offset=*/0, sizeof(pattern), &pattern,
                          sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));

  // The parked operation is resumed by the transition's post-drain flush, after
  // admission has already closed. It must report the recorded failure rather
  // than the closed admission.
  IREE_EXPECT_STATUS_IS(kInjectedFailureCode,
                        iree_hal_semaphore_wait(parked_semaphore, parked_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  WaitForSubmittedEpoch(&libhsa_, queue);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
}

// A producer driven from the queue's own post-drain flush, which runs inside
// the terminal transition's drain. Standing in for any thread that reaches a
// submission entry point while that transition is running, it records what the
// queue answered and posts when it has run.
struct RacingProducer {
  // Queue the producer submits to.
  iree_hal_amdgpu_host_queue_t* queue = NULL;
  // Buffer the submitted fill targets.
  iree_hal_buffer_t* target_buffer = NULL;
  // Action storage owned by the test for the duration of the transition.
  iree_hal_amdgpu_host_queue_post_drain_action_t action = {};
  // Status the submission returned. Owned by the test.
  iree_status_t submit_status = iree_ok_status();
  // Posted once the submission has been attempted.
  iree_notification_t ran;
  // Whether the submission has been attempted.
  iree_atomic_int32_t has_run = IREE_ATOMIC_VAR_INIT(0);

  RacingProducer() { iree_notification_initialize(&ran); }
  ~RacingProducer() { iree_notification_deinitialize(&ran); }
};

static void RacingProducerSubmit(void* user_data) {
  RacingProducer* producer = (RacingProducer*)user_data;
  const uint32_t pattern = 0xCACE1104u;
  producer->submit_status = iree_hal_queue_fill(
      &producer->queue->base, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), producer->target_buffer,
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE);
  iree_atomic_store(&producer->has_run, 1, iree_memory_order_release);
  iree_notification_post(&producer->ran, IREE_ALL_WAITERS);
}

static bool RacingProducerHasRun(void* user_data) {
  RacingProducer* producer = (RacingProducer*)user_data;
  return iree_atomic_load(&producer->has_run, iree_memory_order_acquire) != 0;
}

// A fault can arrive while a producer is mid-flight, and the queue gets exactly
// one failure drain. If that drain runs before admission closes, work published
// behind it is admitted to a queue whose completion thread is already leaving,
// and nothing ever drains it: the notification entry, its semaphores, its
// frontier state and its retained resources all stay live until the device is
// destroyed, and a waiter hangs in the meantime.
//
// The transition therefore closes admission under submission_mutex first, which
// waits out whatever publisher is inside it, and only then drains a published
// state that can no longer grow. A post-drain action is a producer at the
// sharpest point of that window - the flush runs from inside the drain - so
// what the queue answers it is what the ordering decides.
TEST_F(HostQueueFailureTest, ProducerDuringTheFailureDrainIsRejected) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));

  // Stands in for the faulted GPU: nothing submitted below can retire, so the
  // completion thread wakes for the failure and for nothing else.
  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  EnqueueRawBlockingBarrier(queue, blocker_signal);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  const iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  const uint32_t pattern = 0xCACE1105u;
  IREE_ASSERT_OK(
      iree_hal_queue_fill(test_device.queue(), iree_hal_semaphore_list_empty(),
                          signal_list, target_buffer,
                          /*target_offset=*/0, sizeof(pattern), &pattern,
                          sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  RacingProducer producer;
  producer.queue = queue;
  producer.target_buffer = target_buffer;
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      queue, &producer.action, RacingProducerSubmit, &producer);

  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));

  iree_notification_await(&producer.ran, RacingProducerHasRun, &producer,
                          iree_infinite_timeout());

  // Admission was already closed when the flush reached the producer, so its
  // submission never joined a published state the drain had finished with.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, producer.submit_status);
  iree_status_free(producer.submit_status);
  EXPECT_EQ(HostQueueErrorStatusCode(queue), kInjectedFailureCode);
  IREE_EXPECT_STATUS_IS(kInjectedFailureCode,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);
  WaitForSubmittedEpoch(&libhsa_, queue);
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));
}

// Deinitialization frees the recorded failure, so it must also take it out of
// the slot. Every reader of the slot reads it to clone a status from, and the
// post-drain pass deinitialization itself runs after the free is one of them,
// so a slot left pointing at the freed status is read as a live one.
TEST_F(HostQueueFailureTest, TeardownReleasesTheRecordedQueueFailure) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_amdgpu_host_queue_record_failure(
      queue, iree_make_status(kInjectedFailureCode, "injected queue failure"));
  ASSERT_EQ(HostQueueErrorStatusCode(queue), kInjectedFailureCode);

  // Frontier deassignment is the teardown the device destructor performs, and
  // the queue storage is inline in the physical device, so the slot is still
  // readable after the queue it belongs to has been torn down.
  iree_hal_amdgpu_physical_device_deassign_frontier(
      test_device.logical_device()->physical_devices[0]);

  EXPECT_EQ(iree_atomic_load(&queue->error_status, iree_memory_order_acquire),
            0);
}

}  // namespace
}  // namespace iree::hal::amdgpu
