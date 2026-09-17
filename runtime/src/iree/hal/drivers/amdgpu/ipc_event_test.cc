// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/ipc_event.h"

#include <array>
#include <atomic>
#include <cstring>
#include <thread>

#include "iree/async/semaphore.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/ipc_event_monitor.h"
#include "iree/hal/drivers/amdgpu/ipc_event_monitor_test_util.h"
#include "iree/hal/drivers/amdgpu/ipc_event_test_util.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static_assert(sizeof(iree_hal_amdgpu_ipc_event_token_t) == 32);

TEST(IpcEventTest, RetainNullIsNoOp) {
  iree_hal_amdgpu_ipc_event_retain(nullptr);
}

TEST(IpcEventTest, RejectsNonAmdgpuDevice) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view("ipc-event-test");
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));

  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_ipc_event_create(device, &event));
  EXPECT_EQ(event, nullptr);

  iree_hal_amdgpu_ipc_event_token_t token = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_ipc_event_import(device, token, &event));
  EXPECT_EQ(event, nullptr);

  iree_hal_device_release(device);
}

#if !IREE_HAL_AMDGPU_LIBHSA_STATIC

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa,
                           const iree_hal_amdgpu_topology_t* topology,
                           iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), &options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_amdgpu_libhsa_t* device_libhsa() const {
    auto* logical_device =
        reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(base_device_);
    return &logical_device->system->libhsa;
  }

 private:
  // Owns the parameters and frontier tracker used to create the device.
  iree::hal::cts::DeviceCreateContext create_context_;
  // Logical device owned by the fixture.
  iree_hal_device_t* base_device_ = nullptr;
  // Group retaining the logical device's topology association.
  iree_hal_device_group_t* device_group_ = nullptr;
};

class IpcEventGpuTest : public ::testing::Test {
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
    iree_hal_amdgpu_ipc_event_monitor_shutdown();
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  void SetUp() override {
    iree_status_t status =
        device_.Initialize(&libhsa_, &topology_, host_allocator_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "AMDGPU logical device unavailable, skipping tests";
    }
  }

  void TearDown() override {
    // The monitor must release all carrier and destination-device references
    // before the fixture begins logical-device teardown.
    iree_hal_amdgpu_ipc_event_monitor_shutdown();
  }

  // Allocator shared by suite fixtures.
  static iree_allocator_t host_allocator_;
  // Dynamically loaded HSA dispatch table.
  static iree_hal_amdgpu_libhsa_t libhsa_;
  // GPU topology shared by suite fixtures.
  static iree_hal_amdgpu_topology_t topology_;
  // Logical device owned by each individual test.
  TestLogicalDevice device_;
};

iree_allocator_t IpcEventGpuTest::host_allocator_;
iree_hal_amdgpu_libhsa_t IpcEventGpuTest::libhsa_;
iree_hal_amdgpu_topology_t IpcEventGpuTest::topology_;

static bool AtomicFlagIsSet(void* user_data) {
  auto* flag = static_cast<iree_atomic_int32_t*>(user_data);
  return iree_atomic_load(flag, iree_memory_order_acquire) != 0;
}

struct SignalGate {
  // Wakes the test after the hooked monitor operation enters the gate.
  iree_notification_t entered_notification;
  // Wakes the monitor after the test permits the operation to continue.
  iree_notification_t continue_notification;
  // True after the hooked monitor operation enters the gate.
  iree_atomic_int32_t entered;
  // True after the test permits the hooked monitor operation to continue.
  iree_atomic_int32_t may_continue;
  // Signal handle selected for gating.
  uint64_t target_signal_handle;
  // Thread that entered the gate.
  std::thread::id worker_thread;
};

static void SignalGateInitialize(uint64_t target_signal_handle,
                                 SignalGate* gate) {
  iree_notification_initialize(&gate->entered_notification);
  iree_notification_initialize(&gate->continue_notification);
  iree_atomic_store(&gate->entered, 0, iree_memory_order_relaxed);
  iree_atomic_store(&gate->may_continue, 0, iree_memory_order_relaxed);
  gate->target_signal_handle = target_signal_handle;
  gate->worker_thread = {};
}

static void SignalGateAwaitEntered(SignalGate* gate) {
  iree_notification_await(&gate->entered_notification, AtomicFlagIsSet,
                          &gate->entered, iree_infinite_timeout());
}

static void SignalGateContinue(SignalGate* gate) {
  iree_atomic_store(&gate->may_continue, 1, iree_memory_order_release);
  iree_notification_post(&gate->continue_notification, IREE_ALL_WAITERS);
}

static void SignalGateDeinitialize(SignalGate* gate) {
  iree_notification_deinitialize(&gate->continue_notification);
  iree_notification_deinitialize(&gate->entered_notification);
}

static void GateRecordCallback(void* user_data) {
  auto* gate = static_cast<SignalGate*>(user_data);
  if (iree_atomic_exchange(&gate->entered, 1, iree_memory_order_acq_rel) == 0) {
    gate->worker_thread = std::this_thread::get_id();
    iree_notification_post(&gate->entered_notification, IREE_ALL_WAITERS);
    iree_notification_await(&gate->continue_notification, AtomicFlagIsSet,
                            &gate->may_continue, iree_infinite_timeout());
  }
}

static void NotifySignalGate(void* user_data) {
  auto* gate = static_cast<SignalGate*>(user_data);
  if (iree_atomic_exchange(&gate->entered, 1, iree_memory_order_acq_rel) == 0) {
    gate->worker_thread = std::this_thread::get_id();
    iree_notification_post(&gate->entered_notification, IREE_ALL_WAITERS);
  }
}

static std::atomic<uint64_t> captured_signal_handle{0};
static hsa_signal_value_t(HSA_API* forward_signal_load_scacquire)(
    hsa_signal_t signal) = nullptr;

static hsa_signal_value_t HSA_API CaptureSignalLoad(hsa_signal_t signal) {
  captured_signal_handle.store(signal.handle, std::memory_order_release);
  return forward_signal_load_scacquire(signal);
}

static iree_status_t CaptureEventSignalHandle(
    iree_hal_amdgpu_ipc_event_t* event, iree_hal_amdgpu_libhsa_t* libhsa,
    uint64_t* out_signal_handle) {
  auto original_signal_load = libhsa->hsa_signal_load_scacquire;
  forward_signal_load_scacquire = original_signal_load;
  captured_signal_handle.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_load_scacquire = &CaptureSignalLoad;
  bool reached = false;
  iree_status_t status = iree_hal_amdgpu_ipc_event_query(event, &reached);
  libhsa->hsa_signal_load_scacquire = original_signal_load;
  forward_signal_load_scacquire = nullptr;
  *out_signal_handle = captured_signal_handle.load(std::memory_order_acquire);
  return status;
}

static std::atomic<int> ipc_signal_attach_call_count{0};
static hsa_status_t(HSA_API* forward_ipc_signal_attach)(
    const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) = nullptr;

static hsa_status_t HSA_API
CountIpcSignalAttach(const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) {
  ipc_signal_attach_call_count.fetch_add(1, std::memory_order_relaxed);
  return forward_ipc_signal_attach(handle, signal);
}

static hsa_status_t HSA_API RejectIpcSignalAttach(
    const hsa_amd_ipc_signal_t* handle, hsa_signal_t* signal) {
  (void)handle;
  (void)signal;
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

struct FirstImportCandidateGate {
  // Blocks the first candidate after allocation.
  SignalGate gate;
  // Counts candidate-allocation hook invocations.
  std::atomic<int> invocation_count;
};

static void GateFirstImportCandidate(void* user_data) {
  auto* candidate_gate = static_cast<FirstImportCandidateGate*>(user_data);
  if (candidate_gate->invocation_count.fetch_add(
          1, std::memory_order_relaxed) == 0) {
    GateRecordCallback(&candidate_gate->gate);
  }
}

struct RefDecrementObserver {
  // Invocation count indexed by decrement kind.
  std::array<std::atomic<int>,
             IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_COUNT>
      count;
  // Last old reference count indexed by decrement kind.
  std::array<std::atomic<int32_t>,
             IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_COUNT>
      old_ref_count;
};

static void InitializeRefDecrementObserver(RefDecrementObserver* observer) {
  for (size_t i = 0; i < IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_COUNT;
       ++i) {
    observer->count[i].store(0, std::memory_order_relaxed);
    observer->old_ref_count[i].store(0, std::memory_order_relaxed);
  }
}

static void ObserveRefDecrement(void* user_data, int32_t decrement_kind,
                                int32_t old_ref_count) {
  auto* observer = static_cast<RefDecrementObserver*>(user_data);
  if (decrement_kind < 0 ||
      decrement_kind >= IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_COUNT) {
    return;
  }
  observer->old_ref_count[decrement_kind].store(old_ref_count,
                                                std::memory_order_relaxed);
  observer->count[decrement_kind].fetch_add(1, std::memory_order_release);
}

static SignalGate* active_signal_destroy_gate = nullptr;
static std::atomic<uint64_t> counted_destroy_signal_handle{0};
static std::atomic<int> counted_signal_destroy_count{0};
static hsa_status_t(HSA_API* forward_signal_destroy)(hsa_signal_t signal) =
    nullptr;

static hsa_status_t HSA_API CountTargetSignalDestroy(hsa_signal_t signal) {
  if (signal.handle ==
      counted_destroy_signal_handle.load(std::memory_order_acquire)) {
    counted_signal_destroy_count.fetch_add(1, std::memory_order_relaxed);
  }
  return forward_signal_destroy(signal);
}

static hsa_status_t HSA_API GateSignalDestroy(hsa_signal_t signal) {
  SignalGate* gate = active_signal_destroy_gate;
  if (gate && signal.handle == gate->target_signal_handle &&
      iree_atomic_exchange(&gate->entered, 1, iree_memory_order_acq_rel) == 0) {
    gate->worker_thread = std::this_thread::get_id();
    iree_notification_post(&gate->entered_notification, IREE_ALL_WAITERS);
    iree_notification_await(&gate->continue_notification, AtomicFlagIsSet,
                            &gate->may_continue, iree_infinite_timeout());
  }
  return forward_signal_destroy(signal);
}

static SignalGate* active_signal_store_gate = nullptr;
static std::atomic<uint64_t> stored_signal_handle{0};
static std::atomic<uint64_t> counted_completion_signal_handle{0};
static std::atomic<int> counted_completion_store_count{0};
static void(HSA_API* forward_signal_store_screlease)(
    hsa_signal_t signal, hsa_signal_value_t value) = nullptr;

static void HSA_API GateFirstCompletionStore(hsa_signal_t signal,
                                             hsa_signal_value_t value) {
  if (value == 1) {
    uint64_t expected_handle = 0;
    stored_signal_handle.compare_exchange_strong(expected_handle, signal.handle,
                                                 std::memory_order_acq_rel);
  }
  if (value == 0 && signal.handle == counted_completion_signal_handle.load(
                                         std::memory_order_acquire)) {
    counted_completion_store_count.fetch_add(1, std::memory_order_relaxed);
  }
  forward_signal_store_screlease(signal, value);
  SignalGate* gate = active_signal_store_gate;
  if (gate && value == 0 && signal.handle == gate->target_signal_handle &&
      iree_atomic_exchange(&gate->entered, 1, iree_memory_order_acq_rel) == 0) {
    gate->worker_thread = std::this_thread::get_id();
    iree_notification_post(&gate->entered_notification, IREE_ALL_WAITERS);
    iree_notification_await(&gate->continue_notification, AtomicFlagIsSet,
                            &gate->may_continue, iree_infinite_timeout());
  }
}

// Models the production interval after the semaphore release-store becomes
// acquire-visible and before its pending timepoint is detached for dispatch.
static void PublishSemaphoreTimelineWithoutDispatch(
    iree_hal_semaphore_t* semaphore, uint64_t value) {
  auto* async_semaphore = reinterpret_cast<iree_async_semaphore_t*>(semaphore);
  iree_atomic_store(&async_semaphore->timeline_value, (int64_t)value,
                    iree_memory_order_release);
}

// Models the corresponding failure interval after the sticky failure CAS and
// before failure dispatch detaches the pending timepoint.
static void PublishSemaphoreFailureWithoutDispatch(
    iree_hal_semaphore_t* semaphore) {
  auto* async_semaphore = reinterpret_cast<iree_async_semaphore_t*>(semaphore);
  const intptr_t failure =
      reinterpret_cast<intptr_t>(iree_status_from_code(IREE_STATUS_ABORTED));
  intptr_t expected = 0;
  const bool stored = iree_atomic_compare_exchange_strong(
      &async_semaphore->failure_status, &expected, failure,
      iree_memory_order_release, iree_memory_order_acquire);
  IREE_ASSERT(stored);
}

static SignalGate* active_signal_load_gate = nullptr;

static hsa_signal_value_t HSA_API GateFirstSignalLoad(hsa_signal_t signal) {
  const hsa_signal_value_t value = forward_signal_load_scacquire(signal);
  SignalGate* gate = active_signal_load_gate;
  if (gate && signal.handle == gate->target_signal_handle &&
      iree_atomic_exchange(&gate->entered, 1, iree_memory_order_acq_rel) == 0) {
    gate->worker_thread = std::this_thread::get_id();
    iree_notification_post(&gate->entered_notification, IREE_ALL_WAITERS);
    iree_notification_await(&gate->continue_notification, AtomicFlagIsSet,
                            &gate->may_continue, iree_infinite_timeout());
  }
  return value;
}

struct BlockingMonitorOperation {
  // Intrusive monitor operation; must remain the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t operation;
  // Blocks the monitor's first poll before a target operation is published.
  SignalGate gate;
};

static bool PollBlockingMonitorOperation(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  auto* blocker = reinterpret_cast<BlockingMonitorOperation*>(operation);
  (void)was_requested;
  if (iree_atomic_exchange(&blocker->gate.entered, 1,
                           iree_memory_order_acq_rel) == 0) {
    blocker->gate.worker_thread = std::this_thread::get_id();
    iree_notification_post(&blocker->gate.entered_notification,
                           IREE_ALL_WAITERS);
    iree_notification_await(&blocker->gate.continue_notification,
                            AtomicFlagIsSet, &blocker->gate.may_continue,
                            iree_infinite_timeout());
  }
  return is_shutting_down;
}

static void BlockingMonitorOperationStart(BlockingMonitorOperation* blocker) {
  SignalGateInitialize(/*target_signal_handle=*/0, &blocker->gate);
  iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
      PollBlockingMonitorOperation, IREE_DURATION_INFINITE,
      &blocker->operation);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&blocker->operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&blocker->operation);
  SignalGateAwaitEntered(&blocker->gate);
}

static void BlockingMonitorOperationContinue(
    BlockingMonitorOperation* blocker) {
  SignalGateContinue(&blocker->gate);
}

static void BlockingMonitorOperationDeinitialize(
    BlockingMonitorOperation* blocker) {
  SignalGateDeinitialize(&blocker->gate);
}

static iree_hal_semaphore_t* observed_proxy_semaphore = nullptr;
static std::atomic<uint64_t> observed_proxy_value{UINT64_MAX};
static std::atomic<int32_t> observed_proxy_status{IREE_STATUS_UNKNOWN};
static std::atomic<bool> observed_proxy_resolved{false};
static uint64_t observed_destroy_signal_handle = 0;

static hsa_status_t HSA_API
ObserveProxyBeforeSignalDestroy(hsa_signal_t signal) {
  if (signal.handle == observed_destroy_signal_handle) {
    uint64_t value = UINT64_MAX;
    iree_status_t status =
        iree_hal_semaphore_query(observed_proxy_semaphore, &value);
    observed_proxy_status.store(iree_status_code(status),
                                std::memory_order_relaxed);
    if (iree_status_is_ok(status)) {
      observed_proxy_value.store(value, std::memory_order_relaxed);
      observed_proxy_resolved.store(value >= 1, std::memory_order_release);
    } else {
      observed_proxy_resolved.store(true, std::memory_order_release);
    }
    iree_status_free(status);
  }
  return forward_signal_destroy(signal);
}

TEST_F(IpcEventGpuTest, FreshCarrierIsPending) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));

  bool reached = true;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_FALSE(reached);

  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, SourceReopensForeignReexportAsCanonicalCarrier) {
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  auto original_attach = libhsa->hsa_amd_ipc_signal_attach;
  forward_ipc_signal_attach = original_attach;
  ipc_signal_attach_call_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_amd_ipc_signal_attach = &CountIpcSignalAttach;

  iree_hal_amdgpu_ipc_event_t* source = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &source));
  iree_hal_amdgpu_ipc_event_token_t source_token = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_export(source, &source_token));
  iree_hal_amdgpu_ipc_event_t* foreign_import = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_import(
      device_.base_device(), source_token, &foreign_import));

  // A wire adapter may replace the exporting PID when another process
  // re-exports an imported event, but the lower-layer ROCr token is unchanged.
  // Reopening that foreign-PID handle in the source process must therefore
  // recover the original carrier and its one generation lock.
  iree_hal_amdgpu_ipc_event_token_t reexported_token = {};
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_export(foreign_import, &reexported_token));
  iree_hal_amdgpu_ipc_event_t* source_reopened = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_import(
      device_.base_device(), reexported_token, &source_reopened));

  EXPECT_EQ(source, foreign_import);
  EXPECT_EQ(source, source_reopened);
  EXPECT_EQ(0, ipc_signal_attach_call_count.load(std::memory_order_relaxed));

  libhsa->hsa_amd_ipc_signal_attach = original_attach;
  forward_ipc_signal_attach = nullptr;
  iree_hal_amdgpu_ipc_event_release(source_reopened);
  iree_hal_amdgpu_ipc_event_release(foreign_import);
  iree_hal_amdgpu_ipc_event_release(source);
}

TEST_F(IpcEventGpuTest, DuplicateForeignImportsShareOneAttachment) {
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  hsa_signal_t owner_signal = {0};
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, HSA_AMD_SIGNAL_IPC, &owner_signal));
  hsa_amd_ipc_signal_t hsa_token = {};
  IREE_ASSERT_OK(iree_hsa_amd_ipc_signal_create(IREE_LIBHSA(libhsa),
                                                owner_signal, &hsa_token));
  iree_hal_amdgpu_ipc_event_token_t token = {};
  memcpy(token.data, &hsa_token, sizeof(hsa_token));

  auto original_attach = libhsa->hsa_amd_ipc_signal_attach;
  forward_ipc_signal_attach = original_attach;
  ipc_signal_attach_call_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_amd_ipc_signal_attach = &CountIpcSignalAttach;

  iree_hal_amdgpu_ipc_event_t* first = nullptr;
  iree_hal_amdgpu_ipc_event_t* second = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_import(device_.base_device(), token, &first));
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_import(device_.base_device(), token, &second));
  EXPECT_EQ(first, second);
  EXPECT_EQ(1, ipc_signal_attach_call_count.load(std::memory_order_relaxed));

  iree_hal_amdgpu_ipc_event_release(second);
  iree_hal_amdgpu_ipc_event_release(first);
  libhsa->hsa_amd_ipc_signal_attach = original_attach;
  forward_ipc_signal_attach = nullptr;
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_signal_destroy_raw(libhsa, owner_signal));
}

TEST_F(IpcEventGpuTest, CandidateDiscardDecrementExecutesOutsideAssertions) {
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  hsa_signal_t owner_signal = {0};
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, HSA_AMD_SIGNAL_IPC, &owner_signal));
  hsa_amd_ipc_signal_t hsa_token = {};
  IREE_ASSERT_OK(iree_hsa_amd_ipc_signal_create(IREE_LIBHSA(libhsa),
                                                owner_signal, &hsa_token));
  iree_hal_amdgpu_ipc_event_token_t token = {};
  memcpy(token.data, &hsa_token, sizeof(hsa_token));

  FirstImportCandidateGate candidate_gate;
  SignalGateInitialize(/*target_signal_handle=*/0, &candidate_gate.gate);
  candidate_gate.invocation_count.store(0, std::memory_order_relaxed);
  RefDecrementObserver observer;
  InitializeRefDecrementObserver(&observer);
  iree_hal_amdgpu_ipc_event_set_import_candidate_hook_for_testing(
      GateFirstImportCandidate, &candidate_gate);
  iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(
      ObserveRefDecrement, &observer);

  iree_hal_amdgpu_ipc_event_t* first = nullptr;
  iree_status_code_t first_status_code = IREE_STATUS_UNKNOWN;
  std::thread first_import_thread([&] {
    first_status_code = iree_status_consume_code(
        iree_hal_amdgpu_ipc_event_import(device_.base_device(), token, &first));
  });
  SignalGateAwaitEntered(&candidate_gate.gate);

  iree_hal_amdgpu_ipc_event_t* second = nullptr;
  iree_status_t second_status =
      iree_hal_amdgpu_ipc_event_import(device_.base_device(), token, &second);
  SignalGateContinue(&candidate_gate.gate);
  first_import_thread.join();
  iree_hal_amdgpu_ipc_event_set_import_candidate_hook_for_testing(nullptr,
                                                                  nullptr);
  iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(nullptr,
                                                               nullptr);

  EXPECT_EQ(IREE_STATUS_OK, first_status_code);
  IREE_EXPECT_OK(second_status);
  EXPECT_EQ(first, second);
  EXPECT_EQ(
      1,
      observer
          .count[IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_CANDIDATE_DISCARD]
          .load(std::memory_order_acquire));
  EXPECT_EQ(
      1,
      observer
          .old_ref_count
              [IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_CANDIDATE_DISCARD]
          .load(std::memory_order_relaxed));

  iree_hal_amdgpu_ipc_event_release(second);
  iree_hal_amdgpu_ipc_event_release(first);
  SignalGateDeinitialize(&candidate_gate.gate);
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_signal_destroy_raw(libhsa, owner_signal));
}

TEST_F(IpcEventGpuTest, AttachFailureDecrementExecutesOutsideAssertions) {
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  hsa_signal_t owner_signal = {0};
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, HSA_AMD_SIGNAL_IPC, &owner_signal));
  hsa_amd_ipc_signal_t hsa_token = {};
  IREE_ASSERT_OK(iree_hsa_amd_ipc_signal_create(IREE_LIBHSA(libhsa),
                                                owner_signal, &hsa_token));
  iree_hal_amdgpu_ipc_event_token_t token = {};
  memcpy(token.data, &hsa_token, sizeof(hsa_token));

  RefDecrementObserver observer;
  InitializeRefDecrementObserver(&observer);
  iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(
      ObserveRefDecrement, &observer);
  auto original_attach = libhsa->hsa_amd_ipc_signal_attach;
  libhsa->hsa_amd_ipc_signal_attach = &RejectIpcSignalAttach;
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  iree_status_t status =
      iree_hal_amdgpu_ipc_event_import(device_.base_device(), token, &event);
  libhsa->hsa_amd_ipc_signal_attach = original_attach;
  iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(nullptr,
                                                               nullptr);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(nullptr, event);
  EXPECT_EQ(
      1, observer
             .count[IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_ATTACH_FAILURE]
             .load(std::memory_order_acquire));
  EXPECT_EQ(
      1, observer
             .old_ref_count
                 [IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_ATTACH_FAILURE]
             .load(std::memory_order_relaxed));

  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_signal_destroy_raw(libhsa, owner_signal));
}

TEST_F(IpcEventGpuTest, RegistryAnchorDecrementExecutesOutsideAssertions) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  RefDecrementObserver observer;
  InitializeRefDecrementObserver(&observer);
  iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(
      ObserveRefDecrement, &observer);
  iree_hal_amdgpu_ipc_event_release(event);
  iree_hal_amdgpu_ipc_event_set_ref_decrement_hook_for_testing(nullptr,
                                                               nullptr);

  EXPECT_EQ(
      1,
      observer
          .count[IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_REGISTRY_ANCHOR]
          .load(std::memory_order_acquire));
  EXPECT_EQ(
      1, observer
             .old_ref_count
                 [IREE_HAL_AMDGPU_IPC_EVENT_REF_DECREMENT_KIND_REGISTRY_ANCHOR]
             .load(std::memory_order_relaxed));
}

TEST_F(IpcEventGpuTest, ConcurrentFinalCallerReleasesDoNotAccessFreedCarrier) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_ipc_event_token_t token = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_export(event, &token));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));

  // Suspend the only caller before it acquires the registry mutex. Its caller
  // reference must keep the carrier alive while a concurrent import retains
  // and releases the same canonical carrier. Once resumed, the original
  // release owns the serialized 2 -> 1 transition and destroys exactly once.
  SignalGate release_gate;
  SignalGateInitialize(/*target_signal_handle=*/0, &release_gate);
  iree_hal_amdgpu_ipc_event_set_before_anchored_release_lock_hook_for_testing(
      GateRecordCallback, &release_gate);
  auto original_signal_destroy = libhsa->hsa_signal_destroy;
  forward_signal_destroy = original_signal_destroy;
  counted_destroy_signal_handle.store(event_signal_handle,
                                      std::memory_order_release);
  counted_signal_destroy_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_destroy = &CountTargetSignalDestroy;

  std::thread final_release_thread(
      [&] { iree_hal_amdgpu_ipc_event_release(event); });
  SignalGateAwaitEntered(&release_gate);
  iree_hal_amdgpu_ipc_event_t* imported_event = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_import(device_.base_device(), token,
                                                  &imported_event));
  EXPECT_EQ(event, imported_event);
  iree_hal_amdgpu_ipc_event_release(imported_event);
  SignalGateContinue(&release_gate);
  final_release_thread.join();

  iree_hal_amdgpu_ipc_event_set_before_anchored_release_lock_hook_for_testing(
      nullptr, nullptr);
  libhsa->hsa_signal_destroy = original_signal_destroy;
  forward_signal_destroy = nullptr;
  EXPECT_EQ(1, counted_signal_destroy_count.load(std::memory_order_acquire));
  counted_destroy_signal_handle.store(0, std::memory_order_relaxed);
  SignalGateDeinitialize(&release_gate);

  // Exercise the same final-caller transition under a wider release fan-in.
  constexpr size_t kIterationCount = 16;
  constexpr size_t kReleaseThreadCount = 8;
  for (size_t iteration = 0; iteration < kIterationCount; ++iteration) {
    iree_hal_amdgpu_ipc_event_t* stress_event = nullptr;
    IREE_ASSERT_OK(
        iree_hal_amdgpu_ipc_event_create(device_.base_device(), &stress_event));
    for (size_t i = 1; i < kReleaseThreadCount; ++i) {
      iree_hal_amdgpu_ipc_event_retain(stress_event);
    }
    iree_notification_t start_notification;
    iree_notification_initialize(&start_notification);
    iree_atomic_int32_t start;
    iree_atomic_store(&start, 0, iree_memory_order_relaxed);
    std::array<std::thread, kReleaseThreadCount> release_threads;
    for (std::thread& release_thread : release_threads) {
      release_thread = std::thread([&] {
        iree_notification_await(&start_notification, AtomicFlagIsSet, &start,
                                iree_infinite_timeout());
        iree_hal_amdgpu_ipc_event_release(stress_event);
      });
    }
    iree_atomic_store(&start, 1, iree_memory_order_release);
    iree_notification_post(&start_notification, IREE_ALL_WAITERS);
    for (std::thread& release_thread : release_threads) {
      release_thread.join();
    }
    iree_notification_deinitialize(&start_notification);
  }
}

TEST_F(IpcEventGpuTest, FirstRecordDoesNotCompleteInitialPendingGeneration) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {};
  iree_hal_amdgpu_ipc_event_wait_t* wait = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_reserve(device_.base_device(),
                                                        &wait_point, &wait));
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_wait_arm(event, wait));

  SignalGate load_gate;
  SignalGateInitialize(event_signal_handle, &load_gate);
  auto original_signal_load = libhsa->hsa_signal_load_scacquire;
  forward_signal_load_scacquire = original_signal_load;
  active_signal_load_gate = &load_gate;
  libhsa->hsa_signal_load_scacquire = &GateFirstSignalLoad;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_commit(wait));
  SignalGateAwaitEntered(&load_gate);

  iree_hal_semaphore_t* recorded_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &recorded_semaphore));
  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, recorded_semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);

  SignalGate second_load_gate;
  SignalGateInitialize(event_signal_handle, &second_load_gate);
  active_signal_load_gate = &second_load_gate;
  SignalGateContinue(&load_gate);
  SignalGateAwaitEntered(&second_load_gate);

  uint64_t proxy_value = UINT64_MAX;
  IREE_EXPECT_OK(iree_hal_semaphore_query(wait_point.semaphore, &proxy_value));
  EXPECT_EQ(0u, proxy_value)
      << "the first record was mistaken for completion of the initial state";

  IREE_ASSERT_OK(iree_hal_semaphore_signal(recorded_semaphore, 1,
                                           /*frontier=*/nullptr));
  SignalGateContinue(&second_load_gate);
  IREE_EXPECT_OK(iree_hal_semaphore_wait(wait_point.semaphore, wait_point.value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  active_signal_load_gate = nullptr;
  libhsa->hsa_signal_load_scacquire = original_signal_load;
  forward_signal_load_scacquire = nullptr;
  SignalGateDeinitialize(&second_load_gate);
  SignalGateDeinitialize(&load_gate);
  iree_hal_semaphore_release(recorded_semaphore);
  iree_hal_semaphore_release(wait_point.semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, LocalFastRerecordCannotHidePriorCompletion) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();

  iree_hal_semaphore_t* first_semaphore = nullptr;
  iree_hal_semaphore_t* second_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &first_semaphore));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &second_semaphore));

  auto original_signal_store = libhsa->hsa_signal_store_screlease;
  forward_signal_store_screlease = original_signal_store;
  stored_signal_handle.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_store_screlease = &GateFirstCompletionStore;

  iree_hal_amdgpu_ipc_event_record_t* first_record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, first_semaphore, /*recorded_value=*/1, &first_record));
  iree_hal_amdgpu_ipc_event_record_commit(first_record);
  const uint64_t event_signal_handle =
      stored_signal_handle.load(std::memory_order_acquire);
  ASSERT_NE(0u, event_signal_handle);

  SignalGate store_gate;
  SignalGateInitialize(event_signal_handle, &store_gate);
  active_signal_store_gate = &store_gate;

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {};
  iree_hal_amdgpu_ipc_event_wait_t* wait = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_reserve(device_.base_device(),
                                                        &wait_point, &wait));
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_wait_arm(event, wait));
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_commit(wait));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(first_semaphore, 1,
                                           /*frontier=*/nullptr));
  SignalGateAwaitEntered(&store_gate);

  // The monitor has release-stored the prior completion but is paused before
  // it can poll the wait. Rearm locally while signal 0 is visible.
  iree_hal_amdgpu_ipc_event_record_t* second_record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, second_semaphore, /*recorded_value=*/1, &second_record));
  iree_hal_amdgpu_ipc_event_record_commit(second_record);

  uint64_t proxy_value = UINT64_MAX;
  IREE_EXPECT_OK(iree_hal_semaphore_query(wait_point.semaphore, &proxy_value));
  EXPECT_EQ(0u, proxy_value);
  SignalGateContinue(&store_gate);
  IREE_EXPECT_OK(iree_hal_semaphore_wait(wait_point.semaphore, wait_point.value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(second_semaphore, 1,
                                           /*frontier=*/nullptr));
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_wait(event));
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  active_signal_store_gate = nullptr;
  libhsa->hsa_signal_store_screlease = original_signal_store;
  forward_signal_store_screlease = nullptr;
  SignalGateDeinitialize(&store_gate);
  iree_hal_semaphore_release(wait_point.semaphore);
  iree_hal_semaphore_release(second_semaphore);
  iree_hal_semaphore_release(first_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, EnqueueAbortPreservesCompletedGeneration) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));

  iree_hal_semaphore_t* completed_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &completed_semaphore));
  iree_hal_amdgpu_ipc_event_record_t* completed_record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, completed_semaphore, /*recorded_value=*/1, &completed_record));
  iree_hal_amdgpu_ipc_event_record_commit(completed_record);
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait(event));

  iree_hal_semaphore_t* rejected_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &rejected_semaphore));
  iree_hal_amdgpu_ipc_event_record_t* rejected_record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, rejected_semaphore, /*recorded_value=*/1, &rejected_record));

  bool reached = false;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached)
      << "record preparation exposed a speculative pending generation";
  iree_hal_amdgpu_ipc_event_record_abort(rejected_record);
  // Cancellation must exclude the pending callback. Signalling its source
  // after abort cannot publish a speculative generation.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(rejected_semaphore, 1,
                                           /*frontier=*/nullptr));
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached);

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {};
  iree_hal_amdgpu_ipc_event_wait_t* wait = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_reserve(device_.base_device(),
                                                        &wait_point, &wait));
  EXPECT_FALSE(iree_hal_amdgpu_ipc_event_wait_arm(event, wait));
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_wait_commit(wait));

  iree_hal_semaphore_release(wait_point.semaphore);
  iree_hal_semaphore_release(rejected_semaphore);
  iree_hal_semaphore_release(completed_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, SynchronousRecordCallbackIsSafeToAbort) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_semaphore_t* completed_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &completed_semaphore));

  // The timepoint callback runs synchronously during prepare, before the queue
  // transaction decides whether to commit. Abort must consume that readiness
  // without exposing completion through the carrier.
  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, completed_semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_abort(record);

  bool reached = true;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_FALSE(reached);
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  iree_hal_semaphore_release(completed_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, RecordAbortJoinsDispatchedCallback) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);
  iree_hal_semaphore_t* pending_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &pending_semaphore));

  SignalGate callback_gate;
  SignalGateInitialize(/*target_signal_handle=*/0, &callback_gate);
  SignalGate cancel_miss_gate;
  SignalGateInitialize(/*target_signal_handle=*/0, &cancel_miss_gate);
  iree_hal_amdgpu_ipc_event_set_record_callback_hook_for_testing(
      GateRecordCallback, &callback_gate);
  iree_hal_amdgpu_ipc_event_set_record_cancel_miss_hook_for_testing(
      NotifySignalGate, &cancel_miss_gate);
  auto original_signal_store = libhsa->hsa_signal_store_screlease;
  forward_signal_store_screlease = original_signal_store;
  counted_completion_signal_handle.store(event_signal_handle,
                                         std::memory_order_release);
  counted_completion_store_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_store_screlease = &GateFirstCompletionStore;
  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, pending_semaphore, /*recorded_value=*/1, &record));

  iree_status_code_t signal_status_code = IREE_STATUS_UNKNOWN;
  std::thread signal_thread([&] {
    signal_status_code = iree_status_consume_code(
        iree_hal_semaphore_signal(pending_semaphore, 1, /*frontier=*/nullptr));
  });
  SignalGateAwaitEntered(&callback_gate);

  // Entering the callback proves the timepoint was detached, so abort must
  // take the cancel-false join path before it can free embedded storage.
  std::atomic<bool> abort_returned{false};
  bool abort_was_pending = false;
  int completion_store_count_during_abort = -1;
  bool reached_during_abort = true;
  iree_status_code_t query_status_code_during_abort = IREE_STATUS_UNKNOWN;
  std::thread controller_thread([&] {
    SignalGateAwaitEntered(&cancel_miss_gate);
    abort_was_pending = !abort_returned.load(std::memory_order_acquire);
    completion_store_count_during_abort =
        counted_completion_store_count.load(std::memory_order_acquire);
    query_status_code_during_abort = iree_status_consume_code(
        iree_hal_amdgpu_ipc_event_query(event, &reached_during_abort));
    SignalGateContinue(&callback_gate);
  });

  iree_hal_amdgpu_ipc_event_record_abort(record);
  abort_returned.store(true, std::memory_order_release);
  controller_thread.join();
  signal_thread.join();
  EXPECT_TRUE(abort_was_pending);
  EXPECT_EQ(0, completion_store_count_during_abort);
  EXPECT_EQ(IREE_STATUS_OK, query_status_code_during_abort);
  EXPECT_FALSE(reached_during_abort);
  EXPECT_TRUE(abort_returned.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_OK, signal_status_code);
  iree_hal_amdgpu_ipc_event_set_record_callback_hook_for_testing(nullptr,
                                                                 nullptr);
  iree_hal_amdgpu_ipc_event_set_record_cancel_miss_hook_for_testing(nullptr,
                                                                    nullptr);
  libhsa->hsa_signal_store_screlease = original_signal_store;
  forward_signal_store_screlease = nullptr;
  counted_completion_signal_handle.store(0, std::memory_order_relaxed);

  bool reached = true;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_FALSE(reached);
  EXPECT_EQ(0, counted_completion_store_count.load(std::memory_order_acquire));
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  SignalGateDeinitialize(&cancel_miss_gate);
  SignalGateDeinitialize(&callback_gate);
  iree_hal_semaphore_release(pending_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, ProducerFailureResolvesTransport) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_semaphore_t* failed_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &failed_semaphore));
  iree_hal_semaphore_fail(
      failed_semaphore,
      iree_make_status(IREE_STATUS_ABORTED, "injected producer failure"));

  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, failed_semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_wait(event));

  bool reached = false;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached);

  iree_hal_semaphore_release(failed_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, ShutdownDoesNotPublishPendingProducerCompletion) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_ipc_event_token_t token = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_export(event, &token));
  iree_hal_semaphore_t* pending_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &pending_semaphore));

  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, pending_semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  // Accepted producer work has not established its release sequence. Shutdown
  // abandons this transport generation pending instead of advertising output
  // visibility that does not yet exist. The carrier remembers that no monitor
  // operation remains capable of completing this local generation.
  bool reached = true;
  iree_status_t status = iree_hal_amdgpu_ipc_event_query(event, &reached);
  EXPECT_EQ(IREE_STATUS_ABORTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_FALSE(reached);

  // Canonical registry reuse preserves the same failed generation across a
  // monitor restart. A later record preparation must fail promptly rather than
  // waiting for the cancelled prior timepoint, and other local uses report the
  // same permanent state.
  iree_hal_amdgpu_ipc_event_t* reopened_event = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_import(device_.base_device(), token,
                                                  &reopened_event));
  EXPECT_EQ(event, reopened_event);
  iree_hal_semaphore_t* replacement_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &replacement_semaphore));
  iree_hal_amdgpu_ipc_event_record_t* replacement_record = nullptr;
  status = iree_hal_amdgpu_ipc_event_record_prepare(
      reopened_event, replacement_semaphore, /*recorded_value=*/1,
      &replacement_record);
  EXPECT_EQ(IREE_STATUS_ABORTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(nullptr, replacement_record);

  status = iree_hal_amdgpu_ipc_event_wait(reopened_event);
  EXPECT_EQ(IREE_STATUS_ABORTED, iree_status_code(status));
  iree_status_free(status);
  iree_hal_amdgpu_ipc_event_token_t rejected_token = {};
  status = iree_hal_amdgpu_ipc_event_export(reopened_event, &rejected_token);
  EXPECT_EQ(IREE_STATUS_ABORTED, iree_status_code(status));
  iree_status_free(status);

  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  iree_hal_semaphore_release(replacement_semaphore);
  iree_hal_amdgpu_ipc_event_release(reopened_event);
  iree_hal_semaphore_release(pending_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest,
       ShutdownReconcilesReachedSemaphoreAfterSuccessfulCancellation) {
  BlockingMonitorOperation blocker;
  BlockingMonitorOperationStart(&blocker);

  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  auto original_signal_store = libhsa->hsa_signal_store_screlease;
  forward_signal_store_screlease = original_signal_store;
  counted_completion_signal_handle.store(event_signal_handle,
                                         std::memory_order_release);
  counted_completion_store_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_store_screlease = &GateFirstCompletionStore;

  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);
  PublishSemaphoreTimelineWithoutDispatch(semaphore, 1);

  std::thread shutdown_thread(
      [] { iree_hal_amdgpu_ipc_event_monitor_shutdown(); });
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_monitor_await_shutdown(
      iree_infinite_timeout()));
  BlockingMonitorOperationContinue(&blocker);
  shutdown_thread.join();

  bool reached = false;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached);
  EXPECT_EQ(1, counted_completion_store_count.load(std::memory_order_acquire));

  libhsa->hsa_signal_store_screlease = original_signal_store;
  forward_signal_store_screlease = nullptr;
  counted_completion_signal_handle.store(0, std::memory_order_relaxed);
  BlockingMonitorOperationDeinitialize(&blocker);
  iree_hal_semaphore_release(semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest,
       ShutdownReconcilesFailedSemaphoreAfterSuccessfulCancellation) {
  BlockingMonitorOperation blocker;
  BlockingMonitorOperationStart(&blocker);

  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  auto original_signal_store = libhsa->hsa_signal_store_screlease;
  forward_signal_store_screlease = original_signal_store;
  counted_completion_signal_handle.store(event_signal_handle,
                                         std::memory_order_release);
  counted_completion_store_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_store_screlease = &GateFirstCompletionStore;

  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);
  PublishSemaphoreFailureWithoutDispatch(semaphore);

  std::thread shutdown_thread(
      [] { iree_hal_amdgpu_ipc_event_monitor_shutdown(); });
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_monitor_await_shutdown(
      iree_infinite_timeout()));
  BlockingMonitorOperationContinue(&blocker);
  shutdown_thread.join();

  bool reached = false;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached);
  EXPECT_EQ(1, counted_completion_store_count.load(std::memory_order_acquire));

  libhsa->hsa_signal_store_screlease = original_signal_store;
  forward_signal_store_screlease = nullptr;
  counted_completion_signal_handle.store(0, std::memory_order_relaxed);
  BlockingMonitorOperationDeinitialize(&blocker);
  iree_hal_semaphore_release(semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, ShutdownJoinsDispatchedRecordCallback) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  SignalGate callback_gate;
  SignalGateInitialize(/*target_signal_handle=*/0, &callback_gate);
  SignalGate cancel_miss_gate;
  SignalGateInitialize(/*target_signal_handle=*/0, &cancel_miss_gate);
  iree_hal_amdgpu_ipc_event_set_record_callback_hook_for_testing(
      GateRecordCallback, &callback_gate);
  iree_hal_amdgpu_ipc_event_set_record_cancel_miss_hook_for_testing(
      NotifySignalGate, &cancel_miss_gate);
  auto original_signal_store = libhsa->hsa_signal_store_screlease;
  forward_signal_store_screlease = original_signal_store;
  counted_completion_signal_handle.store(event_signal_handle,
                                         std::memory_order_release);
  counted_completion_store_count.store(0, std::memory_order_relaxed);
  libhsa->hsa_signal_store_screlease = &GateFirstCompletionStore;

  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);

  iree_status_code_t signal_status_code = IREE_STATUS_UNKNOWN;
  std::thread signal_thread([&] {
    signal_status_code = iree_status_consume_code(
        iree_hal_semaphore_signal(semaphore, 1, /*frontier=*/nullptr));
  });
  SignalGateAwaitEntered(&callback_gate);

  std::atomic<bool> shutdown_returned{false};
  std::thread shutdown_thread([&] {
    iree_hal_amdgpu_ipc_event_monitor_shutdown();
    shutdown_returned.store(true, std::memory_order_release);
  });
  SignalGateAwaitEntered(&cancel_miss_gate);
  EXPECT_FALSE(shutdown_returned.load(std::memory_order_acquire));
  EXPECT_EQ(0, counted_completion_store_count.load(std::memory_order_acquire));
  bool reached = true;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_FALSE(reached);

  SignalGateContinue(&callback_gate);
  signal_thread.join();
  shutdown_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, signal_status_code);
  EXPECT_TRUE(shutdown_returned.load(std::memory_order_acquire));
  EXPECT_EQ(1, counted_completion_store_count.load(std::memory_order_acquire));
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached);

  iree_hal_amdgpu_ipc_event_set_record_callback_hook_for_testing(nullptr,
                                                                 nullptr);
  iree_hal_amdgpu_ipc_event_set_record_cancel_miss_hook_for_testing(nullptr,
                                                                    nullptr);
  libhsa->hsa_signal_store_screlease = original_signal_store;
  forward_signal_store_screlease = nullptr;
  counted_completion_signal_handle.store(0, std::memory_order_relaxed);
  SignalGateDeinitialize(&cancel_miss_gate);
  SignalGateDeinitialize(&callback_gate);
  iree_hal_semaphore_release(semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, ShutdownPublishesCompletionObservedOnFirstRecordPoll) {
  BlockingMonitorOperation blocker;
  BlockingMonitorOperationStart(&blocker);

  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_semaphore_t* completed_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &completed_semaphore));

  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, completed_semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);

  std::thread shutdown_thread(
      [] { iree_hal_amdgpu_ipc_event_monitor_shutdown(); });
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_monitor_await_shutdown(
      iree_infinite_timeout()));
  BlockingMonitorOperationContinue(&blocker);
  shutdown_thread.join();

  bool reached = false;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_TRUE(reached)
      << "shutdown overrode producer completion visible to the first poll";

  BlockingMonitorOperationDeinitialize(&blocker);
  iree_hal_semaphore_release(completed_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, ShutdownPreservesCompletionObservedOnFirstWaitPoll) {
  BlockingMonitorOperation blocker;
  BlockingMonitorOperationStart(&blocker);

  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {};
  iree_hal_amdgpu_ipc_event_wait_t* wait = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_reserve(device_.base_device(),
                                                        &wait_point, &wait));
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_wait_arm(event, wait));
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_commit(wait));

  hsa_signal_t event_signal = {};
  event_signal.handle = event_signal_handle;
  iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa), event_signal, 0);

  std::thread shutdown_thread(
      [] { iree_hal_amdgpu_ipc_event_monitor_shutdown(); });
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_monitor_await_shutdown(
      iree_infinite_timeout()));
  BlockingMonitorOperationContinue(&blocker);
  shutdown_thread.join();

  uint64_t proxy_value = 0;
  iree_status_t status =
      iree_hal_semaphore_query(wait_point.semaphore, &proxy_value);
  IREE_EXPECT_OK(status);
  EXPECT_GE(proxy_value, wait_point.value)
      << "shutdown cancelled a transport completion visible to the first poll";

  BlockingMonitorOperationDeinitialize(&blocker);
  iree_hal_semaphore_release(wait_point.semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, ShutdownCancelsRerecordBlockedOnPendingGeneration) {
  // Keep the monitor from abandoning the first record until the blocked
  // preparation has observed admission closure and cancelled itself.
  BlockingMonitorOperation blocker;
  BlockingMonitorOperationStart(&blocker);

  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();

  iree_hal_semaphore_t* pending_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &pending_semaphore));
  iree_hal_amdgpu_ipc_event_record_t* first_record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, pending_semaphore, /*recorded_value=*/1, &first_record));
  iree_hal_amdgpu_ipc_event_record_commit(first_record);

  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);
  SignalGate load_gate;
  SignalGateInitialize(event_signal_handle, &load_gate);
  auto original_signal_load = libhsa->hsa_signal_load_scacquire;
  forward_signal_load_scacquire = original_signal_load;
  active_signal_load_gate = &load_gate;
  libhsa->hsa_signal_load_scacquire = &GateFirstSignalLoad;

  iree_hal_amdgpu_ipc_event_record_t* second_record = nullptr;
  iree_status_code_t prepare_status_code = IREE_STATUS_UNKNOWN;
  std::thread prepare_thread([&] {
    prepare_status_code =
        iree_status_consume_code(iree_hal_amdgpu_ipc_event_record_prepare(
            event, pending_semaphore, /*recorded_value=*/2, &second_record));
  });
  SignalGateAwaitEntered(&load_gate);

  std::thread shutdown_thread(
      [] { iree_hal_amdgpu_ipc_event_monitor_shutdown(); });
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_monitor_await_shutdown(
      iree_infinite_timeout()));
  SignalGateContinue(&load_gate);
  prepare_thread.join();

  EXPECT_EQ(IREE_STATUS_CANCELLED, prepare_status_code);
  EXPECT_EQ(nullptr, second_record);

  BlockingMonitorOperationContinue(&blocker);
  shutdown_thread.join();
  bool reached = true;
  iree_status_t query_status = iree_hal_amdgpu_ipc_event_query(event, &reached);
  EXPECT_EQ(IREE_STATUS_ABORTED, iree_status_code(query_status));
  iree_status_free(query_status);
  EXPECT_FALSE(reached)
      << "cancelling a blocked rerecord manufactured transport completion";

  BlockingMonitorOperationDeinitialize(&blocker);
  active_signal_load_gate = nullptr;
  libhsa->hsa_signal_load_scacquire = original_signal_load;
  forward_signal_load_scacquire = nullptr;
  SignalGateDeinitialize(&load_gate);
  iree_hal_semaphore_release(pending_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest, HostWaitDoesNotBlockResolvingRecord) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));

  // Hold the monitor before publishing the wait so registration and host
  // blocking can be observed without racing a poll that resolves the proxy.
  BlockingMonitorOperation blocker;
  BlockingMonitorOperationStart(&blocker);

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {};
  iree_hal_amdgpu_ipc_event_wait_t* wait = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_reserve(device_.base_device(),
                                                        &wait_point, &wait));
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_wait_arm(event, wait));
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_commit(wait));

  iree_status_code_t wait_status_code = IREE_STATUS_UNKNOWN;
  std::atomic<bool> wait_returned{false};
  std::thread wait_thread([&] {
    wait_status_code = iree_status_consume_code(iree_hal_semaphore_wait(
        wait_point.semaphore, wait_point.value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
    wait_returned.store(true, std::memory_order_release);
  });

  // The proxy has no producer metadata, so its host wait uses an async
  // timepoint. Seeing that timepoint on the pending list proves the thread is
  // blocked inside the wait rather than merely scheduled to enter it.
  auto* async_semaphore =
      reinterpret_cast<iree_async_semaphore_t*>(wait_point.semaphore);
  for (;;) {
    iree_slim_mutex_lock(&async_semaphore->mutex);
    const bool is_wait_registered = async_semaphore->timepoints_head != nullptr;
    iree_slim_mutex_unlock(&async_semaphore->mutex);
    if (is_wait_registered) break;
    std::this_thread::yield();
  }
  EXPECT_FALSE(wait_returned.load(std::memory_order_acquire));

  // The committed wait released the generation mutex before the host thread
  // blocked on its destination proxy. A record must remain able to commit and
  // resolve that proxy without depending on the blocked host thread.
  iree_hal_semaphore_t* recorded_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &recorded_semaphore));
  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, recorded_semaphore, /*recorded_value=*/1, &record));
  iree_hal_amdgpu_ipc_event_record_commit(record);
  EXPECT_FALSE(wait_returned.load(std::memory_order_acquire));

  BlockingMonitorOperationContinue(&blocker);
  wait_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, wait_status_code);
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  BlockingMonitorOperationDeinitialize(&blocker);
  iree_hal_semaphore_release(wait_point.semaphore);
  iree_hal_semaphore_release(recorded_semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

TEST_F(IpcEventGpuTest,
       ShutdownFailsPendingProxyBeforeCarrierAndDestinationRelease) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);

  iree_hal_amdgpu_ipc_event_wait_point_t wait_point = {};
  iree_hal_amdgpu_ipc_event_wait_t* wait = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_reserve(device_.base_device(),
                                                        &wait_point, &wait));
  ASSERT_TRUE(iree_hal_amdgpu_ipc_event_wait_arm(event, wait));

  auto original_signal_destroy = libhsa->hsa_signal_destroy;
  forward_signal_destroy = original_signal_destroy;
  observed_proxy_semaphore = wait_point.semaphore;
  observed_destroy_signal_handle = event_signal_handle;
  observed_proxy_value.store(UINT64_MAX, std::memory_order_relaxed);
  observed_proxy_status.store(IREE_STATUS_UNKNOWN, std::memory_order_relaxed);
  observed_proxy_resolved.store(false, std::memory_order_relaxed);
  libhsa->hsa_signal_destroy = &ObserveProxyBeforeSignalDestroy;

  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_wait_commit(wait));
  iree_hal_amdgpu_ipc_event_release(event);
  event = nullptr;
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  libhsa->hsa_signal_destroy = original_signal_destroy;
  forward_signal_destroy = nullptr;
  observed_proxy_semaphore = nullptr;
  observed_destroy_signal_handle = 0;

  EXPECT_TRUE(observed_proxy_resolved.load(std::memory_order_acquire));
  EXPECT_EQ(IREE_STATUS_CANCELLED,
            observed_proxy_status.load(std::memory_order_relaxed));
  uint64_t proxy_value = UINT64_MAX;
  iree_status_t status =
      iree_hal_semaphore_query(wait_point.semaphore, &proxy_value);
  EXPECT_EQ(IREE_STATUS_CANCELLED, iree_status_code(status));
  iree_status_free(status);
  iree_hal_semaphore_release(wait_point.semaphore);
}

TEST_F(IpcEventGpuTest, RecordCompletionNeverBlocksCarrierReleaseCaller) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_amdgpu_libhsa_t* libhsa = device_.device_libhsa();
  uint64_t event_signal_handle = 0;
  IREE_ASSERT_OK(CaptureEventSignalHandle(event, libhsa, &event_signal_handle));
  ASSERT_NE(0u, event_signal_handle);

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_record_prepare(
      event, semaphore, /*recorded_value=*/1, &record));

  SignalGate destroy_gate;
  SignalGateInitialize(event_signal_handle, &destroy_gate);
  auto original_signal_destroy = libhsa->hsa_signal_destroy;
  forward_signal_destroy = original_signal_destroy;
  active_signal_destroy_gate = &destroy_gate;
  libhsa->hsa_signal_destroy = &GateSignalDestroy;

  const std::thread::id release_thread = std::this_thread::get_id();
  iree_hal_amdgpu_ipc_event_release(event);
  event = nullptr;
  iree_hal_amdgpu_ipc_event_record_commit(record);
  SignalGateAwaitEntered(&destroy_gate);

  // Commit returned while final carrier destruction was gated on the monitor,
  // so neither semaphore completion nor carrier release ran teardown inline.
  EXPECT_NE(release_thread, destroy_gate.worker_thread);
  SignalGateContinue(&destroy_gate);
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  active_signal_destroy_gate = nullptr;
  libhsa->hsa_signal_destroy = original_signal_destroy;
  forward_signal_destroy = nullptr;
  SignalGateDeinitialize(&destroy_gate);
  iree_hal_semaphore_release(semaphore);
}

TEST_F(IpcEventGpuTest, WorkerCreationFailureLeavesCarrierPending) {
  iree_hal_amdgpu_ipc_event_t* event = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_create(device_.base_device(), &event));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_.base_device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  iree_hal_amdgpu_ipc_event_monitor_set_fail_thread_create_for_testing(true);
  iree_hal_amdgpu_ipc_event_record_t* record = nullptr;
  iree_status_t status = iree_hal_amdgpu_ipc_event_record_prepare(
      event, semaphore, /*recorded_value=*/1, &record);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(nullptr, record);
  iree_hal_amdgpu_ipc_event_monitor_set_fail_thread_create_for_testing(false);

  bool reached = true;
  IREE_EXPECT_OK(iree_hal_amdgpu_ipc_event_query(event, &reached));
  EXPECT_FALSE(reached);
  iree_hal_semaphore_release(semaphore);
  iree_hal_amdgpu_ipc_event_release(event);
}

#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

}  // namespace
}  // namespace iree::hal::amdgpu
