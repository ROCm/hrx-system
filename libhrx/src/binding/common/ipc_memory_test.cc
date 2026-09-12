// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/ipc_memory.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include "common/internal.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_device_size_t kAllocationSize = 4096;

class TestSignal {
 public:
  TestSignal() { iree_notification_initialize(&notification_); }
  ~TestSignal() { iree_notification_deinitialize(&notification_); }

  TestSignal(const TestSignal&) = delete;
  TestSignal& operator=(const TestSignal&) = delete;

  void Set() {
    is_set_.store(true, std::memory_order_release);
    iree_notification_post(&notification_, IREE_ALL_WAITERS);
  }

  void Wait() {
    (void)iree_notification_await(&notification_, IsSetCondition, this,
                                  iree_infinite_timeout());
  }

  bool IsSet() const { return is_set_.load(std::memory_order_acquire); }

 private:
  static bool IsSetCondition(void* user_data) {
    return static_cast<TestSignal*>(user_data)->IsSet();
  }

  std::atomic<bool> is_set_{false};
  iree_notification_t notification_;
};

struct FakeIpcMemoryState {
  // Storage returned by successful attach and alias callbacks.
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT)
      std::array<uint8_t, kAllocationSize> shared_storage = {};
  // Alternate storage used to exercise process-wide address mismatches.
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT)
      std::array<uint8_t, kAllocationSize> other_storage = {};
  // Storage backing an ordinary context-owned resource in lifetime tests.
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT)
      std::array<uint8_t, kAllocationSize> context_resource_storage = {};
  // Number of backend attachment attempts.
  std::atomic<int> attach_count{0};
  // Number of attached HAL buffers released.
  std::atomic<int> attached_buffer_release_count{0};
  // Number of context-local alias attempts.
  std::atomic<int> alias_count{0};
  // Number of alias HAL buffers released.
  std::atomic<int> alias_buffer_release_count{0};
  // Number of ordinary context-owned HAL buffers released.
  std::atomic<int> context_resource_release_count{0};
  // Whether ordinary context teardown observed IPC detachment first.
  std::atomic<bool> context_resource_saw_ipc_release{false};
  // Whether ordinary context-owned buffer release should pause.
  std::atomic<bool> block_context_resource_release{false};
  // Posted after ordinary context-owned buffer release has begun.
  TestSignal context_resource_release_blocked;
  // Posted by the test to allow ordinary context-owned buffer release to end.
  TestSignal allow_context_resource_release;
  // Whether alias callbacks return |other_storage| instead of shared storage.
  std::atomic<bool> alias_uses_other_storage{false};
  // Whether attach callbacks return |other_storage| instead of shared storage.
  std::atomic<bool> attach_uses_other_storage{false};
  // Optional status code returned instead of attaching memory.
  std::atomic<iree_status_code_t> attach_failure_code{IREE_STATUS_OK};
  // One-based attach attempt to pause in the callback, or zero to never pause.
  std::atomic<int> blocked_attach_attempt{0};
  // Posted after the selected attach callback has been entered.
  TestSignal attach_blocked;
  // Posted by the test to allow the selected attach callback to return.
  TestSignal allow_attach;
  // Whether final attached-buffer release should pause in its callback.
  std::atomic<bool> block_attached_buffer_release{false};
  // Posted after final attached-buffer release has begun.
  TestSignal attached_buffer_release_blocked;
  // Posted by the test to allow final attached-buffer release to finish.
  TestSignal allow_attached_buffer_release;
  // Whether final attached-buffer release has completed.
  std::atomic<bool> attached_buffer_release_completed{false};
  // One-based attach attempt that records whether release already completed.
  std::atomic<int> observe_release_on_attach_attempt{0};
  // Release-completion state observed by the selected attach callback.
  std::atomic<bool> observed_release_completed{false};
  // Number of backend export callbacks entered.
  std::atomic<int> export_count{0};
  // Whether export should pause before reading its retained inputs.
  std::atomic<bool> block_export{false};
  // Posted after the selected export callback has been entered.
  TestSignal export_blocked;
  // Posted by the test to allow the selected export callback to return.
  TestSignal allow_export;
};

std::atomic<FakeIpcMemoryState*> g_fake_ipc_memory_state{nullptr};

void CountAttachedBufferRelease(void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  auto* state = static_cast<FakeIpcMemoryState*>(user_data);
  state->attached_buffer_release_count.fetch_add(1, std::memory_order_relaxed);
  if (state->block_attached_buffer_release.load(std::memory_order_acquire)) {
    state->attached_buffer_release_blocked.Set();
    state->allow_attached_buffer_release.Wait();
  }
  state->attached_buffer_release_completed.store(true,
                                                 std::memory_order_release);
}

void CountAliasBufferRelease(void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  auto* state = static_cast<FakeIpcMemoryState*>(user_data);
  state->alias_buffer_release_count.fetch_add(1, std::memory_order_relaxed);
}

void CountContextResourceRelease(void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  auto* state = static_cast<FakeIpcMemoryState*>(user_data);
  if (state->block_context_resource_release.load(std::memory_order_acquire)) {
    state->context_resource_release_blocked.Set();
    state->allow_context_resource_release.Wait();
  }
  state->context_resource_saw_ipc_release.store(
      state->attached_buffer_release_count.load(std::memory_order_acquire) == 1,
      std::memory_order_release);
  state->context_resource_release_count.fetch_add(1, std::memory_order_release);
}

iree_status_t CreateTrackedBuffer(std::array<uint8_t, kAllocationSize>& storage,
                                  iree_hal_buffer_release_fn_t release_fn,
                                  FakeIpcMemoryState* state,
                                  iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_release_callback_t release_callback = {
      /*.fn=*/release_fn,
      /*.user_data=*/state,
  };
  return iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      storage.size(), iree_make_byte_span(storage.data(), storage.size()),
      release_callback, iree_allocator_system(), out_buffer);
}

iree_status_t AttachMemory(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size, iree_hal_buffer_t** out_buffer) {
  (void)context;
  (void)descriptor;
  (void)view_size;
  FakeIpcMemoryState* state =
      g_fake_ipc_memory_state.load(std::memory_order_acquire);
  if (!state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fake IPC memory state is unavailable");
  }
  const int attach_attempt =
      state->attach_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (attach_attempt ==
      state->blocked_attach_attempt.load(std::memory_order_acquire)) {
    state->attach_blocked.Set();
    state->allow_attach.Wait();
  }
  if (attach_attempt == state->observe_release_on_attach_attempt.load(
                            std::memory_order_acquire)) {
    state->observed_release_completed.store(
        state->attached_buffer_release_completed.load(
            std::memory_order_acquire),
        std::memory_order_release);
  }
  const iree_status_code_t failure_code =
      state->attach_failure_code.load(std::memory_order_acquire);
  if (failure_code != IREE_STATUS_OK) {
    return iree_status_from_code(failure_code);
  }
  auto& storage =
      state->attach_uses_other_storage.load(std::memory_order_acquire)
          ? state->other_storage
          : state->shared_storage;
  return CreateTrackedBuffer(storage, CountAttachedBufferRelease, state,
                             out_buffer);
}

iree_status_t AliasMemory(iree_hal_streaming_context_t* context,
                          iree_hal_buffer_t* attached_buffer,
                          iree_hal_buffer_t** out_buffer) {
  (void)context;
  (void)attached_buffer;
  FakeIpcMemoryState* state =
      g_fake_ipc_memory_state.load(std::memory_order_acquire);
  if (!state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fake IPC memory state is unavailable");
  }
  state->alias_count.fetch_add(1, std::memory_order_relaxed);
  auto& storage =
      state->alias_uses_other_storage.load(std::memory_order_acquire)
          ? state->other_storage
          : state->shared_storage;
  return CreateTrackedBuffer(storage, CountAliasBufferRelease, state,
                             out_buffer);
}

iree_status_t ExportMemory(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor) {
  FakeIpcMemoryState* state =
      g_fake_ipc_memory_state.load(std::memory_order_acquire);
  if (!state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fake IPC memory state is unavailable");
  }
  state->export_count.fetch_add(1, std::memory_order_relaxed);
  if (state->block_export.load(std::memory_order_acquire)) {
    state->export_blocked.Set();
    state->allow_export.Wait();
  }

  // Read both retained inputs after the deterministic pause so the caller can
  // destroy the streaming wrapper before this backend work resumes.
  memset(out_descriptor, 0, sizeof(*out_descriptor));
  out_descriptor->allocation_size = iree_hal_buffer_byte_length(buffer);
  out_descriptor->exporter_process_id = 1234;
  out_descriptor->exporter_device_ordinal = context->device_ordinal;
  return iree_ok_status();
}

class IpcMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

    InitializeDeviceEntry(/*ordinal=*/0, hrx_device, &device_entries_[0]);
    InitializeDeviceEntry(/*ordinal=*/1, hrx_device, &device_entries_[1]);

    // Export accepts an explicit context-list owner so this CPU-only fixture
    // need not initialize the process-global GPU device registry.
    memset(&device_registry_, 0, sizeof(device_registry_));
    device_registry_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&device_registry_.context_list.mutex);
    iree_notification_initialize(&device_registry_.context_list.changed);

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entries_[0], context_flags, iree_allocator_system(),
        &contexts_[0]));
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entries_[1], context_flags, iree_allocator_system(),
        &contexts_[1]));
    iree_hal_streaming_register_context(&device_registry_, contexts_[0]);
    iree_hal_streaming_register_context(&device_registry_, contexts_[1]);

    iree_hal_streaming_ipc_memory_registry_initialize(&registry_,
                                                      iree_allocator_system());
    for (size_t i = 0; i < sizeof(descriptor_.token); ++i) {
      descriptor_.token[i] = static_cast<uint8_t>(i + 1);
    }
    descriptor_.allocation_size = kAllocationSize;
    descriptor_.byte_offset = 0;
    descriptor_.exporter_process_id = 1234;
    descriptor_.exporter_device_ordinal = 0;
    g_fake_ipc_memory_state.store(&state_, std::memory_order_release);
  }

  void TearDown() override {
    g_fake_ipc_memory_state.store(nullptr, std::memory_order_release);
    iree_hal_streaming_ipc_memory_registry_deinitialize(&registry_);
    iree_hal_streaming_context_release(contexts_[1]);
    iree_hal_streaming_context_release(contexts_[0]);
    EXPECT_EQ(nullptr, device_registry_.context_list.head);
    EXPECT_EQ(nullptr, device_registry_.context_list.tail);
    iree_notification_deinitialize(&device_registry_.context_list.changed);
    iree_slim_mutex_deinitialize(&device_registry_.context_list.mutex);
    DeinitializeDeviceEntry(&device_entries_[1]);
    DeinitializeDeviceEntry(&device_entries_[0]);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  static void InitializeDeviceEntry(iree_host_size_t ordinal,
                                    hrx_device_t hrx_device,
                                    iree_hal_streaming_device_t* device_entry) {
    memset(device_entry, 0, sizeof(*device_entry));
    device_entry->ordinal = ordinal;
    device_entry->hrx_device = hrx_device;
    device_entry->hal_device = hrx_device_hal(hrx_device);
    iree_slim_mutex_initialize(&device_entry->primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry->graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry->block_pool);
  }

  static void DeinitializeDeviceEntry(
      iree_hal_streaming_device_t* device_entry) {
    iree_arena_block_pool_deinitialize(&device_entry->block_pool);
    iree_slim_mutex_deinitialize(&device_entry->graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry->primary_context_mutex);
  }

  iree_status_t CreateContextOnDevice(
      iree_host_size_t device_ordinal,
      iree_hal_streaming_context_t** out_context) {
    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    return iree_hal_streaming_context_create(
        &device_entries_[device_ordinal], context_flags,
        iree_allocator_system(), out_context);
  }

  iree_status_t CreateRegisteredContextOnDevice(
      iree_host_size_t device_ordinal,
      iree_hal_streaming_context_t** out_context) {
    IREE_RETURN_IF_ERROR(CreateContextOnDevice(device_ordinal, out_context));
    iree_hal_streaming_register_context(&device_registry_, *out_context);
    return iree_ok_status();
  }

  void InstallTrackedContextResource(iree_hal_streaming_context_t* context) {
    iree_hal_buffer_t* hal_buffer = nullptr;
    IREE_ASSERT_OK(CreateTrackedBuffer(state_.context_resource_storage,
                                       CountContextResourceRelease, &state_,
                                       &hal_buffer));
    iree_hal_streaming_buffer_t* streaming_buffer = nullptr;
    iree_status_t status = iree_hal_streaming_memory_wrap_buffer(
        context, hal_buffer, IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
        &streaming_buffer);
    iree_hal_buffer_release(hal_buffer);
    IREE_ASSERT_OK(status);
    ASSERT_NE(nullptr, streaming_buffer);
    ASSERT_EQ(nullptr, context->pageable_h2d_staging_buffer);
    context->pageable_h2d_staging_buffer = streaming_buffer;
    context->pageable_h2d_staging_size = kAllocationSize;
  }

  static bool RegistryContains(
      iree_hal_streaming_device_registry_t* device_registry,
      const iree_hal_streaming_context_t* handle) {
    bool found = false;
    iree_slim_mutex_lock(&device_registry->context_list.mutex);
    for (iree_hal_streaming_context_t* context =
             device_registry->context_list.head;
         context; context = context->context_list_entry.next) {
      if (context == handle) {
        found = true;
        break;
      }
    }
    iree_slim_mutex_unlock(&device_registry->context_list.mutex);
    return found;
  }

  bool RegistryContains(const iree_hal_streaming_context_t* handle) {
    return RegistryContains(&device_registry_, handle);
  }

  iree_status_t Import(iree_hal_streaming_context_t* context,
                       void** out_device_ptr) {
    return ImportDescriptor(context, descriptor_, out_device_ptr);
  }

  iree_status_t ImportDescriptor(
      iree_hal_streaming_context_t* context,
      const iree_hal_streaming_ipc_memory_descriptor_t& descriptor,
      void** out_device_ptr) {
    return iree_hal_streaming_ipc_memory_import(
        &registry_, context, &descriptor, kAllocationSize, AttachMemory,
        AliasMemory, out_device_ptr);
  }


  void InsertCollision(iree_hal_streaming_context_t* context, void* device_ptr,
                       void* marker) {
    IREE_ASSERT_OK(HRX_CALL(hrx_buffer_table_insert(
        &context->buffer_table, reinterpret_cast<uint64_t>(device_ptr),
        /*host_ptr=*/nullptr, kAllocationSize, /*buffer=*/nullptr, marker)));
  }

  void ExpectCollision(iree_hal_streaming_context_t* context, void* device_ptr,
                       void* marker) {
    void* found_marker = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_buffer_table_find(
        &context->buffer_table, reinterpret_cast<uint64_t>(device_ptr),
        /*out_buffer=*/nullptr, /*out_offset=*/nullptr, &found_marker)));
    EXPECT_EQ(marker, found_marker);
  }

  void RemoveCollision(iree_hal_streaming_context_t* context,
                       void* device_ptr) {
    IREE_ASSERT_OK(HRX_CALL(hrx_buffer_table_remove(
        &context->buffer_table, reinterpret_cast<uint64_t>(device_ptr))));
  }

  void ExpectIpcMapping(iree_hal_streaming_context_t* context,
                        void* device_ptr) {
    iree_hal_streaming_buffer_ref_t ref = {};
    IREE_ASSERT_OK(iree_hal_streaming_memory_lookup_range(
        context, reinterpret_cast<uint64_t>(device_ptr), /*size=*/1, &ref));
    ASSERT_NE(nullptr, ref.buffer);
    EXPECT_FALSE(ref.buffer->is_device_freeable);
  }

  void ExpectNoMapping(iree_hal_streaming_context_t* context,
                       void* device_ptr) {
    iree_hal_streaming_buffer_ref_t ref = {};
    iree_status_t status = iree_hal_streaming_memory_lookup_range(
        context, reinterpret_cast<uint64_t>(device_ptr), /*size=*/1, &ref);
    EXPECT_EQ(IREE_STATUS_NOT_FOUND, iree_status_code(status));
    iree_status_free(status);
    EXPECT_EQ(nullptr, ref.buffer);
  }

  void ExpectImportFailure(iree_hal_streaming_context_t* context,
                           iree_status_code_t expected_code) {
    void* device_ptr = reinterpret_cast<void*>(uintptr_t{1});
    iree_status_t status = Import(context, &device_ptr);
    EXPECT_EQ(expected_code, iree_status_code(status));
    iree_status_free(status);
    EXPECT_EQ(nullptr, device_ptr);
  }

  FakeIpcMemoryState state_;
  iree_hal_streaming_device_t device_entries_[2] = {};
  iree_hal_streaming_context_t* contexts_[2] = {};
  iree_hal_streaming_device_registry_t device_registry_ = {};
  iree_hal_streaming_ipc_memory_registry_t registry_ = {};
  iree_hal_streaming_ipc_memory_descriptor_t descriptor_ = {};
};

TEST_F(IpcMemoryTest,
       FinalDestructorPublishesUnlinkAfterContextResourceTeardown) {
  iree_hal_streaming_device_registry_t isolated_registry = {};
  isolated_registry.host_allocator = iree_allocator_system();
  iree_slim_mutex_initialize(&isolated_registry.context_list.mutex);
  iree_notification_initialize(&isolated_registry.context_list.changed);

  iree_hal_streaming_context_t* context = nullptr;
  IREE_ASSERT_OK(CreateContextOnDevice(/*device_ordinal=*/0, &context));
  iree_hal_streaming_register_context(&isolated_registry, context);
  InstallTrackedContextResource(context);
  state_.block_context_resource_release.store(true, std::memory_order_release);

  iree_hal_streaming_context_t* raw_handle = context;
  std::thread release_thread(
      [context] { iree_hal_streaming_context_release(context); });
  state_.context_resource_release_blocked.Wait();

  // Final destruction is in progress, but weak membership must remain valid
  // until the context-owned resource release above is allowed to finish.
  EXPECT_EQ(0, iree_atomic_ref_count_load(&raw_handle->ref_count));
  EXPECT_TRUE(RegistryContains(&isolated_registry, raw_handle));
  EXPECT_EQ(
      0, state_.context_resource_release_count.load(std::memory_order_acquire));

  iree_slim_mutex_lock(&isolated_registry.context_list.mutex);
  const iree_wait_token_t wait_token =
      iree_notification_prepare_wait(&isolated_registry.context_list.changed);
  iree_slim_mutex_unlock(&isolated_registry.context_list.mutex);
  state_.allow_context_resource_release.Set();
  EXPECT_TRUE(iree_notification_commit_wait(
      &isolated_registry.context_list.changed, wait_token, IREE_DURATION_ZERO,
      iree_timeout_as_deadline_ns(iree_make_timeout_ms(5000))));
  release_thread.join();

  EXPECT_EQ(
      1, state_.context_resource_release_count.load(std::memory_order_acquire));
  EXPECT_FALSE(RegistryContains(&isolated_registry, raw_handle));
  EXPECT_EQ(nullptr, isolated_registry.context_list.head);
  EXPECT_EQ(nullptr, isolated_registry.context_list.tail);
  iree_notification_deinitialize(&isolated_registry.context_list.changed);
  iree_slim_mutex_deinitialize(&isolated_registry.context_list.mutex);
}

TEST_F(IpcMemoryTest, ContextListDrainWaitsForZeroReferenceDestructor) {
  iree_hal_streaming_device_registry_t isolated_registry = {};
  isolated_registry.host_allocator = iree_allocator_system();
  iree_slim_mutex_initialize(&isolated_registry.context_list.mutex);
  iree_notification_initialize(&isolated_registry.context_list.changed);

  iree_hal_streaming_context_t* context = nullptr;
  IREE_ASSERT_OK(CreateContextOnDevice(/*device_ordinal=*/0, &context));
  iree_hal_streaming_register_context(&isolated_registry, context);
  InstallTrackedContextResource(context);
  state_.block_context_resource_release.store(true, std::memory_order_release);

  iree_hal_streaming_context_t* raw_handle = context;
  std::thread release_thread(
      [context] { iree_hal_streaming_context_release(context); });
  state_.context_resource_release_blocked.Wait();

  // A nonblocking cleanup poll must not detach a zero-reference entry owned by
  // its final destructor.
  iree_status_t poll_status = iree_hal_streaming_context_list_drain(
      &isolated_registry, iree_immediate_timeout());
  EXPECT_EQ(IREE_STATUS_DEADLINE_EXCEEDED, iree_status_code(poll_status));
  iree_status_free(poll_status);
  EXPECT_TRUE(RegistryContains(&isolated_registry, raw_handle));

  iree_status_code_t drain_status_code = IREE_STATUS_OK;
  TestSignal drain_started;
  TestSignal drain_completed;
  std::thread drain_thread([&] {
    drain_started.Set();
    drain_status_code =
        iree_status_consume_code(iree_hal_streaming_context_list_drain(
            &isolated_registry, iree_infinite_timeout()));
    drain_completed.Set();
  });
  drain_started.Wait();
  EXPECT_FALSE(drain_completed.IsSet());

  state_.allow_context_resource_release.Set();
  drain_completed.Wait();
  release_thread.join();
  drain_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, drain_status_code);
  EXPECT_EQ(
      1, state_.context_resource_release_count.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, isolated_registry.context_list.head);
  EXPECT_EQ(nullptr, isolated_registry.context_list.tail);
  iree_notification_deinitialize(&isolated_registry.context_list.changed);
  iree_slim_mutex_deinitialize(&isolated_registry.context_list.mutex);
}

TEST_F(IpcMemoryTest, HandleDestroyRejectsInvalidAndConcurrentRepeats) {
  iree_hal_streaming_context_t* context = nullptr;
  IREE_ASSERT_OK(
      CreateRegisteredContextOnDevice(/*device_ordinal=*/0, &context));

  iree_hal_streaming_context_t* rejected_context = nullptr;
  iree_status_t invalid_status =
      iree_hal_streaming_context_begin_handle_destroy(
          &device_registry_,
          reinterpret_cast<iree_hal_streaming_context_t*>(uintptr_t{1}),
          &rejected_context);
  EXPECT_EQ(IREE_STATUS_NOT_FOUND, iree_status_code(invalid_status));
  iree_status_free(invalid_status);
  EXPECT_EQ(nullptr, rejected_context);

  iree_hal_streaming_context_t* retained_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_context_begin_handle_destroy(
      &device_registry_, context, &retained_context));

  iree_status_code_t concurrent_status_code = IREE_STATUS_OK;
  iree_hal_streaming_context_t* concurrent_context = nullptr;
  std::thread concurrent_thread([&] {
    concurrent_status_code = iree_status_consume_code(
        iree_hal_streaming_context_begin_handle_destroy(
            &device_registry_, context, &concurrent_context));
  });
  concurrent_thread.join();
  EXPECT_EQ(IREE_STATUS_FAILED_PRECONDITION, concurrent_status_code);
  EXPECT_EQ(nullptr, concurrent_context);

  iree_hal_streaming_context_commit_handle_destroy(retained_context);
  iree_hal_streaming_context_t* raw_handle = context;
  iree_slim_mutex_lock(&device_registry_.context_list.mutex);
  std::thread release_thread([=] {
    iree_hal_streaming_context_release(raw_handle);
    iree_hal_streaming_context_release(retained_context);
  });
  while (iree_atomic_ref_count_load(&raw_handle->ref_count) != 0) {
    std::this_thread::yield();
  }

  TestSignal repeated_started;
  iree_status_code_t repeated_status_code = IREE_STATUS_OK;
  iree_hal_streaming_context_t* repeated_context = nullptr;
  std::thread repeated_thread([&] {
    repeated_started.Set();
    repeated_status_code = iree_status_consume_code(
        iree_hal_streaming_context_begin_handle_destroy(
            &device_registry_, raw_handle, &repeated_context));
  });
  repeated_started.Wait();
  iree_slim_mutex_unlock(&device_registry_.context_list.mutex);
  release_thread.join();
  repeated_thread.join();

  EXPECT_EQ(IREE_STATUS_NOT_FOUND, repeated_status_code);
  EXPECT_EQ(nullptr, repeated_context);
  EXPECT_FALSE(RegistryContains(raw_handle));
}

TEST_F(IpcMemoryTest, DeviceFreeLookupRejectsBareWrappedBuffer) {
  ASSERT_EQ(nullptr, iree_hal_streaming_device_registry());

  iree_hal_buffer_t* hal_buffer = nullptr;
  IREE_ASSERT_OK(CreateTrackedBuffer(
      state_.shared_storage, CountAliasBufferRelease, &state_, &hal_buffer));
  iree_hal_streaming_buffer_t* wrapped_buffer = nullptr;
  // Module globals use this same borrowed bare-wrapper path.
  IREE_ASSERT_OK(iree_hal_streaming_memory_wrap_buffer(
      contexts_[0], hal_buffer, IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
      &wrapped_buffer));
  ASSERT_NE(nullptr, wrapped_buffer);
  EXPECT_FALSE(wrapped_buffer->is_device_freeable);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_memory_free_device(
                            contexts_[0], wrapped_buffer->device_ptr));
  iree_hal_streaming_memory_release_wrapped_buffer(wrapped_buffer);
  iree_hal_buffer_release(hal_buffer);

  // This CPU-only fixture intentionally has no global device registry. An
  // owned allocation passes the production device-free lookup and then fails
  // at the subsequent global synchronization boundary.
  iree_hal_streaming_buffer_t* device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      contexts_[0], kAllocationSize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE,
      &device_buffer));
  ASSERT_NE(nullptr, device_buffer);
  EXPECT_TRUE(device_buffer->is_device_freeable);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_memory_free_device(
                            contexts_[0], device_buffer->device_ptr));
  iree_hal_streaming_memory_release_wrapped_buffer(device_buffer);

  iree_hal_streaming_buffer_t* managed_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_managed(
      contexts_[0], kAllocationSize, /*allocation_flags=*/0, &managed_buffer));
  ASSERT_NE(nullptr, managed_buffer);
  EXPECT_TRUE(managed_buffer->is_device_freeable);
  ASSERT_NE(nullptr, managed_buffer->host_ptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_streaming_memory_free_device(
          contexts_[0], reinterpret_cast<iree_hal_streaming_deviceptr_t>(
                            managed_buffer->host_ptr)));
  iree_hal_streaming_memory_release_wrapped_buffer(managed_buffer);
}

TEST_F(IpcMemoryTest, ExportSurvivesWrapperFreeAfterLockedSnapshot) {
  iree_hal_streaming_buffer_t* device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      contexts_[0], kAllocationSize,
      IREE_HAL_STREAMING_MEMORY_FLAG_SHARING_EXPORT, &device_buffer));
  ASSERT_NE(nullptr, device_buffer);
  const uint64_t device_ptr = device_buffer->device_ptr;
  state_.block_export.store(true, std::memory_order_release);

  iree_hal_streaming_ipc_memory_descriptor_t exported_descriptor = {};
  iree_host_size_t owner_device_ordinal = 99;
  iree_status_code_t export_status_code = IREE_STATUS_OK;
  TestSignal export_completed;
  std::thread export_thread([&] {
    export_status_code =
        iree_status_consume_code(iree_hal_streaming_ipc_memory_export(
            &device_registry_, contexts_[0], device_ptr, ExportMemory,
            &exported_descriptor, &owner_device_ordinal));
    export_completed.Set();
  });
  state_.export_blocked.Wait();
  EXPECT_EQ(1, state_.export_count.load(std::memory_order_acquire));
  EXPECT_FALSE(export_completed.IsSet());

  // The callback begins only after the locked snapshot is complete. Removing
  // the table entry and synchronously destroying its wrapper must not strand
  // or invalidate the retained backend buffer used when export resumes.
  iree_hal_streaming_memory_release_wrapped_buffer(device_buffer);
  device_buffer = nullptr;
  ExpectNoMapping(contexts_[0], reinterpret_cast<void*>(device_ptr));
  EXPECT_FALSE(export_completed.IsSet());

  state_.allow_export.Set();
  export_thread.join();
  EXPECT_EQ(IREE_STATUS_OK, export_status_code);
  EXPECT_TRUE(export_completed.IsSet());
  EXPECT_GE(exported_descriptor.allocation_size, kAllocationSize);
  EXPECT_EQ(0u, exported_descriptor.byte_offset);
  EXPECT_EQ(1234, exported_descriptor.exporter_process_id);
  EXPECT_EQ(contexts_[0]->device_ordinal,
            exported_descriptor.exporter_device_ordinal);
  EXPECT_EQ(contexts_[0]->device_ordinal, owner_device_ordinal);
}

TEST_F(IpcMemoryTest, ExportRejectsBufferFreedBeforeLockedSnapshot) {
  iree_hal_streaming_buffer_t* device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
      contexts_[0], kAllocationSize,
      IREE_HAL_STREAMING_MEMORY_FLAG_SHARING_EXPORT, &device_buffer));
  ASSERT_NE(nullptr, device_buffer);
  const uint64_t device_ptr = device_buffer->device_ptr;
  iree_hal_streaming_memory_release_wrapped_buffer(device_buffer);
  ExpectNoMapping(contexts_[0], reinterpret_cast<void*>(device_ptr));

  iree_hal_streaming_ipc_memory_descriptor_t exported_descriptor;
  memset(&exported_descriptor, 0xA5, sizeof(exported_descriptor));
  iree_host_size_t owner_device_ordinal = 99;
  iree_status_t status = iree_hal_streaming_ipc_memory_export(
      &device_registry_, contexts_[0], device_ptr, ExportMemory,
      &exported_descriptor, &owner_device_ordinal);

  EXPECT_EQ(IREE_STATUS_NOT_FOUND, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(0, state_.export_count.load(std::memory_order_acquire));
  EXPECT_EQ(0u, owner_device_ordinal);
  const iree_hal_streaming_ipc_memory_descriptor_t empty_descriptor = {};
  EXPECT_EQ(0, memcmp(&empty_descriptor, &exported_descriptor,
                      sizeof(exported_descriptor)));
}

TEST_F(IpcMemoryTest, AttachWrapperFailureUnlinksAndCanRetry) {
  const int32_t context_references_before =
      iree_atomic_ref_count_load(&contexts_[0]->ref_count);
  void* shared_ptr = state_.shared_storage.data();
  int collision_marker = 0;
  InsertCollision(contexts_[0], shared_ptr, &collision_marker);

  ExpectImportFailure(contexts_[0], IREE_STATUS_NOT_FOUND);

  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(1, state_.attached_buffer_release_count.load());
  EXPECT_EQ(0, state_.alias_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
  EXPECT_EQ(context_references_before,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  ExpectCollision(contexts_[0], shared_ptr, &collision_marker);

  RemoveCollision(contexts_[0], shared_ptr);
  void* imported_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &imported_ptr));
  EXPECT_EQ(shared_ptr, imported_ptr);
  EXPECT_EQ(2, state_.attach_count.load());
  EXPECT_EQ(1, state_.attached_buffer_release_count.load());
  EXPECT_NE(nullptr, registry_.imports);
  EXPECT_EQ(context_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  ExpectIpcMapping(contexts_[0], shared_ptr);

  IREE_ASSERT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     imported_ptr));
  EXPECT_EQ(2, state_.attached_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
  EXPECT_EQ(context_references_before,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  ExpectNoMapping(contexts_[0], shared_ptr);
}

TEST_F(IpcMemoryTest, AliasWrapperFailureKeepsAnchorReadyAndCanRetry) {
  const int32_t anchor_references_before =
      iree_atomic_ref_count_load(&contexts_[0]->ref_count);
  const int32_t alias_references_before =
      iree_atomic_ref_count_load(&contexts_[1]->ref_count);
  void* shared_ptr = state_.shared_storage.data();

  void* anchor_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &anchor_ptr));
  ASSERT_EQ(shared_ptr, anchor_ptr);
  iree_hal_streaming_ipc_memory_import_t* anchor_import = registry_.imports;
  ASSERT_NE(nullptr, anchor_import);
  int collision_marker = 0;
  InsertCollision(contexts_[1], shared_ptr, &collision_marker);

  ExpectImportFailure(contexts_[1], IREE_STATUS_NOT_FOUND);

  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(0, state_.attached_buffer_release_count.load());
  EXPECT_EQ(1, state_.alias_count.load());
  EXPECT_EQ(1, state_.alias_buffer_release_count.load());
  EXPECT_EQ(anchor_import, registry_.imports);
  EXPECT_EQ(anchor_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  EXPECT_EQ(alias_references_before,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectIpcMapping(contexts_[0], anchor_ptr);
  ExpectCollision(contexts_[1], shared_ptr, &collision_marker);

  RemoveCollision(contexts_[1], shared_ptr);
  void* alias_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[1], &alias_ptr));
  EXPECT_EQ(anchor_ptr, alias_ptr);
  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(2, state_.alias_count.load());
  EXPECT_EQ(1, state_.alias_buffer_release_count.load());
  EXPECT_EQ(anchor_import, registry_.imports);
  EXPECT_EQ(alias_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectIpcMapping(contexts_[1], alias_ptr);

  IREE_ASSERT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     anchor_ptr));
  EXPECT_EQ(0, state_.attached_buffer_release_count.load());
  EXPECT_EQ(1, state_.alias_buffer_release_count.load());
  EXPECT_EQ(anchor_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  EXPECT_EQ(alias_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectIpcMapping(contexts_[0], anchor_ptr);
  ExpectIpcMapping(contexts_[1], alias_ptr);

  IREE_ASSERT_OK(
      iree_hal_streaming_ipc_memory_close(&registry_, contexts_[1], alias_ptr));
  EXPECT_EQ(1, state_.attached_buffer_release_count.load());
  EXPECT_EQ(2, state_.alias_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
  EXPECT_EQ(anchor_references_before,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  EXPECT_EQ(alias_references_before,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectNoMapping(contexts_[0], anchor_ptr);
  ExpectNoMapping(contexts_[1], alias_ptr);
}

TEST_F(IpcMemoryTest, AliasAddressMismatchUnregistersAndCanRetry) {
  const int32_t anchor_references_before =
      iree_atomic_ref_count_load(&contexts_[0]->ref_count);
  const int32_t alias_references_before =
      iree_atomic_ref_count_load(&contexts_[1]->ref_count);
  void* shared_ptr = state_.shared_storage.data();
  void* other_ptr = state_.other_storage.data();

  void* anchor_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &anchor_ptr));
  ASSERT_EQ(shared_ptr, anchor_ptr);
  iree_hal_streaming_ipc_memory_import_t* anchor_import = registry_.imports;
  ASSERT_NE(nullptr, anchor_import);

  state_.alias_uses_other_storage = true;
  ExpectImportFailure(contexts_[1], IREE_STATUS_NOT_FOUND);

  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(0, state_.attached_buffer_release_count.load());
  EXPECT_EQ(1, state_.alias_count.load());
  EXPECT_EQ(1, state_.alias_buffer_release_count.load());
  EXPECT_EQ(anchor_import, registry_.imports);
  EXPECT_EQ(anchor_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  EXPECT_EQ(alias_references_before,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectIpcMapping(contexts_[0], anchor_ptr);
  ExpectNoMapping(contexts_[1], other_ptr);

  state_.alias_uses_other_storage = false;
  void* alias_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[1], &alias_ptr));
  EXPECT_EQ(anchor_ptr, alias_ptr);
  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(2, state_.alias_count.load());
  EXPECT_EQ(1, state_.alias_buffer_release_count.load());
  EXPECT_EQ(anchor_import, registry_.imports);
  EXPECT_EQ(alias_references_before + 1,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectIpcMapping(contexts_[1], alias_ptr);

  IREE_ASSERT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     anchor_ptr));
  IREE_ASSERT_OK(
      iree_hal_streaming_ipc_memory_close(&registry_, contexts_[1], alias_ptr));
  EXPECT_EQ(1, state_.attached_buffer_release_count.load());
  EXPECT_EQ(2, state_.alias_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
  EXPECT_EQ(anchor_references_before,
            iree_atomic_ref_count_load(&contexts_[0]->ref_count));
  EXPECT_EQ(alias_references_before,
            iree_atomic_ref_count_load(&contexts_[1]->ref_count));
  ExpectNoMapping(contexts_[0], anchor_ptr);
  ExpectNoMapping(contexts_[1], alias_ptr);
}

TEST_F(IpcMemoryTest, SameDeviceContextCannotOpenDuplicateHandle) {
  void* imported_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &imported_ptr));
  ASSERT_EQ(state_.shared_storage.data(), imported_ptr);

  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  iree_hal_streaming_context_t* other_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entries_[0], context_flags, iree_allocator_system(),
      &other_context));

  void* duplicate_ptr = reinterpret_cast<void*>(uintptr_t{1});
  iree_status_t status = Import(other_context, &duplicate_ptr);
  EXPECT_EQ(IREE_STATUS_ALREADY_EXISTS, iree_status_code(status));
  const bool unexpectedly_opened = iree_status_is_ok(status);
  iree_status_free(status);
  EXPECT_EQ(nullptr, duplicate_ptr);
  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(0, state_.alias_count.load());
  ExpectIpcMapping(contexts_[0], imported_ptr);
  ExpectNoMapping(other_context, imported_ptr);

  if (unexpectedly_opened) {
    IREE_EXPECT_OK(iree_hal_streaming_ipc_memory_close(
        &registry_, other_context, duplicate_ptr));
  }
  IREE_EXPECT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     imported_ptr));
  iree_hal_streaming_context_release(other_context);
  EXPECT_EQ(1, state_.attached_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
}

TEST_F(IpcMemoryTest, SameSourceDeviceUsesOneContextAcrossHandles) {
  void* first_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &first_ptr));
  ASSERT_EQ(state_.shared_storage.data(), first_ptr);

  iree_hal_streaming_ipc_memory_descriptor_t second_descriptor = descriptor_;
  ++second_descriptor.token[0];
  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  iree_hal_streaming_context_t* other_context = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entries_[0], context_flags, iree_allocator_system(),
      &other_context));

  void* rejected_ptr = reinterpret_cast<void*>(uintptr_t{1});
  iree_status_t rejected_status =
      ImportDescriptor(other_context, second_descriptor, &rejected_ptr);
  EXPECT_EQ(IREE_STATUS_ALREADY_EXISTS, iree_status_code(rejected_status));
  iree_status_free(rejected_status);
  EXPECT_EQ(nullptr, rejected_ptr);
  EXPECT_EQ(1, state_.attach_count.load());
  ExpectNoMapping(other_context, first_ptr);

  state_.attach_uses_other_storage = true;
  void* second_ptr = nullptr;
  IREE_ASSERT_OK(
      ImportDescriptor(contexts_[0], second_descriptor, &second_ptr));
  EXPECT_EQ(state_.other_storage.data(), second_ptr);
  EXPECT_EQ(2, state_.attach_count.load());
  ExpectIpcMapping(contexts_[0], first_ptr);
  ExpectIpcMapping(contexts_[0], second_ptr);

  IREE_EXPECT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     second_ptr));
  IREE_EXPECT_OK(
      iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0], first_ptr));
  iree_hal_streaming_context_release(other_context);
  EXPECT_EQ(2, state_.attached_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
}

TEST_F(IpcMemoryTest, ConcurrentDistinctHandlesWaitThenRejectCompetingContext) {
  iree_hal_streaming_ipc_memory_descriptor_t second_descriptor = descriptor_;
  ++second_descriptor.token[0];
  iree_hal_streaming_context_t* competing_context = nullptr;
  IREE_ASSERT_OK(
      CreateContextOnDevice(/*device_ordinal=*/0, &competing_context));
  state_.blocked_attach_attempt.store(1, std::memory_order_release);

  void* first_ptr = nullptr;
  iree_status_code_t first_status_code = IREE_STATUS_OK;
  std::thread first_thread([&] {
    first_status_code =
        iree_status_consume_code(Import(contexts_[0], &first_ptr));
  });
  state_.attach_blocked.Wait();

  TestSignal second_started;
  TestSignal second_completed;
  void* second_ptr = reinterpret_cast<void*>(uintptr_t{1});
  iree_status_code_t second_status_code = IREE_STATUS_OK;
  std::thread second_thread([&] {
    second_started.Set();
    second_status_code = iree_status_consume_code(
        ImportDescriptor(competing_context, second_descriptor, &second_ptr));
    second_completed.Set();
  });
  second_started.Wait();

  EXPECT_FALSE(second_completed.IsSet());
  EXPECT_EQ(1, state_.attach_count.load(std::memory_order_acquire));
  state_.allow_attach.Set();
  first_thread.join();
  second_thread.join();

  EXPECT_EQ(IREE_STATUS_OK, first_status_code);
  EXPECT_EQ(IREE_STATUS_ALREADY_EXISTS, second_status_code);
  EXPECT_EQ(state_.shared_storage.data(), first_ptr);
  EXPECT_EQ(nullptr, second_ptr);
  EXPECT_TRUE(second_completed.IsSet());
  EXPECT_EQ(1, state_.attach_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, state_.alias_count.load(std::memory_order_acquire));

  if (first_ptr) {
    IREE_EXPECT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                       first_ptr));
  }
  iree_hal_streaming_context_release(competing_context);
  EXPECT_EQ(
      1, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, registry_.imports);
}

TEST_F(IpcMemoryTest, OpenWaitsForFinalDetachBeforeAttaching) {
  void* first_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &first_ptr));
  ASSERT_EQ(state_.shared_storage.data(), first_ptr);

  iree_hal_streaming_ipc_memory_descriptor_t second_descriptor = descriptor_;
  ++second_descriptor.token[0];
  iree_hal_streaming_context_t* next_context = nullptr;
  IREE_ASSERT_OK(CreateContextOnDevice(/*device_ordinal=*/0, &next_context));
  state_.block_attached_buffer_release.store(true, std::memory_order_release);
  state_.observe_release_on_attach_attempt.store(2, std::memory_order_release);

  iree_status_code_t close_status_code = IREE_STATUS_OK;
  std::thread close_thread([&] {
    close_status_code =
        iree_status_consume_code(iree_hal_streaming_ipc_memory_close(
            &registry_, contexts_[0], first_ptr));
  });
  state_.attached_buffer_release_blocked.Wait();

  TestSignal open_started;
  TestSignal open_completed;
  void* second_ptr = nullptr;
  iree_status_code_t open_status_code = IREE_STATUS_OK;
  std::thread open_thread([&] {
    open_started.Set();
    open_status_code = iree_status_consume_code(
        ImportDescriptor(next_context, second_descriptor, &second_ptr));
    open_completed.Set();
  });
  open_started.Wait();

  EXPECT_FALSE(open_completed.IsSet());
  EXPECT_EQ(1, state_.attach_count.load(std::memory_order_acquire));
  EXPECT_FALSE(
      state_.attached_buffer_release_completed.load(std::memory_order_acquire));
  state_.allow_attached_buffer_release.Set();
  close_thread.join();
  open_thread.join();

  EXPECT_EQ(IREE_STATUS_OK, close_status_code);
  EXPECT_EQ(IREE_STATUS_OK, open_status_code);
  EXPECT_EQ(state_.shared_storage.data(), second_ptr);
  EXPECT_TRUE(
      state_.observed_release_completed.load(std::memory_order_acquire));
  EXPECT_EQ(2, state_.attach_count.load(std::memory_order_acquire));
  EXPECT_EQ(
      1, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  ExpectNoMapping(contexts_[0], first_ptr);
  ExpectIpcMapping(next_context, second_ptr);

  if (second_ptr) {
    IREE_EXPECT_OK(iree_hal_streaming_ipc_memory_close(&registry_, next_context,
                                                       second_ptr));
  }
  iree_hal_streaming_context_release(next_context);
  EXPECT_EQ(
      2, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, registry_.imports);
}

TEST_F(IpcMemoryTest, ConcurrentDuplicateUsesOneAttachAndTwoReferences) {
  state_.blocked_attach_attempt.store(1, std::memory_order_release);

  void* first_ptr = nullptr;
  iree_status_code_t first_status_code = IREE_STATUS_OK;
  std::thread first_thread([&] {
    first_status_code =
        iree_status_consume_code(Import(contexts_[0], &first_ptr));
  });
  state_.attach_blocked.Wait();

  TestSignal second_started;
  TestSignal second_completed;
  void* second_ptr = nullptr;
  iree_status_code_t second_status_code = IREE_STATUS_OK;
  std::thread second_thread([&] {
    second_started.Set();
    second_status_code =
        iree_status_consume_code(Import(contexts_[0], &second_ptr));
    second_completed.Set();
  });
  second_started.Wait();

  EXPECT_FALSE(second_completed.IsSet());
  EXPECT_EQ(1, state_.attach_count.load(std::memory_order_acquire));
  state_.allow_attach.Set();
  first_thread.join();
  second_thread.join();

  EXPECT_EQ(IREE_STATUS_OK, first_status_code);
  EXPECT_EQ(IREE_STATUS_OK, second_status_code);
  ASSERT_EQ(state_.shared_storage.data(), first_ptr);
  ASSERT_EQ(first_ptr, second_ptr);
  EXPECT_EQ(1, state_.attach_count.load(std::memory_order_acquire));

  IREE_ASSERT_OK(
      iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0], first_ptr));
  EXPECT_EQ(
      0, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  EXPECT_NE(nullptr, registry_.imports);
  ExpectIpcMapping(contexts_[0], second_ptr);

  IREE_ASSERT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     second_ptr));
  EXPECT_EQ(
      1, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, registry_.imports);
  ExpectNoMapping(contexts_[0], second_ptr);
}


TEST_F(IpcMemoryTest, SynchronizeFailureRestoresReadyWithoutClosing) {
  void* imported_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &imported_ptr));
  ASSERT_EQ(state_.shared_storage.data(), imported_ptr);

  iree_hal_semaphore_t* failed_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      contexts_[0]->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &failed_semaphore));
  iree_hal_semaphore_fail(
      failed_semaphore,
      iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                       "injected IPC close synchronization failure"));

  iree_slim_mutex_lock(&contexts_[0]->event_record_mutex);
  iree_hal_semaphore_t* original_semaphore =
      contexts_[0]->event_record_timeline.semaphore;
  const uint64_t original_pending_value =
      contexts_[0]->event_record_timeline.pending_value;
  contexts_[0]->event_record_timeline.semaphore = failed_semaphore;
  contexts_[0]->event_record_timeline.pending_value = 1;
  iree_slim_mutex_unlock(&contexts_[0]->event_record_mutex);

  iree_status_t close_status = iree_hal_streaming_ipc_memory_close(
      &registry_, contexts_[0], imported_ptr);

  iree_slim_mutex_lock(&contexts_[0]->event_record_mutex);
  contexts_[0]->event_record_timeline.semaphore = original_semaphore;
  contexts_[0]->event_record_timeline.pending_value = original_pending_value;
  iree_slim_mutex_unlock(&contexts_[0]->event_record_mutex);
  iree_hal_semaphore_release(failed_semaphore);

  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(close_status));
  iree_status_free(close_status);
  EXPECT_EQ(
      0, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  EXPECT_NE(nullptr, registry_.imports);
  ExpectIpcMapping(contexts_[0], imported_ptr);

  IREE_ASSERT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     imported_ptr));
  EXPECT_EQ(
      1, state_.attached_buffer_release_count.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, registry_.imports);
  ExpectNoMapping(contexts_[0], imported_ptr);
}

TEST_F(IpcMemoryTest, AttachResourceExhaustionIsPreserved) {
  state_.attach_failure_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  ExpectImportFailure(contexts_[0], IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(1, state_.attach_count.load());
  EXPECT_EQ(0, state_.attached_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);

  state_.attach_failure_code = IREE_STATUS_OK;
  void* imported_ptr = nullptr;
  IREE_ASSERT_OK(Import(contexts_[0], &imported_ptr));
  EXPECT_EQ(state_.shared_storage.data(), imported_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_ipc_memory_close(&registry_, contexts_[0],
                                                     imported_ptr));
  EXPECT_EQ(2, state_.attach_count.load());
  EXPECT_EQ(1, state_.attached_buffer_release_count.load());
  EXPECT_EQ(nullptr, registry_.imports);
}

}  // namespace
