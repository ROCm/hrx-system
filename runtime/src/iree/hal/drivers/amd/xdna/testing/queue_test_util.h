// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_TESTING_QUEUE_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_TESTING_QUEUE_TEST_UTIL_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/event.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/xdna/device.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/hal/drivers/amd/xdna/memory.h"
#include "iree/hal/drivers/amd/xdna/queue.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"
#include "iree/hal/drivers/amd/xdna/testing/allocation_tracker.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::xdna::testing {

enum class Outcome {
  kSuccess,
  kRejectSubmission,
  kRetiredFailure,
  kNotificationFailure,
  kRefreshFailure,
  kUnretiredFailure,
  kInactiveWithoutFailure,
  kEarlyWake,
};

// Only the libamdf dependency is replaced. Device construction, executable
// loading, buffer capture, native notification, proactor polling, semaphore
// publication, and teardown all use their production implementations.
struct NativeProvider {
  // Holds a selected native call while independent production work progresses.
  struct MemoryGate {
    // Guards this dependency's explicit readiness predicates.
    std::mutex mutex;
    // Publishes entry and release without timing assumptions.
    std::condition_variable condition;
    // Exact native size to intercept; zero leaves calls unblocked.
    uint64_t byte_length = 0;
    // The selected native call has reached the gate.
    bool entered = false;
    // The test has permitted that native call to return.
    bool released = false;

    void Arm(uint64_t length) {
      std::lock_guard<std::mutex> lock(mutex);
      byte_length = length;
      entered = released = false;
    }
    void Enter(uint64_t length) {
      std::unique_lock<std::mutex> lock(mutex);
      if (length != byte_length) {
        return;
      }
      entered = true;
      condition.notify_all();
      condition.wait(lock, [&] { return released; });
      byte_length = 0;
    }
    void AwaitEntry() {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [&] { return entered; });
    }
    void Release() {
      std::lock_guard<std::mutex> lock(mutex);
      released = true;
      condition.notify_all();
    }
  };

  struct Memory {
    // Provider observing native ownership transitions.
    NativeProvider* owner;
    // Storage backing the native host mapping.
    std::vector<uint8_t> contents;
    // Aligned native address independent of host virtual placement.
    uint64_t address;
  };

  NativeProvider() {
    iree_notification_initialize(&submission_notification);
    api.instance_destroy = InstanceDestroy;
    api.instance_create = InstanceCreate;
    api.instance_enumerate_memory_scopes = EnumerateMemoryScopes;
    api.endpoint_enumerate = EnumerateEndpoints;
    api.memory_create = MemoryCreate;
    api.memory_query_info = MemoryQueryInfo;
    api.memory_destroy = MemoryDestroy;
    api.memory_map = MemoryMap;
    api.host_mapping_destroy = MappingDestroy;
    api.host_mapping_query_info = MappingInfo;
    api.host_mapping_cache_control = CacheControl;
    api.memory_query_address = MemoryAddress;
    api.memory_scope_query_info = MemoryScopeQueryInfo;
    api.memory_scope_query_device_profile = MemoryScopeQueryDeviceProfile;
    api.memory_scope_query_pair_info = MemoryScopeQueryPairInfo;
    api.device_destroy = DeviceDestroy;
    api.kernel_queue_query_info = QueueInfo;
    api.kernel_queue_query_status = QueryStatus;
    api.kernel_queue_request_notification = Notify;
    api.kernel_queue_refresh_status = Refresh;
    api.kernel_queue_wait = Wait;
    api.kernel_queue_destroy = QueueDestroy;
    xdna.kernel_queue_create = QueueCreate;
    xdna.kernel_queue_submit = Submit;
    xdna.context_destroy = ContextDestroy;
    xdna.context_query_result_buffer_info = ResultInfo;
    xdna.context_attach_result_buffer = ResultAttach;
    xdna.context_collect_result_buffer = ResultCollect;
    xdna.context_detach_result_buffer = ResultDetach;
  }

  ~NativeProvider() {
    iree_notification_deinitialize(&submission_notification);
  }

  static amdf_status_t AMDF_CALL InstanceCreate(
      const amdf_instance_create_info_t* info, amdf_instance_t** out_instance) {
    *out_instance = reinterpret_cast<amdf_instance_t*>(construction_instance);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  EnumerateMemoryScopes(amdf_instance_t* instance, uint32_t capacity,
                        amdf_memory_scope_t** scopes, uint32_t* out_count) {
    *out_count = 1;
    if (!capacity) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    }
    scopes[0] = reinterpret_cast<amdf_memory_scope_t*>(instance);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  EnumerateEndpoints(amdf_instance_t* instance, uint32_t capacity,
                     amdf_endpoint_summary_t* endpoints, uint32_t* out_count) {
    *out_count = 1;
    if (capacity) {
      endpoints[0] = {};
      endpoints[0].engine_kind = AMDF_ENGINE_KIND_XDNA;
    }
    return AMDF_STATUS_OK;
  }

  static bool SubmissionEntered(void* user_data) {
    auto* self = static_cast<NativeProvider*>(user_data);
    return self->submission_entered.load(std::memory_order_acquire);
  }

  static bool SubmissionReleased(void* user_data) {
    auto* self = static_cast<NativeProvider*>(user_data);
    return self->submission_released.load(std::memory_order_acquire);
  }

  void BlockSubmission() {
    submission_entered.store(false, std::memory_order_relaxed);
    submission_released.store(false, std::memory_order_relaxed);
    block_submission.store(true, std::memory_order_release);
  }

  void AwaitBlockedSubmission() {
    iree_notification_await(&submission_notification, SubmissionEntered, this,
                            iree_infinite_timeout());
  }

  void ReleaseSubmission() {
    submission_released.store(true, std::memory_order_release);
    block_submission.store(false, std::memory_order_release);
    iree_notification_post(&submission_notification, IREE_ALL_WAITERS);
  }

  void ReturnBusyOnNextSubmission() {
    busy_submission_attempt.store(
        submission_attempt_count.load(std::memory_order_acquire) + 1,
        std::memory_order_release);
  }

  void BlockWait() {
    wait_entered.store(false, std::memory_order_relaxed);
    wait_released.store(false, std::memory_order_relaxed);
    block_wait.store(true, std::memory_order_release);
  }

  void AwaitBlockedWait() {
    iree_notification_await(
        &submission_notification,
        +[](void* user_data) {
          return static_cast<NativeProvider*>(user_data)->wait_entered.load(
              std::memory_order_acquire);
        },
        this, iree_infinite_timeout());
  }

  void ReleaseWait() {
    wait_released.store(true, std::memory_order_release);
    block_wait.store(false, std::memory_order_release);
    iree_notification_post(&submission_notification, IREE_ALL_WAITERS);
  }

  size_t FlushCount() {
    std::lock_guard<std::mutex> lock(mutex);
    return flushes.size();
  }

  // Finite result controls model the native dependency. Marker transport uses
  // the explicit measured fixture layout; hardware CTS covers native decoding.
  static amdf_status_t AMDF_CALL ResultInfo(
      amdf_xdna_context_t* context, amdf_xdna_result_buffer_info_t* info) {
    auto* self = reinterpret_cast<NativeProvider*>(context);
    info->memory_scope = reinterpret_cast<amdf_memory_scope_t*>(self);
    info->memory_profile_ordinal = 0;
    info->binding_mode = self->enforce_single_command_memory
                             ? AMDF_XDNA_RESULT_BUFFER_BINDING_RANGE
                             : AMDF_XDNA_RESULT_BUFFER_BINDING_WHOLE_MEMORY;
    info->minimum_byte_length = 24;
    info->maximum_byte_length = self->result.capacity;
    info->byte_offset_alignment = 4;
    info->byte_length_granularity = 4;
    info->timer_clock.counter_bit_count = 64;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL
  ResultAttach(amdf_xdna_context_t* context,
               const amdf_xdna_result_buffer_binding_t* binding) {
    auto* self = reinterpret_cast<NativeProvider*>(context);
    EXPECT_FALSE(self->result.binding.memory);
    ++self->result.attaches;
    if (self->result.unsupported_attachment) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    self->result.binding = *binding;
    self->result.count = 0;
    return self->result.fail_control == 1 ? self->result.failure_status
                                          : AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL
  ResultCollect(amdf_xdna_context_t* context,
                const amdf_xdna_result_buffer_binding_t* binding) {
    auto* self = reinterpret_cast<NativeProvider*>(context);
    EXPECT_EQ(binding->memory, self->result.binding.memory);
    EXPECT_EQ(binding->byte_offset, self->result.binding.byte_offset);
    EXPECT_EQ(binding->byte_length, self->result.binding.byte_length);
    EXPECT_GE(self->checked_retired_submission.load(),
              self->last_accepted_submission.load());
    ++self->result.collects;
    if (self->result.corrupt) {
      auto* memory = reinterpret_cast<Memory*>(binding->memory);
      memory->contents[binding->byte_offset] = 0;
    }
    return self->result.fail_control == 2 ? self->result.failure_status
                                          : AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL
  ResultDetach(amdf_xdna_context_t* context,
               const amdf_xdna_result_buffer_binding_t* binding) {
    auto* self = reinterpret_cast<NativeProvider*>(context);
    EXPECT_EQ(binding->memory, self->result.binding.memory);
    ++self->result.detaches;
    if (self->result.fail_control == 3) {
      return self->result.failure_status;
    }
    self->result.binding = {};
    return AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL MemoryCreate(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
      amdf_memory_t** out_memory) {
    auto* self = reinterpret_cast<NativeProvider*>(scope);
    self->allocation_gate.Enter(info->byte_length);
    const bool is_command_memory =
        info->access_count == 1 &&
        iree_any_bit_set(info->accesses[0].requirements.address_kinds,
                         UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE);
    if (is_command_memory && self->enforce_single_command_memory &&
        self->command_memory_create_count != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    auto* memory = new Memory{self, std::vector<uint8_t>(info->byte_length),
                              self->next_address.fetch_add(
                                  iree_host_align(info->byte_length, 32768))};
    self->command_memory_create_count += is_command_memory ? 1 : 0;
    ++self->live_memories;
    ++self->memory_create_count;
    *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryQueryInfo(amdf_memory_t* handle,
                                                 amdf_memory_info_t* out_info) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    out_info->byte_length = memory->contents.size();
    out_info->alignment = 32768;
    out_info->native_allocation_byte_length = memory->contents.size();
    out_info->native_allocation_granularity = 4096;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryDestroy(amdf_memory_t* handle) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    memory->owner->release_gate.Enter(memory->contents.size());
    --memory->owner->live_memories;
    delete memory;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryMap(amdf_memory_t* memory,
                                           const amdf_memory_map_info_t* info,
                                           amdf_host_mapping_t** out_mapping) {
    *out_mapping = reinterpret_cast<amdf_host_mapping_t*>(memory);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MappingDestroy(amdf_host_mapping_t* mapping) {
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MappingInfo(amdf_host_mapping_t* handle,
                                             amdf_host_mapping_info_t* info) {
    auto* memory = reinterpret_cast<Memory*>(handle);
    info->pointer = memory->contents.data();
    info->byte_length = memory->contents.size();
    info->flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    info->cacheability = AMDF_HOST_CACHEABILITY_UNCACHED;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL CacheControl(
      amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
      uint64_t offset, uint64_t length) {
    auto* memory = reinterpret_cast<Memory*>(mapping);
    std::lock_guard<std::mutex> lock(memory->owner->mutex);
    memory->owner->flushes.push_back({memory, offset, length});
    if (memory->owner->fail_cache_publication_countdown &&
        --memory->owner->fail_cache_publication_countdown == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryAddress(amdf_memory_t* handle,
                                               uint32_t access_ordinal,
                                               amdf_memory_address_kind_t kind,
                                               uint64_t* out_address) {
    *out_address = reinterpret_cast<Memory*>(handle)->address;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryScopeQueryInfo(
      amdf_memory_scope_t* scope, amdf_memory_scope_info_t* out_info) {
    (void)scope;
    out_info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
    out_info->memory_profile_count = 1;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryScopeQueryDeviceProfile(
      amdf_memory_scope_t* scope, uint32_t profile_ordinal,
      uint32_t access_count, const amdf_memory_device_access_t* accesses,
      amdf_memory_profile_t* out_profile,
      amdf_memory_access_capabilities_t* out_capabilities) {
    auto* self = reinterpret_cast<NativeProvider*>(scope);
    EXPECT_EQ(profile_ordinal, 0u);
    EXPECT_EQ(access_count, 1u);
    if (access_count != 1) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    EXPECT_EQ(accesses[0].device, reinterpret_cast<amdf_device_t*>(self));
    EXPECT_EQ(accesses[0].requirements.access,
              AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
    EXPECT_EQ(accesses[0].requirements.flags, AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
    EXPECT_EQ(accesses[0].requirements.address_kinds,
              UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA);
    out_capabilities[0].guaranteed_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    out_capabilities[0].supported_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    out_capabilities[0].guaranteed_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    out_capabilities[0].supported_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    out_capabilities[0].address_kinds = UINT64_C(1)
                                        << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    out_profile->ordinal = profile_ordinal;
    out_profile->memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    out_profile->roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    out_profile->guaranteed_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    out_profile->supported_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    out_profile->allocation.maximum_byte_length = UINT64_C(1) << 30;
    out_profile->allocation.byte_length_granularity = 4096;
    out_profile->allocation.minimum_alignment = 4096;
    out_profile->allocation.maximum_alignment = UINT64_C(1) << 20;
    out_profile->allocation.native_byte_length_granularity = 4096;
    out_profile->host_mapping.maximum_byte_length = UINT64_C(1) << 30;
    out_profile->host_mapping.byte_offset_granularity = 1;
    out_profile->host_mapping.byte_length_granularity = 1;
    out_profile->host_mapping.supported_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryScopeQueryPairInfo(
      amdf_memory_scope_t* scope, const amdf_memory_profile_pair_query_t* query,
      amdf_memory_pair_info_t* out_info) {
    auto* self = reinterpret_cast<NativeProvider*>(scope);
    EXPECT_EQ(query->memory_profile_ordinal, 0u);
    EXPECT_EQ(query->access_count, 1u);
    if (query->access_count != 1) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    EXPECT_EQ(query->accesses[0].device,
              reinterpret_cast<amdf_device_t*>(self));
    out_info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    out_info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    out_info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL
  QueueCreate(amdf_xdna_context_t* context,
              const amdf_xdna_kernel_queue_create_info_t* info,
              amdf_kernel_queue_t** out_queue) {
    *out_queue = reinterpret_cast<amdf_kernel_queue_t*>(context);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueueInfo(amdf_kernel_queue_t* queue,
                                           amdf_kernel_queue_info_t* info) {
    info->command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA;
    info->maximum_pending_submission_count =
        reinterpret_cast<NativeProvider*>(queue)->pending_capacity;
    info->maximum_command_count =
        reinterpret_cast<NativeProvider*>(queue)->maximum_command_count;
    info->notification_types = AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD |
                               AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL
  QueryStatus(amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* status) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    status->retired_submission =
        self->checked_retired_submission.load(std::memory_order_acquire);
    status->state =
        (amdf_queue_state_t)self->checked_state.load(std::memory_order_acquire);
    status->terminal_status =
        self->checked_terminal_status.load(std::memory_order_acquire);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL
  Submit(amdf_kernel_queue_t* queue,
         const amdf_xdna_kernel_queue_submission_info_t* info,
         uint64_t* out_submission) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    const uint64_t submission_ordinal =
        self->submission_attempt_count.fetch_add(1, std::memory_order_acq_rel) +
        1;
    EXPECT_GT(info->command_count, 0u);
    EXPECT_LE(info->command_count, self->maximum_command_count);
    if (self->block_submission.load(std::memory_order_acquire)) {
      self->submission_entered.store(true, std::memory_order_release);
      iree_notification_post(&self->submission_notification, IREE_ALL_WAITERS);
      iree_notification_await(&self->submission_notification,
                              SubmissionReleased, self,
                              iree_infinite_timeout());
    }
    if (self->preceding_completion && submission_ordinal == 2) {
      uint64_t value = 0;
      IREE_EXPECT_OK(
          iree_hal_semaphore_query(self->preceding_completion, &value));
      EXPECT_EQ(value, 1u);
      EXPECT_EQ(self->live_memories, 3u);
    }
    if (submission_ordinal ==
        self->busy_submission_attempt.load(std::memory_order_acquire)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    }
    if (self->outcome == Outcome::kRejectSubmission ||
        submission_ordinal == self->reject_submission_attempt) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    {
      std::lock_guard<std::mutex> lock(self->mutex);
      for (uint32_t i = 0; i < info->command_count; ++i) {
        const auto& command = info->commands[i];
        auto* memory = reinterpret_cast<Memory*>(command.memory);
        self->AppendTimerResults(command);
        self->pending_commands.push_back(
            {memory, submission_ordinal * self->point_stride,
             command.byte_offset,
             std::vector<uint8_t>(
                 memory->contents.begin() + command.byte_offset,
                 memory->contents.begin() + command.byte_offset +
                     command.byte_length)});
      }
    }
    *out_submission = submission_ordinal * self->point_stride;
    self->last_accepted_submission.store(*out_submission,
                                         std::memory_order_release);
    self->submission_count.fetch_add(1, std::memory_order_release);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Notify(amdf_kernel_queue_t* queue,
                                        uint64_t submission,
                                        const amdf_native_event_t* event) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    ++self->notification_count;
    if (self->outcome == Outcome::kNotificationFailure) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    self->notification = *event;
    if (!self->hold_retirement) {
      self->Wake();
    }
    return AMDF_STATUS_OK;
  }

  void AppendTimerResults(const amdf_xdna_kernel_command_t& command) {
    if (!result.binding.memory || command.byte_length != 72) {
      return;
    }
    auto* memory = reinterpret_cast<Memory*>(command.memory);
    const uint8_t* bytes = memory->contents.data() + command.byte_offset;
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes + 16), 0x83u);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes + 60), 0x83u);
    auto* destination = reinterpret_cast<Memory*>(result.binding.memory);
    ASSERT_LE((result.count + 1) * 24, result.binding.byte_length);
    uint8_t* target = destination->contents.data() +
                      result.binding.byte_offset + result.count * 24;
    // Cross the low-word wrap so the consumer must combine both timer words.
    const uint64_t begin = UINT64_C(0xFFFFFFFE);
    const uint64_t end = begin + 100 + result.count;
    iree_unaligned_store_le_u32(target, iree_unaligned_load_le_u32(bytes + 24));
    iree_unaligned_store_le_u32(target + 4, begin >> 32);
    iree_unaligned_store_le_u32(target + 8, (uint32_t)begin);
    iree_unaligned_store_le_u32(target + 12,
                                iree_unaligned_load_le_u32(bytes + 68));
    iree_unaligned_store_le_u32(target + 16, end >> 32);
    iree_unaligned_store_le_u32(target + 20, (uint32_t)end);
    ++result.count;
  }

  void Wake() {
    // Borrow the queue-owned native event solely to deliver its wake hint.
    iree_async_event_native_t borrowed = {};
#if defined(IREE_ASYNC_HAVE_EVENTFD)
    borrowed.signal_primitive = iree_async_primitive_from_fd(
        static_cast<int>(notification.payload.file_descriptor));
#elif defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
    borrowed.signal_primitive = iree_async_primitive_from_win32_handle(
        reinterpret_cast<uintptr_t>(notification.payload.native_handle));
#endif
    borrowed.wait_primitive = borrowed.signal_primitive;
    iree_async_event_native_set(&borrowed);
  }

  static amdf_status_t AMDF_CALL Refresh(amdf_kernel_queue_t* queue,
                                         amdf_kernel_queue_status_t* status) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    ++self->refresh_count;
    EXPECT_GT(self->live_memories, 0u);
    if (self->outcome == Outcome::kRefreshFailure) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    const bool failed = self->outcome == Outcome::kRetiredFailure ||
                        self->outcome == Outcome::kUnretiredFailure;
    const bool inactive =
        failed || self->outcome == Outcome::kInactiveWithoutFailure;
    status->state =
        inactive ? AMDF_QUEUE_STATE_DEVICE_LOST : AMDF_QUEUE_STATE_ACTIVE;
    status->terminal_status =
        failed ? amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST)
               : AMDF_STATUS_OK;
    status->retired_submission =
        self->outcome == Outcome::kUnretiredFailure ||
                self->outcome == Outcome::kInactiveWithoutFailure ||
                (self->outcome == Outcome::kEarlyWake &&
                 self->refresh_count == 1)
            ? 0
            : std::min(self->last_accepted_submission.load(
                           std::memory_order_acquire),
                       self->retirement_limit);
    {
      std::lock_guard<std::mutex> lock(self->mutex);
      for (const auto& command : self->pending_commands) {
        EXPECT_TRUE(
            std::equal(command.bytes.begin(), command.bytes.end(),
                       command.memory->contents.begin() + command.offset));
      }
      self->pending_commands.erase(
          std::remove_if(
              self->pending_commands.begin(), self->pending_commands.end(),
              [&](const PendingCommand& command) {
                return command.submission <= status->retired_submission;
              }),
          self->pending_commands.end());
    }
    self->checked_retired_submission.store(status->retired_submission,
                                           std::memory_order_release);
    self->checked_state.store(status->state, std::memory_order_release);
    self->checked_terminal_status.store(status->terminal_status,
                                        std::memory_order_release);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Wait(amdf_kernel_queue_t* queue,
                                      uint64_t submission,
                                      uint64_t timeout_nanoseconds,
                                      uint64_t poll_duration_nanoseconds) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    ++self->wait_count;
    if (self->block_wait.load(std::memory_order_acquire)) {
      self->wait_entered.store(true, std::memory_order_release);
      iree_notification_post(&self->submission_notification, IREE_ALL_WAITERS);
      iree_notification_await(
          &self->submission_notification,
          +[](void* user_data) {
            return static_cast<NativeProvider*>(user_data)->wait_released.load(
                std::memory_order_acquire);
          },
          self, iree_infinite_timeout());
    }
    amdf_kernel_queue_status_t checked = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(checked),
    };
    const amdf_status_t status = Refresh(queue, &checked);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (checked.retired_submission >= submission ||
        !amdf_status_is_ok(checked.terminal_status)) {
      return checked.terminal_status;
    }
    (void)timeout_nanoseconds;
    (void)poll_duration_nanoseconds;
    return amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
  }
  static amdf_status_t AMDF_CALL QueueDestroy(amdf_kernel_queue_t* queue) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    ++self->queue_destroy_count;
    return self->destroy_status;
  }
  static amdf_status_t AMDF_CALL DeviceDestroy(amdf_device_t* device) {
    (void)device;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL InstanceDestroy(amdf_instance_t* instance) {
    auto* self = reinterpret_cast<NativeProvider*>(instance);
    EXPECT_EQ(self->context_destroy_count, 1u);
    EXPECT_EQ(self->live_memories, 0u);
    ++self->instance_destroy_count;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL ContextDestroy(amdf_xdna_context_t* context) {
    auto* self = reinterpret_cast<NativeProvider*>(context);
    self->context_destroy_saw_queue = self->queue_destroy_count == 1;
    ++self->context_destroy_count;
    return AMDF_STATUS_OK;
  }

  struct PendingCommand {
    // Borrowed native command storage, immutable through checked retirement.
    Memory* memory;
    // Opaque point returned when these bytes were accepted.
    uint64_t submission;
    // Instruction range start within the allocation.
    uint64_t offset;
    // Exact bytes observed at native acceptance.
    std::vector<uint8_t> bytes;
  };
  struct Flush {
    // Borrowed mapping identifying the publication target.
    Memory* memory;
    // First byte requested for publication.
    uint64_t offset;
    // Number of bytes requested for publication.
    uint64_t length;
  };
  // Exact publication attempt rejected after any previously accepted prefix.
  uint64_t reject_submission_attempt = 0;
  // Prepared provider capacity, independent of the number of commands per call.
  uint32_t pending_capacity = 1;
  // Maximum number of commands accepted as one ordered native array.
  uint32_t maximum_command_count = 1;
  // Fails cache publication when this nonzero countdown reaches zero.
  uint32_t fail_cache_publication_countdown = 0;
  // Nonunit stride verifies that the HAL does not derive ring slots from
  // points.
  uint64_t point_stride = 1;
  // Highest native point that a checked refresh may retire.
  uint64_t retirement_limit = UINT64_MAX;
  // Test controls native retirement independently of HAL readiness processing.
  bool hold_retirement = false;
  // Borrowed notification destination live through each accepted obligation.
  amdf_native_event_t notification = {};
  // Command bytes that must remain unchanged while accepted work is pending.
  std::vector<PendingCommand> pending_commands;
  // Exact cache-publication ranges issued by executable preparation.
  std::vector<Flush> flushes;
  // Serializes controlled provider vectors shared by publisher and proactor.
  std::mutex mutex;
  // Native acquisition held by the independent-progression witness.
  MemoryGate allocation_gate;
  // Native release held by the final-owner lifetime witness.
  MemoryGate release_gate;
  // Dependency instance supplied to the synchronous native instance factory.
  inline static thread_local NativeProvider* construction_instance = nullptr;
  // Explicit gate used to hold one native publication without wall-clock waits.
  iree_notification_t submission_notification;
  // True when the next native submission must wait at the explicit gate.
  std::atomic<bool> block_submission{false};
  // True after the publisher has entered the explicit gate.
  std::atomic<bool> submission_entered{false};
  // True when the publisher may leave the explicit gate.
  std::atomic<bool> submission_released{false};
  // True when a checked wait must stop at the explicit gate.
  std::atomic<bool> block_wait{false};
  // True after a checked waiter has entered the explicit gate.
  std::atomic<bool> wait_entered{false};
  // True when the checked waiter may leave the explicit gate.
  std::atomic<bool> wait_released{false};
  // Total native backing creations, including destroyed allocations.
  std::atomic<size_t> memory_create_count{0};
  // Context-private instruction allocations accepted by the provider.
  std::atomic<size_t> command_memory_create_count{0};
  // True when the context accepts only one complete instruction aperture.
  bool enforce_single_command_memory = true;
  // Native finite result association and failure injection.
  struct {
    // Active borrowed result destination.
    amdf_xdna_result_buffer_binding_t binding = {};
    // Maximum result bytes admitted by the dependency.
    uint64_t capacity = 128 * 1024;
    // Accepted measured commands appended in native order.
    uint64_t count = 0;
    // Number of attachment attempts.
    uint32_t attaches = 0;
    // Number of collection attempts.
    uint32_t collects = 0;
    // Number of removal attempts.
    uint32_t detaches = 0;
    // Native control to fail ambiguously: 1 attach, 2 collect, or 3 detach.
    uint32_t fail_control = 0;
    // Failure reported after the selected control may have made progress.
    amdf_status_t failure_status =
        amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    // Reject attachment without establishing any native binding.
    bool unsupported_attachment = false;
    // Collection supplies a malformed marker when true.
    bool corrupt = false;
  } result;
  // Common native services consumed by the real HAL.
  amdf_api_t api = {};
  // Native XDNA services consumed by the real HAL.
  amdf_xdna_api_t xdna = {};
  // Selected externally possible submission/observation outcome.
  Outcome outcome = Outcome::kSuccess;
  // Native queue teardown result, including its handle-consumption semantics.
  amdf_status_t destroy_status = AMDF_STATUS_OK;
  // Borrowed timeline that must be published before the second native submit.
  iree_hal_semaphore_t* preceding_completion = nullptr;
  // Next native allocation address, aligned for instruction storage.
  std::atomic<uint64_t> next_address{0x1000000};
  // Number of native memory resources not yet destroyed.
  std::atomic<size_t> live_memories{0};
  // Number of native submissions fully accepted by the controlled provider.
  std::atomic<uint64_t> submission_count{0};
  // Number of native submission attempts used to assign opaque points.
  std::atomic<uint64_t> submission_attempt_count{0};
  // Exact attempt on which the provider returns one transient API BUSY result.
  std::atomic<uint64_t> busy_submission_attempt{0};
  // Highest opaque native point accepted by the controlled provider.
  std::atomic<uint64_t> last_accepted_submission{0};
  // Number of one-shot wake requests.
  size_t notification_count = 0;
  // Number of checked native observation calls.
  size_t refresh_count = 0;
  // Number of checked waits issued directly by a semaphore waiter.
  size_t wait_count = 0;
  // Greatest native point established by refresh or wait.
  std::atomic<uint64_t> checked_retired_submission{0};
  // Cached queue state established by refresh or wait.
  std::atomic<uint32_t> checked_state{AMDF_QUEUE_STATE_ACTIVE};
  // Cached terminal status established by refresh or wait.
  std::atomic<uint64_t> checked_terminal_status{AMDF_STATUS_OK};
  // Number of native queue destruction attempts.
  size_t queue_destroy_count = 0;
  // Number of parent context destruction attempts.
  size_t context_destroy_count = 0;
  // Number of fabric instance destruction attempts after its native children.
  size_t instance_destroy_count = 0;
  // True when parent destruction observed terminal native queue destruction.
  bool context_destroy_saw_queue = false;
};

enum {
  IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA = 0,
  IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST = 1,
  IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT = 2,
};

inline const iree_hal_buffer_binding_layout_t* DirectBufferBindingLayout(void) {
  static const uint16_t types[] = {
      IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
      IREE_HAL_BUFFER_INTERFACE_HOST,
  };
  static const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = sizeof(iree_hal_buffer_native_binding_t) *
                     IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT,
      .binding_count = IREE_ARRAYSIZE(types),
      .host_binding_index = IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST,
      .types = types,
  };
  return &layout;
}

class QueueHarness {
 public:
  enum class CapturedResourceState {
    kRetained,
    kReleased,
  };

  ~QueueHarness() {
    ReleaseDevice();
    if (device_release_requested) {
      PollUntilDeviceDestroyed();
    }
    ReleasePool();
  }

  void Initialize(iree_allocator_t allocator = iree_allocator_system()) {
    host_allocator = allocator;
    auto options = iree_async_proactor_pool_options_default();
    options.runner = {};  // This test owns the polling thread.
    IREE_ASSERT_OK(iree_async_proactor_pool_create(1, nullptr, options,
                                                   host_allocator, &pool));
    IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 0, &proactor));
    IREE_ASSERT_OK(iree_allocator_malloc(host_allocator, sizeof(*context),
                                         reinterpret_cast<void**>(&context)));
    context->host_allocator = host_allocator;
    NativeProvider::construction_instance = &native;
    iree_status_t fabric_status =
        iree_hal_amd_fabric_create(&native.api, AMDF_NATIVE_LIFETIME_PROCESS,
                                   host_allocator, &context->fabric);
    NativeProvider::construction_instance = nullptr;
    IREE_ASSERT_OK(fabric_status);
    context->api = &native.api;
    context->xdna = &native.xdna;
    context->handle = reinterpret_cast<amdf_xdna_context_t*>(&native);
    context->target = iree::hal::amd::xdna::testing::MakeImageTarget();
    std::strcpy(context->endpoint_info.target_id,
                "amd.xdna.strix_halo.17f0_11");
    context->event_sink = {
        +[](void* user_data, const iree_hal_device_event_t* event) {
          ++static_cast<QueueHarness*>(user_data)->diagnostic_count;
        },
        this};
    context->device = reinterpret_cast<amdf_device_t*>(&native);
    context->data_source.scope =
        reinterpret_cast<amdf_memory_scope_t*>(&native);
    context->data_source.profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    context->data_source.profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE;
    context->data_source.profile.supported_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE;
    context->data_source.profile.allocation.byte_length_granularity = 4;
    context->data_source.profile.allocation.minimum_alignment = 4;
    context->data_source.profile.allocation.maximum_alignment = 32768;
    context->data_source.profile.allocation.maximum_byte_length = UINT32_MAX;
    context->data_source.capabilities.guaranteed_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    context->data_source.address_kind = AMDF_MEMORY_ADDRESS_XDNA_DMA;
    context->data_source.access.device =
        reinterpret_cast<amdf_device_t*>(&native);
    context->data_source.access.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    context->data_source.access.requirements.flags =
        AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    context->data_source.access.requirements.address_kinds =
        UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    context->command_source = context->data_source;
    context->command_source.profile.allocation.maximum_byte_length =
        8 * 1024 * 1024;
    if (native.enforce_single_command_memory) {
      context->command_source.profile.allocation.byte_length_granularity =
          8 * 1024 * 1024;
    }
    context->command_source.address_kind = AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    context->command_source.access.requirements.access =
        AMDF_MEMORY_ACCESS_READ;
    context->command_source.access.requirements.address_kinds =
        UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = pool;
    params.event_sink = context->event_sink;
    IREE_ASSERT_OK(iree_hal_amd_xdna_device_create(
        context, IREE_SV("controlled XDNA provider"), &params, host_allocator,
        &device));
    queue = iree_hal_device_queue(device, 0, 0);
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &done));
  }

  void SealDeviceGroup() {
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), host_allocator,
        &tracker));
    iree_status_t status = iree_hal_device_group_create_from_device(
        device, tracker, host_allocator, &group);
    iree_async_frontier_tracker_release(tracker);
    IREE_ASSERT_OK(status);
  }

  void Submit(
      iree_hal_semaphore_t* completion = nullptr,
      CapturedResourceState resource_state = CapturedResourceState::kRetained) {
    if (!completion) {
      completion = done;
    }
    SubmitWithWaits({}, completion, /*completion_value=*/1,
                    /*barriers=*/nullptr, resource_state);
  }

  void SubmitAfter(iree_hal_semaphore_t* dependency, uint64_t dependency_value,
                   iree_hal_semaphore_t* completion = nullptr,
                   uint64_t completion_value = 1) {
    if (!completion) {
      completion = done;
    }
    SubmitWithWaits({1, &dependency, &dependency_value}, completion,
                    completion_value);
  }

  void SubmitWithWaits(
      iree_hal_semaphore_list_t waits, iree_hal_semaphore_t* completion,
      uint64_t completion_value,
      const iree_hal_queue_barriers_t* barriers = nullptr,
      CapturedResourceState resource_state = CapturedResourceState::kRetained) {
    if (completion == done) {
      completion_timepoint.callback =
          +[](void* user_data, iree_async_semaphore_timepoint_t* timepoint,
              iree_status_t status) {
            auto* self = static_cast<QueueHarness*>(user_data);
            self->live_memories_at_completion = self->native.live_memories;
            self->completion_status = iree_status_code(status);
            iree_status_free(status);
          };
      completion_timepoint.user_data = this;
      IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
          reinterpret_cast<iree_async_semaphore_t*>(done), 1,
          &completion_timepoint));
    }
    const size_t previous_live_memories = native.live_memories;
    iree_hal_executable_t* executable = nullptr;
    iree_hal_executable_function_t function;
    ASSERT_NO_FATAL_FAILURE(LoadExecutable(
        iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
    iree_hal_buffer_t* buffer = nullptr;
    ASSERT_NO_FATAL_FAILURE(MakeBuffer(&buffer));
    auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, waits, {1, &completion, &completion_value}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
        barriers, IREE_HAL_DISPATCH_FLAG_NONE));
    iree_hal_executable_release(executable);
    iree_hal_buffer_release(buffer);
    const size_t expected_live_memories =
        resource_state == CapturedResourceState::kRetained
            ? previous_live_memories + 2
            : previous_live_memories;
    EXPECT_EQ(native.live_memories, expected_live_memories);
  }

  void LoadExecutable(
      const iree::hal::amd::xdna::testing::ImageFixture& fixture,
      iree_hal_executable_t** out_executable,
      iree_hal_executable_function_t* out_function) {
    auto bytes = fixture.Build();
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device));
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_ASSERT_OK(iree_hal_executable_load(
        iree_hal_device_queue_family(device, 0), &targets->targets[0], &params,
        out_executable));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        *out_executable, IREE_SV("main"), out_function));
  }

  void MakeBuffer(
      iree_hal_buffer_t** out_buffer,
      iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_ALL,
      iree_hal_buffer_usage_t allowed_usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                                              IREE_HAL_BUFFER_USAGE_MAPPING,
      iree_device_size_t allocation_size = 64) {
    *out_buffer = nullptr;
    // Supply the memory namespace independently of queue completion topology.
    // Tests that exercise group construction supply the real group namespace;
    // the remaining queue tests control only the completion frontier
    // themselves.
    auto* memory_family =
        const_cast<iree_hal_queue_family_t*>(iree_hal_queue_family(queue));
    if (!memory_family->memory.domain) {
      memory_family->memory.domain = this;
      memory_family->memory.queue_scope_id = 2;
    }
    DirectBufferStorage* storage = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                         reinterpret_cast<void**>(&storage)));
    memset(storage, 0, sizeof(*storage));
    storage->context = context;
    storage->host_allocator = host_allocator;
    iree_status_t status = iree_hal_amd_xdna_memory_allocate(
        context, &context->data_source, allocation_size, /*alignment=*/1,
        &storage->memory);
    if (iree_status_is_ok(status)) {
      status = iree_hal_memory_contract_create(
          memory_family->memory.domain,
          group ? iree_hal_device_group_memory_scope_count(group) : 4,
          DirectBufferBindingLayout(), host_allocator, &storage->contract);
    }
    if (iree_status_is_ok(status)) {
      storage->contract->allowed_access = allowed_access;
      storage->contract->allowed_usage = allowed_usage;
      storage->contract->host.access = allowed_access;
      storage->contract->host.modes =
          IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT;
      const auto* family = iree_hal_queue_family(queue);
      for (uint32_t id = family->memory.queue_scope_id;
           id <= family->memory.queue_scope_id + 1; ++id) {
        auto* access = &storage->contract->scopes[id];
        access->usage = allowed_usage;
        access->interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
        access->bindings[IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA] =
            IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA;
      }
      storage->bindings[IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA]
          .device_address = storage->memory.device_address;
      storage->bindings[IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST].host_pointer =
          storage->memory.storage.contents.data;
      iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                           IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                           IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      if (iree_any_bit_set(context->data_source.capabilities.guaranteed_flags,
                           AMDF_MEMORY_FLAG_HOST_COHERENT)) {
        memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
      }
      if (storage->memory.storage.host_cacheability ==
          IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
        memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
      }
      status = iree_hal_amd_fabric_buffer_wrap(
          iree_hal_buffer_placement_t{
              .device = device,
              .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          },
          context->api,
          iree_hal_amd_fabric_buffer_storage_t{
              .memory = &storage->memory.storage,
              .bindings = storage->bindings,
              .host_binding_index = IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST,
          },
          allocation_size, memory_type, allowed_access, allowed_usage,
          iree_hal_buffer_release_callback_t{
              .fn = ReleaseDirectBufferStorage,
              .user_data = storage,
          },
          host_allocator, out_buffer);
      if (iree_status_is_ok(status)) {
        (*out_buffer)->memory.contract = storage->contract;
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_xdna_memory_deinitialize(context, &storage->memory);
      iree_hal_memory_contract_release(storage->contract);
      iree_allocator_free(host_allocator, storage);
    }
    IREE_ASSERT_OK(status);
  }

  iree_status_t CreateSlabPool(iree_hal_buffer_usage_t usage,
                               iree_hal_pool_t** out_pool) {
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queue),
        .usage = usage,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    return iree_hal_slab_pool_create(group, scope, &options, host_allocator,
                                     out_pool);
  }

  iree_status_t CreateCachedTLSFPool(iree_hal_buffer_usage_t usage,
                                     iree_hal_pool_t** out_pool) {
    *out_pool = nullptr;
    iree_hal_pool_t* source_pool = nullptr;
    IREE_RETURN_IF_ERROR(CreateSlabPool(usage, &source_pool));

    iree_hal_tlsf_pool_options_t tlsf_options = {};
    tlsf_options.tlsf_options.range_length = 64 * 1024;
    tlsf_options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
    tlsf_options.tlsf_options.frontier_capacity = 64;
    iree_hal_pool_reservation_request_t backing_request = {};
    iree_status_t status = iree_hal_tlsf_pool_query_backing_request(
        source_pool, &tlsf_options, &backing_request);

    iree_hal_pool_t* cache = nullptr;
    if (iree_status_is_ok(status)) {
      iree_hal_slab_cache_options_t cache_options;
      iree_hal_slab_cache_options_initialize(&cache_options);
      cache_options.slab = backing_request;
      cache_options.max_count = 1;
      status = iree_hal_slab_cache_create(source_pool, &cache_options,
                                          host_allocator, &cache);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_tlsf_pool_create(cache, &tlsf_options, host_allocator,
                                         out_pool);
    }
    iree_hal_pool_release(cache);
    iree_hal_pool_release(source_pool);
    return status;
  }

  void PollUntilSubmitted(uint64_t count) {
    while (native.submission_count < count) {
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }

  void PollUntilNotificationCount(size_t count) {
    while (native.notification_count < count) {
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }

  void RegisterNativeObserver() {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  }

  void PollUntilDone(iree_status_code_t expected,
                     iree_hal_semaphore_t* completion = nullptr) {
    if (!completion) {
      completion = done;
    }
    while (true) {
      uint64_t value = 0;
      iree_status_t status = iree_hal_semaphore_query(completion, &value);
      if (!iree_status_is_ok(status) || value == 1) {
        if (completion == done) {
          EXPECT_EQ(expected, completion_status);
        }
        IREE_EXPECT_STATUS_IS(expected, status);
        break;
      }
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }

  void PollUntilValue(iree_hal_semaphore_t* semaphore, uint64_t target_value) {
    struct Wait {
      // Poll owner to wake when another thread publishes the semaphore.
      iree_async_proactor_t* proactor;
      // Publishes callback final access to the waiting test.
      std::atomic<bool> completed{false};
      // Owned terminal result, transferred after completed is published.
      iree_status_t status = iree_ok_status();
    } wait = {proactor};
    iree_async_semaphore_timepoint_t timepoint = {};
    timepoint.user_data = &wait;
    timepoint.callback =
        +[](void* user_data, iree_async_semaphore_timepoint_t* timepoint,
            iree_status_t status) {
          auto* wait = static_cast<Wait*>(user_data);
          auto* proactor = wait->proactor;
          wait->status = status;
          wait->completed.store(true, std::memory_order_release);
          iree_async_proactor_wake(proactor);
        };
    IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(semaphore), target_value,
        &timepoint));
    while (!wait.completed.load(std::memory_order_acquire)) {
      // A failed progress engine cannot leave a stack-owned registration live.
      IREE_CHECK_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
    IREE_ASSERT_OK(wait.status);
  }

  void ReleaseDevice() {
    if (!device) {
      return;
    }
    iree_hal_semaphore_release(done);
    done = nullptr;
    iree_hal_device_release(device);
    device = nullptr;
    iree_hal_device_group_release(group);
    group = nullptr;
    device_release_requested = true;
  }

  void PollUntilDeviceDestroyed() {
    while (!native.context_destroy_count) {
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }

  void PollUntilQueueDestroyAttempted() {
    while (!native.queue_destroy_count) {
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }

  void ReleasePool() {
    iree_async_proactor_pool_release(pool);
    pool = nullptr;
    proactor = nullptr;
  }

  struct DirectBufferStorage {
    // Borrowed native lifetime owner.
    iree_hal_amd_xdna_context_t* context;
    // Allocator owning this fixture storage.
    iree_allocator_t host_allocator;
    // Native backing with its XDNA address.
    iree_hal_amd_xdna_memory_t memory;
    // Prepared scope and binding facts, borrowed by the buffer view.
    iree_hal_memory_contract_t* contract;
    // Published native addresses in the fixture layout.
    iree_hal_buffer_native_binding_t
        bindings[IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT];
  };

  static void ReleaseDirectBufferStorage(void* user_data,
                                         iree_hal_buffer_t* buffer) {
    (void)buffer;
    auto* storage = static_cast<DirectBufferStorage*>(user_data);
    iree_hal_amd_xdna_memory_deinitialize(storage->context, &storage->memory);
    iree_hal_memory_contract_release(storage->contract);
    iree_allocator_free(storage->host_allocator, storage);
  }

  // Mocked native dependency shared with the native allocation worker.
  NativeProvider native;
  // Allocator used by the production device and shared proactor.
  iree_allocator_t host_allocator = iree_allocator_system();
  // Pool retaining the real platform proactor without a polling runner.
  iree_async_proactor_pool_t* pool = nullptr;
  // Proactor borrowed from the pool and shared with the HAL device.
  iree_async_proactor_t* proactor = nullptr;
  // Owned production HAL device.
  iree_hal_device_t* device = nullptr;
  // Borrowed native context owned by |device|.
  iree_hal_amd_xdna_context_t* context = nullptr;
  // Optional sealed group assigning exact memory scopes to |device|.
  iree_hal_device_group_t* group = nullptr;
  // Queue borrowed from the production HAL device.
  iree_hal_queue_t* queue = nullptr;
  // Owned completion timeline.
  iree_hal_semaphore_t* done = nullptr;
  // Observes ownership synchronously when the HAL publishes completion.
  iree_async_semaphore_timepoint_t completion_timepoint = {};
  // Captured terminal status, checked alongside the public semaphore result.
  iree_status_code_t completion_status = IREE_STATUS_UNKNOWN;
  // Native resources still owned at the instant completion becomes observable.
  size_t live_memories_at_completion = SIZE_MAX;
  // Number of published driver failure diagnostics.
  size_t diagnostic_count = 0;
  // True after the public device reference has entered terminal shutdown.
  bool device_release_requested = false;
};

}  // namespace iree::hal::amd::xdna::testing

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_TESTING_QUEUE_TEST_UTIL_H_
