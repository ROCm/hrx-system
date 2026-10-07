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
#include "iree/hal/drivers/amd/xdna/device.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#if defined(IREE_ASYNC_HAVE_EVENTFD) || defined(IREE_ASYNC_HAVE_WIN32_HANDLE)

struct HostAllocationCounters {
  std::atomic<uint64_t> allocations{0};
  std::atomic<uint64_t> reallocations{0};
  std::atomic<uint64_t> frees{0};
  std::atomic<int64_t> live_allocations{0};
};

struct HostAllocationSnapshot {
  uint64_t allocations;
  uint64_t reallocations;
  uint64_t frees;
};

iree_status_t TrackingAllocatorControl(void* self,
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

iree_allocator_t TrackingAllocator(HostAllocationCounters* counters) {
  return {counters, TrackingAllocatorControl};
}

HostAllocationSnapshot SnapshotHostAllocations(
    const HostAllocationCounters& counters) {
  return {
      counters.allocations.load(),
      counters.reallocations.load(),
      counters.frees.load(),
  };
}

void ExpectHostAllocationsUnchanged(const HostAllocationSnapshot& before,
                                    const HostAllocationCounters& counters) {
  const HostAllocationSnapshot after = SnapshotHostAllocations(counters);
  EXPECT_EQ(after.allocations, before.allocations);
  EXPECT_EQ(after.reallocations, before.reallocations);
  EXPECT_EQ(after.frees, before.frees);
}

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
    api.memory_create = MemoryCreate;
    api.memory_destroy = MemoryDestroy;
    api.memory_map = MemoryMap;
    api.host_mapping_destroy = MappingDestroy;
    api.host_mapping_query_info = MappingInfo;
    api.host_mapping_cache_control = CacheControl;
    api.memory_query_address = MemoryAddress;
    api.kernel_queue_query_info = QueueInfo;
    api.kernel_queue_request_notification = Notify;
    api.kernel_queue_refresh_status = Refresh;
    api.kernel_queue_destroy = QueueDestroy;
    xdna.kernel_queue_create = QueueCreate;
    xdna.kernel_queue_submit = Submit;
    xdna.context_destroy = ContextDestroy;
  }

  ~NativeProvider() {
    iree_notification_deinitialize(&submission_notification);
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

  size_t FlushCount() {
    std::lock_guard<std::mutex> lock(mutex);
    return flushes.size();
  }

  static amdf_status_t AMDF_CALL MemoryCreate(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
      amdf_memory_t** out_memory) {
    auto* self = reinterpret_cast<NativeProvider*>(scope);
    auto* memory = new Memory{self, std::vector<uint8_t>(info->byte_length),
                              self->next_address};
    self->next_address += iree_host_align(info->byte_length, 32768);
    ++self->live_memories;
    ++self->memory_create_count;
    *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryDestroy(amdf_memory_t* handle) {
    auto* memory = reinterpret_cast<Memory*>(handle);
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
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MemoryAddress(amdf_memory_t* handle,
                                               uint32_t access_ordinal,
                                               amdf_memory_address_kind_t kind,
                                               uint64_t* out_address) {
    *out_address = reinterpret_cast<Memory*>(handle)->address;
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
    info->maximum_command_count = 1;
    info->notification_types = AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD |
                               AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
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
    EXPECT_EQ(info->command_count, 1u);
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
    if (self->outcome == Outcome::kRejectSubmission) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    const auto& command = info->commands[0];
    auto* memory = reinterpret_cast<Memory*>(command.memory);
    {
      std::lock_guard<std::mutex> lock(self->mutex);
      self->pending_commands.push_back(
          {memory, submission_ordinal * self->point_stride, command.byte_offset,
           std::vector<uint8_t>(memory->contents.begin() + command.byte_offset,
                                memory->contents.begin() + command.byte_offset +
                                    command.byte_length)});
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
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueueDestroy(amdf_kernel_queue_t* queue) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    ++self->queue_destroy_count;
    return self->destroy_status;
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
  // Prepared provider capacity, independent of the number of commands per call.
  uint32_t pending_capacity = 1;
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
  // Explicit gate used to hold one native publication without wall-clock waits.
  iree_notification_t submission_notification;
  // True when the next native submission must wait at the explicit gate.
  std::atomic<bool> block_submission{false};
  // True after the publisher has entered the explicit gate.
  std::atomic<bool> submission_entered{false};
  // True when the publisher may leave the explicit gate.
  std::atomic<bool> submission_released{false};
  // Total native backing creations, including destroyed allocations.
  size_t memory_create_count = 0;
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
  uint64_t next_address = 0x1000000;
  // Number of native memory resources not yet destroyed.
  size_t live_memories = 0;
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
  // Number of native queue destruction attempts.
  size_t queue_destroy_count = 0;
  // Number of parent context destruction attempts.
  size_t context_destroy_count = 0;
  // True when parent destruction observed terminal native queue destruction.
  bool context_destroy_saw_queue = false;
};

class QueueHarness {
 public:
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
    iree_hal_amd_xdna_context_t* context = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(host_allocator, sizeof(*context),
                                         reinterpret_cast<void**>(&context)));
    context->host_allocator = host_allocator;
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

  void Submit(iree_hal_semaphore_t* completion = nullptr) {
    if (!completion) {
      completion = done;
    }
    SubmitWithWaits({}, completion, /*completion_value=*/1);
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

  void SubmitWithWaits(iree_hal_semaphore_list_t waits,
                       iree_hal_semaphore_t* completion,
                       uint64_t completion_value) {
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
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, 0));
    iree_hal_executable_release(executable);
    iree_hal_buffer_release(buffer);
    EXPECT_EQ(native.live_memories, previous_live_memories + 3);
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

  void MakeBuffer(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, 64, out_buffer));
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
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
    while (value < target_value) {
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
      IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
    }
  }

  void ReleaseDevice() {
    if (!device) {
      return;
    }
    iree_hal_semaphore_release(done);
    done = nullptr;
    iree_hal_device_release(device);
    device = nullptr;
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

  // Mocked native dependency; all access runs on this test's polling thread.
  NativeProvider native;
  // Allocator used by the production device and shared proactor.
  iree_allocator_t host_allocator = iree_allocator_system();
  // Pool retaining the real platform proactor without a polling runner.
  iree_async_proactor_pool_t* pool = nullptr;
  // Proactor borrowed from the pool and shared with the HAL device.
  iree_async_proactor_t* proactor = nullptr;
  // Owned production HAL device.
  iree_hal_device_t* device = nullptr;
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

TEST(XdnaQueueTest, ReadyDispatchPublishesBeforeProactorProgress) {
  QueueHarness harness;
  harness.native.hold_retirement = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());

  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  EXPECT_EQ(harness.native.submission_count, 1u);
  EXPECT_EQ(harness.native.notification_count, 0u);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &value));
  EXPECT_EQ(value, 0u);

  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
  EXPECT_EQ(harness.native.notification_count, 1u);
  harness.native.hold_retirement = false;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
}

TEST(XdnaQueueTest, InexactReachedFrontierUsesQueuedPublication) {
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
  EXPECT_EQ(harness.native.submission_count, 0u);
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_EQ(harness.native.submission_count, 1u);
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
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
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
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
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
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
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

  ASSERT_NO_FATAL_FAILURE(harness.Submit());
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
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
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
  IREE_ASSERT_OK(iree_hal_semaphore_signal(
      gate, 1, iree_async_single_frontier_as_const_frontier(&host_frontier)));

  harness.native.ReturnBusyOnNextSubmission();
  harness.native.BlockSubmission();
  ASSERT_NO_FATAL_FAILURE(harness.SubmitAfter(gate, 1));
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
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, 0));
    first_submission_returned.store(true, std::memory_order_release);
  });
  harness.native.AwaitBlockedSubmission();
  EXPECT_EQ(harness.native.submission_attempt_count, 1u);
  iree_status_t second_status = iree_hal_queue_dispatch(
      harness.queue, {}, {1, &completions[1], &value}, executable, function,
      iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, 0);
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
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(harness.LoadExecutable(
      iree::hal::amd::xdna::testing::ImageFixture(), &executable, &function));
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  std::array<iree_hal_semaphore_t*, 2> completions = {};
  for (size_t i = 0; i < buffers.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffers[i]));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &completions[i]));
  }
  // Two initial executable allocations and two caller buffers.
  EXPECT_EQ(harness.native.memory_create_count, 4u);
  for (uint64_t iteration = 1; iteration <= 4; ++iteration) {
    harness.native.hold_retirement = true;
    harness.native.flushes.clear();
    for (size_t i = 0; i < buffers.size(); ++i) {
      auto binding =
          iree_hal_make_buffer_ref(buffers[(i + iteration) % 2], 0, 64);
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          harness.queue, {}, {1, &completions[i], &iteration}, executable,
          function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
          {1, &binding}, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(iteration * 2));
    ASSERT_EQ(harness.native.pending_commands.size(), 2u);
    const auto& first = harness.native.pending_commands[0];
    const auto& second = harness.native.pending_commands[1];
    EXPECT_NE(first.memory, second.memory);
    EXPECT_FALSE(std::equal(first.bytes.begin() + 8, first.bytes.end(),
                            second.bytes.begin() + 8));
    // The static DMA catalog is shared, so only one command allocation grows.
    EXPECT_EQ(harness.native.memory_create_count, 5u);
    if (iteration > 1) {
      ASSERT_EQ(harness.native.flushes.size(), 2u);
      for (const auto& flush : harness.native.flushes) {
        EXPECT_EQ(flush.offset, 8u);
        EXPECT_EQ(flush.length, 8u);
      }
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
  for (auto* completion : completions) {
    iree_hal_semaphore_release(completion);
  }
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_executable_release(executable);
  EXPECT_EQ(harness.native.live_memories, 0u);
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
  EXPECT_EQ(harness.native.live_memories, 6u);
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
  EXPECT_EQ(harness.native.live_memories, 0u);
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
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, 0));
  }
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(2));
  EXPECT_EQ(harness.native.memory_create_count, 9u);
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
  EXPECT_EQ(harness.native.live_memories, 0u);
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
  iree_hal_transfer_operation_t transfer = {};
  transfer.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  transfer.update.target_buffer = buffer;
  transfer.update.source_buffer = &payload;
  transfer.update.length = sizeof(payload);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_transfer(harness.queue, {},
                                         {1, &produced, &value}, 1, &transfer));
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
  iree_hal_transfer_operation_t transfer = {};
  transfer.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  transfer.update.target_buffer = buffer;
  transfer.update.source_buffer = &payload;
  transfer.update.length = sizeof(payload);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      harness.queue, {}, {1, &host_completion, &value}, 1, &transfer));

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
  iree_hal_buffer_t* dispatch_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&dispatch_buffer));
  const iree_hal_buffer_ref_t binding =
      iree_hal_make_buffer_ref(dispatch_buffer, 0, 64);

  iree_hal_semaphore_t* ready_completion = nullptr;
  iree_hal_semaphore_t* reached_gate = nullptr;
  iree_hal_semaphore_t* wait_gate = nullptr;
  iree_hal_semaphore_t* wait_completion = nullptr;
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
        {1, &binding}, IREE_HAL_DISPATCH_FLAG_NONE));
    harness.PollUntilValue(ready_completion, iteration);
  }
  check_steady_state("ready dispatch", before, native_allocations_before);

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
        IREE_HAL_DISPATCH_FLAG_NONE));
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
          IREE_HAL_DISPATCH_FLAG_NONE));
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
      {1, &binding}, IREE_HAL_DISPATCH_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(submission_count + 1));
  ASSERT_NO_FATAL_FAILURE(
      harness.PollUntilNotificationCount(notification_count + 1));
  harness.native.ReturnBusyOnNextSubmission();
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      harness.queue, {}, {1, &capacity_completions[1], &retry_value},
      executable, function, iree_hal_make_static_dispatch_config(1, 1, 1), {},
      {1, &binding}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                          iree_infinite_timeout(), nullptr));
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
  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_hal_buffer_t* transfer_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(harness.device), buffer_params,
      kLargeUpdateLength, &transfer_buffer));
  std::array<uint8_t, 64> small_update;
  small_update.fill(0x5A);
  std::vector<uint8_t> large_update(kLargeUpdateLength, 0xA5);

  auto run_updates = [&](iree_const_byte_span_t source,
                         iree_hal_semaphore_t* completion) {
    iree_hal_transfer_operation_t transfer = {};
    transfer.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
    transfer.update.source_buffer = source.data;
    transfer.update.target_buffer = transfer_buffer;
    transfer.update.length = source.data_length;
    for (uint64_t iteration = 1; iteration <= kTotalIterations; ++iteration) {
      if (iteration == kWarmupIterations + 1) {
        before = SnapshotHostAllocations(allocation_counters);
        native_allocations_before = harness.native.memory_create_count;
      }
      IREE_ASSERT_OK(iree_hal_queue_transfer(
          harness.queue, {}, {1, &completion, &iteration}, 1, &transfer));
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
  iree_hal_semaphore_release(wait_gate);
  iree_hal_semaphore_release(reached_gate);
  iree_hal_semaphore_release(ready_completion);
  iree_hal_buffer_release(dispatch_buffer);
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
      {1, &binding}, 0));
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
  EXPECT_EQ(harness.live_memories_at_completion, 0u);
  EXPECT_EQ(harness.native.live_memories, 0u);
  EXPECT_EQ(harness.native.notification_count, 0u);
}

TEST(XdnaQueueTest, RetiredFailureReleasesCaptureBeforePublishingFailure) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.outcome = Outcome::kRetiredFailure;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_DATA_LOSS));
  EXPECT_EQ(harness.live_memories_at_completion, 0u);
  EXPECT_EQ(harness.native.live_memories, 0u);
  EXPECT_EQ(harness.native.refresh_count, 1u);
}

TEST(XdnaQueueTest, EarlyWakeRearmsUntilCheckedRetirement) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.outcome = Outcome::kEarlyWake;
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  EXPECT_EQ(harness.live_memories_at_completion, 0u);
  EXPECT_EQ(harness.native.live_memories, 0u);
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
  EXPECT_EQ(harness.native.live_memories, 0u);
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
  iree_async_semaphore_timepoint_t release_timepoint = {};
  release_timepoint.callback =
      +[](void* user_data, iree_async_semaphore_timepoint_t* timepoint,
          iree_status_t status) {
        IREE_EXPECT_OK(status);
        static_cast<QueueHarness*>(user_data)->ReleaseDevice();
      };
  release_timepoint.user_data = &harness;
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(harness.done), 1,
      &release_timepoint));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDeviceDestroyed());
  EXPECT_EQ(harness.completion_status, IREE_STATUS_OK);
  EXPECT_EQ(harness.live_memories_at_completion, 0u);
  EXPECT_EQ(harness.native.live_memories, 0u);
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 1u);
  EXPECT_TRUE(harness.native.context_destroy_saw_queue);
  EXPECT_EQ(harness.diagnostic_count, 0u);
  iree_hal_semaphore_release(first);
  harness.ReleasePool();
  EXPECT_EQ(allocation_counters.live_allocations.load(), 0);
}

TEST(XdnaQueueTest, ConsumingCleanupFailureReleasesParent) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.destroy_status =
      amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  harness.ReleaseDevice();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDeviceDestroyed());
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 1u);
  EXPECT_TRUE(harness.native.context_destroy_saw_queue);
  EXPECT_EQ(harness.diagnostic_count, 1u);
}

// These paths deliberately preserve live native ownership until process exit.
// Child processes verify the retained resources and use _Exit so deliberate
// retention does not become an accidental LeakSanitizer failure at exit.
TEST(XdnaQueueDeathTest, ObserverFailuresPreserveAcceptedOwnership) {
  for (Outcome outcome :
       {Outcome::kNotificationFailure, Outcome::kRefreshFailure,
        Outcome::kUnretiredFailure, Outcome::kInactiveWithoutFailure}) {
    SCOPED_TRACE(static_cast<int>(outcome));
    EXPECT_EXIT(
        {
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
        },
        ::testing::ExitedWithCode(0), "");
  }
}

TEST(XdnaQueueDeathTest, ObserverFailurePreservesEveryPendingInvocation) {
  EXPECT_EXIT(
      {
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
        ASSERT_NO_FATAL_FAILURE(
            harness.PollUntilDone(IREE_STATUS_INTERNAL, second));
        EXPECT_EQ(harness.native.live_memories, 6u);
        harness.ReleaseDevice();
        iree_hal_semaphore_release(second);
        EXPECT_EQ(harness.native.queue_destroy_count, 0u);
        EXPECT_EQ(harness.native.context_destroy_count, 0u);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(XdnaQueueDeathTest, BusyNativeDestroyPreservesParent) {
  EXPECT_EXIT(
      {
        QueueHarness harness;
        ASSERT_NO_FATAL_FAILURE(harness.Initialize());
        harness.native.destroy_status =
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
        harness.ReleaseDevice();
        ASSERT_NO_FATAL_FAILURE(harness.PollUntilQueueDestroyAttempted());
        EXPECT_EQ(harness.native.queue_destroy_count, 1u);
        EXPECT_EQ(harness.native.context_destroy_count, 0u);
        EXPECT_EQ(harness.diagnostic_count, 1u);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

#endif  // IREE_ASYNC_HAVE_EVENTFD || IREE_ASYNC_HAVE_WIN32_HANDLE

}  // namespace
