// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "iree/async/event.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/xdna/device.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/hal/drivers/amd/xdna/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#if defined(IREE_ASYNC_HAVE_EVENTFD) || defined(IREE_ASYNC_HAVE_WIN32_HANDLE)

enum class Outcome {
  kSuccess,
  kRejectSubmission,
  kRetiredFailure,
  kNotificationFailure,
  kRefreshFailure,
  kUnretiredFailure,
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
    ++self->submission_count;
    EXPECT_EQ(info->command_count, 1u);
    if (self->preceding_completion && self->submission_count == 2) {
      uint64_t value = 0;
      IREE_EXPECT_OK(
          iree_hal_semaphore_query(self->preceding_completion, &value));
      EXPECT_EQ(value, 1u);
      EXPECT_EQ(self->live_memories, 3u);
    }
    if (self->outcome == Outcome::kRejectSubmission) {
      return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    const auto& command = info->commands[0];
    auto* memory = reinterpret_cast<Memory*>(command.memory);
    self->pending_commands.push_back(
        {memory, self->submission_count * self->point_stride,
         command.byte_offset,
         std::vector<uint8_t>(memory->contents.begin() + command.byte_offset,
                              memory->contents.begin() + command.byte_offset +
                                  command.byte_length)});
    *out_submission = self->submission_count * self->point_stride;
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
    status->state =
        failed ? AMDF_QUEUE_STATE_DEVICE_LOST : AMDF_QUEUE_STATE_ACTIVE;
    status->terminal_status =
        failed ? amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST)
               : AMDF_STATUS_OK;
    status->retired_submission =
        self->outcome == Outcome::kUnretiredFailure ||
                (self->outcome == Outcome::kEarlyWake &&
                 self->refresh_count == 1)
            ? 0
            : std::min(self->submission_count * self->point_stride,
                       self->retirement_limit);
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
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueueDestroy(amdf_kernel_queue_t* queue) {
    auto* self = reinterpret_cast<NativeProvider*>(queue);
    ++self->queue_destroy_count;
    return self->destroy_status;
  }
  static amdf_status_t AMDF_CALL ContextDestroy(amdf_xdna_context_t* context) {
    ++reinterpret_cast<NativeProvider*>(context)->context_destroy_count;
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
  // Number of native submission calls.
  uint64_t submission_count = 0;
  // Number of one-shot wake requests.
  size_t notification_count = 0;
  // Number of checked native observation calls.
  size_t refresh_count = 0;
  // Number of native queue destruction attempts.
  size_t queue_destroy_count = 0;
  // Number of parent context destruction attempts.
  size_t context_destroy_count = 0;
};

class QueueHarness {
 public:
  ~QueueHarness() {
    ReleaseDevice();
    iree_async_proactor_pool_release(pool);
  }

  void Initialize() {
    auto options = iree_async_proactor_pool_options_default();
    options.runner = {};  // This test owns the polling thread.
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, options, iree_allocator_system(), &pool));
    IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 0, &proactor));
    iree_hal_amd_xdna_context_t* context = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*context),
                                         reinterpret_cast<void**>(&context)));
    context->host_allocator = iree_allocator_system();
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
        context, IREE_SV("controlled XDNA provider"), &params,
        iree_allocator_system(), &device));
    queue = iree_hal_device_queue(device, 0, 0);
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &done));
  }

  void Submit(iree_hal_semaphore_t* completion = nullptr) {
    if (!completion) {
      completion = done;
    }
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
    uint64_t value = 1;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, {}, {1, &completion, &value}, executable, function,
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

  void ReleaseDevice() {
    iree_hal_semaphore_release(done);
    done = nullptr;
    iree_hal_device_release(device);
    device = nullptr;
  }

  // Mocked native dependency; all access runs on this test's polling thread.
  NativeProvider native;
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
};

TEST(XdnaQueueTest, DeviceCreatesOwnedHostCompatibleSemaphores) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  EXPECT_TRUE(iree_hal_amd_xdna_semaphore_isa(harness.done));
  EXPECT_TRUE(
      iree_hal_amd_xdna_semaphore_is_local(harness.done, harness.device));
  EXPECT_EQ(iree_hal_device_query_semaphore_compatibility(harness.device,
                                                          harness.done),
            IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY);
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
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
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
  while (harness.device) {
    IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                            iree_infinite_timeout(), nullptr));
  }
  EXPECT_EQ(harness.completion_status, IREE_STATUS_OK);
  EXPECT_EQ(harness.live_memories_at_completion, 0u);
  EXPECT_EQ(harness.native.live_memories, 0u);
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 1u);
  EXPECT_EQ(harness.diagnostic_count, 0u);
  iree_hal_semaphore_release(first);
}

TEST(XdnaQueueTest, ConsumingCleanupFailureReleasesParent) {
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  harness.native.destroy_status =
      amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  harness.ReleaseDevice();
  EXPECT_EQ(harness.native.queue_destroy_count, 1u);
  EXPECT_EQ(harness.native.context_destroy_count, 1u);
  EXPECT_EQ(harness.diagnostic_count, 1u);
}

// These paths deliberately preserve live native ownership until process exit.
// Child processes verify the retained resources and use _Exit so deliberate
// retention does not become an accidental LeakSanitizer failure at exit.
TEST(XdnaQueueDeathTest, ObserverFailuresPreserveAcceptedOwnership) {
  for (Outcome outcome :
       {Outcome::kNotificationFailure, Outcome::kRefreshFailure,
        Outcome::kUnretiredFailure}) {
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
          EXPECT_GE(harness.diagnostic_count, 2u);
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
        EXPECT_EQ(harness.native.queue_destroy_count, 1u);
        EXPECT_EQ(harness.native.context_destroy_count, 0u);
        EXPECT_EQ(harness.diagnostic_count, 1u);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

#endif  // IREE_ASYNC_HAVE_EVENTFD || IREE_ASYNC_HAVE_WIN32_HANDLE

}  // namespace
