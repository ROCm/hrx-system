// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <cstring>
#include <vector>

#include "iree/async/event.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/xdna/device.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
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
    info->maximum_pending_submission_count = 1;
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
    *out_submission = self->submission_count;
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
    // Borrow the queue-owned native event solely to deliver this wake hint.
    iree_async_event_native_t borrowed = {};
#if defined(IREE_ASYNC_HAVE_EVENTFD)
    borrowed.signal_primitive = iree_async_primitive_from_fd(
        static_cast<int>(event->payload.file_descriptor));
#elif defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
    borrowed.signal_primitive = iree_async_primitive_from_win32_handle(
        reinterpret_cast<uintptr_t>(event->payload.native_handle));
#endif
    borrowed.wait_primitive = borrowed.signal_primitive;
    iree_async_event_native_set(&borrowed);
    return AMDF_STATUS_OK;
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
            : self->submission_count;
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
    auto bytes = iree::hal::amd::xdna::testing::ImageFixture().Build();
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device));
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    iree_hal_executable_t* executable = nullptr;
    IREE_ASSERT_OK(
        iree_hal_executable_load(iree_hal_device_queue_family(device, 0),
                                 &targets->targets[0], &params, &executable));
    iree_hal_executable_function_t function;
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable, IREE_SV("main"), &function));
    iree_hal_buffer_t* buffer = nullptr;
    iree_hal_buffer_params_t buffer_params = {};
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), buffer_params, 64, &buffer));
    auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
    uint64_t value = 1;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, {}, {1, &completion, &value}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, 0));
    iree_hal_executable_release(executable);
    iree_hal_buffer_release(buffer);
    EXPECT_EQ(native.live_memories, previous_live_memories + 3);
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
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
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
