// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/drivers/amd/gpu/queue/file_transfer.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::fabric::FakeNative;
using iree::hal::amd::fabric::MemoryBackendTest;
using iree::hal::cts::Ref;

// Real file I/O, staging, allocation, and SDMA publication share the production
// ownership path. Native dependencies provide coherent entry/exit boundaries
// and controlled libamdf retirement. Accepted commands can still access their
// operands after notification fails, until the explicit native stop.
class FileTransferStagingTest : public MemoryBackendTest,
                                public ::testing::WithParamInterface<bool> {
 protected:
  struct BoundaryQueue : iree_hal_queue_t {
    // File operations borrowing this queue until all captured uses retire.
    uint32_t host_owners = 0;
    // Visibility boundaries completed by this coherent dependency.
    uint32_t boundaries = 0;
  };

  static FileTransferStagingTest* FromQueue(amdf_kernel_queue_t* queue) {
    return reinterpret_cast<FileTransferStagingTest*>(queue);
  }

  void SetUp() override {
    native_.queue_family_ordinals[0] = 0;
    ASSERT_NO_FATAL_FAILURE(MemoryBackendTest::SetUp());
    native_.user_data = this;
    native_.required_flags = 0;
    native_.host_coherent = true;
    native_.host_mappable = false;
    native_.host_visibility_guaranteed = false;
    native_.api.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                          amdf_memory_address_kind_t kind,
                                          uint64_t* address) {
      EXPECT_EQ(kind, AMDF_MEMORY_ADDRESS_GPU);
      *address = reinterpret_cast<uintptr_t>(
          reinterpret_cast<FakeNative::Memory*>(memory)->data);
      return AMDF_STATUS_OK;
    };
    native_.api.memory_query_pair_info = [](const amdf_memory_site_t* producer,
                                            const amdf_memory_site_t* consumer,
                                            amdf_memory_pair_info_t* info) {
      EXPECT_TRUE(producer->kind == AMDF_MEMORY_SITE_KIND_DEVICE ||
                  producer->kind == AMDF_MEMORY_SITE_KIND_FILE_IO);
      EXPECT_TRUE(consumer->kind == AMDF_MEMORY_SITE_KIND_DEVICE ||
                  consumer->kind == AMDF_MEMORY_SITE_KIND_FILE_IO);
      info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
      info->release.kind = info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
      return AMDF_STATUS_OK;
    };
    for (uint32_t i = 0; i < families_.size(); ++i) {
      binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                  AMDF_MEMORY_ADDRESS_GPU};
      native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
      native_families_[i].host_usage = 0;
      native_families_[i].copy_family_ordinal = 0;
      native_.copy_family_ordinals[i] = 0;
    }
    IREE_ASSERT_OK(CreatePool(
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        pool_.out(), UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        IREE_HAL_MEMORY_ACCESS_NONE));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(8192, 4096),
        iree_infinite_timeout(), buffer_.out()));
    const auto view = iree_hal_buffer_memory_view(buffer_);
    const auto* facts =
        reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
            view.bindings + view.contract->binding_layout.binding_count);
    ASSERT_EQ(facts->memory->mapping, nullptr);
    buffer_contents_ = static_cast<uint8_t*>(
        reinterpret_cast<FakeNative::Memory*>(facts->memory->handle)->data);

    ASSERT_NO_FATAL_FAILURE(InitializeNativeQueue());
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator_, &proactor_));
    IREE_ASSERT_OK(iree_hal_host_semaphore_create(proactor_, 0, allocator_,
                                                  completion_.out()));
    native_.host_mappable = native_.host_visibility_guaranteed = true;
    native_.file_io_enabled = true;
    native_.host_mapping_features = AMDF_HOST_MAPPING_FEATURE_FILE_IO;
    amdf_memory_profile_t profile;
    bool available = false;
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_staging_query_profile(
        &context_, &profile, &available));
    ASSERT_TRUE(available);
    const amdf_memory_device_access_t access = {
        .device = native_.devices[0],
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
            },
    };
    const iree_hal_amd_fabric_memory_options_t control = {
        .scope = native_.scope(),
        .profile = profile,
        .access_count = 1,
        .accesses = &access,
        .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_queue_allocate(
        &context_, 0, AMDF_QUEUE_PUBLICATION_MODE_KERNEL, &control,
        &publication_, 0, allocator_, &sdma_));
    native_.required_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_FILE_IO;
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_staging_allocate(
        devices_[0], &context_, 0, &profile, sdma_, proactor_, 2, 4096,
        UINT32_MAX, allocator_, &staging_));
    iree_async_semaphore_wait_context_initialize(&waits_);
    iree_hal_amd_gpu_capture_pool_initialize(4096, allocator_, &captures_);
    queue_.queue_family = families_[0];
    executor_ = {
        .queue = &queue_,
        .context = &context_,
        .captures = &captures_,
        .begin =
            [](iree_hal_queue_t* queue) {
              ++static_cast<BoundaryQueue*>(queue)->host_owners;
            },
        .end =
            [](iree_hal_queue_t* queue) {
              --static_cast<BoundaryQueue*>(queue)->host_owners;
            },
        .staging = staging_,
        .waits = &waits_,
        .native_family_ordinal = 0,
        .barrier_size = sizeof(uint32_t),
        .add_effects =
            [](iree_hal_queue_t*, iree_hal_memory_effects_t effects,
               void* barrier) {
              *static_cast<uint32_t*>(barrier) |= effects.bits;
            },
        .submit =
            [](iree_hal_queue_t* queue,
               const iree_hal_transfer_operation_t* operation, const void*,
               const void*, iree_hal_amd_gpu_retirement_callback_t callback) {
              EXPECT_EQ(operation,
                        nullptr);  // All payloads must pass through SDMA.
              auto* boundary = static_cast<BoundaryQueue*>(queue);
              iree_async_single_frontier_t frontier;
              iree_async_single_frontier_initialize(&frontier, 7,
                                                    ++boundary->boundaries);
              callback.fn(callback.user_data, iree_ok_status(), &frontier,
                          true);
              return iree_ok_status();
            },
    };
  }

  void InitializeNativeQueue() {
    gpu_.kernel_queue_create = [](amdf_device_t* device,
                                  const amdf_gpu_kernel_queue_create_info_t*,
                                  amdf_kernel_queue_t** queue) {
      const auto* native_device = reinterpret_cast<FakeNative::Device*>(device);
      *queue = reinterpret_cast<amdf_kernel_queue_t*>(
          native_device->owner->user_data);
      return AMDF_STATUS_OK;
    };
    native_.api.kernel_queue_query_info = [](amdf_kernel_queue_t*,
                                             amdf_kernel_queue_info_t* info) {
      info->maximum_pending_submission_count = 64;
      info->notification_types = UINT64_C(1) << AMDF_NATIVE_EVENT_TYPE_EVENTFD;
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_submit =
        [](amdf_kernel_queue_t* queue,
           const amdf_gpu_kernel_queue_submission_info_t* info,
           uint64_t* point) {
          auto* self = FromQueue(queue);
          EXPECT_FALSE(self->native_stopped_);
          EXPECT_EQ(info->command_count, 1u);
          const auto& command = info->commands[0];
          const auto* memory =
              reinterpret_cast<FakeNative::Memory*>(command.memory);
          const auto* words = reinterpret_cast<const uint32_t*>(
              static_cast<const uint8_t*>(memory->data) + command.byte_offset);
          self->commands_.emplace_back(words, words + command.byte_length / 4);
          *point = ++self->submitted_;
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_request_notification =
        [](amdf_kernel_queue_t* queue, uint64_t, const amdf_native_event_t*) {
          return FromQueue(queue)->notification_status_;
        };
    native_.api.kernel_queue_refresh_status =
        [](amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* status) {
          status->retired_submission = FromQueue(queue)->retired_;
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_wait = [](amdf_kernel_queue_t* queue,
                                       uint64_t point, uint64_t, uint64_t) {
      return point <= FromQueue(queue)->retired_
                 ? AMDF_STATUS_OK
                 : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    };
    gpu_.kernel_queue_inactivate = [](amdf_kernel_queue_t* queue) {
      FromQueue(queue)->native_stopped_ = true;
      return AMDF_STATUS_OK;
    };
    native_.api.kernel_queue_destroy = [](amdf_kernel_queue_t*) {
      return AMDF_STATUS_OK;
    };
    IREE_ASSERT_OK(iree_allocator_malloc(allocator_, sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = allocator_;
    fabric_->api = &native_.api;
    fabric_->system_scope = native_.scope();
    context_.fabric = fabric_;
    context_.api = &native_.api;
    context_.gpu = &gpu_;
    context_.device = native_.devices[0];
    context_.families = &family_;
    context_.kernel_notification.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    context_.event_sink = {
        [](void* user_data, const iree_hal_device_event_t* event) {
          auto* self = static_cast<FileTransferStagingTest*>(user_data);
          // File-owner diagnostics are distinct from native queue diagnostics.
          if (event->source.device) {
            ++self->file_failures_;
          }
        },
        this};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    iree_notification_initialize(&publication_.notification);
  }

  void TearDown() override {
    buffer_.reset();
    file_.reset();
    if (sdma_ && queue_.host_owners) {
      PollCopies(amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
      StopCopies();
      PumpUntil([&] { return queue_.host_owners == 0; });
    }
    pool_.reset();
    if (staging_) {
      std::atomic<bool> finished = false;
      std::thread cleanup([&] {
        EXPECT_TRUE(iree_hal_amd_gpu_file_staging_free(staging_));
        finished.store(true);
        iree_async_proactor_wake(proactor_);
      });
      PumpUntil([&] { return finished.load(); });
      cleanup.join();
    }
    if (sdma_) {
      EXPECT_EQ(iree_hal_amd_gpu_sdma_queue_free(sdma_), AMDF_STATUS_OK);
    }
    if (fabric_) {
      iree_notification_deinitialize(&publication_.notification);
      iree_hal_amd_gpu_context_deinitialize_memory(&context_);
      iree_hal_amd_fabric_release(fabric_);
    }
    iree_hal_amd_gpu_capture_pool_deinitialize(&captures_);
    iree_async_semaphore_wait_context_deinitialize(&waits_);
    completion_.reset();
    iree_async_proactor_release(proactor_);
    if (file_path_.Exists()) {
      EXPECT_TRUE(file_path_.Remove());
    }
    MemoryBackendTest::TearDown();
  }

  template <typename Predicate>
  void PumpUntil(Predicate ready) {
    while (!ready()) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  void Submit(bool fail, size_t length = 64) {
    expected_.resize(length);
    for (size_t i = 0; i < expected_.size(); ++i) {
      expected_[i] = uint8_t(i * 7);
    }
    std::vector<uint8_t> zeros(length);
    const auto& initial = GetParam() ? zeros : expected_;
    std::memcpy(buffer_contents_, GetParam() ? expected_.data() : zeros.data(),
                expected_.size());
    IREE_ASSERT_OK(iree_io_file_contents_write(
        file_path_.path_view(),
        iree_make_const_byte_span(initial.data(), initial.size()), allocator_));
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_ASYNC,
        file_path_.path_view(), allocator_, &handle));
    iree_status_t status = iree_hal_fd_file_from_handle(
        IREE_HAL_MEMORY_ACCESS_ALL, handle, proactor_, allocator_, file_.out());
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);
    if (fail) {
      notification_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
    }
    const uint32_t barrier = 0;
    iree_hal_semaphore_t* completion = completion_;
    uint64_t value = 1;
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit(
        &executor_, {}, {1, &completion, &value}, file_,
        GetParam() ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
        0, buffer_, 0, expected_.size(), &barrier, &barrier));
    const uint64_t chunks = (length + 4095) / 4096;
    PumpUntil([&] { return submitted_ == chunks || queue_.host_owners == 0; });
    ASSERT_EQ(submitted_, chunks);
    const auto source = iree_hal_amd_gpu_sdma_queue_publication_source(sdma_);
    while (source.publish(source.user_data)) {
    }
  }

  void ExpectPending() {
    uint64_t value = UINT64_MAX;
    IREE_EXPECT_OK(iree_hal_semaphore_query(completion_, &value));
    EXPECT_EQ(value, 0u);
    EXPECT_EQ(queue_.host_owners, 1u);
  }

  void ExpectCompletion(iree_status_code_t expected) {
    uint64_t value = UINT64_MAX;
    iree_status_t status = iree_hal_semaphore_query(completion_, &value);
    EXPECT_EQ(iree_status_code(status), expected);
    iree_status_free(status);
    if (expected == IREE_STATUS_OK) {
      EXPECT_EQ(value, 1u);
    }
    EXPECT_EQ(queue_.host_owners, 0u);
  }

  void ExecuteAcceptedCopy() {
    ASSERT_FALSE(native_stopped_);
    ASSERT_EQ(commands_.size(), submitted_);
    size_t offset = 0;
    for (const auto& commands : commands_) {
      ASSERT_EQ(commands.size(), 11u);
      ASSERT_EQ(commands[0], 1u);  // COPY_LINEAR.
      ASSERT_EQ(commands[7], 5u);  // FENCE.
      const auto address = [&](size_t index) {
        return uint64_t(commands[index]) |
               (uint64_t(commands[index + 1]) << 32);
      };
      const auto* source = reinterpret_cast<const void*>(address(3));
      auto* target = reinterpret_cast<void*>(address(5));
      const size_t length = commands[1] + 1;
      ASSERT_LE(offset + length, expected_.size());
      std::memcpy(target, source, length);
      EXPECT_EQ(std::memcmp(target, expected_.data() + offset, length), 0);
      *reinterpret_cast<uint32_t*>(address(8)) = commands[10];
      offset += length;
    }
    EXPECT_EQ(offset, expected_.size());
  }

  void PollCopies(amdf_status_t status = AMDF_STATUS_OK) {
    auto source = iree_hal_amd_gpu_sdma_queue_completion_source(sdma_);
    iree_hal_amd_gpu_completion_wait_t wait = {};
    iree_hal_amd_gpu_completion_t completion = {};
    iree_slim_mutex_initialize(&completion.drain_mutex);
    auto poll = [&](amdf_status_t status) {
      iree_slim_mutex_lock(&completion.drain_mutex);
      iree_hal_amd_gpu_completion_scope_t scope = {
          .completion = &completion,
          .source = &source,
          .fn = source.fn,
      };
      const bool ready =
          source.fn(source.user_data, status, &scope, &wait, nullptr);
      EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
      iree_slim_mutex_unlock(&completion.drain_mutex);
      return ready;
    };
    while (poll(status)) {
    }
    iree_slim_mutex_deinitialize(&completion.drain_mutex);
    // Completion failure wakes the distinct publication owner to deliver its
    // retained-copy notifications. Drive that owner before polling file I/O.
    const auto publication =
        iree_hal_amd_gpu_sdma_queue_publication_source(sdma_);
    while (publication.publish(publication.user_data)) {
    }
  }

  void StopCopies() {
    amdf_status_t status = AMDF_STATUS_OK;
    ASSERT_TRUE(iree_hal_amd_gpu_sdma_queue_try_inactivate(sdma_, &status));
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_TRUE(native_stopped_);
    ASSERT_TRUE(iree_hal_amd_gpu_sdma_queue_retire_failed(sdma_));
  }

  // Application-scoped policy outlives every captured buffer and reservation.
  Ref<iree_hal_pool_t> pool_;
  // Caller-owned buffer reference independent of the captured transfer.
  Ref<iree_hal_buffer_t> buffer_;
  // Borrowed dependency bytes, used only while the buffer's backing is alive.
  uint8_t* buffer_contents_ = nullptr;
  // Platform file retained by the production transfer after caller release.
  Ref<iree_hal_file_t> file_;
  // Small fixture file, removed after all asynchronous references retire.
  iree::testing::TempFilePath file_path_{"iree_file_staging_retirement"};
  // Payload oracle shared by the real disk and native dependency.
  std::vector<uint8_t> expected_;
  // File I/O and completion owner, polled only by the test thread.
  iree_async_proactor_t* proactor_ = nullptr;
  // Native GPU dependency table.
  amdf_gpu_api_t gpu_ = {};
  // Native storage domain owning the production allocation worker.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Borrowed memory and native queue services.
  iree_hal_amd_gpu_context_t context_ = {};
  // Coherent native SDMA family with no extra cache packets.
  amdf_queue_family_info_t family_ = {};
  // Publication readiness storage owned by the device fixture.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Production SDMA queue, including real command and retirement storage.
  iree_hal_amd_gpu_sdma_queue_t* sdma_ = nullptr;
  // Production lazy file staging and registration owner.
  iree_hal_amd_gpu_file_staging_t* staging_ = nullptr;
  // File submission captures, returned only after final observation.
  iree_hal_amd_gpu_capture_pool_t captures_ = {};
  // Queue's dependency associations, joined before file work begins.
  iree_async_semaphore_wait_context_t waits_ = {};
  // Coherent entry/exit boundary dependency.
  BoundaryQueue queue_;
  // Borrowed execution contract for the complete production file pipeline.
  iree_hal_amd_gpu_file_executor_t executor_ = {};
  // Accepted native words; addresses continue borrowing real backing.
  std::vector<std::vector<uint32_t>> commands_;
  // Accepted native submission count.
  uint64_t submitted_ = 0;
  // Checked native retirement, independent from reported failure.
  uint64_t retired_ = 0;
  // Injected notification error after successful native acceptance.
  amdf_status_t notification_status_ = AMDF_STATUS_OK;
  // Explicit proof that native execution can no longer access operands.
  bool native_stopped_ = false;
  // Public outcome must follow every accepted file and native use.
  Ref<iree_hal_semaphore_t> completion_;
  // File-owner diagnostics may precede public completion and native retirement.
  uint32_t file_failures_ = 0;
};

TEST_P(FileTransferStagingTest, HealthyCopyPublishesBytesBeforeCompletion) {
  ASSERT_NO_FATAL_FAILURE(Submit(false));
  ExpectPending();
  ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
  retired_ = submitted_;
  PollCopies();
  PumpUntil([&] { return queue_.host_owners == 0; });
  ExpectCompletion(IREE_STATUS_OK);
  EXPECT_EQ(file_failures_, 0u);
  EXPECT_EQ(queue_.boundaries, 2u);
  if (GetParam()) {
    // Close the caller's exclusive handle before reopening the completed file.
    file_.reset();
    iree_io_file_contents_t* contents = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(file_path_.path_view(),
                                              allocator_, &contents));
    EXPECT_EQ(contents->buffer.data_length, expected_.size());
    if (contents->buffer.data_length == expected_.size()) {
      EXPECT_EQ(std::memcmp(contents->buffer.data, expected_.data(),
                            expected_.size()),
                0);
    }
    iree_io_file_contents_free(contents);
  } else {
    EXPECT_EQ(std::memcmp(buffer_contents_, expected_.data(), expected_.size()),
              0);
  }
}

TEST_P(FileTransferStagingTest, EarlyErrorKeepsPublicCompletionPending) {
  ASSERT_NO_FATAL_FAILURE(Submit(true));
  PumpUntil([&] { return file_failures_ != 0; });
  ExpectPending();
  buffer_.reset();
  file_.reset();
  EXPECT_EQ(native_.destroy_count.load(), 0);
  ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
  ExpectPending();
  EXPECT_EQ(native_.destroy_count.load(), 0);
  ASSERT_NO_FATAL_FAILURE(StopCopies());
  PumpUntil([&] { return queue_.host_owners == 0; });
  ExpectCompletion(IREE_STATUS_DATA_LOSS);
  // Pool ownership is application-scoped, not retained by its buffers. The
  // operation's failed output must follow release of its captured operands.
  pool_.reset();
  EXPECT_EQ(native_.destroy_count.load(), 1);
}

TEST_P(FileTransferStagingTest, RetirementBeforeFilePollPreservesFailure) {
  ASSERT_NO_FATAL_FAILURE(Submit(false));
  PollCopies(amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
  ASSERT_NO_FATAL_FAILURE(StopCopies());
  // Both native notifications precede the file proactor's handoff. Coalescing
  // their delivery must neither lose the failure nor relinquish ownership
  // twice. Public completion also joins the file-owner callback.
  ExpectPending();
  buffer_.reset();
  file_.reset();
  EXPECT_EQ(native_.destroy_count.load(), 0);
  PumpUntil([&] { return queue_.host_owners == 0; });
  ExpectCompletion(IREE_STATUS_DATA_LOSS);
  pool_.reset();
  EXPECT_EQ(native_.destroy_count.load(), 1);
}

TEST_P(FileTransferStagingTest, FailureJoinsEveryAcceptedChunk) {
  ASSERT_NO_FATAL_FAILURE(Submit(false, 8192));
  ASSERT_EQ(submitted_, 2u);
  PollCopies(amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  PumpUntil([&] { return file_failures_ != 0; });
  ExpectPending();
  buffer_.reset();
  file_.reset();
  ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
  ExpectPending();
  EXPECT_EQ(native_.destroy_count.load(), 0);
  ASSERT_NO_FATAL_FAILURE(StopCopies());
  PumpUntil([&] { return queue_.host_owners == 0; });
  ExpectCompletion(IREE_STATUS_DATA_LOSS);
  pool_.reset();
  EXPECT_EQ(native_.destroy_count.load(), 1);
}

INSTANTIATE_TEST_SUITE_P(ReadAndWrite, FileTransferStagingTest,
                         ::testing::Bool());

}  // namespace
