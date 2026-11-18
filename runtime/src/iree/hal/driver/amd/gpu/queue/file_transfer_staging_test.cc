// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/native_queue.h"
#include "iree/hal/driver/amd/status.h"
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
    // Exact queue selected by the file caller; staging owns no native queue.
    iree_hal_amd_gpu_sdma_native_queue_t* native = nullptr;
    struct Copy {
      // Placed command dominated by this dependency's lifetime.
      iree_hal_amd_gpu_sdma_submission_t command = {};
      // Prepared native copy, immutable until checked retirement.
      std::vector<uint32_t> words;
      // File owner notified by native failure and final retirement.
      iree_hal_amd_gpu_retirement_callback_t callback = {};
    };
    // Stable accepted records, retained through the explicit native stop.
    std::vector<std::unique_ptr<Copy>> copies;
    // Exact metadata prefixes acquired through the native download dependency.
    std::vector<uint64_t> downloads;
    // A controlled first-use visibility dependency, completed by the test.
    iree_hal_amd_gpu_retirement_callback_t visibility_callback = {};
    // Native release remains outstanding while later I/O must remain unissued.
    bool defer_visibility = false;
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
    native_.api.host_mapping_cache_control = [](amdf_host_mapping_t* mapping,
                                                amdf_host_cache_operation_t,
                                                uint64_t, uint64_t) {
      // The native dependency exposes coherent host/device bytes even
      // when the public pool contract does not promise host coherence.
      EXPECT_NE(mapping, nullptr);
      return AMDF_STATUS_OK;
    };
    for (uint32_t i = 0; i < families_.size(); ++i) {
      binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                  AMDF_MEMORY_ADDRESS_GPU};
      native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
      native_families_[i].host_usage = 0;
    }
    IREE_ASSERT_OK(CreatePool(
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        IREE_HAL_MEMORY_ACCESS_NONE, pool_.out()));
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
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_native_queue_allocate(
        &context_, 0, AMDF_QUEUE_PUBLICATION_MODE_KERNEL, &control,
        &publication_, 0, AMDF_QUEUE_PRIORITY_NORMAL,
        AMDF_QUEUE_PLACEMENT_DEFAULT, allocator_, &sdma_));
    native_.required_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_FILE_IO;
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_staging_allocate(
        devices_[0], &context_, 0, &profile, false, proactor_, 2, 4096,
        UINT32_MAX, allocator_, &staging_));
    iree_async_semaphore_wait_context_initialize(&waits_);
    iree_hal_amd_gpu_capture_pool_initialize(4096, allocator_, &captures_);
    queue_.queue_family = families_[0];
    queue_.native = sdma_;
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
        .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
        .barrier_size = sizeof(uint32_t),
        .add_effects =
            [](iree_hal_queue_t*, iree_hal_memory_effects_t effects,
               void* barrier) {
              *static_cast<uint32_t*>(barrier) |= effects.bits;
            },
        .submit =
            [](iree_hal_queue_t* queue, iree_host_size_t count,
               const iree_hal_amd_gpu_transfer_operation_t* operations,
               const void* before, const void* after,
               iree_hal_amd_gpu_retirement_callback_t callback) {
              auto* boundary = static_cast<BoundaryQueue*>(queue);
              if (count) {
                auto copy = std::make_unique<BoundaryQueue::Copy>();
                copy->words.resize(count * 7);
                for (iree_host_size_t i = 0; i < count; ++i) {
                  const auto& operation = operations[i];
                  uint64_t source = 0, target = 0;
                  if (operation.type ==
                      IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
                    source = reinterpret_cast<uintptr_t>(operation.source.host);
                  } else {
                    IREE_RETURN_IF_ERROR(
                        iree_hal_amd_gpu_buffer_reference_resolve(
                            &operation.source.device, &source));
                  }
                  if (operation.type ==
                      IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD) {
                    target = reinterpret_cast<uintptr_t>(operation.target.host);
                    boundary->downloads.push_back(operation.length);
                  } else {
                    IREE_RETURN_IF_ERROR(
                        iree_hal_amd_gpu_buffer_reference_resolve(
                            &operation.target.device, &target));
                  }
                  iree_hal_amd_gpu_sdma_emit_copy(copy->words.data() + i * 7, 0,
                                                  source, target,
                                                  operation.length);
                }
                EXPECT_EQ(before ? *static_cast<const uint32_t*>(before) : 0u,
                          0u);
                EXPECT_EQ(after ? *static_cast<const uint32_t*>(after) : 0u,
                          0u);
                copy->command.words = copy->words.data();
                copy->command.word_count = copy->words.size();
                copy->command.completion.notify =
                    [](void* user_data, amdf_status_t status,
                       iree_hal_amd_gpu_sdma_retirement_t retirement) {
                      auto* copy = static_cast<BoundaryQueue::Copy*>(user_data);
                      copy->callback.fn(
                          copy->callback.user_data,
                          IREE_HAL_AMD_STATUS_FROM_AMDF(status, "native copy"),
                          nullptr, retirement == IREE_HAL_AMD_GPU_SDMA_RETIRED);
                    };
                copy->command.completion.user_data = copy.get();
                copy->callback = callback;
                auto* command = &copy->command;
                boundary->copies.push_back(std::move(copy));
                iree_hal_amd_gpu_sdma_native_queue_submit(boundary->native,
                                                          command);
                return iree_ok_status();
              }
              if (boundary->defer_visibility && before &&
                  *static_cast<const uint32_t*>(before)) {
                boundary->visibility_callback = callback;
                return iree_ok_status();
              }
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
      EXPECT_EQ(iree_hal_amd_gpu_sdma_native_queue_free(sdma_), AMDF_STATUS_OK);
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

  void PrepareFile(size_t length) {
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
  }

  void Submit(bool fail, size_t length = 64) {
    ASSERT_NO_FATAL_FAILURE(PrepareFile(length));
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
    EXPECT_EQ(queue_.copies.size(), chunks);
    const auto source =
        iree_hal_amd_gpu_sdma_native_queue_publication_source(sdma_);
    while (source.publish(source.user_data)) {
    }
  }

  void CompleteAndCheckTransfer() {
    ExpectPending();
    ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
    retired_ = submitted_;
    PollCopies();
    PumpUntil([&] { return queue_.host_owners == 0; });
    ExpectCompletion(IREE_STATUS_OK);
    EXPECT_EQ(file_failures_, 0u);
    EXPECT_EQ(queue_.boundaries, 2u);
    if (GetParam()) {
      // Close the caller's exclusive handle before reopening the completed
      // file.
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
      EXPECT_EQ(
          std::memcmp(buffer_contents_, expected_.data(), expected_.size()), 0);
    }
  }

  // Drives only the external native and file dependencies. The production
  // owner decides when to acquire metadata, submit I/O, and publish copies.
  void RunToCompletion() {
    const auto publication =
        iree_hal_amd_gpu_sdma_native_queue_publication_source(sdma_);
    while (queue_.host_owners) {
      while (publication.publish(publication.user_data)) {
      }
      if (executed_ < commands_.size()) {
        ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
        retired_ = submitted_;
        PollCopies();
      } else {
        IREE_ASSERT_OK(iree_async_proactor_poll(
            proactor_, iree_infinite_timeout(), nullptr));
      }
    }
  }

  void IndirectRows(bool host_readable, bool malformed);
  void DirectRows(bool indirect);

  struct SubmissionObserver {
    // Forwarding dependency; native file I/O still executes unchanged.
    iree_async_proactor_vtable_t vtable = {};
    // Original submission entry point beneath the observer.
    const iree_async_proactor_vtable_t* original = nullptr;
    // Synchronously observed file requests, independent of completion timing.
    mutable uint64_t file_requests = 0;
  };

  void ObserveFileSubmissions() {
    observer_.original = proactor_->vtable;
    observer_.vtable = *proactor_->vtable;
    observer_.vtable.submit = [](iree_async_proactor_t* proactor,
                                 iree_async_operation_list_t operations) {
      const auto* observer =
          reinterpret_cast<const SubmissionObserver*>(proactor->vtable);
      for (iree_host_size_t i = 0; i < operations.count; ++i) {
        const auto type = operations.values[i]->type;
        if (type == IREE_ASYNC_OPERATION_TYPE_FILE_READ ||
            type == IREE_ASYNC_OPERATION_TYPE_FILE_WRITE) {
          ++observer->file_requests;
        }
      }
      return observer->original->submit(proactor, operations);
    };
    proactor_->vtable = &observer_.vtable;
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
    while (executed_ < commands_.size()) {
      const auto& commands = commands_[executed_++];
      ASSERT_GE(commands.size(), 11u);
      ASSERT_EQ((commands.size() - 4) % 7, 0u);
      const auto address = [&](size_t index) {
        return uint64_t(commands[index]) |
               (uint64_t(commands[index + 1]) << 32);
      };
      size_t offset = 0;
      for (; offset + 4 < commands.size(); offset += 7) {
        ASSERT_EQ(commands[offset], 1u);  // COPY_LINEAR.
        const auto* source = reinterpret_cast<const void*>(address(offset + 3));
        auto* target = reinterpret_cast<void*>(address(offset + 5));
        std::memcpy(target, source, commands[offset + 1] + 1);
      }
      ASSERT_EQ(commands[offset], 5u);  // FENCE.
      *reinterpret_cast<uint32_t*>(address(offset + 1)) = commands[offset + 3];
    }
  }

  void PollCopies(amdf_status_t status = AMDF_STATUS_OK) {
    auto source = iree_hal_amd_gpu_sdma_native_queue_completion_source(sdma_);
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
        iree_hal_amd_gpu_sdma_native_queue_publication_source(sdma_);
    while (publication.publish(publication.user_data)) {
    }
  }

  void StopCopies() {
    amdf_status_t status = AMDF_STATUS_OK;
    ASSERT_TRUE(
        iree_hal_amd_gpu_sdma_native_queue_try_inactivate(sdma_, &status));
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_TRUE(native_stopped_);
    ASSERT_TRUE(iree_hal_amd_gpu_sdma_native_queue_retire_failed(sdma_));
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
  // Optional observer whose lifetime includes every file/proactor callback.
  SubmissionObserver observer_;
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
  iree_hal_amd_gpu_sdma_native_queue_t* sdma_ = nullptr;
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
  // Native dependency commands whose borrowed operands have been consumed.
  size_t executed_ = 0;
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
  ASSERT_NO_FATAL_FAILURE(CompleteAndCheckTransfer());
}

TEST_P(FileTransferStagingTest, GroupsCapturedRowsAndPreservesPadding) {
  ASSERT_NO_FATAL_FAILURE(PrepareFile(8192));
  std::vector<iree_hal_file_transfer_t> rows(64);
  for (size_t i = 0; i < rows.size(); ++i) {
    rows[i] = {
        .file_offset = i * 128,
        .buffer_offset = i * 128,
        .row_length = 90,
        .row_count = 1,
        .slice_count = 1,
    };
  }
  const iree_hal_file_binding_t file_binding = {file_, 0, expected_.size()};
  const iree_hal_buffer_binding_t buffer_binding = {buffer_, 0,
                                                    expected_.size()};
  const iree_hal_file_transfer_list_t list = {
      .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
      .host = {rows.size(), rows.data()},
  };
  Ref<iree_hal_semaphore_t> ready;
  IREE_ASSERT_OK(
      iree_hal_host_semaphore_create(proactor_, 0, allocator_, ready.out()));
  iree_hal_semaphore_t* producer = ready;
  iree_hal_semaphore_t* completion = completion_;
  uint64_t value = 1;
  const uint32_t barrier = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor_, {1, &producer, &value}, {1, &completion, &value},
      GetParam() ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
      {1, &file_binding}, {1, &buffer_binding}, &list, &barrier, &barrier));
  EXPECT_EQ(submitted_, 0u);
  // Host metadata was captured before return; later edits cannot alter the I/O.
  rows.clear();
  rows.shrink_to_fit();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready, 1, nullptr));
  PumpUntil([&] { return submitted_ == 2 || queue_.host_owners == 0; });
  ASSERT_EQ(submitted_, 2u);
  ASSERT_EQ(queue_.copies.size(), 2u);
  for (const auto& copy : queue_.copies) {
    EXPECT_EQ(copy->words.size(), 32 * 7u);
  }
  const auto publication =
      iree_hal_amd_gpu_sdma_native_queue_publication_source(sdma_);
  while (publication.publish(publication.user_data)) {
  }
  ASSERT_NO_FATAL_FAILURE(ExecuteAcceptedCopy());
  retired_ = submitted_;
  PollCopies();
  PumpUntil([&] { return queue_.host_owners == 0; });
  ExpectCompletion(IREE_STATUS_OK);
  iree_io_file_contents_t* contents = nullptr;
  const uint8_t* actual = buffer_contents_;
  if (GetParam()) {
    file_.reset();
    IREE_ASSERT_OK(iree_io_file_contents_read(file_path_.path_view(),
                                              allocator_, &contents));
    actual = contents->buffer.data;
  }
  for (size_t i = 0; i < expected_.size(); ++i) {
    EXPECT_EQ(actual[i], i % 128 < 90 ? expected_[i] : 0) << i;
  }
  if (contents) {
    iree_io_file_contents_free(contents);
  }
}

void FileTransferStagingTest::IndirectRows(bool host_readable, bool malformed) {
  ASSERT_NO_FATAL_FAILURE(PrepareFile(8192));
  constexpr size_t kCount = 65;
  constexpr size_t kCapacity = 68;
  constexpr size_t kMetadataLength = 64 + kCapacity * 64;
  native_.host_mappable = native_.host_visibility_guaranteed = host_readable;
  native_.required_flags = host_readable ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0;
  Ref<iree_hal_pool_t> metadata_pool;
  IREE_ASSERT_OK(CreatePool(
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      host_readable ? IREE_HAL_MEMORY_ACCESS_ALL : IREE_HAL_MEMORY_ACCESS_NONE,
      metadata_pool.out()));
  Ref<iree_hal_buffer_t> metadata;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      metadata_pool,
      iree_hal_make_pool_reservation_request(kMetadataLength, 4096),
      iree_infinite_timeout(), metadata.out()));
  native_.host_mappable = native_.host_visibility_guaranteed = true;
  native_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_FILE_IO;
  EXPECT_EQ(metadata.get()->host_binding_index !=
                IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE,
            host_readable);
  iree_hal_amd_gpu_buffer_reference_t reference;
  IREE_ASSERT_OK(iree_hal_amd_gpu_buffer_reference_initialize(
      queue_.queue_family, IREE_HAL_MEMORY_SITE_QUEUE, metadata, 0,
      &reference));
  uint64_t address = 0;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_buffer_reference_resolve(&reference, &address));
  auto* published = reinterpret_cast<uint8_t*>(address);
  std::memset(published, 0xFF, kMetadataLength);
  std::vector<uint8_t> expected_buffer(buffer_contents_,
                                       buffer_contents_ + 8192);
  std::vector<uint8_t> expected_file =
      GetParam() ? std::vector<uint8_t>(8192) : expected_;
  const iree_hal_file_binding_t file_binding = {file_, 17, 8192 - 17};
  const iree_hal_buffer_binding_t buffer_binding = {buffer_, 13, 8192 - 13};
  const iree_hal_file_transfer_list_t list = {
      .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
      .indirect = {{metadata, 64, kCapacity * 64}, metadata, 0},
  };
  Ref<iree_hal_semaphore_t> ready;
  IREE_ASSERT_OK(
      iree_hal_host_semaphore_create(proactor_, 0, allocator_, ready.out()));
  iree_hal_semaphore_t* producer = ready;
  iree_hal_semaphore_t* completion = completion_;
  uint64_t value = 1;
  const uint32_t barrier = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor_, {1, &producer, &value}, {1, &completion, &value},
      GetParam() ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
      {1, &file_binding}, {1, &buffer_binding}, &list, &barrier, &barrier));
  EXPECT_EQ(submitted_, 0u);
  // Count and records are deliberately invalid at enqueue. The producer
  // publishes only the live prefix; trailing capacity stays invalid forever.
  iree_unaligned_store_le_u64(published, kCount);
  for (size_t i = 0; i < kCount; ++i) {
    auto* record = published + 64 + i * 64;
    iree_unaligned_store_le_u64(record, i * 91);
    iree_unaligned_store_le_u64(record + 8, i * 96);
    const uint64_t length = i % 32 == 31 ? 0 : 90;
    iree_unaligned_store_le_u64(record + 16, length);
    if (malformed && i == kCount - 1) {
      iree_unaligned_store_le_u64(record, UINT64_C(1) << 48);
    }
    if (GetParam()) {
      std::memcpy(expected_file.data() + 17 + i * 91,
                  buffer_contents_ + 13 + i * 96, length);
    } else {
      std::memcpy(expected_buffer.data() + 13 + i * 96,
                  expected_.data() + 17 + i * 91, length);
    }
  }
  metadata.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(ready, 1, nullptr));
  ASSERT_NO_FATAL_FAILURE(RunToCompletion());
  ExpectCompletion(malformed ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_OK);
  if (host_readable) {
    EXPECT_TRUE(queue_.downloads.empty());
  } else {
    // Two full pages and one live record, never the inactive capacity tail.
    EXPECT_EQ(queue_.downloads, (std::vector<uint64_t>{8, 2048, 2048, 64}));
  }
  if (!malformed) {
    if (GetParam()) {
      file_.reset();
      iree_io_file_contents_t* contents = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(file_path_.path_view(),
                                                allocator_, &contents));
      EXPECT_EQ(contents->buffer.data_length, expected_file.size());
      EXPECT_EQ(std::memcmp(contents->buffer.data, expected_file.data(),
                            expected_file.size()),
                0);
      iree_io_file_contents_free(contents);
    } else {
      EXPECT_EQ(std::memcmp(buffer_contents_, expected_buffer.data(),
                            expected_buffer.size()),
                0);
    }
  }
}

TEST_P(FileTransferStagingTest, CpuReadableIndirectRowsUsePublishedPrefix) {
  IndirectRows(true, false);
}

void FileTransferStagingTest::DirectRows(bool indirect) {
  buffer_.reset();
  pool_.reset();
  native_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER,
                 UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                 IREE_HAL_MEMORY_ACCESS_ALL, pool_.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(8192, 4096),
      iree_infinite_timeout(), buffer_.out()));
  iree_byte_span_t contents;
  IREE_ASSERT_OK(iree_hal_buffer_native_host_span(buffer_, 0, 8192, &contents));
  buffer_contents_ = contents.data;
  ASSERT_NO_FATAL_FAILURE(PrepareFile(8192));
  native_.api.memory_query_pair_info = [](const amdf_memory_site_t* producer,
                                          const amdf_memory_site_t*,
                                          amdf_memory_pair_info_t* info) {
    info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    info->release.kind = info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    if (producer->kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
      info->release.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
      info->release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE;
      info->release.operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
    }
    return AMDF_STATUS_OK;
  };
  ObserveFileSubmissions();
  queue_.defer_visibility = true;
  const iree_hal_file_transfer_t transfer = {
      .row_length = 90,
      .row_count = 8,
      .slice_count = 8,
      .file_strides = {128, 1024},
      .buffer_strides = {128, 1024},
  };
  iree_hal_file_transfer_list_t list = {
      .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
      .host = {1, &transfer},
  };
  Ref<iree_hal_buffer_t> metadata;
  if (indirect) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(128, 64),
        iree_infinite_timeout(), metadata.out()));
    iree_byte_span_t bytes;
    IREE_ASSERT_OK(iree_hal_buffer_native_host_span(metadata, 0, 128, &bytes));
    const uint64_t words[16] = {1,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                (UINT64_C(1) << 48) | 90,
                                (UINT64_C(8) << 48) | 128,
                                128,
                                (UINT64_C(8) << 48) | 1024,
                                1024,
                                0};
    for (size_t i = 0; i < IREE_ARRAYSIZE(words); ++i) {
      iree_unaligned_store_le_u64(bytes.data + i * 8, words[i]);
    }
    list = {
        .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
        .indirect = {{metadata, 64, 64}, metadata, 0},
    };
  }
  const iree_hal_file_binding_t file_binding = {file_, 0, 8192};
  const iree_hal_buffer_binding_t buffer_binding = {buffer_, 0, 8192};
  iree_hal_semaphore_t* completion = completion_;
  uint64_t value = 1;
  const uint32_t barrier = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor_, {}, {1, &completion, &value},
      GetParam() ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
      {1, &file_binding}, {1, &buffer_binding}, &list, &barrier, &barrier));
  ExpectPending();
  ASSERT_NE(queue_.visibility_callback.fn, nullptr);
  EXPECT_EQ(observer_.file_requests, 0u);
  queue_.visibility_callback.fn(queue_.visibility_callback.user_data,
                                iree_ok_status(), nullptr, true);
  ASSERT_NO_FATAL_FAILURE(RunToCompletion());
  ExpectCompletion(IREE_STATUS_OK);
  EXPECT_EQ(observer_.file_requests, 64u);
  EXPECT_TRUE(queue_.copies.empty());
  iree_io_file_contents_t* file_contents = nullptr;
  const uint8_t* actual = buffer_contents_;
  if (GetParam()) {
    file_.reset();
    IREE_ASSERT_OK(iree_io_file_contents_read(file_path_.path_view(),
                                              allocator_, &file_contents));
    actual = file_contents->buffer.data;
  }
  for (size_t i = 0; i < expected_.size(); ++i) {
    EXPECT_EQ(actual[i], i % 128 < 90 ? expected_[i] : 0) << i;
  }
  if (file_contents) {
    iree_io_file_contents_free(file_contents);
  }
}

TEST_P(FileTransferStagingTest, DirectHostWindowsJoinFirstUseVisibility) {
  DirectRows(false);
}

TEST_P(FileTransferStagingTest, DirectIndirectWindowsJoinFirstUseVisibility) {
  DirectRows(true);
}

TEST_P(FileTransferStagingTest, PrivateIndirectRowsUseBoundedDownloads) {
  IndirectRows(false, false);
}

TEST_P(FileTransferStagingTest, MalformedIndirectRowsDrainAcceptedWork) {
  IndirectRows(false, true);
}

// The registration dependency can decline an optional preparation while every
// file operation, staging allocation, copy, and completion remains real.
class FileTransferUnregisteredStagingTest : public FileTransferStagingTest {
 protected:
  void SetUp() override {
    FileTransferStagingTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    if (!iree_any_bit_set(
            iree_async_proactor_query_capabilities(proactor_),
            IREE_ASYNC_PROACTOR_CAPABILITY_CONCURRENT_BUFFER_REGISTRATION)) {
      GTEST_SKIP() << "Backend does not attempt prepared registration";
    }
    registration_.vtable = *proactor_->vtable;
    registration_.vtable.prepare_buffer =
        [](iree_async_proactor_t* proactor,
           iree_async_buffer_registration_state_t*, iree_byte_span_t,
           iree_async_buffer_access_flags_t,
           iree_async_buffer_registration_flags_t flags,
           iree_async_buffer_registration_entry_t** out_entry) {
          const auto* registration =
              reinterpret_cast<const Registration*>(proactor->vtable);
          registration->preparations.fetch_add(1, std::memory_order_relaxed);
          *out_entry = nullptr;
          if (iree_any_bit_set(flags,
                               IREE_ASYNC_BUFFER_REGISTRATION_FLAG_OPTIONAL)) {
            return iree_ok_status();
          }
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "required registration exceeds pin budget");
        };
    proactor_->vtable = &registration_.vtable;
  }

  struct Registration {
    // Native proactor methods except for optional preparation admission.
    iree_async_proactor_vtable_t vtable = {};
    // Actual cold preparation calls, observed after transfer completion.
    mutable std::atomic<uint32_t> preparations = 0;
  };
  // Forwarding registration policy, alive through proactor destruction.
  Registration registration_;
};

TEST_P(FileTransferUnregisteredStagingTest, ReadWriteAndTrimWithoutPins) {
  ASSERT_NO_FATAL_FAILURE(Submit(false, 8192));
  ASSERT_NO_FATAL_FAILURE(CompleteAndCheckTransfer());
  EXPECT_GT(registration_.preparations.load(), 0u);
  iree_hal_amd_gpu_file_staging_trim(staging_);
}

INSTANTIATE_TEST_SUITE_P(ReadAndWrite, FileTransferUnregisteredStagingTest,
                         ::testing::Bool());

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

TEST_P(FileTransferStagingTest, FailedQueueDoesNotPoisonSharedFileStaging) {
  ASSERT_NO_FATAL_FAILURE(Submit(true));
  PumpUntil([&] { return file_failures_ != 0; });
  ExpectPending();

  // A second queue shares staging but has independent execution and completion.
  // Its dependency executes copies inline while the failed queue still owns
  // its accepted operands. The two requests touch disjoint device ranges.
  BoundaryQueue other;
  other.queue_family = queue_.queue_family;
  auto executor = executor_;
  executor.queue = &other;
  executor.submit = [](iree_hal_queue_t* queue, iree_host_size_t count,
                       const iree_hal_amd_gpu_transfer_operation_t* operations,
                       const void*, const void*,
                       iree_hal_amd_gpu_retirement_callback_t callback) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      const auto* operation = &operations[i];
      EXPECT_EQ(operation->type, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY);
      uint64_t source = 0, target = 0;
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->source.device, &source));
      IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_buffer_reference_resolve(
          &operation->target.device, &target));
      std::memcpy(reinterpret_cast<void*>(target),
                  reinterpret_cast<const void*>(source), operation->length);
    }
    if (!count) {
      ++static_cast<BoundaryQueue*>(queue)->boundaries;
    }
    callback.fn(callback.user_data, iree_ok_status(), nullptr, true);
    return iree_ok_status();
  };
  Ref<iree_hal_semaphore_t> completed;
  IREE_ASSERT_OK(iree_hal_host_semaphore_create(proactor_, 0, allocator_,
                                                completed.out()));
  if (GetParam()) {
    std::memcpy(buffer_contents_ + 4096, expected_.data(), expected_.size());
  }
  iree_hal_semaphore_t* timeline = completed;
  uint64_t value = 1;
  const uint32_t barrier = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit(
      &executor, {}, {1, &timeline, &value}, file_,
      GetParam() ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ,
      0, buffer_, 4096, expected_.size(), &barrier, &barrier));
  PumpUntil([&] { return other.host_owners == 0; });
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(completed, &observed));
  EXPECT_EQ(observed, 1u);
  EXPECT_EQ(other.boundaries, 2u);
  EXPECT_EQ(
      std::memcmp(buffer_contents_ + 4096, expected_.data(), expected_.size()),
      0);
  ExpectPending();
  ASSERT_NO_FATAL_FAILURE(StopCopies());
  PumpUntil([&] { return queue_.host_owners == 0; });
  ExpectCompletion(IREE_STATUS_DATA_LOSS);
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
