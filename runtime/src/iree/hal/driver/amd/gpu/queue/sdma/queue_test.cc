// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/queue.h"

#include <cstring>
#include <semaphore>
#include <tuple>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/driver/amd/gpu/queue/aql/signal.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/command_buffer.h"
#include "iree/hal/driver/amd/gpu/queue/semaphore.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::amd::fabric::FakeNative;
using iree::hal::amd::fabric::MemoryBackendTest;
using iree::hal::cts::Ref;

// Actual public queue, recorder, pools, dependency tracking, and completion
// owners. Only native command execution/retirement is controlled by this
// fixture; physical visibility and packet behavior belong to native CTS.
class QueueTest : public MemoryBackendTest,
                  public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  bool SubmissionGcr() const { return std::get<0>(GetParam()); }
  bool AqlSignals() const { return std::get<1>(GetParam()); }

  static QueueTest* FromNative(amdf_kernel_queue_t* queue) {
    return reinterpret_cast<QueueTest*>(queue);
  }

  void SetUp() override {
    native_.queue_family_ordinals[0] = 0;
    ASSERT_NO_FATAL_FAILURE(MemoryBackendTest::SetUp());
    native_.user_data = this;
    native_.before_create = [this] {
      if (hold_acquisition_.exchange(false)) {
        acquisition_entered_.release();
        acquisition_resume_.acquire();
      }
    };
    native_.required_flags = 0;
    native_.host_coherent = true;
    native_.host_mappable = native_.host_visibility_guaranteed = true;
    native_.file_io_enabled = true;
    native_.host_mapping_features = AMDF_HOST_MAPPING_FEATURE_FILE_IO;
    native_.api.memory_scope_query_pair_info =
        [](amdf_memory_scope_t*, const amdf_memory_profile_pair_query_t* query,
           amdf_memory_pair_info_t* info) {
          for (uint32_t i = 0; i < query->access_count; ++i) {
            EXPECT_EQ(query->accesses[i].requirements.access,
                      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
          }
          info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
          info->release.kind = info->acquire.kind =
              AMDF_CACHE_TRANSITION_KIND_NONE;
          return AMDF_STATUS_OK;
        };
    native_.api.host_mapping_cache_control =
        [](amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
           uint64_t offset, uint64_t length) {
          auto* memory = reinterpret_cast<FakeNative::Memory*>(mapping);
          EXPECT_LE(offset + length, memory->byte_length);
          EXPECT_EQ(operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
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
    native_.api.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                          amdf_memory_address_kind_t,
                                          uint64_t* address) {
      *address = reinterpret_cast<uintptr_t>(
          reinterpret_cast<FakeNative::Memory*>(memory)->data);
      return AMDF_STATUS_OK;
    };
    native_.api.memory_create = [](amdf_memory_scope_t* scope,
                                   const amdf_memory_create_info_t* info,
                                   amdf_memory_t** out_memory,
                                   amdf_memory_release_info_t* out_release) {
      if (out_release) {
        *out_release = {};
      }
      auto* storage = reinterpret_cast<FakeNative::Scope*>(scope);
      auto* native = storage->owner;
      if (native->before_create) {
        native->before_create();
      }
      if (!amdf_status_is_ok(native->create_failure)) {
        return native->create_failure;
      }
      FakeNative::CheckAccesses(storage, info->access_count, info->accesses);
      amdf_memory_profile_t profile = {};
      std::vector<amdf_memory_access_capabilities_t> capabilities(
          info->access_count);
      EXPECT_EQ(native->api.memory_scope_query_device_profile(
                    scope, info->memory_profile_ordinal, info->access_count,
                    info->accesses, &profile, capabilities.data()),
                AMDF_STATUS_OK);
      EXPECT_EQ(info->required_flags & ~profile.supported_flags, 0u);
      auto* memory = new FakeNative::Memory{
          native,
          nullptr,
          info->byte_length,
          uint64_t(++native->create_count),
          info->placement,
          storage->memory_class,
          false,
          std::vector<amdf_memory_device_access_t>(
              info->accesses, info->accesses + info->access_count)};
      IREE_CHECK_OK(iree_allocator_malloc_aligned(
          iree_allocator_system(), info->byte_length, info->minimum_alignment,
          0, &memory->data));
      {
        std::lock_guard<std::mutex> lock(native->memory_mutex);
        ++native->live_memories;
      }
      *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
      return AMDF_STATUS_OK;
    };
    for (uint32_t i = 0; i < families_.size(); ++i) {
      binding_descriptions_[i] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                  AMDF_MEMORY_ADDRESS_GPU};
      native_.address_kinds[i] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
      native_families_[i].host_usage = 0;
    }
    gpu_.kernel_queue_create = [](amdf_device_t* device,
                                  const amdf_gpu_kernel_queue_create_info_t*,
                                  amdf_kernel_queue_t** queue) {
      *queue = reinterpret_cast<amdf_kernel_queue_t*>(
          reinterpret_cast<FakeNative::Device*>(device)->owner->user_data);
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
          auto* self = FromNative(queue);
          EXPECT_FALSE(self->stopped_);
          const auto& command = info->commands[0];
          auto* memory = reinterpret_cast<FakeNative::Memory*>(command.memory);
          const auto* words = reinterpret_cast<const uint32_t*>(
              static_cast<uint8_t*>(memory->data) + command.byte_offset);
          self->commands_.emplace_back(words, words + command.byte_length / 4);
          *point = self->commands_.size();
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_request_notification =
        [](amdf_kernel_queue_t* queue, uint64_t, const amdf_native_event_t*) {
          return FromNative(queue)->notification_status_;
        };
    native_.api.kernel_queue_refresh_status =
        [](amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* status) {
          status->retired_submission = FromNative(queue)->retired_;
          return AMDF_STATUS_OK;
        };
    native_.api.kernel_queue_wait = [](amdf_kernel_queue_t* queue,
                                       uint64_t point, uint64_t, uint64_t) {
      return point <= FromNative(queue)->retired_
                 ? AMDF_STATUS_OK
                 : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    };
    gpu_.kernel_queue_inactivate = [](amdf_kernel_queue_t* queue) {
      FromNative(queue)->stopped_ = true;
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
    local_scope_ =
        reinterpret_cast<amdf_memory_scope_t*>(&native_.local_scopes[0]);
    context_.memory_scope_count = 1;
    context_.memory_scopes = &local_scope_;
    context_.families = &native_family_;
    context_.kernel_notification.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    context_.event_sink = {[](void* user_data, const iree_hal_device_event_t*) {
                             ++static_cast<QueueTest*>(user_data)->diagnostics_;
                           },
                           this};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    iree_notification_initialize(&publication_.notification);
    iree_slim_mutex_initialize(&completion_.drain_mutex);
    iree_hal_amd_gpu_signal_cache_initialize(&publication_, 0, &signals_);
    access_ = {
        .device = context_.device,
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
            },
    };
    amdf_memory_access_capabilities_t capabilities = {};
    amdf_memory_profile_t profile = {};
    ASSERT_EQ(native_.api.memory_scope_query_device_profile(
                  native_.scope(), 0, 1, &access_, &profile, &capabilities),
              AMDF_STATUS_OK);
    memory_ = {
        .scope = native_.scope(),
        .profile = profile,
        .access_count = 1,
        .accesses = &access_,
        .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_pool_initialize(
        &context_, 0, 1, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        allocator_, &programs_));
    bool file_available = false;
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_staging_query_profile(
        &context_, &profile, &file_available));
    ASSERT_TRUE(file_available);
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_staging_allocate(
        devices_[0], &context_, 0, &profile, SubmissionGcr(),
        iree::hal::amd::fabric::test_proactor(), 2, 4096, UINT32_MAX,
        allocator_, &files_));
    iree_hal_amd_gpu_allocation_context_initialize(
        devices_[0], &publication_, 1, allocator_, &allocations_);
    options_ = {
        .family_ordinal = 0,
        .queue_count = 1,
        .publication_mode = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        .barriers = {.submission = SubmissionGcr()},
        .signal_layout =
            {
                .stride = 8,
                .initialize = iree_hal_amd_gpu_pm4_signal_storage_initialize,
            },
        .signals = &signals_,
        .memory = &memory_,
        .publication = &publication_,
        .allocations = &allocations_,
        .file_staging = files_,
        .staging_capacity = 256,
        .capture_block_size = 4096,
        .capture_block_count = 4,
    };
    if (AqlSignals()) {
      options_.signal_layout = {
          .stride = sizeof(iree_hal_amd_gpu_aql_signal_t),
          .value_offset = offsetof(iree_hal_amd_gpu_aql_signal_t, value),
          .pending_value = 1,
          .initialize = iree_hal_amd_gpu_aql_signal_storage_initialize,
      };
    }
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_queue_create(
        families_[0], &context_, &options_, &params, allocator_, queue_.out()));
    IREE_ASSERT_OK(
        CreatePool(IREE_HAL_BUFFER_USAGE_TRANSFER,
                   UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                   IREE_HAL_MEMORY_ACCESS_NONE, pool_.out()));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(32768, 4096),
        iree_infinite_timeout(), buffer_.out()));
    const auto view = iree_hal_buffer_memory_view(buffer_);
    const auto* facts =
        reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
            view.bindings + view.contract->binding_layout.binding_count);
    bytes_ = static_cast<uint8_t*>(
        reinterpret_cast<FakeNative::Memory*>(facts->memory->handle)->data);
  }

  void TearDown() override {
    if (queue_) {
      EXPECT_TRUE(lifecycle().is_idle(queue_));
      EXPECT_EQ(lifecycle().shutdown(queue_), AMDF_STATUS_OK);
      queue_.reset();
    }
    buffer_.reset();
    pool_.reset();
    if (files_) {
      std::atomic<bool> joined = false;
      std::thread cleanup([&] {
        EXPECT_TRUE(iree_hal_amd_gpu_file_staging_free(files_));
        joined.store(true);
        iree_async_proactor_wake(iree::hal::amd::fabric::test_proactor());
      });
      while (!joined.load()) {
        IREE_EXPECT_OK(
            iree_async_proactor_poll(iree::hal::amd::fabric::test_proactor(),
                                     iree_infinite_timeout(), nullptr));
      }
      cleanup.join();
    }
    iree_hal_amd_gpu_allocation_context_deinitialize(&allocations_);
    iree_hal_amd_gpu_program_pool_deinitialize(&programs_);
    iree_hal_amd_gpu_signal_cache_deinitialize(&context_, &signals_);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_hal_amd_fabric_release(fabric_);
    iree_slim_mutex_deinitialize(&completion_.drain_mutex);
    iree_notification_deinitialize(&publication_.notification);
    MemoryBackendTest::TearDown();
  }

  const iree_hal_amd_gpu_queue_lifecycle_t& lifecycle() {
    return iree_hal_amd_gpu_sdma_queue_lifecycle;
  }

  Ref<iree_hal_semaphore_t> Semaphore(bool native = false) {
    Ref<iree_hal_semaphore_t> semaphore;
    if (native) {
      IREE_CHECK_OK(iree_hal_amd_gpu_semaphore_create(
          iree::hal::amd::fabric::test_proactor(), nullptr, 0, 1, allocator_,
          semaphore.out()));
    } else {
      IREE_CHECK_OK(iree_hal_host_semaphore_create(
          iree::hal::amd::fabric::test_proactor(), 0, allocator_,
          semaphore.out()));
    }
    return semaphore;
  }

  static iree_hal_semaphore_list_t Point(iree_hal_semaphore_t** semaphore,
                                         uint64_t* value) {
    return {1, semaphore, value};
  }

  void Publish() {
    const auto allocation =
        iree_hal_amd_gpu_allocation_context_publication_source(&allocations_);
    while (allocation.publish(allocation.user_data)) {
    }
    const auto source = lifecycle().publication_source(queue_);
    while (source.publish(source.user_data)) {
    }
  }

  void Poll(amdf_status_t status = AMDF_STATUS_OK) {
    auto source = lifecycle().completion_source(queue_);
    iree_hal_amd_gpu_completion_wait_t wait = {};
    bool ready = true;
    while (ready) {
      iree_hal_amd_gpu_completion_copy_t copy = {};
      iree_slim_mutex_lock(&completion_.drain_mutex);
      iree_hal_amd_gpu_completion_scope_t scope = {
          .completion = &completion_,
          .source = &source,
          .fn = source.fn,
      };
      ready = source.fn(source.user_data, status, &scope, &wait, &copy);
      if (copy.fn) {
        copy.fn(copy.user_data);
      }
      EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
      iree_slim_mutex_unlock(&completion_.drain_mutex);
      ready |= copy.fn != nullptr;
    }
  }

  void ExecuteAccepted() {
    ASSERT_FALSE(stopped_);
    while (retired_ < commands_.size()) {
      const auto& words = commands_[retired_];
      size_t cursor = 0;
      while (cursor < words.size()) {
        const auto address = [&](size_t index) {
          return uint64_t(words[cursor + index]) |
                 (uint64_t(words[cursor + index + 1]) << 32);
        };
        switch (words[cursor] & 255) {
          case 0:
            ++cursor;
            break;
          case 1: {
            ASSERT_LE(cursor + 7, words.size());
            std::memcpy(reinterpret_cast<void*>(address(5)),
                        reinterpret_cast<const void*>(address(3)),
                        words[cursor + 1] + 1);
            cursor += 7;
            break;
          }
          case 5: {
            ASSERT_LE(cursor + 4, words.size());
            *reinterpret_cast<uint32_t*>(address(1)) = words[cursor + 3];
            cursor += 4;
            break;
          }
          case 8: {
            ASSERT_LE(cursor + 6, words.size());
            ASSERT_EQ(*reinterpret_cast<uint32_t*>(address(1)),
                      words[cursor + 3]);
            cursor += 6;
            break;
          }
          case 11: {
            ASSERT_LE(cursor + 5, words.size());
            auto* target = reinterpret_cast<uint8_t*>(address(1));
            const auto* pattern =
                reinterpret_cast<const uint8_t*>(&words[cursor + 3]);
            for (uint32_t i = 0; i <= words[cursor + 4]; ++i) {
              target[i] = pattern[i & 3];
            }
            cursor += 5;
            break;
          }
          case 13: {
            ASSERT_LE(cursor + 3, words.size());
            EXPECT_EQ(address(1) % 32, 0u);
            *reinterpret_cast<uint64_t*>(address(1)) =
                UINT64_C(0x7135ac020000) + retired_;
            cursor += 3;
            break;
          }
          default:
            FAIL() << "unexpected native packet " << words[cursor];
        }
      }
      ++retired_;
    }
  }

  void Drain() {
    while (!lifecycle().is_idle(queue_)) {
      Publish();
      ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
      Poll();
      iree_status_t status =
          iree_async_proactor_poll(iree::hal::amd::fabric::test_proactor(),
                                   iree_immediate_timeout(), nullptr);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_free(status);
      } else {
        IREE_ASSERT_OK(status);
      }
    }
  }

  void ExpectValue(iree_hal_semaphore_t* semaphore, uint64_t expected) {
    uint64_t value = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(value, expected);
  }

  // Real native allocation provider and command-family entry points.
  amdf_gpu_api_t gpu_ = {};
  // Fabric and context retain asynchronous allocation/release ownership.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Queue borrows this device domain until every callback joins.
  iree_hal_amd_gpu_context_t context_ = {};
  // Qualified device-local program storage, distinct from host staging.
  amdf_memory_scope_t* local_scope_ = nullptr;
  // Qualified coherent SDMA dependency, with optional submission cache
  // boundaries.
  amdf_queue_family_info_t native_family_ = {};
  // Production publication source; the fixture controls its invocation order.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Production completion scopes coordinate worker-copy tails.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Native signal cells outlive queue captures and reader caches.
  iree_hal_amd_gpu_signal_cache_t signals_ = {};
  // Explicit native device access borrowed by the placement policy.
  amdf_memory_device_access_t access_ = {};
  // Coherent control and staging placement.
  iree_hal_amd_fabric_memory_options_t memory_ = {};
  // Recorded private data uses the production placement registry.
  iree_hal_amd_gpu_program_pool_t programs_ = {};
  // Shared file staging belongs to the device, never to a hidden copy queue.
  iree_hal_amd_gpu_file_staging_t* files_ = nullptr;
  // Shared transient allocation service, progressed independently of SDMA.
  iree_hal_amd_gpu_allocation_context_t allocations_ = {};
  // Immutable construction facts retained for recorder creation.
  iree_hal_amd_gpu_sdma_queue_options_t options_ = {};
  // Public execution owner under test.
  Ref<iree_hal_queue_t> queue_;
  // Application pool remains alive through all captured buffer uses.
  Ref<iree_hal_pool_t> pool_;
  // Actual buffer and native binding used by public operations.
  Ref<iree_hal_buffer_t> buffer_;
  // Dependency's backing bytes; never obtained through a HAL host mapping.
  uint8_t* bytes_ = nullptr;
  // Accepted immutable native spans, independent of HAL operation captures.
  std::vector<std::vector<uint32_t>> commands_;
  // Native checked retirement, advanced only after dependency execution.
  uint64_t retired_ = 0;
  // Notification error after irrevocable native acceptance.
  amdf_status_t notification_status_ = AMDF_STATUS_OK;
  // Native access cessation, distinct from failure notification.
  bool stopped_ = false;
  // Reported failures remain distinct from checked retirement.
  uint32_t diagnostics_ = 0;
  // One cold allocation can be held independently from already-ready work.
  std::atomic<bool> hold_acquisition_ = false;
  // Explicit native-entry evidence, without sleeps or timing assumptions.
  std::binary_semaphore acquisition_entered_{0};
  // Releases the held acquisition even after a test assertion fails.
  std::binary_semaphore acquisition_resume_{0};

  struct AcquisitionGate {
    // Borrowed native dependency resumed before fixture teardown.
    std::binary_semaphore* resume;
    ~AcquisitionGate() { resume->release(); }
  };
};

TEST_P(QueueTest, HostTransferWrapsNativeSlotsAndRetainsBorrowedBytes) {
  std::vector<uint8_t> source(20003), target(source.size(), 0);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = uint8_t(i * 13 + 17);
  }
  auto uploaded = Semaphore();
  auto downloaded = Semaphore();
  iree_hal_semaphore_t* upload = uploaded;
  iree_hal_semaphore_t* download = downloaded;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_upload(queue_, {}, Point(&upload, &one),
                                       source.data(), buffer_, 3, source.size(),
                                       nullptr));
  IREE_ASSERT_OK(iree_hal_queue_download(
      queue_, Point(&upload, &one), Point(&download, &one), buffer_, 3,
      target.data(), target.size(), nullptr));
  Publish();
  EXPECT_EQ(commands_.size(), 64u);
  ExpectValue(downloaded, 0);
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectValue(uploaded, 1);
  ExpectValue(downloaded, 1);
  EXPECT_EQ(target, source);
  EXPECT_GT(commands_.size(), 128u);
}

TEST_P(QueueTest, UnreadyInputDoesNotHoldIndependentReadyWork) {
  auto gate = Semaphore();
  auto blocked = Semaphore();
  auto independent = Semaphore();
  iree_hal_semaphore_t* input = gate;
  iree_hal_semaphore_t* first = blocked;
  iree_hal_semaphore_t* second = independent;
  uint64_t one = 1;
  const uint32_t pattern = 0x53c1a72e;
  IREE_ASSERT_OK(iree_hal_queue_fill(queue_, Point(&input, &one),
                                     Point(&first, &one), buffer_, 0, 1024,
                                     &pattern, sizeof(pattern), nullptr, 0));
  IREE_ASSERT_OK(iree_hal_queue_fill(queue_, {}, Point(&second, &one), buffer_,
                                     4096, 1024, &pattern, sizeof(pattern),
                                     nullptr, 0));
  Publish();
  ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
  Poll();
  ExpectValue(blocked, 0);
  ExpectValue(independent, 1);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectValue(blocked, 1);
  EXPECT_EQ(std::memcmp(bytes_, bytes_ + 4096, 1024), 0);
}

TEST_P(QueueTest, ReusableRecordingCapturesIndependentBindingTables) {
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_command_buffer_create(
      families_[0], &programs_, 0, options_.barriers, 0,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 1, allocator_, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const std::vector<uint8_t> input = {19, 37, 67, 89, 131, 193, 251};
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      recording, input.data(), 0,
      iree_hal_make_indirect_buffer_ref(0, 0, input.size()), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  auto first = Semaphore();
  auto second = Semaphore();
  iree_hal_semaphore_t* outputs[] = {first, second};
  uint64_t one = 1;
  iree_hal_buffer_binding_t binding = {buffer_, 17, input.size()};
  IREE_ASSERT_OK(iree_hal_queue_execute(queue_, {}, Point(&outputs[0], &one),
                                        recording, {1, &binding}, 0));
  binding.offset = 8201;
  IREE_ASSERT_OK(iree_hal_queue_execute(queue_, {}, Point(&outputs[1], &one),
                                        recording, {1, &binding}, 0));
  binding = {};
  recording.reset();
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectValue(first, 1);
  ExpectValue(second, 1);
  EXPECT_EQ(std::memcmp(bytes_ + 17, input.data(), input.size()), 0);
  EXPECT_EQ(std::memcmp(bytes_ + 8201, input.data(), input.size()), 0);
}

TEST_P(QueueTest, NativeFailureRetainsAcceptedOperandsUntilStop) {
  auto completed = Semaphore();
  iree_hal_semaphore_t* output = completed;
  uint64_t one = 1;
  std::vector<uint8_t> input(3073, 0x97);
  notification_status_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  IREE_ASSERT_OK(iree_hal_queue_upload(queue_, {}, Point(&output, &one),
                                       input.data(), buffer_, 0, input.size(),
                                       nullptr));
  Publish();
  ExpectValue(completed, 0);
  buffer_.reset();
  ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
  ExpectValue(completed, 0);
  amdf_status_t status = AMDF_STATUS_OK;
  ASSERT_TRUE(lifecycle().try_inactivate(queue_, &status));
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_TRUE(lifecycle().retire_failed(queue_, IREE_STATUS_DATA_LOSS));
  uint64_t value = 0;
  iree_status_t outcome = iree_hal_semaphore_query(completed, &value);
  EXPECT_EQ(iree_status_code(outcome), IREE_STATUS_DATA_LOSS);
  iree_status_free(outcome);
  EXPECT_TRUE(lifecycle().is_idle(queue_));
}

TEST_P(QueueTest, NativeDependencyPublishesBeforeHostRetirement) {
  auto producer = Semaphore(true);
  auto consumer = Semaphore(true);
  iree_hal_semaphore_t* before = producer;
  iree_hal_semaphore_t* after = consumer;
  uint64_t one = 1;
  const uint32_t pattern = 0x71bca345;
  IREE_ASSERT_OK(iree_hal_queue_fill(queue_, {}, Point(&before, &one), buffer_,
                                     0, 512, &pattern, sizeof(pattern), nullptr,
                                     0));
  IREE_ASSERT_OK(iree_hal_queue_copy(queue_, Point(&before, &one),
                                     Point(&after, &one), buffer_, 0, buffer_,
                                     4096, 512, nullptr, 0));
  Publish();
  EXPECT_EQ(commands_.size(), SubmissionGcr() ? 1u : 2u);
  ExpectValue(producer, 0);
  ExpectValue(consumer, 0);
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(std::memcmp(bytes_, bytes_ + 4096, 512), 0);
  ExpectValue(consumer, 1);
}

TEST_P(QueueTest, FailedDependencyStillJoinsAnUnresolvedSibling) {
  auto failed = Semaphore(true);
  auto pending = Semaphore(true);
  auto output = Semaphore(true);
  iree_hal_semaphore_t* waits[] = {failed, pending};
  iree_hal_semaphore_t* done = output;
  uint64_t values[] = {1, 1};
  iree_hal_semaphore_fail(failed, iree_status_from_code(IREE_STATUS_DATA_LOSS));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queue_, {2, waits, values},
                                        Point(&done, values), nullptr, 0));
  Publish();
  EXPECT_TRUE(commands_.empty());
  ExpectValue(output, 0);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(pending, 1, nullptr));
  ASSERT_NO_FATAL_FAILURE(Drain());
  uint64_t value = 0;
  iree_status_t status = iree_hal_semaphore_query(output, &value);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_DATA_LOSS);
  iree_status_free(status);
}

TEST_P(QueueTest, PreparationFailureJoinsAlreadyPublishedInputs) {
  auto allocation_gate = Semaphore();
  auto allocated = Semaphore(true);
  auto produced = Semaphore(true);
  auto failed = Semaphore(true);
  iree_hal_semaphore_t* points[] = {allocation_gate, allocated, produced,
                                    failed};
  uint64_t one = 1;
  const auto request = iree_hal_make_pool_reservation_request(4096, 4096);
  Ref<iree_hal_buffer_t> pending;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue_, Point(&points[0], &one),
                                       Point(&points[1], &one), pool_, 1,
                                       &request, pending.out()));
  IREE_ASSERT_OK(
      iree_hal_queue_barrier(queue_, {}, Point(&points[2], &one), nullptr, 0));
  // A missing allocation dependency is diagnosed during address resolution.
  // Failure still joins the captured producer before releasing its outputs.
  const uint32_t pattern = 0x753917bc;
  IREE_ASSERT_OK(iree_hal_queue_fill(queue_, Point(&points[2], &one),
                                     Point(&points[3], &one), pending, 0, 4096,
                                     &pattern, sizeof(pattern), nullptr, 0));
  Publish();
  ExpectValue(failed, 0);
  ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
  Poll();
  Publish();
  uint64_t value = 0;
  iree_status_t status = iree_hal_semaphore_query(failed, &value);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(allocation_gate, 1, nullptr));
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectValue(allocated, 1);
}

TEST_P(QueueTest, FileReadRecordingAndWriteUseThePublicTransferQueue) {
  iree::testing::TempFilePath path("iree_sdma_public_file");
  std::vector<uint8_t> expected(8197);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint8_t(i * 17 + 37);
  }
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(expected.data(), expected.size()), allocator_));
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), allocator_, &handle));
  Ref<iree_hal_file_t> file;
  iree_status_t status = iree_hal_fd_file_from_handle(
      IREE_HAL_MEMORY_ACCESS_ALL, handle,
      iree::hal::amd::fabric::test_proactor(), allocator_, file.out());
  iree_io_file_handle_release(handle);
  IREE_ASSERT_OK(status);
  auto loaded = Semaphore(true);
  auto copied = Semaphore(true);
  auto saved = Semaphore(true);
  iree_hal_semaphore_t* points[] = {loaded, copied, saved};
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_read(queue_, {}, Point(&points[0], &one), file,
                                     0, buffer_, 5, expected.size(), nullptr,
                                     0));
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_command_buffer_create(
      families_[0], &programs_, 0, options_.barriers, 0,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, allocator_, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording, iree_hal_make_buffer_ref(buffer_, 5, expected.size()),
      iree_hal_make_buffer_ref(buffer_, 16387, expected.size()), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queue_, Point(&points[0], &one), Point(&points[1], &one), recording,
      iree_hal_buffer_binding_table_empty(), 0));
  IREE_ASSERT_OK(iree_hal_queue_write(queue_, Point(&points[1], &one),
                                      Point(&points[2], &one), buffer_, 16387,
                                      file, 0, expected.size(), nullptr, 0));
  recording.reset();
  file.reset();
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectValue(saved, 1);
  EXPECT_EQ(std::memcmp(bytes_ + 16387, expected.data(), expected.size()), 0);
  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(
      iree_io_file_contents_read(path.path_view(), allocator_, &contents));
  EXPECT_EQ(contents->buffer.data_length, expected.size());
  EXPECT_EQ(
      std::memcmp(contents->buffer.data, expected.data(), expected.size()), 0);
  iree_io_file_contents_free(contents);
  EXPECT_TRUE(path.Remove());
}

TEST_P(QueueTest, PoolAllocationAndDeallocationFollowTheSameTimeline) {
  auto allocated = Semaphore(true);
  auto filled = Semaphore(true);
  auto freed = Semaphore(true);
  iree_hal_semaphore_t* points[] = {allocated, filled, freed};
  uint64_t one = 1;
  const auto request = iree_hal_make_pool_reservation_request(16384, 4096);
  Ref<iree_hal_buffer_t> transient;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue_, {}, Point(&points[0], &one),
                                       pool_, 1, &request, transient.out()));
  const uint32_t pattern = 0x835b4e29;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queue_, Point(&points[0], &one), Point(&points[1], &one), transient, 0,
      16384, &pattern, sizeof(pattern), nullptr, 0));
  iree_hal_buffer_t* buffer = transient;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queue_, Point(&points[1], &one),
                                         Point(&points[2], &one), 1, &buffer));
  transient.reset();
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectValue(freed, 1);
}

TEST_P(QueueTest, PartialRecordingReservationResumesBeforeCloneAcquisition) {
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_command_buffer_create(
      families_[0], &programs_, 0, options_.barriers, 0,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 1, allocator_, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  std::vector<uint8_t> expected(600);
  for (uint32_t i = 0; i < expected.size(); ++i) {
    expected[i] = uint8_t(i * 23 + 19);
    IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
        recording, &expected[i], 0, iree_hal_make_indirect_buffer_ref(0, i, 1),
        0));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  auto first = Semaphore(true);
  auto second = Semaphore(true);
  iree_hal_semaphore_t* outputs[] = {first, second};
  uint64_t one = 1;
  {
    AcquisitionGate gate{&acquisition_resume_};
    hold_acquisition_ = true;
    iree_hal_buffer_binding_t binding = {buffer_, 0, expected.size()};
    IREE_ASSERT_OK(iree_hal_queue_execute(queue_, {}, Point(&outputs[0], &one),
                                          recording, {1, &binding}, 0));
    binding.offset = 8192;
    IREE_ASSERT_OK(iree_hal_queue_execute(queue_, {}, Point(&outputs[1], &one),
                                          recording, {1, &binding}, 0));
    Publish();
    acquisition_entered_.acquire();
    ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
    Poll();
    Publish();
    ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
    Poll();
    ExpectValue(first, 1);
    ExpectValue(second, 1);
    EXPECT_EQ(std::memcmp(bytes_, expected.data(), expected.size()), 0);
    EXPECT_EQ(std::memcmp(bytes_ + 8192, expected.data(), expected.size()), 0);
    recording.reset();
  }
  ASSERT_NO_FATAL_FAILURE(Drain());
}

TEST_P(QueueTest, ReturnedSignalsResumeWhileNativeGrowthIsBlocked) {
  if (SubmissionGcr()) {
    GTEST_SKIP() << "Submission GCR uses host-retired dependencies";
  }
  std::vector<Ref<iree_hal_semaphore_t>> retained;
  using Reader = std::unique_ptr<iree_hal_amd_gpu_signal_t,
                                 decltype(&iree_hal_amd_gpu_signal_release)>;
  std::vector<Reader> readers;
  uint64_t one = 1;
  for (uint32_t i = 0; i < 64; ++i) {
    retained.push_back(Semaphore(true));
    iree_hal_semaphore_t* output = retained.back();
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queue_, {}, Point(&output, &one), nullptr, 0));
    Publish();
    iree_hal_amd_gpu_semaphore_wait_t wait;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_semaphore_acquire_wait(output, one, &context_, &wait));
    ASSERT_EQ(wait.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
    readers.emplace_back(wait.signal, iree_hal_amd_gpu_signal_release);
  }
  ASSERT_NO_FATAL_FAILURE(Drain());
  auto resumed = Semaphore(true);
  iree_hal_semaphore_t* output = resumed;
  {
    AcquisitionGate gate{&acquisition_resume_};
    hold_acquisition_ = true;
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queue_, {}, Point(&output, &one), nullptr, 0));
    Publish();
    acquisition_entered_.acquire();
    readers.front().reset();
    Publish();
    EXPECT_EQ(commands_.size(), 65u);
    ASSERT_NO_FATAL_FAILURE(ExecuteAccepted());
    Poll();
    ExpectValue(resumed, 1);
  }
  ASSERT_NO_FATAL_FAILURE(Drain());
}

TEST_P(QueueTest, TimestampAndHostCallFollowNativeRetirement) {
  auto stamped = Semaphore(true);
  auto called = Semaphore(true);
  iree_hal_semaphore_t* points[] = {stamped, called};
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_timestamp(queue_, {}, Point(&points[0], &one),
                                          buffer_, 8, nullptr, 0));
  bool invoked = false;
  uint64_t arguments[4] = {13, 29, 37, 53};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queue_, Point(&points[0], &one), Point(&points[1], &one),
      iree_hal_make_host_call(
          [](void* user_data, const uint64_t args[4],
             iree_hal_host_call_context_t*) {
            EXPECT_EQ(args[0], 13u);
            EXPECT_EQ(args[3], 53u);
            *static_cast<bool*>(user_data) = true;
            return iree_ok_status();
          },
          &invoked),
      arguments, 0));
  Publish();
  EXPECT_FALSE(invoked);
  ExpectValue(called, 0);
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_TRUE(invoked);
  ExpectValue(called, 1);
  uint64_t timestamp = 0;
  std::memcpy(&timestamp, bytes_ + 8, sizeof(timestamp));
  EXPECT_EQ(timestamp, UINT64_C(0x7135ac020000));
}

TEST_P(QueueTest, DeviceFailureCancelsUnsubmittedDependencyWaits) {
  auto input = Semaphore();
  auto completed = Semaphore();
  iree_hal_semaphore_t* waits = input;
  iree_hal_semaphore_t* signals = completed;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queue_, Point(&waits, &one),
                                        Point(&signals, &one), nullptr, 0));
  Publish();
  EXPECT_TRUE(commands_.empty());
  lifecycle().notify_failure(queue_);
  Publish();
  amdf_status_t status = AMDF_STATUS_OK;
  ASSERT_TRUE(lifecycle().try_inactivate(queue_, &status));
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_TRUE(lifecycle().retire_failed(queue_, IREE_STATUS_ABORTED));
  EXPECT_TRUE(lifecycle().is_idle(queue_));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(completed, &value));
  ExpectValue(input, 0);
}

INSTANTIATE_TEST_SUITE_P(CacheVisibilityAndSignalLayout, QueueTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));
}  // namespace
