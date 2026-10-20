// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/storage/memory.h"

namespace {

// The dependency owns real shared sections, including their native geometry
// and mapping bindings. A selected failed release consumes the caller's slab
// but keeps its native backing until fixture destruction. No pool statistics,
// queue behavior, virtual mappings, or admission decisions are replaced.
class RetainingProvider : public iree_hal_slab_provider_t {
 public:
  explicit RetainingProvider(iree_hal_slab_provider_t* source)
      : source_(source) {
    iree_hal_slab_provider_initialize(&vtable_, iree_allocator_system(), this);
  }

  ~RetainingProvider() {
    EXPECT_EQ(held_, nullptr);
    for (const auto& slab : abandoned_) {
      iree_hal_slab_provider_release_slab_sync(source_, &slab, nullptr);
    }
    iree_hal_slab_provider_release(source_);
  }

  void HoldNextRelease() {
    std::lock_guard<std::mutex> lock(mutex_);
    hold_release_ = true;
  }

  void AbandonHeldRelease() {
    iree_hal_slab_release_request_t* request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [&] { return held_ != nullptr; });
      request = std::exchange(held_, nullptr);
    }
    iree_hal_slab_release_info_t info = {};
    for (iree_host_size_t i = 0; i < request->count; ++i) {
      info = iree_hal_slab_release_info_join(info, Abandon(request->slabs[i]));
    }
    request->completion.fn(request->completion.user_data, info);
  }

  void FailNextConstruction() { fail_construction_.store(true); }

 private:
  iree_hal_slab_release_info_t Abandon(const iree_hal_slab_t& slab) {
    abandoned_.push_back(slab);
    return {iree_make_status(IREE_STATUS_INTERNAL,
                             "native section release retained its backing"),
            slab.native_allocation.byte_length};
  }

  static iree_status_t Acquire(iree_hal_slab_provider_t* base,
                               iree_device_size_t length,
                               iree_device_size_t alignment,
                               iree_hal_slab_t* out_slab,
                               iree_hal_slab_release_info_t* out_release_info) {
    auto* self = static_cast<RetainingProvider*>(base);
    IREE_RETURN_IF_ERROR(iree_hal_slab_provider_acquire_slab_sync(
        self->source_, length, alignment, out_slab, out_release_info));
    if (self->fail_construction_.exchange(false)) {
      *out_release_info = self->Abandon(*out_slab);
      *out_slab = {};
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "construction failed after acquiring backing");
    }
    return iree_ok_status();
  }

  static iree_hal_slab_release_info_t Release(iree_hal_slab_provider_t* base,
                                              const iree_hal_slab_t* slab) {
    iree_hal_slab_release_info_t info = {};
    iree_hal_slab_provider_release_slab_sync(
        static_cast<RetainingProvider*>(base)->source_, slab, &info);
    return info;
  }

  // Shared native source, consumed by this fixture's final destruction.
  iree_hal_slab_provider_t* source_;
  // Actual backing excluded from reuse until after the accounting owner dies.
  std::vector<iree_hal_slab_t> abandoned_;
  // Controls a selected unpublished construction failure on the task worker.
  std::atomic<bool> fail_construction_ = false;
  // Protects capture of an asynchronous release without blocking its worker.
  std::mutex mutex_;
  // Joins the selected release's arrival with the test's explicit completion.
  std::condition_variable condition_;
  // The next release transfers its completion descriptor to held_.
  bool hold_release_ = false;
  // Borrowed caller-owned request, retained until the test completes it.
  iree_hal_slab_release_request_t* held_ = nullptr;

  // All ordinary operations preserve the real shared provider's behavior.
  static const iree_hal_slab_provider_vtable_t vtable_;
};

const iree_hal_slab_provider_vtable_t RetainingProvider::vtable_ = {
    .destroy =
        [](iree_hal_slab_provider_t* base) {
          delete static_cast<RetainingProvider*>(base);
        },
    .acquire_slabs =
        [](iree_hal_slab_provider_t* base,
           iree_hal_slab_acquire_request_t* request) {
          iree_hal_slab_provider_acquire_slabs_inline(base, request, Acquire,
                                                      Release);
        },
    .release_slabs =
        [](iree_hal_slab_provider_t* base,
           iree_hal_slab_release_request_t* request) {
          auto* self = static_cast<RetainingProvider*>(base);
          {
            std::lock_guard<std::mutex> lock(self->mutex_);
            if (std::exchange(self->hold_release_, false)) {
              self->held_ = request;
              self->condition_.notify_one();
              return;
            }
          }
          iree_hal_slab_provider_release_slabs(self->source_, request);
        },
    .wrap_buffer =
        [](iree_hal_slab_provider_t* base, const iree_hal_slab_t* slab,
           iree_device_size_t offset, iree_device_size_t length,
           iree_hal_memory_access_t access, iree_hal_buffer_usage_t usage,
           iree_hal_buffer_release_callback_t callback,
           iree_hal_buffer_t** out_buffer) {
          return iree_hal_slab_provider_wrap_buffer(
              static_cast<RetainingProvider*>(base)->source_, slab, offset,
              length, access, usage, callback, out_buffer);
        },
    .validate_asan_options =
        [](const iree_hal_slab_provider_t* base,
           const iree_hal_asan_pool_options_t* options) {
          return iree_hal_slab_provider_validate_asan_options(
              static_cast<const RetainingProvider*>(base)->source_, options);
        },
    .prefault =
        [](iree_hal_slab_provider_t* base, const iree_hal_slab_t* slab,
           iree_device_size_t offset, iree_device_size_t length) {
          iree_hal_slab_provider_prefault(
              static_cast<RetainingProvider*>(base)->source_, slab, offset,
              length);
        },
    .trim =
        [](iree_hal_slab_provider_t* base, iree_hal_pool_trim_flags_t flags) {
          iree_hal_slab_provider_trim(
              static_cast<RetainingProvider*>(base)->source_, flags);
        },
    .query_stats =
        [](const iree_hal_slab_provider_t* base,
           iree_hal_slab_provider_visited_set_t* visited,
           iree_hal_slab_provider_stats_t* out_stats) {
          iree_hal_slab_provider_query_stats(
              static_cast<const RetainingProvider*>(base)->source_, visited,
              out_stats);
        },
    .query_properties =
        [](const iree_hal_slab_provider_t* base,
           iree_hal_slab_provider_properties_t* out_properties) {
          iree_hal_slab_provider_query_properties(
              static_cast<const RetainingProvider*>(base)->source_,
              out_properties);
        },
};

class MemoryLossTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(
        loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
    const loom_serve_device_options_t options = {
        .driver = loom_serve_host_task_driver(host),
        .uri = IREE_SV("task"),
    };
    auto status = loom_serve_device_group_allocate(
        host, 1, &options, nullptr, iree_allocator_system(), &group_);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    auto* device = loom_serve_device_group_at(group_, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution_));
    auto* group = loom_serve_device_group(device);
    auto* queue = loom_serve_device_transfer_queue(device);
    iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(queue),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
    };
    iree_hal_pool_scope_t scope = {.family_count = 1, .families = &family};
    iree_hal_pool_t* source = nullptr;
    const iree_hal_slab_pool_options_t source_options = {};
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group, scope, &source_options,
                                             iree_allocator_system(), &source));
    iree_hal_slab_provider_t* shared = nullptr;
    IREE_ASSERT_OK(
        iree_hal_shared_slab_provider_create(iree_allocator_system(), &shared));
    provider_ = new RetainingProvider(shared);
    const iree_hal_passthrough_pool_options_t pool_options = {
        .memory_contract = source->memory_contract,
        .epoch_query = source->epoch_query,
    };
    status = iree_hal_passthrough_pool_create(
        pool_options, provider_, source->notification, source->frontier_tracker,
        iree_allocator_system(), &source_);
    iree_hal_pool_release(source);
    IREE_ASSERT_OK(status);
    family.interfaces = 0;
    iree_hal_virtual_buffer_capabilities_t capabilities;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        group, scope, iree_allocator_system(), &capabilities));
    IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(
        &capabilities, source_, kSlabSize, &geometry_));
    const loom_serve_memory_geometry_t geometries[] = {geometry_, geometry_};
    IREE_ASSERT_OK(loom_serve_memory_pool_allocate(
        group, scope, source_, queue, geometries, 3 * kSlabSize,
        iree_allocator_system(), &pool_));
    for (auto& buffer : buffers_) {
      IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
          pool_, geometry_, kSlabSize, 64, &statistics_, &buffer));
    }
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    for (auto* buffer : buffers_) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
    }
    loom_serve_memory_pool_trim(pool_, 0);
    IREE_EXPECT_OK(loom_serve_memory_pool_await(pool_));
    const auto statistics = loom_serve_memory_pool_statistics(pool_);
    EXPECT_EQ(statistics.committed_bytes, statistics.abandoned_bytes);
    EXPECT_EQ(statistics.releasing_bytes, 0u);
    loom_serve_memory_pool_free(pool_);
    iree_hal_pool_release(source_);
    // Only now may fixture teardown dispose the retained native payload. It
    // never supplies recovery credit to the production accounting owner.
    iree_hal_slab_provider_release(provider_);
    loom_serve_execution_release(execution_);
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
  }

  void CheckContents(size_t index, uint32_t value, bool fill = false) {
    auto* buffer = loom_serve_virtual_buffer_handle(buffers_[index]);
    std::array<uint32_t, 16> observed = {};
    const iree_hal_transfer_operation_t operations[] = {
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffer,
                    .length = sizeof(observed),
                    .pattern = &value,
                    .pattern_length = sizeof(value),
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = buffer,
                    .target = observed.data(),
                    .length = sizeof(observed),
                },
        },
    };
    uint64_t completion = 0;
    if (fill) {
      IREE_ASSERT_OK(loom_serve_execution_transfer(execution_, 1, operations,
                                                   &completion));
    }
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution_, 1, operations + 1,
                                                 &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
    for (auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  iree_status_t Prepare(uint64_t count,
                        loom_serve_memory_admission_t* out_admission) {
    return loom_serve_memory_pool_prepare(pool_, {{count * kSlabSize, 0}, 0},
                                          out_admission);
  }

  void ExpectLoss(iree_status_t status) {
    char text[2048];
    iree_host_size_t length = 0;
    EXPECT_TRUE(iree_status_format(status, sizeof(text), text, &length));
    EXPECT_NE(std::string(text, length).find("native section release"),
              std::string::npos);
    EXPECT_NE(std::string(text, length).find("physical admission"),
              std::string::npos);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, status);
  }

  // Exact source/target-compatible physical unit for the three-unit cap.
  static constexpr uint64_t kSlabSize = 256 * 1024;
  // Task execution and progress owners, outliving every pool and reservation.
  loom_serve_device_group_t* group_ = nullptr;
  // Real transfer timeline used to verify mapped output after loss and reuse.
  loom_serve_execution_t* execution_ = nullptr;
  // Dependency retaining native payload, outliving the accounting owner.
  RetainingProvider* provider_ = nullptr;
  // Real source pool which publishes loss and capacity notifications.
  iree_hal_pool_t* source_ = nullptr;
  // Serialized serving admission owner under test.
  loom_serve_memory_pool_t* pool_ = nullptr;
  // Qualified native replacement geometry used by all reservations.
  loom_serve_memory_geometry_t geometry_ = {};
  // Logical accounting shared by this fixture's reservations.
  loom_serve_memory_statistics_t statistics_ = {};
  // Three initially live roots and a fourth which can reuse surviving backing.
  std::array<loom_serve_virtual_buffer_t*, 4> buffers_ = {};
};

TEST_F(MemoryLossTest, AcceptedWaitEndsWithoutSpendingLostCapacity) {
  for (size_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffers_[i], 0, kSlabSize));
    CheckContents(i, 100 + i, true);
  }
  loom_serve_virtual_buffer_begin_trim(buffers_[2]);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffers_[2]));
  provider_->HoldNextRelease();
  loom_serve_memory_pool_trim(pool_, 2 * kSlabSize);
  loom_serve_memory_admission_t admission;
  IREE_EXPECT_OK(Prepare(1, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  const auto waiting = loom_serve_memory_pool_statistics(pool_);
  EXPECT_EQ(waiting.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(waiting.releasing_bytes, kSlabSize);
  EXPECT_EQ(waiting.abandoned_bytes, 0u);
  provider_->AbandonHeldRelease();
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  EXPECT_FALSE(loom_serve_memory_pool_is_pending(pool_));
  const auto lost = loom_serve_memory_pool_statistics(pool_);
  EXPECT_EQ(lost.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(lost.releasing_bytes, 0u);
  EXPECT_EQ(lost.abandoned_bytes, kSlabSize);
  EXPECT_EQ(lost.released_bytes, kSlabSize);
  ExpectLoss(Prepare(1, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  // Even recovering the lost unit could not satisfy this larger request.
  IREE_EXPECT_OK(Prepare(2, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool_));

  loom_serve_virtual_buffer_begin_trim(buffers_[0]);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffers_[0]));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffers_[3], 0, kSlabSize));
  CheckContents(3, 999, true);
  CheckContents(1, 101);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            3 * kSlabSize);
}

TEST_F(MemoryLossTest, FailedConstructionChargesLossWithoutLogicalReturns) {
  provider_->FailNextConstruction();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(buffers_[0], 0, kSlabSize));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  const auto failed = loom_serve_memory_pool_statistics(pool_);
  EXPECT_EQ(failed.committed_bytes, kSlabSize);
  EXPECT_EQ(failed.abandoned_bytes, kSlabSize);
  EXPECT_EQ(failed.pending_bytes, 0u);
  EXPECT_EQ(failed.releasing_bytes, 0u);
  EXPECT_EQ(failed.released_bytes, 0u);
  EXPECT_EQ(statistics_.committed_bytes, 0u);
  loom_serve_memory_admission_t admission;
  ExpectLoss(Prepare(3, &admission));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffers_[1], 0, kSlabSize));
  CheckContents(1, 123, true);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffers_[2], 0, kSlabSize));
  CheckContents(2, 456, true);
  ExpectLoss(Prepare(1, &admission));
}

}  // namespace
