// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/gather.h"

#include <array>
#include <future>
#include <string>

#include "iree/io/file_contents.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/testing/device.h"

namespace {

// The storage dependency inserts a real queue prerequisite after enumeration.
// Reads still pass through the production index provider and native file I/O.
// This makes accepted-but-not-retired reads deterministic without sleeps.
struct GatedProvider : iree_io_parameter_provider_t {
  // Real checkpoint provider retained through this forwarding capability.
  iree_io_parameter_provider_t* provider;
  // Retained storage prerequisite, released by the outer test owner.
  iree_hal_semaphore_t* gate;
  // Published after all reads have been accepted by the real provider.
  std::promise<iree_status_code_t>* submitted;
  // Published only after the gather callback releases its provider claim.
  std::promise<void>* released;

  GatedProvider(iree_io_parameter_provider_t* provider,
                iree_hal_semaphore_t* gate,
                std::promise<iree_status_code_t>* submitted,
                std::promise<void>* released)
      : provider(provider),
        gate(gate),
        submitted(submitted),
        released(released) {
    iree_atomic_ref_count_init(&ref_count);
    iree_io_parameter_provider_retain(provider);
    iree_hal_semaphore_retain(gate);
    static const iree_io_parameter_provider_vtable_t table = {
        .destroy =
            [](iree_io_parameter_provider_t* base) {
              auto* self = static_cast<GatedProvider*>(base);
              auto* released = self->released;
              iree_io_parameter_provider_release(self->provider);
              iree_hal_semaphore_release(self->gate);
              delete self;
              released->set_value();
            },
        .gather =
            [](iree_io_parameter_provider_t* base, iree_hal_device_t* device,
               iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_string_view_t scope,
               iree_hal_buffer_t* target, iree_host_size_t count,
               iree_io_parameter_enumerator_t enumerator) {
              auto* self = static_cast<GatedProvider*>(base);
              EXPECT_EQ(waits.count, 0u);
              uint64_t value = 1;
              auto status = iree_io_parameter_provider_gather(
                  self->provider, device, queue, {1, &self->gate, &value},
                  signals, scope, target, count, enumerator);
              self->submitted->set_value(iree_status_code(status));
              return status;
            },
    };
    vtable = &table;
  }
};

class ParameterGatherTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr size_t kRows = 128;
  static constexpr size_t kRequests = 48;
  static constexpr size_t kLength = 16;
  enum RunFlagBits {
    kSplat = 1u << 0,
    kCancelReadiness = 1u << 1,
    kInvalidSpan = 1u << 2,
    kCancelProducer = 1u << 3,
  };

  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_host_create(GetParam(), nullptr, allocator_, &host_));
    iree_hal_driver_t* driver = nullptr;
    IREE_ASSERT_OK(loom_serve_test_driver_acquire(host_, &driver));
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = loom_serve_test_device()->device_uri,
    };
    auto status = loom_serve_device_group_allocate(host_, 1, &options,
                                                   allocator_, &group_);
    iree_hal_driver_release(driver);
    IREE_ASSERT_OK(status);
    owner_ = loom_serve_device_group_at(group_, 0);
    device_ = loom_serve_device_handle(owner_);
    queue_ = loom_serve_device_transfer_queue(owner_);
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  iree_hal_buffer_t* Buffer(size_t length,
                            loom_serve_retirement_t* retirement) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(owner_),
        iree_hal_make_pool_reservation_request(length, 256),
        iree_infinite_timeout(), &buffer));
    IREE_CHECK_OK(loom_serve_retirement_track(retirement, &buffer, allocator_));
    return buffer;
  }

  iree_hal_semaphore_t* Semaphore() {
    iree_hal_semaphore_t* result = nullptr;
    IREE_CHECK_OK(
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                  0, IREE_HAL_SEMAPHORE_FLAG_NONE, &result));
    return result;
  }

  static uint8_t Byte(size_t domain, size_t row, size_t byte) {
    return 1 + (domain * 61 + row * 17 + byte * 13) % 255;
  }

  void Run(uint32_t flags) {
    const bool cancel = (flags & kCancelReadiness) != 0;
    const bool splat = (flags & kSplat) != 0;
    const bool invalid = (flags & kInvalidSpan) != 0;
    const bool cancelled_producer = (flags & kCancelProducer) != 0;
    uint64_t one = 1;
    loom_serve_retirement_t descriptors_retirement;
    loom_serve_retirement_t target_retirement[2];
    loom_serve_retirement_initialize(&descriptors_retirement);
    for (auto& retirement : target_retirement) {
      loom_serve_retirement_initialize(&retirement);
    }
    auto* descriptor_root =
        Buffer(64 + 2 * kRequests * 24, &descriptors_retirement);
    iree_hal_buffer_t* descriptors = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        descriptor_root, 64, 2 * kRequests * 24, allocator_, &descriptors));
    iree_hal_buffer_release(descriptor_root);
    std::array<uint8_t, 2 * kRequests * 24> records = {};
    for (size_t i = 0; i < 2 * kRequests; ++i) {
      iree_unaligned_store_le_u64(
          records.data() + i * 24,
          (((i % kRequests) * 23 + 37) % kRows) * kLength);
      iree_unaligned_store_le_u64(records.data() + i * 24 + 8,
                                  (i % kRequests) * kLength);
      iree_unaligned_store_le_u64(records.data() + i * 24 + 16, kLength);
    }
    // Only A's eighth span is invalid. Its first seven accepted reads still
    // borrow the destination even though the callback reports an error.
    if (invalid) {
      iree_unaligned_store_le_u64(records.data() + 7 * 24, kRows * kLength);
    }
    auto* produced = Semaphore();
    auto* producer_gate = Semaphore();
    IREE_ASSERT_OK(iree_hal_queue_update(
        queue_, {1, &producer_gate, &one}, {1, &produced, &one}, records.data(),
        0, descriptors, 0, records.size(), nullptr, IREE_HAL_UPDATE_FLAG_NONE));

    std::array<iree_hal_semaphore_t*, 2> gates, consumed;
    std::array<iree_hal_fence_t*, 2> ready = {};
    std::array<std::array<uint8_t, kRequests * kLength>, 2> outputs = {};
    std::promise<iree_status_code_t> submitted[2];
    std::promise<void> released[2];
    struct FileOwner {
      // Owns the native handle through every imported file reference.
      iree_io_file_handle_t* handle = nullptr;
      // Final native close, independent of target-buffer retirement.
      std::promise<void> closed;
    } files[2];
    iree::testing::TempFilePath paths[] = {
        iree::testing::TempFilePath("parameter_gather_a"),
        iree::testing::TempFilePath("parameter_gather_b")};
    for (size_t domain = 0; domain < 2; ++domain) {
      iree_io_parameter_index_t* index = nullptr;
      IREE_ASSERT_OK(iree_io_parameter_index_create(allocator_, &index));
      iree_io_parameter_index_entry_t entry = {
          .key = IREE_SV("same-key"),
          .length = kRows * kLength,
      };
      iree_io_file_handle_t* file = nullptr;
      if (splat) {
        entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
        entry.storage.splat.pattern_length = 4;
        for (size_t i = 0; i < 4; ++i) {
          entry.storage.splat.pattern[i] = Byte(domain, 0, i);
        }
      } else {
        std::array<uint8_t, kRows * kLength> bytes;
        for (size_t row = 0; row < kRows; ++row) {
          for (size_t i = 0; i < kLength; ++i) {
            bytes[row * kLength + i] = Byte(domain, row, i);
          }
        }
        IREE_ASSERT_OK(iree_io_file_contents_write(
            paths[domain].path_view(),
            iree_make_const_byte_span(bytes.data(), bytes.size()), allocator_));
        IREE_ASSERT_OK(iree_io_file_handle_open(
            IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
                IREE_IO_FILE_MODE_ASYNC,
            paths[domain].path_view(), allocator_, &files[domain].handle));
        auto status = iree_io_file_handle_wrap(
            IREE_IO_FILE_ACCESS_READ, IREE_IO_FILE_MODE_ASYNC,
            iree_io_file_handle_primitive(files[domain].handle),
            {[](void* payload, iree_io_file_handle_primitive_t) {
               auto* owner = static_cast<FileOwner*>(payload);
               iree_io_file_handle_release(owner->handle);
               owner->closed.set_value();
             },
             &files[domain]},
            allocator_, &file);
        if (!iree_status_is_ok(status)) {
          iree_io_file_handle_release(files[domain].handle);
        }
        IREE_ASSERT_OK(status);
        entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
        entry.storage.file.handle = file;
      }
      IREE_ASSERT_OK(iree_io_parameter_index_add(index, &entry));
      iree_io_file_handle_release(file);
      iree_io_parameter_provider_t* provider = nullptr;
      IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
          iree_string_view_empty(), index, IREE_HOST_SIZE_MAX, allocator_,
          &provider));
      iree_io_parameter_index_release(index);
      gates[domain] = Semaphore();
      consumed[domain] = Semaphore();
      auto* gated = new GatedProvider(provider, gates[domain],
                                      &submitted[domain], &released[domain]);
      iree_io_parameter_provider_release(provider);
      auto* target = Buffer(outputs[domain].size(), &target_retirement[domain]);
      std::string key = "same-key";
      IREE_ASSERT_OK(loom_serve_parameter_gather(
          gated, device_, queue_, {1, &produced, &one},
          iree_make_string_view(key.data(), key.size()), descriptors,
          domain * kRequests * 24, kRequests, target, allocator_,
          &ready[domain]));
      key.assign(key.size(), 'x');
      iree_io_parameter_provider_release(gated);
      IREE_ASSERT_OK(iree_hal_queue_download(
          queue_, iree_hal_fence_semaphore_list(ready[domain]),
          {1, &consumed[domain], &one}, target, 0, outputs[domain].data(),
          outputs[domain].size(), nullptr));
      iree_hal_buffer_release(target);
    }
    iree_hal_buffer_release(descriptors);
    if (cancelled_producer) {
      iree_hal_semaphore_fail(producer_gate,
                              iree_status_from_code(IREE_STATUS_CANCELLED));
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_signal(producer_gate, 1, nullptr));
    }
    for (size_t domain = 0; domain < 2; ++domain) {
      if (!cancelled_producer) {
        EXPECT_EQ(
            submitted[domain].get_future().get(),
            invalid && domain == 0 ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_OK);
      }
      released[domain].get_future().wait();
      if (!cancelled_producer) {
        EXPECT_FALSE(loom_serve_retirement_is_idle(&target_retirement[domain]));
      }
    }
    loom_serve_retirement_await(&descriptors_retirement);

    // Table B completes while A is held; submission order is not a dependency.
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gates[1], 1, nullptr));
    IREE_EXPECT_STATUS_IS(
        cancelled_producer ? IREE_STATUS_CANCELLED : IREE_STATUS_OK,
        iree_hal_semaphore_wait(consumed[1], 1, iree_infinite_timeout(), 0));
    loom_serve_retirement_await(&target_retirement[1]);
    if (!cancelled_producer) {
      EXPECT_FALSE(loom_serve_retirement_is_idle(&target_retirement[0]));
    }
    IREE_EXPECT_STATUS_IS(cancelled_producer ? IREE_STATUS_CANCELLED
                          : invalid          ? IREE_STATUS_OUT_OF_RANGE
                                             : IREE_STATUS_DEFERRED,
                          iree_hal_fence_query(ready[0]));
    if (cancel) {
      iree_hal_fence_fail(ready[0],
                          iree_status_from_code(IREE_STATUS_CANCELLED));
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_CANCELLED,
          iree_hal_semaphore_wait(consumed[0], 1, iree_infinite_timeout(), 0));
      // Failure released the consumer, not the accepted gated file reads.
      EXPECT_FALSE(loom_serve_retirement_is_idle(&target_retirement[0]));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gates[0], 1, nullptr));
    if (!cancel) {
      IREE_EXPECT_STATUS_IS(
          cancelled_producer ? IREE_STATUS_CANCELLED
          : invalid          ? IREE_STATUS_OUT_OF_RANGE
                             : IREE_STATUS_OK,
          iree_hal_semaphore_wait(consumed[0], 1, iree_infinite_timeout(), 0));
    }
    for (size_t domain = 0; domain < 2; ++domain) {
      loom_serve_retirement_await(&target_retirement[domain]);
      if (!cancelled_producer && (domain == 1 || (!cancel && !invalid))) {
        for (size_t i = 0; i < kRequests; ++i) {
          for (size_t j = 0; j < kLength; ++j) {
            EXPECT_EQ(outputs[domain][i * kLength + j],
                      Byte(domain, splat ? 0 : (i * 23 + 37) % kRows,
                           splat ? j % 4 : j));
          }
        }
      }
      iree_hal_fence_release(ready[domain]);
      iree_hal_semaphore_release(consumed[domain]);
      iree_hal_semaphore_release(gates[domain]);
      loom_serve_retirement_deinitialize(&target_retirement[domain]);
      if (!splat) {
        files[domain].closed.get_future().wait();
        EXPECT_TRUE(paths[domain].Remove());
      }
    }
    iree_hal_semaphore_release(produced);
    iree_hal_semaphore_release(producer_gate);
    loom_serve_retirement_deinitialize(&descriptors_retirement);
  }

  // Host allocator, valid until all queue and file owners have retired.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Native worker and proactor owner.
  loom_serve_host_t* host_ = nullptr;
  // Shared device/storage lifetime enclosing all operations.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed serving-device capabilities.
  loom_serve_device_t* owner_ = nullptr;
  // Borrowed physical device.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed exact transfer queue.
  iree_hal_queue_t* queue_ = nullptr;
};

TEST_P(ParameterGatherTest, IndependentFileDomains) { Run(0); }
TEST_P(ParameterGatherTest, IndependentSplatDomains) { Run(kSplat); }
TEST_P(ParameterGatherTest, FailedReadinessRetainsAcceptedFiles) {
  Run(kCancelReadiness);
}
TEST_P(ParameterGatherTest, FailedReadinessRetainsAcceptedSplats) {
  Run(kCancelReadiness | kSplat);
}
TEST_P(ParameterGatherTest, CancelledProducerRetiresWithoutReading) {
  Run(kCancelProducer);
}
TEST_P(ParameterGatherTest, RejectedSpanRetainsEarlierReads) {
  Run(kInvalidSpan);
}

TEST_P(ParameterGatherTest, DescriptorBoundsAndEmptyGather) {
  iree_io_parameter_index_t* index = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_create(allocator_, &index));
  iree_io_parameter_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      iree_string_view_empty(), index, IREE_HOST_SIZE_MAX, allocator_,
      &provider));
  iree_io_parameter_index_release(index);
  loom_serve_retirement_t retirement;
  loom_serve_retirement_initialize(&retirement);
  auto* descriptors = Buffer(48, &retirement);
  auto* target = Buffer(16, &retirement);
  struct InvalidRange {
    // Descriptor start relative to the source view.
    iree_device_size_t offset;
    // Number of complete 24-byte records requested.
    iree_host_size_t count;
  };
  for (const auto range :
       {InvalidRange{49, 0}, InvalidRange{1, 2}, InvalidRange{UINT64_MAX, 1},
        InvalidRange{0, IREE_HOST_SIZE_MAX}}) {
    iree_hal_fence_t* ready = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_serve_parameter_gather(
            provider, device_, queue_, iree_hal_semaphore_list_empty(),
            IREE_SV("unused"), descriptors, range.offset, range.count, target,
            allocator_, &ready));
    EXPECT_EQ(ready, nullptr);
  }
  iree_hal_fence_t* empty_ready = nullptr;
  auto* producer = Semaphore();
  uint64_t value = 1;
  IREE_ASSERT_OK(loom_serve_parameter_gather(
      provider, device_, queue_, {1, &producer, &value}, IREE_SV("unused"),
      descriptors, 48, 0, target, allocator_, &empty_ready));
  iree_io_parameter_provider_release(provider);
  iree_hal_buffer_release(descriptors);
  iree_hal_buffer_release(target);
  // No byte storage is retained by an empty gather, but the dependency remains.
  EXPECT_TRUE(loom_serve_retirement_is_idle(&retirement));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        iree_hal_fence_query(empty_ready));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(producer, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_fence_wait(empty_ready, iree_infinite_timeout(), 0));
  iree_hal_fence_release(empty_ready);
  iree_hal_semaphore_release(producer);
  loom_serve_retirement_await(&retirement);
  loom_serve_retirement_deinitialize(&retirement);
}

INSTANTIATE_TEST_SUITE_P(Workers, ParameterGatherTest,
                         ::testing::Values(1u, 4u));

}  // namespace
