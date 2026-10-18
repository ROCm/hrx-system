// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/sdma/queue.h"

#include <array>
#include <atomic>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Native dependencies expose explicit admission and checked-retirement points.
// The real transport owns encoding, publication, capacity, callbacks, and
// native storage lifetime. Hardware CTS separately verifies the resulting
// payloads.
class QueueTest : public ::testing::Test {
 protected:
  struct Copy {
    // Placed record retained through its terminal notification.
    iree_hal_amd_gpu_sdma_copy_t operation = {};
    // Number of delivered terminal outcomes.
    uint32_t notifications = 0;
    // Observed native result, independent of safe storage reclamation.
    amdf_status_t status = AMDF_STATUS_OK;
    // Whether the callback returned operand ownership.
    iree_hal_amd_gpu_sdma_retirement_t retirement =
        IREE_HAL_AMD_GPU_SDMA_UNRETIRED;

    Copy() {
      operation = {
          .source = 0x100000,
          .target = 0x200000,
          .length = 4096,
          .notify =
              [](void* user_data, amdf_status_t status,
                 iree_hal_amd_gpu_sdma_retirement_t retirement) {
                auto* self = static_cast<Copy*>(user_data);
                ++self->notifications;
                self->status = status;
                self->retirement = retirement;
              },
          .user_data = this,
      };
    }
  };

  void SetUp() override {
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory) {
      auto* self = reinterpret_cast<QueueTest*>(scope);
      self->storage_.resize((info->byte_length + 7) / 8);
      *out_memory = reinterpret_cast<amdf_memory_t*>(self);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_info = [](amdf_memory_t* memory,
                                amdf_memory_info_t* info) {
      auto* self = reinterpret_cast<QueueTest*>(memory);
      info->byte_length = self->storage_.size() * 8;
      info->alignment = 64;
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t* memory, uint32_t,
                                   amdf_memory_address_kind_t,
                                   uint64_t* address) {
      *address = reinterpret_cast<uintptr_t>(
          reinterpret_cast<QueueTest*>(memory)->storage_.data());
      return AMDF_STATUS_OK;
    };
    api_.memory_map = [](amdf_memory_t* memory, const amdf_memory_map_info_t*,
                         amdf_host_mapping_t** mapping) {
      *mapping = reinterpret_cast<amdf_host_mapping_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_query_info = [](amdf_host_mapping_t* mapping,
                                      amdf_host_mapping_info_t* info) {
      auto* self = reinterpret_cast<QueueTest*>(mapping);
      info->pointer = self->storage_.data();
      info->byte_length = self->storage_.size() * 8;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_destroy = [](amdf_host_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* memory) {
      ++reinterpret_cast<QueueTest*>(memory)->destroyed_;
      return AMDF_STATUS_OK;
    };
    api_.instance_destroy = [](amdf_instance_t*) { return AMDF_STATUS_OK; };
    gpu_.kernel_queue_create = [](amdf_device_t* device,
                                  const amdf_gpu_kernel_queue_create_info_t*,
                                  amdf_kernel_queue_t** queue) {
      *queue = reinterpret_cast<amdf_kernel_queue_t*>(device);
      return AMDF_STATUS_OK;
    };
    api_.kernel_queue_query_info = [](amdf_kernel_queue_t* queue,
                                      amdf_kernel_queue_info_t* info) {
      info->maximum_pending_submission_count =
          reinterpret_cast<QueueTest*>(queue)->capacity_;
      info->notification_types = UINT64_C(1) << AMDF_NATIVE_EVENT_TYPE_EVENTFD;
      return AMDF_STATUS_OK;
    };
    gpu_.kernel_queue_submit =
        [](amdf_kernel_queue_t* queue,
           const amdf_gpu_kernel_queue_submission_info_t* info,
           uint64_t* point) {
          auto* self = reinterpret_cast<QueueTest*>(queue);
          if (!amdf_status_is_ok(self->submit_status_)) {
            return self->submit_status_;
          }
          const auto& command = info->commands[0];
          const auto* words = reinterpret_cast<const uint32_t*>(
              reinterpret_cast<const uint8_t*>(self->storage_.data()) +
              command.byte_offset);
          self->last_commands_.assign(words, words + command.byte_length / 4);
          *point = ++self->submitted_;
          return AMDF_STATUS_OK;
        };
    api_.kernel_queue_request_notification =
        [](amdf_kernel_queue_t* queue, uint64_t, const amdf_native_event_t*) {
          return reinterpret_cast<QueueTest*>(queue)->notification_status_;
        };
    api_.kernel_queue_refresh_status = [](amdf_kernel_queue_t* queue,
                                          amdf_kernel_queue_status_t* status) {
      auto* self = reinterpret_cast<QueueTest*>(queue);
      if (self->hold_refresh_.exchange(false)) {
        self->entered_.release();
        self->resume_.acquire();
      }
      status->retired_submission = self->retired_;
      return self->refresh_status_;
    };
    api_.kernel_queue_wait = [](amdf_kernel_queue_t* queue, uint64_t point,
                                uint64_t, uint64_t) {
      auto* self = reinterpret_cast<QueueTest*>(queue);
      return point <= self->retired_
                 ? AMDF_STATUS_OK
                 : amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
    };
    api_.kernel_queue_destroy = [](amdf_kernel_queue_t*) {
      return AMDF_STATUS_OK;
    };
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = iree_allocator_system();
    fabric_->api = &api_;
    context_.fabric = fabric_;
    context_.api = &api_;
    context_.gpu = &gpu_;
    context_.device = reinterpret_cast<amdf_device_t*>(this);
    context_.families = &family_;
    context_.kernel_notification.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    context_.event_sink = {[](void*, const iree_hal_device_event_t*) {},
                           nullptr};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    iree_notification_initialize(&publication_.notification);
  }

  void TearDown() override {
    // Native ownership is joined explicitly even after an observer failure.
    retired_ = submitted_;
    if (queue_) {
      EXPECT_EQ(iree_hal_amd_gpu_sdma_queue_free(queue_), AMDF_STATUS_OK);
    }
    iree_notification_deinitialize(&publication_.notification);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_hal_amd_fabric_release(fabric_);
  }

  void Allocate() {
    const iree_hal_amd_fabric_memory_options_t control = {
        .scope = reinterpret_cast<amdf_memory_scope_t*>(this),
        .profile =
            {
                .allocation =
                    {
                        .maximum_byte_length = UINT32_MAX,
                        .byte_length_granularity = 64,
                        .minimum_alignment = 64,
                        .maximum_alignment = 4096,
                    },
            },
        .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_queue_allocate(
        &context_, 0, AMDF_QUEUE_PUBLICATION_MODE_KERNEL, &control,
        &publication_, 0, iree_allocator_system(), &queue_));
  }

  void Publish() {
    const auto source = iree_hal_amd_gpu_sdma_queue_publication_source(queue_);
    while (source.publish(source.user_data)) {
    }
  }

  void Poll(amdf_status_t status = AMDF_STATUS_OK) {
    const auto source = iree_hal_amd_gpu_sdma_queue_completion_source(queue_);
    while (source.fn(source.user_data, status)) {
    }
  }

  // Native API dependencies, immutable after construction.
  amdf_api_t api_ = {};
  // Native GPU extension dependency.
  amdf_gpu_api_t gpu_ = {};
  // Native control allocation, retained until the production owner releases it.
  std::vector<uint64_t> storage_;
  // Last native command range, captured before return from submission.
  std::vector<uint32_t> last_commands_;
  // Number of completed native backing destructions.
  uint32_t destroyed_ = 0;
  // Native advertised capacity, selected before construction.
  uint32_t capacity_ = 64;
  // Last accepted native submission.
  uint64_t submitted_ = 0;
  // Explicitly checked native retirement frontier.
  uint64_t retired_ = 0;
  // Result of the next native submission dependency call.
  amdf_status_t submit_status_ = AMDF_STATUS_OK;
  // Result of native wake registration.
  amdf_status_t notification_status_ = AMDF_STATUS_OK;
  // Result of checked native retirement refresh.
  amdf_status_t refresh_status_ = AMDF_STATUS_OK;
  // Holds one observer inside its native refresh call.
  std::atomic<bool> hold_refresh_ = false;
  // Observer has entered the held call.
  std::binary_semaphore entered_{0};
  // Releases that native call without fabricating retirement.
  std::binary_semaphore resume_{0};
  // Real fabric domain registry owns the native acquisition/release worker.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Borrowed context passed to the production transport.
  iree_hal_amd_gpu_context_t context_ = {};
  // Exact native encoding capability facts.
  amdf_queue_family_info_t family_ = {};
  // Source readiness storage; tests drive its real source deterministically.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Production object under test.
  iree_hal_amd_gpu_sdma_queue_t* queue_ = nullptr;
};

TEST_F(QueueTest, KernelRetirementOwnsCapacityAndCommandBytes) {
  capacity_ = 2;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  std::array<Copy, 3> copies;
  for (auto& copy : copies) {
    iree_hal_amd_gpu_sdma_queue_submit(queue_, &copy.operation);
  }
  EXPECT_EQ(submitted_, 2u);
  Poll();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 0u);
  }
  retired_ = 1;
  Poll();
  EXPECT_EQ(copies[0].notifications, 1u);
  EXPECT_EQ(copies[0].retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  EXPECT_EQ(copies[1].notifications, 0u);
  Publish();
  EXPECT_EQ(submitted_, 3u);
  retired_ = 3;
  Poll();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 1u);
  }
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_queue_is_idle(queue_));
}

TEST_F(QueueTest, EmptyNativeBusyFailsWithoutInventingAWake) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  submit_status_ = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  Copy copy;
  iree_hal_amd_gpu_sdma_queue_submit(queue_, &copy.operation);
  EXPECT_EQ(copy.notifications, 1u);
  EXPECT_EQ(copy.status, submit_status_);
  EXPECT_EQ(copy.retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_queue_is_idle(queue_));
}

TEST_F(QueueTest,
       FailedObserverRejectsQueuedWorkWithoutReclaimingAcceptedCopies) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  std::array<Copy, 66> copies;
  for (uint32_t i = 0; i < 65; ++i) {
    iree_hal_amd_gpu_sdma_queue_submit(queue_, &copies[i].operation);
  }
  EXPECT_EQ(submitted_, 64u);
  const auto failure = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  Poll(failure);
  Publish();
  for (uint32_t i = 0; i < 64; ++i) {
    EXPECT_EQ(copies[i].notifications, 1u);
    EXPECT_EQ(copies[i].status, failure);
    EXPECT_EQ(copies[i].retirement, IREE_HAL_AMD_GPU_SDMA_UNRETIRED);
  }
  EXPECT_EQ(copies[64].notifications, 1u);
  EXPECT_EQ(copies[64].retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  iree_hal_amd_gpu_sdma_queue_submit(queue_, &copies[65].operation);
  EXPECT_EQ(copies[65].notifications, 1u);
  EXPECT_EQ(copies[65].retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  EXPECT_EQ(submitted_, 64u);
  EXPECT_FALSE(iree_hal_amd_gpu_sdma_queue_is_idle(queue_));
  EXPECT_EQ(iree_hal_amd_gpu_sdma_queue_free(queue_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(destroyed_, 0u);
  Publish();
  Poll();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 1u);
  }
}

TEST_F(QueueTest, FailureReportingJoinsAnActiveObserver) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  std::array<Copy, 2> copies;
  iree_hal_amd_gpu_sdma_queue_submit(queue_, &copies[0].operation);
  hold_refresh_.store(true);
  std::thread observer([&] { Poll(); });
  entered_.acquire();
  notification_status_ = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  iree_hal_amd_gpu_sdma_queue_submit(queue_, &copies[1].operation);
  Publish();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 0u);
  }
  resume_.release();
  observer.join();
  Publish();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 1u);
    EXPECT_EQ(copy.status, notification_status_);
    EXPECT_EQ(copy.retirement, IREE_HAL_AMD_GPU_SDMA_UNRETIRED);
  }
}

TEST_F(QueueTest, BoundedCopiesSplitPacketsAndReuseRetiredCommandSlots) {
  family_.format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  Copy copy;
  copy.operation.length = IREE_HAL_AMD_GPU_SDMA_MAX_COPY_LENGTH;
  copy.operation.effects.bits =
      IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
      IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
  for (uint32_t i = 0; i < 257; ++i) {
    iree_hal_amd_gpu_sdma_queue_submit(queue_, &copy.operation);
    EXPECT_EQ(last_commands_.size(), 5u + 17u * 7u + 5u + 4u);
    uint32_t copied = 0;
    for (uint32_t j = 0; j < 17; ++j) {
      const auto* words = &last_commands_[5 + j * 7];
      EXPECT_EQ(words[0], 1u);
      EXPECT_LE(words[1] + 1, 0x3fff00u);
      EXPECT_EQ(words[3], copy.operation.source + copied);
      EXPECT_EQ(words[5], copy.operation.target + copied);
      copied += words[1] + 1;
    }
    EXPECT_EQ(copied, copy.operation.length);
    EXPECT_EQ(last_commands_.back(), i + 1u);
    retired_ = submitted_;
    Poll();
    EXPECT_EQ(copy.notifications, i + 1u);
  }
}

}  // namespace
