// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/native_queue.h"

#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/event.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Native dependencies expose explicit admission and checked-retirement points.
// The real transport owns encoding, publication, capacity, callbacks, and
// native storage lifetime. Hardware CTS separately verifies the resulting
// payloads.
class QueueTest : public ::testing::Test {
 protected:
  struct Submission {
    // Immutable prepared span retained independently of native command storage.
    std::array<uint32_t, IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS> words = {};
    // Placed record retained through its terminal notification.
    iree_hal_amd_gpu_sdma_submission_t operation = {};
    // Number of irrevocable native acceptances, distinct from retirement.
    uint32_t acceptances = 0;
    // Optional native publication operand sampled by acceptance.
    const uint64_t* native_frontier = nullptr;
    // Native operand value observed before exposing retirement ownership.
    uint64_t observed_frontier = UINT64_MAX;
    // Number of delivered terminal outcomes.
    uint32_t notifications = 0;
    // Observed native result, independent of safe storage reclamation.
    amdf_status_t status = AMDF_STATUS_OK;
    // Whether the callback returned operand ownership.
    iree_hal_amd_gpu_sdma_retirement_t retirement =
        IREE_HAL_AMD_GPU_SDMA_UNRETIRED;
    // Optional native stop proof required before returning accepted operands.
    const bool* native_stopped = nullptr;

    Submission()
        : operation{
              .words = words.data(),
              .word_count = 7,
              .accepted =
                  [](void* user_data) {
                    auto* self = static_cast<Submission*>(user_data);
                    ++self->acceptances;
                    if (self->native_frontier) {
                      self->observed_frontier = *self->native_frontier;
                    }
                  },
              .completion =
                  {
                      .notify =
                          [](void* user_data, amdf_status_t status,
                             iree_hal_amd_gpu_sdma_retirement_t retirement) {
                            auto* self = static_cast<Submission*>(user_data);
                            ++self->notifications;
                            self->status = status;
                            self->retirement = retirement;
                            if (self->native_stopped &&
                                retirement == IREE_HAL_AMD_GPU_SDMA_RETIRED) {
                              EXPECT_TRUE(*self->native_stopped);
                            }
                          },
                      .user_data = this,
                  },
              .user_data = this,
          } {
      iree_hal_amd_gpu_sdma_emit_copy(words.data(), 0, 0x100000, 0x200000,
                                      4096);
    }
  };

  void SetUp() override {
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory,
                            amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
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
    api_.memory_destroy = [](amdf_memory_t* memory,
                             amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
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
      info->notification_types = AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD |
                                 AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
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
    gpu_.kernel_queue_inactivate = [](amdf_kernel_queue_t* queue) {
      auto* self = reinterpret_cast<QueueTest*>(queue);
      EXPECT_EQ(self->destroyed_, 0u);
      ++self->native_stops_;
      self->native_stopped_ = true;
      return AMDF_STATUS_OK;
    };
    gpu_.user_queue_inactivate = [](amdf_user_queue_t* queue) {
      auto* self = reinterpret_cast<QueueTest*>(queue);
      EXPECT_EQ(self->destroyed_, 0u);
      ++self->native_stops_;
      self->native_stopped_ = true;
      return AMDF_STATUS_OK;
    };
    gpu_.user_queue_create = [](amdf_device_t* device,
                                const amdf_gpu_user_queue_create_info_t* info,
                                amdf_user_queue_t** queue) {
      auto* self = reinterpret_cast<QueueTest*>(device);
      self->user_.words.resize(info->ring_byte_length / 4);
      *queue = reinterpret_cast<amdf_user_queue_t*>(self);
      return AMDF_STATUS_OK;
    };
    api_.user_queue_map = [](amdf_user_queue_t* queue, amdf_device_t*,
                             amdf_user_queue_mapping_t** mapping) {
      *mapping = reinterpret_cast<amdf_user_queue_mapping_t*>(queue);
      return AMDF_STATUS_OK;
    };
    api_.user_queue_query_info = [](amdf_user_queue_t*,
                                    amdf_user_queue_info_t* info) {
      info->placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL;
      return AMDF_STATUS_OK;
    };
    api_.user_queue_mapping_query_info =
        [](amdf_user_queue_mapping_t* mapping,
           amdf_user_queue_mapping_info_t* info) {
          auto* self = reinterpret_cast<QueueTest*>(mapping);
          info->ring_address =
              reinterpret_cast<uintptr_t>(self->user_.words.data());
          info->ring_byte_length = self->user_.words.size() * 4;
          info->write_index_address =
              reinterpret_cast<uintptr_t>(&self->user_.write_index);
          info->doorbell_address =
              reinterpret_cast<uintptr_t>(&self->user_.doorbell);
          info->index_bits = info->doorbell_bits = 64;
          info->host_publish_executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
          return AMDF_STATUS_OK;
        };
    api_.user_queue_mapping_flush = [](amdf_user_queue_mapping_t* mapping) {
      auto* self = reinterpret_cast<QueueTest*>(mapping);
      ++self->user_.flush_count;
      EXPECT_EQ(self->user_.write_index, self->user_.doorbell);
      return self->user_.status;
    };
    api_.user_queue_wait_consumed = [](amdf_user_queue_t* queue, uint64_t,
                                       uint64_t, uint64_t) {
      return reinterpret_cast<QueueTest*>(queue)->user_.status;
    };
    api_.user_queue_mapping_destroy = [](amdf_user_queue_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api_.user_queue_destroy = [](amdf_user_queue_t*) { return AMDF_STATUS_OK; };
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
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
        &context_, AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
    context_.event_sink = {[](void*, const iree_hal_device_event_t*) {},
                           nullptr};
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    iree_notification_initialize(&publication_.notification);
  }

  void TearDown() override {
    // Native ownership is joined explicitly even after an observer failure.
    retired_ = submitted_;
    user_.status = AMDF_STATUS_OK;
    if (queue_) {
      EXPECT_EQ(iree_hal_amd_gpu_sdma_native_queue_free(queue_),
                AMDF_STATUS_OK);
    }
    iree_notification_deinitialize(&publication_.notification);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    iree_async_event_native_deinitialize(&context_.kernel_event);
    iree_hal_amd_fabric_release(fabric_);
  }

  void Allocate(amdf_queue_publication_modes_t mode =
                    AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
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
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_native_queue_allocate(
        &context_, 0, mode, &control, &publication_, 0,
        AMDF_QUEUE_PRIORITY_NORMAL, AMDF_QUEUE_PLACEMENT_DEFAULT,
        iree_allocator_system(), &queue_));
  }

  void Publish() {
    const auto source =
        iree_hal_amd_gpu_sdma_native_queue_publication_source(queue_);
    while (source.publish(source.user_data)) {
    }
  }

  iree_hal_amd_gpu_completion_wait_t Poll(
      amdf_status_t status = AMDF_STATUS_OK,
      iree_hal_amd_gpu_completion_copy_t* out_copy = nullptr) {
    auto source = iree_hal_amd_gpu_sdma_native_queue_completion_source(queue_);
    iree_hal_amd_gpu_completion_wait_t wait = {};
    iree_hal_amd_gpu_completion_t completion = {};
    iree_slim_mutex_initialize(&completion.drain_mutex);
    auto poll = [&](amdf_status_t status) {
      wait = {};
      iree_slim_mutex_lock(&completion.drain_mutex);
      iree_hal_amd_gpu_completion_scope_t scope = {
          .completion = &completion,
          .source = &source,
          .fn = source.fn,
      };
      const bool ready =
          source.fn(source.user_data, status, &scope, &wait, out_copy);
      EXPECT_FALSE(iree_hal_amd_gpu_completion_scope_finish(&scope));
      iree_slim_mutex_unlock(&completion.drain_mutex);
      return ready;
    };
    while (poll(status) && !(out_copy && out_copy->fn)) {
    }
    iree_slim_mutex_deinitialize(&completion.drain_mutex);
    EXPECT_FALSE(wait.pending);
    return wait;
  }

  // Native API dependencies, immutable after construction.
  amdf_api_t api_ = {};
  // Native GPU extension dependency.
  amdf_gpu_api_t gpu_ = {};
  // Native control allocation, retained until the production owner releases it.
  std::vector<uint64_t> storage_;
  // Native USER publication operands and release dependency.
  struct {
    // Command storage returned by the native mapping.
    std::vector<uint32_t> words;
    // Coherent producer index, untouched until release succeeds.
    uint64_t write_index = 0;
    // Notification aperture, untouched until index commitment.
    uint64_t doorbell = 0;
    // Number of attempted native releases.
    uint32_t flush_count = 0;
    // Terminal native result independently supplied to release and join.
    amdf_status_t status = AMDF_STATUS_OK;
  } user_;
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
  iree_hal_amd_gpu_sdma_native_queue_t* queue_ = nullptr;
  // Number of native access-cessation calls made before final copy retirement.
  uint32_t native_stops_ = 0;
  // Access-cessation proof checked by operand-release callbacks.
  bool native_stopped_ = false;
};

TEST_F(QueueTest,
       FailedUserReleaseFailsWaitersWithoutReturningAcceptedOperands) {
  ASSERT_NO_FATAL_FAILURE(Allocate(AMDF_QUEUE_PUBLICATION_MODE_USER));
  user_.status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  Submission accepted;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &accepted.operation);
  Publish();
  EXPECT_EQ(user_.flush_count, 1u);
  EXPECT_EQ(user_.write_index, 0u);
  EXPECT_EQ(user_.doorbell, 0u);
  EXPECT_EQ(accepted.acceptances, 0u);
  EXPECT_EQ(accepted.notifications, 1u);
  EXPECT_EQ(accepted.status, user_.status);
  EXPECT_EQ(accepted.retirement, IREE_HAL_AMD_GPU_SDMA_UNRETIRED);
  EXPECT_FALSE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
  EXPECT_EQ(iree_hal_amd_gpu_sdma_native_queue_free(queue_), user_.status);
  EXPECT_EQ(destroyed_, 0u);
  Submission rejected;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &rejected.operation);
  EXPECT_EQ(rejected.acceptances, 0u);
  EXPECT_EQ(rejected.notifications, 1u);
  EXPECT_EQ(rejected.status, user_.status);
  EXPECT_EQ(rejected.retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  EXPECT_EQ(user_.flush_count, 1u);
  Publish();
  EXPECT_EQ(accepted.notifications, 1u);
}

TEST_F(QueueTest, KernelRetirementOwnsCapacityAndCommandBytes) {
  capacity_ = 2;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  const auto idle = Poll();
  EXPECT_EQ(idle.address, nullptr);
  EXPECT_FALSE(idle.pending);
  std::array<Submission, 3> copies;
  for (auto& copy : copies) {
    iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copy.operation);
  }
  EXPECT_EQ(submitted_, 2u);
  EXPECT_EQ(copies[0].acceptances, 1u);
  EXPECT_EQ(copies[1].acceptances, 1u);
  EXPECT_EQ(copies[2].acceptances, 0u);
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
  EXPECT_EQ(copies[2].acceptances, 1u);
  retired_ = 3;
  Poll();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 1u);
  }
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
}

TEST_F(QueueTest, MetadataProgressesPastExhaustedNativeCapacity) {
  capacity_ = 2;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  std::array<Submission, 3> copies;
  for (auto& copy : copies) {
    iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copy.operation);
  }
  EXPECT_EQ(submitted_, 2u);
  bool metadata_processed = false;
  iree_hal_amd_gpu_sdma_metadata_t metadata = {
      .process =
          [](void* user_data, amdf_status_t status) {
            *static_cast<bool*>(user_data) = true;
          },
      .user_data = &metadata_processed,
  };
  iree_hal_amd_gpu_sdma_native_queue_defer_metadata(queue_, &metadata);
  Publish();
  EXPECT_TRUE(metadata_processed);
  EXPECT_EQ(submitted_, 2u);
  EXPECT_EQ(retired_, 0u);
  EXPECT_EQ(copies[2].acceptances, 0u);
  retired_ = 2;
  Poll();
  Publish();
  EXPECT_EQ(submitted_, 3u);
  retired_ = 3;
  Poll();
  for (auto& copy : copies) {
    EXPECT_EQ(copy.notifications, 1u);
  }
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
}

TEST_F(QueueTest, UserAcceptanceFollowsReleaseAndPrecedesIndexCommitment) {
  ASSERT_NO_FATAL_FAILURE(Allocate(AMDF_QUEUE_PUBLICATION_MODE_USER));
  const auto idle = Poll();
  ASSERT_NE(idle.address, nullptr);
  EXPECT_TRUE(idle.is_64_bit);
  EXPECT_FALSE(idle.pending);
  Submission submission;
  submission.native_frontier = &user_.write_index;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &submission.operation);
  EXPECT_NE(
      iree_atomic_load(static_cast<const iree_atomic_int64_t*>(idle.address),
                       iree_memory_order_acquire),
      idle.value);
  EXPECT_EQ(submission.acceptances, 1u);
  EXPECT_EQ(submission.observed_frontier, 0u);
  EXPECT_EQ(user_.flush_count, 1u);
  EXPECT_GT(user_.write_index, 0u);
  EXPECT_EQ(user_.doorbell, user_.write_index);
  EXPECT_EQ(submission.notifications, 0u);
  // Retire through failure cessation so this test does not pretend that
  // writing the native index establishes execution completion.
  Poll(amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  Publish();
  amdf_status_t status = AMDF_STATUS_OK;
  ASSERT_TRUE(
      iree_hal_amd_gpu_sdma_native_queue_try_inactivate(queue_, &status));
  EXPECT_EQ(status, AMDF_STATUS_OK);
  ASSERT_TRUE(iree_hal_amd_gpu_sdma_native_queue_retire_failed(queue_));
  EXPECT_EQ(submission.acceptances, 1u);
  EXPECT_EQ(submission.retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
}

TEST_F(QueueTest, KernelCompletionCannotReturnCaptureDuringAcceptance) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  using Owner = decltype(this);
  struct Observation {
    // Actual queue fixture owning the native dependency controls.
    Owner owner;
    // Acceptance callback has finished publishing producer metadata.
    bool accepted = false;
    // Number of safely returned command captures.
    uint32_t retirements = 0;
  } observation = {this};
  iree_hal_amd_gpu_sdma_submission_t submission = {
      .accepted =
          [](void* user_data) {
            auto* observation = static_cast<Observation*>(user_data);
            EXPECT_EQ(observation->owner->submitted_, 1u);
            observation->owner->retired_ = 1;
            observation->owner->Poll();
            EXPECT_EQ(observation->retirements, 0u);
            observation->accepted = true;
          },
      .completion =
          {
              .notify =
                  [](void* user_data, amdf_status_t status,
                     iree_hal_amd_gpu_sdma_retirement_t retirement) {
                    auto* observation = static_cast<Observation*>(user_data);
                    EXPECT_TRUE(observation->accepted);
                    EXPECT_EQ(status, AMDF_STATUS_OK);
                    EXPECT_EQ(retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
                    ++observation->retirements;
                  },
              .user_data = &observation,
          },
      .user_data = &observation,
  };
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &submission);
  EXPECT_TRUE(observation.accepted);
  EXPECT_EQ(observation.retirements, 0u);
  // An empty span still owns a native completion and its payload boundary.
  ASSERT_EQ(last_commands_.size(), 4u);
  EXPECT_EQ(last_commands_[0], 5u);
  Poll();
  EXPECT_EQ(observation.retirements, 1u);
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
}

TEST_F(QueueTest, EmptyNativeBusyFailsWithoutInventingAWake) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  submit_status_ = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  Submission copy;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copy.operation);
  EXPECT_EQ(copy.notifications, 1u);
  EXPECT_EQ(copy.acceptances, 0u);
  EXPECT_EQ(copy.status, submit_status_);
  EXPECT_EQ(copy.retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
}

TEST_F(QueueTest,
       FailedObserverRejectsQueuedWorkWithoutReclaimingAcceptedCopies) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  std::array<Submission, 80> copies;
  for (uint32_t i = 0; i < 79; ++i) {
    iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copies[i].operation);
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
  for (uint32_t i = 64; i < 79; ++i) {
    EXPECT_EQ(copies[i].notifications, 1u);
    EXPECT_EQ(copies[i].status, failure);
    EXPECT_EQ(copies[i].retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  }
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copies[79].operation);
  EXPECT_EQ(copies[79].notifications, 1u);
  EXPECT_EQ(copies[79].retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
  EXPECT_EQ(submitted_, 64u);
  EXPECT_FALSE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
  EXPECT_EQ(iree_hal_amd_gpu_sdma_native_queue_free(queue_),
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
  std::array<Submission, 2> copies;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copies[0].operation);
  hold_refresh_.store(true);
  std::thread observer([&] { Poll(); });
  entered_.acquire();
  EXPECT_FALSE(iree_hal_amd_gpu_sdma_native_queue_is_host_idle(queue_));
  notification_status_ = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copies[1].operation);
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

TEST_F(QueueTest, NativeStopRetiresFaultedCopiesWithoutCompletionWrites) {
  for (const auto mode :
       {AMDF_QUEUE_PUBLICATION_MODE_USER, AMDF_QUEUE_PUBLICATION_MODE_KERNEL}) {
    SCOPED_TRACE(mode);
    ASSERT_NO_FATAL_FAILURE(Allocate(mode));
    std::array<Submission, 2> copies;
    for (auto& copy : copies) {
      copy.native_stopped = &native_stopped_;
      iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copy.operation);
    }
    const auto failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
    Poll(failure);
    Publish();
    EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_host_idle(queue_));
    EXPECT_FALSE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
    for (auto& copy : copies) {
      EXPECT_EQ(copy.retirement, IREE_HAL_AMD_GPU_SDMA_UNRETIRED);
    }
    amdf_status_t status = AMDF_STATUS_OK;
    ASSERT_TRUE(
        iree_hal_amd_gpu_sdma_native_queue_try_inactivate(queue_, &status));
    EXPECT_EQ(status, AMDF_STATUS_OK);
    EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_retire_failed(queue_));
    for (auto& copy : copies) {
      EXPECT_EQ(copy.status, failure);
      EXPECT_EQ(copy.retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
      EXPECT_EQ(copy.notifications, 2u);
    }
    EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
    ASSERT_TRUE(
        iree_hal_amd_gpu_sdma_native_queue_try_inactivate(queue_, &status));
    EXPECT_EQ(native_stops_, 1u);
    EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_retire_failed(queue_));
    user_.status = failure;
    EXPECT_EQ(iree_hal_amd_gpu_sdma_native_queue_free(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
    native_stops_ = destroyed_ = 0;
    native_stopped_ = false;
    user_.status = AMDF_STATUS_OK;
  }
}

TEST_F(QueueTest, PreparedCommandSpansReuseRetiredNativeSlots) {
  family_.format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  Submission submission;
  auto* cursor =
      iree_hal_amd_gpu_sdma_emit_cache(submission.words.data(), 0xc3c0);
  constexpr uint32_t kLength = 64 * 1024 * 1024;
  constexpr uint64_t kSource = 0x1234fffffff1;
  constexpr uint64_t kTarget = 0x567800000003;
  for (uint32_t offset = 0; offset < kLength;) {
    const uint32_t length =
        iree_min(kLength - offset, IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
    cursor = iree_hal_amd_gpu_sdma_emit_copy(cursor, family_.format_features,
                                             kSource + offset, kTarget + offset,
                                             length);
    offset += length;
  }
  cursor = iree_hal_amd_gpu_sdma_emit_cache(cursor, 0x8040);
  const uint32_t word_count = cursor - submission.words.data();
  const auto original_words = submission.words;
  for (uint32_t i = 0; i < 257; ++i) {
    // Alternate an exact span and a full slot with trailing native NOPs.
    submission.operation.word_count =
        (i & 1) ? IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS : word_count;
    iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &submission.operation);
    ASSERT_EQ(last_commands_.size(), submission.operation.word_count + 4u);
    uint32_t copied = 0;
    for (uint32_t j = 0; j < 17; ++j) {
      const auto* words = &last_commands_[5 + j * 7];
      EXPECT_EQ(words[0], 1u);
      EXPECT_LE(words[1] + 1, 0x3fff00u);
      EXPECT_EQ(uint64_t{words[3]} | (uint64_t{words[4]} << 32),
                kSource + copied);
      EXPECT_EQ(uint64_t{words[5]} | (uint64_t{words[6]} << 32),
                kTarget + copied);
      copied += words[1] + 1;
    }
    EXPECT_EQ(copied, kLength);
    EXPECT_EQ(last_commands_.back(), i + 1u);
    EXPECT_EQ(submission.acceptances, i + 1u);
    retired_ = submitted_;
    Poll();
    EXPECT_EQ(submission.notifications, i + 1u);
    EXPECT_EQ(submission.words, original_words);
  }
}

TEST_F(QueueTest, SubmissionCacheTransitionsRetireAtNativePoint) {
  family_.format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  // KERNEL cache work belongs to the submission boundary. The prepared
  // commands contain only the copy, and its in-stream completion is
  // insufficient.
  Submission copy;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &copy.operation);
  ASSERT_EQ(last_commands_.size(), 11u);
  EXPECT_EQ(last_commands_,
            (std::vector<uint32_t>{1, 4095, 0, 0x100000, 0, 0x200000, 0, 5,
                                   last_commands_[8], last_commands_[9], 1}));
  const uint64_t completion_address =
      last_commands_[8] | (uint64_t{last_commands_[9]} << 32);
  // The in-stream marker may precede the native cache-release epilogue.
  // Publishing that marker alone must not return the operands to the caller.
  iree_atomic_store(reinterpret_cast<iree_atomic_int32_t*>(completion_address),
                    1, iree_memory_order_release);
  Poll();
  EXPECT_EQ(copy.notifications, 0u);
  retired_ = submitted_;
  Poll();
  EXPECT_EQ(copy.notifications, 1u);
  EXPECT_EQ(copy.status, AMDF_STATUS_OK);
  EXPECT_EQ(copy.retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
}

TEST_F(QueueTest, AcceptanceDetachesReusablePublicationFromEveryRetirement) {
  capacity_ = 2;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  struct Publication {
    // Actual native publisher, shared by both command spans.
    iree_hal_amd_gpu_sdma_native_queue_t* queue;
    // Reused caller entry, destroyed immediately after final acceptance.
    iree_hal_amd_gpu_sdma_submission_t* entry;
    // Separate per-span lifetime and outcome observations.
    std::array<Submission, 2> retirements;
    // Number of accepted spans, independent of native completion.
    uint32_t accepted = 0;
  } publication = {queue_, new iree_hal_amd_gpu_sdma_submission_t{}};
  auto* entry = publication.entry;
  entry->words = publication.retirements[0].words.data();
  entry->word_count = 7;
  entry->completion = publication.retirements[0].operation.completion;
  entry->user_data = &publication;
  entry->accepted = [](void* user_data) {
    auto* self = static_cast<Publication*>(user_data);
    if (++self->accepted == 2) {
      delete self->entry;
      self->entry = nullptr;
    } else {
      self->entry->word_count = 0;
      self->entry->completion = self->retirements[1].operation.completion;
      iree_hal_amd_gpu_sdma_native_queue_defer(self->queue, self->entry);
    }
  };
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, entry);
  EXPECT_EQ(publication.accepted, 1u);
  const auto first = last_commands_;
  Publish();
  EXPECT_EQ(publication.accepted, 2u);
  EXPECT_EQ(publication.entry, nullptr);
  EXPECT_EQ(submitted_, 2u);
  // The first KERNEL command slot is immutable despite entry reuse/destruction.
  EXPECT_EQ(memcmp(storage_.data(), first.data(), first.size() * 4), 0);
  EXPECT_EQ(publication.retirements[0].notifications, 0u);
  EXPECT_EQ(publication.retirements[1].notifications, 0u);
  retired_ = 1;
  Poll();
  EXPECT_EQ(publication.retirements[0].notifications, 1u);
  EXPECT_EQ(publication.retirements[1].notifications, 0u);
  retired_ = 2;
  Poll();
  EXPECT_EQ(publication.retirements[1].notifications, 1u);
}

TEST_F(QueueTest, PreparationParksOnlyItsOperationAndDefersCallbackWork) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  struct Pending {
    // Span retaining its operands while dependency readiness is unresolved.
    Submission span;
    // Explicit external dependency state, independent of native capacity.
    bool ready = false;
    // Preparation entries observed through the real exclusive publisher.
    uint32_t attempts = 0;
    // Native position offered by the most recent preparation turn.
    uint64_t position = UINT64_MAX;
  } pending;
  pending.span.operation.user_data = &pending;
  pending.span.operation.accepted = nullptr;
  pending.span.operation.prepare = [](void* data, uint64_t position) {
    auto* self = static_cast<Pending*>(data);
    ++self->attempts;
    self->position = position;
    return self->ready;
  };
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &pending.span.operation);
  EXPECT_EQ(pending.attempts, 1u);
  EXPECT_EQ(submitted_, 0u);
  Submission independent;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &independent.operation);
  EXPECT_EQ(submitted_, 1u);
  pending.ready = true;
  iree_hal_amd_gpu_sdma_native_queue_defer(queue_, &pending.span.operation);
  // An allocation/dependency callback only enqueues; native work belongs to
  // the existing publication worker's subsequent turn.
  EXPECT_EQ(pending.attempts, 1u);
  EXPECT_EQ(submitted_, 1u);
  Publish();
  EXPECT_EQ(pending.attempts, 2u);
  EXPECT_EQ(pending.position, 1u);
  EXPECT_EQ(submitted_, 2u);
  retired_ = 2;
  Poll();
  EXPECT_EQ(independent.notifications, 1u);
  EXPECT_EQ(pending.span.notifications, 1u);
}

TEST_F(QueueTest, PreparedCandidateSurvivesNativeCapacityRetry) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  Submission first, second, third;
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &first.operation);
  uint32_t preparations = 0;
  second.operation.accepted = nullptr;
  second.operation.user_data = &preparations;
  second.operation.prepare = [](void* data, uint64_t position) {
    EXPECT_EQ(position, 1u);
    ++*static_cast<uint32_t*>(data);
    return true;
  };
  submit_status_ = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &second.operation);
  EXPECT_EQ(preparations, 1u);
  EXPECT_EQ(submitted_, 1u);
  const std::vector<uint64_t> prepared(storage_.begin() + 128,
                                       storage_.begin() + 256);
  uint32_t later_preparations = 0;
  third.operation.accepted = nullptr;
  third.operation.user_data = &later_preparations;
  third.operation.prepare = [](void* data, uint64_t position) {
    EXPECT_EQ(position, 2u);
    ++*static_cast<uint32_t*>(data);
    return true;
  };
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &third.operation);
  bool metadata_processed = false;
  iree_hal_amd_gpu_sdma_metadata_t metadata = {
      .process =
          [](void* data, amdf_status_t status) {
            EXPECT_EQ(status, AMDF_STATUS_OK);
            *static_cast<bool*>(data) = true;
          },
      .user_data = &metadata_processed,
  };
  iree_hal_amd_gpu_sdma_native_queue_defer_metadata(queue_, &metadata);
  Publish();
  EXPECT_TRUE(metadata_processed);
  EXPECT_EQ(later_preparations, 0u);
  EXPECT_EQ(preparations, 1u);
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(std::memcmp(storage_.data() + 128, prepared.data(),
                        prepared.size() * sizeof(uint64_t)),
            0);
  submit_status_ = AMDF_STATUS_OK;
  retired_ = 1;
  Poll();
  Publish();
  EXPECT_EQ(preparations, 1u);
  EXPECT_EQ(later_preparations, 1u);
  EXPECT_EQ(submitted_, 3u);
  EXPECT_EQ(std::memcmp(storage_.data() + 128, prepared.data(),
                        prepared.size() * sizeof(uint64_t)),
            0);
  retired_ = 3;
  Poll();
  EXPECT_EQ(first.notifications, 1u);
  EXPECT_EQ(second.notifications, 1u);
  EXPECT_EQ(third.notifications, 1u);
}

TEST_F(QueueTest, NativeCompletionJoinsPayloadAndWorkerCopyBeforeRetirement) {
  ASSERT_NO_FATAL_FAILURE(Allocate(AMDF_QUEUE_PUBLICATION_MODE_USER));
  struct Payload {
    // Final native outcome observation retained through the worker copy.
    Submission span;
    // Producer payload cleanup, later than its device completion marker.
    bool dependencies_retired = false;
    // Worker-owned download action has finished its bounded host access.
    bool copied = false;
  } payload;
  payload.span.operation.completion = {
      .ready =
          [](void* data, iree_hal_amd_gpu_completion_copy_t* out_copy) {
            auto* self = static_cast<Payload*>(data);
            if (!self->dependencies_retired) {
              return false;
            }
            if (!self->copied && out_copy) {
              *out_copy = {
                  .fn =
                      [](void* data) {
                        static_cast<Payload*>(data)->copied = true;
                      },
                  .user_data = self,
              };
            }
            return self->copied;
          },
      .notify =
          [](void* data, amdf_status_t status,
             iree_hal_amd_gpu_sdma_retirement_t retirement) {
            auto* self = static_cast<Payload*>(data);
            EXPECT_EQ(status, AMDF_STATUS_OK);
            EXPECT_EQ(retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
            EXPECT_TRUE(self->copied);
            ++self->span.notifications;
          },
      .user_data = &payload,
  };
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &payload.span.operation);
  ASSERT_EQ(user_.words[7], 5u);
  const uint64_t address = user_.words[8] | (uint64_t{user_.words[9]} << 32);
  iree_atomic_store(reinterpret_cast<iree_atomic_int32_t*>(address), 1,
                    iree_memory_order_release);
  Poll();
  EXPECT_EQ(payload.span.notifications, 0u);
  payload.dependencies_retired = true;
  Poll();  // A helping waiter cannot take the bulk-copy action.
  EXPECT_FALSE(payload.copied);
  iree_hal_amd_gpu_completion_copy_t copy = {};
  Poll(AMDF_STATUS_OK, &copy);
  ASSERT_NE(copy.fn, nullptr);
  EXPECT_EQ(payload.span.notifications, 0u);
  EXPECT_FALSE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
  copy.fn(copy.user_data);
  Poll();
  EXPECT_EQ(payload.span.notifications, 1u);
  EXPECT_TRUE(iree_hal_amd_gpu_sdma_native_queue_is_idle(queue_));
}

TEST_F(QueueTest, RetirementSlotRemainsOwnedThroughNotificationTail) {
  capacity_ = 1;
  ASSERT_NO_FATAL_FAILURE(Allocate());
  struct Tail {
    // Signals the last callback's live access to its slot-owned metadata.
    std::binary_semaphore entered{0};
    // Explicitly releases that access before returning native capacity.
    std::binary_semaphore resume{0};
  } tail;
  Submission first, second;
  first.operation.completion = {
      .notify =
          [](void* data, amdf_status_t status,
             iree_hal_amd_gpu_sdma_retirement_t retirement) {
            EXPECT_EQ(status, AMDF_STATUS_OK);
            EXPECT_EQ(retirement, IREE_HAL_AMD_GPU_SDMA_RETIRED);
            auto* self = static_cast<Tail*>(data);
            self->entered.release();
            self->resume.acquire();
          },
      .user_data = &tail,
  };
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &first.operation);
  retired_ = 1;
  std::thread observer([&] { Poll(); });
  tail.entered.acquire();
  iree_hal_amd_gpu_sdma_native_queue_submit(queue_, &second.operation);
  EXPECT_EQ(submitted_, 1u);
  EXPECT_EQ(second.acceptances, 0u);
  tail.resume.release();
  observer.join();
  Publish();
  EXPECT_EQ(submitted_, 2u);
  EXPECT_EQ(second.acceptances, 1u);
  retired_ = 2;
  Poll();
  EXPECT_EQ(second.notifications, 1u);
}

}  // namespace
