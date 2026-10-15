// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/kmq/storage.h"

#include <array>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Native API dependency only: the subject owns real command reservations,
// pending descriptors, accepted points, and cleanup. Native execution and
// notification delivery are exercised by the GPU CTS.
struct NativeQueue {
  // Coherent executable allocation returned through the fabric API.
  std::array<uint32_t, 1024> words = {};
  // Ordered native lifetime and publication calls.
  std::vector<std::string> calls;
  // Operation to reject, or empty for successful native calls.
  std::string fail_operation;
  // Exact native result returned at the selected failure boundary.
  amdf_status_t failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  // Independent native point sequence, deliberately different from slot
  // indices.
  uint64_t next_point = 17;
  // Checked native progress, never inferred from command bytes or
  // notifications.
  uint64_t retired = 0;
  // Terminal execution failure returned inside a successful status query.
  amdf_status_t terminal_status = AMDF_STATUS_OK;
  // Complete descriptor consumed during the last attempted submission.
  amdf_gpu_kernel_command_t submitted = {};
  // Notification targets registered by the completion observer.
  std::vector<uint64_t> arms;
  // Final accepted native point joined during cleanup.
  uint64_t waited = 0;
  // Whether native registration observes completion before it returns.
  bool retire_during_arm = false;

  amdf_status_t Call(const char* operation) {
    calls.emplace_back(operation);
    return fail_operation == operation ? failure : AMDF_STATUS_OK;
  }

  static NativeQueue* From(amdf_kernel_queue_t* queue) {
    return reinterpret_cast<NativeQueue*>(queue);
  }
  static amdf_status_t AMDF_CALL
  Create(amdf_device_t* device, const amdf_gpu_kernel_queue_create_info_t* info,
         amdf_kernel_queue_t** out_queue) {
    auto* self = reinterpret_cast<NativeQueue*>(device);
    EXPECT_EQ(info->queue_family_ordinal, 3u);
    EXPECT_EQ(info->maximum_pending_submission_count, 4u);
    amdf_status_t status = self->Call("create");
    if (amdf_status_is_ok(status)) {
      *out_queue = reinterpret_cast<amdf_kernel_queue_t*>(self);
    }
    return status;
  }
  static amdf_status_t AMDF_CALL QueryInfo(amdf_kernel_queue_t* queue,
                                           amdf_kernel_queue_info_t* info) {
    info->notification_types = AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT;
    return From(queue)->Call("query");
  }
  static amdf_status_t AMDF_CALL
  Submit(amdf_kernel_queue_t* queue,
         const amdf_gpu_kernel_queue_submission_info_t* info, uint64_t* point) {
    auto* self = From(queue);
    EXPECT_EQ(info->command_count, 1u);
    self->submitted = info->commands[0];
    amdf_status_t status = self->Call("submit");
    if (amdf_status_is_ok(status)) {
      *point = self->next_point;
      self->next_point += 13;
    }
    return status;
  }
  static amdf_status_t AMDF_CALL Refresh(amdf_kernel_queue_t* queue,
                                         amdf_kernel_queue_status_t* status) {
    auto* self = From(queue);
    status->retired_submission = self->retired;
    status->terminal_status = self->terminal_status;
    return self->Call("refresh");
  }
  static amdf_status_t AMDF_CALL Arm(amdf_kernel_queue_t* queue, uint64_t point,
                                     const amdf_native_event_t* event) {
    auto* self = From(queue);
    EXPECT_EQ(event->type, AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT);
    amdf_status_t status = self->Call("arm");
    if (amdf_status_is_ok(status)) {
      self->arms.push_back(point);
      if (self->retire_during_arm) {
        self->retired = point;
      }
    }
    return status;
  }
  static amdf_status_t AMDF_CALL Wait(amdf_kernel_queue_t* queue,
                                      uint64_t point, uint64_t timeout,
                                      uint64_t poll_duration) {
    auto* self = From(queue);
    EXPECT_EQ(timeout, AMDF_TIMEOUT_INFINITE);
    EXPECT_EQ(poll_duration, 0u);
    self->waited = point;
    return self->Call("wait");
  }
  static amdf_status_t AMDF_CALL Destroy(amdf_kernel_queue_t* queue) {
    return From(queue)->Call("destroy");
  }
  static amdf_status_t AMDF_CALL Allocate(amdf_memory_scope_t* scope,
                                          const amdf_memory_create_info_t* info,
                                          amdf_memory_t** memory) {
    auto* self = reinterpret_cast<NativeQueue*>(scope);
    EXPECT_EQ(info->byte_length, sizeof(self->words));
    amdf_status_t status = self->Call("allocate");
    if (amdf_status_is_ok(status)) {
      *memory = reinterpret_cast<amdf_memory_t*>(self);
    }
    return status;
  }
  static amdf_status_t AMDF_CALL MemoryInfo(amdf_memory_t* memory,
                                            amdf_memory_info_t* info) {
    auto* self = reinterpret_cast<NativeQueue*>(memory);
    info->byte_length = sizeof(self->words);
    return self->Call("memory_info");
  }
  static amdf_status_t AMDF_CALL Map(amdf_memory_t* memory,
                                     const amdf_memory_map_info_t* info,
                                     amdf_host_mapping_t** mapping) {
    auto* self = reinterpret_cast<NativeQueue*>(memory);
    EXPECT_EQ(info->byte_length, sizeof(self->words));
    *mapping = reinterpret_cast<amdf_host_mapping_t*>(self);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MappingInfo(amdf_host_mapping_t* mapping,
                                             amdf_host_mapping_info_t* info) {
    auto* self = reinterpret_cast<NativeQueue*>(mapping);
    info->pointer = self->words.data();
    info->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Unmap(amdf_host_mapping_t* mapping) {
    return reinterpret_cast<NativeQueue*>(mapping)->Call("unmap");
  }
  static amdf_status_t AMDF_CALL Free(amdf_memory_t* memory) {
    return reinterpret_cast<NativeQueue*>(memory)->Call("free");
  }
};

class KernelStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_.kernel_queue_query_info = NativeQueue::QueryInfo;
    api_.kernel_queue_refresh_status = NativeQueue::Refresh;
    api_.kernel_queue_request_notification = NativeQueue::Arm;
    api_.kernel_queue_wait = NativeQueue::Wait;
    api_.kernel_queue_destroy = NativeQueue::Destroy;
    api_.memory_create = NativeQueue::Allocate;
    api_.memory_query_info = NativeQueue::MemoryInfo;
    api_.memory_map = NativeQueue::Map;
    api_.host_mapping_query_info = NativeQueue::MappingInfo;
    api_.host_mapping_destroy = NativeQueue::Unmap;
    api_.memory_destroy = NativeQueue::Free;
    gpu_.kernel_queue_create = NativeQueue::Create;
    gpu_.kernel_queue_submit = NativeQueue::Submit;
    context_.api = &api_;
    context_.gpu = &gpu_;
    context_.device = reinterpret_cast<amdf_device_t*>(&native_);
    context_.kernel_notification.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
    options_.scope = reinterpret_cast<amdf_memory_scope_t*>(&native_);
    options_.profile.allocation.byte_length_granularity = 64;
    options_.profile.allocation.maximum_byte_length = sizeof(native_.words);
    options_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  }
  void TearDown() override {
    if (storage_) {
      EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_free(storage_),
                AMDF_STATUS_OK);
    }
  }
  iree_status_t Allocate() {
    return iree_hal_amd_gpu_pm4_kmq_storage_allocate(
        &context_, 3, sizeof(native_.words), 4, &options_,
        iree_allocator_system(), &storage_, &ring_);
  }
  void Prepare() {
    iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(
        &ring_, ring_.reserved_words, 30, &reservation));
    storage_->pending = {
        .memory = storage_->memory.handle,
        .byte_offset = uint64_t(reservation.words - ring_.words) * 4,
        .byte_length = 128,
    };
  }

  // Native dependency and API tables borrowed by the real owner.
  NativeQueue native_;
  // Core native calls consumed through the captured context.
  amdf_api_t api_ = {};
  // Protocol-specific create and submit calls.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed device facts and notification destination.
  iree_hal_amd_gpu_context_t context_ = {};
  // Coherent executable memory policy.
  iree_hal_amd_fabric_memory_options_t options_ = {};
  // Actual owner under test, including partial construction failures.
  iree_hal_amd_gpu_pm4_kmq_storage_t* storage_ = nullptr;
  // Hot command reservation state, independent of native resource ownership.
  iree_hal_amd_gpu_pm4_ring_t ring_ = {};
};

TEST_F(KernelStorageTest, CapturesOpaquePointsWithoutInferringProgress) {
  IREE_ASSERT_OK(Allocate());
  EXPECT_EQ(ring_.words, native_.words.data());
  EXPECT_EQ(ring_.capacity, native_.words.size());
  for (uint64_t i = 0; i < 12; ++i) {
    Prepare();
    EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, i),
              AMDF_STATUS_OK);
    const uint64_t point = 17 + i * 13;
    EXPECT_EQ(storage_->points[i & 3], point);
    EXPECT_EQ(storage_->pending.byte_length, 0u);
    EXPECT_EQ(native_.submitted.byte_length, 128u);
    EXPECT_EQ(native_.submitted.byte_offset, i * 128);
    uint64_t retired = UINT64_MAX;
    EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_refresh(storage_, &retired),
              AMDF_STATUS_OK);
    EXPECT_EQ(retired, i ? point - 13 : 0);
    native_.retired = point;
    EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_refresh(storage_, &retired),
              AMDF_STATUS_OK);
    EXPECT_EQ(retired, point);
  }
}

TEST_F(KernelStorageTest, BusyRetainsTheExactPreparedPiece) {
  IREE_ASSERT_OK(Allocate());
  Prepare();
  const auto pending = storage_->pending;
  native_.fail_operation = "submit";
  native_.failure = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, 0),
            native_.failure);
  EXPECT_EQ(storage_->pending.memory, pending.memory);
  EXPECT_EQ(storage_->pending.byte_offset, pending.byte_offset);
  EXPECT_EQ(storage_->pending.byte_length, pending.byte_length);
  EXPECT_EQ(storage_->last_submission, 0u);
  EXPECT_EQ(storage_->points[0], 0u);
  native_.fail_operation.clear();
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(storage_->points[0], 17u);
  EXPECT_EQ(native_.submitted.byte_offset, pending.byte_offset);
  EXPECT_EQ(native_.submitted.byte_length, pending.byte_length);
}

TEST_F(KernelStorageTest, OneArmPerPointIncludingCompletionDuringRegistration) {
  IREE_ASSERT_OK(Allocate());
  Prepare();
  ASSERT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, 0),
            AMDF_STATUS_OK);
  native_.retire_during_arm = true;
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_arm(storage_, 17),
              AMDF_STATUS_OK);
  }
  EXPECT_EQ(native_.arms, (std::vector<uint64_t>{17}));
  uint64_t retired = 0;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_refresh(storage_, &retired),
            AMDF_STATUS_OK);
  EXPECT_EQ(retired, 17u);
  Prepare();
  ASSERT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, 1),
            AMDF_STATUS_OK);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_arm(storage_, 30), AMDF_STATUS_OK);
  EXPECT_EQ(native_.arms, (std::vector<uint64_t>{17, 30}));
}

TEST_F(KernelStorageTest, FailedArmAndTerminalStatusRemainFailures) {
  IREE_ASSERT_OK(Allocate());
  Prepare();
  ASSERT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, 0),
            AMDF_STATUS_OK);
  native_.fail_operation = "arm";
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_arm(storage_, 17),
            native_.failure);
  EXPECT_EQ(storage_->armed_submission, 0u);
  EXPECT_EQ(storage_->last_submission, 17u);
  native_.fail_operation.clear();
  native_.terminal_status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  uint64_t retired = 0;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_refresh(storage_, &retired),
            native_.terminal_status);
}

TEST_F(KernelStorageTest, CleanupJoinsAcceptedTailBeforeReleasingCommandBytes) {
  IREE_ASSERT_OK(Allocate());
  Prepare();
  ASSERT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_submit(storage_, 0),
            AMDF_STATUS_OK);
  native_.calls.clear();
  native_.fail_operation = "wait";
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_free(storage_), native_.failure);
  EXPECT_NE(storage_->queue, nullptr);
  EXPECT_NE(storage_->memory.handle, nullptr);
  EXPECT_EQ(native_.calls, (std::vector<std::string>{"wait"}));
  native_.fail_operation.clear();
  native_.calls.clear();
  EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_free(storage_), AMDF_STATUS_OK);
  storage_ = nullptr;
  EXPECT_EQ(native_.waited, 17u);
  EXPECT_EQ(native_.calls,
            (std::vector<std::string>{"wait", "destroy", "unmap", "free"}));
}

TEST_F(KernelStorageTest, PartialConstructionPreservesItsCleanupOwner) {
  for (const char* operation : {"create", "query", "allocate", "memory_info"}) {
    SCOPED_TRACE(operation);
    native_.fail_operation = operation;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Allocate());
    ASSERT_NE(storage_, nullptr);
    EXPECT_EQ(ring_.words, nullptr);
    native_.fail_operation.clear();
    EXPECT_EQ(iree_hal_amd_gpu_pm4_kmq_storage_free(storage_), AMDF_STATUS_OK);
    storage_ = nullptr;
  }
}

}  // namespace
