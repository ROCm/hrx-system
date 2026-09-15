// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/user_queue.h"

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/device.h"

namespace {

struct FakeMapping {
  amdf_user_queue_mapping_t base;
  amdf_status_t destroy_status;
};

struct FakeQueue {
  amdf_user_queue_t base;
  amdf_status_t map_status;
  amdf_status_t query_status;
  amdf_status_t wait_status;
  amdf_status_t destroy_status;
  uint64_t waited_index;
};

static amdf_status_t FakeMappingDestroyNative(
    amdf_user_queue_mapping_t* base_mapping) {
  return reinterpret_cast<FakeMapping*>(base_mapping)->destroy_status;
}

static const amdf_user_queue_mapping_vtable_t kFakeMappingVtable = {
    .destroy_native = FakeMappingDestroyNative,
};

static amdf_status_t FakeMap(amdf_user_queue_t* base_queue,
                             amdf_device_t* producer_device,
                             amdf_user_queue_mapping_t** out_mapping) {
  auto* queue = reinterpret_cast<FakeQueue*>(base_queue);
  if (!amdf_status_is_ok(queue->map_status)) return queue->map_status;
  FakeMapping* mapping = nullptr;
  amdf_status_t status = amdf_calloc(
      base_queue->host_allocator, sizeof(*mapping), amdf_alignof(FakeMapping),
      reinterpret_cast<void**>(&mapping));
  if (!amdf_status_is_ok(status)) return status;
  amdf_user_queue_mapping_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      .structure_size = sizeof(info),
      .queue_id = base_queue->info.queue_id,
      .queue_reset_epoch = base_queue->info.reset_epoch,
      .command_type = base_queue->info.command_type,
      .format_version = base_queue->info.format_version,
      .ring_address = UINT64_C(0x1000),
      .ring_byte_length = base_queue->info.ring_byte_length,
      .read_index_address = UINT64_C(0x2000),
      .write_index_address = UINT64_C(0x2008),
      .doorbell_address = UINT64_C(0x3000),
      .index_bits = 64,
      .doorbell_bits = 64,
  };
  if (producer_device != nullptr) info.producer_device_id.words[0] = 17;
  status = amdf_user_queue_mapping_initialize(
      &mapping->base, &kFakeMappingVtable, base_queue, producer_device, &info);
  if (amdf_status_is_ok(status)) {
    *out_mapping = &mapping->base;
  } else {
    amdf_free(base_queue->host_allocator, mapping);
  }
  return status;
}

static amdf_status_t FakeQueryStatus(amdf_user_queue_t* base_queue,
                                     amdf_user_queue_status_t* out_status) {
  auto* queue = reinterpret_cast<FakeQueue*>(base_queue);
  out_status->state = AMDF_QUEUE_STATE_ACTIVE;
  out_status->reset_epoch = base_queue->info.reset_epoch;
  out_status->producer_index = 41;
  out_status->consumed_index = 37;
  return queue->query_status;
}

static amdf_status_t FakeWaitConsumed(amdf_user_queue_t* base_queue,
                                      uint64_t published_index,
                                      uint64_t timeout_nanoseconds,
                                      uint64_t poll_duration_nanoseconds) {
  (void)timeout_nanoseconds;
  (void)poll_duration_nanoseconds;
  auto* queue = reinterpret_cast<FakeQueue*>(base_queue);
  queue->waited_index = published_index;
  return queue->wait_status;
}

static amdf_status_t FakeQueueDestroyNative(amdf_user_queue_t* base_queue) {
  return reinterpret_cast<FakeQueue*>(base_queue)->destroy_status;
}

static const amdf_user_queue_vtable_t kFakeQueueVtable = {
    .map = FakeMap,
    .query_status = FakeQueryStatus,
    .wait_consumed = FakeWaitConsumed,
    .destroy_native = FakeQueueDestroyNative,
};

static amdf_status_t CreateFakeQueue(amdf_device_t* device,
                                     amdf_user_queue_t** out_queue) {
  FakeQueue* queue = nullptr;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*queue),
                  amdf_alignof(FakeQueue), reinterpret_cast<void**>(&queue));
  if (!amdf_status_is_ok(status)) return status;
  const amdf_user_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
      .structure_size = sizeof(info),
      .device_id = {.words = {7, 11}},
      .queue_id = {.words = {13, 17}},
      .reset_epoch = 1,
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = 1,
      .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
      .capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      .roles = AMDF_QUEUE_ROLE_COMPUTE,
      .ring_byte_length = 4096,
  };
  status = amdf_user_queue_initialize(&queue->base, &kFakeQueueVtable, device,
                                      &info);
  if (amdf_status_is_ok(status)) {
    queue->map_status = AMDF_STATUS_OK;
    queue->query_status = AMDF_STATUS_OK;
    queue->wait_status = AMDF_STATUS_OK;
    queue->destroy_status = AMDF_STATUS_OK;
    *out_queue = &queue->base;
  } else {
    amdf_free(device->host_allocator, queue);
  }
  return status;
}

TEST(UserQueueTest, PreservesOutputsAndEnforcesMappingLifetime) {
  amdf_device_t device = {};
  device.host_allocator = amdf_allocator_system();
  amdf_child_tracker_initialize(&device.children);
  amdf_user_queue_t* queue = nullptr;
  ASSERT_EQ(CreateFakeQueue(&device, &queue), AMDF_STATUS_OK);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(amdf_child_tracker_count(&device.children), 1u);

  amdf_user_queue_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(amdf_user_queue_query_info(queue, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  EXPECT_EQ(info.ring_byte_length, 4096u);

  auto* fake_queue = reinterpret_cast<FakeQueue*>(queue);
  fake_queue->query_status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  amdf_user_queue_status_t queue_status;
  std::memset(&queue_status, 0xA5, sizeof(queue_status));
  queue_status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
  queue_status.structure_size = sizeof(queue_status);
  queue_status.next = nullptr;
  const amdf_user_queue_status_t original_status = queue_status;
  EXPECT_EQ(
      amdf_status_code(amdf_user_queue_query_status(queue, &queue_status)),
      AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(std::memcmp(&queue_status, &original_status, sizeof(queue_status)),
            0);
  fake_queue->query_status = AMDF_STATUS_OK;

  auto* const sentinel =
      reinterpret_cast<amdf_user_queue_mapping_t*>(uintptr_t{1});
  amdf_user_queue_mapping_t* mapping = sentinel;
  fake_queue->map_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_map(queue, nullptr, &mapping)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(mapping, sentinel);

  fake_queue->map_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_user_queue_map(queue, nullptr, &mapping), AMDF_STATUS_OK);
  ASSERT_NE(mapping, nullptr);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_destroy(queue)),
            AMDF_STATUS_CODE_BUSY);

  amdf_user_queue_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_EQ(amdf_user_queue_mapping_query_info(mapping, &mapping_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_info.ring_address, UINT64_C(0x1000));
  EXPECT_EQ(mapping_info.index_bits, 64u);

  auto* fake_mapping = reinterpret_cast<FakeMapping*>(mapping);
  fake_mapping->destroy_status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_mapping_destroy(mapping)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_destroy(queue)),
            AMDF_STATUS_CODE_BUSY);
  fake_mapping->destroy_status = AMDF_STATUS_OK;
  EXPECT_EQ(amdf_user_queue_mapping_destroy(mapping), AMDF_STATUS_OK);

  EXPECT_EQ(amdf_user_queue_wait_consumed(queue, 41, 10, 2), AMDF_STATUS_OK);
  EXPECT_EQ(fake_queue->waited_index, 41u);
  EXPECT_EQ(amdf_user_queue_destroy(queue), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_child_tracker_count(&device.children), 0u);
}

}  // namespace
