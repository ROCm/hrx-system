// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/util/user_queue.h"

#include "gtest/gtest.h"

void GpuUserQueue::Initialize(
    const amdf_api_t* api, const amdf_gpu_api_t* gpu_api, amdf_device_t* device,
    const amdf_queue_family_info_t& family,
    amdf_queue_producer_mode_t producer_mode,
    const amdf_gpu_queue_scratch_t& scratch,
    amdf_user_queue_capabilities_t required_capabilities,
    uint64_t ring_byte_length) {
  amdf_gpu_user_queue_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create),
      .queue_family_ordinal = family.ordinal,
      .priority = AMDF_QUEUE_PRIORITY_NORMAL,
      .producer_mode = producer_mode,
      .required_capabilities = required_capabilities,
      .ring_byte_length = ring_byte_length,
      .scratch = scratch};
  ASSERT_EQ(gpu_api->user_queue_create(device, &create, &queue),
            AMDF_STATUS_OK);
  info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(api->user_queue_query_info(queue, &info), AMDF_STATUS_OK);
  ASSERT_EQ(info.queue_family_ordinal, family.ordinal);
  ASSERT_EQ(info.command_type, family.command_type);
  ASSERT_EQ(info.format_version, family.format_version);
  ASSERT_EQ(info.format_features, family.format_features);
  ASSERT_EQ(info.producer_mode, producer_mode);
  ASSERT_EQ(info.priority, AMDF_QUEUE_PRIORITY_NORMAL);
  ASSERT_EQ(info.capabilities & required_capabilities, required_capabilities);
  ASSERT_EQ(api->user_queue_map(queue, nullptr, &mapping), AMDF_STATUS_OK);
  host.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
  host.structure_size = sizeof(host);
  ASSERT_EQ(api->user_queue_mapping_query_info(mapping, &host), AMDF_STATUS_OK);
  ASSERT_TRUE(amdf_queue_id_is_equal(&host.queue_id, &info.queue_id));
  ASSERT_EQ(host.queue_reset_epoch, info.reset_epoch);
  ASSERT_EQ(host.command_type, info.command_type);
  ASSERT_EQ(host.format_version, info.format_version);
  ASSERT_EQ(host.format_features, info.format_features);
  ASSERT_EQ(host.ring_byte_length, info.ring_byte_length);
  ASSERT_EQ(host.index_bits, 64u);
  ASSERT_EQ(host.doorbell_bits, 64u);
  ASSERT_EQ(host.metadata_ring_byte_length, 0u);
  ASSERT_NE(host.ring_address, 0u);
  ASSERT_NE(host.read_index_address, 0u);
  ASSERT_NE(host.write_index_address, 0u);
  ASSERT_NE(host.doorbell_address, 0u);
  ASSERT_EQ(host.read_index_address % sizeof(uint64_t), 0u);
  ASSERT_EQ(host.write_index_address % sizeof(uint64_t), 0u);
  ASSERT_EQ(host.doorbell_address % sizeof(uint64_t), 0u);
  ASSERT_EQ(GpuLoadAcquire<uint64_t>(host.read_index_address), 0u);
  ASSERT_EQ(GpuLoadAcquire<uint64_t>(host.write_index_address), 0u);
  if ((required_capabilities & AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER) !=
      0) {
    ASSERT_EQ(api->user_queue_map(queue, device, &producer.mapping),
              AMDF_STATUS_OK);
    producer.info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
    producer.info.structure_size = sizeof(producer.info);
    ASSERT_EQ(
        api->user_queue_mapping_query_info(producer.mapping, &producer.info),
        AMDF_STATUS_OK);
    ASSERT_TRUE(
        amdf_queue_id_is_equal(&producer.info.queue_id, &info.queue_id));
    ASSERT_EQ(producer.info.queue_reset_epoch, info.reset_epoch);
    ASSERT_EQ(producer.info.command_type, info.command_type);
    ASSERT_EQ(producer.info.format_version, info.format_version);
    ASSERT_EQ(producer.info.format_features, info.format_features);
    ASSERT_EQ(producer.info.ring_byte_length, info.ring_byte_length);
    ASSERT_EQ(producer.info.index_bits, 64u);
    ASSERT_EQ(producer.info.doorbell_bits, 64u);
    ASSERT_NE(producer.info.ring_address, 0u);
    ASSERT_NE(producer.info.read_index_address, 0u);
    ASSERT_NE(producer.info.write_index_address, 0u);
    ASSERT_NE(producer.info.doorbell_address, 0u);
    ASSERT_EQ(producer.info.read_index_address % sizeof(uint64_t), 0u);
    ASSERT_EQ(producer.info.write_index_address % sizeof(uint64_t), 0u);
    ASSERT_EQ(producer.info.doorbell_address % sizeof(uint64_t), 0u);
  }
}

bool GpuUserQueue::Release(const amdf_api_t* api) {
  if (producer.mapping != nullptr) {
    const amdf_status_t status =
        api->user_queue_mapping_destroy(producer.mapping);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
    producer.mapping = nullptr;
  }
  if (mapping != nullptr) {
    const amdf_status_t status = api->user_queue_mapping_destroy(mapping);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
    mapping = nullptr;
  }
  if (queue != nullptr) {
    const amdf_status_t status = api->user_queue_destroy(queue);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      queue = nullptr;
    }
    EXPECT_EQ(status, AMDF_STATUS_OK);
    // Handle consumption does not establish final native access. The fixture
    // preserves all backing after rejection or failed native removal.
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  return true;
}

void GpuUserQueue::PublishStream(uint64_t producer_index) {
  ASSERT_NE(info.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  GpuStoreRelease(host.write_index_address, producer_index);
  GpuStoreRelease(host.doorbell_address, producer_index);
}

void GpuUserQueue::WaitConsumed(const amdf_api_t* api,
                                uint64_t producer_index) {
  ASSERT_EQ(api->user_queue_wait_consumed(queue, producer_index,
                                          AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
}
