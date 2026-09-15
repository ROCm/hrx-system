// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/user_queue.h"

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/device.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/user_queue.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/user_queue.h"

struct FakeNativeState;

struct amdf_gpu_umd_device_t {
  FakeNativeState* state;
};

struct amdf_gpu_umd_user_queue_t {
  FakeNativeState* state;
};

struct amdf_gpu_umd_user_queue_mapping_t {
  FakeNativeState* state;
};

struct FakeNativeState {
  amdf_status_t create_status = AMDF_STATUS_OK;
  amdf_status_t map_status = AMDF_STATUS_OK;
  amdf_status_t mapping_destroy_status = AMDF_STATUS_OK;
  amdf_status_t query_status = AMDF_STATUS_OK;
  amdf_status_t wait_status = AMDF_STATUS_OK;
  amdf_status_t destroy_status = AMDF_STATUS_OK;
  amdf_gpu_umd_user_queue_t queue = {this};
  amdf_gpu_umd_user_queue_mapping_t mapping = {this};
  amdf_gpu_umd_user_queue_create_info_t observed_create = {};
  amdf_gpu_umd_device_t* observed_producer = nullptr;
  uint64_t waited_index = 0;
};

namespace {

amdf_queue_family_info_t current_family;

struct TestGpuDevice {
  amdf_device_t base = {};
  amdf_gpu_umd_device_t umd = {};
  amdf_gpu_device_info_t info = {};
};

class GpuUserQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitializeDevice(&consumer_, 7, 1, &consumer_state_);
    InitializeDevice(&producer_, 11, 3, &producer_state_);
    current_family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(current_family),
        .ordinal = 0,
        .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
        .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
        .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_CACHE_CONTROL,
        .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
        .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
        .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                   AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
        .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
        .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
        .minimum_ring_byte_length = 4096,
        .maximum_ring_byte_length = 65536,
        .ring_byte_length_alignment = 4096,
    };
    InitializeScratch();
  }

  void TearDown() override {
    consumer_state_.mapping_destroy_status = AMDF_STATUS_OK;
    consumer_state_.destroy_status = AMDF_STATUS_OK;
    if (mapping_ != nullptr) {
      EXPECT_EQ(amdf_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
    }
    if (queue_ != nullptr) {
      EXPECT_EQ(amdf_user_queue_destroy(queue_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(amdf_child_tracker_count(&scratch_.children), 0u);
    EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
    EXPECT_EQ(amdf_child_tracker_count(&producer_.base.children), 0u);
  }

  static void InitializeDevice(TestGpuDevice* device, uint64_t id,
                               uint64_t reset_epoch, FakeNativeState* state) {
    device->base.host_allocator = amdf_allocator_system();
    device->base.endpoint = reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1});
    device->base.engine_kind = AMDF_ENGINE_KIND_GPU;
    amdf_child_tracker_initialize(&device->base.children);
    device->umd.state = state;
    device->info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
    device->info.structure_size = sizeof(device->info);
    device->info.id.words[0] = id;
    device->info.reset_epoch = reset_epoch;
  }

  void InitializeScratch() {
    scratch_.accesses = scratch_accesses_;
    scratch_.info.access_count = 2;
    scratch_accesses_[0].device = &producer_.base;
    scratch_accesses_[1].device = &consumer_.base;
    scratch_.info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    scratch_.info.structure_size = sizeof(scratch_.info);
    scratch_accesses_[1].info.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    scratch_accesses_[1].info.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    scratch_accesses_[1].info.address_kinds = UINT64_C(1)
                                              << AMDF_MEMORY_ADDRESS_GPU;
    scratch_.info.byte_length = 16384;
    scratch_accesses_[1].addresses[AMDF_MEMORY_ADDRESS_GPU] =
        UINT64_C(0x800000);
    scratch_accesses_[1].info.reset_epoch = consumer_.info.reset_epoch;
    amdf_child_tracker_initialize(&scratch_.children);
  }

  amdf_gpu_user_queue_create_info_t MakeCreateInfo() const {
    amdf_gpu_user_queue_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    create_info.priority = AMDF_QUEUE_PRIORITY_NORMAL;
    create_info.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
    create_info.required_capabilities =
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
    create_info.scratch.memory = const_cast<amdf_memory_t*>(&scratch_);
    create_info.scratch.access_ordinal = 1;
    create_info.scratch.byte_offset = 4096;
    create_info.scratch.byte_length = 8192;
    create_info.scratch.maximum_private_segment_byte_length = 256;
    create_info.scratch.maximum_wave_count = 32;
    return create_info;
  }

  FakeNativeState consumer_state_;
  FakeNativeState producer_state_;
  TestGpuDevice consumer_;
  TestGpuDevice producer_;
  amdf_memory_t scratch_ = {};
  // Prepared access facts supplied by the memory dependency, in caller order.
  amdf_memory_access_state_t scratch_accesses_[2] = {};
  amdf_user_queue_t* queue_ = nullptr;
  amdf_user_queue_mapping_t* mapping_ = nullptr;
};

TEST_F(GpuUserQueueTest, RetainsScratchAndMappingsAcrossDestroyRetries) {
  const amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  ASSERT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            AMDF_STATUS_OK);
  ASSERT_NE(queue_, nullptr);
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 1u);
  EXPECT_EQ(amdf_child_tracker_count(&scratch_.children), 1u);
  EXPECT_EQ(consumer_state_.observed_create.scratch.device_address,
            scratch_accesses_[1].addresses[AMDF_MEMORY_ADDRESS_GPU] +
                create_info.scratch.byte_offset);
  EXPECT_EQ(consumer_state_.observed_create.scratch.byte_length,
            create_info.scratch.byte_length);

  amdf_user_queue_info_t queue_info = {};
  queue_info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO;
  queue_info.structure_size = sizeof(queue_info);
  ASSERT_EQ(amdf_user_queue_query_info(queue_, &queue_info), AMDF_STATUS_OK);
  EXPECT_TRUE(amdf_queue_id_is_valid(&queue_info.queue_id));
  EXPECT_EQ(queue_info.priority, AMDF_QUEUE_PRIORITY_NORMAL);
  EXPECT_EQ(queue_info.ring_byte_length, 4096u);

  ASSERT_EQ(amdf_user_queue_map(queue_, nullptr, &mapping_), AMDF_STATUS_OK);
  ASSERT_NE(mapping_, nullptr);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);

  amdf_user_queue_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_EQ(amdf_user_queue_mapping_query_info(mapping_, &mapping_info),
            AMDF_STATUS_OK);
  EXPECT_TRUE(
      amdf_queue_id_is_equal(&mapping_info.queue_id, &queue_info.queue_id));
  EXPECT_FALSE(amdf_device_id_is_equal(&mapping_info.producer_device_id,
                                       &consumer_.info.id));
  EXPECT_EQ(mapping_info.queue_reset_epoch, consumer_.info.reset_epoch);
  EXPECT_EQ(mapping_info.producer_reset_epoch, 0u);
  EXPECT_EQ(mapping_info.ring_address, UINT64_C(0x1000));
  EXPECT_EQ(mapping_info.doorbell_bits, 64u);

  consumer_state_.mapping_destroy_status =
      amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_mapping_destroy(mapping_)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);
  consumer_state_.mapping_destroy_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;

  consumer_state_.destroy_status = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 1u);
  EXPECT_EQ(amdf_child_tracker_count(&scratch_.children), 1u);
  consumer_state_.destroy_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
  EXPECT_EQ(amdf_child_tracker_count(&scratch_.children), 0u);
}

TEST_F(GpuUserQueueTest, PreservesPublicOutputsAcrossNativeFailures) {
  auto* const queue_sentinel =
      reinterpret_cast<amdf_user_queue_t*>(uintptr_t{1});
  queue_ = queue_sentinel;
  consumer_state_.create_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  const amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(queue_, queue_sentinel);
  queue_ = nullptr;
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
  EXPECT_EQ(amdf_child_tracker_count(&scratch_.children), 0u);

  consumer_state_.create_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            AMDF_STATUS_OK);
  auto* const mapping_sentinel =
      reinterpret_cast<amdf_user_queue_mapping_t*>(uintptr_t{1});
  mapping_ = mapping_sentinel;
  consumer_state_.map_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_map(queue_, nullptr, &mapping_)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(mapping_, mapping_sentinel);
  mapping_ = nullptr;

  consumer_state_.query_status =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  amdf_user_queue_status_t status;
  std::memset(&status, 0xA5, sizeof(status));
  status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
  status.structure_size = sizeof(status);
  status.next = nullptr;
  const amdf_user_queue_status_t original = status;
  EXPECT_EQ(amdf_status_code(amdf_user_queue_query_status(queue_, &status)),
            AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(std::memcmp(&status, &original, sizeof(status)), 0);
}

TEST_F(GpuUserQueueTest, DeviceMappingRetainsExactProducer) {
  amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  create_info.required_capabilities =
      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
  ASSERT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            AMDF_STATUS_OK);
  ASSERT_EQ(amdf_user_queue_map(queue_, &producer_.base, &mapping_),
            AMDF_STATUS_OK);
  EXPECT_EQ(consumer_state_.observed_producer, &producer_.umd);
  EXPECT_EQ(amdf_child_tracker_count(&producer_.base.children), 1u);

  amdf_user_queue_mapping_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(amdf_user_queue_mapping_query_info(mapping_, &info),
            AMDF_STATUS_OK);
  EXPECT_TRUE(
      amdf_device_id_is_equal(&info.producer_device_id, &producer_.info.id));
  EXPECT_EQ(info.producer_reset_epoch, producer_.info.reset_epoch);

  ASSERT_EQ(amdf_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  EXPECT_EQ(amdf_child_tracker_count(&producer_.base.children), 0u);
}

TEST_F(GpuUserQueueTest, RejectsInvalidScratchWithoutPublishingOutput) {
  amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  create_info.scratch.byte_length = 0;
  auto* const sentinel = reinterpret_cast<amdf_user_queue_t*>(uintptr_t{1});
  queue_ = sentinel;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(queue_, sentinel);
  queue_ = nullptr;

  create_info = MakeCreateInfo();
  create_info.scratch.byte_offset = scratch_.info.byte_length;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(queue_, nullptr);
}

TEST_F(GpuUserQueueTest, RejectsNonPowerOfTwoRingWithoutPublishingOutput) {
  amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  create_info.ring_byte_length = 12288;
  auto* const sentinel = reinterpret_cast<amdf_user_queue_t*>(uintptr_t{1});
  queue_ = sentinel;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(queue_, sentinel);
  queue_ = nullptr;
}

TEST_F(GpuUserQueueTest, ValidatesTheSelectedScratchAccess) {
  amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  create_info.scratch.access_ordinal = 2;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(queue_, nullptr);

  create_info.scratch.access_ordinal = 0;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(queue_, nullptr);

  create_info.scratch.access_ordinal = 1;
  scratch_accesses_[1].info.access = AMDF_MEMORY_ACCESS_READ;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(queue_, nullptr);

  scratch_accesses_[1].info.access |= AMDF_MEMORY_ACCESS_WRITE;
  scratch_accesses_[1].info.reset_epoch = consumer_.info.reset_epoch + 1;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_FAILED_PRECONDITION);
  EXPECT_EQ(queue_, nullptr);

  scratch_accesses_[1].info.reset_epoch = consumer_.info.reset_epoch;
  scratch_accesses_[1].info.address_kinds =
      UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
  EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(&consumer_.base,
                                                        &create_info, &queue_)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(queue_, nullptr);
}

}  // namespace

extern "C" {

amdf_status_t AMDF_CALL amdf_endpoint_query_queue_family_info(
    amdf_endpoint_t*, uint32_t ordinal, amdf_queue_family_info_t* out_info) {
  if (ordinal != current_family.ordinal) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  *out_info = current_family;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_endpoint_register_device(amdf_endpoint_t*) {
  return AMDF_STATUS_OK;
}

void amdf_endpoint_unregister_device(amdf_endpoint_t*) {}

amdf_allocator_t amdf_endpoint_host_allocator(const amdf_endpoint_t*) {
  return amdf_allocator_system();
}

amdf_instance_t* amdf_endpoint_get_instance(const amdf_endpoint_t*) {
  return reinterpret_cast<amdf_instance_t*>(uintptr_t{1});
}

amdf_gpu_umd_device_t* amdf_gpu_device_get_umd(amdf_device_t* device) {
  return &reinterpret_cast<TestGpuDevice*>(device)->umd;
}

const amdf_gpu_device_info_t* amdf_gpu_device_get_info(
    const amdf_device_t* device) {
  return &reinterpret_cast<const TestGpuDevice*>(device)->info;
}

amdf_status_t amdf_memory_register_child(amdf_memory_t* memory) {
  return amdf_child_tracker_register(&memory->children);
}

void amdf_memory_unregister_child(amdf_memory_t* memory) {
  amdf_child_tracker_unregister(&memory->children);
}

amdf_status_t amdf_gpu_umd_user_queue_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_umd_user_queue_t** out_queue,
    amdf_gpu_umd_user_queue_result_t* out_result) {
  FakeNativeState* state = device->state;
  state->observed_create = *create_info;
  if (!amdf_status_is_ok(state->create_status)) return state->create_status;
  *out_result = {
      .queue_id = {.words = {19, 23}},
      .capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
      .ring_byte_length = create_info->ring_byte_length == 0
                              ? UINT64_C(4096)
                              : create_info->ring_byte_length,
  };
  *out_queue = &state->queue;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_user_queue_map(
    amdf_gpu_umd_user_queue_t* queue, amdf_gpu_umd_device_t* producer_device,
    amdf_gpu_umd_user_queue_mapping_t** out_mapping,
    amdf_gpu_umd_user_queue_mapping_result_t* out_result) {
  FakeNativeState* state = queue->state;
  state->observed_producer = producer_device;
  if (!amdf_status_is_ok(state->map_status)) return state->map_status;
  *out_result = {
      .ring_address = UINT64_C(0x1000),
      .read_index_address = UINT64_C(0x2000),
      .write_index_address = UINT64_C(0x2008),
      .doorbell_address = UINT64_C(0x3000),
      .index_bits = 64,
      .doorbell_bits = 64,
  };
  *out_mapping = &state->mapping;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_user_queue_mapping_destroy(
    amdf_gpu_umd_user_queue_mapping_t* mapping) {
  return mapping->state->mapping_destroy_status;
}

amdf_status_t amdf_gpu_umd_user_queue_query_status(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status) {
  out_status->state = AMDF_QUEUE_STATE_ACTIVE;
  out_status->reset_epoch = 1;
  out_status->producer_index = 13;
  out_status->consumed_index = 11;
  return queue->state->query_status;
}

amdf_status_t amdf_gpu_umd_user_queue_wait_consumed(
    amdf_gpu_umd_user_queue_t* queue, uint64_t published_index,
    const amdf_wait_deadline_t*) {
  queue->state->waited_index = published_index;
  return queue->state->wait_status;
}

amdf_status_t amdf_gpu_umd_user_queue_destroy(
    amdf_gpu_umd_user_queue_t* queue) {
  return queue->state->destroy_status;
}

}  // extern "C"
