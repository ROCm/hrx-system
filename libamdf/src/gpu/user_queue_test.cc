// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/user_queue.h"

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/device.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/event.h"
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

struct amdf_gpu_event_t {
  // Exact device borrowed by this dependency event.
  amdf_device_t* device;
  // Immutable native notification operands.
  amdf_gpu_event_info_t info;
};

struct FakeNativeState {
  // Native construction calls reached after public input validation.
  uint32_t create_call_count = 0;
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
  // Last caller backing resolved before native resume.
  amdf_gpu_umd_queue_scratch_t resumed_scratch = {};
  // Timestamp selections that reached the native dependency.
  uint32_t timestamp_call_count = 0;
  // Last validated timestamp selection passed to the native dependency.
  bool timestamps_enabled = false;
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
  }

  amdf_gpu_user_queue_create_info_t MakeCreateInfo() const {
    amdf_gpu_user_queue_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
    };
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

TEST_F(GpuUserQueueTest, RetainsQueueAcrossRejectedDestruction) {
  const amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  ASSERT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            AMDF_STATUS_OK);
  ASSERT_NE(queue_, nullptr);
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 1u);
  EXPECT_EQ(consumer_state_.observed_create.scratch.device_address,
            scratch_accesses_[1].addresses[AMDF_MEMORY_ADDRESS_GPU] +
                create_info.scratch.byte_offset);
  EXPECT_EQ(consumer_state_.observed_create.scratch.byte_length,
            create_info.scratch.byte_length);

  amdf_user_queue_info_t queue_info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
      .structure_size = sizeof(queue_info),
  };
  ASSERT_EQ(amdf_user_queue_query_info(queue_, &queue_info), AMDF_STATUS_OK);
  EXPECT_TRUE(amdf_queue_id_is_valid(&queue_info.queue_id));
  EXPECT_EQ(queue_info.priority, AMDF_QUEUE_PRIORITY_NORMAL);
  EXPECT_EQ(queue_info.ring_byte_length, 4096u);

  ASSERT_EQ(amdf_user_queue_map(queue_, nullptr, &mapping_), AMDF_STATUS_OK);
  ASSERT_NE(mapping_, nullptr);
  EXPECT_EQ(amdf_status_code(amdf_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);

  amdf_user_queue_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
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
  consumer_state_.destroy_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
}

TEST_F(GpuUserQueueTest, NativeReleaseFailureConsumesQueue) {
  const amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  ASSERT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            AMDF_STATUS_OK);
  consumer_state_.destroy_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);
  EXPECT_EQ(amdf_user_queue_destroy(queue_), consumer_state_.destroy_status);
  queue_ = nullptr;
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
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

  amdf_user_queue_mapping_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(amdf_user_queue_mapping_query_info(mapping_, &info),
            AMDF_STATUS_OK);
  EXPECT_TRUE(
      amdf_device_id_is_equal(&info.producer_device_id, &producer_.info.id));
  EXPECT_EQ(info.producer_reset_epoch, producer_.info.reset_epoch);

  ASSERT_EQ(amdf_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  EXPECT_EQ(amdf_child_tracker_count(&producer_.base.children), 0u);
}

TEST_F(GpuUserQueueTest, DispatchTimestampsRequireExplicitQueueSelection) {
  current_family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  current_family.format_version = AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1;
  current_family.format_features =
      AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS;
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    auto create_info = MakeCreateInfo();
    create_info.flags =
        enabled ? AMDF_GPU_USER_QUEUE_FLAG_DISPATCH_TIMESTAMPS : 0;
    ASSERT_EQ(
        amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
        AMDF_STATUS_OK);
    EXPECT_EQ(consumer_state_.observed_create.dispatch_timestamps, enabled);
    EXPECT_EQ(consumer_state_.observed_create.command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    ASSERT_EQ(amdf_user_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
  }
  EXPECT_EQ(consumer_state_.create_call_count, 2u);
}

TEST_F(GpuUserQueueTest, RejectsUnavailableDispatchTimestampsBeforeCreation) {
  struct Case {
    // Native language selected by the family.
    amdf_queue_command_type_t command_type;
    // Available operations in that language's feature namespace.
    amdf_queue_format_features_t format_features;
  };
  // Format bits are local to their command language. Even a coincident bit in
  // a PM4/SDMA family must not enable an AQL-only native property.
  constexpr Case kCases[] = {
      {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 0},
      {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
       AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS},
      {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 0},
      {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
       AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS},
      {AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, 0},
  };
  auto create_info = MakeCreateInfo();
  create_info.flags = AMDF_GPU_USER_QUEUE_FLAG_DISPATCH_TIMESTAMPS;
  auto* const sentinel = reinterpret_cast<amdf_user_queue_t*>(uintptr_t{1});
  for (const auto& test : kCases) {
    SCOPED_TRACE(test.command_type);
    SCOPED_TRACE(test.format_features);
    current_family.command_type = test.command_type;
    current_family.format_features = test.format_features;
    auto* output = sentinel;
    EXPECT_EQ(
        amdf_gpu_user_queue_create(&consumer_.base, &create_info, &output),
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(output, sentinel);
  }
  EXPECT_EQ(consumer_state_.create_call_count, 0u);
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
}

TEST_F(GpuUserQueueTest, TimestampSelectionValidatesBeforeNativeMutation) {
  EXPECT_EQ(amdf_gpu_user_queue_set_dispatch_timestamps(nullptr, 1),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  for (auto command_type :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
        AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA}) {
    for (auto features :
         {amdf_queue_format_features_t{0},
          amdf_queue_format_features_t{
              AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS}}) {
      current_family.command_type = command_type;
      current_family.format_features = features;
      auto create_info = MakeCreateInfo();
      ASSERT_EQ(
          amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
          AMDF_STATUS_OK);
      const uint32_t before = consumer_state_.timestamp_call_count;
      EXPECT_EQ(amdf_gpu_user_queue_set_dispatch_timestamps(queue_, 2),
                amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
      const bool supported =
          command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL && features;
      for (uint32_t enabled : {0u, 1u}) {
        EXPECT_EQ(amdf_gpu_user_queue_set_dispatch_timestamps(queue_, enabled),
                  supported
                      ? AMDF_STATUS_OK
                      : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
        if (supported) {
          EXPECT_EQ(consumer_state_.timestamps_enabled, enabled != 0);
        }
      }
      EXPECT_EQ(consumer_state_.timestamp_call_count,
                before + (supported ? 2u : 0u));
      ASSERT_EQ(amdf_user_queue_destroy(queue_), AMDF_STATUS_OK);
      queue_ = nullptr;
    }
  }
}

TEST_F(GpuUserQueueTest, ScratchServiceValidatesExactEventAndBacking) {
  current_family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  current_family.format_features = AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE;
  amdf_gpu_event_t event = {
      .device = &producer_.base,
      .info = {.mailbox_address = 0x12340000, .interrupt_context = 17},
  };
  amdf_gpu_aql_scratch_service_info_t service = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_SERVICE_INFO,
      .structure_size = sizeof(service),
      .event = &event,
  };
  auto create_info = MakeCreateInfo();
  create_info.next = &service;
  EXPECT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(consumer_state_.create_call_count, 0u);
  event.device = &consumer_.base;
  ASSERT_EQ(amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
            AMDF_STATUS_OK);
  EXPECT_TRUE(consumer_state_.observed_create.scratch_service);
  EXPECT_EQ(consumer_state_.observed_create.inactive_event.mailbox_address,
            event.info.mailbox_address);
  auto scratch = create_info.scratch;
  scratch.access_ordinal = 0;
  EXPECT_EQ(amdf_gpu_user_queue_resume_scratch(queue_, &scratch),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(consumer_state_.resumed_scratch.device_address, 0u);
  ASSERT_EQ(amdf_gpu_user_queue_resume_scratch(queue_, &create_info.scratch),
            AMDF_STATUS_OK);
  EXPECT_EQ(consumer_state_.resumed_scratch.device_address, 0x801000u);

  amdf_gpu_aql_scratch_request_t request = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_REQUEST,
      .structure_size = sizeof(request),
  };
  ASSERT_EQ(amdf_gpu_user_queue_query_scratch(queue_, &request),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_gpu_user_queue_trim_scratch(queue_), AMDF_STATUS_OK);
  EXPECT_EQ(request.byte_length, 4096u);
  EXPECT_EQ(request.structure_size, sizeof(request));
  const auto saved = request;
  consumer_state_.query_status =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_user_queue_query_scratch(queue_, &request),
            consumer_state_.query_status);
  EXPECT_EQ(amdf_gpu_user_queue_trim_scratch(queue_),
            consumer_state_.query_status);
  EXPECT_EQ(std::memcmp(&saved, &request, sizeof(request)), 0);
}

TEST_F(GpuUserQueueTest, ScratchServiceRequiresQualifiedAqlFamily) {
  amdf_gpu_aql_scratch_service_info_t service = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_SERVICE_INFO,
      .structure_size = sizeof(service),
  };
  auto create_info = MakeCreateInfo();
  create_info.next = &service;
  for (auto command_type :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL}) {
    current_family.command_type = command_type;
    current_family.format_features =
        command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
            ? 0
            : AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE;
    EXPECT_EQ(
        amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  }
  EXPECT_EQ(consumer_state_.create_call_count, 0u);
}

TEST_F(GpuUserQueueTest, RejectsUnknownQueueFlagsBeforeCreation) {
  auto create_info = MakeCreateInfo();
  auto* const sentinel = reinterpret_cast<amdf_user_queue_t*>(uintptr_t{1});
  for (const uint32_t flags : {2u, 3u, UINT32_MAX}) {
    SCOPED_TRACE(flags);
    create_info.flags = flags;
    auto* output = sentinel;
    EXPECT_EQ(
        amdf_gpu_user_queue_create(&consumer_.base, &create_info, &output),
        amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
    EXPECT_EQ(output, sentinel);
  }
  EXPECT_EQ(consumer_state_.create_call_count, 0u);
  EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
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

TEST_F(GpuUserQueueTest, RejectsInvalidRingCapacityWithoutNativeConstruction) {
  amdf_gpu_user_queue_create_info_t create_info = MakeCreateInfo();
  auto* const sentinel = reinterpret_cast<amdf_user_queue_t*>(uintptr_t{1});
  for (uint64_t capacity : {UINT64_C(2048), UINT64_C(12288), UINT64_C(131072),
                            UINT64_C(1) << 32, UINT64_MAX}) {
    SCOPED_TRACE(capacity);
    create_info.ring_byte_length = capacity;
    auto* queue = sentinel;
    EXPECT_EQ(amdf_status_code(amdf_gpu_user_queue_create(
                  &consumer_.base, &create_info, &queue)),
              AMDF_STATUS_CODE_OUT_OF_RANGE);
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(consumer_state_.observed_create.ring_byte_length, 0u);
    EXPECT_EQ(amdf_child_tracker_count(&consumer_.base.children), 0u);
  }
}

TEST_F(GpuUserQueueTest, PreservesRequestedRingCapacity) {
  for (uint64_t capacity : {UINT64_C(4096), UINT64_C(16384), UINT64_C(65536)}) {
    SCOPED_TRACE(capacity);
    auto create_info = MakeCreateInfo();
    create_info.ring_byte_length = capacity;
    ASSERT_EQ(
        amdf_gpu_user_queue_create(&consumer_.base, &create_info, &queue_),
        AMDF_STATUS_OK);
    EXPECT_EQ(consumer_state_.observed_create.ring_byte_length, capacity);
    amdf_user_queue_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(amdf_user_queue_query_info(queue_, &info), AMDF_STATUS_OK);
    EXPECT_EQ(info.ring_byte_length, capacity);
    ASSERT_EQ(amdf_user_queue_map(queue_, nullptr, &mapping_), AMDF_STATUS_OK);
    amdf_user_queue_mapping_info_t mapping_info = {
        .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
        .structure_size = sizeof(mapping_info),
    };
    ASSERT_EQ(amdf_user_queue_mapping_query_info(mapping_, &mapping_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(mapping_info.ring_byte_length, capacity);
    ASSERT_EQ(amdf_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
    mapping_ = nullptr;
    ASSERT_EQ(amdf_user_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
  }
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

amdf_status_t amdf_gpu_umd_user_queue_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_umd_user_queue_t** out_queue,
    amdf_gpu_umd_user_queue_result_t* out_result) {
  FakeNativeState* state = device->state;
  ++state->create_call_count;
  state->observed_create = *create_info;
  if (!amdf_status_is_ok(state->create_status)) {
    return state->create_status;
  }
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
  if (!amdf_status_is_ok(state->map_status)) {
    return state->map_status;
  }
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

amdf_device_t* amdf_gpu_event_get_device(const amdf_gpu_event_t* event) {
  return event->device;
}

amdf_status_t AMDF_CALL amdf_gpu_event_query_info(
    amdf_gpu_event_t* event, amdf_gpu_event_info_t* out_info) {
  *out_info = event->info;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_user_queue_query_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    amdf_gpu_aql_scratch_request_t* out_request) {
  out_request->byte_length = 4096;
  return queue->state->query_status;
}

amdf_status_t amdf_gpu_umd_user_queue_trim_scratch(
    amdf_gpu_umd_user_queue_t* queue) {
  return queue->state->query_status;
}

amdf_status_t amdf_gpu_umd_user_queue_set_dispatch_timestamps(
    amdf_gpu_umd_user_queue_t* queue, bool enabled) {
  ++queue->state->timestamp_call_count;
  queue->state->timestamps_enabled = enabled;
  return queue->state->query_status;
}

amdf_status_t amdf_gpu_umd_user_queue_resume_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    const amdf_gpu_umd_queue_scratch_t* scratch) {
  queue->state->resumed_scratch = *scratch;
  return AMDF_STATUS_OK;
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
