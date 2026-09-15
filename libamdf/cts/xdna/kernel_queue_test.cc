// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "gtest/gtest.h"
#include "xdna_device_fixture.h"

namespace {

std::array<uint8_t, 64> MakeNoOpTransaction() {
  std::array<uint8_t, 64> bytes = {};
  bytes[1] = 1;   // Transaction version 0.1.
  bytes[2] = 4;   // AIE2P generation.
  bytes[3] = 6;   // Physical row count.
  bytes[4] = 1;   // Logical column count.
  bytes[5] = 1;   // Memory row count.
  bytes[8] = 12;  // Twelve four-byte XAIE_IO_NOOP operations.
  bytes[12] = bytes.size();
  for (size_t offset = 16; offset < bytes.size(); offset += 4)
    bytes[offset] = 5;
  return bytes;
}

class XdnaKernelQueueTest : public XdnaContextFixture {
 protected:
  void TearDown() override {
    if (queue_) {
      EXPECT_EQ(api_->kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    }
    if (mapping_) {
      EXPECT_EQ(api_->host_mapping_destroy(mapping_), AMDF_STATUS_OK);
    }
    if (memory_) {
      EXPECT_EQ(api_->memory_destroy(memory_), AMDF_STATUS_OK);
    }
    if (sibling_.memory) {
      EXPECT_EQ(api_->memory_destroy(sibling_.memory), AMDF_STATUS_OK);
    }
    if (sibling_.context) {
      EXPECT_EQ(xdna_api_->context_destroy(sibling_.context), AMDF_STATUS_OK);
    }
    XdnaContextFixture::TearDown();
  }

  void AllocateInstructions(amdf_xdna_context_t* context,
                            amdf_memory_t** out_memory) {
    uint32_t count = 0;
    ASSERT_EQ(amdf_status_code(xdna_api_->context_enumerate_memory_scopes(
                  context, 0, nullptr, &count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    ASSERT_EQ(count, 1u);
    amdf_memory_scope_t* scope = nullptr;
    ASSERT_EQ(
        xdna_api_->context_enumerate_memory_scopes(context, 1, &scope, &count),
        AMDF_STATUS_OK);
    amdf_memory_scope_info_t scope_info = {};
    scope_info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
    scope_info.structure_size = sizeof(scope_info);
    ASSERT_EQ(api_->memory_scope_query_info(scope, &scope_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(scope_info.kind, AMDF_MEMORY_SCOPE_KIND_PRIVATE);
    EXPECT_EQ(scope_info.memory_profile_count, 1u);

    amdf_xdna_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    instruction_stride_ = endpoint_info.instruction.address_alignment;
    memory_access_.requirements.access |= AMDF_MEMORY_ACCESS_EXECUTE;
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(api_->memory_scope_query_device_profile(
                  scope, 0, 1, &memory_access_, &profile, &capabilities),
              AMDF_STATUS_OK);
    EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_PRIVATE);
    EXPECT_NE(capabilities.address_kinds &
                  (uint64_t{1} << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE),
              0u);

    const uint64_t granularity = profile.allocation.byte_length_granularity;
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.access_count = 1;
    create.accesses = &memory_access_;
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length = ((instruction_stride_ + MakeNoOpTransaction().size() +
                           granularity - 1) /
                          granularity) *
                         granularity;
    create.minimum_alignment = profile.allocation.minimum_alignment;
    ASSERT_EQ(api_->memory_create(scope, &create, out_memory), AMDF_STATUS_OK);
  }

  void CreateInstructions() {
    ASSERT_NO_FATAL_FAILURE(AllocateInstructions(context_, &memory_));

    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map.byte_length = instruction_stride_ + MakeNoOpTransaction().size();
    ASSERT_EQ(api_->memory_map(memory_, &map, &mapping_), AMDF_STATUS_OK);
    amdf_host_mapping_info_t mapping_info = {};
    mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping_info.structure_size = sizeof(mapping_info);
    ASSERT_EQ(api_->host_mapping_query_info(mapping_, &mapping_info),
              AMDF_STATUS_OK);
    pointer_ = static_cast<uint8_t*>(mapping_info.pointer);
    const auto transaction = MakeNoOpTransaction();
    std::memcpy(pointer_, transaction.data(), transaction.size());
    std::memcpy(pointer_ + instruction_stride_, transaction.data(),
                transaction.size());
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, map.byte_length),
        AMDF_STATUS_OK);
  }

  uint32_t QueryKernelQueueFamily() {
    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    EXPECT_TRUE(amdf_status_is_ok(
        api_->endpoint_query_info(endpoint_, &endpoint_info)));
    for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
         ++ordinal) {
      amdf_queue_family_info_t family_info = {};
      family_info.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family_info.structure_size = sizeof(family_info);
      EXPECT_TRUE(amdf_status_is_ok(api_->endpoint_query_queue_family_info(
          endpoint_, ordinal, &family_info)));
      if (family_info.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
          (family_info.publication_modes &
           AMDF_QUEUE_PUBLICATION_MODE_KERNEL) != 0) {
        return ordinal;
      }
    }
    ADD_FAILURE() << "materialized XDNA device has no kernel queue family";
    return UINT32_MAX;
  }

  void CreateQueue() {
    std::cerr << "[XDNA] Acquiring queue and admitting firmware" << std::endl;
    amdf_xdna_kernel_queue_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    create_info.queue_family_ordinal = QueryKernelQueueFamily();
    ASSERT_NE(create_info.queue_family_ordinal, UINT32_MAX);
    ASSERT_TRUE(amdf_status_is_ok(
        xdna_api_->kernel_queue_create(context_, &create_info, &queue_)));
  }

  // Caller-owned instruction backing, released before the context.
  amdf_memory_t* memory_ = nullptr;
  // Explicit mapping used to publish instructions and verify immutability.
  amdf_host_mapping_t* mapping_ = nullptr;
  // Borrowed host view owned by mapping_.
  uint8_t* pointer_ = nullptr;
  // Target-qualified placement distance between the two immutable ranges.
  uint64_t instruction_stride_ = 0;
  // Exclusive native transport lease.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Independent context and private backing used for scope isolation coverage.
  struct {
    // Second live context borrowing the shared device.
    amdf_xdna_context_t* context = nullptr;
    // Separate instruction backing destroyed before its context.
    amdf_memory_t* memory = nullptr;
  } sibling_;
};

TEST_F(XdnaKernelQueueTest, SubmitsImmutableRangesAndReacquiresQueue) {
  ASSERT_NO_FATAL_FAILURE(CreateInstructions());
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  amdf_xdna_kernel_command_t command = {};
  command.memory = memory_;
  command.byte_length = MakeNoOpTransaction().size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;

  for (uint64_t i = 0; i < 3; ++i) {
    command.byte_offset = (i % 2) * instruction_stride_;
    uint64_t submission = 0;
    ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
              AMDF_STATUS_OK);
    EXPECT_EQ(submission, i + 1);
    ASSERT_EQ(
        api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
        AMDF_STATUS_OK);
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    ASSERT_EQ(api_->kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
    EXPECT_EQ(status.retired_submission, submission);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
  }
  const auto expected = MakeNoOpTransaction();
  ASSERT_EQ(api_->host_mapping_cache_control(
                mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                instruction_stride_ + expected.size()),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pointer_, expected.data(), expected.size()), 0);
  EXPECT_EQ(std::memcmp(pointer_ + instruction_stride_, expected.data(),
                        expected.size()),
            0);

  ASSERT_EQ(api_->kernel_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  uint64_t submission = 0;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
            AMDF_STATUS_OK);
  EXPECT_EQ(submission, 1u);
  ASSERT_EQ(
      api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
      AMDF_STATUS_OK);
}

TEST_F(XdnaKernelQueueTest, PrivateBackingIsQualifiedByExactContext) {
  ASSERT_NO_FATAL_FAILURE(CreateInstructions());
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  amdf_xdna_context_create_info_t create = {};
  create.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.logical_column_count = 1;
  create.physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
  create.acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  ASSERT_EQ(xdna_api_->context_create(device_, &create, &sibling_.context),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(
      AllocateInstructions(sibling_.context, &sibling_.memory));
  EXPECT_NE(memory_, sibling_.memory);
  amdf_xdna_kernel_command_t command = {};
  command.memory = sibling_.memory;
  command.byte_length = MakeNoOpTransaction().size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;
  uint64_t submission = UINT64_MAX;
  EXPECT_EQ(amdf_status_code(
                xdna_api_->kernel_queue_submit(queue_, &submit, &submission)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(submission, UINT64_MAX);

  // Destroying the sibling's backing and context cannot release or invalidate
  // this context's instruction allocation or queue.
  ASSERT_EQ(api_->memory_destroy(sibling_.memory), AMDF_STATUS_OK);
  sibling_.memory = nullptr;
  ASSERT_EQ(xdna_api_->context_destroy(sibling_.context), AMDF_STATUS_OK);
  sibling_.context = nullptr;
  command.memory = memory_;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
            AMDF_STATUS_OK);
  ASSERT_EQ(
      api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
      AMDF_STATUS_OK);
}

}  // namespace
