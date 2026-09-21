// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "gtest/gtest.h"
#include "util/native_event.h"
#include "xdna_device_fixture.h"

namespace {

std::vector<uint8_t> MakeNoOpTransaction() {
  std::vector<uint8_t> bytes(64);
  bytes[1] = 1;   // Transaction version 0.1.
  bytes[2] = 4;   // AIE2P generation.
  bytes[3] = 6;   // Physical row count.
  bytes[4] = 1;   // Logical column count.
  bytes[5] = 1;   // Memory row count.
  bytes[8] = 12;  // Twelve four-byte XAIE_IO_NOOP operations.
  bytes[12] = bytes.size();
  for (size_t offset = 16; offset < bytes.size(); offset += 4) {
    bytes[offset] = 5;
  }
  return bytes;
}

enum class RegisterOperation : uint8_t {
  kWrite = 0,
  kMaskedWrite = 3,
  kPoll = 4,
};

void WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
  }
}

// AIE-RT transaction 0.1 register operations carry a 64-bit array offset and
// their own byte length. Polling verifies values through the actual firmware
// consumer without a debug output buffer or application tile program.
void AppendRegisterOperation(std::vector<uint8_t>& bytes,
                             RegisterOperation operation, uint32_t address,
                             uint32_t value, uint32_t mask) {
  const size_t offset = bytes.size();
  const uint32_t byte_length = operation == RegisterOperation::kWrite ? 24 : 32;
  bytes.resize(offset + byte_length);
  bytes[offset] = static_cast<uint8_t>(operation);
  WriteU32(bytes, offset + 8, address);
  WriteU32(bytes, offset + 16, value);
  if (operation == RegisterOperation::kWrite) {
    WriteU32(bytes, offset + 20, byte_length);
  } else {
    WriteU32(bytes, offset + 20, mask);
    WriteU32(bytes, offset + 24, byte_length);
  }
  uint32_t operation_count = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    operation_count |= uint32_t{bytes[8 + i]} << (i * 8);
  }
  WriteU32(bytes, 8, operation_count + 1);
  WriteU32(bytes, 12, static_cast<uint32_t>(bytes.size()));
}

class XdnaKernelQueueTest : public XdnaContextFixture {
 protected:
  void TearDown() override {
    if (queue_) {
      const auto status = DestroyQueue();
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(status)) {
        return;
      }
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

  amdf_status_t DestroyQueue() {
    const auto status = api_->kernel_queue_destroy(queue_);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      queue_ = nullptr;
    }
    return status;
  }

  void AllocateInstructions(amdf_xdna_context_t* context,
                            uint64_t second_range_byte_length,
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

    amdf_xdna_device_info_t device_info = {};
    device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
    device_info.structure_size = sizeof(device_info);
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    instruction_stride_ = device_info.instruction.address_alignment;
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
    create.byte_length =
        ((instruction_stride_ + second_range_byte_length + granularity - 1) /
         granularity) *
        granularity;
    create.minimum_alignment = profile.allocation.minimum_alignment;
    const auto& geometry = profile.allocation;
    const uint64_t native_granularity = geometry.native_byte_length_granularity;
    ASSERT_GT(native_granularity, 0u);
    ASSERT_LE(geometry.maximum_byte_length,
              UINT64_MAX - geometry.native_byte_length_prefix -
                  (native_granularity - 1));
    const uint64_t native_byte_length =
        ((create.byte_length + geometry.native_byte_length_prefix +
          native_granularity - 1) /
         native_granularity) *
        native_granularity;
    ASSERT_EQ(api_->memory_create(scope, &create, out_memory), AMDF_STATUS_OK);
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(*out_memory, &info), AMDF_STATUS_OK);
    EXPECT_EQ(info.source_byte_offset, geometry.native_byte_length_prefix);
    EXPECT_EQ(info.native_allocation_byte_length, native_byte_length);
    EXPECT_EQ(info.native_allocation_granularity, native_granularity);
  }

  void CreateInstructions(const std::vector<uint8_t>& first,
                          const std::vector<uint8_t>& second) {
    ASSERT_NO_FATAL_FAILURE(
        AllocateInstructions(context_, second.size(), &memory_));
    ASSERT_LE(first.size(), instruction_stride_);

    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map.byte_length = instruction_stride_ + second.size();
    ASSERT_EQ(api_->memory_map(memory_, &map, &mapping_), AMDF_STATUS_OK);
    amdf_host_mapping_info_t mapping_info = {};
    mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping_info.structure_size = sizeof(mapping_info);
    ASSERT_EQ(api_->host_mapping_query_info(mapping_, &mapping_info),
              AMDF_STATUS_OK);
    pointer_ = static_cast<uint8_t*>(mapping_info.pointer);
    std::memcpy(pointer_, first.data(), first.size());
    std::memcpy(pointer_ + instruction_stride_, second.data(), second.size());
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

  void CreateQueue(uint32_t capacity = 0) {
    std::cerr << "[XDNA] Acquiring queue and admitting firmware" << std::endl;
    amdf_xdna_kernel_queue_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    create_info.queue_family_ordinal = QueryKernelQueueFamily();
    create_info.maximum_pending_submission_count = capacity;
    ASSERT_NE(create_info.queue_family_ordinal, UINT32_MAX);
    ASSERT_TRUE(amdf_status_is_ok(
        xdna_api_->kernel_queue_create(context_, &create_info, &queue_)));
  }

  void RunPipelinedTransactions(uint32_t capacity) {
    amdf_xdna_endpoint_info_t identity = {};
    identity.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    identity.structure_size = sizeof(identity);
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &identity),
              AMDF_STATUS_OK);
    if (identity.architecture != AMDF_XDNA_ARCHITECTURE_AIE2P) {
      GTEST_SKIP() << "register transaction encoder requires AIE2P";
    }
    auto first = MakeNoOpTransaction();
    auto second = MakeNoOpTransaction();
    // Each independent command establishes and checks its own SRAM contents.
    // No state is assumed to survive scheduling between native submissions.
    for (uint32_t word = 0; word < 16; ++word) {
      const uint32_t address = (1u << 20) + 0x1000 + word * 4;
      const uint32_t value = 0x13579bdfu + word * 0x10203u;
      AppendRegisterOperation(first, RegisterOperation::kWrite, address, value,
                              UINT32_MAX);
      AppendRegisterOperation(first, RegisterOperation::kPoll, address, value,
                              UINT32_MAX);
      AppendRegisterOperation(second, RegisterOperation::kWrite, address,
                              ~value, UINT32_MAX);
      AppendRegisterOperation(second, RegisterOperation::kPoll, address, ~value,
                              UINT32_MAX);
    }
    ASSERT_NO_FATAL_FAILURE(CreateInstructions(first, second));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(capacity));
    amdf_kernel_queue_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->kernel_queue_query_info(queue_, &info), AMDF_STATUS_OK);
    ASSERT_NE(info.maximum_pending_submission_count, 0u);
    const uint32_t command_count =
        std::min(info.maximum_pending_submission_count, uint32_t{4096});
    amdf_xdna_kernel_command_t command = {};
    command.memory = memory_;
    command.byte_length = first.size();
    amdf_xdna_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = &command;
    uint64_t previous = 0;
    for (uint32_t round = 0; round < 3; ++round) {
      amdf_status_t status = AMDF_STATUS_OK;
      uint64_t last = 0;
      for (uint32_t i = 0; i < command_count && amdf_status_is_ok(status);
           ++i) {
        command.byte_offset = ((i + round) % 2) * instruction_stride_;
        uint64_t point = 0;
        status = xdna_api_->kernel_queue_submit(queue_, &submit, &point);
        if (amdf_status_is_ok(status)) {
          EXPECT_GT(point, previous);
          previous = last = point;
        }
      }
      // A rejected suffix does not cancel its accepted prefix.
      if (last != 0) {
        ASSERT_EQ(
            api_->kernel_queue_wait(queue_, last, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  instruction_stride_ + second.size()),
              AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(pointer_, first.data(), first.size()), 0);
    EXPECT_EQ(std::memcmp(pointer_ + instruction_stride_, second.data(),
                          second.size()),
              0);
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

class XdnaKernelQueueNotificationTest : public XdnaKernelQueueTest {
 protected:
  void TearDown() override {
    if (notification_pending_) {
      const auto consumed = amdf::cts::WaitNativeEvent(event_);
      EXPECT_TRUE(consumed);
      if (!consumed) {
        return;
      }
      notification_pending_ = false;
    }
    XdnaKernelQueueTest::TearDown();
    if (queue_ == nullptr && event_.type != AMDF_NATIVE_EVENT_TYPE_NONE) {
      EXPECT_TRUE(amdf::cts::DestroyNativeEvent(&event_));
    }
  }

  void CreateEvent() {
    amdf_kernel_queue_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->kernel_queue_query_info(queue_, &info), AMDF_STATUS_OK);
    if ((info.notification_types &
         (UINT64_C(1) << amdf::cts::NativeEventType())) == 0) {
      GTEST_SKIP() << "queue does not support this platform's native events";
    }
    ASSERT_TRUE(amdf::cts::CreateNativeEvent(&event_));
  }

  void RequestNotification(uint64_t submission) {
    // This stack descriptor expires before delivery. Only the native event
    // itself remains owned by the caller.
    const amdf_native_event_t borrowed = event_;
    ASSERT_EQ(
        api_->kernel_queue_request_notification(queue_, submission, &borrowed),
        AMDF_STATUS_OK);
    notification_pending_ = true;
  }

  void ConsumeNotification() {
    ASSERT_TRUE(amdf::cts::WaitNativeEvent(event_));
    notification_pending_ = false;
  }

  void RefreshThroughNotification(uint64_t last) {
    amdf_kernel_queue_status_t checked = {};
    checked.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    checked.structure_size = sizeof(checked);
    ASSERT_EQ(api_->kernel_queue_query_status(queue_, &checked),
              AMDF_STATUS_OK);
    while (checked.retired_submission < last) {
      // Use an actual accepted point; public identities need not be dense.
      ASSERT_NO_FATAL_FAILURE(RequestNotification(last));
      ASSERT_NO_FATAL_FAILURE(ConsumeNotification());
      ASSERT_EQ(api_->kernel_queue_refresh_status(queue_, &checked),
                AMDF_STATUS_OK);
      ASSERT_EQ(checked.terminal_status, AMDF_STATUS_OK);
    }
  }

  // Caller-owned event, separate from queue packet storage and checked
  // progress.
  amdf_native_event_t event_ = {};
  // One outstanding wake in this client flow, reconciled even on test failure.
  bool notification_pending_ = false;
};

TEST_F(XdnaKernelQueueNotificationTest,
       ChecksCompletedBatchesAndReusesPacketStorage) {
  const auto transaction = MakeNoOpTransaction();
  ASSERT_NO_FATAL_FAILURE(CreateInstructions(transaction, transaction));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(3));
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  if (IsSkipped()) {
    return;
  }
  amdf_xdna_kernel_command_t command = {};
  command.memory = memory_;
  command.byte_length = transaction.size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;

  // Notify the first accepted native point before a successor can capture it.
  uint64_t last = 0;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &last),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(RefreshThroughNotification(last));
  for (uint32_t round = 0; round < 3; ++round) {
    for (uint32_t i = 0; i < 3; ++i) {
      command.byte_offset = ((round + i) % 2) * instruction_stride_;
      ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &last),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(RefreshThroughNotification(last));
  }
  ASSERT_EQ(api_->host_mapping_cache_control(
                mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                instruction_stride_ + transaction.size()),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pointer_, transaction.data(), transaction.size()), 0);
  EXPECT_EQ(std::memcmp(pointer_ + instruction_stride_, transaction.data(),
                        transaction.size()),
            0);
}

TEST_F(XdnaKernelQueueNotificationTest,
       OneShotRequestsRemainFreshAfterCompletionAndSlotReuse) {
  const auto transaction = MakeNoOpTransaction();
  ASSERT_NO_FATAL_FAILURE(CreateInstructions(transaction, transaction));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(1));
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  if (IsSkipped()) {
    return;
  }
  amdf_xdna_kernel_command_t command = {};
  command.memory = memory_;
  command.byte_length = transaction.size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;
  uint64_t first = 0;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &first),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->kernel_queue_wait(queue_, first, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);

  // Already-checked requests signal before return, so both notifications are
  // delivered before consumption and may coalesce into one readiness edge.
  ASSERT_NO_FATAL_FAILURE(RequestNotification(first));
  ASSERT_NO_FATAL_FAILURE(RequestNotification(first));
  ASSERT_NO_FATAL_FAILURE(ConsumeNotification());
  bool ready = false;
  ASSERT_TRUE(amdf::cts::TryConsumeNativeEvent(event_, &ready));
  EXPECT_FALSE(ready);

  uint64_t second = 0;
  command.byte_offset = instruction_stride_;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &second),
            AMDF_STATUS_OK);
  ASSERT_TRUE(amdf::cts::TryConsumeNativeEvent(event_, &ready));
  EXPECT_FALSE(ready);  // Submission did not implicitly rearm the event.
  ASSERT_NO_FATAL_FAILURE(RequestNotification(first));
  ASSERT_NO_FATAL_FAILURE(ConsumeNotification());
  amdf_kernel_queue_status_t checked = {};
  checked.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
  checked.structure_size = sizeof(checked);
  ASSERT_EQ(api_->kernel_queue_query_status(queue_, &checked), AMDF_STATUS_OK);
  EXPECT_EQ(checked.retired_submission, first);
  ASSERT_NO_FATAL_FAILURE(RefreshThroughNotification(second));
  ASSERT_NO_FATAL_FAILURE(RequestNotification(first));
  ASSERT_NO_FATAL_FAILURE(ConsumeNotification());
  ASSERT_TRUE(amdf::cts::TryConsumeNativeEvent(event_, &ready));
  EXPECT_FALSE(ready);
}

TEST_F(XdnaKernelQueueTest, SubmitsImmutableRangesAndReacquiresQueue) {
  const auto transaction = MakeNoOpTransaction();
  ASSERT_NO_FATAL_FAILURE(CreateInstructions(transaction, transaction));
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  amdf_xdna_kernel_command_t command = {};
  command.memory = memory_;
  command.byte_length = MakeNoOpTransaction().size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;

  uint64_t previous_submission = 0;
  for (uint64_t i = 0; i < 3; ++i) {
    command.byte_offset = (i % 2) * instruction_stride_;
    uint64_t submission = 0;
    ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
              AMDF_STATUS_OK);
    EXPECT_GT(submission, previous_submission);
    // Observation does not release command ownership, even if the device has
    // already completed this short transaction. This batch has no capacity
    // pressure; checked retirement occurs in the explicit wait below.
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    ASSERT_EQ(api_->kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
    EXPECT_EQ(status.retired_submission, previous_submission);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
        AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
    EXPECT_EQ(status.retired_submission, submission);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
    previous_submission = submission;
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

  ASSERT_EQ(DestroyQueue(), AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  uint64_t submission = 0;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
            AMDF_STATUS_OK);
  EXPECT_NE(submission, 0u);
  ASSERT_EQ(
      api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
      AMDF_STATUS_OK);
}

TEST_F(XdnaKernelQueueTest, PipelinesCommandsAtConfiguredCapacity) {
  RunPipelinedTransactions(8);
}

TEST_F(XdnaKernelQueueTest, PipelinesCommandsAtDefaultCapacity) {
  RunPipelinedTransactions(0);
}

TEST_F(XdnaKernelQueueTest, RefreshRetiresBatchesAndReusesPacketStorage) {
  const auto transaction = MakeNoOpTransaction();
  ASSERT_NO_FATAL_FAILURE(CreateInstructions(transaction, transaction));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(3));
  amdf_kernel_queue_status_t checked = {};
  checked.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
  checked.structure_size = sizeof(checked);
  ASSERT_EQ(api_->kernel_queue_refresh_status(queue_, &checked),
            AMDF_STATUS_OK);
  EXPECT_EQ(checked.retired_submission, 0u);
  amdf_xdna_kernel_command_t command = {};
  command.memory = memory_;
  command.byte_length = transaction.size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;
  uint64_t last = 0;
  for (uint32_t round = 0; round < 3; ++round) {
    for (uint32_t i = 0; i < 3; ++i) {
      command.byte_offset = ((round + i) % 2) * instruction_stride_;
      ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &last),
                AMDF_STATUS_OK);
    }
    // This caller explicitly polls checked progress. The outer CTS harness
    // bounds a hang; no execution deadline or native wait changes the result.
    while (checked.retired_submission < last) {
      const uint64_t previous = checked.retired_submission;
      ASSERT_EQ(api_->kernel_queue_refresh_status(queue_, &checked),
                AMDF_STATUS_OK);
      ASSERT_EQ(checked.terminal_status, AMDF_STATUS_OK);
      EXPECT_GE(checked.retired_submission, previous);
      ASSERT_LE(checked.retired_submission, last);
      if (checked.retired_submission < last) {
        std::this_thread::yield();
      }
    }
    ASSERT_EQ(api_->kernel_queue_query_status(queue_, &checked),
              AMDF_STATUS_OK);
    EXPECT_EQ(checked.retired_submission, last);
    EXPECT_EQ(checked.state, AMDF_QUEUE_STATE_ACTIVE);
  }
  ASSERT_EQ(api_->host_mapping_cache_control(
                mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                instruction_stride_ + transaction.size()),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pointer_, transaction.data(), transaction.size()), 0);
  EXPECT_EQ(std::memcmp(pointer_ + instruction_stride_, transaction.data(),
                        transaction.size()),
            0);
}

TEST_F(XdnaKernelQueueTest, PrivateBackingIsQualifiedByExactContext) {
  const auto transaction = MakeNoOpTransaction();
  ASSERT_NO_FATAL_FAILURE(CreateInstructions(transaction, transaction));
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  amdf_xdna_context_create_info_t create = {};
  create.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.logical_column_count = 1;
  create.physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
  create.acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  ASSERT_EQ(xdna_api_->context_create(device_, &create, &sibling_.context),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(AllocateInstructions(
      sibling_.context, transaction.size(), &sibling_.memory));
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
  ASSERT_EQ(api_->memory_destroy(std::exchange(sibling_.memory, nullptr)),
            AMDF_STATUS_OK);
  ASSERT_EQ(xdna_api_->context_destroy(sibling_.context), AMDF_STATUS_OK);
  sibling_.context = nullptr;
  command.memory = memory_;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
            AMDF_STATUS_OK);
  ASSERT_EQ(
      api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
      AMDF_STATUS_OK);
}

TEST_F(XdnaKernelQueueTest,
       PreservesApplicationStateWithinSubmissionsAcrossQueueLeases) {
  amdf_xdna_endpoint_info_t identity = {};
  identity.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
  identity.structure_size = sizeof(identity);
  ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &identity),
            AMDF_STATUS_OK);
  ASSERT_EQ(identity.architecture, AMDF_XDNA_ARCHITECTURE_AIE2P);
  amdf_xdna_context_info_t context_info = {};
  context_info.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO;
  context_info.structure_size = sizeof(context_info);
  ASSERT_EQ(xdna_api_->context_query_info(context_, &context_info),
            AMDF_STATUS_OK);
  ASSERT_EQ(context_info.row_count, 6u);

  auto transaction = MakeNoOpTransaction();
  // One AIE2P column has a memory row followed by four compute rows. Hold each
  // core in reset and publish each lock's first credit for this command.
  // Admission must not leave a sample acquire or DMA request to consume it.
  for (uint32_t row = 2; row < context_info.row_count; ++row) {
    const uint32_t base = row << 20;
    AppendRegisterOperation(transaction, RegisterOperation::kMaskedWrite,
                            base + 0x32000, 2, 3);
    for (uint32_t lock = 0; lock < 16; ++lock) {
      const uint32_t address = base + 0x1F000 + lock * 16;
      AppendRegisterOperation(transaction, RegisterOperation::kWrite, address,
                              1, UINT32_MAX);
      AppendRegisterOperation(transaction, RegisterOperation::kPoll, address, 1,
                              0x3F);
    }
  }
  // Populate independent 64-byte application regions in memory-tile SRAM and
  // every core's data memory.
  for (uint32_t row = 1; row < context_info.row_count; ++row) {
    for (uint32_t word = 0; word < 16; ++word) {
      const uint32_t address = (row << 20) + 0x1000 + word * 4;
      const uint32_t value = 0xC0DE0000u | (row << 8) | word;
      AppendRegisterOperation(transaction, RegisterOperation::kWrite, address,
                              value, UINT32_MAX);
      AppendRegisterOperation(transaction, RegisterOperation::kPoll, address,
                              value, UINT32_MAX);
    }
  }
  // Observe every initialized region again after all writes. The native
  // command owns this state interval; time-sliced contexts need not retain
  // tile state across separate submissions, even on the same queue.
  for (uint32_t row = 2; row < context_info.row_count; ++row) {
    for (uint32_t lock = 0; lock < 16; ++lock) {
      AppendRegisterOperation(transaction, RegisterOperation::kPoll,
                              (row << 20) + 0x1F000 + lock * 16, 1, 0x3F);
    }
  }
  for (uint32_t row = 1; row < context_info.row_count; ++row) {
    for (uint32_t word = 0; word < 16; ++word) {
      AppendRegisterOperation(transaction, RegisterOperation::kPoll,
                              (row << 20) + 0x1000 + word * 4,
                              0xC0DE0000u | (row << 8) | word, UINT32_MAX);
    }
  }
  ASSERT_NO_FATAL_FAILURE(CreateInstructions(transaction, transaction));
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  amdf_xdna_kernel_command_t command = {};
  command.memory = memory_;
  command.byte_length = transaction.size();
  amdf_xdna_kernel_queue_submission_info_t submit = {};
  submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
  submit.structure_size = sizeof(submit);
  submit.command_count = 1;
  submit.commands = &command;
  for (uint32_t generation = 0; generation < 8; ++generation) {
    SCOPED_TRACE(generation);
    if (generation == 4) {
      ASSERT_EQ(DestroyQueue(), AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(CreateQueue());
    }
    command.byte_offset = (generation % 2) * instruction_stride_;
    uint64_t submission = 0;
    ASSERT_EQ(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
              AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
        AMDF_STATUS_OK);
  }
  ASSERT_EQ(api_->host_mapping_cache_control(
                mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                instruction_stride_ + transaction.size()),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pointer_, transaction.data(), transaction.size()), 0);
  EXPECT_EQ(std::memcmp(pointer_ + instruction_stride_, transaction.data(),
                        transaction.size()),
            0);
}

}  // namespace
