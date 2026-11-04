// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "shared/include/d3dkmt_types.h"
#include "wkmi.h"

namespace {

constexpr size_t kGuardByteLength = 16;
constexpr uint8_t kGuardValue = 0xA5;

class GuardedPayload {
 public:
  explicit GuardedPayload(size_t payload_byte_length)
      : payload_byte_length_(payload_byte_length),
        storage_(kGuardByteLength + payload_byte_length + kGuardByteLength,
                 kGuardValue) {
    std::fill_n(static_cast<uint8_t*>(data()), payload_byte_length_, 0);
  }

  void* data() { return storage_.data() + kGuardByteLength; }

  const uint8_t* payload() const { return storage_.data() + kGuardByteLength; }

  size_t payload_byte_length() const { return payload_byte_length_; }

  bool GuardsIntact() const {
    return std::all_of(storage_.begin(), storage_.begin() + kGuardByteLength,
                       [](uint8_t value) { return value == kGuardValue; }) &&
           std::all_of(storage_.end() - kGuardByteLength, storage_.end(),
                       [](uint8_t value) { return value == kGuardValue; });
  }

 private:
  size_t payload_byte_length_;
  std::vector<uint8_t> storage_;
};

bool PayloadsEqual(const GuardedPayload& a, const GuardedPayload& b) {
  return a.payload_byte_length() == b.payload_byte_length() &&
         std::equal(a.payload(), a.payload() + a.payload_byte_length(),
                    b.payload());
}

TEST(WkmiPayloadTest, PublishedSizesMatchPinnedArtifact) {
  EXPECT_EQ(Wkmi::GetContextPrivDataSize(), 424);
  EXPECT_EQ(Wkmi::GetHwQueuePrivDataSize(), 64);
  EXPECT_EQ(Wkmi::GetSubmitPrivDataSize(), 72);
}

TEST(WkmiPayloadTest, FillersAreBoundedAndDeterministic) {
  GuardedPayload context_a(Wkmi::GetContextPrivDataSize());
  GuardedPayload context_b(Wkmi::GetContextPrivDataSize());
  Wkmi::FillinContextPrivData(context_a.data(), true, 3, 0x1234, false);
  Wkmi::FillinContextPrivData(context_b.data(), true, 3, 0x1234, false);
  EXPECT_TRUE(context_a.GuardsIntact());
  EXPECT_TRUE(context_b.GuardsIntact());
  EXPECT_TRUE(PayloadsEqual(context_a, context_b));

  GuardedPayload queue_a(Wkmi::GetHwQueuePrivDataSize());
  GuardedPayload queue_b(Wkmi::GetHwQueuePrivDataSize());
  Wkmi::FillinHwQueuePrivData(queue_a.data(), true, Wkmi::kNormal,
                              Wkmi::kHwQueuePm4);
  Wkmi::FillinHwQueuePrivData(queue_b.data(), true, Wkmi::kNormal,
                              Wkmi::kHwQueuePm4);
  EXPECT_TRUE(queue_a.GuardsIntact());
  EXPECT_TRUE(queue_b.GuardsIntact());
  EXPECT_TRUE(PayloadsEqual(queue_a, queue_b));

  GuardedPayload submit_a(Wkmi::GetSubmitPrivDataSize());
  GuardedPayload submit_b(Wkmi::GetSubmitPrivDataSize());
  Wkmi::FillinSubmitPrivData(submit_a.data(), 0x12, 0x1000, 64, true);
  Wkmi::FillinSubmitPrivData(submit_b.data(), 0x12, 0x1000, 64, true);
  EXPECT_TRUE(submit_a.GuardsIntact());
  EXPECT_TRUE(submit_b.GuardsIntact());
  EXPECT_TRUE(PayloadsEqual(submit_a, submit_b));
}

TEST(WkmiPayloadTest, SemanticInputsReachOpaqueRecords) {
  GuardedPayload context_a(Wkmi::GetContextPrivDataSize());
  GuardedPayload context_b(Wkmi::GetContextPrivDataSize());
  Wkmi::FillinContextPrivData(context_a.data(), false, 2, 0, false);
  Wkmi::FillinContextPrivData(context_b.data(), true, 3, 0x1000, true);
  EXPECT_FALSE(PayloadsEqual(context_a, context_b));

  GuardedPayload queue_a(Wkmi::GetHwQueuePrivDataSize());
  GuardedPayload queue_b(Wkmi::GetHwQueuePrivDataSize());
  uint32_t* doorbell = nullptr;
  Wkmi::FillinHwQueuePrivData(queue_a.data(), false, Wkmi::kLow,
                              Wkmi::kHwQueuePm4);
  Wkmi::FillinHwQueuePrivData(queue_b.data(), true, Wkmi::kHigh,
                              Wkmi::kHwQueueSdma, 0x2000, 4096, 0, 0, 0,
                              &doorbell);
  EXPECT_FALSE(PayloadsEqual(queue_a, queue_b));
  EXPECT_GE(reinterpret_cast<uint8_t*>(doorbell), queue_b.payload());
  EXPECT_LT(reinterpret_cast<uint8_t*>(doorbell),
            queue_b.payload() + queue_b.payload_byte_length());

  GuardedPayload submit_a(Wkmi::GetSubmitPrivDataSize());
  GuardedPayload submit_b(Wkmi::GetSubmitPrivDataSize());
  Wkmi::FillinSubmitPrivData(submit_a.data(), 0, 0x3000, 64, true);
  Wkmi::FillinSubmitPrivData(submit_b.data(), 0x34, 0x4000, 128, false);
  EXPECT_FALSE(PayloadsEqual(submit_a, submit_b));
}

}  // namespace
