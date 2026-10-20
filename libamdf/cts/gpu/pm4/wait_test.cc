// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <string>

#include "libamdf/cts/gpu/pm4/command_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

namespace {

constexpr size_t kMemoryByteLength = 4096;
using MemoryWords = std::array<uint64_t, kMemoryByteLength / sizeof(uint64_t)>;

MemoryWords MakeGuardWords(uint64_t seed) {
  MemoryWords words;
  for (size_t i = 0; i < words.size(); ++i) {
    words[i] = seed ^ (i * UINT64_C(0x030507090b0d0f11));
  }
  return words;
}

MemoryWords ReadWords(const GpuMemory& memory) {
  MemoryWords words;
  std::memcpy(words.data(), memory.host.pointer, kMemoryByteLength);
  return words;
}

struct WaitCase {
  // Stable GoogleTest parameter name describing the field partition.
  const char* name;
  // Memory operand width in bytes, either four or eight.
  size_t byte_length;
  // Comparison applied to the masked memory operand.
  Pm4MemoryComparison comparison;
  // Initial value that does not satisfy the comparison.
  uint64_t initial_value;
  // Packet reference value, with no bits outside mask set.
  uint64_t reference;
  // Significant bits of the polled memory operand.
  uint64_t mask;
  // Value published by the finite producer, satisfying the comparison.
  uint64_t final_value;
};

// Both operand widths cover every conditional function with full and partial
// masks. Ordered boundaries keep the compared value below the sign bit;
// the explicit unsigned GE cases separately exercise that interpretation.
constexpr std::array<WaitCase, 25> kWaitCases = {{
    {"Equal32", 4, Pm4MemoryComparison::kEqual, 0, 0x13579bdf, UINT32_MAX,
     0x13579bdf},
    {"MaskedEqual32", 4, Pm4MemoryComparison::kEqual, 0xaa000011, 0x5a000000,
     0xff000000, 0x5a123456},
    {"NotEqual32", 4, Pm4MemoryComparison::kNotEqual, 0x80000000, 0x80000000,
     UINT32_MAX, 0},
    {"Less32EqualBoundary", 4, Pm4MemoryComparison::kLess, 0x40000000,
     0x40000000, UINT32_MAX, 0x3fffffff},
    {"LessOrEqual32EqualBoundary", 4, Pm4MemoryComparison::kLessOrEqual,
     0x40000001, 0x40000000, UINT32_MAX, 0x40000000},
    {"Greater32EqualBoundary", 4, Pm4MemoryComparison::kGreater, 0x40000000,
     0x40000000, UINT32_MAX, 0x40000001},
    {"UnsignedGreaterOrEqual32Equal", 4, Pm4MemoryComparison::kGreaterOrEqual,
     0x7fffffff, 0x80000000, UINT32_MAX, 0x80000000},
    {"UnsignedGreaterOrEqual32Above", 4, Pm4MemoryComparison::kGreaterOrEqual,
     0x7fffffff, 0x80000000, UINT32_MAX, 0x80000001},
    {"MaskedLess32EqualBoundary", 4, Pm4MemoryComparison::kLess, 0xa540005a,
     0x00400000, 0x00ffff00, 0xa53fff5a},
    {"MaskedLessOrEqual32Below", 4, Pm4MemoryComparison::kLessOrEqual,
     0xa540015a, 0x00400000, 0x00ffff00, 0xa53fff5a},
    {"MaskedNotEqual32", 4, Pm4MemoryComparison::kNotEqual, 0xa540005a,
     0x00400000, 0x00ffff00, 0xa540015a},
    {"MaskedGreaterOrEqual32Equal", 4, Pm4MemoryComparison::kGreaterOrEqual,
     0xa53fff5a, 0x00400000, 0x00ffff00, 0xa540005a},
    {"MaskedGreater32EqualBoundary", 4, Pm4MemoryComparison::kGreater,
     0xa540005a, 0x00400000, 0x00ffff00, 0xa540015a},
    {"Equal64HighHalf", 8, Pm4MemoryComparison::kEqual,
     UINT64_C(0x123456780badc0de), UINT64_C(0x876543210badc0de), UINT64_MAX,
     UINT64_C(0x876543210badc0de)},
    {"MaskedEqual64HighHalf", 8, Pm4MemoryComparison::kEqual,
     UINT64_C(0x1000000013579bdf), UINT64_C(0x8000000000000000),
     UINT64_C(0xffffffff00000000), UINT64_C(0x8000000013579bdf)},
    {"NotEqual64HighHalf", 8, Pm4MemoryComparison::kNotEqual,
     UINT64_C(0x123456780badc0de), UINT64_C(0x123456780badc0de), UINT64_MAX,
     UINT64_C(0x876543210badc0de)},
    {"Less64HighHalf", 8, Pm4MemoryComparison::kLess,
     UINT64_C(0x0000000200000001), UINT64_C(0x0000000200000001), UINT64_MAX,
     UINT64_C(0x0000000100000001)},
    {"LessOrEqual64EqualBoundary", 8, Pm4MemoryComparison::kLessOrEqual,
     UINT64_C(0x0000000100000001), UINT64_C(0x0000000100000000), UINT64_MAX,
     UINT64_C(0x0000000100000000)},
    {"Greater64EqualBoundary", 8, Pm4MemoryComparison::kGreater,
     UINT64_C(0x0000000100000000), UINT64_C(0x0000000100000000), UINT64_MAX,
     UINT64_C(0x0000000100000001)},
    {"MaskedLess64Epoch", 8, Pm4MemoryComparison::kLess,
     UINT64_C(0xbfffffffffffffff), UINT64_C(0x3fffffffffffffff),
     UINT64_C(0x7fffffffffffffff), UINT64_C(0xbffffffffffffffe)},
    {"UnsignedGreaterOrEqual64Carry", 8, Pm4MemoryComparison::kGreaterOrEqual,
     UINT64_C(0x00000000ffffffff), UINT64_C(0x0000000100000000), UINT64_MAX,
     UINT64_C(0x0000000100000001)},
    {"MaskedNotEqual64", 8, Pm4MemoryComparison::kNotEqual,
     UINT64_C(0xa54000de13579bdf), UINT64_C(0x0040000000000000),
     UINT64_C(0x00ffff0000000000), UINT64_C(0xa54001de13579bdf)},
    {"MaskedLessOrEqual64Below", 8, Pm4MemoryComparison::kLessOrEqual,
     UINT64_C(0xa54001de13579bdf), UINT64_C(0x0040000000000000),
     UINT64_C(0x00ffff0000000000), UINT64_C(0xa53fffde13579bdf)},
    {"MaskedGreaterOrEqual64Equal", 8, Pm4MemoryComparison::kGreaterOrEqual,
     UINT64_C(0xa53fffde13579bdf), UINT64_C(0x0040000000000000),
     UINT64_C(0x00ffff0000000000), UINT64_C(0xa54000de13579bdf)},
    {"MaskedGreater64EqualBoundary", 8, Pm4MemoryComparison::kGreater,
     UINT64_C(0xa54000de13579bdf), UINT64_C(0x0040000000000000),
     UINT64_C(0x00ffff0000000000), UINT64_C(0xa54001de13579bdf)},
}};

class Pm4WaitTest : public Pm4CommandTest,
                    public ::testing::WithParamInterface<WaitCase> {
 protected:
  Pm4WaitTest() : Pm4CommandTest(AMDF_QUEUE_ROLE_ATOMIC) {}

  enum class Producer { kCopy, kAtomicStore, kHostStore };
  void RunHandoff(Producer producer_kind);

  void ReleaseWait() {
    if (pending_control_ == nullptr) {
      return;
    }
    const auto address =
        reinterpret_cast<uintptr_t>(pending_control_->host.pointer);
    if (GetParam().byte_length == 4) {
      GpuStoreRelease<uint32_t>(address,
                                static_cast<uint32_t>(GetParam().final_value));
    } else {
      GpuStoreRelease<uint64_t>(address, GetParam().final_value);
    }
    pending_control_ = nullptr;
  }

  void TearDown() override {
    // An early assertion can leave the consumer published without its
    // producer. Release that finite wait before retiring either queue or
    // releasing any of their backing. This does not establish completion.
    ReleaseWait();
    Pm4CommandTest::TearDown();
  }

  void EmitWait(Pm4CommandWriter& commands, uint64_t address) {
    const auto& parameters = GetParam();
    RecordProperty("wait_operand_byte_length", parameters.byte_length);
    RecordProperty("wait_comparison", static_cast<int>(parameters.comparison));
    RecordProperty("wait_reference", std::to_string(parameters.reference));
    RecordProperty("wait_mask", std::to_string(parameters.mask));
    RecordProperty("wait_initial", std::to_string(parameters.initial_value));
    RecordProperty("wait_final", std::to_string(parameters.final_value));
    if (parameters.byte_length == 4) {
      commands.WaitMemory32(
          address, static_cast<uint32_t>(parameters.reference),
          parameters.comparison, static_cast<uint32_t>(parameters.mask));
    } else {
      commands.WaitMemory64(address, parameters.reference,
                            parameters.comparison, parameters.mask);
    }
  }

  // Case-owned control backing retained through the consumer's final use.
  GpuMemory* pending_control_ = nullptr;
};

TEST_P(Pm4WaitTest, AlreadySatisfiedOperandAllowsFollowingWork) {
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kMemoryByteLength, &control));
  auto* values = static_cast<uint64_t*>(control->host.pointer);
  auto expected = MakeGuardWords(UINT64_C(0x39c4e7a625d18b03));
  expected[0] = GetParam().byte_length == 4
                    ? (expected[0] & UINT64_C(0xffffffff00000000)) |
                          GetParam().final_value
                    : GetParam().final_value;
  expected[8] = expected[0];
  expected[16] = (expected[16] & UINT64_C(0xffffffff00000000)) | 1;
  std::memcpy(values, expected.data(), kMemoryByteLength);
  values[8] = ~expected[8];
  values[16] &= UINT64_C(0xffffffff00000000);
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size_bytes(), 256u);
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  commands.SystemBarrier();
  EmitWait(commands, control->device_address);
  commands.CopyData64(control->device_address, control->device_address + 64);
  commands.SystemBarrier();
  commands.WriteData32(control->device_address + 128, 1);
  commands.PadToEightWords();
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(
      GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(values + 16), 1));
  // Observe payload before retirement queries can add synchronization.
  EXPECT_EQ(ReadWords(*control), expected);
  RecordProperty("memory_checked_bytes", kMemoryByteLength);
  EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
}

void Pm4WaitTest::RunHandoff(Producer producer_kind) {
  constexpr uint64_t kPayload = UINT64_C(0x2468ace013579bdf);
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* source = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kMemoryByteLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kMemoryByteLength, &intermediate));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, kMemoryByteLength, &target));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kMemoryByteLength, &control));
  if (producer_kind != Producer::kCopy) {
    constexpr auto kOperations =
        AMDF_ATOMIC_OPERATION_STORE | AMDF_ATOMIC_OPERATION_WAIT;
    const auto operations =
        GetParam().byte_length == 4
            ? control->access_info.atomic_operations_32 &
                  family_.atomic_capabilities.operations_without_dispatch_32
            : control->access_info.atomic_operations_64 &
                  family_.atomic_capabilities.operations_without_dispatch_64;
    if ((operations & kOperations) != kOperations) {
      GTEST_SKIP() << "queue and backing do not qualify native store/wait";
    }
    const auto device_site = control->DeviceSite(family_.ordinal);
    const auto producer_site = producer_kind == Producer::kHostStore
                                   ? control->HostSite()
                                   : device_site;
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair),
    };
    ASSERT_EQ(api_->memory_query_pair_info(&producer_site, &device_site, &pair),
              AMDF_STATUS_OK);
    const auto reach = GetParam().byte_length == 4 ? pair.atomic_reach.scope_32
                                                   : pair.atomic_reach.scope_64;
    const auto required_scope = producer_kind == Producer::kHostStore
                                    ? AMDF_ATOMIC_SCOPE_SYSTEM
                                    : AMDF_ATOMIC_SCOPE_DEVICE;
    if (reach < required_scope) {
      GTEST_SKIP() << "backing does not qualify producer/consumer atomic reach";
    }
    RecordProperty("atomic_reach", reach);
  }
  auto* input = static_cast<uint64_t*>(source->host.pointer);
  auto* milestones = static_cast<uint64_t*>(control->host.pointer);
  auto expected_source = MakeGuardWords(UINT64_C(0x6d2ac491579b03e8));
  auto expected_intermediate = MakeGuardWords(UINT64_C(0xb730e85a913df246));
  auto expected_target = MakeGuardWords(UINT64_C(0x4962d5e713bfa80c));
  auto expected_control = MakeGuardWords(UINT64_C(0x25a64bc3e07d198f));
  expected_source[0] =
      GetParam().byte_length == 4
          ? (expected_source[0] & UINT64_C(0xffffffff00000000)) |
                GetParam().final_value
          : GetParam().final_value;
  expected_source[8] = kPayload;
  expected_intermediate[0] = expected_target[0] = kPayload;
  expected_control[0] =
      GetParam().byte_length == 4
          ? (expected_control[0] & UINT64_C(0xffffffff00000000)) |
                GetParam().final_value
          : GetParam().final_value;
  for (size_t i : {8u, 16u, 24u}) {
    expected_control[i] =
        (expected_control[i] & UINT64_C(0xffffffff00000000)) | 1;
  }
  std::memcpy(input, expected_source.data(), kMemoryByteLength);
  std::memcpy(intermediate->host.pointer, expected_intermediate.data(),
              kMemoryByteLength);
  std::memcpy(target->host.pointer, expected_target.data(), kMemoryByteLength);
  std::memcpy(milestones, expected_control.data(), kMemoryByteLength);
  *static_cast<uint64_t*>(intermediate->host.pointer) = ~kPayload;
  *static_cast<uint64_t*>(target->host.pointer) = ~kPayload;
  milestones[0] = GetParam().byte_length == 4
                      ? (expected_control[0] & UINT64_C(0xffffffff00000000)) |
                            GetParam().initial_value
                      : GetParam().initial_value;
  for (size_t i : {8u, 16u, 24u}) {
    milestones[i] &= UINT64_C(0xffffffff00000000);
  }

  GpuCommandQueue* producer = nullptr;
  GpuCommandQueue* consumer = nullptr;
  if (producer_kind != Producer::kHostStore) {
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  }
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  ASSERT_GT(consumer->words().size_bytes(), 256u);
  size_t producer_word_count = 0;
  if (producer != nullptr) {
    ASSERT_GE(producer->words().size_bytes(), 256u);
    Pm4CommandWriter produce(producer->words().data(), *pm4_profile_);
    produce.SystemBarrier();
    produce.CopyData64(source->device_address + 64,
                       intermediate->device_address);
    produce.SystemBarrier();
    if (producer_kind == Producer::kAtomicStore) {
      if (GetParam().byte_length == 4) {
        produce.Store32(control->device_address,
                        static_cast<uint32_t>(GetParam().final_value));
      } else {
        produce.Store64(control->device_address, GetParam().final_value);
      }
    } else if (GetParam().byte_length == 4) {
      produce.CopyData32(source->device_address, control->device_address);
    } else {
      produce.CopyData64(source->device_address, control->device_address);
    }
    produce.SystemBarrier();
    produce.WriteData32(control->device_address + 128, 1);
    produce.PadToEightWords();
    producer_word_count = produce.word_count();
  }

  Pm4CommandWriter consume(consumer->words().data(), *pm4_profile_);
  consume.SystemBarrier();
  consume.WriteData32(control->device_address + 64, 1);
  EmitWait(consume, control->device_address);
  consume.SystemBarrier();
  consume.CopyData64(intermediate->device_address, target->device_address);
  consume.SystemBarrier();
  consume.WriteData32(control->device_address + 192, 1);
  consume.PadToEightWords();

  // Readiness belongs to the consumer's confirmed prefix. The host then
  // publishes the finite producer, without any sleep or intermediate host
  // retirement that could substitute for the tested memory dependency.
  pending_control_ = control;
  ASSERT_NO_FATAL_FAILURE(
      consumer->Publish(api_, gpu_api_, consume.word_count()));
  ASSERT_TRUE(
      GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(milestones + 8), 1));
  EXPECT_EQ(
      GpuLoadAcquire<uint32_t>(reinterpret_cast<uintptr_t>(milestones + 24)),
      0u);
  if (producer_kind == Producer::kHostStore) {
    std::memcpy(intermediate->host.pointer, &kPayload, sizeof(kPayload));
    GpuStoreRelease<uint32_t>(reinterpret_cast<uintptr_t>(milestones + 16), 1);
    ReleaseWait();
  } else {
    ASSERT_NO_FATAL_FAILURE(
        producer->Publish(api_, gpu_api_, producer_word_count));
  }
  ASSERT_TRUE(
      GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(milestones + 24), 1));
  // The consumer's own marker must suffice for observing its output.
  EXPECT_EQ(ReadWords(*target), expected_target);
  ASSERT_TRUE(
      GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(milestones + 16), 1));
  EXPECT_EQ(ReadWords(*intermediate), expected_intermediate);
  EXPECT_EQ(ReadWords(*control), expected_control);
  EXPECT_EQ(ReadWords(*source), expected_source);
  RecordProperty("memory_checked_bytes", 4 * kMemoryByteLength);
  if (producer != nullptr) {
    EXPECT_NO_FATAL_FAILURE(producer->WaitRetired(api_));
  }
  EXPECT_NO_FATAL_FAILURE(consumer->WaitRetired(api_));
}

TEST_P(Pm4WaitTest, ConsumerWaitPrecedesProducerPublication) {
  RunHandoff(Producer::kCopy);
}

TEST_P(Pm4WaitTest, NativeAtomicStoreReleasesWaitingConsumer) {
  RunHandoff(Producer::kAtomicStore);
}

TEST_P(Pm4WaitTest, HostStoreReleasesWaitingConsumer) {
  RunHandoff(Producer::kHostStore);
}

INSTANTIATE_TEST_SUITE_P(Comparison, Pm4WaitTest,
                         ::testing::ValuesIn(kWaitCases),
                         [](const ::testing::TestParamInfo<WaitCase>& info) {
                           return info.param.name;
                         });

class Pm4UnconditionalWaitTest : public Pm4CommandTest {};

TEST_F(Pm4UnconditionalWaitTest, BothWidthsContinueForEveryOperandRelation) {
  const struct {
    // Memory operand width in bytes.
    size_t byte_length;
    // Backed operand below, equal to or above the reference.
    uint64_t value;
    // Full-width reference compared to the operand.
    uint64_t reference;
  } cases[] = {
      {4, 0x003fffff, 0x00400000},
      {4, 0x00400000, 0x00400000},
      {4, 0x00400001, 0x00400000},
      {8, UINT64_C(0x00000000ffffffff), UINT64_C(0x0000000100000000)},
      {8, UINT64_C(0x0000000100000000), UINT64_C(0x0000000100000000)},
      {8, UINT64_C(0x0000000100000001), UINT64_C(0x0000000100000000)},
  };
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kMemoryByteLength, &control));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size_bytes(), 1024u);
  auto* values = static_cast<uint64_t*>(control->host.pointer);
  auto expected = MakeGuardWords(UINT64_C(0x1f7ca9036db5e248));
  expected[128] = (expected[128] & UINT64_C(0xffffffff00000000)) | 1;
  std::memcpy(values, expected.data(), kMemoryByteLength);
  values[128] &= UINT64_C(0xffffffff00000000);

  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  commands.SystemBarrier();
  for (size_t i = 0; i < std::size(cases); ++i) {
    const auto& test = cases[i];
    const size_t source_word = i * 8;
    const size_t target_word = 64 + source_word;
    expected[source_word] =
        test.byte_length == 4
            ? (expected[source_word] & UINT64_C(0xffffffff00000000)) |
                  test.value
            : test.value;
    expected[target_word] = expected[source_word];
    values[source_word] = expected[source_word];
    values[target_word] = ~expected[target_word];
    const uint64_t address =
        control->device_address + source_word * sizeof(uint64_t);
    if (test.byte_length == 4) {
      commands.WaitMemory32(address, static_cast<uint32_t>(test.reference),
                            Pm4MemoryComparison::kAlways);
    } else {
      commands.WaitMemory64(address, test.reference,
                            Pm4MemoryComparison::kAlways);
    }
    commands.CopyData64(
        address, control->device_address + target_word * sizeof(uint64_t));
  }
  commands.SystemBarrier();
  commands.WriteData32(control->device_address + 128 * sizeof(uint64_t), 1);
  commands.PadToEightWords();
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(
      GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(values + 128), 1));
  EXPECT_EQ(ReadWords(*control), expected);
  RecordProperty("wait_comparison", 0);
  RecordProperty("wait_count", std::size(cases));
  RecordProperty("memory_checked_bytes", kMemoryByteLength);
  EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
}

}  // namespace
