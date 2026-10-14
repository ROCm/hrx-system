// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using Ring = iree_hal_amd_gpu_pm4_ring_t;
using Reservation = iree_hal_amd_gpu_pm4_ring_reservation_t;
constexpr uint32_t kUnwritten = 0xa5a5a5a5;

void ExpectNop(const uint32_t* words, uint32_t count) {
  ASSERT_GE(count, 2u);
  ASSERT_LE(count, 16385u);
  EXPECT_EQ(words[0], 0xc0001000u | ((count - 2) << 16));
  for (uint32_t i = 1; i < count; ++i) {
    EXPECT_EQ(words[i], 0u);
  }
}

TEST(RingTest, InitializationDoesNotTouchNativeStorage) {
  std::array<uint32_t, 64> words;
  words.fill(kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  EXPECT_EQ(ring.words, words.data());
  EXPECT_EQ(ring.capacity, 64u);
  EXPECT_EQ(ring.reserved_words, 0u);
  for (uint32_t word : words) {
    EXPECT_EQ(word, kUnwritten);
  }
}

TEST(RingTest, AllPublicationPaddingResidues) {
  constexpr uint32_t kExtents[] = {16, 24, 24, 24, 24, 24, 24, 32};
  for (uint32_t residue = 0; residue < 8; ++residue) {
    SCOPED_TRACE(residue);
    std::array<uint32_t, 128> words;
    words.fill(kUnwritten);
    Ring ring;
    iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
    Reservation reservation;
    const uint32_t body = 16 + residue;
    ASSERT_TRUE(
        iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, body, &reservation));
    EXPECT_EQ(reservation.words, words.data());
    EXPECT_EQ(reservation.position, 0u);
    EXPECT_EQ(ring.reserved_words, kExtents[residue]);
    for (uint32_t i = 0; i < body; ++i) {
      EXPECT_EQ(words[i], kUnwritten);
    }
    if (residue) {
      ExpectNop(words.data() + body, kExtents[residue] - body);
    }
    EXPECT_EQ(words[kExtents[residue]], kUnwritten);
  }
}

TEST(RingTest, InsufficientCreditLeavesEverythingUnchanged) {
  std::array<uint32_t, 64> words;
  words.fill(kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  Reservation reservation;
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 24, &reservation));
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 24, &reservation));
  const Reservation previous = reservation;
  const auto previous_words = words;
  // Filling every DWORD would alias the native empty-ring position.
  EXPECT_FALSE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 16, &reservation));
  EXPECT_EQ(ring.reserved_words, 48u);
  EXPECT_EQ(reservation.words, previous.words);
  EXPECT_EQ(reservation.position, previous.position);
  EXPECT_EQ(words, previous_words);
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 24, 16, &reservation));
  EXPECT_EQ(reservation.position, 48u);
  EXPECT_EQ(ring.reserved_words, 64u);
}

TEST(RingTest, WrapPreservesTheUnconsumedNotificationTail) {
  std::array<uint32_t, 64> words;
  words.fill(kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  Reservation first;
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 24, &first));
  iree_hal_amd_gpu_pm4_emit_nop(first.words, 24);
  Reservation second;
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 16, 24, &second));
  iree_hal_amd_gpu_pm4_emit_nop(second.words, 24);
  // The completion marker in the second publication ends at 40; its trailing
  // event at [40, 48) remains unconsumed even after that completion is
  // observed.
  const auto before_wrap = words;
  Reservation third;
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 16, 24, &third));
  EXPECT_EQ(words, before_wrap);
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 40, 24, &third));
  EXPECT_EQ(third.position, 64u);
  EXPECT_EQ(third.words, words.data());
  EXPECT_EQ(ring.reserved_words, 88u);
  ExpectNop(words.data() + 48, 16);
  for (uint32_t i = 40; i < 48; ++i) {
    EXPECT_EQ(words[i], before_wrap[i]);
  }
}

TEST(RingTest, CompletionCreditProgressesAcrossEveryBodyResidueAndWrap) {
  for (uint32_t capacity : {64u, 128u, 1024u, 65536u}) {
    SCOPED_TRACE(capacity);
    std::vector<uint32_t> words(capacity, kUnwritten);
    Ring ring;
    iree_hal_amd_gpu_pm4_ring_initialize(words.data(), capacity, &ring);
    uint64_t consumed = 0;
    // Completed release packets return credit immediately. The final eight
    // DWORDs model the independent event tail, followed by native padding.
    for (uint32_t iteration = 0; iteration < 1024; ++iteration) {
      const uint32_t body = 16 + iteration % (capacity == 64 ? 7 : 17);
      const uint64_t previous_end = ring.reserved_words;
      Reservation reservation;
      ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, consumed, body,
                                                        &reservation));
      EXPECT_EQ(reservation.words,
                words.data() + (reservation.position & (capacity - 1)));
      EXPECT_EQ(ring.reserved_words & 7, 0u);
      EXPECT_LT(ring.reserved_words - consumed, capacity);
      if (reservation.position != previous_end) {
        ExpectNop(words.data() + (previous_end & (capacity - 1)),
                  static_cast<uint32_t>(reservation.position - previous_end));
      }
      const uint32_t padding =
          static_cast<uint32_t>(ring.reserved_words - reservation.position) -
          body;
      if (padding) {
        ExpectNop(reservation.words + body, padding);
      }
      iree_hal_amd_gpu_pm4_emit_nop(reservation.words, body);
      consumed = reservation.position + body - 8;
    }
  }
}

TEST(RingTest, LargestAdmittedSpanUsesValidWrapPackets) {
  std::vector<uint32_t> words(65536, kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  uint64_t consumed = 0;
  for (uint32_t iteration = 0; iteration < 12; ++iteration) {
    const uint32_t body = iteration % 2 ? 16382 : 8199;
    const uint64_t previous_end = ring.reserved_words;
    Reservation reservation;
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, consumed, body,
                                                      &reservation));
    if (reservation.position != previous_end) {
      ExpectNop(words.data() + (previous_end & 65535),
                static_cast<uint32_t>(reservation.position - previous_end));
    }
    iree_hal_amd_gpu_pm4_emit_nop(reservation.words, body);
    consumed = reservation.position + body - 8;
  }
  EXPECT_GT(ring.reserved_words, words.size());
}

TEST(RingTest, CompletedMaximumTailsCannotStrandTheNextPublication) {
  std::array<uint32_t, 1024> words;
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  uint64_t consumed = 0;
  for (uint32_t iteration = 0; iteration < 1024; ++iteration) {
    // A 495-DWORD body needs nine DWORDs of padding. Eight notification
    // DWORDs and that padding remain live after the release marker completes.
    // Alternating a shorter body also varies the amount of wrap padding.
    const uint32_t body = iteration % 2 ? 495 : 247;
    Reservation reservation;
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, consumed, body,
                                                      &reservation));
    EXPECT_LT(ring.reserved_words - consumed, words.size());
    consumed = reservation.position + body - 8;
    EXPECT_EQ(ring.reserved_words - consumed, 17u);
  }
}

// Native allocation and mapping dependency; reservation and ownership cleanup
// use the real ring implementation. This fake never executes PM4 commands.
struct NativeRing {
  // Emulated native primary-ring storage returned by the mapping query.
  std::array<uint32_t, 1024> words = {};
  // Host publication operand returned with the primary-ring view.
  uint64_t write_index = 0;
  // Emulated write-only aperture; native execution uses the hardware CTS.
  uint64_t doorbell = 0;
  // Creation input whose exact pointer and scratch payload must be forwarded.
  const amdf_gpu_user_queue_create_info_t* create_info = nullptr;
  // Native operation to fail, or empty for successful calls.
  std::string fail_operation;
  // Original native result supplied at the selected failure boundary.
  amdf_status_t failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  // Ordered calls made by the real owner.
  std::vector<std::string> calls;
  // Whether an owned native queue handle is still live.
  bool queue_live = false;
  // Whether the host producer mapping is still live.
  bool mapping_live = false;
  // Final published index joined during destruction.
  uint64_t consumed_target = 0;

  amdf_status_t Call(const char* operation) {
    calls.push_back(operation);
    return fail_operation == operation ? failure : AMDF_STATUS_OK;
  }

  static amdf_status_t AMDF_CALL
  Create(amdf_device_t* device, const amdf_gpu_user_queue_create_info_t* info,
         amdf_user_queue_t** out_queue) {
    auto* self = reinterpret_cast<NativeRing*>(device);
    EXPECT_EQ(info, self->create_info);
    EXPECT_FALSE(self->queue_live);
    amdf_status_t status = self->Call("create");
    if (amdf_status_is_ok(status)) {
      self->queue_live = true;
      *out_queue = reinterpret_cast<amdf_user_queue_t*>(self);
    }
    return status;
  }

  static amdf_status_t AMDF_CALL Map(amdf_user_queue_t* queue,
                                     amdf_device_t* producer,
                                     amdf_user_queue_mapping_t** out_mapping) {
    auto* self = reinterpret_cast<NativeRing*>(queue);
    EXPECT_TRUE(self->queue_live);
    EXPECT_FALSE(self->mapping_live);
    EXPECT_EQ(producer, nullptr);
    amdf_status_t status = self->Call("map");
    if (amdf_status_is_ok(status)) {
      self->mapping_live = true;
      *out_mapping = reinterpret_cast<amdf_user_queue_mapping_t*>(self);
    }
    return status;
  }

  static amdf_status_t AMDF_CALL
  Query(amdf_user_queue_mapping_t* mapping,
        amdf_user_queue_mapping_info_t* out_info) {
    auto* self = reinterpret_cast<NativeRing*>(mapping);
    EXPECT_TRUE(self->mapping_live);
    EXPECT_EQ(out_info->type, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
    EXPECT_EQ(out_info->structure_size, sizeof(*out_info));
    amdf_status_t status = self->Call("query");
    if (amdf_status_is_ok(status)) {
      out_info->command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
      out_info->format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1;
      out_info->ring_address = reinterpret_cast<uintptr_t>(self->words.data());
      out_info->ring_byte_length = sizeof(self->words);
      out_info->write_index_address =
          reinterpret_cast<uintptr_t>(&self->write_index);
      out_info->doorbell_address = reinterpret_cast<uintptr_t>(&self->doorbell);
      out_info->index_bits = 64;
      out_info->doorbell_bits = 64;
    }
    return status;
  }

  static amdf_status_t AMDF_CALL Wait(amdf_user_queue_t* queue,
                                      uint64_t published_index,
                                      uint64_t timeout,
                                      uint64_t poll_duration) {
    auto* self = reinterpret_cast<NativeRing*>(queue);
    EXPECT_TRUE(self->queue_live);
    EXPECT_EQ(timeout, AMDF_TIMEOUT_INFINITE);
    EXPECT_EQ(poll_duration, 0u);
    self->consumed_target = published_index;
    return self->Call("wait");
  }

  static amdf_status_t AMDF_CALL Unmap(amdf_user_queue_mapping_t* mapping) {
    auto* self = reinterpret_cast<NativeRing*>(mapping);
    EXPECT_TRUE(self->queue_live);
    EXPECT_TRUE(self->mapping_live);
    amdf_status_t status = self->Call("unmap");
    if (amdf_status_is_ok(status)) {
      self->mapping_live = false;
    }
    return status;
  }

  static amdf_status_t AMDF_CALL Destroy(amdf_user_queue_t* queue) {
    auto* self = reinterpret_cast<NativeRing*>(queue);
    EXPECT_TRUE(self->queue_live);
    EXPECT_FALSE(self->mapping_live);
    amdf_status_t status = self->Call("destroy");
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      self->queue_live = false;
    }
    return status;
  }
};

class RingStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_.user_queue_map = NativeRing::Map;
    api_.user_queue_mapping_query_info = NativeRing::Query;
    api_.user_queue_wait_consumed = NativeRing::Wait;
    api_.user_queue_mapping_destroy = NativeRing::Unmap;
    api_.user_queue_destroy = NativeRing::Destroy;
    gpu_.user_queue_create = NativeRing::Create;
    context_.api = &api_;
    context_.gpu = &gpu_;
    context_.device = reinterpret_cast<amdf_device_t*>(&native_);
    native_.create_info = &create_info_;
  }

  iree_status_t Initialize() {
    return iree_hal_amd_gpu_pm4_ring_storage_initialize(
        &context_, &create_info_, &storage_, &ring_, &mapping_info_);
  }

  amdf_status_t Deinitialize(uint64_t published_words = 0) {
    return iree_hal_amd_gpu_pm4_ring_storage_deinitialize(&storage_,
                                                          published_words);
  }

  void TearDown() override {
    EXPECT_FALSE(native_.mapping_live);
    EXPECT_FALSE(native_.queue_live);
  }

  // Dependency state; the native API tables borrow it through the device.
  NativeRing native_;
  // Controlled core native resource operations.
  amdf_api_t api_ = {};
  // Controlled GPU queue constructor.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed native context retained through all cleanup attempts.
  iree_hal_amd_gpu_context_t context_ = {};
  // Admitted queue configuration forwarded intact to the native constructor.
  amdf_gpu_user_queue_create_info_t create_info_ = {
      .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_info_),
      .queue_family_ordinal = 3,
      .priority = AMDF_QUEUE_PRIORITY_HIGH,
      .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
      .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      .ring_byte_length = 4096,
  };
  // Embedded real native owner under test.
  iree_hal_amd_gpu_pm4_ring_storage_t storage_ = {};
  // Independently stored hot reservation state.
  Ring ring_ = {};
  // Immutable publication addresses returned on successful initialization.
  amdf_user_queue_mapping_info_t mapping_info_ = {};
};

TEST_F(RingStorageTest, CapturesMappedStorageWithoutPublication) {
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(storage_.context, &context_);
  EXPECT_EQ(ring_.words, native_.words.data());
  EXPECT_EQ(ring_.capacity, native_.words.size());
  EXPECT_EQ(ring_.reserved_words, 0u);
  EXPECT_EQ(mapping_info_.write_index_address,
            reinterpret_cast<uintptr_t>(&native_.write_index));
  EXPECT_EQ(mapping_info_.doorbell_address,
            reinterpret_cast<uintptr_t>(&native_.doorbell));
  EXPECT_EQ(native_.write_index, 0u);
  EXPECT_EQ(native_.doorbell, 0u);
  EXPECT_EQ(Deinitialize(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.calls, (std::vector<std::string>{"create", "map", "query",
                                                     "unmap", "destroy"}));
  EXPECT_EQ(Deinitialize(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.calls.size(), 5u);
}

TEST_F(RingStorageTest, ConstructionFailurePreservesPartialOwnerForCleanup) {
  for (const char* operation : {"create", "map", "query"}) {
    SCOPED_TRACE(operation);
    native_.fail_operation = operation;
    ring_.reserved_words = 713;
    mapping_info_.format_version = 57;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
    EXPECT_EQ(ring_.reserved_words, 713u);
    EXPECT_EQ(mapping_info_.format_version, 57u);
    EXPECT_EQ(native_.queue_live, native_.fail_operation != "create");
    EXPECT_EQ(native_.mapping_live, native_.fail_operation == "query");
    EXPECT_EQ(Deinitialize(), AMDF_STATUS_OK);
    EXPECT_FALSE(native_.queue_live);
    EXPECT_FALSE(native_.mapping_live);
  }
}

TEST_F(RingStorageTest, JoinsThePublishedTailBeforeRemovingProducerView) {
  IREE_ASSERT_OK(Initialize());
  Reservation reservation;
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring_, 0, 71, &reservation));
  // A release at body end minus the eight-DWORD notification proves only 63
  // DWORDs consumed. Destruction must join all 80, including nine-DWORD
  // padding.
  EXPECT_EQ(ring_.reserved_words, 80u);
  EXPECT_EQ(Deinitialize(ring_.reserved_words), AMDF_STATUS_OK);
  EXPECT_EQ(native_.consumed_target, 80u);
  EXPECT_EQ(native_.calls,
            (std::vector<std::string>{"create", "map", "query", "wait", "unmap",
                                      "destroy"}));
}

TEST_F(RingStorageTest, FailedConsumptionPreservesQueueAndMapping) {
  IREE_ASSERT_OK(Initialize());
  native_.fail_operation = "wait";
  native_.failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(Deinitialize(80), native_.failure);
  EXPECT_TRUE(native_.queue_live);
  EXPECT_TRUE(native_.mapping_live);
  EXPECT_NE(storage_.queue, nullptr);
  EXPECT_NE(storage_.mapping, nullptr);
  EXPECT_EQ(native_.calls.back(), "wait");
  native_.fail_operation.clear();
  EXPECT_EQ(Deinitialize(80), AMDF_STATUS_OK);
}

TEST_F(RingStorageTest, FailedUnmapPreservesQueueAndMapping) {
  IREE_ASSERT_OK(Initialize());
  native_.fail_operation = "unmap";
  EXPECT_EQ(Deinitialize(), native_.failure);
  EXPECT_TRUE(native_.queue_live);
  EXPECT_TRUE(native_.mapping_live);
  native_.fail_operation.clear();
  EXPECT_EQ(Deinitialize(), AMDF_STATUS_OK);
}

TEST_F(RingStorageTest, ApiBusyPreservesQueueWithoutRepeatingSuccessfulUnmap) {
  IREE_ASSERT_OK(Initialize());
  native_.fail_operation = "destroy";
  native_.failure = amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(Deinitialize(), native_.failure);
  EXPECT_TRUE(native_.queue_live);
  EXPECT_FALSE(native_.mapping_live);
  EXPECT_NE(storage_.queue, nullptr);
  EXPECT_EQ(storage_.mapping, nullptr);
  native_.fail_operation.clear();
  EXPECT_EQ(Deinitialize(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.calls,
            (std::vector<std::string>{"create", "map", "query", "unmap",
                                      "destroy", "destroy"}));
}

TEST_F(RingStorageTest, NativeRemovalFailureCannotBecomeSuccessfulReclamation) {
  IREE_ASSERT_OK(Initialize());
  native_.fail_operation = "destroy";
  // An error with the same numeric code in a native domain is not API BUSY.
  native_.failure =
      amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(Deinitialize(), native_.failure);
  EXPECT_FALSE(native_.queue_live);
  EXPECT_FALSE(native_.mapping_live);
  EXPECT_EQ(storage_.queue, nullptr);
  EXPECT_EQ(storage_.mapping, nullptr);
  EXPECT_EQ(storage_.context, &context_);
  EXPECT_EQ(storage_.removal_status, native_.failure);
  const size_t call_count = native_.calls.size();
  native_.fail_operation.clear();
  EXPECT_EQ(Deinitialize(), native_.failure);
  EXPECT_EQ(native_.calls.size(), call_count);
}

TEST_F(RingStorageTest, FailedRollbackKeepsItsNativeOwner) {
  native_.fail_operation = "query";
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
  native_.fail_operation = "unmap";
  EXPECT_EQ(Deinitialize(), native_.failure);
  EXPECT_TRUE(native_.queue_live);
  EXPECT_TRUE(native_.mapping_live);
  native_.fail_operation.clear();
  EXPECT_EQ(Deinitialize(), AMDF_STATUS_OK);
}

}  // namespace
