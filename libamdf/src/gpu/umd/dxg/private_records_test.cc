// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/private_records.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"

namespace {

template <size_t N>
using Record = std::array<uint8_t, N>;

template <size_t N>
struct GuardedRecord {
  static constexpr uint8_t kGuardValue = 0xA5;

  GuardedRecord() { bytes.fill(kGuardValue); }

  void* data() { return bytes.data() + 8; }

  Record<N> payload() const {
    Record<N> result;
    std::memcpy(result.data(), bytes.data() + 8, N);
    return result;
  }

  void ExpectGuardsIntact() const {
    for (size_t i = 0; i < 8; ++i) {
      EXPECT_EQ(bytes[i], kGuardValue);
      EXPECT_EQ(bytes[8 + N + i], kGuardValue);
    }
  }

  std::array<uint8_t, N + 16> bytes;
};

template <size_t N>
void Store32(Record<N>& record, size_t offset, uint32_t value) {
  std::memcpy(record.data() + offset, &value, sizeof(value));
}

template <size_t N>
void Store64(Record<N>& record, size_t offset, uint64_t value) {
  std::memcpy(record.data() + offset, &value, sizeof(value));
}

void Store16(void* data, size_t offset, uint16_t value) {
  std::memcpy(static_cast<uint8_t*>(data) + offset, &value, sizeof(value));
}

void Store32(void* data, size_t offset, uint32_t value) {
  std::memcpy(static_cast<uint8_t*>(data) + offset, &value, sizeof(value));
}

void Store64(void* data, size_t offset, uint64_t value) {
  std::memcpy(static_cast<uint8_t*>(data) + offset, &value, sizeof(value));
}

Record<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH>
MakeCommonAllocationRecord(uint64_t byte_length) {
  Record<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> expected = {};
  Store32(expected, 0x00, 0x1D8);
  Store64(expected, 0x3C, UINT64_C(0x1D800000001));
  Store64(expected, 0x44, UINT64_C(0x8000000001));
  Store32(expected, 0x54, static_cast<uint32_t>(byte_length));
  Store32(expected, 0x58, 0x1000);
  Store64(expected, 0xA8, byte_length);
  Store32(expected, 0xF0, 0xC4);
  Store32(expected, 0x104, 0x114);
  return expected;
}

TEST(PrivateRecordsTest, AbiLayoutsAreFixed) {
  static_assert(sizeof(amdf_dxg_private_adapter_query_t) == 32);
  static_assert(sizeof(amdf_dxg_private_adapter_info_t) == 120);
  static_assert(sizeof(amdf_dxg_private_queue_profile_t) == 16);
  static_assert(sizeof(amdf_dxg_private_allocation_info_t) == 16);
  static_assert(sizeof(amdf_dxg_private_allocation_metadata_t) == 32);
  EXPECT_EQ(AMDF_DXG_PRIVATE_LINKED_QUERY_SCRATCH_BYTE_LENGTH, 0xF160u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_SINGLE_QUERY_SCRATCH_BYTE_LENGTH, 0x3A00u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH, 0x40u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH, 0x218u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_PROXY_RESOURCE_RECORD_BYTE_LENGTH, 0x40u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH, 0x1A8u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH, 0x40u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH, 0x48u);
}

TEST(PrivateRecordsTest, InitializesAllocationDriverRecordExactly) {
  GuardedRecord<AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH> record;
  amdf_dxg_private_initialize_allocation_driver_record(record.data());

  Record<AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH> expected = {};
  expected[0x08] = 0x80;
  Store32(expected, 0x3C, AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
  EXPECT_EQ(record.payload(), expected);
  record.ExpectGuardsIntact();
}

TEST(PrivateRecordsTest, InitializesWslSystemAndRegisteredRecordsExactly) {
  constexpr uint64_t kByteLength = UINT64_C(0x123456789);
  struct TestCase {
    amdf_dxg_private_allocation_domain_t domain;
    int32_t gfx_ip_major;
    amdf_dxg_private_allocation_flags_t flags;
    bool expects_uncached_mapping;
  };
  const TestCase test_cases[] = {
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM, 11, 0, false},
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM, 11,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN, true},
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST, 12,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN, false},
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST, 12,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED, true},
  };
  for (const TestCase& test_case : test_cases) {
    GuardedRecord<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> record;
    const amdf_dxg_private_allocation_info_t info = {
        .byte_length = kByteLength,
        .domain = test_case.domain,
        .flags = test_case.flags,
    };
    const amdf_dxg_private_adapter_info_t adapter_info = {
        .gfx_ip_major = test_case.gfx_ip_major,
        .is_discrete = 1,
    };
    amdf_dxg_private_initialize_wsl_allocation_record(&info, &adapter_info,
                                                      record.data());

    auto expected = MakeCommonAllocationRecord(kByteLength);
    Store32(expected, 0x50, 0x04000008);
    expected[0x5C] = 1;
    expected[0x60] = 4;
    if (test_case.expects_uncached_mapping) {
      Store32(expected, 0x1D0, 4);
    }
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
  }
}

TEST(PrivateRecordsTest, InitializesEveryWslLocalMemorySegmentShapeExactly) {
  constexpr uint64_t kByteLength = UINT64_C(0x123456789);
  struct TestCase {
    uint32_t is_discrete;
    uint64_t invisible_byte_length;
    uint8_t segment_flags;
    uint8_t segment_count;
    std::array<uint8_t, 3> segment_types;
  };
  const TestCase test_cases[] = {
      {1, 0, 1, 1, {1, 0, 0}},
      {1, 0x100000, 3, 2, {1, 2, 0}},
      {0, 0, 9, 2, {1, 4, 0}},
      {0, 0x100000, 11, 3, {1, 2, 4}},
  };
  for (const TestCase& test_case : test_cases) {
    GuardedRecord<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> record;
    const amdf_dxg_private_allocation_info_t info = {
        .byte_length = kByteLength,
        .domain = AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL,
        .flags = 0,
    };
    const amdf_dxg_private_adapter_info_t adapter_info = {
        .gfx_ip_major = 11,
        .is_discrete = test_case.is_discrete,
        .invisible_local_heap_byte_length = test_case.invisible_byte_length,
    };
    amdf_dxg_private_initialize_wsl_allocation_record(&info, &adapter_info,
                                                      record.data());

    auto expected = MakeCommonAllocationRecord(kByteLength);
    expected[0x50] = test_case.segment_flags;
    expected[0x5C] = test_case.segment_count;
    std::memcpy(expected.data() + 0x60, test_case.segment_types.data(),
                test_case.segment_types.size());
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
  }
}

TEST(PrivateRecordsTest, InitializesWddmSystemAndRegisteredRecordsExactly) {
  constexpr uint64_t kByteLength = UINT64_C(0x123456789);
  struct TestCase {
    amdf_dxg_private_allocation_domain_t domain;
    int32_t gfx_ip_major;
    amdf_dxg_private_allocation_flags_t flags;
    uint8_t native_queue_flags;
    uint32_t expected_segment_flags;
    uint8_t expected_queue_byte;
    uint8_t expected_cache_policy;
  };
  const TestCase test_cases[] = {
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM, 11,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN, 0, 0x24002008, 0, 4},
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST, 12,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN, 0, 0x04000008, 0, 0},
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST, 12,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED |
           AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE,
       AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL, 0x04000008, 1, 4},
      {AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_REGISTERED_HOST, 12,
       AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE, 0, 0x04000008, 0, 0},
  };
  for (const TestCase& test_case : test_cases) {
    GuardedRecord<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> record;
    const amdf_dxg_private_allocation_info_t info = {
        .byte_length = kByteLength,
        .domain = test_case.domain,
        .flags = test_case.flags,
    };
    const amdf_dxg_private_adapter_info_t adapter_info = {
        .gfx_ip_major = test_case.gfx_ip_major,
        .native_queue_flags = test_case.native_queue_flags,
    };
    amdf_dxg_private_initialize_wddm_allocation_record(&info, &adapter_info,
                                                       record.data());

    auto expected = MakeCommonAllocationRecord(kByteLength);
    Store32(expected, 0x50, test_case.expected_segment_flags);
    expected[0x5C] = 1;
    expected[0x60] = 4;
    expected[0xA6] = test_case.expected_queue_byte;
    if (test_case.domain == AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM) {
      expected[0xA7] = 1;
    }
    if (test_case.expected_queue_byte != 0) {
      expected[0xC0] = 0x10;
    }
    expected[0x1D0] = test_case.expected_cache_policy;
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
  }
}

TEST(PrivateRecordsTest, InitializesEveryWddmLocalMemorySegmentShapeExactly) {
  constexpr uint64_t kByteLength = UINT64_C(0x123456789);
  struct TestCase {
    uint64_t invisible_byte_length;
    uint8_t segment_flags;
    uint8_t segment_count;
    std::array<uint8_t, 3> segment_types;
  };
  const TestCase test_cases[] = {
      {0, 5, 2, {1, 3, 0}},
      {0x100000, 7, 3, {2, 1, 3}},
  };
  for (const TestCase& test_case : test_cases) {
    GuardedRecord<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> record;
    const amdf_dxg_private_allocation_info_t info = {
        .byte_length = kByteLength,
        .domain = AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL,
        .flags = AMDF_DXG_PRIVATE_ALLOCATION_FLAG_QUEUE_STORAGE,
    };
    const amdf_dxg_private_adapter_info_t adapter_info = {
        .gfx_ip_major = 11,
        .native_queue_flags = AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL,
        .invisible_local_heap_byte_length = test_case.invisible_byte_length,
    };
    amdf_dxg_private_initialize_wddm_allocation_record(&info, &adapter_info,
                                                       record.data());

    auto expected = MakeCommonAllocationRecord(kByteLength);
    expected[0x50] = test_case.segment_flags;
    expected[0x5C] = test_case.segment_count;
    std::memcpy(expected.data() + 0x60, test_case.segment_types.data(),
                test_case.segment_types.size());
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
  }
}

TEST(PrivateRecordsTest, DecodesReturnedAllocationMetadataExactly) {
  Record<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> record = {};
  Store32(record, 0x40, 0xC5);
  Store64(record, 0xA8, UINT64_C(0x123456789ABC));
  Store32(record, 0x104, 4);
  Store32(record, 0x10C, 32);
  Store32(record, 0x140, 0x1234);
  record[0x1B8] = 0x2C;
  Store32(record, 0x1D4, 7);

  amdf_dxg_private_allocation_metadata_t metadata = {};
  ASSERT_TRUE(amdf_dxg_private_decode_allocation_metadata(
      record.data(), record.size(), &metadata));
  EXPECT_EQ(metadata.byte_length, UINT64_C(0x123456789ABC));
  EXPECT_EQ(metadata.swizzle_mode, 32u);
  EXPECT_EQ(metadata.tile_swizzle, 0x1234u);
  EXPECT_EQ(metadata.compression_mode, 7u);
  EXPECT_EQ(metadata.maximum_compressed_block_size, 2u);
  EXPECT_EQ(metadata.maximum_uncompressed_block_size, 3u);
}

TEST(PrivateRecordsTest, RejectsIncompleteAllocationMetadataWithoutPublishing) {
  Record<AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH> record = {};
  struct TestCase {
    size_t byte_length;
    uint32_t surface_size;
    uint32_t layout_version;
  };
  const TestCase test_cases[] = {
      {0x1D7, 0xC5, 4},
      {record.size(), 0xC4, 4},
      {record.size(), 0xC5, 3},
  };
  for (const TestCase& test_case : test_cases) {
    Store32(record, 0x40, test_case.surface_size);
    Store32(record, 0x104, test_case.layout_version);
    amdf_dxg_private_allocation_metadata_t metadata;
    std::memset(&metadata, 0xA5, sizeof(metadata));
    const auto original = metadata;
    EXPECT_FALSE(amdf_dxg_private_decode_allocation_metadata(
        record.data(), test_case.byte_length, &metadata));
    EXPECT_EQ(std::memcmp(&metadata, &original, sizeof(metadata)), 0);
  }
}

TEST(PrivateRecordsTest, InitializesContextRecordExactly) {
  GuardedRecord<AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH> record;
  amdf_dxg_private_initialize_context_record(
      /*supports_firmware_state_shadowing=*/1, /*scheduler_id=*/5,
      UINT64_C(0x123456789ABCDEF0), record.data());

  Record<AMDF_DXG_PRIVATE_CONTEXT_RECORD_BYTE_LENGTH> expected = {};
  Store64(expected, 0x00, UINT64_C(0x0010010000000040));
  Store32(expected, 0x38, 0x00200005);
  Store32(expected, 0x40, 0x168);
  Store64(expected, 0x150, UINT64_C(0x123456789ABCDEF0));
  EXPECT_EQ(record.payload(), expected);
  record.ExpectGuardsIntact();
}

TEST(PrivateRecordsTest, InitializesEveryPm4QueuePriorityExactly) {
  const uint32_t expected_priority_bits[] = {0x1000, 0x1000, 0x2000};
  for (uint32_t priority = AMDF_DXG_PRIVATE_QUEUE_PRIORITY_LOW;
       priority <= AMDF_DXG_PRIVATE_QUEUE_PRIORITY_HIGH; ++priority) {
    GuardedRecord<AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH> record;
    amdf_dxg_private_initialize_hardware_queue_record(
        /*supports_firmware_state_shadowing=*/1, priority, 0xDEF0,
        record.data());

    Record<AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH> expected = {};
    Store32(expected, 0x00, AMDF_DXG_PRIVATE_HARDWARE_QUEUE_RECORD_BYTE_LENGTH);
    Store32(expected, 0x10, expected_priority_bits[priority] | 0x20000);
    Store32(expected, 0x30, 0xDEF0);
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
  }
}

TEST(PrivateRecordsTest, InitializesHardwareAndSoftwareSubmissionsExactly) {
  for (uint32_t uses_hardware_queue : {0u, 1u}) {
    GuardedRecord<AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH> record;
    amdf_dxg_private_initialize_submission_record(
        /*queue_handle=*/0x34, UINT64_C(0x123456789ABCDEF0), 0x87654321,
        uses_hardware_queue, record.data());

    Record<AMDF_DXG_PRIVATE_SUBMISSION_RECORD_BYTE_LENGTH> expected = {};
    Store64(expected, 0x00, UINT64_C(0x100000006));
    Store64(expected, 0x08, UINT64_C(1) << 38);
    Store32(expected, 0x10, 0x87654321);
    expected[0x14] = 4;
    expected[0x1C] = 1;
    Store64(expected, 0x20, UINT64_C(0x123456789ABCDEF0));
    if (!uses_hardware_queue) {
      Store32(expected, 0x28, 0x34);
    }
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
  }
}

TEST(PrivateRecordsTest, DerivesComputeAndHardwareSdmaQueueProfiles) {
  amdf_dxg_private_adapter_info_t info = {};
  info.pm4_scheduler_id = 6;
  info.scheduler_ids[0] = 4;
  info.scheduler_ids[1] = 5;
  info.scheduler_ids[2] = 6;
  info.scheduler_ids[3] = 7;
  info.hardware_scheduling_node_mask = (1u << 2) | (1u << 3);

  amdf_dxg_private_queue_profile_t pm4_profile = {};
  amdf_dxg_private_queue_profile_t sdma_profile = {};
  amdf_dxg_private_initialize_queue_profiles(&info, &pm4_profile,
                                             &sdma_profile);
  EXPECT_TRUE(pm4_profile.is_supported);
  EXPECT_EQ(pm4_profile.scheduler_id, 6u);
  EXPECT_EQ(pm4_profile.node_ordinal, 2u);
  EXPECT_TRUE(pm4_profile.uses_hardware_scheduling);
  EXPECT_TRUE(sdma_profile.is_supported);
  EXPECT_EQ(sdma_profile.scheduler_id, 7u);
  EXPECT_EQ(sdma_profile.node_ordinal, 3u);
  EXPECT_TRUE(sdma_profile.uses_hardware_scheduling);
}

TEST(PrivateRecordsTest, AllowsSoftwareSdmaOnlyWithoutPersistentUserQueues) {
  amdf_dxg_private_adapter_info_t info = {};
  info.pm4_scheduler_id = 5;
  info.scheduler_ids[0] = 5;
  info.scheduler_ids[1] = 4;
  info.hardware_scheduling_node_mask = 1u << 0;

  amdf_dxg_private_queue_profile_t pm4_profile = {};
  amdf_dxg_private_queue_profile_t sdma_profile = {};
  amdf_dxg_private_initialize_queue_profiles(&info, &pm4_profile,
                                             &sdma_profile);
  EXPECT_TRUE(pm4_profile.is_supported);
  EXPECT_TRUE(sdma_profile.is_supported);
  EXPECT_EQ(sdma_profile.scheduler_id, 4u);
  EXPECT_FALSE(sdma_profile.uses_hardware_scheduling);

  info.user_queue_byte_length = 4096;
  amdf_dxg_private_initialize_queue_profiles(&info, &pm4_profile,
                                             &sdma_profile);
  EXPECT_TRUE(pm4_profile.is_supported);
  EXPECT_FALSE(sdma_profile.is_supported);
}

TEST(PrivateRecordsTest, DoesNotSubstituteAnUnrelatedComputeScheduler) {
  amdf_dxg_private_adapter_info_t info = {};
  info.pm4_scheduler_id = 5;
  info.scheduler_ids[0] = 0;
  info.hardware_scheduling_node_mask = 1;
  amdf_dxg_private_queue_profile_t pm4_profile = {};
  amdf_dxg_private_queue_profile_t sdma_profile = {};
  amdf_dxg_private_initialize_queue_profiles(&info, &pm4_profile,
                                             &sdma_profile);
  EXPECT_FALSE(pm4_profile.is_supported);
}

struct FakePrivateQuery {
  std::array<uint32_t, 4> byte_lengths = {};
  uint32_t call_count = 0;
  uint32_t failure_call = UINT32_MAX;
  uint32_t failure_status = 0;
  uint32_t physical_adapter_count = 2;
  uint32_t family = 0x96;
  uint32_t asic_revision = 0x10;
  uint32_t encoded_gfxip = 0x000B0500;
  uint16_t vendor_id = 0x1002;
  uint8_t native_queue_bits = (1u << 2) | (1u << 5);
  bool supports_wddm2 = true;
  bool corrupt_checksum = false;
  bool valid_compute_unit_mask = true;
  bool include_last_compute_unit_mask = false;
};

uint32_t HeaderChecksum(const uint8_t* header) {
  uint32_t checksum = 0;
  for (uint32_t i = 0; i < 0x960 / sizeof(uint32_t); ++i) {
    uint32_t word = 0;
    std::memcpy(&word, header + i * sizeof(uint32_t), sizeof(word));
    if (i * sizeof(uint32_t) == 0x8AC) {
      word = 0;
    }
    checksum += word ^ i;
  }
  return checksum;
}

void FillBaseRecord(uint8_t* record, uint32_t family) {
  Store16(record, 0x2C, 0x150E);
  Store32(record, 0x30, family);
  record[0x1F4] = 0x80;
}

void FillBaseRecord(uint8_t* record, const FakePrivateQuery& fake) {
  FillBaseRecord(record, fake.family);
  record[0x1F4] = fake.supports_wddm2 ? 0x80 : 0;
}

void FillExtendedRecord(uint8_t* record, const FakePrivateQuery& fake) {
  record[0x208] = 0x20;
  record[0xC46] = fake.valid_compute_unit_mask ? 0x21 : 0x20;
  Store32(record, 0xA08, fake.asic_revision);
  Store32(record, 0xAF4, fake.encoded_gfxip);
  Store32(record, 0xC5C, 32);
  Store32(record, 0xC60, 2);
  Store64(record, 0xC64, UINT64_C(2) << 32);
  Store32(record, 0xC68, 2);
  Store32(record, 0xC6C, 3);
  Store64(record, 0xA38, 0x12345000);
  Store16(record, 0xD04, 0x000F);
  Store16(record, 0xD06, 0x0003);
  if (fake.include_last_compute_unit_mask) {
    Store16(record, 0xD08, 1);
  }
  std::memset(record + 0x89C, 0xFF, AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT);
  record[0x89C] = 0;
  record[0x89D] = 4;
  record[0x89E] = 5;
  record[0x89F] = 7;
  record[0x2E03] = 1;
  record[0x35C0] = fake.native_queue_bits;
  Store64(record, 0x35C4, (UINT64_C(1) << 1) | (UINT64_C(1) << 2));
}

uint32_t FakeQuery(void* context, uint32_t adapter_handle, void* buffer,
                   uint32_t buffer_byte_length) {
  auto* fake = static_cast<FakePrivateQuery*>(context);
  const uint32_t call = fake->call_count++;
  if (call < fake->byte_lengths.size()) {
    fake->byte_lengths[call] = buffer_byte_length;
  }
  if (call == fake->failure_call) {
    return fake->failure_status;
  }
  EXPECT_EQ(adapter_handle, 0x42u);

  auto* bytes = static_cast<uint8_t*>(buffer);
  if (buffer_byte_length == 0x960) {
    Store32(bytes, 0x2C, 0);
    return 0;
  }
  if (buffer_byte_length == 0x260) {
    FillBaseRecord(bytes, *fake);
    return 0;
  }
  if (buffer_byte_length == 0x3A00) {
    FillExtendedRecord(bytes, *fake);
    return 0;
  }

  Store32(bytes, 0x08, fake->physical_adapter_count);
  Store16(bytes, 0x0C, fake->vendor_id);
  Store16(bytes, 0x0E, fake->vendor_id);
  Store32(bytes, 0x2C, 0);
  Store32(bytes, 0x30, 1);
  if (buffer_byte_length >= 0xBC0) {
    FillBaseRecord(bytes + 0x960, 0x91);
  }
  if (buffer_byte_length >= 0xE20) {
    FillBaseRecord(bytes + 0x960 + 0x260, *fake);
  }
  if (buffer_byte_length >= 0x7D60) {
    FillExtendedRecord(bytes + 0x960, *fake);
    FillExtendedRecord(bytes + 0x960 + 0x3A00, *fake);
  }

  uint32_t checksum = HeaderChecksum(bytes);
  if (fake->corrupt_checksum) {
    ++checksum;
  }
  Store32(bytes, 0x8AC, checksum);
  return 0;
}

struct AdapterQueryFixture {
  AdapterQueryFixture() {
    scratch.fill(0);
    query = {
        .adapter_handle = 0x42,
        .physical_adapter_index = 1,
        .query_context = &fake,
        .query = FakeQuery,
        .scratch = scratch.data(),
    };
  }

  FakePrivateQuery fake;
  Record<AMDF_DXG_PRIVATE_LINKED_QUERY_SCRATCH_BYTE_LENGTH> scratch;
  amdf_dxg_private_adapter_query_t query;
};

TEST(PrivateRecordsTest, QueriesSelectedPhysicalAdapterExactly) {
  AdapterQueryFixture fixture;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_SUCCESS);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(fixture.fake.call_count, 3u);
  EXPECT_EQ(fixture.fake.byte_lengths,
            (std::array<uint32_t, 4>{0xBC0, 0xE20, 0x7D60, 0}));
  EXPECT_EQ(info.asic_revision, 0x10u);
  EXPECT_EQ(info.gfx_ip_major, 11);
  EXPECT_EQ(info.gfx_ip_minor, 5);
  EXPECT_EQ(info.gfx_ip_stepping, 0);
  EXPECT_EQ(info.is_discrete, 0u);
  EXPECT_EQ(info.wavefront_size, 32u);
  EXPECT_EQ(info.compute_unit_count, 12u);
  EXPECT_EQ(info.maximum_wave_count_per_compute_unit, 32u);
  EXPECT_EQ(info.maximum_scratch_wave_count_per_compute_unit, 128u);
  EXPECT_EQ(info.local_data_share_byte_length, 64u * 1024);
  EXPECT_EQ(info.shader_engine_count, 2u);
  EXPECT_EQ(info.xcc_count, 1u);
  EXPECT_EQ(info.supports_firmware_state_shadowing, 1u);
  EXPECT_EQ(info.supports_platform_atomics, 1u);
  EXPECT_EQ(info.user_queue_byte_length, 0u);
  EXPECT_EQ(info.pm4_scheduler_id, 5u);
  EXPECT_EQ(info.native_queue_flags, AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL |
                                         AMDF_DXG_PRIVATE_NATIVE_QUEUE_SDMA);
  EXPECT_EQ(info.invisible_local_heap_byte_length, 0x12345000u);
  EXPECT_EQ(info.scheduler_ids[0], 0u);
  EXPECT_EQ(info.scheduler_ids[1], 4u);
  EXPECT_EQ(info.scheduler_ids[2], 5u);
  EXPECT_EQ(info.scheduler_ids[3], 7u);
  EXPECT_EQ(info.hardware_scheduling_node_mask,
            (UINT64_C(1) << 1) | (UINT64_C(1) << 2));
}

TEST(PrivateRecordsTest, ReusesInitialBaseRecordForFirstPhysicalAdapter) {
  AdapterQueryFixture fixture;
  fixture.query.physical_adapter_index = 0;
  fixture.fake.encoded_gfxip = 0x000B0000;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_SUCCESS);
  EXPECT_EQ(fixture.fake.call_count, 2u);
  EXPECT_EQ(fixture.fake.byte_lengths,
            (std::array<uint32_t, 4>{0xBC0, 0x7D60, 0, 0}));
}

TEST(PrivateRecordsTest, QueriesSingleAdapterRecordsExactly) {
  AdapterQueryFixture fixture;
  fixture.query.physical_adapter_index = 0;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  ASSERT_EQ(amdf_dxg_private_query_single_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_SUCCESS);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(fixture.fake.call_count, 3u);
  EXPECT_EQ(fixture.fake.byte_lengths,
            (std::array<uint32_t, 4>{0x960, 0x260, 0x3A00, 0}));
  EXPECT_EQ(info.gfx_ip_major, 11);
  EXPECT_EQ(info.gfx_ip_minor, 5);
  EXPECT_EQ(info.asic_revision, 0x10u);
}

TEST(PrivateRecordsTest, DecodesDriverSelectedComputeScheduler) {
  AdapterQueryFixture fixture;
  fixture.fake.native_queue_bits = (1u << 2) | (1u << 4);
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_SUCCESS);
  EXPECT_EQ(info.pm4_scheduler_id, 6u);
  EXPECT_EQ(info.native_queue_flags, AMDF_DXG_PRIVATE_NATIVE_QUEUE_AQL);
}

TEST(PrivateRecordsTest, DecodesEveryComputeUnitLayout) {
  struct TestCase {
    uint32_t encoded_gfxip;
    uint32_t expected_compute_unit_count;
    uint32_t expected_wave_count;
    uint32_t expected_scratch_wave_count;
  };
  const TestCase test_cases[] = {
      {0x00090000, 12, 40, 64},
      {0x000A0300, 14, 32, 128},
      {0x000B0500, 14, 32, 128},
      {0x000C0000, 14, 32, 256},
  };
  for (const TestCase& test_case : test_cases) {
    AdapterQueryFixture fixture;
    fixture.fake.encoded_gfxip = test_case.encoded_gfxip;
    fixture.fake.include_last_compute_unit_mask = true;
    amdf_dxg_private_adapter_info_t info = {};
    uint32_t native_status = UINT32_MAX;
    ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                    &native_status),
              AMDF_DXG_PRIVATE_RESULT_SUCCESS);
    EXPECT_EQ(native_status, 0u);
    EXPECT_EQ(info.compute_unit_count, test_case.expected_compute_unit_count);
    EXPECT_EQ(info.maximum_wave_count_per_compute_unit,
              test_case.expected_wave_count);
    EXPECT_EQ(info.maximum_scratch_wave_count_per_compute_unit,
              test_case.expected_scratch_wave_count);
  }
}

TEST(PrivateRecordsTest, ExpandsGfx94TopologyAcrossEightXccs) {
  AdapterQueryFixture fixture;
  fixture.fake.encoded_gfxip = 0x00090400;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_SUCCESS);
  EXPECT_EQ(info.compute_unit_count, 96u);
  EXPECT_EQ(info.shader_engine_count, 16u);
  EXPECT_EQ(info.xcc_count, 8u);
}

TEST(PrivateRecordsTest, DecodesFallbackGfxipFamiliesAtRangeBoundaries) {
  struct TestCase {
    uint32_t family;
    uint32_t revision;
    int32_t major;
    int32_t minor;
    int32_t stepping;
  };
  const TestCase test_cases[] = {
      {0x8F, 9, 10, 1, 0},   {0x8F, 10, 10, 1, 1},  {0x8F, 20, 10, 1, 2},
      {0x8F, 40, 10, 3, 0},  {0x8F, 50, 10, 3, 1},  {0x8F, 60, 10, 3, 2},
      {0x8F, 70, 10, 3, 3},  {0x8F, 192, 10, 3, 0}, {0x8F, 200, 10, 3, 1},
      {0x90, 1, 10, 3, 1},   {0x92, 254, 10, 3, 1}, {0x95, 31, 10, 3, 1},
      {0x95, 32, 10, 3, 6},  {0x97, 1, 10, 3, 2},   {0x91, 15, 11, 0, 0},
      {0x91, 16, 11, 0, 2},  {0x91, 32, 11, 0, 1},  {0x94, 1, 11, 0, 1},
      {0x96, 25, 11, 5, 0},  {0x96, 64, 11, 5, 0},  {0x96, 80, 11, 5, 0},
      {0x96, 192, 11, 5, 1}, {0x98, 31, 12, 0, 0},  {0x98, 32, 12, 0, 1},
      {0x98, 64, 12, 0, 0},  {0x98, 80, 12, 0, 1},
  };
  for (const TestCase& test_case : test_cases) {
    AdapterQueryFixture fixture;
    fixture.fake.family = test_case.family;
    fixture.fake.asic_revision = test_case.revision;
    fixture.fake.encoded_gfxip = 0;
    amdf_dxg_private_adapter_info_t info = {};
    uint32_t native_status = UINT32_MAX;
    ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                    &native_status),
              AMDF_DXG_PRIVATE_RESULT_SUCCESS)
        << "family=" << test_case.family << " revision=" << test_case.revision;
    EXPECT_EQ(info.gfx_ip_major, test_case.major);
    EXPECT_EQ(info.gfx_ip_minor, test_case.minor);
    EXPECT_EQ(info.gfx_ip_stepping, test_case.stepping);
  }
}

TEST(PrivateRecordsTest, NormalizesPackedGfxipQuirks) {
  struct TestCase {
    uint32_t encoded_gfxip;
    int32_t major;
    int32_t minor;
    int32_t stepping;
  };
  const TestCase test_cases[] = {
      {0x000B0503, 11, 5, 2},
      {0x000D0001, 13, 1, 0},
  };
  for (const TestCase& test_case : test_cases) {
    AdapterQueryFixture fixture;
    fixture.fake.encoded_gfxip = test_case.encoded_gfxip;
    amdf_dxg_private_adapter_info_t info = {};
    uint32_t native_status = UINT32_MAX;
    ASSERT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                    &native_status),
              AMDF_DXG_PRIVATE_RESULT_SUCCESS);
    EXPECT_EQ(info.gfx_ip_major, test_case.major);
    EXPECT_EQ(info.gfx_ip_minor, test_case.minor);
    EXPECT_EQ(info.gfx_ip_stepping, test_case.stepping);
  }
}

TEST(PrivateRecordsTest, RejectsUnsupportedAdapterIdentitiesWithoutPublishing) {
  struct TestCase {
    uint32_t family;
    uint32_t encoded_gfxip;
    uint16_t vendor_id;
    bool supports_wddm2;
  };
  const TestCase test_cases[] = {
      {0xDEAD, 0, 0x1002, true},
      {0x96, 0x000E0000, 0x1002, true},
      {0x96, 0x000B0500, 0x1234, true},
      {0x96, 0x000B0500, 0x1002, false},
  };
  for (const TestCase& test_case : test_cases) {
    AdapterQueryFixture fixture;
    fixture.fake.family = test_case.family;
    fixture.fake.encoded_gfxip = test_case.encoded_gfxip;
    fixture.fake.vendor_id = test_case.vendor_id;
    fixture.fake.supports_wddm2 = test_case.supports_wddm2;
    amdf_dxg_private_adapter_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const amdf_dxg_private_adapter_info_t original = info;
    uint32_t native_status = UINT32_MAX;
    EXPECT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                    &native_status),
              AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED);
    EXPECT_EQ(native_status, 0u);
    EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
  }
}

TEST(PrivateRecordsTest, PreservesNativeFailureAndOutput) {
  AdapterQueryFixture fixture;
  fixture.fake.failure_call = 1;
  fixture.fake.failure_status = 0xC0000001u;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = 0;
  EXPECT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(native_status, 0xC0000001u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(PrivateRecordsTest, RejectsBadChecksumWithoutPublishing) {
  AdapterQueryFixture fixture;
  fixture.fake.corrupt_checksum = true;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(PrivateRecordsTest, RejectsInvalidComputeUnitMaskWithoutPublishing) {
  AdapterQueryFixture fixture;
  fixture.fake.valid_compute_unit_mask = false;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(PrivateRecordsTest, RejectsUnsupportedPhysicalAdapterWithoutPublishing) {
  AdapterQueryFixture fixture;
  fixture.query.physical_adapter_index = 2;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(amdf_dxg_private_query_linked_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(PrivateRecordsTest, RejectsIndexedSingleAdapterWithoutQuerying) {
  AdapterQueryFixture fixture;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(amdf_dxg_private_query_single_adapter(&fixture.query, &info,
                                                  &native_status),
            AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(fixture.fake.call_count, 0u);
}

}  // namespace
