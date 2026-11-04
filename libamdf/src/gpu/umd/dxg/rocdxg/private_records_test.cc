// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/private_records.h"

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
  static_assert(sizeof(amdf_dxg_private_gfxip_entry_t) == 16);
  static_assert(sizeof(amdf_dxg_private_adapter_query_t) == 56);
  static_assert(sizeof(amdf_dxg_private_adapter_info_t) == 128);
  static_assert(sizeof(amdf_dxg_private_allocation_info_t) == 32);
  EXPECT_EQ(AMDF_DXG_PRIVATE_QUERY_SCRATCH_BYTE_LENGTH, 0xF160u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH, 0x40u);
  EXPECT_EQ(AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH, 0x218u);
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

TEST(PrivateRecordsTest, InitializesSystemAndRegisteredRecordsExactly) {
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
        .invisible_local_heap_byte_length = 0,
        .gfx_ip_major = test_case.gfx_ip_major,
        .domain = test_case.domain,
        .flags = test_case.flags,
        .is_discrete = 1,
    };
    amdf_dxg_private_initialize_allocation_record(&info, record.data());

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

TEST(PrivateRecordsTest, InitializesEveryLocalMemorySegmentShapeExactly) {
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
        .invisible_local_heap_byte_length = test_case.invisible_byte_length,
        .gfx_ip_major = 11,
        .domain = AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL,
        .flags = 0,
        .is_discrete = test_case.is_discrete,
    };
    amdf_dxg_private_initialize_allocation_record(&info, record.data());

    auto expected = MakeCommonAllocationRecord(kByteLength);
    expected[0x50] = test_case.segment_flags;
    expected[0x5C] = test_case.segment_count;
    std::memcpy(expected.data() + 0x60, test_case.segment_types.data(),
                test_case.segment_types.size());
    EXPECT_EQ(record.payload(), expected);
    record.ExpectGuardsIntact();
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

struct FakePrivateQuery {
  std::array<uint32_t, 4> byte_lengths = {};
  uint32_t call_count = 0;
  uint32_t failure_call = UINT32_MAX;
  uint32_t failure_status = 0;
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
  Store32(bytes, 0x08, 2);
  Store16(bytes, 0x0C, 0x1002);
  Store16(bytes, 0x0E, 0x1002);
  Store32(bytes, 0x2C, 0);
  Store32(bytes, 0x30, 1);

  if (call == 2) {
    auto* first = bytes + 0x960;
    auto* second = first + 0x260;
    uint16_t first_device = 0x7448;
    uint16_t second_device = 0x150E;
    uint32_t second_revision = 0x96;
    std::memcpy(first + 0x2C, &first_device, sizeof(first_device));
    std::memcpy(second + 0x2C, &second_device, sizeof(second_device));
    std::memcpy(second + 0x30, &second_revision, sizeof(second_revision));
    second[0x1F4] = 0x80;
  } else if (call == 3) {
    auto* second = bytes + 0x960 + 0x3A00;
    second[0xC46] = fake->valid_compute_unit_mask ? 0x21 : 0x20;
    uint32_t wavefront_size = 32;
    uint32_t shader_engine_count = 2;
    uint64_t shader_array_count = UINT64_C(2) << 32;
    uint32_t compute_units_per_shader_array = 3;
    uint64_t invisible_heap_byte_length = 0x12345000;
    uint16_t compute_unit_masks[] = {0x000F, 0x0003};
    std::memcpy(second + 0xC5C, &wavefront_size, sizeof(wavefront_size));
    std::memcpy(second + 0xC60, &shader_engine_count,
                sizeof(shader_engine_count));
    std::memcpy(second + 0xC64, &shader_array_count,
                sizeof(shader_array_count));
    std::memcpy(second + 0xC6C, &compute_units_per_shader_array,
                sizeof(compute_units_per_shader_array));
    std::memcpy(second + 0xA38, &invisible_heap_byte_length,
                sizeof(invisible_heap_byte_length));
    std::memcpy(second + 0xD04, compute_unit_masks, sizeof(compute_unit_masks));
    if (fake->include_last_compute_unit_mask) {
      uint16_t last_compute_unit_mask = 1;
      std::memcpy(second + 0xD1A, &last_compute_unit_mask,
                  sizeof(last_compute_unit_mask));
    }
    std::memset(second + 0x89C, 0xFF, AMDF_DXG_PRIVATE_SCHEDULER_ID_COUNT);
    second[0x89C] = 0;
    second[0x89D] = 4;
    second[0x89E] = 5;
    second[0x89F] = 7;
    second[0x2E03] = 1;
    uint64_t hws_mask = (UINT64_C(1) << 1) | (UINT64_C(1) << 2);
    std::memcpy(second + 0x35C4, &hws_mask, sizeof(hws_mask));
  }

  uint32_t checksum = HeaderChecksum(bytes);
  if (fake->corrupt_checksum && call == 0) {
    ++checksum;
  }
  std::memcpy(bytes + 0x8AC, &checksum, sizeof(checksum));
  return 0;
}

struct AdapterQueryFixture {
  AdapterQueryFixture() {
    scratch.fill(0);
    query = {
        .structure_size = sizeof(query),
        .adapter_handle = 0x42,
        .physical_adapter_index = 1,
        .gfxip_entry_count = static_cast<uint32_t>(gfxip_entries.size()),
        .gfxip_entries = gfxip_entries.data(),
        .query_context = &fake,
        .query = FakeQuery,
        .scratch = scratch.data(),
        .scratch_byte_length = scratch.size(),
    };
  }

  FakePrivateQuery fake;
  std::array<amdf_dxg_private_gfxip_entry_t, 2> gfxip_entries = {{
      {.device_id = 0x7448,
       .reserved = 0,
       .major = 11,
       .minor = 0,
       .stepping = 0},
      {.device_id = 0x150E,
       .reserved = 0,
       .major = 11,
       .minor = 5,
       .stepping = 0},
  }};
  Record<AMDF_DXG_PRIVATE_QUERY_SCRATCH_BYTE_LENGTH> scratch;
  amdf_dxg_private_adapter_query_t query;
};

TEST(PrivateRecordsTest, QueriesSelectedPhysicalAdapterExactly) {
  AdapterQueryFixture fixture;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  ASSERT_EQ(
      amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
      AMDF_DXG_PRIVATE_RESULT_SUCCESS);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(fixture.fake.call_count, 4u);
  EXPECT_EQ(fixture.fake.byte_lengths,
            (std::array<uint32_t, 4>{0xBC0, 0xD60, 0xE20, 0x7D60}));
  EXPECT_EQ(info.physical_adapter_count, 2u);
  EXPECT_EQ(info.vendor_id, 0x1002u);
  EXPECT_EQ(info.device_id, 0x150Eu);
  EXPECT_EQ(info.asic_revision, 0x96u);
  EXPECT_EQ(info.gfx_ip_major, 11);
  EXPECT_EQ(info.gfx_ip_minor, 5);
  EXPECT_EQ(info.gfx_ip_stepping, 0);
  EXPECT_EQ(info.is_discrete, 0u);
  EXPECT_EQ(info.supports_wddm2, 1u);
  EXPECT_EQ(info.wavefront_size, 32u);
  EXPECT_EQ(info.compute_unit_count, 12u);
  EXPECT_EQ(info.maximum_wave_count_per_compute_unit, 32u);
  EXPECT_EQ(info.maximum_scratch_wave_count_per_compute_unit, 128u);
  EXPECT_EQ(info.local_data_share_byte_length, 64u * 1024);
  EXPECT_EQ(info.shader_engine_count, 2u);
  EXPECT_EQ(info.supports_firmware_state_shadowing, 1u);
  EXPECT_EQ(info.supports_platform_atomics, 1u);
  EXPECT_EQ(info.user_queue_byte_length, 0u);
  EXPECT_EQ(info.invisible_local_heap_byte_length, 0x12345000u);
  EXPECT_EQ(info.scheduler_ids[0], 0u);
  EXPECT_EQ(info.scheduler_ids[1], 4u);
  EXPECT_EQ(info.scheduler_ids[2], 5u);
  EXPECT_EQ(info.scheduler_ids[3], 7u);
  EXPECT_EQ(info.hardware_scheduling_node_mask,
            (UINT64_C(1) << 1) | (UINT64_C(1) << 2));
}

TEST(PrivateRecordsTest, DecodesEveryComputeUnitLayout) {
  struct TestCase {
    int32_t gfx_ip_major;
    uint32_t expected_compute_unit_count;
    uint32_t expected_wave_count;
    uint32_t expected_scratch_wave_count;
  };
  const TestCase test_cases[] = {
      {9, 12, 40, 64},
      {10, 12, 32, 128},
      {11, 14, 32, 128},
      {12, 14, 32, 256},
  };
  for (const TestCase& test_case : test_cases) {
    AdapterQueryFixture fixture;
    fixture.gfxip_entries[1].major = test_case.gfx_ip_major;
    fixture.fake.include_last_compute_unit_mask = true;
    amdf_dxg_private_adapter_info_t info = {};
    uint32_t native_status = UINT32_MAX;
    ASSERT_EQ(
        amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
        AMDF_DXG_PRIVATE_RESULT_SUCCESS);
    EXPECT_EQ(native_status, 0u);
    EXPECT_EQ(info.compute_unit_count, test_case.expected_compute_unit_count);
    EXPECT_EQ(info.maximum_wave_count_per_compute_unit,
              test_case.expected_wave_count);
    EXPECT_EQ(info.maximum_scratch_wave_count_per_compute_unit,
              test_case.expected_scratch_wave_count);
  }
}

TEST(PrivateRecordsTest, PreservesNativeFailureAndOutput) {
  AdapterQueryFixture fixture;
  fixture.fake.failure_call = 2;
  fixture.fake.failure_status = 0xC0000001u;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = 0;
  EXPECT_EQ(
      amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
      AMDF_DXG_PRIVATE_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(native_status, 0xC0000001u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(PrivateRecordsTest, ReportsRequiredScratchWithoutQuerying) {
  AdapterQueryFixture fixture;
  fixture.query.scratch_byte_length =
      AMDF_DXG_PRIVATE_QUERY_SCRATCH_BYTE_LENGTH - 1;
  amdf_dxg_private_adapter_info_t info = {};
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(
      amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
      AMDF_DXG_PRIVATE_RESULT_BUFFER_TOO_SMALL);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(fixture.fake.call_count, 0u);
}

TEST(PrivateRecordsTest, RejectsBadChecksumWithoutPublishing) {
  AdapterQueryFixture fixture;
  fixture.fake.corrupt_checksum = true;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(
      amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
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
  EXPECT_EQ(
      amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
      AMDF_DXG_PRIVATE_RESULT_MALFORMED_RECORD);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

TEST(PrivateRecordsTest, RejectsUnsupportedPhysicalAdapterWithoutPublishing) {
  AdapterQueryFixture fixture;
  fixture.query.gfxip_entry_count = 1;
  amdf_dxg_private_adapter_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_dxg_private_adapter_info_t original = info;
  uint32_t native_status = UINT32_MAX;
  EXPECT_EQ(
      amdf_dxg_private_query_adapter(&fixture.query, &info, &native_status),
      AMDF_DXG_PRIVATE_RESULT_UNSUPPORTED);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(std::memcmp(&info, &original, sizeof(info)), 0);
}

}  // namespace
