// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer_table.h"

#include "hrx_internal.h"

using iree_hal_streaming_deviceptr_t = uint64_t;
using iree_hal_streaming_any_ptr_t = uint64_t;
using iree_hal_streaming_buffer_table_t = hrx_buffer_table_t;

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Dummy buffer structure for testing.
//
// NOTE: this is not the layout of the real structure, but in this isolated
// test that does not even initialize a HAL device we can't use the real
// structure without partially populating it. I figure it's more confusing to
// have the partially populated structure than to have one that's totally
// different. At least we have tests!
struct iree_hal_streaming_buffer_t {
  iree_hal_streaming_deviceptr_t device_ptr;
  void* host_ptr;
  size_t size;
  // Padding to make allocation pattern impacts a bit more realistic.
  uint64_t padding[5];
};

static iree_status_t BufferTableStatus(hrx_status_t status) {
  if (hrx_status_is_ok(status)) {
    return iree_ok_status();
  }
  iree_status_code_t code = (iree_status_code_t)hrx_status_code(status);
  hrx_status_ignore(status);
  return iree_make_status(code);
}

static iree_status_t iree_hal_streaming_buffer_table_allocate(
    iree_allocator_t allocator, iree_hal_streaming_buffer_table_t** out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  *out_table = nullptr;
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*table), (void**)&table));
  hrx_buffer_table_initialize(table);
  *out_table = table;
  return iree_ok_status();
}

static void iree_hal_streaming_buffer_table_free(
    iree_hal_streaming_buffer_table_t* table) {
  if (!table) {
    return;
  }
  hrx_buffer_table_deinitialize(table);
  iree_allocator_free(iree_allocator_system(), table);
}

static iree_status_t iree_hal_streaming_buffer_table_insert(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_buffer_t* buffer) {
  return BufferTableStatus(
      hrx_buffer_table_insert(table, buffer->device_ptr, buffer->host_ptr,
                              buffer->size, (hrx_buffer_t)buffer, nullptr));
}

static iree_status_t iree_hal_streaming_buffer_table_insert_reserved(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_buffer_t* buffer) {
  return BufferTableStatus(hrx_buffer_table_insert_reserved(
      table, buffer->device_ptr, buffer->host_ptr, buffer->size,
      (hrx_buffer_t)buffer, nullptr));
}

static iree_status_t iree_hal_streaming_buffer_table_remove(
    iree_hal_streaming_buffer_table_t* table, uint64_t any_ptr) {
  return BufferTableStatus(hrx_buffer_table_remove(table, any_ptr));
}

static iree_status_t iree_hal_streaming_buffer_table_lookup(
    iree_hal_streaming_buffer_table_t* table, uint64_t any_ptr,
    iree_hal_streaming_buffer_t** out_buffer) {
  hrx_buffer_t buffer = nullptr;
  iree_status_t status = BufferTableStatus(
      hrx_buffer_table_find(table, any_ptr, &buffer, nullptr, nullptr));
  if (out_buffer) {
    *out_buffer = (iree_hal_streaming_buffer_t*)buffer;
  }
  return status;
}

static iree_status_t iree_hal_streaming_buffer_table_lookup_range(
    iree_hal_streaming_buffer_table_t* table, uint64_t any_ptr, size_t size,
    iree_hal_streaming_buffer_t** out_buffer) {
  hrx_buffer_t buffer = nullptr;
  iree_status_t status = BufferTableStatus(hrx_buffer_table_find_range(
      table, any_ptr, size, &buffer, nullptr, nullptr));
  if (out_buffer) {
    *out_buffer = (iree_hal_streaming_buffer_t*)buffer;
  }
  return status;
}
namespace iree::hal::stream {
namespace {

using ::iree::testing::status::StatusIs;

struct EntryCallbackState {
  void* expected_user_data;
  size_t expected_offset;
  int call_count;
};

struct BulkCallbackState {
  int call_count;
  int reject_call;
};

static hrx_status_t AcceptEntryCallback(const hrx_buffer_table_entry_t* entry,
                                        size_t offset, void* user_data) {
  auto* state = static_cast<EntryCallbackState*>(user_data);
  EXPECT_EQ(state->expected_user_data, entry->user_data);
  EXPECT_EQ(state->expected_offset, offset);
  ++state->call_count;
  return hrx_ok_status();
}

static hrx_status_t RejectEntryCallback(const hrx_buffer_table_entry_t* entry,
                                        size_t offset, void* user_data) {
  (void)entry;
  (void)offset;
  (void)user_data;
  return hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                         "entry is unavailable");
}

static hrx_status_t CountBulkEntryCallback(
    const hrx_buffer_table_entry_t* entry, size_t offset, void* user_data) {
  (void)entry;
  (void)offset;
  auto* state = static_cast<BulkCallbackState*>(user_data);
  ++state->call_count;
  return state->call_count == state->reject_call
             ? hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                               "entry is unavailable")
             : hrx_ok_status();
}

// Helper to create a dummy buffer with the given device pointer.
static iree_hal_streaming_buffer_t* CreateDummyBuffer(
    iree_hal_streaming_deviceptr_t device_ptr, size_t size,
    iree_allocator_t allocator, void* host_ptr = nullptr) {
  iree_hal_streaming_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(
      iree_allocator_malloc(allocator, sizeof(*buffer), (void**)&buffer));
  buffer->device_ptr = device_ptr;
  buffer->host_ptr = host_ptr;
  buffer->size = size;
  return buffer;
}

// Helper to free a dummy buffer.
static void FreeDummyBuffer(iree_hal_streaming_buffer_t* buffer,
                            iree_allocator_t allocator) {
  iree_allocator_free(allocator, buffer);
}

//===----------------------------------------------------------------------===//
// Basic operations
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, AllocateAndFree) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
  EXPECT_NE(table, nullptr);

  // Free should be safe even with an empty table.
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, FreeNull) {
  // Should be safe to free NULL.
  iree_hal_streaming_buffer_table_free(nullptr);
}

TEST(BufferTableTest, InsertSingle) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupExact) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, RemoveSingle) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL));

  // Should not be found after removal.
  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

//===----------------------------------------------------------------------===//
// Error conditions
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, DoubleInsert) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  auto* buffer2 = CreateDummyBuffer(0x100000000ULL, 8192, allocator);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));

  // Second insert with same device_ptr should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, buffer2)),
              StatusIs(StatusCode::kAlreadyExists));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
}

TEST(BufferTableTest, RejectsOverlappingRanges) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* existing =
      CreateDummyBuffer(UINT64_C(0x100000000), 0x1000, allocator,
                        reinterpret_cast<void*>(UINT64_C(0x200000000)));
  auto* interior = CreateDummyBuffer(UINT64_C(0x100000800), 0x1000, allocator);
  auto* enclosing = CreateDummyBuffer(UINT64_C(0x0FFFFFF00), 0x2000, allocator);
  auto* alias_overlap =
      CreateDummyBuffer(UINT64_C(0x300000000), 0x1000, allocator,
                        reinterpret_cast<void*>(UINT64_C(0x200000800)));

  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, existing));
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, interior)),
              StatusIs(StatusCode::kAlreadyExists));
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, enclosing)),
              StatusIs(StatusCode::kAlreadyExists));
  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_insert(table, alias_overlap)),
      StatusIs(StatusCode::kAlreadyExists));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(existing, allocator);
  FreeDummyBuffer(interior, allocator);
  FreeDummyBuffer(enclosing, allocator);
  FreeDummyBuffer(alias_overlap, allocator);
}

TEST(BufferTableTest, RejectsInvalidAllocationRanges) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* null_device = CreateDummyBuffer(0, 0x1000, allocator);
  auto* empty = CreateDummyBuffer(UINT64_C(0x100000000), 0, allocator);
  auto* device_overflow = CreateDummyBuffer(UINT64_MAX, 1, allocator);
  auto* host_overflow =
      CreateDummyBuffer(UINT64_C(0x100000000), 2, allocator,
                        reinterpret_cast<void*>(UINT64_MAX - 1));

  for (auto* buffer : {null_device, empty, device_overflow, host_overflow}) {
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, buffer)),
                StatusIs(StatusCode::kInvalidArgument));
  }

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(null_device, allocator);
  FreeDummyBuffer(empty, allocator);
  FreeDummyBuffer(device_overflow, allocator);
  FreeDummyBuffer(host_overflow, allocator);
}

TEST(BufferTableTest, IndexesOutOfOrderDeviceAndHostRanges) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* high =
      CreateDummyBuffer(UINT64_C(0x300000000), 0x1000, allocator,
                        reinterpret_cast<void*>(UINT64_C(0x400000000)));
  auto* low = CreateDummyBuffer(UINT64_C(0x100000000), 0x1000, allocator,
                                reinterpret_cast<void*>(UINT64_C(0x200000000)));
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, high));
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, low));

  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, UINT64_C(0x100000800), &found));
  EXPECT_EQ(low, found);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, UINT64_C(0x400000800), &found));
  EXPECT_EQ(high, found);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(high, allocator);
  FreeDummyBuffer(low, allocator);
}

TEST(BufferTableTest, RemovalRepairsMovedEntryAliases) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* first =
      CreateDummyBuffer(UINT64_C(0x100000000), 0x1000, allocator,
                        reinterpret_cast<void*>(UINT64_C(0x200000000)));
  auto* middle =
      CreateDummyBuffer(UINT64_C(0x300000000), 0x1000, allocator,
                        reinterpret_cast<void*>(UINT64_C(0x400000000)));
  auto* last =
      CreateDummyBuffer(UINT64_C(0x500000000), 0x1000, allocator,
                        reinterpret_cast<void*>(UINT64_C(0x600000000)));
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, first));
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, middle));
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, last));

  IREE_ASSERT_OK(
      iree_hal_streaming_buffer_table_remove(table, UINT64_C(0x100000000)));
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, UINT64_C(0x500000800), &found));
  EXPECT_EQ(last, found);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, UINT64_C(0x600000800), &found));
  EXPECT_EQ(last, found);
  IREE_ASSERT_OK(
      iree_hal_streaming_buffer_table_remove(table, UINT64_C(0x600000000)));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, UINT64_C(0x400000800), &found));
  EXPECT_EQ(middle, found);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(first, allocator);
  FreeDummyBuffer(middle, allocator);
  FreeDummyBuffer(last, allocator);
}

TEST(BufferTableTest, LookupMissing) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));
  EXPECT_EQ(found, nullptr);

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, RemoveMissing) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL)),
      StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, LookupRangeInvalidSize) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100000000ULL, 0, &found)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, LookupRangeOverflow) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;
  // Request a range that would overflow.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, UINT64_MAX - 10, 20, &found)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, RetainedRangeSurvivesTableRemoval) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_atomic_ref_count_init(&buffer->ref_count);
  buffer->size = 4096;

  constexpr uint64_t kDevicePointer = UINT64_C(0x100000000);
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, kDevicePointer, /*host_ptr=*/nullptr, buffer->size, buffer,
      /*user_data=*/nullptr)));

  hrx_buffer_table_retained_ref_t retained = {};
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_find_range_retain(
      table, kDevicePointer + 32, /*size=*/8, &retained)));
  EXPECT_EQ(buffer, retained.buffer);
  EXPECT_EQ(kDevicePointer, retained.device_ptr);
  EXPECT_EQ(nullptr, retained.host_ptr);
  EXPECT_EQ(4096u, retained.size);
  EXPECT_EQ(32u, retained.offset);
  EXPECT_EQ(2, iree_atomic_ref_count_load(&buffer->ref_count));

  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, kDevicePointer)));
  hrx_buffer_release(buffer);
  EXPECT_EQ(4096u, retained.buffer->size);
  EXPECT_EQ(1, iree_atomic_ref_count_load(&retained.buffer->ref_count));
  hrx_buffer_release(retained.buffer);
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, RetainedRangeSnapshotsHostAlias) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_atomic_ref_count_init(&buffer->ref_count);
  buffer->size = 4096;

  constexpr uint64_t kDevicePointer = UINT64_C(0x100000000);
  constexpr uintptr_t kHostPointer = UINT64_C(0x200000000);
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, kDevicePointer, reinterpret_cast<void*>(kHostPointer),
      buffer->size, buffer, /*user_data=*/nullptr)));

  hrx_buffer_table_retained_ref_t retained = {};
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_find_range_retain(
      table, kHostPointer + 24, /*size=*/8, &retained)));
  EXPECT_EQ(buffer, retained.buffer);
  EXPECT_EQ(kDevicePointer, retained.device_ptr);
  EXPECT_EQ(reinterpret_cast<void*>(kHostPointer), retained.host_ptr);
  EXPECT_EQ(4096u, retained.size);
  EXPECT_EQ(24u, retained.offset);

  hrx_buffer_release(retained.buffer);
  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, kDevicePointer)));
  hrx_buffer_release(buffer);
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, RetainedRangeAcquiresAndSnapshotsOpaquePayload) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_atomic_ref_count_init(&buffer->ref_count);
  buffer->size = 4096;

  constexpr uint64_t kDevicePointer = UINT64_C(0x100000000);
  uint64_t payload = 0;
  IREE_ASSERT_OK(BufferTableStatus(
      hrx_buffer_table_insert(table, kDevicePointer, /*host_ptr=*/nullptr,
                              buffer->size, buffer, &payload)));

  EntryCallbackState callback_state = {
      .expected_user_data = &payload,
      .expected_offset = 32,
      .call_count = 0,
  };
  hrx_buffer_table_retained_ref_t retained = {};
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_find_range_retain_if(
      table, kDevicePointer + callback_state.expected_offset, /*size=*/8,
      AcceptEntryCallback, &callback_state, &retained)));
  EXPECT_EQ(1, callback_state.call_count);
  EXPECT_EQ(&payload, retained.user_data);
  EXPECT_EQ(buffer, retained.buffer);
  EXPECT_EQ(2, iree_atomic_ref_count_load(&buffer->ref_count));

  hrx_buffer_release(retained.buffer);
  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, kDevicePointer)));
  hrx_buffer_release(buffer);
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, RejectedRangeAcquisitionLeavesEntryUnchanged) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_atomic_ref_count_init(&buffer->ref_count);
  buffer->size = 4096;
  constexpr uint64_t kDevicePointer = UINT64_C(0x100000000);
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, kDevicePointer, /*host_ptr=*/nullptr, buffer->size, buffer,
      /*user_data=*/nullptr)));

  hrx_buffer_table_retained_ref_t retained = {};
  EXPECT_THAT(Status(BufferTableStatus(hrx_buffer_table_find_range_retain_if(
                  table, kDevicePointer, /*size=*/8, RejectEntryCallback,
                  /*callback_user_data=*/nullptr, &retained))),
              StatusIs(StatusCode::kFailedPrecondition));
  EXPECT_EQ(nullptr, retained.buffer);
  EXPECT_EQ(1, iree_atomic_ref_count_load(&buffer->ref_count));

  hrx_buffer_t found = nullptr;
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_find(
      table, kDevicePointer, &found, /*out_offset=*/nullptr,
      /*out_user_data=*/nullptr)));
  EXPECT_EQ(buffer, found);

  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, kDevicePointer)));
  hrx_buffer_release(buffer);
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, BulkLookupRetainsEachAllocationOnce) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  hrx_buffer_t first = nullptr;
  hrx_buffer_t second = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*first), (void**)&first));
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*second), (void**)&second));
  memset(first, 0, sizeof(*first));
  memset(second, 0, sizeof(*second));
  iree_atomic_ref_count_init(&first->ref_count);
  iree_atomic_ref_count_init(&second->ref_count);
  first->size = 0x100;
  second->size = 0x100;
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, UINT64_C(0x1000), reinterpret_cast<void*>(UINT64_C(0x3000)),
      first->size, first, reinterpret_cast<void*>(UINT64_C(1)))));
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, UINT64_C(0x5000), /*host_ptr=*/nullptr, second->size, second,
      reinterpret_cast<void*>(UINT64_C(2)))));

  const hrx_buffer_table_range_request_t requests[] = {
      {.address = UINT64_C(0x1010), .length = 4},
      {.address = UINT64_C(0x3020), .length = 8},
      {.address = UINT64_C(0x5080), .length = 8},
      {.address = UINT64_C(0x9000), .length = 4},
      {.address = UINT64_C(0x1030), .length = 4},
  };
  hrx_buffer_table_retained_ref_t refs[IREE_ARRAYSIZE(requests)] = {};
  hrx_buffer_table_range_match_t matches[IREE_ARRAYSIZE(requests)] = {};
  BulkCallbackState callback_state = {};
  size_t ref_count = 0;
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_find_ranges_retain_if(
      table, IREE_ARRAYSIZE(requests), requests, CountBulkEntryCallback,
      &callback_state, IREE_ARRAYSIZE(refs), refs, &ref_count, matches)));

  EXPECT_EQ(2u, ref_count);
  EXPECT_EQ(2, callback_state.call_count);
  EXPECT_EQ(first, refs[0].buffer);
  EXPECT_EQ(second, refs[1].buffer);
  EXPECT_EQ(matches[0].ref_index, matches[1].ref_index);
  EXPECT_EQ(matches[0].ref_index, matches[4].ref_index);
  EXPECT_NE(matches[0].ref_index, matches[2].ref_index);
  EXPECT_EQ(SIZE_MAX, matches[3].ref_index);
  EXPECT_EQ(0x10u, matches[0].offset);
  EXPECT_EQ(0x20u, matches[1].offset);
  EXPECT_EQ(0x80u, matches[2].offset);
  EXPECT_EQ(0x30u, matches[4].offset);

  for (size_t i = 0; i < ref_count; ++i) {
    hrx_buffer_release(refs[i].buffer);
  }
  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, UINT64_C(0x1000))));
  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, UINT64_C(0x5000))));
  hrx_buffer_release(first);
  hrx_buffer_release(second);
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, BulkLookupReportsRetainedPrefixOnCallbackFailure) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  hrx_buffer_t first = nullptr;
  hrx_buffer_t second = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*first), (void**)&first));
  IREE_ASSERT_OK(
      iree_allocator_malloc(allocator, sizeof(*second), (void**)&second));
  memset(first, 0, sizeof(*first));
  memset(second, 0, sizeof(*second));
  iree_atomic_ref_count_init(&first->ref_count);
  iree_atomic_ref_count_init(&second->ref_count);
  first->size = 0x100;
  second->size = 0x100;
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, UINT64_C(0x1000), /*host_ptr=*/nullptr, first->size, first,
      /*user_data=*/nullptr)));
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_insert(
      table, UINT64_C(0x5000), /*host_ptr=*/nullptr, second->size, second,
      /*user_data=*/nullptr)));

  const hrx_buffer_table_range_request_t requests[] = {
      {.address = UINT64_C(0x1010), .length = 4},
      {.address = UINT64_C(0x5010), .length = 4},
  };
  hrx_buffer_table_retained_ref_t refs[IREE_ARRAYSIZE(requests)] = {};
  hrx_buffer_table_range_match_t matches[IREE_ARRAYSIZE(requests)] = {};
  BulkCallbackState callback_state = {.call_count = 0, .reject_call = 2};
  size_t ref_count = 0;
  EXPECT_THAT(
      Status(BufferTableStatus(hrx_buffer_table_find_ranges_retain_if(
          table, IREE_ARRAYSIZE(requests), requests, CountBulkEntryCallback,
          &callback_state, IREE_ARRAYSIZE(refs), refs, &ref_count, matches))),
      StatusIs(StatusCode::kFailedPrecondition));
  EXPECT_EQ(1u, ref_count);
  EXPECT_EQ(2, callback_state.call_count);
  EXPECT_EQ(0u, matches[0].ref_index);
  EXPECT_EQ(SIZE_MAX, matches[1].ref_index);

  hrx_buffer_release(refs[0].buffer);
  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, UINT64_C(0x1000))));
  IREE_ASSERT_OK(
      BufferTableStatus(hrx_buffer_table_remove(table, UINT64_C(0x5000))));
  hrx_buffer_release(first);
  hrx_buffer_release(second);
  iree_hal_streaming_buffer_table_free(table);
}

//===----------------------------------------------------------------------===//
// Multiple operations
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, InsertMultiple) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 100;
  std::vector<iree_hal_streaming_buffer_t*> buffers;

  // Insert multiple buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify all can be looked up.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, addr, &found));
    EXPECT_EQ(found, buffers[i]);
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RemoveFIFO) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 50;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Remove in FIFO order.
  for (size_t i = 0; i < count; ++i) {
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, addresses[i]));

    // Verify it's gone.
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[i], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RemoveLIFO) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 50;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Remove in LIFO order.
  for (size_t i = count; i > 0; --i) {
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_remove(table, addresses[i - 1]));

    // Verify it's gone.
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[i - 1], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RemoveRandom) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 50;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Create random removal order.
  std::vector<size_t> removal_order(count);
  std::iota(removal_order.begin(), removal_order.end(), 0);
  std::mt19937 gen(0x12345678);
  std::shuffle(removal_order.begin(), removal_order.end(), gen);

  // Remove in random order.
  for (size_t index : removal_order) {
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_remove(table, addresses[index]));

    // Verify it's gone.
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[index], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

//===----------------------------------------------------------------------===//
// Range lookup tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, LookupRangeExact) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x8000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup exact range.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100000000ULL, 0x8000, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeWithin) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x8000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup range within buffer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100001000ULL, 0x2000, &found));
  EXPECT_EQ(found, buffer);

  // Lookup at the end of the buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100007000ULL, 0x1000, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeOutOfBounds) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x8000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  iree_hal_streaming_buffer_t* found = nullptr;

  // Range starts before buffer.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x0FFFFFFF00ULL, 0x1000, &found)),
              StatusIs(StatusCode::kNotFound));

  // Range extends past buffer end.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100007000ULL, 0x2000, &found)),
              StatusIs(StatusCode::kNotFound));

  // Range starts after buffer end.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100008000ULL, 0x1000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeMultipleBuffers) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  // Create adjacent buffers.
  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator);
  auto* buffer2 = CreateDummyBuffer(0x100004000ULL, 0x4000, allocator);
  auto* buffer3 = CreateDummyBuffer(0x100008000ULL, 0x4000, allocator);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer2));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer3));

  iree_hal_streaming_buffer_t* found = nullptr;

  // Range within first buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100001000ULL, 0x2000, &found));
  EXPECT_EQ(found, buffer1);

  // Range within second buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100005000ULL, 0x2000, &found));
  EXPECT_EQ(found, buffer2);

  // Range spanning two buffers should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100003000ULL, 0x2000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
  FreeDummyBuffer(buffer3, allocator);
}

//===----------------------------------------------------------------------===//
// Edge cases and stress tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, EmptyTableOperations) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;

  // All operations should fail gracefully on empty table.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100000000ULL, 0x1000, &found)),
              StatusIs(StatusCode::kNotFound));

  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL)),
      StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, LargeScale) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 10000;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Generate random addresses.
  std::mt19937_64 gen(0xDEADBEEF);
  std::uniform_int_distribution<iree_hal_streaming_deviceptr_t> dist(
      0x100000000ULL, 0x700000000000ULL);

  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = dist(gen) & ~0xFULL;  // Align to 16.
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 0x1000, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify random lookups.
  std::uniform_int_distribution<size_t> index_dist(0, count - 1);
  for (size_t i = 0; i < 100; ++i) {
    size_t index = index_dist(gen);
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
        table, addresses[index], &found));
    EXPECT_EQ(found, buffers[index]);
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, SparseAddresses) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 100;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Create sparse addresses with exponentially growing gaps.
  iree_hal_streaming_deviceptr_t base = 0x100000000ULL;
  for (size_t i = 0; i < count; ++i) {
    base += 0x10000 * (i + 1);  // Gaps: 0x10000, 0x20000, 0x30000, etc.
    addresses.push_back(base);
    auto* buffer = CreateDummyBuffer(base, 0x1000, allocator);
    buffers.push_back(buffer);
    IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify all can be looked up.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
    EXPECT_EQ(found, buffers[i]);
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, InsertRemoveInsert) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 0x1000, allocator);
  auto* buffer2 = CreateDummyBuffer(0x100000000ULL, 0x2000, allocator);

  // Insert, remove, then insert again with same address.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer2));

  // Should find the second buffer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer2);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
}

TEST(BufferTableTest, ReservedInsertSurvivesCapacityPressure) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* removed_buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, removed_buffer));
  const size_t reserved_capacity = table->capacity;
  hrx_buffer_table_entry_t removed_entry = {};
  size_t removed_offset = SIZE_MAX;
  IREE_ASSERT_OK(BufferTableStatus(hrx_buffer_table_remove_reserved_if(
      table, removed_buffer->device_ptr, /*callback=*/nullptr,
      /*callback_user_data=*/nullptr, &removed_entry, &removed_offset)));
  EXPECT_EQ(removed_buffer->device_ptr, removed_entry.device_ptr);
  EXPECT_EQ((hrx_buffer_t)removed_buffer, removed_entry.buffer);
  EXPECT_EQ(0u, removed_offset);
  EXPECT_EQ(1u, table->reserved_insert_count);

  std::vector<iree_hal_streaming_buffer_t*> competing_buffers;
  competing_buffers.reserve(reserved_capacity - 1);
  for (size_t i = 0; i < reserved_capacity - 1; ++i) {
    auto* buffer =
        CreateDummyBuffer(0x200000000ULL + i * 0x10000, 4096, allocator);
    competing_buffers.push_back(buffer);
    IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }
  EXPECT_EQ(table->capacity, reserved_capacity);
  EXPECT_EQ(table->reserved_insert_count, 1u);

  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_insert_reserved(table, removed_buffer));
  EXPECT_EQ(table->capacity, reserved_capacity);
  EXPECT_EQ(table->count, reserved_capacity);
  EXPECT_EQ(table->reserved_insert_count, 0u);

  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, removed_buffer->device_ptr, &found));
  EXPECT_EQ(found, removed_buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(removed_buffer, allocator);
  for (auto* buffer : competing_buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RejectedRemovalDoesNotReserveOrRemove) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  hrx_buffer_table_entry_t removed_entry = {};
  EXPECT_THAT(Status(BufferTableStatus(hrx_buffer_table_remove_reserved_if(
                  table, buffer->device_ptr, RejectEntryCallback,
                  /*callback_user_data=*/nullptr, &removed_entry,
                  /*out_offset=*/nullptr))),
              StatusIs(StatusCode::kFailedPrecondition));
  EXPECT_EQ(nullptr, removed_entry.buffer);
  EXPECT_EQ(0u, table->reserved_insert_count);

  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_buffer_table_lookup(
      table, buffer->device_ptr, &found));
  EXPECT_EQ(buffer, found);

  IREE_ASSERT_OK(
      iree_hal_streaming_buffer_table_remove(table, buffer->device_ptr));
  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, MixedOperations) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 100;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert initial set.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 0x4000, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Remove every other buffer.
  for (size_t i = 0; i < count; i += 2) {
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, addresses[i]));
  }

  // Verify remaining buffers can be found.
  for (size_t i = 1; i < count; i += 2) {
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
    EXPECT_EQ(found, buffers[i]);
  }

  // Verify removed buffers cannot be found.
  for (size_t i = 0; i < count; i += 2) {
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[i], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  // Add new buffers at removed addresses.
  std::vector<iree_hal_streaming_buffer_t*> new_buffers;
  for (size_t i = 0; i < count; i += 2) {
    auto* buffer = CreateDummyBuffer(addresses[i], 0x8000, allocator);
    new_buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify all buffers can now be found.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
    if (i % 2 == 0) {
      // Should find the new buffer.
      EXPECT_EQ(found, new_buffers[i / 2]);
    } else {
      // Should find the original buffer.
      EXPECT_EQ(found, buffers[i]);
    }
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
  for (auto* buffer : new_buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

//===----------------------------------------------------------------------===//
// Host pointer tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, InsertWithHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  // Create a buffer with both device and host pointers.
  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Should be able to lookup by device pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer);

  // Should be able to lookup by host pointer.
  found = nullptr;
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, host_addr, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, RemoveByHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Remove using host pointer.
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, host_addr));

  // Should not be found by either pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));
  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_lookup(table, host_addr, &found)),
      StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeWithHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup range using host pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, host_addr + 0x1000, 0x2000, &found));
  EXPECT_EQ(found, buffer);

  // Range extending beyond buffer should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, host_addr + 0x2000, 0x3000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, MixedHostPointers) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  // Insert buffers with and without host pointers.
  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  void* host_ptr2 = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer2 = CreateDummyBuffer(0x110000000ULL, 4096, allocator, host_ptr2);
  auto* buffer3 = CreateDummyBuffer(0x120000000ULL, 4096, allocator);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer2));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer3));

  // Verify all can be looked up by device pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer1);
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x110000000ULL, &found));
  EXPECT_EQ(found, buffer2);
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x120000000ULL, &found));
  EXPECT_EQ(found, buffer3);

  // Only buffer2 should be found by host pointer.
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr2;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, host_addr, &found));
  EXPECT_EQ(found, buffer2);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
  FreeDummyBuffer(buffer3, allocator);
}

//===----------------------------------------------------------------------===//
// Mid-buffer lookup tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, LookupMidBufferDevice) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup at various offsets within the buffer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL,
                                                        &found));  // Start
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, 0x100001000ULL,
                                                        &found));  // Middle
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, 0x100003FFFULL,
                                                        &found));  // Last byte
  EXPECT_EQ(found, buffer);

  // Just past the end should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100004000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupMidBufferHost) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup at various offsets within the host buffer range.
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, host_addr,
                                                        &found));  // Start
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, host_addr + 0x1000, &found));  // Middle
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, host_addr + 0x3FFF, &found));  // Last byte
  EXPECT_EQ(found, buffer);

  // Just past the end should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, host_addr + 0x4000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, RemoveMidBuffer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Remove using a pointer in the middle of the buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, 0x100002000ULL));

  // Buffer should be gone.
  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, DoubleInsertHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 4096, allocator, host_ptr);
  // Different device pointer but same host pointer.
  auto* buffer2 = CreateDummyBuffer(0x110000000ULL, 4096, allocator, host_ptr);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));

  // Second insert with same host_ptr should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, buffer2)),
              StatusIs(StatusCode::kAlreadyExists));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
}

}  // namespace
}  // namespace iree::hal::stream
