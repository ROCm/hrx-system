// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_buffer.h"

#include <array>
#include <cstring>
#include <string>

#include "iree/hal/replay/file_reader.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ReplayRecorderBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_io_file_handle_t* file_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(recording_.data(), recording_.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &file_handle));
    iree_status_t status = iree_hal_replay_recorder_create(
        file_handle, nullptr, iree_allocator_system(), &recorder_);
    iree_io_file_handle_release(file_handle);
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
            IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
        sizeof(storage_), iree_make_byte_span(storage_, sizeof(storage_)),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &native_buffer_));
  }

  void TearDown() override {
    iree_hal_buffer_release(native_buffer_);
    if (recorder_) {
      IREE_EXPECT_STATUS_IS(close_status_,
                            iree_hal_replay_recorder_close(recorder_));
      iree_hal_replay_recorder_release(recorder_);
    }
  }

  // Destination for the real recorder's capture stream.
  std::array<uint8_t, 4096> recording_ = {};
  // Recorder retained until the wrapped buffers have been released.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Expected terminal capture result, independent of forwarded HAL failures.
  iree_status_code_t close_status_ = IREE_STATUS_OK;
  // Native host backing exposed by the wrapped buffer.
  alignas(64) uint8_t storage_[128] = {};
  // Whole native allocation from which the recorded view is created.
  iree_hal_buffer_t* native_buffer_ = nullptr;
};

TEST_F(ReplayRecorderBufferTest, ExportsNestedSubspansOfANativeView) {
  iree_hal_buffer_t* native_view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      native_buffer_, 32, 64, iree_allocator_system(), &native_view));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_recorder_buffer_create_proxy(
      recorder_, /*device_id=*/1, /*buffer_id=*/2,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, /*placement_device=*/nullptr, native_view,
      iree_allocator_system(), &buffer));
  iree_hal_buffer_release(native_view);

  iree_hal_external_storage_t external_buffer = {};
  iree_hal_buffer_range_t logical_range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      &external_buffer, &logical_range));
  EXPECT_EQ(storage_, external_buffer.payload.host_pointer);
  EXPECT_EQ(sizeof(storage_), external_buffer.byte_length);
  EXPECT_EQ(32u, logical_range.byte_offset);
  EXPECT_EQ(64u, logical_range.byte_length);
  iree_hal_external_storage_release(&external_buffer);

  iree_hal_buffer_t* whole_view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                         iree_allocator_system(), &whole_view));
  EXPECT_EQ(buffer, whole_view);
  iree_hal_buffer_release(whole_view);

  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 14, 16,
                                         iree_allocator_system(), &subspan));
  iree_hal_buffer_t* nested_subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(subspan, 4, 8, iree_allocator_system(),
                                         &nested_subspan));
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      &external_buffer, &logical_range));
  EXPECT_EQ(storage_, external_buffer.payload.host_pointer);
  EXPECT_EQ(sizeof(storage_), external_buffer.byte_length);
  EXPECT_EQ(50u, logical_range.byte_offset);
  EXPECT_EQ(8u, logical_range.byte_length);
  iree_hal_external_storage_release(&external_buffer);
  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS),
      &external_buffer, &logical_range));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(storage_),
            external_buffer.payload.device_address);
  EXPECT_EQ(sizeof(storage_), external_buffer.byte_length);
  EXPECT_EQ(50u, logical_range.byte_offset);
  EXPECT_EQ(8u, logical_range.byte_length);
  iree_hal_external_storage_release(&external_buffer);
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      nested_subspan, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
  EXPECT_EQ(storage_ + 50, mapping.contents.data);
  EXPECT_EQ(8u, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(nested_subspan);
}

TEST_F(ReplayRecorderBufferTest, ScopedMappingsCaptureViewRelativeIdentity) {
  iree_hal_buffer_t* native_view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      native_buffer_, 32, 64, iree_allocator_system(), &native_view));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_recorder_buffer_create_proxy(
      recorder_, /*device_id=*/98, /*buffer_id=*/99,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, nullptr, native_view,
      iree_allocator_system(), &buffer));
  iree_hal_buffer_release(native_view);
  std::array<iree_hal_buffer_mapping_t, 2> mappings;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 2, 8, &mappings[0]));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 4, 4, &mappings[1]));
  iree_hal_buffer_release(buffer);
  std::memset(mappings[0].contents.data, 0xA1, 8);
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&mappings[0], 1, 2));
  IREE_ASSERT_OK(iree_hal_buffer_mapping_invalidate_range(&mappings[1], 1, 2));
  EXPECT_EQ(storage_ + 36, mappings[1].contents.data);
  EXPECT_EQ(0xA1, mappings[1].contents.data[0]);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mappings[0]));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mappings[1]));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));

  // Both maps use the same captured buffer but have independent lifetimes.
  const std::array<iree_hal_replay_operation_code_t, 6> operations = {
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE,
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE,
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE,
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE,
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE,
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE,
  };
  const std::array<uint64_t, 6> offsets = {2, 4, 3, 5, 2, 4};
  const std::array<uint64_t, 6> lengths = {8, 4, 2, 2, 8, 4};
  const std::array<size_t, 6> owners = {0, 1, 0, 1, 0, 1};
  std::array<iree_hal_replay_object_id_t, 2> mapping_ids = {};
  auto contents =
      iree_make_const_byte_span(recording_.data(), recording_.size());
  iree_hal_replay_file_header_t header;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_header(contents, &header, &offset));
  contents.data_length = header.file_length;
  size_t operation_index = 0;
  while (offset < contents.data_length) {
    iree_hal_replay_file_record_t record;
    IREE_ASSERT_OK(
        iree_hal_replay_file_parse_record(contents, offset, &record, &offset));
    if (record.header.record_type !=
        IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION) {
      continue;
    }
    ASSERT_LT(operation_index, operations.size());
    EXPECT_EQ(operations[operation_index], record.header.operation_code);
    EXPECT_EQ(99u, record.header.object_id);
    if (operation_index < mapping_ids.size()) {
      mapping_ids[operation_index] = record.header.related_object_id;
      EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, mapping_ids[operation_index]);
    }
    EXPECT_EQ(mapping_ids[owners[operation_index]],
              record.header.related_object_id);
    uint64_t range[2];
    ASSERT_GE(record.payload.data_length, sizeof(range));
    std::memcpy(range, record.payload.data, sizeof(range));
    EXPECT_EQ(offsets[operation_index], range[0]);
    EXPECT_EQ(lengths[operation_index], range[1]);
    ++operation_index;
  }
  EXPECT_EQ(operations.size(), operation_index);
  EXPECT_NE(mapping_ids[0], mapping_ids[1]);
}

TEST_F(ReplayRecorderBufferTest, ScopedMappingUnmapsAfterCaptureCloses) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_recorder_buffer_create_proxy(
      recorder_, /*device_id=*/98, /*buffer_id=*/99,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, nullptr, native_buffer_,
      iree_allocator_system(), &buffer));
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 16, &mapping));
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
  std::memset(mapping.contents.data, 0xB2, 16);
  IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
  EXPECT_EQ(nullptr, mapping.buffer);
  EXPECT_EQ(0xB2, storage_[0]);
}

// Native mapping dependency with observable teardown. The recorder and public
// HAL mapping lifecycle remain real; only the wrapped backend is controlled.
struct MappingBuffer {
  // Native resource header at offset zero.
  iree_hal_buffer_t base;
  // Mapped host bytes borrowed by the recorder until unmap.
  uint8_t storage[16] = {};
  // Number of native mappings created.
  int map_count = 0;
  // Number of native mappings consumed, including failed unmaps.
  int unmap_count = 0;
  // Terminal native result returned after consuming the mapping.
  iree_status_code_t unmap_status = IREE_STATUS_OK;
  // Whether every native owner has been released.
  bool destroyed = false;
};

static const iree_hal_buffer_vtable_t kMappingBufferVTable = {
    .destroy =
        [](iree_hal_buffer_t* base_buffer) {
          auto* buffer = reinterpret_cast<MappingBuffer*>(base_buffer);
          EXPECT_EQ(buffer->map_count, buffer->unmap_count);
          buffer->destroyed = true;
        },
    .export_range = nullptr,
    .map_range =
        [](iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
           iree_hal_memory_access_t memory_access,
           iree_hal_buffer_map_flags_t flags, iree_device_size_t byte_offset,
           iree_device_size_t byte_length, iree_hal_buffer_mapping_t* mapping) {
          auto* buffer = reinterpret_cast<MappingBuffer*>(base_buffer);
          ++buffer->map_count;
          mapping->contents =
              iree_make_byte_span(buffer->storage + byte_offset, byte_length);
          return iree_ok_status();
        },
    .unmap_range =
        [](iree_hal_buffer_t* base_buffer, iree_device_size_t byte_offset,
           iree_device_size_t byte_length, iree_hal_buffer_mapping_t* mapping) {
          auto* buffer = reinterpret_cast<MappingBuffer*>(base_buffer);
          ++buffer->unmap_count;
          return iree_status_from_code(buffer->unmap_status);
        },
};

enum class CaptureState { kOpen, kClosed, kFailed, kSnapshotAllocationFailure };

class ReplayRecorderBufferUnmapTest
    : public ReplayRecorderBufferTest,
      public ::testing::WithParamInterface<iree_status_code_t> {
 protected:
  void CheckUnmap(CaptureState capture_state, iree_hal_memory_access_t access) {
    MappingBuffer native = {};
    native.unmap_status = GetParam();
    iree_hal_buffer_initialize(
        iree_hal_buffer_placement_undefined(), &native.base,
        sizeof(native.storage), 0, sizeof(native.storage),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED, &kMappingBufferVTable,
        &native.base);
    bool fail_allocations = false;
    const iree_allocator_t allocator = {
        .self = &fail_allocations,
        .ctl =
            [](void* self, iree_allocator_command_t command,
               const void* parameters, void** pointer) {
              if (*static_cast<bool*>(self) &&
                  (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                   command == IREE_ALLOCATOR_COMMAND_CALLOC)) {
                return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
              }
              auto allocator = iree_allocator_system();
              return allocator.ctl(allocator.self, command, parameters,
                                   pointer);
            },
    };
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_replay_recorder_buffer_create_proxy(
        recorder_, /*device_id=*/1, /*buffer_id=*/2,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, /*placement_device=*/nullptr,
        &native.base, allocator, &buffer));
    iree_hal_buffer_release(&native.base);
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, access,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
    EXPECT_EQ(native.map_count, 1);
    // The mapping is now the only owner of the recording wrapper.
    iree_hal_buffer_release(buffer);

    switch (capture_state) {
      case CaptureState::kOpen:
        break;
      case CaptureState::kClosed:
        IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder_));
        break;
      case CaptureState::kFailed: {
        // A real capture write exceeds the fixed recording destination.
        const std::string scope(recording_.size(), 'x');
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_OUT_OF_RANGE,
            iree_hal_replay_recorder_scope_begin(
                recorder_, iree_make_string_view(scope.data(), scope.size())));
        close_status_ = IREE_STATUS_OUT_OF_RANGE;
        break;
      }
      case CaptureState::kSnapshotAllocationFailure:
        fail_allocations = true;
        close_status_ = IREE_STATUS_RESOURCE_EXHAUSTED;
        break;
    }

    IREE_EXPECT_STATUS_IS(GetParam(), iree_hal_buffer_unmap_range(&mapping));
    EXPECT_EQ(native.unmap_count, 1);
    EXPECT_TRUE(native.destroyed);
    EXPECT_EQ(mapping.buffer, nullptr);
    EXPECT_EQ(mapping.contents.data, nullptr);
    // Public teardown consumes the record even when native unmap failed.
    IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
    EXPECT_EQ(native.unmap_count, 1);
  }
};

TEST_P(ReplayRecorderBufferUnmapTest, OpenCapture) {
  CheckUnmap(CaptureState::kOpen, IREE_HAL_MEMORY_ACCESS_WRITE);
}

TEST_P(ReplayRecorderBufferUnmapTest, ClosedCapture) {
  CheckUnmap(CaptureState::kClosed, IREE_HAL_MEMORY_ACCESS_WRITE);
}

TEST_P(ReplayRecorderBufferUnmapTest, FailedCapture) {
  CheckUnmap(CaptureState::kFailed, IREE_HAL_MEMORY_ACCESS_WRITE);
}

TEST_P(ReplayRecorderBufferUnmapTest, SnapshotAllocationFailure) {
  CheckUnmap(CaptureState::kSnapshotAllocationFailure,
             IREE_HAL_MEMORY_ACCESS_WRITE);
}

TEST_P(ReplayRecorderBufferUnmapTest, ClosedReadCapture) {
  CheckUnmap(CaptureState::kClosed, IREE_HAL_MEMORY_ACCESS_READ);
}

INSTANTIATE_TEST_SUITE_P(NativeResult, ReplayRecorderBufferUnmapTest,
                         ::testing::Values(IREE_STATUS_OK,
                                           IREE_STATUS_DATA_LOSS));

}  // namespace
