// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {
constexpr iree_device_size_t kDefaultAllocationSize = 1024;
constexpr iree_host_size_t kSourceBufferSlot = 0;
constexpr iree_host_size_t kTargetBufferSlot = 1;
constexpr iree_host_size_t kBindingSlotCount = 2;
}  // namespace

class CommandBufferCopyBufferTest : public CtsTestBase<> {
 protected:
  // Returns a buffer ref using either direct or indirect mode.
  // In direct mode the slot is ignored; in indirect mode the buffer is ignored.
  iree_hal_buffer_ref_t MakeRef(iree_hal_buffer_t* buffer,
                                iree_host_size_t slot,
                                iree_device_size_t offset,
                                iree_device_size_t length) {
    if (recording_mode() == RecordingMode::kIndirect) {
      return iree_hal_make_indirect_buffer_ref(slot, offset, length);
    }
    return iree_hal_make_buffer_ref(buffer, offset, length);
  }

  iree_host_size_t binding_capacity() {
    return recording_mode() == RecordingMode::kIndirect ? kBindingSlotCount : 0;
  }

  iree_status_t SubmitWithBindings(iree_hal_command_buffer_t* command_buffer,
                                   iree_hal_buffer_t* source_buffer,
                                   iree_hal_buffer_t* target_buffer) {
    if (recording_mode() == RecordingMode::kIndirect) {
      const iree_hal_buffer_binding_t bindings[] = {
          /*kSourceBufferSlot=*/{source_buffer, 0, IREE_HAL_WHOLE_BUFFER},
          /*kTargetBufferSlot=*/{target_buffer, 0, IREE_HAL_WHOLE_BUFFER},
      };
      return SubmitCommandBufferAndWait(
          command_buffer,
          iree_hal_buffer_binding_table_t{IREE_ARRAYSIZE(bindings), bindings});
    }
    return SubmitCommandBufferAndWait(command_buffer);
  }

  void RunCopyBufferTest(iree_device_size_t source_offset,
                         iree_device_size_t target_offset,
                         iree_device_size_t length) {
    iree_device_size_t buffer_size = source_offset + length;
    if (target_offset + length > buffer_size) {
      buffer_size = target_offset + length;
    }
    buffer_size += 16;

    std::vector<uint8_t> source_data = MakeDeterministicBytes(buffer_size);
    Ref<iree_hal_buffer_t> source_buffer;
    IREE_ASSERT_OK(CreateDeviceBufferWithData(source_data.data(), buffer_size,
                                              source_buffer.out()));
    Ref<iree_hal_buffer_t> target_buffer;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, target_buffer.out()));

    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                       IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                       binding_capacity(),
                                       command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(
        command_buffer,
        MakeRef(source_buffer, kSourceBufferSlot, source_offset, length),
        MakeRef(target_buffer, kTargetBufferSlot, target_offset, length),
        IREE_HAL_COPY_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    IREE_ASSERT_OK(
        SubmitWithBindings(command_buffer, source_buffer, target_buffer));

    std::vector<uint8_t> expected(buffer_size, 0);
    std::memcpy(expected.data() + target_offset,
                source_data.data() + source_offset, length);
    auto actual_data = ReadBufferBytes(target_buffer, 0, buffer_size);
    EXPECT_THAT(actual_data, ContainerEq(expected));
  }
};

TEST_P(CommandBufferCopyBufferTest, RectanglesPreservePaddingAcrossTiles) {
  constexpr size_t kWidth = 37, kRows = 2049, kSlices = 2;
  constexpr size_t kSourceStride = 48, kTargetStride = 60;
  constexpr size_t kSourceSlice = kRows * kSourceStride + 24;
  constexpr size_t kTargetSlice = kRows * kTargetStride + 40;
  constexpr size_t kSourceCapacity = kSlices * kSourceSlice + 32;
  constexpr size_t kTargetCapacity = kSlices * kTargetSlice + 32;
  const auto source = MakeDeterministicBytes(kSourceCapacity);
  std::vector<uint8_t> expected(kTargetCapacity, 0);
  for (size_t z = 0; z < kSlices; ++z) {
    for (size_t y = 0; y < kRows; ++y) {
      memcpy(expected.data() + 7 + z * kTargetSlice + y * kTargetStride,
             source.data() + 3 + z * kSourceSlice + y * kSourceStride, kWidth);
    }
  }
  Ref<iree_hal_buffer_t> source_buffer, target_buffer;
  IREE_ASSERT_OK(CreateDeviceBufferWithData(source.data(), source.size(),
                                            source_buffer.out()));
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kTargetCapacity, target_buffer.out()));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                     binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const iree_hal_transfer_region_t region = {
      .row_length = kWidth,
      .row_count = kRows,
      .slice_count = kSlices,
      .source_strides =
          {
              .row_stride = kSourceStride,
              .slice_stride = kSourceSlice,
          },
      .target_strides =
          {
              .row_stride = kTargetStride,
              .slice_stride = kTargetSlice,
          },
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, MakeRef(source_buffer, 0, 3, kSourceCapacity - 3),
      MakeRef(target_buffer, 1, 7, kTargetCapacity - 7), &region, 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_ASSERT_OK(
      SubmitWithBindings(command_buffer, source_buffer, target_buffer));
  std::vector<uint8_t> actual(kTargetCapacity);
  IREE_ASSERT_OK(
      DownloadBufferData(target_buffer, 0, actual.data(), actual.size()));
  EXPECT_THAT(actual, ContainerEq(expected));
}

TEST_P(CommandBufferCopyBufferTest, InterleavedRowsMaySharePadding) {
  constexpr size_t kWidth = 17, kStride = 64, kRows = 32;
  constexpr size_t kCapacity = kStride * kRows;
  const auto initial = MakeDeterministicBytes(kCapacity);
  auto expected = initial;
  for (size_t y = 0; y < kRows; ++y) {
    memcpy(expected.data() + y * kStride + 23, initial.data() + y * kStride + 3,
           kWidth);
  }
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(
      CreateDeviceBufferWithData(initial.data(), initial.size(), buffer.out()));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                     binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const iree_hal_transfer_region_t region = {
      .row_length = kWidth,
      .row_count = kRows,
      .slice_count = 1,
      .source_strides = {.row_stride = kStride},
      .target_strides = {.row_stride = kStride},
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, MakeRef(buffer, 0, 3, kCapacity - 3),
      MakeRef(buffer, 1, 23, kCapacity - 23), &region, 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_ASSERT_OK(SubmitWithBindings(command_buffer, buffer, buffer));
  std::vector<uint8_t> actual(kCapacity);
  IREE_ASSERT_OK(DownloadBufferData(buffer, 0, actual.data(), actual.size()));
  EXPECT_THAT(actual, ContainerEq(expected));
}

TEST_P(CommandBufferCopyBufferTest, CopyWholeBuffer) {
  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize, i8_val);

  Ref<iree_hal_buffer_t> source_buffer;
  IREE_ASSERT_OK(CreateDeviceBufferWithData(
      reference_buffer.data(), kDefaultAllocationSize, source_buffer.out()));
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(kDefaultAllocationSize,
                                                 target_buffer.out()));

  // Copy the source buffer to the target buffer.
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                     binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(
      command_buffer,
      MakeRef(source_buffer, kSourceBufferSlot, 0, kDefaultAllocationSize),
      MakeRef(target_buffer, kTargetBufferSlot, 0, kDefaultAllocationSize),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(
      SubmitWithBindings(command_buffer, source_buffer, target_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(DownloadBufferData(target_buffer, /*source_offset=*/0,
                                    actual_data.data(),
                                    kDefaultAllocationSize));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferCopyBufferTest, CopySubBuffer) {
  uint8_t i8_val = 0x88;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize, 0);
  std::memset(reference_buffer.data() + 8, i8_val,
              kDefaultAllocationSize / 2 - 4);

  std::vector<uint8_t> source_buffer_data(kDefaultAllocationSize, i8_val);
  Ref<iree_hal_buffer_t> source_buffer;
  IREE_ASSERT_OK(CreateDeviceBufferWithData(source_buffer_data.data(),
                                            source_buffer_data.size() / 2,
                                            source_buffer.out()));
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(kDefaultAllocationSize,
                                                 target_buffer.out()));

  // Zero-fill, copy a sub-region, then zero-fill the remainder.
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                     binding_capacity(), command_buffer.out()));
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer_linear(
      command_buffer,
      MakeRef(target_buffer, kTargetBufferSlot, /*offset=*/0, /*length=*/8),
      &zero_val, /*pattern_length=*/sizeof(zero_val), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(
      command_buffer,
      MakeRef(source_buffer, kSourceBufferSlot, /*offset=*/4,
              /*length=*/kDefaultAllocationSize / 2 - 4),
      MakeRef(target_buffer, kTargetBufferSlot, /*offset=*/8,
              /*length=*/kDefaultAllocationSize / 2 - 4),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer_linear(
      command_buffer,
      MakeRef(target_buffer, kTargetBufferSlot,
              /*offset=*/8 + kDefaultAllocationSize / 2 - 4,
              /*length=*/kDefaultAllocationSize -
                  (8 + kDefaultAllocationSize / 2 - 4)),
      &zero_val, /*pattern_length=*/sizeof(zero_val), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(
      SubmitWithBindings(command_buffer, source_buffer, target_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(DownloadBufferData(target_buffer, /*source_offset=*/0,
                                    actual_data.data(),
                                    kDefaultAllocationSize));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferCopyBufferTest, CopySizeAndAlignmentClasses) {
  struct AlignmentCase {
    const char* name = nullptr;
    iree_device_size_t source_offset = 0;
    iree_device_size_t target_offset = 0;
  };
  const AlignmentCase alignment_cases[] = {
      {.name = "aligned16", .source_offset = 0, .target_offset = 0},
      {.name = "aligned8_not16", .source_offset = 8, .target_offset = 8},
      {.name = "aligned4_not8", .source_offset = 4, .target_offset = 4},
      {.name = "byte_misaligned", .source_offset = 1, .target_offset = 2},
  };
  const iree_device_size_t common_sizes[] = {
      4, 8, 16, 31, 32, 33, 64, 128, 256, 1024, 4 * 1024, 16 * 1024, 64 * 1024,
  };

  for (iree_device_size_t length : common_sizes) {
    for (const AlignmentCase& alignment_case : alignment_cases) {
      SCOPED_TRACE(::testing::Message()
                   << alignment_case.name
                   << " source_offset=" << alignment_case.source_offset
                   << " target_offset=" << alignment_case.target_offset
                   << " length=" << length);
      RunCopyBufferTest(alignment_case.source_offset,
                        alignment_case.target_offset, length);
    }
  }
}

TEST_P(CommandBufferCopyBufferTest, BulkSizeAndAlignmentClasses) {
  // Exercise exact tails across grid-stride and bulk-transfer boundaries.
  // The helper checks the untouched prefix and suffix as well.
  for (iree_device_size_t length :
       {65537, 262143, 262144, 262145, 1048575, 1048576, 1048577}) {
    SCOPED_TRACE(length);
    RunCopyBufferTest(/*source_offset=*/16, /*target_offset=*/32, length);
    RunCopyBufferTest(/*source_offset=*/1, /*target_offset=*/3, length);
  }
}

TEST_P(CommandBufferCopyBufferTest, FixedGeometryRequiresItsWholeBindingSpan) {
  const iree_hal_transfer_region_t region = {
      .row_length = 17,
      .row_count = 3,
      .slice_count = 2,
      .source_strides = {32, 128},
      .target_strides = {64, 256},
  };
  Ref<iree_hal_buffer_t> source, target;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(512, source.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(512, target.out()));
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(CreateCommandBuffer(0, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                     binding_capacity(), recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  if (recording_mode() == RecordingMode::kIndirect) {
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        recording, MakeRef(source, 0, 7, IREE_HAL_WHOLE_BUFFER),
        MakeRef(target, 1, 11, IREE_HAL_WHOLE_BUFFER), &region, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    iree_hal_buffer_binding_t bindings[] = {
        {source, 0, 7 + 128 + 64 + 17},
        {target, 0, 11 + 256 + 128 + 17},
    };
    IREE_ASSERT_OK(
        iree_hal_command_buffer_validate_submission(recording, {2, bindings}));
    --bindings[0].length;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_command_buffer_validate_submission(recording, {2, bindings}));
    ++bindings[0].length;
    --bindings[1].length;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_command_buffer_validate_submission(recording, {2, bindings}));
  } else {
    Ref<iree_hal_buffer_t> short_target;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(target, 0, 11 + 256 + 128 + 16,
                                           iree_allocator_system(),
                                           short_target.out()));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_command_buffer_copy_buffer(
            recording, MakeRef(source, 0, 7, IREE_HAL_WHOLE_BUFFER),
            MakeRef(short_target, 1, 11, IREE_HAL_WHOLE_BUFFER), &region, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  }
}

TEST_P(CommandBufferCopyBufferTest, RemainingRanges) {
  const bool indirect = recording_mode() == RecordingMode::kIndirect;
  Ref<iree_hal_command_buffer_t> reusable_command_buffer;
  if (indirect) {
    IREE_ASSERT_OK(CreateCommandBuffer(
        /*mode=*/0, IREE_HAL_COMMAND_CATEGORY_TRANSFER, binding_capacity(),
        reusable_command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(reusable_command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(
        reusable_command_buffer,
        MakeRef(nullptr, kSourceBufferSlot, 7, IREE_HAL_WHOLE_BUFFER),
        MakeRef(nullptr, kTargetBufferSlot, 11, IREE_HAL_WHOLE_BUFFER),
        IREE_HAL_COPY_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(reusable_command_buffer));
  }

  for (uint32_t length : {0, 13, 65536 + 13, 2 * 65536 + 13}) {
    SCOPED_TRACE(length);
    const iree_device_size_t table_offset = indirect ? length % 8 : 0;
    const iree_device_size_t source_offset = table_offset + 7;
    const iree_device_size_t target_offset = table_offset + 11;
    const iree_device_size_t padding = indirect ? 16 : 0;
    auto source_data = MakeDeterministicBytes(source_offset + length + padding);
    Ref<iree_hal_buffer_t> source;
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        source_data.data(), source_data.size(), source.out()));
    Ref<iree_hal_buffer_t> target;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(target_offset + length + padding,
                                            target.out()));
    if (indirect) {
      const iree_hal_buffer_binding_t bindings[] = {
          {source, table_offset, 7 + length},
          {target, table_offset, 11 + length},
      };
      IREE_ASSERT_OK(
          SubmitCommandBufferAndWait(reusable_command_buffer, {2, bindings}));
    } else {
      Ref<iree_hal_command_buffer_t> command_buffer;
      IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                         IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0,
                                         command_buffer.out()));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
      IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer_linear(
          command_buffer,
          MakeRef(source, kSourceBufferSlot, 7, IREE_HAL_WHOLE_BUFFER),
          MakeRef(target, kTargetBufferSlot, 11, IREE_HAL_WHOLE_BUFFER),
          IREE_HAL_COPY_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
      IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
    }
    std::vector<uint8_t> expected(target_offset + length + padding, 0);
    std::memcpy(expected.data() + target_offset,
                source_data.data() + source_offset, length);
    EXPECT_THAT(ReadBufferBytes(target, 0, expected.size()),
                ContainerEq(expected));
  }
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferCopyBufferTest);

}  // namespace iree::hal::cts
