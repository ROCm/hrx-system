// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/buffer.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string FormatMemoryType(iree_hal_memory_type_t memory_type) {
  iree_bitfield_string_temp_t temporary;
  const iree_string_view_t value =
      iree_hal_memory_type_format(memory_type, &temporary);
  return std::string(value.data, value.size);
}

TEST(MemoryTypeTest, EncodesLocalityIndependentlyFromCoherence) {
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_HOST_LOCAL, 0x42u);
  EXPECT_EQ(
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      0x46u);
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            0x72u);
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            0x76u);
}

TEST(MemoryTypeTest, RoundTripsOrthogonalLocalityAndCoherence) {
  static const struct {
    iree_hal_memory_type_t memory_type;
    const char* value;
  } cases[] = {
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL, "HOST_LOCAL"},
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
       "HOST_LOCAL|HOST_COHERENT"},
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
       "HOST_LOCAL|DEVICE_LOCAL"},
      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
       "HOST_LOCAL|DEVICE_LOCAL|HOST_COHERENT"},
  };
  for (const auto& test_case : cases) {
    EXPECT_EQ(FormatMemoryType(test_case.memory_type), test_case.value);
    iree_hal_memory_type_t parsed_memory_type = 0;
    IREE_EXPECT_OK(iree_hal_memory_type_parse(
        iree_make_cstring_view(test_case.value), &parsed_memory_type));
    EXPECT_EQ(parsed_memory_type, test_case.memory_type);
  }
}

TEST(BufferRangeTest, AcceptsContainedRanges) {
  iree_hal_buffer_t buffer = {};
  buffer.byte_length = 16;

  IREE_EXPECT_OK(iree_hal_buffer_validate_range(&buffer, 0, 16));
  IREE_EXPECT_OK(iree_hal_buffer_validate_range(&buffer, 7, 9));
  IREE_EXPECT_OK(iree_hal_buffer_validate_range(&buffer, 16, 0));
}

TEST(BufferRangeTest, RejectsOutOfRangeAndOverflowingRanges) {
  iree_hal_buffer_t buffer = {};
  buffer.byte_length = 16;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_buffer_validate_range(&buffer, 17, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_buffer_validate_range(&buffer, 15, 2));

  buffer.byte_length = IREE_DEVICE_SIZE_MAX;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_buffer_validate_range(&buffer, IREE_DEVICE_SIZE_MAX - 3, 8));
}

TEST(BufferPermissionTest, ValidatesAccessAndUsage) {
  IREE_EXPECT_OK(iree_hal_buffer_validate_access(IREE_HAL_MEMORY_ACCESS_READ,
                                                 IREE_HAL_MEMORY_ACCESS_READ));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_access(IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_MEMORY_ACCESS_WRITE));

  IREE_EXPECT_OK(
      iree_hal_buffer_validate_usage(IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
                                     IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_usage(IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
                                     IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
}

}  // namespace
