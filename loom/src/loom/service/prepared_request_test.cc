// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/prepared_request.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(PreparedRequestsTest, PacksStableRangesAndOwnsBytes) {
  const auto allocator = iree_allocator_system();
  std::array<loom_service_prepared_request_t, 3> requests = {};
  const std::array<iree_string_view_t, 3> inputs = {
      IREE_SV("first"), IREE_SV(""), IREE_SV("last")};
  for (size_t i = 0; i < requests.size(); ++i) {
    IREE_ASSERT_OK(iree_vm_buffer_create(inputs[i].size, 8, allocator,
                                         &requests[i].bytes));
    iree_byte_span_t bytes;
    IREE_ASSERT_OK(
        iree_vm_buffer_map_write(requests[i].bytes, 0, inputs[i].size, &bytes));
    if (bytes.data_length) {
      memcpy(bytes.data, inputs[i].data, bytes.data_length);
    }
  }
  iree_vm_buffer_t* ranges = nullptr;
  iree_vm_buffer_t* packed = nullptr;
  auto status = loom_service_prepared_requests_pack(
      requests.size(), requests.data(), allocator, &ranges, &packed);
  for (const auto& request : requests) {
    iree_vm_buffer_release(request.bytes);
  }
  IREE_ASSERT_OK(status);
  iree_const_byte_span_t records;
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      ranges, 0, iree_vm_buffer_length(ranges), &records));
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      packed, 0, iree_vm_buffer_length(packed), &bytes));
  EXPECT_EQ(records.data_length, 3 * 16u);
  EXPECT_EQ(bytes.data_length, 3 * 16u + 9u);
  EXPECT_EQ(memcmp(bytes.data, records.data, records.data_length), 0);
  EXPECT_EQ(memcmp(bytes.data + records.data_length, "firstlast", 9), 0);
  const uint64_t expected[] = {48, 5, 53, 0, 53, 4};
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_EQ(iree_unaligned_load_le_u64(records.data + i * 8), expected[i]);
  }
  iree_vm_buffer_release(ranges);
  iree_vm_buffer_release(packed);
}

}  // namespace
