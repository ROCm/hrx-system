// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include <type_traits>

#include "iree/testing/gtest.h"
#include "loomc/iree.h"

namespace {

static_assert(std::is_same_v<loomc_status_t, iree_status_t>);
static_assert(
    std::is_same_v<loomc_allocator_command_t, iree_allocator_command_t>);
static_assert(
    std::is_same_v<loomc_allocator_ctl_fn_t, iree_allocator_ctl_fn_t>);

TEST(IreeAdapterTest, StringViewRoundTripsWithoutCopying) {
  iree_string_view_t iree_value = IREE_SV("some text");
  loomc_string_view_t loomc_value = loomc_string_view_from_iree(iree_value);
  EXPECT_EQ(loomc_value.data, iree_value.data);
  EXPECT_EQ(loomc_value.size, iree_value.size);

  iree_string_view_t round_trip = iree_string_view_from_loomc(loomc_value);
  EXPECT_EQ(round_trip.data, iree_value.data);
  EXPECT_EQ(round_trip.size, iree_value.size);
}

TEST(IreeAdapterTest, ByteSpanRoundTripsWithoutCopying) {
  const uint8_t data[] = {1, 2, 3, 4};
  iree_const_byte_span_t iree_value = iree_make_const_byte_span(data, 4);
  loomc_byte_span_t loomc_value = loomc_byte_span_from_iree(iree_value);
  EXPECT_EQ(loomc_value.data, iree_value.data);
  EXPECT_EQ(loomc_value.data_length, iree_value.data_length);

  iree_const_byte_span_t round_trip =
      iree_const_byte_span_from_loomc(loomc_value);
  EXPECT_EQ(round_trip.data, iree_value.data);
  EXPECT_EQ(round_trip.data_length, iree_value.data_length);
}

TEST(IreeAdapterTest, IreeAllocatorServicesLoomCalls) {
  iree_allocator_t iree_allocator = iree_allocator_system();
  loomc_allocator_t loomc_allocator = loomc_allocator_from_iree(iree_allocator);
  ASSERT_TRUE(loomc_allocator_is_valid(loomc_allocator));

  void* data = NULL;
  loomc_status_t status = loomc_allocator_malloc(loomc_allocator, 32, &data);
  loomc_status_code_t status_code = loomc_status_code(status);
  loomc_status_free(status);
  ASSERT_EQ(status_code, LOOMC_STATUS_OK);
  ASSERT_NE(data, nullptr);
  memset(data, 0x5A, 32);
  loomc_allocator_free(loomc_allocator, data);

  iree_allocator_t round_trip = iree_allocator_from_loomc(loomc_allocator);
  EXPECT_EQ(round_trip.self, iree_allocator.self);
  EXPECT_EQ(round_trip.ctl, iree_allocator.ctl);
}

TEST(IreeAdapterTest, LoomAllocatorServicesIreeCalls) {
  loomc_allocator_t loomc_allocator = loomc_allocator_system();
  iree_allocator_t iree_allocator = iree_allocator_from_loomc(loomc_allocator);

  void* data = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator, 32, &data);
  bool status_ok = iree_status_is_ok(status);
  iree_status_free(status);
  ASSERT_TRUE(status_ok);
  ASSERT_NE(data, nullptr);
  memset(data, 0xA5, 32);
  iree_allocator_free(iree_allocator, data);
}

}  // namespace
