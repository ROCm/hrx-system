// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/inspection.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct Inventory {
  static bool RequestAt(void* self, iree_host_size_t index,
                        loom_serve_http_request_snapshot_t* out_snapshot) {
    auto* inventory = static_cast<Inventory*>(self);
    ++inventory->queries;
    if (index >= 66) {
      return false;
    }
    *out_snapshot = {.id = index == 64 ? 9u : 0u,
                     .phase = "queued",
                     .arrival_time = 1000000000,
                     .counters = {.requests = 1}};
    return true;
  }

  static bool SessionAt(void* self, iree_host_size_t index,
                        loom_serve_http_session_snapshot_t* out_snapshot) {
    auto* inventory = static_cast<Inventory*>(self);
    ++inventory->queries;
    if (index) {
      return false;
    }
    *out_snapshot = {.id = 3,
                     .name = IREE_SV("a\"b"),
                     .storage = "cache",
                     .position = 7,
                     .resident_tokens = 8,
                     .counters = {.requests = 2, .completed = 2}};
    return true;
  }

  // Number of owner slots examined, independent of how many were occupied.
  size_t queries = 0;
};

TEST(InspectionTest, RejectsSignsEmptyAndOverflowOffsets) {
  iree_host_size_t offset = 0;
  for (const char* value :
       {"", "-1", "+1", " 1", "1x", "1:2", "18446744073709551616"}) {
    EXPECT_FALSE(loom_serve_http_inspection_offset(
        iree_make_cstring_view(value), &offset));
  }
  EXPECT_TRUE(loom_serve_http_inspection_offset(IREE_SV("00123"), &offset));
  EXPECT_EQ(offset, 123u);
  const auto maximum = std::to_string(SIZE_MAX);
  EXPECT_TRUE(loom_serve_http_inspection_offset(
      iree_make_cstring_view(maximum.c_str()), &offset));
  EXPECT_EQ(offset, SIZE_MAX);
}

TEST(InspectionTest, BoundsEmptySlotsAndFormatsPublishedMetadata) {
  Inventory inventory;
  const loom_serve_http_inspection_t inspection = {
      .request_at = Inventory::RequestAt, .session_at = Inventory::SessionAt};
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  const auto view = [&] {
    const auto value = iree_string_builder_view(&output);
    return std::string(value.data, value.size);
  };
  IREE_ASSERT_OK(loom_serve_http_inspection_format(
      &inspection, &inventory, LOOM_SERVE_HTTP_INSPECT_REQUESTS, 0, 2000000000,
      &output));
  EXPECT_EQ(inventory.queries, 64u);
  EXPECT_EQ(view(), R"(,"data":[],"next_offset":64)");
  iree_string_builder_reset(&output);
  IREE_ASSERT_OK(loom_serve_http_inspection_format(
      &inspection, &inventory, LOOM_SERVE_HTTP_INSPECT_REQUESTS, 64, 2000000000,
      &output));
  EXPECT_EQ(inventory.queries, 67u);
  EXPECT_EQ(
      view(),
      R"(,"data":[{"id":9,"phase":"queued","session":null,"age_seconds":1.000000000,"counters":{"requests":1,"completed":0,"rejected":0,"cancelled":0,"failed":0,"input_tokens":0,"output_tokens":0,"batches":0}}],"next_offset":null)");
  iree_string_builder_reset(&output);
  IREE_ASSERT_OK(loom_serve_http_inspection_format(
      &inspection, &inventory, LOOM_SERVE_HTTP_INSPECT_REQUESTS, SIZE_MAX,
      2000000000, &output));
  EXPECT_EQ(view(), R"(,"data":[],"next_offset":null)");
  iree_string_builder_reset(&output);
  IREE_ASSERT_OK(loom_serve_http_inspection_format(
      &inspection, &inventory, LOOM_SERVE_HTTP_INSPECT_SESSIONS, 0, 2000000000,
      &output));
  EXPECT_EQ(
      view(),
      R"(,"data":[{"id":3,"name":"a\"b","request":null,"storage":"cache","position":7,"resident_tokens":8,"reserved_tokens":0,"counters":{"requests":2,"completed":2,"rejected":0,"cancelled":0,"failed":0,"input_tokens":0,"output_tokens":0,"batches":0}}],"next_offset":null)");
  iree_string_builder_deinitialize(&output);
}

TEST(InspectionTest, FormattingFailureStopsQuerying) {
  Inventory inventory;
  const loom_serve_http_inspection_t inspection = {.session_at =
                                                       Inventory::SessionAt};
  char bytes[16];
  iree_string_builder_t output;
  iree_string_builder_initialize_with_storage(bytes, sizeof(bytes), &output);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_http_inspection_format(
                            &inspection, &inventory,
                            LOOM_SERVE_HTTP_INSPECT_SESSIONS, 0, 0, &output));
  EXPECT_EQ(inventory.queries, 1u);
  iree_string_builder_deinitialize(&output);
}

}  // namespace
