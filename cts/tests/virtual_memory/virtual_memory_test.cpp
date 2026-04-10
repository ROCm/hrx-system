// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(HrxTestFixture, "query virtual memory support",
                 "[virtual_memory][query]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  bool supported = false;
  size_t min_page = 0, rec_page = 0;
  REQUIRE_OK(hrx().allocator_query_virtual_memory(
      alloc, HRX_MEMORY_TYPE_DEVICE_LOCAL, &supported, &min_page, &rec_page));

  // local-task does not support VM — just verify the query doesn't crash
  // and returns consistent values.
  if (supported) {
    REQUIRE(min_page > 0);
    REQUIRE(rec_page >= min_page);
  } else {
    REQUIRE(min_page == 0);
    REQUIRE(rec_page == 0);
  }
}

TEST_CASE_METHOD(HrxTestFixture, "reserve fails gracefully without VM support",
                 "[virtual_memory][reserve]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  bool supported = false;
  size_t min_page = 0, rec_page = 0;
  REQUIRE_OK(hrx().allocator_query_virtual_memory(
      alloc, HRX_MEMORY_TYPE_DEVICE_LOCAL, &supported, &min_page, &rec_page));

  if (!supported) {
    // Reserve should fail with UNAVAILABLE.
    hrx_buffer_t vbuf = nullptr;
    hrx_status_t s =
        hrx().allocator_virtual_memory_reserve(alloc, 0, 1024 * 1024, &vbuf);
    REQUIRE(!hrx_status_is_ok(s));
    hrx().status_ignore(s);
  }
  // If VM is supported, a full reserve/map/unmap/release cycle would go here.
}
