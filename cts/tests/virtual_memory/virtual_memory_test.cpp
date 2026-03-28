// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(PyreTestFixture, "query virtual memory support",
                 "[virtual_memory][query]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  bool supported = false;
  size_t min_page = 0, rec_page = 0;
  REQUIRE_OK(pyre().allocator_query_virtual_memory(
      alloc, PYRE_MEMORY_TYPE_DEVICE_LOCAL, &supported, &min_page, &rec_page));

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

TEST_CASE_METHOD(PyreTestFixture, "reserve fails gracefully without VM support",
                 "[virtual_memory][reserve]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  bool supported = false;
  size_t min_page = 0, rec_page = 0;
  REQUIRE_OK(pyre().allocator_query_virtual_memory(
      alloc, PYRE_MEMORY_TYPE_DEVICE_LOCAL, &supported, &min_page, &rec_page));

  if (!supported) {
    // Reserve should fail with UNAVAILABLE.
    pyre_buffer_t vbuf = nullptr;
    pyre_status_t s = pyre().allocator_virtual_memory_reserve(
        alloc, 0, 1024 * 1024, &vbuf);
    REQUIRE(!pyre_status_is_ok(s));
    pyre().status_ignore(s);
  }
  // If VM is supported, a full reserve/map/unmap/release cycle would go here.
}
