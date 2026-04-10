// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE("OK status is null", "[status]") {
  hrx_status_t ok = hrx_ok_status();
  REQUIRE(hrx_status_is_ok(ok));
  REQUIRE(ok == nullptr);
}

TEST_CASE("Make and inspect error status", "[status]") {
  hrx_status_t err = hrx().make_status(HRX_STATUS_NOT_FOUND, "test msg");
  REQUIRE(!hrx_status_is_ok(err));
  REQUIRE(hrx().status_code(err) == HRX_STATUS_NOT_FOUND);

  char *msg = nullptr;
  size_t len = 0;
  REQUIRE_OK(hrx().status_to_string(err, &msg, &len));
  REQUIRE(msg != nullptr);
  REQUIRE(std::strcmp(msg, "test msg") == 0);
  hrx().status_free_message(msg);
  hrx().status_ignore(err);
}

TEST_CASE("Status ignore frees resources", "[status]") {
  hrx_status_t err = hrx().make_status(HRX_STATUS_INTERNAL, "to ignore");
  hrx().status_ignore(err);
  // No crash = pass.
}

TEST_CASE("OK status to_string", "[status]") {
  char *msg = nullptr;
  size_t len = 0;
  REQUIRE_OK(hrx().status_to_string(hrx_ok_status(), &msg, &len));
  REQUIRE(msg != nullptr);
  REQUIRE(std::strcmp(msg, "OK") == 0);
  hrx().status_free_message(msg);
}

TEST_CASE("All status codes are distinct", "[status]") {
  REQUIRE(HRX_STATUS_OK != HRX_STATUS_INVALID_ARGUMENT);
  REQUIRE(HRX_STATUS_INVALID_ARGUMENT != HRX_STATUS_NOT_FOUND);
  REQUIRE(HRX_STATUS_NOT_FOUND != HRX_STATUS_OUT_OF_RANGE);
  REQUIRE(HRX_STATUS_OUT_OF_RANGE != HRX_STATUS_UNIMPLEMENTED);
  REQUIRE(HRX_STATUS_UNIMPLEMENTED != HRX_STATUS_INTERNAL);
  REQUIRE(HRX_STATUS_INTERNAL != HRX_STATUS_UNAVAILABLE);
  REQUIRE(HRX_STATUS_UNAVAILABLE != HRX_STATUS_OUT_OF_MEMORY);
  REQUIRE(HRX_STATUS_OUT_OF_MEMORY != HRX_STATUS_DEADLINE_EXCEEDED);
}
