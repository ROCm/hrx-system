// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE("OK status is null", "[status]") {
  pyre_status_t ok = pyre_ok_status();
  REQUIRE(pyre_status_is_ok(ok));
  REQUIRE(ok == nullptr);
}

TEST_CASE("Make and inspect error status", "[status]") {
  pyre_status_t err = pyre().make_status(PYRE_STATUS_NOT_FOUND, "test msg");
  REQUIRE(!pyre_status_is_ok(err));
  REQUIRE(pyre().status_code(err) == PYRE_STATUS_NOT_FOUND);

  char* msg = nullptr;
  size_t len = 0;
  REQUIRE_OK(pyre().status_to_string(err, &msg, &len));
  REQUIRE(msg != nullptr);
  REQUIRE(std::strcmp(msg, "test msg") == 0);
  pyre().status_free_message(msg);
  pyre().status_ignore(err);
}

TEST_CASE("Status ignore frees resources", "[status]") {
  pyre_status_t err = pyre().make_status(PYRE_STATUS_INTERNAL, "to ignore");
  pyre().status_ignore(err);
  // No crash = pass.
}

TEST_CASE("OK status to_string", "[status]") {
  char* msg = nullptr;
  size_t len = 0;
  REQUIRE_OK(pyre().status_to_string(pyre_ok_status(), &msg, &len));
  REQUIRE(msg != nullptr);
  REQUIRE(std::strcmp(msg, "OK") == 0);
  pyre().status_free_message(msg);
}

TEST_CASE("All status codes are distinct", "[status]") {
  REQUIRE(PYRE_STATUS_OK != PYRE_STATUS_INVALID_ARGUMENT);
  REQUIRE(PYRE_STATUS_INVALID_ARGUMENT != PYRE_STATUS_NOT_FOUND);
  REQUIRE(PYRE_STATUS_NOT_FOUND != PYRE_STATUS_OUT_OF_RANGE);
  REQUIRE(PYRE_STATUS_OUT_OF_RANGE != PYRE_STATUS_UNIMPLEMENTED);
  REQUIRE(PYRE_STATUS_UNIMPLEMENTED != PYRE_STATUS_INTERNAL);
  REQUIRE(PYRE_STATUS_INTERNAL != PYRE_STATUS_UNAVAILABLE);
  REQUIRE(PYRE_STATUS_UNAVAILABLE != PYRE_STATUS_OUT_OF_MEMORY);
  REQUIRE(PYRE_STATUS_OUT_OF_MEMORY != PYRE_STATUS_DEADLINE_EXCEEDED);
}
