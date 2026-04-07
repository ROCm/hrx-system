// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PyreTestFixture, "Mem pool create and release",
                 "[mem_pool][lifecycle]") {
  pyre_mem_pool_props_t props = {};
  props.location_type = 1;  // device
  props.location_id = 0;

  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));
  REQUIRE(pool != nullptr);

  pyre().mem_pool_release(pool);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool retain and release",
                 "[mem_pool][lifecycle]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  pyre().mem_pool_retain(pool);
  pyre().mem_pool_release(pool);
  pyre().mem_pool_release(pool);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool create with NULL device fails",
                 "[mem_pool][lifecycle]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  pyre_status_t status = pyre().mem_pool_create(nullptr, &props, &pool);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool create with NULL props fails",
                 "[mem_pool][lifecycle]") {
  pyre_mem_pool_t pool = nullptr;
  pyre_status_t status = pyre().mem_pool_create(device_, nullptr, &pool);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Default attribute values
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PyreTestFixture, "Mem pool default attributes",
                 "[mem_pool][attr]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  uint64_t val = UINT64_MAX;

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 1);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_RESERVED_MEM_HIGH, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_USED_MEM_CURRENT, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_USED_MEM_HIGH, &val));
  REQUIRE(val == 0);

  pyre().mem_pool_release(pool);
}

//===----------------------------------------------------------------------===//
// Attribute set and get
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PyreTestFixture, "Mem pool set release threshold",
                 "[mem_pool][attr]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, 1024 * 1024));

  uint64_t val = 0;
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == 1024 * 1024);

  pyre().mem_pool_release(pool);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool set reuse flags",
                 "[mem_pool][attr]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  // Toggle follow event dependencies off.
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 0));
  uint64_t val = 1;
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 0);

  // Toggle allow internal dependencies on.
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, 1));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &val));
  REQUIRE(val == 1);

  // Toggle allow opportunistic on.
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, 1));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &val));
  REQUIRE(val == 1);

  pyre().mem_pool_release(pool);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool set read-only attribute fails",
                 "[mem_pool][attr]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  // Stats are read-only; setting them should fail.
  pyre_status_t status = pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, 999);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);

  status = pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_USED_MEM_HIGH, 999);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);

  pyre().mem_pool_release(pool);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool get with NULL pool fails",
                 "[mem_pool][attr]") {
  uint64_t val = 0;
  pyre_status_t status = pyre().mem_pool_get_attribute(
      nullptr, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool set with NULL pool fails",
                 "[mem_pool][attr]") {
  pyre_status_t status = pyre().mem_pool_set_attribute(
      nullptr, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Trim
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PyreTestFixture, "Mem pool trim succeeds",
                 "[mem_pool][trim]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  REQUIRE_OK(pyre().mem_pool_trim(pool, 0));
  REQUIRE_OK(pyre().mem_pool_trim(pool, 1024 * 1024));

  pyre().mem_pool_release(pool);
}

TEST_CASE_METHOD(PyreTestFixture, "Mem pool trim with NULL pool fails",
                 "[mem_pool][trim]") {
  pyre_status_t status = pyre().mem_pool_trim(nullptr, 0);
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Multiple pools
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PyreTestFixture, "Multiple pools are independent",
                 "[mem_pool][lifecycle]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool_a = nullptr;
  pyre_mem_pool_t pool_b = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool_a));
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool_b));

  // Set different thresholds.
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool_a, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, 100));
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool_b, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, 200));

  uint64_t val_a = 0, val_b = 0;
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool_a, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val_a));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool_b, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val_b));
  REQUIRE(val_a == 100);
  REQUIRE(val_b == 200);

  pyre().mem_pool_release(pool_b);
  pyre().mem_pool_release(pool_a);
}

//===----------------------------------------------------------------------===//
// Attribute toggle round-trip
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(PyreTestFixture, "Mem pool attribute toggle round-trip",
                 "[mem_pool][attr]") {
  pyre_mem_pool_props_t props = {};
  pyre_mem_pool_t pool = nullptr;
  REQUIRE_OK(pyre().mem_pool_create(device_, &props, &pool));

  uint64_t val = 0;

  // Default is true; toggle to false, then back.
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 0));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 1));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 1);

  // Release threshold round-trip with large value.
  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, UINT64_MAX));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == UINT64_MAX);

  REQUIRE_OK(pyre().mem_pool_set_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0));
  REQUIRE_OK(pyre().mem_pool_get_attribute(
      pool, PYRE_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == 0);

  pyre().mem_pool_release(pool);
}
