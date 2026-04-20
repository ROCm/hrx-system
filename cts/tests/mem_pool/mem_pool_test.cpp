// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(HRXTestFixture, "Mem pool create and release",
                 "[mem_pool][lifecycle]") {
  hrx_mem_pool_props_t props = {};
  props.location_type = 1;  // device
  props.location_id = 0;

  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));
  REQUIRE(pool != nullptr);

  hrx().mem_pool_release(pool);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool retain and release",
                 "[mem_pool][lifecycle]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  hrx().mem_pool_retain(pool);
  hrx().mem_pool_release(pool);
  hrx().mem_pool_release(pool);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool create with NULL device fails",
                 "[mem_pool][lifecycle]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  hrx_status_t status = hrx().mem_pool_create(nullptr, &props, &pool);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool create with NULL props fails",
                 "[mem_pool][lifecycle]") {
  hrx_mem_pool_t pool = nullptr;
  hrx_status_t status = hrx().mem_pool_create(device_, nullptr, &pool);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Default attribute values
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(HRXTestFixture, "Mem pool default attributes",
                 "[mem_pool][attr]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  uint64_t val = UINT64_MAX;

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 1);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_HIGH, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_USED_MEM_CURRENT, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_USED_MEM_HIGH, &val));
  REQUIRE(val == 0);

  hrx().mem_pool_release(pool);
}

//===----------------------------------------------------------------------===//
// Attribute set and get
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(HRXTestFixture, "Mem pool set release threshold",
                 "[mem_pool][attr]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 1024 * 1024));

  uint64_t val = 0;
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == 1024 * 1024);

  hrx().mem_pool_release(pool);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool set reuse flags",
                 "[mem_pool][attr]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  // Toggle follow event dependencies off.
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 0));
  uint64_t val = 1;
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 0);

  // Toggle allow internal dependencies on.
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, 1));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &val));
  REQUIRE(val == 1);

  // Toggle allow opportunistic on.
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, 1));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &val));
  REQUIRE(val == 1);

  hrx().mem_pool_release(pool);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool set read-only attribute fails",
                 "[mem_pool][attr]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  // Stats are read-only; setting them should fail.
  hrx_status_t status = hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, 999);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);

  status = hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_USED_MEM_HIGH, 999);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);

  hrx().mem_pool_release(pool);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool get with NULL pool fails",
                 "[mem_pool][attr]") {
  uint64_t val = 0;
  hrx_status_t status = hrx().mem_pool_get_attribute(
      nullptr, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool set with NULL pool fails",
                 "[mem_pool][attr]") {
  hrx_status_t status = hrx().mem_pool_set_attribute(
      nullptr, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Trim
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(HRXTestFixture, "Mem pool trim succeeds",
                 "[mem_pool][trim]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  REQUIRE_OK(hrx().mem_pool_trim(pool, 0));
  REQUIRE_OK(hrx().mem_pool_trim(pool, 1024 * 1024));

  hrx().mem_pool_release(pool);
}

TEST_CASE_METHOD(HRXTestFixture, "Mem pool trim with NULL pool fails",
                 "[mem_pool][trim]") {
  hrx_status_t status = hrx().mem_pool_trim(nullptr, 0);
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Multiple pools
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(HRXTestFixture, "Multiple pools are independent",
                 "[mem_pool][lifecycle]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool_a = nullptr;
  hrx_mem_pool_t pool_b = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool_a));
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool_b));

  // Set different thresholds.
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool_a, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 100));
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool_b, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 200));

  uint64_t val_a = 0, val_b = 0;
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool_a, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val_a));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool_b, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val_b));
  REQUIRE(val_a == 100);
  REQUIRE(val_b == 200);

  hrx().mem_pool_release(pool_b);
  hrx().mem_pool_release(pool_a);
}

//===----------------------------------------------------------------------===//
// Attribute toggle round-trip
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(HRXTestFixture, "Mem pool attribute toggle round-trip",
                 "[mem_pool][attr]") {
  hrx_mem_pool_props_t props = {};
  hrx_mem_pool_t pool = nullptr;
  REQUIRE_OK(hrx().mem_pool_create(device_, &props, &pool));

  uint64_t val = 0;

  // Default is true; toggle to false, then back.
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 0));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 0);

  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, 1));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &val));
  REQUIRE(val == 1);

  // Release threshold round-trip with large value.
  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, UINT64_MAX));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == UINT64_MAX);

  REQUIRE_OK(hrx().mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0));
  REQUIRE_OK(hrx().mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, &val));
  REQUIRE(val == 0);

  hrx().mem_pool_release(pool);
}
