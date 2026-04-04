// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(PyreTestFixture, "device_allocator returns non-null",
                 "[allocator]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);
  REQUIRE(alloc != nullptr);
}

TEST_CASE_METHOD(PyreTestFixture, "device_allocator is idempotent",
                 "[allocator]") {
  pyre_allocator_t a1 = pyre().device_allocator(device_);
  pyre_allocator_t a2 = pyre().device_allocator(device_);
  REQUIRE(a1 == a2);
}

TEST_CASE_METHOD(PyreTestFixture, "allocator_allocate_buffer device local",
                 "[allocator][alloc]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_DEVICE_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_allocate_buffer(alloc, params, 4096, &buf));
  REQUIRE(buf != nullptr);

  size_t size = 0;
  REQUIRE_OK(pyre().buffer_get_size(buf, &size));
  REQUIRE(size == 4096);

  pyre().buffer_release(buf);
}

TEST_CASE_METHOD(PyreTestFixture, "allocator_allocate_buffer host mappable",
                 "[allocator][alloc]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_allocate_buffer(alloc, params, 256, &buf));

  // Should be mappable.
  void* ptr = nullptr;
  REQUIRE_OK(pyre().buffer_map(buf, PYRE_MAP_WRITE, 0, 256, &ptr));
  REQUIRE(ptr != nullptr);
  memset(ptr, 0xCD, 256);
  REQUIRE_OK(pyre().buffer_unmap(buf));

  pyre().buffer_release(buf);
}

TEST_CASE_METHOD(PyreTestFixture, "allocator_import_buffer from host ptr",
                 "[allocator][import]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);

  // Heap allocator requires 64-byte alignment.
  pyre_host_allocator_t ha = pyre().host_allocator_system();
  void* host_raw = nullptr;
  REQUIRE_OK(pyre().host_allocator_malloc_aligned(ha, 128, 64, 0, &host_raw));
  uint8_t* host_data = (uint8_t*)host_raw;
  memset(host_data, 0xAA, 128);

  pyre_buffer_params_t params = {};
  params.type = PYRE_MEMORY_TYPE_HOST_LOCAL | PYRE_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = PYRE_BUFFER_USAGE_DEFAULT | PYRE_BUFFER_USAGE_MAPPING_SCOPED;

  pyre_buffer_t buf = nullptr;
  REQUIRE_OK(pyre().allocator_import_buffer(alloc, params, host_data,
                                            128, &buf));
  REQUIRE(buf != nullptr);

  size_t size = 0;
  REQUIRE_OK(pyre().buffer_get_size(buf, &size));
  REQUIRE(size == 128);

  pyre().buffer_release(buf);
  pyre().host_allocator_free_aligned(ha, host_raw);
}

TEST_CASE_METHOD(PyreTestFixture, "allocator retain/release",
                 "[allocator][refcount]") {
  pyre_allocator_t alloc = pyre().device_allocator(device_);
  // Retain then release — should not crash.
  pyre().allocator_retain(alloc);
  pyre().allocator_release(alloc);
}
