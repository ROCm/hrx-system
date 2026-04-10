// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(HrxTestFixture, "device_allocator returns non-null",
                 "[allocator]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);
  REQUIRE(alloc != nullptr);
}

TEST_CASE_METHOD(HrxTestFixture, "device_allocator is idempotent",
                 "[allocator]") {
  hrx_allocator_t a1 = hrx().device_allocator(device_);
  hrx_allocator_t a2 = hrx().device_allocator(device_);
  REQUIRE(a1 == a2);
}

TEST_CASE_METHOD(HrxTestFixture, "allocator_allocate_buffer device local",
                 "[allocator][alloc]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_DEVICE_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 4096, &buf));
  REQUIRE(buf != nullptr);

  size_t size = 0;
  REQUIRE_OK(hrx().buffer_get_size(buf, &size));
  REQUIRE(size == 4096);

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "allocator_allocate_buffer host mappable",
                 "[allocator][alloc]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, 256, &buf));

  // Should be mappable.
  void *ptr = nullptr;
  REQUIRE_OK(hrx().buffer_map(buf, HRX_MAP_WRITE, 0, 256, &ptr));
  REQUIRE(ptr != nullptr);
  memset(ptr, 0xCD, 256);
  REQUIRE_OK(hrx().buffer_unmap(buf));

  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "allocator_import_buffer from host ptr",
                 "[allocator][import]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  // Heap allocator requires 64-byte alignment.
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *host_raw = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc_aligned(ha, 128, 64, 0, &host_raw));
  uint8_t *host_data = (uint8_t *)host_raw;
  memset(host_data, 0xAA, 128);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED;

  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(
      hrx().allocator_import_buffer(alloc, params, host_data, 128, &buf));
  REQUIRE(buf != nullptr);

  size_t size = 0;
  REQUIRE_OK(hrx().buffer_get_size(buf, &size));
  REQUIRE(size == 128);

  hrx().buffer_release(buf);
  hrx().host_allocator_free_aligned(ha, host_raw);
}

TEST_CASE_METHOD(HrxTestFixture, "allocator retain/release",
                 "[allocator][refcount]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);
  // Retain then release — should not crash.
  hrx().allocator_retain(alloc);
  hrx().allocator_release(alloc);
}
