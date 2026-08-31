// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include "hrx_test_fixture.hpp"

namespace {

static void IgnoreCleanupStatus(hrx_status_t status) {
  if (!hrx_status_is_ok(status)) hrx().status_ignore(status);
}

class VirtualMemoryResources {
 public:
  explicit VirtualMemoryResources(hrx_allocator_t allocator)
      : allocator(allocator) {}

  ~VirtualMemoryResources() {
    if (is_mapped) {
      IgnoreCleanupStatus(hrx().allocator_virtual_memory_unmap(
          allocator, virtual_buffer, /*virtual_offset=*/0, size));
    }
    if (physical_memory) {
      IgnoreCleanupStatus(
          hrx().allocator_physical_memory_free(allocator, physical_memory));
    }
    if (virtual_buffer) {
      IgnoreCleanupStatus(
          hrx().allocator_virtual_memory_release(allocator, virtual_buffer));
    }
  }

  // Unowned allocator that must outlive all resources in this helper.
  hrx_allocator_t allocator = nullptr;
  // Owned virtual address reservation released by the destructor.
  hrx_buffer_t virtual_buffer = nullptr;
  // Owned physical allocation freed by the destructor.
  hrx_physical_memory_t physical_memory = nullptr;
  // Size in bytes of the reservation, allocation, and mapping.
  size_t size = 0;
  // Whether the physical allocation is currently mapped.
  bool is_mapped = false;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "query virtual memory support",
                 "[virtual_memory][query]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  bool supported = false;
  size_t min_page = 0, rec_page = 0;
  REQUIRE_OK(hrx().allocator_query_virtual_memory(
      alloc, HRX_MEMORY_TYPE_DEVICE_LOCAL, &supported, &min_page, &rec_page));

  // The task driver does not support VM — just verify the query doesn't crash
  // and returns consistent values.
  if (supported) {
    REQUIRE(min_page > 0);
    REQUIRE(rec_page >= min_page);
  } else {
    REQUIRE(min_page == 0);
    REQUIRE(rec_page == 0);
  }
}

TEST_CASE_METHOD(HrxTestFixture, "virtual memory lifecycle",
                 "[virtual_memory][lifecycle]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  bool supported = false;
  size_t min_page = 0, rec_page = 0;
  REQUIRE_OK(hrx().allocator_query_virtual_memory(
      alloc, HRX_MEMORY_TYPE_DEVICE_LOCAL, &supported, &min_page, &rec_page));

  if (!supported) {
    hrx_buffer_t vbuf = nullptr;
    hrx_status_t s =
        hrx().allocator_virtual_memory_reserve(alloc, 0, 1024 * 1024, &vbuf);
    REQUIRE(!hrx_status_is_ok(s));
    hrx().status_ignore(s);
    return;
  }

  REQUIRE(min_page > 0);
  REQUIRE(rec_page >= min_page);
  VirtualMemoryResources resources(alloc);
  resources.size = rec_page;

  REQUIRE_OK(hrx().allocator_virtual_memory_reserve(
      alloc, /*affinity=*/0, resources.size, &resources.virtual_buffer));
  REQUIRE(resources.virtual_buffer != nullptr);
  REQUIRE_OK(hrx().allocator_physical_memory_allocate(
      alloc, HRX_MEMORY_TYPE_DEVICE_LOCAL, resources.size,
      &resources.physical_memory));
  REQUIRE(resources.physical_memory != nullptr);
  REQUIRE_OK(hrx().allocator_virtual_memory_map(
      alloc, resources.virtual_buffer, /*virtual_offset=*/0,
      resources.physical_memory, /*physical_offset=*/0, resources.size));
  resources.is_mapped = true;
  REQUIRE_OK(hrx().allocator_virtual_memory_protect(
      alloc, resources.virtual_buffer, /*virtual_offset=*/0, resources.size,
      HRX_MEMORY_PROTECTION_READ_WRITE));

  REQUIRE_OK(hrx().allocator_virtual_memory_unmap(
      alloc, resources.virtual_buffer, /*virtual_offset=*/0, resources.size));
  resources.is_mapped = false;
  REQUIRE_OK(
      hrx().allocator_physical_memory_free(alloc, resources.physical_memory));
  resources.physical_memory = nullptr;
  REQUIRE_OK(
      hrx().allocator_virtual_memory_release(alloc, resources.virtual_buffer));
  resources.virtual_buffer = nullptr;
}
