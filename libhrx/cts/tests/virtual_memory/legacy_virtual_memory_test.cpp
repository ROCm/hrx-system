// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "hrx_test_fixture.hpp"

namespace {

class LegacyVirtualMemoryResources {
 public:
  explicit LegacyVirtualMemoryResources(hrx_allocator_t allocator)
      : allocator(allocator) {}

  ~LegacyVirtualMemoryResources() {
    if (is_mapped) {
      hrx_status_t status = hrx().allocator_virtual_memory_unmap(
          allocator, virtual_buffer, /*virtual_offset=*/0, size);
      const bool unmapped = hrx_status_is_ok(status);
      CHECK(unmapped);
      hrx().status_ignore(status);
      if (!unmapped) {
        return;
      }
    }
    if (physical_memory) {
      hrx_status_t status =
          hrx().allocator_physical_memory_free(allocator, physical_memory);
      CHECK(hrx_status_is_ok(status));
      hrx().status_ignore(status);
    }
    if (virtual_buffer) {
      hrx_status_t status =
          hrx().allocator_virtual_memory_release(allocator, virtual_buffer);
      CHECK(hrx_status_is_ok(status));
      hrx().status_ignore(status);
    }
  }

  // Borrowed allocator that outlives the reservation and physical handle.
  hrx_allocator_t allocator;
  // Scalar reservation consumed only by a successful allocator release.
  hrx_buffer_t virtual_buffer = nullptr;
  // Opaque physical storage retained until every mapping is removed.
  hrx_physical_memory_t physical_memory = nullptr;
  // Length of the reservation, physical allocation, and mapping, in bytes.
  size_t size = 0;
  // Whether teardown must first remove the native mapping.
  bool is_mapped = false;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "query legacy virtual memory support",
                 "[legacy_virtual_memory][query]") {
  bool supported = false;
  size_t minimum_page_size = 1;
  size_t recommended_page_size = 1;
  REQUIRE_OK(hrx().allocator_query_virtual_memory(
      hrx().device_allocator(device_), HRX_MEMORY_TYPE_DEVICE_LOCAL, &supported,
      &minimum_page_size, &recommended_page_size));
  REQUIRE(supported == is_gpu());
  if (supported) {
    CHECK(minimum_page_size > 0);
    CHECK(recommended_page_size >= minimum_page_size);
  } else {
    CHECK(minimum_page_size == 0);
    CHECK(recommended_page_size == 0);
  }
}

TEST_CASE_METHOD(HrxTestFixture, "legacy virtual memory lifecycle",
                 "[legacy_virtual_memory][lifecycle]") {
  hrx_allocator_t allocator = hrx().device_allocator(device_);
  bool supported = false;
  size_t minimum_page_size = 0;
  size_t recommended_page_size = 0;
  REQUIRE_OK(hrx().allocator_query_virtual_memory(
      allocator, HRX_MEMORY_TYPE_DEVICE_LOCAL, &supported, &minimum_page_size,
      &recommended_page_size));
  REQUIRE(supported == is_gpu());
  if (!supported) {
    hrx_buffer_t virtual_buffer = nullptr;
    hrx_status_t status = hrx().allocator_virtual_memory_reserve(
        allocator, /*affinity=*/0, 65536, &virtual_buffer);
    CHECK(!hrx_status_is_ok(status));
    CHECK(virtual_buffer == nullptr);
    hrx().status_ignore(status);
    return;
  }

  REQUIRE(minimum_page_size > 0);
  REQUIRE(recommended_page_size >= minimum_page_size);
  LegacyVirtualMemoryResources resources(allocator);
  resources.size = recommended_page_size;
  REQUIRE_OK(hrx().allocator_virtual_memory_reserve(
      allocator, /*affinity=*/0, resources.size, &resources.virtual_buffer));
  REQUIRE_OK(hrx().allocator_physical_memory_allocate(
      allocator, HRX_MEMORY_TYPE_DEVICE_LOCAL, resources.size,
      &resources.physical_memory));
  REQUIRE_OK(hrx().allocator_virtual_memory_map(
      allocator, resources.virtual_buffer, /*virtual_offset=*/0,
      resources.physical_memory, /*physical_offset=*/0, resources.size));
  resources.is_mapped = true;
  REQUIRE_OK(hrx().allocator_virtual_memory_protect(
      allocator, resources.virtual_buffer, /*virtual_offset=*/0, resources.size,
      HRX_MEMORY_PROTECTION_READ_WRITE));
  const std::array<uint32_t, 4> expected = {7, 13, 19, 23};
  REQUIRE_OK(hrx().synchronous_h2d(transfer_queue_, expected.data(),
                                   resources.virtual_buffer, 0,
                                   sizeof(expected)));
  REQUIRE_OK(hrx().allocator_virtual_memory_protect_scoped(
      allocator, resources.virtual_buffer, /*virtual_offset=*/0, resources.size,
      HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL, HRX_MEMORY_PROTECTION_READ_WRITE));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, resources.virtual_buffer, 0,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);

  REQUIRE_OK(hrx().allocator_virtual_memory_unmap(
      allocator, resources.virtual_buffer, /*virtual_offset=*/0,
      resources.size));
  resources.is_mapped = false;
  REQUIRE_OK(hrx().allocator_physical_memory_free(allocator,
                                                  resources.physical_memory));
  resources.physical_memory = nullptr;
  REQUIRE_OK(hrx().allocator_virtual_memory_release(allocator,
                                                    resources.virtual_buffer));
  resources.virtual_buffer = nullptr;
}
