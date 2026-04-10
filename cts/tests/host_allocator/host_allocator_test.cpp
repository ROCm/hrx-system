// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

// Host allocator tests don't need a device — just the loader.

TEST_CASE("host_allocator_system returns non-null ctl", "[host_allocator]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  REQUIRE(ha.ctl != nullptr);
}

TEST_CASE("host_allocator_malloc zeroed", "[host_allocator][malloc]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *ptr = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc(ha, 256, &ptr));
  REQUIRE(ptr != nullptr);

  // Must be zeroed.
  unsigned char *data = (unsigned char *)ptr;
  for (int i = 0; i < 256; i++) {
    REQUIRE(data[i] == 0);
  }

  hrx().host_allocator_free(ha, ptr);
}

TEST_CASE("host_allocator_malloc_uninitialized", "[host_allocator][malloc]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *ptr = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc_uninitialized(ha, 1024, &ptr));
  REQUIRE(ptr != nullptr);

  // Just verify we can write to it without crashing.
  memset(ptr, 0xAB, 1024);

  hrx().host_allocator_free(ha, ptr);
}

TEST_CASE("host_allocator_realloc grow", "[host_allocator][realloc]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *ptr = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc(ha, 64, &ptr));
  memset(ptr, 0x42, 64);

  REQUIRE_OK(hrx().host_allocator_realloc(ha, 256, &ptr));
  REQUIRE(ptr != nullptr);

  // Original 64 bytes preserved.
  unsigned char *data = (unsigned char *)ptr;
  for (int i = 0; i < 64; i++) {
    REQUIRE(data[i] == 0x42);
  }

  hrx().host_allocator_free(ha, ptr);
}

TEST_CASE("host_allocator_clone", "[host_allocator][clone]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  unsigned char src[128];
  memset(src, 0xCD, sizeof(src));

  void *dst = nullptr;
  REQUIRE_OK(hrx().host_allocator_clone(ha, src, sizeof(src), &dst));
  REQUIRE(dst != nullptr);
  REQUIRE(memcmp(src, dst, sizeof(src)) == 0);

  hrx().host_allocator_free(ha, dst);
}

TEST_CASE("host_allocator_free null is no-op", "[host_allocator][free]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  hrx().host_allocator_free(ha, nullptr); // Must not crash.
}

TEST_CASE("host_allocator_malloc_aligned 64-byte",
          "[host_allocator][aligned]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *ptr = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc_aligned(ha, 256, 64, 0, &ptr));
  REQUIRE(ptr != nullptr);
  REQUIRE(((uintptr_t)ptr % 64) == 0);

  // Must be zeroed.
  unsigned char *data = (unsigned char *)ptr;
  for (int i = 0; i < 256; i++) {
    REQUIRE(data[i] == 0);
  }

  hrx().host_allocator_free_aligned(ha, ptr);
}

TEST_CASE("host_allocator_malloc_aligned 4096-byte page",
          "[host_allocator][aligned]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *ptr = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc_aligned(ha, 8192, 4096, 0, &ptr));
  REQUIRE(ptr != nullptr);
  REQUIRE(((uintptr_t)ptr % 4096) == 0);

  hrx().host_allocator_free_aligned(ha, ptr);
}

TEST_CASE("host_allocator_realloc_aligned", "[host_allocator][aligned]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  void *ptr = nullptr;
  REQUIRE_OK(hrx().host_allocator_malloc_aligned(ha, 64, 64, 0, &ptr));
  memset(ptr, 0x99, 64);

  REQUIRE_OK(hrx().host_allocator_realloc_aligned(ha, 256, 64, 0, &ptr));
  REQUIRE(ptr != nullptr);
  REQUIRE(((uintptr_t)ptr % 64) == 0);

  // Original bytes preserved.
  unsigned char *data = (unsigned char *)ptr;
  for (int i = 0; i < 64; i++) {
    REQUIRE(data[i] == 0x99);
  }

  hrx().host_allocator_free_aligned(ha, ptr);
}

TEST_CASE("host_allocator_free_aligned null is no-op",
          "[host_allocator][aligned]") {
  hrx_host_allocator_t ha = hrx().host_allocator_system();
  hrx().host_allocator_free_aligned(ha, nullptr); // Must not crash.
}
