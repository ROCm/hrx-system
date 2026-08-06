// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/managed_global.h"

#include <cstdint>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ManagedGlobalTest, DerivesStorageName) {
  char* storage_name = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_managed_global_storage_name(
      iree_allocator_system(), IREE_SV("counter"), &storage_name));
  EXPECT_STREQ("counter.managed", storage_name);
  iree_allocator_free(iree_allocator_system(), storage_name);
}

TEST(ManagedGlobalTest, RejectsStorageNameSizeOverflow) {
  char* storage_name = reinterpret_cast<char*>(uintptr_t{1});
  iree_status_t status = iree_hal_streaming_managed_global_storage_name(
      iree_allocator_system(),
      iree_make_string_view(nullptr, IREE_HOST_SIZE_MAX), &storage_name);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  EXPECT_EQ(nullptr, storage_name);
  iree_status_ignore(status);
}

}  // namespace
