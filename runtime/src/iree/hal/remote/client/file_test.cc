// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/file.h"

#include <cstdint>

#include "iree/async/proactor_platform.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

TEST(RemoteClientFileTest, ImportsHostAllocation) {
  iree_allocator_t host_allocator = iree_allocator_system();
  uint8_t contents[16] = {0};

  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(contents, sizeof(contents)),
      iree_io_file_handle_release_callback_null(), host_allocator, &handle));

  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_file_import(
      /*queue_affinity=*/0,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, /*proactor=*/nullptr, host_allocator,
      &file));

  EXPECT_TRUE(iree_hal_remote_client_file_isa(file));
  EXPECT_EQ(iree_hal_file_allowed_access(file),
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(iree_hal_file_length(file), sizeof(contents));
  EXPECT_EQ(iree_hal_file_storage_buffer(file), nullptr);
  EXPECT_EQ(iree_hal_file_async_handle(file), nullptr);
  EXPECT_FALSE(iree_hal_file_supports_synchronous_io(file));

  iree_hal_remote_client_file_view_t view;
  IREE_ASSERT_OK(iree_hal_remote_client_file_resolve(file, &view));
  EXPECT_EQ(view.kind, IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION);
  EXPECT_EQ(view.access,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(view.length, sizeof(contents));
  EXPECT_EQ(view.host_allocation.data, contents);
  EXPECT_EQ(view.host_allocation.data_length, sizeof(contents));
  EXPECT_EQ(view.async_file, nullptr);
  EXPECT_EQ(view.remote_file_id, 0u);

  iree_hal_file_release(file);
  iree_io_file_handle_release(handle);
}

TEST(RemoteClientFileTest, ImportsAsyncFileHandle) {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_async_proactor_t* proactor = nullptr;
  iree_async_proactor_options_t proactor_options =
      iree_async_proactor_options_default();
  iree_status_t status = iree_async_proactor_create_platform(
      proactor_options, host_allocator, &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_file");
  const uint8_t contents[16] = {0};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(contents, sizeof(contents)),
      host_allocator));

  iree_io_file_handle_t* handle = nullptr;
  status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), host_allocator, &handle);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
    iree_async_proactor_release(proactor);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);
  EXPECT_TRUE(iree_io_file_handle_uses_async_io(handle));

  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_file_import(
      /*queue_affinity=*/0,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, proactor, host_allocator, &file));

  EXPECT_TRUE(iree_hal_remote_client_file_isa(file));
  EXPECT_EQ(iree_hal_file_allowed_access(file),
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(iree_hal_file_length(file), sizeof(contents));
  EXPECT_EQ(iree_hal_file_storage_buffer(file), nullptr);
  EXPECT_NE(iree_hal_file_async_handle(file), nullptr);
  EXPECT_FALSE(iree_hal_file_supports_synchronous_io(file));

  iree_hal_remote_client_file_view_t view;
  IREE_ASSERT_OK(iree_hal_remote_client_file_resolve(file, &view));
  EXPECT_EQ(view.kind, IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE);
  EXPECT_EQ(view.access,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(view.length, sizeof(contents));
  EXPECT_EQ(view.host_allocation.data, nullptr);
  EXPECT_EQ(view.host_allocation.data_length, 0u);
  EXPECT_EQ(view.async_file, iree_hal_file_async_handle(file));
  EXPECT_EQ(view.remote_file_id, 0u);

  iree_hal_file_release(file);
  iree_io_file_handle_release(handle);
  iree_async_proactor_release(proactor);
}

TEST(RemoteClientFileTest, RejectsSynchronousFileHandleBeforeQueueSubmission) {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_async_proactor_t* proactor = nullptr;
  iree_async_proactor_options_t proactor_options =
      iree_async_proactor_options_default();
  iree_status_t status = iree_async_proactor_create_platform(
      proactor_options, host_allocator, &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);

  iree::testing::TempFilePath path("iree_hal_remote_client_sync_file");
  const uint8_t contents[16] = {0};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(contents, sizeof(contents)),
      host_allocator));

  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_SHARE_READ,
      path.path_view(), host_allocator, &handle));
  EXPECT_FALSE(iree_io_file_handle_uses_async_io(handle));

  iree_hal_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_remote_client_file_import(
          /*queue_affinity=*/0, IREE_HAL_MEMORY_ACCESS_READ, handle,
          IREE_HAL_EXTERNAL_FILE_FLAG_NONE, proactor, host_allocator, &file));
  EXPECT_EQ(file, nullptr);

  iree_io_file_handle_release(handle);
  iree_async_proactor_release(proactor);
}

TEST(RemoteClientFileTest, RejectsDisallowedHandleAccess) {
  iree_allocator_t host_allocator = iree_allocator_system();
  uint8_t contents[16] = {0};

  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ, iree_make_byte_span(contents, sizeof(contents)),
      iree_io_file_handle_release_callback_null(), host_allocator, &handle));

  iree_hal_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_remote_client_file_import(
                            /*queue_affinity=*/0, IREE_HAL_MEMORY_ACCESS_WRITE,
                            handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
                            /*proactor=*/nullptr, host_allocator, &file));
  EXPECT_EQ(file, nullptr);

  iree_io_file_handle_release(handle);
}

}  // namespace
