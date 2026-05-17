// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/file_index.h"

#include <string>

#if !defined(IREE_PLATFORM_WINDOWS)
#include <sys/stat.h>
#include <unistd.h>
#endif  // !IREE_PLATFORM_WINDOWS

#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

#if IREE_FILE_IO_ENABLE

iree_status_t WriteTempContents(iree_string_view_t path) {
  static const char kContents[] = "remote file index test";
  return iree_io_file_contents_write(
      path, iree_make_const_byte_span(kContents, sizeof(kContents) - 1),
      iree_allocator_system());
}

TEST(FileIndexTest, AllowPathStatsFileAndOpensAsyncReadOnly) {
  iree::testing::TempFilePath path("iree_hal_remote_file_index");
  IREE_ASSERT_OK(WriteTempContents(path.path_view()));

  iree_hal_remote_file_index_t* file_index = NULL;
  IREE_ASSERT_OK(
      iree_hal_remote_file_index_create(iree_allocator_system(), &file_index));
  IREE_ASSERT_OK(iree_hal_remote_file_index_allow_path(
      file_index, IREE_SV("model://weights"), path.path_view(),
      IREE_HAL_MEMORY_ACCESS_READ));

  iree_io_file_handle_t* handle = NULL;
  iree_hal_memory_access_t granted_access = 0;
  IREE_ASSERT_OK(iree_hal_remote_file_index_open(
      file_index, IREE_SV("model://weights"), IREE_HAL_MEMORY_ACCESS_READ,
      iree_allocator_system(), &handle, &granted_access));
  EXPECT_EQ(granted_access, IREE_HAL_MEMORY_ACCESS_READ);
  EXPECT_EQ(iree_io_file_handle_access(handle), IREE_IO_FILE_ACCESS_READ);
  EXPECT_TRUE(iree_io_file_handle_uses_async_io(handle));
  iree_io_file_handle_release(handle);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("model://weights"), IREE_HAL_MEMORY_ACCESS_WRITE,
          iree_allocator_system(), &handle, &granted_access));

  iree_hal_remote_file_index_release(file_index);
}

TEST(FileIndexTest, DirectoryEntryResolvesLogicalSuffix) {
  iree::testing::TempFilePath path("iree_hal_remote_file_index");
  IREE_ASSERT_OK(WriteTempContents(path.path_view()));
  iree_string_view_t directory = iree_file_path_dirname(path.path_view());
  iree_string_view_t basename = iree_file_path_basename(path.path_view());

  iree_hal_remote_file_index_t* file_index = NULL;
  IREE_ASSERT_OK(
      iree_hal_remote_file_index_create(iree_allocator_system(), &file_index));
  IREE_ASSERT_OK(iree_hal_remote_file_index_allow_path(
      file_index, IREE_SV("tmp://"), directory, IREE_HAL_MEMORY_ACCESS_READ));

  std::string logical_path =
      std::string("tmp://") + std::string(basename.data, basename.size);

  iree_io_file_handle_t* handle = NULL;
  iree_hal_memory_access_t granted_access = 0;
  IREE_ASSERT_OK(iree_hal_remote_file_index_open(
      file_index,
      iree_make_string_view(logical_path.data(), logical_path.size()),
      IREE_HAL_MEMORY_ACCESS_READ, iree_allocator_system(), &handle,
      &granted_access));
  EXPECT_EQ(granted_access, IREE_HAL_MEMORY_ACCESS_READ);
  iree_io_file_handle_release(handle);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://../secret"), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp:///secret"), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://./secret"), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_remote_file_index_open(
                            file_index, IREE_SV("tmp://secret//weights"),
                            IREE_HAL_MEMORY_ACCESS_READ,
                            iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_remote_file_index_open(
                            file_index, IREE_SV("tmp://secret\\weights"),
                            IREE_HAL_MEMORY_ACCESS_READ,
                            iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://C:weights"), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_remote_file_index_open(
                            file_index, IREE_SV("tmp://weights:stream"),
                            IREE_HAL_MEMORY_ACCESS_READ,
                            iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://CON"), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://LPT1.txt"), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://trailing."), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_remote_file_index_open(
          file_index, IREE_SV("tmp://trailing "), IREE_HAL_MEMORY_ACCESS_READ,
          iree_allocator_system(), &handle, &granted_access));

  iree_hal_remote_file_index_release(file_index);
}

#if !defined(IREE_PLATFORM_WINDOWS)

TEST(FileIndexTest, ExplicitSymlinkFileIsAllowed) {
  iree::testing::TempFilePath target_path("iree_hal_remote_file_index_target");
  IREE_ASSERT_OK(WriteTempContents(target_path.path_view()));
  iree::testing::TempFilePath link_path("iree_hal_remote_file_index_link");
  ASSERT_EQ(symlink(target_path.path().c_str(), link_path.path().c_str()), 0);

  iree_hal_remote_file_index_t* file_index = NULL;
  IREE_ASSERT_OK(
      iree_hal_remote_file_index_create(iree_allocator_system(), &file_index));
  IREE_ASSERT_OK(iree_hal_remote_file_index_allow_path(
      file_index, IREE_SV("model://weights"), link_path.path_view(),
      IREE_HAL_MEMORY_ACCESS_READ));

  iree_io_file_handle_t* handle = NULL;
  iree_hal_memory_access_t granted_access = 0;
  IREE_ASSERT_OK(iree_hal_remote_file_index_open(
      file_index, IREE_SV("model://weights"), IREE_HAL_MEMORY_ACCESS_READ,
      iree_allocator_system(), &handle, &granted_access));
  EXPECT_EQ(granted_access, IREE_HAL_MEMORY_ACCESS_READ);
  iree_io_file_handle_release(handle);

  unlink(link_path.path().c_str());
  iree_hal_remote_file_index_release(file_index);
}

TEST(FileIndexTest, ExplicitSymlinkDirectoryIsAllowed) {
  iree::testing::TempFilePath directory_path("iree_hal_remote_file_index_dir");
  ASSERT_EQ(mkdir(directory_path.path().c_str(), 0777), 0);
  std::string target_file_path = directory_path.path() + "/weights.bin";
  IREE_ASSERT_OK(WriteTempContents(
      iree_make_string_view(target_file_path.data(), target_file_path.size())));

  iree::testing::TempFilePath link_path("iree_hal_remote_file_index_dir_link");
  ASSERT_EQ(symlink(directory_path.path().c_str(), link_path.path().c_str()),
            0);

  iree_hal_remote_file_index_t* file_index = NULL;
  IREE_ASSERT_OK(
      iree_hal_remote_file_index_create(iree_allocator_system(), &file_index));
  IREE_ASSERT_OK(iree_hal_remote_file_index_allow_path(
      file_index, IREE_SV("model://"), link_path.path_view(),
      IREE_HAL_MEMORY_ACCESS_READ));

  iree_io_file_handle_t* handle = NULL;
  iree_hal_memory_access_t granted_access = 0;
  IREE_ASSERT_OK(iree_hal_remote_file_index_open(
      file_index, IREE_SV("model://weights.bin"), IREE_HAL_MEMORY_ACCESS_READ,
      iree_allocator_system(), &handle, &granted_access));
  EXPECT_EQ(granted_access, IREE_HAL_MEMORY_ACCESS_READ);
  iree_io_file_handle_release(handle);

  unlink(link_path.path().c_str());
  unlink(target_file_path.c_str());
  rmdir(directory_path.path().c_str());
  iree_hal_remote_file_index_release(file_index);
}

#endif  // !IREE_PLATFORM_WINDOWS

#endif  // IREE_FILE_IO_ENABLE

}  // namespace
