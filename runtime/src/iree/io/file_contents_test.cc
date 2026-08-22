// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_contents.h"

#include "iree/base/api.h"

#if IREE_FILE_IO_ENABLE

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/path.h"
#include "iree/io/stdio_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace iree {
namespace io {
namespace {

using ::iree::testing::status::StatusIs;

std::string GetUniqueContents(const char* unique_name,
                              iree_host_size_t padded_size) {
  std::string str = std::string("Test with name ") + unique_name + "\n";
  if (str.size() < padded_size) str.resize(padded_size, 0);
  return str;
}

#if defined(IREE_PLATFORM_WINDOWS)

std::wstring ToWin32Path(iree_string_view_t path) {
  wchar_t* converted_path = NULL;
  IREE_CHECK_OK(
      iree_file_path_to_win32(path, iree_allocator_system(), &converted_path));
  std::wstring result(converted_path);
  iree_allocator_free(iree_allocator_system(), converted_path);
  return result;
}

#endif  // IREE_PLATFORM_WINDOWS

TEST(FileContents, ReadWriteContentsPreload) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 32);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, ReadWriteContentsMmap) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 4096);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(
      iree_io_file_contents_map(path.path_view(), IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, StdioFileReadWrite) {
  iree::testing::TempFilePath path("iree_stdio_file_test");
  auto write_contents = GetUniqueContents("StdioFileReadWrite", 32);

  FILE* write_file = NULL;
  IREE_ASSERT_OK(iree_io_stdio_file_open(path.path_view(), "wb",
                                         iree_allocator_system(), &write_file));
  ASSERT_EQ(fwrite(write_contents.data(), 1, write_contents.size(), write_file),
            write_contents.size());
  ASSERT_EQ(fclose(write_file), 0);

  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));
  ASSERT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);
  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, StdioStreamFillMultiBytePattern) {
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
      path.path_view(), iree_allocator_system(), &stream));

  const uint8_t pattern[] = {0x34, 0x12};
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 3, pattern, sizeof(pattern)));

  iree_io_stream_release(stream);

  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));

  const uint8_t expected[] = {0x34, 0x12, 0x34, 0x12, 0x34, 0x12};
  EXPECT_EQ(sizeof(expected), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(expected, read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

// Tests that a file opened for reading can be opened again concurrently.
// Validates FILE_SHARE_READ behavior on Windows. Without it, the second open
// fails with ERROR_SHARING_VIOLATION.
TEST(FileContents, ConcurrentReadOpens) {
  constexpr const char* kUniqueName = "ConcurrentReadOpens";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Write a file to open.
  auto contents = GetUniqueContents(kUniqueName, 4096);
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system()));

  // Open the file twice for reading. Both should succeed.
  iree_io_file_contents_t* read1 = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(path.path_view(),
                                           IREE_IO_FILE_ACCESS_READ,
                                           iree_allocator_system(), &read1));

  iree_io_file_contents_t* read2 = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(path.path_view(),
                                           IREE_IO_FILE_ACCESS_READ,
                                           iree_allocator_system(), &read2));

  // Both should have the same contents.
  EXPECT_EQ(read1->const_buffer.data_length, read2->const_buffer.data_length);
  EXPECT_EQ(memcmp(read1->const_buffer.data, read2->const_buffer.data,
                   read1->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read2);
  iree_io_file_contents_free(read1);
}

TEST(FileContents, ReadAndMapWhileWriterOpen) {
  constexpr const char* kUniqueName = "ReadAndMapWhileWriterOpen";
  iree::testing::TempFilePath path("iree_file_contents_test");

  auto contents = GetUniqueContents(kUniqueName, 4096);
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system()));

  iree_io_file_handle_t* writer = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE,
      path.path_view(), iree_allocator_system(), &writer));

  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));
  ASSERT_EQ(contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);
  iree_io_file_contents_free(read_contents);

  iree_io_file_contents_t* mapped_contents = NULL;
  IREE_ASSERT_OK(
      iree_io_file_contents_map(path.path_view(), IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &mapped_contents));
  ASSERT_EQ(contents.size(), mapped_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(contents.data(), mapped_contents->const_buffer.data,
                   mapped_contents->const_buffer.data_length),
            0);
  iree_io_file_contents_free(mapped_contents);

  iree_io_file_handle_release(writer);
}

TEST(FileContents, CreateOverwritesSharedAsyncFileAndResizes) {
  iree::testing::TempFilePath path("iree_file_contents_test");
  const uint8_t initial_contents[] = {0x11, 0x22, 0x33, 0x44};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(initial_contents, sizeof(initial_contents)),
      iree_allocator_system()));

  iree_io_file_handle_t* shared_reader = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_ASYNC |
          IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_SHARE_WRITE,
      path.path_view(), iree_allocator_system(), &shared_reader));

  iree_io_file_handle_t* replacement = NULL;
  iree_status_t status = iree_io_file_handle_create(
      IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE,
      path.path_view(), /*initial_size=*/2, iree_allocator_system(),
      &replacement);
  iree_io_file_handle_release(replacement);
  iree_io_file_handle_release(shared_reader);
  IREE_ASSERT_OK(status);

  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));
  ASSERT_EQ(read_contents->const_buffer.data_length, 2u);
  EXPECT_EQ(read_contents->const_buffer.data[0], 0u);
  EXPECT_EQ(read_contents->const_buffer.data[1], 0u);
  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, ResizeOpenFilePreservesPrefixAcrossShrinkAndExtend) {
  iree::testing::TempFilePath path("iree_file_contents_test");
  const uint8_t initial_contents[] = {0x11, 0x22, 0x33, 0x44};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(initial_contents, sizeof(initial_contents)),
      iree_allocator_system()));

  iree_io_file_handle_t* handle = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE, path.path_view(),
      iree_allocator_system(), &handle));
  IREE_EXPECT_OK(iree_io_file_handle_resize(handle, 2));
  IREE_EXPECT_OK(iree_io_file_handle_resize(handle, 6));
  iree_io_file_handle_release(handle);

  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));
  ASSERT_EQ(read_contents->const_buffer.data_length, 6);
  EXPECT_EQ(memcmp(read_contents->const_buffer.data, initial_contents, 2), 0);
  iree_io_file_contents_free(read_contents);
}

#if defined(IREE_PLATFORM_WINDOWS)

TEST(FileContents, ReadWriteLongUtf8Path) {
  iree::testing::TempFilePath root_path("iree_file_contents_long_path");
  std::string directory_path = root_path.path();
  std::vector<std::wstring> created_directories;
  do {
    if (!created_directories.empty()) {
      directory_path.append("/0123456789abcdef0123456789abcdef");
    }
    std::wstring win32_directory = ToWin32Path(
        iree_make_string_view(directory_path.data(), directory_path.size()));
    BOOL created = CreateDirectoryW(win32_directory.c_str(), NULL);
    DWORD create_error = created ? ERROR_SUCCESS : GetLastError();
    ASSERT_TRUE(created || create_error == ERROR_ALREADY_EXISTS)
        << "CreateDirectoryW failed with error " << create_error;
    created_directories.push_back(std::move(win32_directory));
  } while (directory_path.size() < 300);

  std::string file_path =
      directory_path + "/payload_\xE8\xB7\xAF\xE5\xBE\x84.bin";
  ASSERT_GT(file_path.size(), 260u);
  iree_string_view_t file_path_view =
      iree_make_string_view(file_path.data(), file_path.size());
  auto write_contents = GetUniqueContents("ReadWriteLongUtf8Path", 32);

  FILE* write_file = NULL;
  IREE_ASSERT_OK(iree_io_stdio_file_open(file_path_view, "wb",
                                         iree_allocator_system(), &write_file));
  ASSERT_EQ(fwrite(write_contents.data(), 1, write_contents.size(), write_file),
            write_contents.size());
  ASSERT_EQ(fclose(write_file), 0);

  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      file_path_view, iree_allocator_system(), &read_contents));
  ASSERT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);
  iree_io_file_contents_free(read_contents);

  iree_io_stream_t* stdio_stream = NULL;
  IREE_ASSERT_OK(
      iree_io_stdio_stream_open(IREE_IO_STDIO_STREAM_MODE_READ, file_path_view,
                                iree_allocator_system(), &stdio_stream));
  iree_io_stream_release(stdio_stream);

  std::wstring win32_file_path = ToWin32Path(file_path_view);
  BOOL deleted = DeleteFileW(win32_file_path.c_str());
  DWORD delete_error = deleted ? ERROR_SUCCESS : GetLastError();
  ASSERT_TRUE(deleted) << "DeleteFileW failed with error " << delete_error;
  for (auto it = created_directories.rbegin(); it != created_directories.rend();
       ++it) {
    BOOL removed = RemoveDirectoryW(it->c_str());
    DWORD remove_error = removed ? ERROR_SUCCESS : GetLastError();
    ASSERT_TRUE(removed) << "RemoveDirectoryW failed with error "
                         << remove_error;
  }
}

#endif  // IREE_PLATFORM_WINDOWS

}  // namespace
}  // namespace io
}  // namespace iree

#endif  // IREE_FILE_IO_ENABLE
