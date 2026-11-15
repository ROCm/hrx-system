// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/async/cts/file/native_file.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/file.h"
#include "iree/async/operations/file.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

#if defined(IREE_PLATFORM_LINUX) && \
    !(defined(__ANDROID_API__) && __ANDROID_API__ < 30)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace iree::async::cts {
namespace {

enum class FileSource { kAsyncOpen, kImport };

struct AlignedBufferDeleter {
  void operator()(uint8_t* pointer) const {
    iree_allocator_free_aligned(iree_allocator_system(), pointer);
  }
};
using AlignedBuffer = std::unique_ptr<uint8_t, AlignedBufferDeleter>;

class FileRequirementsTest : public CtsTestBase<> {
 protected:
  void TearDown() override {
    iree_async_file_release(file_);
    if (path_) {
      EXPECT_TRUE(path_.Remove());
    }
    CtsTestBase::TearDown();
  }

  iree_status_t Open(FileSource source, iree_async_file_flags_t flags,
                     iree_const_byte_span_t contents) {
    path_ = iree::testing::TempFilePath("iree_async_requirements");
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        path_.path_view(), contents, iree_allocator_system()));
    if (source == FileSource::kImport) {
      return ImportNativeFile(proactor_, path_.path_view(),
                              kNativeFileAccessRead | kNativeFileAccessWrite,
                              flags, &file_);
    }
    CompletionTracker completion;
    iree_async_file_open_operation_t operation = {};
    operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
    operation.base.completion_fn = CompletionTracker::Callback;
    operation.base.user_data = &completion;
    operation.path = path_.path().c_str();
    operation.open_flags =
        IREE_ASYNC_FILE_OPEN_FLAG_READ | IREE_ASYNC_FILE_OPEN_FLAG_WRITE;
    if (iree_any_bit_set(flags, IREE_ASYNC_FILE_FLAG_DIRECT)) {
      operation.open_flags |= IREE_ASYNC_FILE_OPEN_FLAG_DIRECT;
    }
    IREE_RETURN_IF_ERROR(
        iree_async_proactor_submit_one(proactor_, &operation.base));
    PollUntil(1);
    EXPECT_EQ(completion.call_count, 1);
    file_ = operation.opened_file;
    return completion.ConsumeStatus();
  }

  void Write(uint64_t offset, uint8_t* data, size_t length) {
    CompletionTracker completion;
    iree_async_file_write_operation_t operation = {};
    operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
    operation.base.completion_fn = CompletionTracker::Callback;
    operation.base.user_data = &completion;
    operation.file = file_;
    operation.offset = offset;
    operation.buffer = iree_async_span_from_ptr(data, length);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
    PollUntil(1);
    EXPECT_EQ(completion.call_count, 1);
    IREE_ASSERT_OK(completion.ConsumeStatus());
    ASSERT_EQ(operation.bytes_written, length);
  }

  void Read(uint64_t offset, uint8_t* data, size_t length,
            size_t expected_length) {
    CompletionTracker completion;
    iree_async_file_read_operation_t operation = {};
    operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
    operation.base.completion_fn = CompletionTracker::Callback;
    operation.base.user_data = &completion;
    operation.file = file_;
    operation.offset = offset;
    operation.buffer = iree_async_span_from_ptr(data, length);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
    PollUntil(1);
    EXPECT_EQ(completion.call_count, 1);
    IREE_ASSERT_OK(completion.ConsumeStatus());
    ASSERT_EQ(operation.bytes_read, expected_length);
  }

  void Close() {
    CompletionTracker completion;
    iree_async_file_close_operation_t operation = {};
    operation.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE;
    operation.base.completion_fn = CompletionTracker::Callback;
    operation.base.user_data = &completion;
    operation.file = file_;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
    file_ = nullptr;  // An accepted close consumes the reference.
    PollUntil(1);
    EXPECT_EQ(completion.call_count, 1);
    IREE_ASSERT_OK(completion.ConsumeStatus());
  }

  void ChangedRange(FileSource source, iree_async_file_flags_t flags) {
    IREE_ASSERT_OK(Open(source, flags, iree_const_byte_span_empty()));
    iree_async_file_requirements_t requirements;
    iree::Status query_status = iree_async_file_query_requirements(
        file_->primitive, file_->flags, &requirements);
    if (query_status.code() == iree::StatusCode::kUnavailable) {
      GTEST_SKIP() << query_status.ToString();
    }
    IREE_ASSERT_OK(query_status);
    const auto& read = requirements.read;
    const auto& write = requirements.write;
    const size_t buffer_alignment =
        std::max(read.buffer_alignment, write.buffer_alignment);
    const uint64_t transfer_alignment = std::max(
        {read.offset_alignment, write.offset_alignment,
         uint64_t(read.length_alignment), uint64_t(write.length_alignment)});
    SCOPED_TRACE(::testing::Message()
                 << "buffer alignment=" << buffer_alignment
                 << ", transfer alignment=" << transfer_alignment);
    ASSERT_TRUE(iree_host_size_is_power_of_two(buffer_alignment));
    ASSERT_TRUE(iree_host_size_is_power_of_two(transfer_alignment));
    // Bound the fixture, not the production API. These are file geometry tests,
    // not large-file throughput measurements.
    ASSERT_LE(transfer_alignment, 1024 * 1024u);
    ASSERT_LE(buffer_alignment, 1024 * 1024u);
    const size_t extent = iree_host_align(17, transfer_alignment);
    const size_t length = extent * 3;
    void* storage = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc_aligned(
        iree_allocator_system(), length + buffer_alignment, buffer_alignment, 0,
        &storage));
    AlignedBuffer owner(static_cast<uint8_t*>(storage));
    // Test the reported buffer alignment, not incidental page alignment from
    // the allocator. On byte-granular platforms this is an odd address.
    uint8_t* data = owner.get() + buffer_alignment;
    std::vector<uint8_t> expected(length);
    for (size_t i = 0; i < length; ++i) {
      expected[i] = uint8_t(i * 37 + 11);
    }
    std::memcpy(data, expected.data(), length);
    ASSERT_NO_FATAL_FAILURE(Write(0, data, length));
    for (size_t i = 0; i < extent; ++i) {
      data[i] = uint8_t(i * 29 + 203);
      expected[extent + i] = data[i];
    }
    ASSERT_NO_FATAL_FAILURE(Write(extent, data, extent));
    std::memset(data, 0, length);
    ASSERT_NO_FATAL_FAILURE(Read(0, data, length, length));
    EXPECT_EQ(std::memcmp(data, expected.data(), length), 0);
    ASSERT_NO_FATAL_FAILURE(Close());

    // A fresh ordinary reader verifies publication and all untouched bytes;
    // readback is not confined to the writer's native handle.
    iree_io_file_contents_t* readback = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(
        path_.path_view(), iree_allocator_system(), &readback));
    EXPECT_EQ(readback->const_buffer.data_length, expected.size());
    if (readback->const_buffer.data_length == expected.size()) {
      EXPECT_EQ(
          std::memcmp(readback->const_buffer.data, expected.data(), length), 0);
    }
    iree_io_file_contents_free(readback);
  }

  void ShortEof(FileSource source, iree_async_file_flags_t flags) {
    const uint8_t contents[] = "odd byte count 17";
    const size_t length = sizeof(contents) - 1;
    IREE_ASSERT_OK(
        Open(source, flags, iree_make_const_byte_span(contents, length)));
    iree_async_file_requirements_t requirements;
    iree::Status query_status = iree_async_file_query_requirements(
        file_->primitive, file_->flags, &requirements);
    if (query_status.code() == iree::StatusCode::kUnavailable) {
      GTEST_SKIP() << query_status.ToString();
    }
    IREE_ASSERT_OK(query_status);
    const auto& read = requirements.read;
    ASSERT_TRUE(iree_host_size_is_power_of_two(read.buffer_alignment));
    ASSERT_TRUE(iree_host_size_is_power_of_two(read.offset_alignment));
    ASSERT_TRUE(iree_host_size_is_power_of_two(read.length_alignment));
    ASSERT_LE(read.buffer_alignment, 1024 * 1024u);
    ASSERT_LE(read.offset_alignment, 1024 * 1024u);
    ASSERT_LE(read.length_alignment, 1024 * 1024u);
    const size_t request_length =
        iree_host_align(length + 1, read.length_alignment);
    void* storage = nullptr;
    IREE_ASSERT_OK(
        iree_allocator_malloc_aligned(iree_allocator_system(), request_length,
                                      read.buffer_alignment, 0, &storage));
    AlignedBuffer data(static_cast<uint8_t*>(storage));
    std::memset(data.get(), 0xCD, request_length);
    ASSERT_NO_FATAL_FAILURE(Read(0, data.get(), request_length, length));
    EXPECT_EQ(std::memcmp(data.get(), contents, length), 0);
    // An exact short count may not be a legal next request offset. Use a new
    // aligned request beyond EOF instead of blindly retrying the suffix.
    const uint64_t eof_offset = iree_host_align(length, read.offset_alignment);
    ASSERT_NO_FATAL_FAILURE(Read(eof_offset, data.get(), request_length, 0));
    ASSERT_NO_FATAL_FAILURE(Close());
  }

 protected:
  // Owned fixture path, empty until Open runs after backend setup succeeds.
  iree::testing::TempFilePath path_;
  // Owned until an accepted FILE_CLOSE consumes it, or teardown releases it.
  iree_async_file_t* file_ = nullptr;
};

TEST_P(FileRequirementsTest, BufferedOpenChangedRange) {
  ChangedRange(FileSource::kAsyncOpen, IREE_ASYNC_FILE_FLAG_NONE);
}

TEST_P(FileRequirementsTest, FailedQueryPreservesOutputAndHandle) {
  uint8_t contents[] = "borrowed handle";
  IREE_ASSERT_OK(Open(FileSource::kImport, IREE_ASYNC_FILE_FLAG_NONE,
                      iree_make_const_byte_span(contents, sizeof(contents))));
  const iree_async_file_requirements_t original = {{2, 4, 8}, {16, 32, 64}};
  auto requirements = original;
  EXPECT_THAT(
      iree::Status(iree_async_file_query_requirements(
          file_->primitive, ~IREE_ASYNC_FILE_FLAG_DIRECT, &requirements)),
      ::iree::testing::status::StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_EQ(std::memcmp(&original, &requirements, sizeof(original)), 0);
  EXPECT_THAT(
      iree::Status(iree_async_file_query_requirements(
          iree_async_primitive_none(), IREE_ASYNC_FILE_FLAG_NONE,
          &requirements)),
      ::iree::testing::status::StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_EQ(std::memcmp(&original, &requirements, sizeof(original)), 0);
  uint8_t readback[sizeof(contents)] = {};
  ASSERT_NO_FATAL_FAILURE(
      Read(0, readback, sizeof(readback), sizeof(readback)));
  EXPECT_EQ(std::memcmp(contents, readback, sizeof(contents)), 0);
}

TEST_P(FileRequirementsTest, BufferedImportChangedRange) {
  ChangedRange(FileSource::kImport, IREE_ASYNC_FILE_FLAG_NONE);
}

TEST_P(FileRequirementsTest, DirectOpenChangedRange) {
  ChangedRange(FileSource::kAsyncOpen, IREE_ASYNC_FILE_FLAG_DIRECT);
}

TEST_P(FileRequirementsTest, DirectImportChangedRange) {
  ChangedRange(FileSource::kImport, IREE_ASYNC_FILE_FLAG_DIRECT);
}

#if defined(IREE_PLATFORM_LINUX) && \
    !(defined(__ANDROID_API__) && __ANDROID_API__ < 30)
TEST_P(FileRequirementsTest, DirectSharedMemoryIsByteGranular) {
  const int descriptor = memfd_create("iree_direct_requirements", MFD_CLOEXEC);
  ASSERT_GE(descriptor, 0);
  // Reopen through shmem's file-open path, which admits O_DIRECT. The
  // anonymous descriptor returned by memfd_create bypasses that open path.
  const std::string path = "/proc/self/fd/" + std::to_string(descriptor);
  auto status =
      ImportNativeFile(proactor_, iree_make_cstring_view(path.c_str()),
                       kNativeFileAccessRead | kNativeFileAccessWrite,
                       IREE_ASYNC_FILE_FLAG_DIRECT, &file_);
  close(descriptor);
  IREE_ASSERT_OK(status);

  iree_async_file_requirements_t requirements;
  IREE_ASSERT_OK(iree_async_file_query_requirements(
      file_->primitive, file_->flags, &requirements));
  EXPECT_EQ(requirements.read.buffer_alignment, 1u);
  EXPECT_EQ(requirements.read.offset_alignment, 1u);
  EXPECT_EQ(requirements.read.length_alignment, 1u);
  EXPECT_EQ(requirements.write.buffer_alignment, 1u);
  EXPECT_EQ(requirements.write.offset_alignment, 1u);
  EXPECT_EQ(requirements.write.length_alignment, 1u);

  alignas(64) uint8_t data[] = "_0123456789";
  ASSERT_NO_FATAL_FAILURE(Write(0, data + 1, 10));
  std::memcpy(data + 1, "abc", 3);
  ASSERT_NO_FATAL_FAILURE(Write(3, data + 1, 3));
  std::memset(data + 1, 0, 10);
  ASSERT_NO_FATAL_FAILURE(Read(1, data + 1, 9, 9));
  EXPECT_EQ(std::memcmp(data + 1, "12abc6789", 9), 0);
  EXPECT_EQ(data[0], '_');
  ASSERT_NO_FATAL_FAILURE(Close());
}
#endif

TEST_P(FileRequirementsTest, BufferedOpenShortEof) {
  ShortEof(FileSource::kAsyncOpen, IREE_ASYNC_FILE_FLAG_NONE);
}

TEST_P(FileRequirementsTest, BufferedImportShortEof) {
  ShortEof(FileSource::kImport, IREE_ASYNC_FILE_FLAG_NONE);
}

TEST_P(FileRequirementsTest, DirectOpenShortEof) {
  ShortEof(FileSource::kAsyncOpen, IREE_ASYNC_FILE_FLAG_DIRECT);
}

TEST_P(FileRequirementsTest, DirectImportShortEof) {
  ShortEof(FileSource::kImport, IREE_ASYNC_FILE_FLAG_DIRECT);
}

CTS_REGISTER_TEST_SUITE(FileRequirementsTest);

}  // namespace
}  // namespace iree::async::cts
