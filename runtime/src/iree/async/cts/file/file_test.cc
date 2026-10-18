// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async file I/O operations.
//
// Tests cover the full file lifecycle: open, read, write, close. All operations
// use positioned (pread/pwrite) semantics, so concurrent reads/writes at
// different offsets are safe.

#include "iree/async/file.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "iree/async/cts/file/native_file.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/event.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/region.h"
#include "iree/async/slab.h"
#include "iree/async/span.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Test fixture with temp file helpers
//===----------------------------------------------------------------------===//

class FileTest : public CtsTestBase<> {
 protected:
  void TearDown() override {
    // Clean up any temp files created during the test.
    for (const auto& path : temp_paths_) {
      EXPECT_TRUE(path.Remove()) << path.path();
    }
    CtsTestBase::TearDown();
  }

  // Creates a temp file with the given contents, returns the path.
  // The file is closed after writing; callers reopen it via file_open or
  // file_import as needed.
  std::string CreateTempFileWithContents(const void* data, size_t length) {
    iree::testing::TempFilePath temp_path("iree_async_cts_file");
    IREE_EXPECT_OK(iree_io_file_contents_write(
        temp_path.path_view(), iree_make_const_byte_span(data, length),
        iree_allocator_system()));
    std::string path = temp_path.path();
    temp_paths_.push_back(std::move(temp_path));
    return path;
  }

  // Creates an empty temp file and returns the path.
  std::string CreateEmptyTempFile() {
    return CreateTempFileWithContents("", 0);
  }

  // Removes a tracked temp file before the test recreates it.
  bool RemoveTempFile(const std::string& path) {
    auto path_it =
        std::find_if(temp_paths_.begin(), temp_paths_.end(),
                     [&path](const iree::testing::TempFilePath& temp_path) {
                       return temp_path.path() == path;
                     });
    return path_it != temp_paths_.end() && path_it->Remove();
  }

  // Opens a temp file for read+write and imports it into the proactor.
  // Returns the imported file handle. The caller must submit a close operation
  // or release the file when done.
  iree_async_file_t* ImportTempFileForReadWrite(const std::string& path) {
    iree_async_file_t* file = nullptr;
    IREE_EXPECT_OK(
        ImportNativeFile(proactor_, iree_make_cstring_view(path.c_str()),
                         kNativeFileAccessRead | kNativeFileAccessWrite,
                         IREE_ASYNC_FILE_FLAG_NONE, &file));
    return file;
  }

  // Opens a temp file for reading only and imports it into the proactor.
  iree_async_file_t* ImportTempFileForRead(const std::string& path) {
    iree_async_file_t* file = nullptr;
    IREE_EXPECT_OK(ImportNativeFile(
        proactor_, iree_make_cstring_view(path.c_str()), kNativeFileAccessRead,
        IREE_ASYNC_FILE_FLAG_NONE, &file));
    return file;
  }

 private:
  std::vector<iree::testing::TempFilePath> temp_paths_;
};

class PreparedFileTest : public FileTest {
 protected:
  void TearDown() override {
    if (attached_) {
      ASSERT_NO_FATAL_FAILURE(SetAttached(false));
    }
    iree_async_buffer_registration_state_deinitialize(&state_);
    iree_async_buffer_registration_state_deinitialize(&source_state_);
    iree_async_proactor_release(source_proactor_);
    iree_async_file_release(file_);
    FileTest::TearDown();
  }

  void SetAttached(bool attached) {
    CompletionTracker completion;
    iree_async_buffer_registration_callback_t callback = {
        +[](void* user_data, iree_status_t status) {
          CompletionTracker::Callback(user_data, nullptr, status,
                                      IREE_ASYNC_COMPLETION_FLAG_NONE);
        },
        &completion};
    if (attached) {
      iree_async_proactor_attach_buffer(proactor_, entry_, callback);
    } else {
      iree_async_proactor_detach_buffer(proactor_, entry_, callback);
    }
    PollUntilCondition([&] { return completion.call_count != 0; },
                       "prepared file buffer transition");
    IREE_ASSERT_OK(completion.ConsumeStatus());
    EXPECT_EQ(completion.call_count, 1);
    attached_ = attached;
  }

  // Canonical preparation owner, independently destroyed before file I/O.
  iree_async_proactor_t* source_proactor_ = nullptr;
  // Source preparation and its native pins, when still owned by the source.
  iree_async_buffer_registration_state_t source_state_ = {};
  // Independent destination registration list.
  iree_async_buffer_registration_state_t state_ = {};
  // Cloned destination view borrowed from state_.
  iree_async_buffer_registration_entry_t* entry_ = nullptr;
  // Whether teardown must first detach the cloned view.
  bool attached_ = false;
  // File owned by the destination proactor.
  iree_async_file_t* file_ = nullptr;
  // Payload storage outlives both views and every file operation.
  std::vector<uint8_t> buffer_;
};

TEST_P(PreparedFileTest, CloneReadsAfterSourceProactorDestruction) {
  if (!iree_any_bit_set(
          capabilities_,
          IREE_ASYNC_PROACTOR_CAPABILITY_CONCURRENT_BUFFER_REGISTRATION)) {
    GTEST_SKIP() << "Backend cannot attach prepared registrations";
  }
  IREE_ASSERT_OK_AND_ASSIGN(
      source_proactor_,
      GetParam().factory(iree_async_proactor_options_default()));
  buffer_.resize(4096);
  iree_async_buffer_registration_entry_t* source_entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_prepare_buffer(
      source_proactor_, &source_state_,
      iree_make_byte_span(buffer_.data(), buffer_.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
      &source_entry));

  for (auto access : {IREE_ASYNC_BUFFER_ACCESS_FLAG_NONE,
                      IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_async_proactor_clone_buffer_registration(
            proactor_, &state_, source_entry, access, &entry_));
    EXPECT_EQ(entry_, nullptr);
    EXPECT_TRUE(iree_async_buffer_registration_state_is_empty(&state_));
  }
  IREE_ASSERT_OK(iree_async_proactor_clone_buffer_registration(
      proactor_, &state_, source_entry, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
      &entry_));
  EXPECT_EQ(entry_->region->base_ptr, buffer_.data());
  EXPECT_EQ(entry_->region->length, buffer_.size());
  EXPECT_EQ(entry_->region->access_flags, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE);
  iree_async_buffer_registration_entry_t* widened_entry = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_clone_buffer_registration(
          proactor_, &state_, entry_, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
          &widened_entry));
  EXPECT_EQ(widened_entry, nullptr);
  EXPECT_EQ(state_.head, entry_);
  EXPECT_EQ(entry_->next, nullptr);

  iree_async_buffer_registration_state_deinitialize(&source_state_);
  iree_async_proactor_release(source_proactor_);
  source_proactor_ = nullptr;

  const std::vector<uint8_t> expected(buffer_.size(), 0xBA);
  file_ = ImportTempFileForRead(
      CreateTempFileWithContents(expected.data(), expected.size()));
  ASSERT_NE(file_, nullptr);
  for (int iteration = 0; iteration < 2; ++iteration) {
    ASSERT_NO_FATAL_FAILURE(SetAttached(true));
    std::fill(buffer_.begin(), buffer_.end(), 0);
    CompletionTracker completion;
    iree_async_file_read_operation_t read = {};
    iree_async_operation_initialize(&read.base,
                                    IREE_ASYNC_OPERATION_TYPE_FILE_READ,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    CompletionTracker::Callback, &completion);
    read.file = file_;
    read.buffer = iree_async_span_make(entry_->region, 0, buffer_.size());
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read.base));
    PollUntilCondition([&] { return completion.call_count != 0; },
                       "cloned buffer file read");
    IREE_ASSERT_OK(completion.ConsumeStatus());
    EXPECT_EQ(read.bytes_read, expected.size());
    EXPECT_EQ(buffer_, expected);
    ASSERT_NO_FATAL_FAILURE(SetAttached(false));
  }
}

CTS_REGISTER_TEST_SUITE(PreparedFileTest);

struct RegisteredReadContext {
  bool callback_fired = false;
  int32_t callback_region_ref_count = 0;
  iree_status_t status = iree_ok_status();
  std::vector<uint8_t> contents;
};

static void registered_read_callback(void* user_data,
                                     iree_async_operation_t* operation,
                                     iree_status_t status,
                                     iree_async_completion_flags_t flags) {
  (void)flags;
  auto* context = static_cast<RegisteredReadContext*>(user_data);
  auto* read_op =
      reinterpret_cast<iree_async_file_read_operation_t*>(operation);
  context->callback_fired = true;
  context->status = status;
  context->callback_region_ref_count =
      iree_atomic_ref_count_load(&read_op->buffer.region->ref_count);
  if (iree_status_is_ok(status)) {
    const uint8_t* data = iree_async_span_ptr(read_op->buffer);
    context->contents.assign(data, data + read_op->bytes_read);
  }
}

//===----------------------------------------------------------------------===//
// File open tests
//===----------------------------------------------------------------------===//

// Open an existing file for reading via async open operation.
TEST_P(FileTest, OpenExistingFileForRead) {
  const char kTestData[] = "hello async file";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_open_operation_t open_op;
  memset(&open_op, 0, sizeof(open_op));
  open_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_op.path = path.c_str();
  open_op.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  open_op.opened_file = nullptr;

  CompletionTracker tracker;
  open_op.base.completion_fn = CompletionTracker::Callback;
  open_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  ASSERT_NE(open_op.opened_file, nullptr);

  // Clean up: close via release (file was opened, not submitted for close).
  iree_async_file_release(open_op.opened_file);
}

// Open a non-existent file without CREATE flag fails with NOT_FOUND.
TEST_P(FileTest, OpenNonExistentFileFailsNotFound) {
  iree_async_file_open_operation_t open_op;
  memset(&open_op, 0, sizeof(open_op));
  open_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_op.path = "/tmp/iree_cts_file_nonexistent_should_not_exist_12345";
  open_op.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  open_op.opened_file = nullptr;

  CompletionTracker tracker;
  open_op.base.completion_fn = CompletionTracker::Callback;
  open_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, tracker.ConsumeStatus());
  EXPECT_EQ(open_op.opened_file, nullptr);
}

// Open with CREATE flag creates a new file.
TEST_P(FileTest, OpenWithCreateCreatesNewFile) {
  // Use a unique path that doesn't exist.
  std::string path = CreateEmptyTempFile();
  // Remove it so the open+CREATE can recreate it.
  ASSERT_TRUE(RemoveTempFile(path)) << path;

  iree_async_file_open_operation_t open_op;
  memset(&open_op, 0, sizeof(open_op));
  open_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_op.path = path.c_str();
  open_op.open_flags =
      IREE_ASYNC_FILE_OPEN_FLAG_WRITE | IREE_ASYNC_FILE_OPEN_FLAG_CREATE;
  open_op.opened_file = nullptr;

  CompletionTracker tracker;
  open_op.base.completion_fn = CompletionTracker::Callback;
  open_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  ASSERT_NE(open_op.opened_file, nullptr);

  iree_async_file_release(open_op.opened_file);
}

//===----------------------------------------------------------------------===//
// File read tests
//===----------------------------------------------------------------------===//

// Read entire file contents at offset 0.
TEST_P(FileTest, ReadEntireFile) {
  const char kTestData[] = "hello async world";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, strlen(kTestData));
  EXPECT_EQ(memcmp(read_buffer, kTestData, strlen(kTestData)), 0);

  iree_async_file_release(file);
}

TEST_P(FileTest, ReadRetainsRegisteredSlabThroughCallback) {
  const char kTestData[] = "registered slab lifetime";
  const iree_host_size_t data_length = strlen(kTestData);
  std::string path = CreateTempFileWithContents(kTestData, data_length);

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  iree_async_slab_options_t slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = 4096;
  slab_options.buffer_count = 1;
  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));

  RegisteredReadContext context;
  iree_async_file_read_operation_t read_op = {};
  iree_async_operation_initialize(
      &read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, registered_read_callback, &context);
  read_op.file = file;
  read_op.buffer = iree_async_span_make(region, 0, data_length);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));

  // The accepted operation now owns the region and its retained slab. Drop
  // both caller references before the backend begins the read.
  iree_async_region_release(region);
  iree_async_slab_release(slab);

  PollUntilCondition([&] { return context.callback_fired; },
                     "registered file read completion");
  IREE_EXPECT_OK(context.status);
  EXPECT_EQ(context.callback_region_ref_count, 1);
  EXPECT_EQ(context.contents.size(), data_length);
  EXPECT_EQ(memcmp(context.contents.data(), kTestData, data_length), 0);

  iree_async_file_release(file);
}

// Read at a non-zero offset within the file.
TEST_P(FileTest, ReadAtOffset) {
  const char kTestData[] = "abcdefghij";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[5] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 5;  // Start at 'f'.
  read_op.buffer = iree_async_span_from_ptr(read_buffer, 5);
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, 5u);
  EXPECT_EQ(memcmp(read_buffer, "fghij", 5), 0);

  iree_async_file_release(file);
}

// Read past EOF produces a short read (bytes_read < requested).
TEST_P(FileTest, ReadPastEOFShortRead) {
  const char kTestData[] = "short";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[1024] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  // Short read: file has 5 bytes, we requested 1024.
  EXPECT_EQ(read_op.bytes_read, strlen(kTestData));
  EXPECT_EQ(memcmp(read_buffer, kTestData, strlen(kTestData)), 0);

  iree_async_file_release(file);
}

// Read at offset beyond EOF returns 0 bytes (EOF).
TEST_P(FileTest, ReadBeyondEOFReturnsZeroBytes) {
  const char kTestData[] = "data";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 1000;  // Well past the 4-byte file.
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, 0u);

  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// File write tests
//===----------------------------------------------------------------------===//

// Write data to a file and verify by reading it back.
TEST_P(FileTest, WriteAndReadBack) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Write data at offset 0.
  const char kWriteData[] = "written by proactor";
  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 0;
  write_op.buffer = iree_async_span_from_ptr(const_cast<char*>(kWriteData),
                                             strlen(kWriteData));
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(write_tracker.call_count, 1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());
  EXPECT_EQ(write_op.bytes_written, strlen(kWriteData));

  // Read it back at offset 0.
  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(read_tracker.call_count, 1);
  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, strlen(kWriteData));
  EXPECT_EQ(memcmp(read_buffer, kWriteData, strlen(kWriteData)), 0);

  iree_async_file_release(file);
}

// Write at a non-zero offset, then read the full file to verify.
TEST_P(FileTest, WriteAtOffset) {
  const char kInitialData[] = "AAAAABBBBB";
  std::string path =
      CreateTempFileWithContents(kInitialData, strlen(kInitialData));

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Overwrite the "BBBBB" portion (offset 5).
  const char kOverwrite[] = "CCCCC";
  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 5;
  write_op.buffer = iree_async_span_from_ptr(const_cast<char*>(kOverwrite),
                                             strlen(kOverwrite));
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(write_tracker.call_count, 1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());
  EXPECT_EQ(write_op.bytes_written, strlen(kOverwrite));

  // Read full file and verify.
  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(read_tracker.call_count, 1);
  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, 10u);
  EXPECT_EQ(memcmp(read_buffer, "AAAAACCCCC", 10), 0);

  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// File close tests
//===----------------------------------------------------------------------===//

// Async close completes successfully and the callback fires.
TEST_P(FileTest, AsyncClose) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  iree_async_file_close_operation_t close_op;
  memset(&close_op, 0, sizeof(close_op));
  close_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE;
  close_op.file = file;

  CompletionTracker tracker;
  close_op.base.completion_fn = CompletionTracker::Callback;
  close_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  // File reference was consumed by the close operation. Do not release.
}

// Write, then close via async close, verify the write persisted.
TEST_P(FileTest, WriteFlushClose) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Write data.
  const char kData[] = "persist me";
  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 0;
  write_op.buffer =
      iree_async_span_from_ptr(const_cast<char*>(kData), strlen(kData));
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());

  // Close the file async.
  iree_async_file_close_operation_t close_op;
  memset(&close_op, 0, sizeof(close_op));
  close_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE;
  close_op.file = file;

  CompletionTracker close_tracker;
  close_op.base.completion_fn = CompletionTracker::Callback;
  close_op.base.user_data = &close_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close_op.base));
  PollUntil(/*min_completions=*/1);
  IREE_EXPECT_OK(close_tracker.ConsumeStatus());

  // Re-open and verify the data persisted.
  iree_async_file_t* reopened = ImportTempFileForRead(path);
  ASSERT_NE(reopened, nullptr);

  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = reopened;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, strlen(kData));
  EXPECT_EQ(memcmp(read_buffer, kData, strlen(kData)), 0);

  iree_async_file_release(reopened);
}

//===----------------------------------------------------------------------===//
// File import and lifecycle tests
//===----------------------------------------------------------------------===//

// Import a file, retain, release -- verify reference counting works.
TEST_P(FileTest, ImportRetainRelease) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Retain bumps refcount.
  iree_async_file_retain(file);

  // First release: refcount goes from 2 to 1.
  iree_async_file_release(file);

  // File should still be usable. Verify with a read (0-byte read at offset 0).
  uint8_t read_buffer[1] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Final release destroys.
  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// Concurrent read operations
//===----------------------------------------------------------------------===//

// Multiple reads at different offsets on the same file complete independently.
TEST_P(FileTest, ConcurrentReadsAtDifferentOffsets) {
  const char kTestData[] = "0123456789abcdefghij";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  // Set up two concurrent reads at different offsets.
  uint8_t buffer_a[5] = {0};
  uint8_t buffer_b[5] = {0};

  iree_async_file_read_operation_t read_a;
  memset(&read_a, 0, sizeof(read_a));
  read_a.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_a.file = file;
  read_a.offset = 0;
  read_a.buffer = iree_async_span_from_ptr(buffer_a, 5);

  iree_async_file_read_operation_t read_b;
  memset(&read_b, 0, sizeof(read_b));
  read_b.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_b.file = file;
  read_b.offset = 10;
  read_b.buffer = iree_async_span_from_ptr(buffer_b, 5);

  CompletionTracker tracker_a;
  read_a.base.completion_fn = CompletionTracker::Callback;
  read_a.base.user_data = &tracker_a;

  CompletionTracker tracker_b;
  read_b.base.completion_fn = CompletionTracker::Callback;
  read_b.base.user_data = &tracker_b;

  // Submit as a batch.
  iree_async_operation_t* ops[] = {&read_a.base, &read_b.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/2);

  EXPECT_EQ(tracker_a.call_count, 1);
  IREE_EXPECT_OK(tracker_a.ConsumeStatus());
  EXPECT_EQ(read_a.bytes_read, 5u);
  EXPECT_EQ(memcmp(buffer_a, "01234", 5), 0);

  EXPECT_EQ(tracker_b.call_count, 1);
  IREE_EXPECT_OK(tracker_b.ConsumeStatus());
  EXPECT_EQ(read_b.bytes_read, 5u);
  EXPECT_EQ(memcmp(buffer_b, "abcde", 5), 0);

  iree_async_file_release(file);
}

TEST_P(FileTest, BufferedFileEventChainPreservesIndependentProgress) {
  const char initial[] = "abcdefghijkl";
  std::string path = CreateTempFileWithContents(initial, sizeof(initial) - 1);
  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  char payload[] = "WXYZ";
  char dependent_result[4] = {};
  char independent_result[4] = {};
  CompletionTracker write_completion;
  CompletionTracker wait_completion;
  CompletionTracker dependent_completion;
  CompletionTracker independent_completion;
  iree_async_file_write_operation_t write = {};
  iree_async_operation_initialize(
      &write.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_LINKED, CompletionTracker::Callback,
      &write_completion);
  write.file = file;
  write.buffer = iree_async_span_from_ptr(payload, sizeof(dependent_result));
  iree_async_event_wait_operation_t wait = {};
  iree_async_operation_initialize(
      &wait.base, IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT,
      IREE_ASYNC_OPERATION_FLAG_LINKED, CompletionTracker::Callback,
      &wait_completion);
  wait.event = event;
  iree_async_file_read_operation_t dependent = {};
  iree_async_operation_initialize(
      &dependent.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, CompletionTracker::Callback,
      &dependent_completion);
  dependent.file = file;
  dependent.buffer =
      iree_async_span_from_ptr(dependent_result, sizeof(dependent_result));
  iree_async_file_read_operation_t independent = {};
  iree_async_operation_initialize(
      &independent.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, CompletionTracker::Callback,
      &independent_completion);
  independent.file = file;
  independent.offset = 8;
  independent.buffer =
      iree_async_span_from_ptr(independent_result, sizeof(independent_result));
  iree_async_operation_t* operations[] = {&write.base, &wait.base,
                                          &dependent.base, &independent.base};
  IREE_ASSERT_OK(iree_async_proactor_submit(
      proactor_,
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations))));

  PollUntilCondition([&] { return independent_completion.call_count == 1; });
  EXPECT_EQ(wait_completion.call_count, 0);
  EXPECT_EQ(dependent_completion.call_count, 0);
  EXPECT_EQ(memcmp(independent_result, "ijkl", 4), 0);

  iree_async_event_set(event);
  PollUntilCondition([&] {
    return write_completion.call_count == 1 &&
           wait_completion.call_count == 1 &&
           dependent_completion.call_count == 1;
  });
  IREE_EXPECT_OK(write_completion.ConsumeStatus());
  IREE_EXPECT_OK(wait_completion.ConsumeStatus());
  IREE_EXPECT_OK(dependent_completion.ConsumeStatus());
  IREE_EXPECT_OK(independent_completion.ConsumeStatus());
  EXPECT_EQ(memcmp(dependent_result, payload, 4), 0);
  iree_async_event_release(event);
  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// Larger I/O tests
//===----------------------------------------------------------------------===//

// Write and read back a larger buffer to exercise multi-page I/O paths.
TEST_P(FileTest, LargeWriteAndReadBack) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Create a 64KB buffer with a known pattern.
  static constexpr size_t kBufferSize = 64 * 1024;
  std::vector<uint8_t> write_buffer(kBufferSize);
  for (size_t i = 0; i < kBufferSize; ++i) {
    write_buffer[i] = static_cast<uint8_t>(i & 0xFF);
  }

  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 0;
  write_op.buffer = iree_async_span_from_ptr(write_buffer.data(), kBufferSize);
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(write_tracker.call_count, 1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());
  EXPECT_EQ(write_op.bytes_written, kBufferSize);

  // Read it back.
  std::vector<uint8_t> read_buffer(kBufferSize, 0);
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer.data(), kBufferSize);
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(read_tracker.call_count, 1);
  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, kBufferSize);
  EXPECT_EQ(write_buffer, read_buffer);

  iree_async_file_release(file);
}

CTS_REGISTER_TEST_SUITE(FileTest);

}  // namespace iree::async::cts
