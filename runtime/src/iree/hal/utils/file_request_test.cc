// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/file_request.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Controls only native acceptance and completion. The placed request, geometry,
// continuation, and terminal-status owner are the production implementation.
struct NativeFile {
  NativeFile() {
    proactor_vtable.submit = Submit;
    iree_async_proactor_initialize(&proactor_vtable, IREE_SV("request test"),
                                   iree_allocator_system(), &proactor);
    native.proactor = &proactor;
    file_vtable.async_handle = +[](iree_hal_file_t* file) {
      return &reinterpret_cast<NativeFile*>(file)->native;
    };
    file_vtable.requirements = +[](iree_hal_file_t* file) {
      return static_cast<const iree_async_file_requirements_t*>(
          &reinterpret_cast<NativeFile*>(file)->requirements);
    };
    iree_hal_resource_initialize(&file_vtable, &resource);
  }

  iree_hal_file_t* file() {
    return reinterpret_cast<iree_hal_file_t*>(&resource);
  }

  static iree_status_t Submit(iree_async_proactor_t* proactor,
                              iree_async_operation_list_t operations) {
    auto* self = iree_containerof(proactor, NativeFile, proactor);
    self->submission_sizes.push_back(operations.count);
    if (self->submission_failure != IREE_STATUS_OK) {
      return iree_status_from_code(self->submission_failure);
    }
    for (iree_host_size_t i = 0; i < operations.count; ++i) {
      self->pending.push_back(operations.values[i]);
    }
    return iree_ok_status();
  }

  void Complete(size_t index, size_t length,
                iree_status_code_t code = IREE_STATUS_OK) {
    ASSERT_LT(index, pending.size());
    auto* operation = pending[index];
    pending.erase(pending.begin() + index);
    if (code == IREE_STATUS_OK) {
      if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
        auto* read =
            reinterpret_cast<iree_async_file_read_operation_t*>(operation);
        ASSERT_LE(length, read->buffer.length);
        std::memcpy(iree_async_span_ptr(read->buffer),
                    bytes.data() + read->offset, length);
        read->bytes_read = length;
      } else {
        auto* write =
            reinterpret_cast<iree_async_file_write_operation_t*>(operation);
        ASSERT_LE(length, write->buffer.length);
        std::memcpy(bytes.data() + write->offset,
                    iree_async_span_ptr(write->buffer), length);
        write->bytes_written = length;
      }
    }
    operation->completion_fn(operation->user_data, operation,
                             iree_status_from_code(code),
                             IREE_ASYNC_COMPLETION_FLAG_NONE);
  }

  // HAL file identity; first so native accessors can recover the fixture.
  iree_hal_resource_t resource = {};
  // Only the immutable facts used by a request are supplied here.
  iree_hal_file_vtable_t file_vtable = {};
  // File-specific read/write geometry, stable through every completion.
  iree_async_file_requirements_t requirements = {{1, 1, 1}, {1, 1, 1}};
  // Native file borrowed by the production request.
  iree_async_file_t native = {};
  // Native acceptance dependency, independent of request progress.
  iree_async_proactor_t proactor = {};
  // Submission-only platform adapter.
  iree_async_proactor_vtable_t proactor_vtable = {};
  // Actual bytes copied by controlled native completions.
  std::array<uint8_t, 128> bytes = {};
  // Accepted operations; callbacks may add partial continuations.
  std::vector<iree_async_operation_t*> pending;
  // Admission calls distinguish one batch from repeated scalar submissions.
  std::vector<iree_host_size_t> submission_sizes;
  // Synchronous native rejection, with no accepted operation or callback.
  iree_status_code_t submission_failure = IREE_STATUS_OK;
};

struct Completion {
  static void Complete(void* user_data, iree_status_t status) {
    auto* self = static_cast<Completion*>(user_data);
    ++self->count;
    self->code = iree_status_code(status);
    iree_status_free(status);
  }
  // Number of terminal callbacks for this independently owned request.
  size_t count = 0;
  // Last terminal status, independent of another request's progress.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
};

class FileRequestTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < bytes_.size(); ++i) {
      file_.bytes[i] = uint8_t(i * 3 + 7);
      bytes_[i] = uint8_t(i * 5 + 19);
    }
  }
  void TearDown() override {
    // Placed request storage remains live until accepted native work retires,
    // including after a failed assertion in the test body.
    while (!file_.pending.empty()) {
      file_.Complete(0, 0, IREE_STATUS_CANCELLED);
    }
  }
  iree_hal_memory_access_t access() const {
    return GetParam() ? IREE_HAL_MEMORY_ACCESS_WRITE
                      : IREE_HAL_MEMORY_ACCESS_READ;
  }
  iree_status_t Prepare(size_t index, uint64_t offset, size_t length,
                        iree_async_span_t span) {
    return iree_hal_file_request_prepare(file_.file(), access(), offset, length,
                                         span, &requests_[index]);
  }
  iree_status_t Submit(size_t index) {
    return iree_hal_file_request_submit(
        &requests_[index], {Completion::Complete, &completions_[index]});
  }
  iree_status_t SubmitBatch() {
    std::array<iree_async_operation_t*, 2> operations;
    for (size_t i = 0; i < operations.size(); ++i) {
      operations[i] = iree_hal_file_request_bind(
          &requests_[i], {Completion::Complete, &completions_[i]});
    }
    return iree_async_proactor_submit(&file_.proactor,
                                      {operations.data(), operations.size()});
  }
  // Controlled native file and submission dependency.
  NativeFile file_;
  // Caller-owned payload and independent request lifetimes.
  alignas(16) std::array<uint8_t, 128> bytes_ = {};
  // Requests share no mutable progress, even for the same native file.
  std::array<iree_hal_file_request_t, 2> requests_ = {};
  // Terminal observations for the two captured requests.
  std::array<Completion, 2> completions_ = {};
  // Optional registered span remains live through accepted completion.
  iree_async_region_t region_ = {};
};

TEST_P(FileRequestTest, RegisteredPartialRequestsRetireIndependently) {
  region_.proactor = &file_.proactor;
  region_.base_ptr = bytes_.data();
  region_.length = bytes_.size();
  region_.buffer_size = 32;
  const auto original_file = file_.bytes;
  const auto original_buffer = bytes_;
  IREE_ASSERT_OK(Prepare(0, 17, 48, iree_async_span_make(&region_, 13, 48)));
  IREE_ASSERT_OK(
      Prepare(1, 65, 16, iree_async_span_from_ptr(bytes_.data() + 80, 16)));
  EXPECT_EQ(iree_hal_file_request_span(&requests_[0]).length, 19u);
  IREE_ASSERT_OK(Submit(0));
  IREE_ASSERT_OK(Submit(1));
  file_.Complete(1, 16);
  EXPECT_EQ(completions_[0].count, 0u);
  EXPECT_EQ(completions_[1].count, 1u);
  file_.Complete(0, 7);
  ASSERT_EQ(file_.pending.size(), 1u);
  const auto& span =
      GetParam() ? requests_[0].write.buffer : requests_[0].read.buffer;
  EXPECT_EQ(span.region, &region_);
  EXPECT_EQ(span.offset, 20u);
  EXPECT_EQ(span.length, 12u);
  EXPECT_EQ(GetParam() ? requests_[0].write.offset : requests_[0].read.offset,
            24u);
  EXPECT_EQ(iree_hal_file_request_span(&requests_[0]).offset, 13u);
  EXPECT_EQ(iree_hal_file_request_span(&requests_[0]).length, 19u);
  file_.Complete(0, 12);
  EXPECT_EQ(iree_hal_file_request_span(&requests_[0]).offset, 13u);
  EXPECT_EQ(iree_hal_file_request_span(&requests_[0]).length, 19u);
  EXPECT_EQ(completions_[0].count, 1u);
  EXPECT_EQ(completions_[0].code, IREE_STATUS_OK);
  EXPECT_EQ(completions_[1].code, IREE_STATUS_OK);
  auto expected_file = original_file;
  auto expected_buffer = original_buffer;
  if (GetParam()) {
    std::memcpy(expected_file.data() + 17, original_buffer.data() + 13, 19);
    std::memcpy(expected_file.data() + 65, original_buffer.data() + 80, 16);
  } else {
    std::memcpy(expected_buffer.data() + 13, original_file.data() + 17, 19);
    std::memcpy(expected_buffer.data() + 80, original_file.data() + 65, 16);
  }
  EXPECT_EQ(file_.bytes, expected_file);
  EXPECT_EQ(bytes_, expected_buffer);
}

TEST_P(FileRequestTest, RejectionDoesNotCompleteAndAcceptedFailureDoes) {
  IREE_ASSERT_OK(
      Prepare(0, 0, 16, iree_async_span_from_ptr(bytes_.data(), 16)));
  file_.submission_failure = IREE_STATUS_RESOURCE_EXHAUSTED;
  EXPECT_THAT(
      iree::Status(Submit(0)),
      iree::testing::status::StatusIs(iree::StatusCode::kResourceExhausted));
  EXPECT_EQ(completions_[0].count, 0u);
  EXPECT_TRUE(file_.pending.empty());
  file_.submission_failure = IREE_STATUS_OK;
  IREE_ASSERT_OK(
      Prepare(0, 0, 16, iree_async_span_from_ptr(bytes_.data(), 16)));
  IREE_ASSERT_OK(Submit(0));
  file_.Complete(0, 7);
  file_.Complete(0, 0, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(requests_[0].bytes_transferred, 7u);
  EXPECT_EQ(completions_[0].count, 1u);
  EXPECT_EQ(completions_[0].code, IREE_STATUS_DATA_LOSS);
}

TEST_P(FileRequestTest, BatchAdmitsOnceAndRetiresPartialIoIndependently) {
  const auto original_file = file_.bytes;
  const auto original_buffer = bytes_;
  IREE_ASSERT_OK(
      Prepare(0, 19, 16, iree_async_span_from_ptr(bytes_.data() + 7, 16)));
  IREE_ASSERT_OK(
      Prepare(1, 73, 24, iree_async_span_from_ptr(bytes_.data() + 49, 24)));
  IREE_ASSERT_OK(SubmitBatch());
  EXPECT_EQ(file_.submission_sizes, (std::vector<iree_host_size_t>{2}));
  ASSERT_EQ(file_.pending.size(), 2u);

  file_.Complete(1, 9);
  EXPECT_EQ(completions_[1].count, 0u);
  EXPECT_EQ(file_.submission_sizes, (std::vector<iree_host_size_t>{2, 1}));
  file_.Complete(0, 16);
  EXPECT_EQ(completions_[0].count, 1u);
  EXPECT_EQ(completions_[1].count, 0u);
  file_.Complete(0, 15);
  EXPECT_EQ(completions_[1].count, 1u);
  EXPECT_EQ(completions_[0].code, IREE_STATUS_OK);
  EXPECT_EQ(completions_[1].code, IREE_STATUS_OK);
  EXPECT_TRUE(file_.pending.empty());

  auto expected_file = original_file;
  auto expected_buffer = original_buffer;
  if (GetParam()) {
    std::memcpy(expected_file.data() + 19, original_buffer.data() + 7, 16);
    std::memcpy(expected_file.data() + 73, original_buffer.data() + 49, 24);
  } else {
    std::memcpy(expected_buffer.data() + 7, original_file.data() + 19, 16);
    std::memcpy(expected_buffer.data() + 49, original_file.data() + 73, 24);
  }
  EXPECT_EQ(file_.bytes, expected_file);
  EXPECT_EQ(bytes_, expected_buffer);
}

TEST_P(FileRequestTest, RejectedBatchRetainsAllRequestsAndCanBeResubmitted) {
  for (size_t i = 0; i < requests_.size(); ++i) {
    IREE_ASSERT_OK(Prepare(
        i, i * 32, 16, iree_async_span_from_ptr(bytes_.data() + i * 32, 16)));
  }
  file_.submission_failure = IREE_STATUS_RESOURCE_EXHAUSTED;
  EXPECT_THAT(
      iree::Status(SubmitBatch()),
      iree::testing::status::StatusIs(iree::StatusCode::kResourceExhausted));
  EXPECT_TRUE(file_.pending.empty());
  EXPECT_EQ(completions_[0].count, 0u);
  EXPECT_EQ(completions_[1].count, 0u);

  file_.submission_failure = IREE_STATUS_OK;
  IREE_ASSERT_OK(SubmitBatch());
  file_.Complete(0, 0, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(completions_[0].count, 1u);
  EXPECT_EQ(completions_[0].code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(completions_[1].count, 0u);
  // A sibling failure does not release another request's accepted native use.
  ASSERT_EQ(file_.pending.size(), 1u);
  file_.Complete(0, 16);
  EXPECT_EQ(completions_[1].count, 1u);
  EXPECT_EQ(completions_[1].code, IREE_STATUS_OK);
  EXPECT_TRUE(file_.pending.empty());
}

TEST_P(FileRequestTest, ZeroProgressFails) {
  IREE_ASSERT_OK(
      Prepare(0, 0, 16, iree_async_span_from_ptr(bytes_.data(), 16)));
  IREE_ASSERT_OK(Submit(0));
  file_.Complete(0, 7);
  file_.Complete(0, 0);
  EXPECT_EQ(completions_[0].count, 1u);
  EXPECT_EQ(completions_[0].code, IREE_STATUS_OUT_OF_RANGE);
}

TEST_P(FileRequestTest, RejectedContinuationRetiresAcceptedPrefix) {
  IREE_ASSERT_OK(
      Prepare(0, 16, 16, iree_async_span_from_ptr(bytes_.data(), 16)));
  IREE_ASSERT_OK(Submit(0));
  file_.submission_failure = IREE_STATUS_RESOURCE_EXHAUSTED;
  file_.Complete(0, 7);
  EXPECT_EQ(requests_[0].bytes_transferred, 7u);
  EXPECT_EQ(completions_[0].count, 1u);
  EXPECT_EQ(completions_[0].code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_TRUE(file_.pending.empty());
  const auto span = iree_hal_file_request_span(&requests_[0]);
  EXPECT_EQ(iree_async_span_ptr(span), bytes_.data());
  EXPECT_EQ(span.length, 16u);
}

TEST_P(FileRequestTest, UnalignedShortIoIsNotResubmitted) {
  file_.requirements = {{8, 8, 8}, {8, 8, 8}};
  IREE_ASSERT_OK(
      Prepare(1, 0, 16, iree_async_span_from_ptr(bytes_.data(), 16)));
  IREE_ASSERT_OK(Submit(1));
  file_.Complete(0, 7);
  EXPECT_EQ(completions_[1].count, 1u);
  EXPECT_EQ(completions_[1].code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_TRUE(file_.pending.empty());
}

INSTANTIATE_TEST_SUITE_P(ReadAndWrite, FileRequestTest, ::testing::Bool());

}  // namespace
