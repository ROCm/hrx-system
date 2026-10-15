// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/file_transfer.h"

#include <array>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "iree/async/file.h"
#include "iree/async/proactor.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Controls the platform I/O dependency so short writes and resubmission errors
// are deterministic. The transfer implementation and native buffer view are
// production code; actual platform files are exercised by the native HAL CTS.
struct FileProvider {
  struct Request {
    // File-relative byte position submitted to the platform.
    uint64_t offset;
    // Requested remaining length.
    iree_host_size_t length;
  };

  FileProvider() {
    proactor_vtable.submit = Submit;
    iree_async_proactor_initialize(&proactor_vtable, IREE_SV("file test"),
                                   iree_allocator_system(), &proactor);
    native_file.proactor = &proactor;
    file_vtable.destroy = Destroy;
    file_vtable.async_handle = AsyncHandle;
    file_vtable.requirements = Requirements;
    iree_hal_resource_initialize(&file_vtable, &resource);
    for (size_t i = 0; i < contents.size(); ++i) {
      contents[i] = uint8_t(i * 3);
    }
  }

  iree_hal_file_t* file() {
    return reinterpret_cast<iree_hal_file_t*>(&resource);
  }

  static iree_async_file_t* AsyncHandle(iree_hal_file_t* file) {
    auto* self = iree_containerof(reinterpret_cast<iree_hal_resource_t*>(file),
                                  FileProvider, resource);
    return &self->native_file;
  }

  static void Destroy(iree_hal_file_t* file) {
    auto* self = iree_containerof(reinterpret_cast<iree_hal_resource_t*>(file),
                                  FileProvider, resource);
    EXPECT_EQ(self->pending, nullptr);
    ++self->release_count;
  }

  static const iree_async_file_requirements_t* Requirements(
      iree_hal_file_t* file) {
    auto* self = iree_containerof(reinterpret_cast<iree_hal_resource_t*>(file),
                                  FileProvider, resource);
    return &self->requirements;
  }

  static iree_status_t Submit(iree_async_proactor_t* proactor,
                              iree_async_operation_list_t operations) {
    auto* self = iree_containerof(proactor, FileProvider, proactor);
    EXPECT_EQ(operations.count, 1u);
    EXPECT_EQ(self->pending, nullptr);
    if (self->submit_code != IREE_STATUS_OK) {
      return iree_make_status(self->submit_code, "platform submission failed");
    }
    self->pending = operations.values[0];
    if (self->pending->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
      auto* read =
          reinterpret_cast<iree_async_file_read_operation_t*>(self->pending);
      self->requests.push_back({read->offset, read->buffer.length});
    } else {
      EXPECT_EQ(self->pending->type, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE);
      auto* write =
          reinterpret_cast<iree_async_file_write_operation_t*>(self->pending);
      self->requests.push_back({write->offset, write->buffer.length});
    }
    return iree_ok_status();
  }

  void Complete(iree_host_size_t byte_count,
                iree_status_code_t code = IREE_STATUS_OK) {
    ASSERT_NE(pending, nullptr);
    auto* operation = pending;
    pending = nullptr;
    if (code == IREE_STATUS_OK) {
      if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
        auto* read =
            reinterpret_cast<iree_async_file_read_operation_t*>(operation);
        ASSERT_LE(byte_count, read->buffer.length);
        ASSERT_LE(read->offset + byte_count, contents.size());
        memcpy(iree_async_span_ptr(read->buffer),
               contents.data() + read->offset, byte_count);
        read->bytes_read = byte_count;
      } else {
        auto* write =
            reinterpret_cast<iree_async_file_write_operation_t*>(operation);
        ASSERT_LE(byte_count, write->buffer.length);
        ASSERT_LE(write->offset + byte_count, contents.size());
        memcpy(contents.data() + write->offset,
               iree_async_span_ptr(write->buffer), byte_count);
        write->bytes_written = byte_count;
      }
    }
    operation->completion_fn(
        operation->user_data, operation,
        code == IREE_STATUS_OK ? iree_ok_status()
                               : iree_make_status(code, "platform I/O failed"),
        IREE_ASYNC_COMPLETION_FLAG_NONE);
  }

  // Controlled platform submission owner.
  iree_async_proactor_t proactor = {};
  // Sole implemented platform operation, submission.
  iree_async_proactor_vtable_t proactor_vtable = {};
  // Imported native file borrowed by the HAL file resource.
  iree_async_file_t native_file = {};
  // Ref-counted HAL file dependency.
  iree_hal_resource_t resource = {};
  // File operations used by the shared transfer.
  iree_hal_file_vtable_t file_vtable = {};
  // Retained native requirements for each submitted request.
  iree_async_file_requirements_t requirements = {{1, 1, 1}, {1, 1, 1}};
  // Platform file bytes.
  std::array<uint8_t, 128> contents;
  // Accepted request awaiting an explicit completion.
  iree_async_operation_t* pending = nullptr;
  // Observed platform requests in submission order.
  std::vector<Request> requests;
  // External failure returned by the next submission, or OK.
  iree_status_code_t submit_code = IREE_STATUS_OK;
  // Number of final HAL file releases.
  size_t release_count = 0;
};

struct BufferStorage {
  struct CacheOperation {
    // Native cache operation requested by the production buffer view.
    amdf_host_cache_operation_t operation;
    // Byte position within the native mapping.
    uint64_t offset;
    // Exact byte count of the requested maintenance.
    uint64_t length;
  };

  static amdf_status_t AMDF_CALL CacheControl(
      amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
      uint64_t offset, uint64_t length) {
    auto* self = reinterpret_cast<BufferStorage*>(mapping);
    self->cache_operations.push_back({operation, offset, length});
    return self->cache_status;
  }

  static void Release(void* user_data, iree_hal_buffer_t* buffer) {
    ++static_cast<BufferStorage*>(user_data)->release_count;
  }

  // Native backing bytes, with guard space around the logical buffer.
  alignas(64) std::array<uint8_t, 256> contents;
  // Mapping information borrowed by the production buffer view.
  iree_hal_amd_fabric_memory_t memory = {};
  // Native dependency providing cache maintenance.
  amdf_api_t api = {};
  // Persistent host execution binding independent of public map grants.
  iree_hal_buffer_native_binding_t binding = {};
  // Observed native cache operations.
  std::vector<CacheOperation> cache_operations;
  // Native cache result under test.
  amdf_status_t cache_status = AMDF_STATUS_OK;
  // Number of final backing-view releases.
  size_t release_count = 0;
};

class FileTransferTest : public ::testing::TestWithParam<
                             std::tuple<iree_hal_memory_access_t, bool>> {
 protected:
  iree_hal_memory_access_t file_access() const {
    return std::get<0>(GetParam());
  }
  bool is_direct() const { return std::get<1>(GetParam()); }

  void SetUp() override {
    provider_.native_file.flags =
        is_direct() ? IREE_ASYNC_FILE_FLAG_DIRECT : IREE_ASYNC_FILE_FLAG_NONE;
    storage_.contents.fill(0xA5);
    storage_.memory.contents =
        iree_make_byte_span(storage_.contents.data(), storage_.contents.size());
    storage_.memory.mapping = reinterpret_cast<amdf_host_mapping_t*>(&storage_);
    storage_.api.host_mapping_cache_control = BufferStorage::CacheControl;
    storage_.binding.host_pointer = storage_.contents.data();
    iree_hal_amd_fabric_buffer_storage_t backing = {
        .memory = &storage_.memory,
        .bindings = &storage_.binding,
        .host_binding_index = 0,
        .offset = 8,
    };
    iree_hal_buffer_t* root = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_fabric_buffer_wrap(
        {}, &storage_.api, backing, 192, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        {BufferStorage::Release, &storage_}, iree_allocator_system(), &root));
    iree_status_t status = iree_hal_buffer_subspan(
        root, 16, 96, iree_allocator_system(), &buffer_);
    iree_hal_buffer_release(root);
    IREE_ASSERT_OK(status);
    if (file_access() == IREE_HAL_MEMORY_ACCESS_WRITE) {
      for (size_t i = 0; i < kLength; ++i) {
        storage_.contents[kNativeOffset + i] = uint8_t(i * 7);
      }
    }
    expected_buffer_ = storage_.contents;
    expected_file_ = provider_.contents;
    if (file_access() == IREE_HAL_MEMORY_ACCESS_READ) {
      memcpy(expected_buffer_.data() + kNativeOffset,
             expected_file_.data() + kFileOffset, kLength);
    } else {
      memcpy(expected_file_.data() + kFileOffset,
             expected_buffer_.data() + kNativeOffset, kLength);
    }
  }

  void TearDown() override {
    // A failed assertion must retire accepted mock I/O before placed state is
    // released, just as a platform completion would do in production.
    if (provider_.pending) {
      provider_.Complete(0, IREE_STATUS_CANCELLED);
    }
    if (initialized_) {
      iree_hal_amd_fabric_file_transfer_deinitialize(&transfer_);
    }
    iree_hal_buffer_release(buffer_);
    if (file_owned_) {
      iree_hal_file_release(provider_.file());
    }
    EXPECT_EQ(provider_.release_count, 1u);
    EXPECT_EQ(storage_.release_count, 1u);
  }

  void Initialize(uint64_t file_offset = kFileOffset,
                  iree_device_size_t buffer_offset = kBufferOffset,
                  iree_allocator_t host_allocator = iree_allocator_system()) {
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_initialize(
        provider_.file(), file_access(), file_offset, buffer_, buffer_offset,
        kLength, host_allocator, &transfer_));
    initialized_ = true;
  }

  iree_status_t Submit() {
    return iree_hal_amd_fabric_file_transfer_submit(
        &transfer_, {+[](void* user_data, iree_status_t status) {
                       auto* self = static_cast<FileTransferTest*>(user_data);
                       ++self->completion_count_;
                       self->completion_code_ = iree_status_code(status);
                       iree_status_free(status);
                     },
                     this});
  }

  // Tested file-relative range, distinct from the buffer-relative range.
  static constexpr size_t kFileOffset = 19;
  // Position within the logical subspan.
  static constexpr size_t kBufferOffset = 7;
  // Native view offset plus subspan offset plus logical operation offset.
  static constexpr size_t kNativeOffset = 8 + 16 + kBufferOffset;
  // Transfer extent covering several deliberately short native requests.
  static constexpr size_t kLength = 32;
  // Controlled platform I/O dependency.
  FileProvider provider_;
  // Controlled native mapping dependency.
  BufferStorage storage_;
  // Logical view with no public mapping permissions.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Shared production operation under test.
  iree_hal_amd_fabric_file_transfer_t transfer_ = {};
  // True after operand capture owns references.
  bool initialized_ = false;
  // True while the fixture owns its initial file reference.
  bool file_owned_ = true;
  // Number of terminal I/O callbacks.
  size_t completion_count_ = 0;
  // Captured terminal I/O result.
  iree_status_code_t completion_code_ = IREE_STATUS_UNKNOWN;
  // Exact expected device backing including unchanged guards.
  std::array<uint8_t, 256> expected_buffer_;
  // Exact expected file bytes including unchanged guards.
  std::array<uint8_t, 128> expected_file_;
};

TEST_P(FileTransferTest, PartialTransfersRetainOperandsAndPublishExactRanges) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  iree_hal_buffer_release(buffer_);
  buffer_ = nullptr;
  iree_hal_file_release(provider_.file());
  file_owned_ = false;
  EXPECT_EQ(provider_.release_count, 0u);
  EXPECT_EQ(storage_.release_count, 0u);
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  EXPECT_EQ(storage_.cache_operations.size(),
            file_access() == IREE_HAL_MEMORY_ACCESS_READ ? 0u : 1u);
  IREE_ASSERT_OK(Submit());
  provider_.Complete(7);
  EXPECT_EQ(completion_count_, 0u);
  provider_.Complete(11);
  EXPECT_EQ(completion_count_, 0u);
  provider_.Complete(14);
  ASSERT_EQ(provider_.requests.size(), 3u);
  EXPECT_EQ(provider_.requests[0].offset, kFileOffset);
  EXPECT_EQ(provider_.requests[0].length, kLength);
  EXPECT_EQ(provider_.requests[1].offset, kFileOffset + 7);
  EXPECT_EQ(provider_.requests[1].length, kLength - 7);
  EXPECT_EQ(provider_.requests[2].offset, kFileOffset + 18);
  EXPECT_EQ(provider_.requests[2].length, kLength - 18);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_finish(&transfer_));
  ASSERT_EQ(storage_.cache_operations.size(), 1u);
  EXPECT_EQ(storage_.cache_operations[0].operation,
            file_access() == IREE_HAL_MEMORY_ACCESS_READ
                ? AMDF_HOST_CACHE_OPERATION_FLUSH
                : AMDF_HOST_CACHE_OPERATION_INVALIDATE);
  EXPECT_EQ(storage_.cache_operations[0].offset, kNativeOffset);
  EXPECT_EQ(storage_.cache_operations[0].length, kLength);
  EXPECT_EQ(storage_.contents, expected_buffer_);
  EXPECT_EQ(provider_.contents, expected_file_);
}

TEST_P(FileTransferTest, ZeroProgressFailsTheExactRange) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  IREE_ASSERT_OK(Submit());
  provider_.Complete(7);
  provider_.Complete(0);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_OUT_OF_RANGE);
  EXPECT_EQ(transfer_.async.bytes_transferred, 7u);
  EXPECT_EQ(provider_.pending, nullptr);
}

TEST_P(FileTransferTest, EngineStagingDoesNotRequireOrTouchAHostBufferView) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  const auto original_buffer = storage_.contents;
  std::array<uint8_t, 16> staging = {};
  std::array<uint8_t, kLength> gathered = {};
  for (size_t offset = 0; offset < kLength; offset += staging.size()) {
    if (file_access() == IREE_HAL_MEMORY_ACCESS_WRITE) {
      memcpy(staging.data(), original_buffer.data() + kNativeOffset + offset,
             staging.size());
    }
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare_chunk(
        &transfer_, iree_make_byte_span(staging.data(), staging.size())));
    EXPECT_EQ(transfer_.async.chunk.offset, offset);
    EXPECT_EQ(transfer_.async.chunk.contents.data, staging.data());
    IREE_ASSERT_OK(Submit());
    provider_.Complete(7);
    provider_.Complete(9);
    EXPECT_EQ(completion_count_, offset / staging.size() + 1);
    EXPECT_EQ(completion_code_, IREE_STATUS_OK);
    EXPECT_EQ(transfer_.async.bytes_transferred, offset + staging.size());
    if (file_access() == IREE_HAL_MEMORY_ACCESS_READ) {
      memcpy(gathered.data() + offset, staging.data(), staging.size());
    }
  }
  EXPECT_EQ(transfer_.buffer_contents.data, nullptr);
  EXPECT_EQ(transfer_.async.staging.data, nullptr);
  EXPECT_TRUE(storage_.cache_operations.empty());
  EXPECT_EQ(storage_.contents, original_buffer);
  EXPECT_EQ(provider_.contents, expected_file_);
  if (file_access() == IREE_HAL_MEMORY_ACCESS_READ) {
    EXPECT_EQ(memcmp(gathered.data(), provider_.contents.data() + kFileOffset,
                     kLength),
              0);
  }
}

TEST_P(FileTransferTest, AlignedPartialCompletionContinues) {
  auto& requirements = file_access() == IREE_HAL_MEMORY_ACCESS_READ
                           ? provider_.requirements.read
                           : provider_.requirements.write;
  requirements = {16, 8, 4};
  ASSERT_NO_FATAL_FAILURE(Initialize(/*file_offset=*/32, /*buffer_offset=*/8));
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  IREE_ASSERT_OK(Submit());
  provider_.Complete(16);
  ASSERT_EQ(provider_.requests.size(), 2u);
  EXPECT_EQ(provider_.requests[1].offset, 48u);
  EXPECT_EQ(provider_.requests[1].length, 16u);
  EXPECT_EQ(completion_count_, 0u);
  provider_.Complete(16);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_finish(&transfer_));
  EXPECT_EQ(std::memcmp(storage_.contents.data() + 32,
                        provider_.contents.data() + 32, 32),
            0);
}

TEST_P(FileTransferTest, UnalignedPartialCompletionDoesNotResubmit) {
  auto& requirements = file_access() == IREE_HAL_MEMORY_ACCESS_READ
                           ? provider_.requirements.read
                           : provider_.requirements.write;
  requirements = {16, 8, 4};
  ASSERT_NO_FATAL_FAILURE(Initialize(/*file_offset=*/32, /*buffer_offset=*/8));
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  IREE_ASSERT_OK(Submit());
  provider_.Complete(4);
  EXPECT_EQ(provider_.requests.size(), 1u);
  EXPECT_EQ(provider_.pending, nullptr);
  EXPECT_EQ(transfer_.async.bytes_transferred, 4u);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_INVALID_ARGUMENT);
}

TEST_P(FileTransferTest, UnalignedInitialRequestKeepsCallerOwnership) {
  auto& requirements = file_access() == IREE_HAL_MEMORY_ACCESS_READ
                           ? provider_.requirements.read
                           : provider_.requirements.write;
  requirements = {16, 8, 4};
  ASSERT_NO_FATAL_FAILURE(Initialize());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  EXPECT_TRUE(provider_.requests.empty());
  EXPECT_EQ(provider_.pending, nullptr);
  EXPECT_EQ(completion_count_, 0u);
}

TEST_P(FileTransferTest, PlatformFailureAfterPartialProgressIsTerminal) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  IREE_ASSERT_OK(Submit());
  provider_.Complete(7);
  provider_.Complete(0, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(transfer_.async.bytes_transferred, 7u);
}

TEST_P(FileTransferTest, ResubmissionFailureIsTerminal) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  IREE_ASSERT_OK(Submit());
  provider_.submit_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  provider_.Complete(7);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(provider_.pending, nullptr);
}

TEST_P(FileTransferTest, RejectedSubmissionRetainsCallerOwnership) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  provider_.submit_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Submit());
  EXPECT_EQ(completion_count_, 0u);
  EXPECT_EQ(provider_.pending, nullptr);
}

TEST_P(FileTransferTest, CacheFailurePropagatesAtItsExecutionBoundary) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  storage_.cache_status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  if (file_access() == IREE_HAL_MEMORY_ACCESS_WRITE) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
    EXPECT_TRUE(provider_.requests.empty());
  } else {
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
    IREE_ASSERT_OK(Submit());
    provider_.Complete(kLength);
    EXPECT_EQ(completion_code_, IREE_STATUS_OK);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          iree_hal_amd_fabric_file_transfer_finish(&transfer_));
  }
}

TEST_P(FileTransferTest, NativeIoUsesOnlyItsPreparedBacking) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
  const auto original_buffer = storage_.contents;
  IREE_ASSERT_OK(Submit());
  void* native_pointer =
      file_access() == IREE_HAL_MEMORY_ACCESS_READ
          ? iree_async_span_ptr(transfer_.async.read.buffer)
          : iree_async_span_ptr(transfer_.async.write.buffer);
  if (is_direct()) {
    EXPECT_EQ(native_pointer, transfer_.async.staging.data);
    EXPECT_NE(native_pointer, storage_.contents.data() + kNativeOffset);
  } else {
    EXPECT_EQ(native_pointer, storage_.contents.data() + kNativeOffset);
    EXPECT_EQ(transfer_.async.staging.data, nullptr);
  }
  provider_.Complete(kLength);
  EXPECT_EQ(completion_count_, 1u);
  EXPECT_EQ(completion_code_, IREE_STATUS_OK);
  if (is_direct()) {
    // Only the host service copies successful reads into the device mapping.
    EXPECT_EQ(storage_.contents, original_buffer);
  }
  IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_finish(&transfer_));
  EXPECT_EQ(storage_.contents, expected_buffer_);
  EXPECT_EQ(provider_.contents, expected_file_);
  EXPECT_EQ(transfer_.async.staging.data, nullptr);
}

TEST_P(FileTransferTest, StagingAllocationFailureKeepsCallerOwnership) {
  const iree_allocator_t failing_allocator = {
      .self = nullptr,
      .ctl =
          +[](void*, iree_allocator_command_t, const void*, void**) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "host allocation failed");
          },
  };
  ASSERT_NO_FATAL_FAILURE(
      Initialize(kFileOffset, kBufferOffset, failing_allocator));
  if (is_direct()) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
    EXPECT_EQ(transfer_.async.staging.data, nullptr);
    EXPECT_TRUE(provider_.requests.empty());
    EXPECT_EQ(completion_count_, 0u);
  } else {
    // Buffered transfers have no staging allocation dependency.
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_prepare(&transfer_));
    IREE_ASSERT_OK(Submit());
    provider_.Complete(kLength);
    IREE_ASSERT_OK(iree_hal_amd_fabric_file_transfer_finish(&transfer_));
    EXPECT_EQ(completion_code_, IREE_STATUS_OK);
    EXPECT_EQ(storage_.contents, expected_buffer_);
    EXPECT_EQ(provider_.contents, expected_file_);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReadAndWrite, FileTransferTest,
    ::testing::Combine(::testing::Values(IREE_HAL_MEMORY_ACCESS_READ,
                                         IREE_HAL_MEMORY_ACCESS_WRITE),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<FileTransferTest::ParamType>& info) {
      std::string name = std::get<0>(info.param) == IREE_HAL_MEMORY_ACCESS_READ
                             ? "Read"
                             : "Write";
      return name + (std::get<1>(info.param) ? "Direct" : "Buffered");
    });

}  // namespace
