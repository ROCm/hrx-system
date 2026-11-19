// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/io/file_contents.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::cts {
namespace {

class VulkanFiniteFileTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &file_pool_));
    IREE_ASSERT_OK(
        iree_async_proactor_pool_get(file_pool_, 0, &file_proactor_));
  }

  void TearDown() override {
    iree_async_proactor_pool_release(file_pool_);
    CtsTestBase<>::TearDown();
  }

  iree_status_t ImportFile(iree_string_view_t path,
                           const std::vector<uint8_t>& bytes,
                           iree_hal_file_t** out_file) {
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        path, iree_make_const_byte_span(bytes.data(), bytes.size()),
        iree_allocator_system()));
    iree_io_file_handle_t* handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
        path, iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_fd_file_from_handle(
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
        file_proactor_, iree_allocator_system(), out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  iree_status_t ImportMemory(std::vector<uint8_t>& bytes,
                             iree_hal_file_t** out_file) {
    iree_io_file_handle_t* handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(bytes.data(), bytes.size()), {},
        iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_memory_file_wrap(
        nullptr, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        handle, 0, iree_allocator_system(), out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  enum class Form { kHostRows, kHostStrided, kIndirectRows, kIndirectStrided };

  void RoundTrip(Form form, size_t count) {
    constexpr size_t kRows = 65, kCapacity = 68, kOrigin = 32;
    constexpr size_t kWidth = 90, kFileStride = 192, kBufferStride = 128;
    constexpr size_t kLength = 2 * kOrigin + kRows * kBufferStride;
    const bool indirect =
        form == Form::kIndirectRows || form == Form::kIndirectStrided;
    const bool strided =
        form == Form::kHostStrided || form == Form::kIndirectStrided;
    const bool invalid = count > kRows;
    SCOPED_TRACE(::testing::Message()
                 << "form=" << static_cast<int>(form) << ", count=" << count);
    std::vector<uint8_t> input =
        MakeDeterministicBytes(2 * kOrigin + kRows * kFileStride);
    std::vector<uint8_t> memory_output(input.size(), 0xCD);
    std::vector<uint8_t> expected_file(memory_output);
    std::vector<uint8_t> expected_memory(memory_output);
    std::vector<uint8_t> expected(kLength, 0xA5);
    std::vector<uint8_t> actual(kLength);
    iree::testing::TempFilePath input_path("vulkan_finite_input");
    iree::testing::TempFilePath output_path("vulkan_finite_output");
    Ref<iree_hal_file_t> input_file, output_file, memory_input, memory_file;
    IREE_ASSERT_OK(ImportFile(input_path.path_view(), input, input_file.out()));
    IREE_ASSERT_OK(
        ImportFile(output_path.path_view(), memory_output, output_file.out()));
    IREE_ASSERT_OK(ImportMemory(input, memory_input.out()));
    IREE_ASSERT_OK(ImportMemory(memory_output, memory_file.out()));
    std::array<uint64_t, 8 + kCapacity * 8> words;
    words.fill(UINT64_MAX);
    words[0] = strided ? 1 : count;
    std::array<iree_hal_file_transfer_t, kRows> rows = {};
    for (size_t i = 0; i < std::min(count, kRows); ++i) {
      // The first page can pack a full native batch; the next page mixes
      // native files and host memory files without changing the payload owner.
      const uint32_t file_index =
          !strided && !invalid && i >= 32 ? static_cast<uint32_t>(i & 1) : 0;
      rows[i] = {
          file_index, 0, i * kFileStride, i * kBufferStride, kWidth, 1, 1,
          {},         {}};
      words[8 + i * 8] = i * kFileStride | (uint64_t{file_index} << 48);
      words[9 + i * 8] = i * kBufferStride;
      words[10 + i * 8] = kWidth;
      memcpy(expected.data() + kOrigin + i * kBufferStride,
             input.data() + kOrigin + i * kFileStride, kWidth);
      auto& output = file_index ? expected_memory : expected_file;
      memcpy(output.data() + kOrigin + i * kFileStride,
             input.data() + kOrigin + i * kFileStride, kWidth);
    }
    if (strided) {
      const uint64_t rows_per_slice = count == kRows ? 5 : count;
      const uint64_t slices = count == kRows ? 13 : 1;
      rows[0] = {0,
                 0,
                 0,
                 0,
                 kWidth,
                 rows_per_slice,
                 slices,
                 {kFileStride, rows_per_slice * kFileStride},
                 {kBufferStride, rows_per_slice * kBufferStride}};
      words[8] = words[9] = 0;
      words[10] = kWidth | (UINT64_C(1) << 48);
      words[11] = kFileStride | (rows_per_slice << 48);
      words[12] = kBufferStride;
      words[13] = rows_per_slice * kFileStride | (slices << 48);
      words[14] = rows_per_slice * kBufferStride;
    } else if (invalid) {
      // The malformed third page must not publish failure until preceding
      // accepted copies and native requests have retired.
      words[8 + kRows * 8] = UINT64_C(2) << 48;
      words[9 + kRows * 8] = 0;
      words[10 + kRows * 8] = kWidth;
    }
    Ref<iree_hal_buffer_t> payload, metadata, independent;
    IREE_ASSERT_OK(
        CreateFilledDeviceBuffer(kLength, uint8_t{0xA5}, payload.out()));
    IREE_ASSERT_OK(
        CreateUninitializedDeviceBuffer(sizeof(words), metadata.out()));
    IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(4, independent.out()));
    SemaphoreList gate(device_, {0}, {1}), ready(device_, {0}, {1});
    SemaphoreList loaded(device_, {0}, {1}), written(device_, {0}, {1});
    SemaphoreList finished(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_upload_linear(transfer_queue_, gate, ready,
                                                words.data(), metadata, 0,
                                                sizeof(words), nullptr));
    std::array<iree_hal_file_binding_t, 2> sources = {{
        {input_file, kOrigin, kRows * kFileStride},
        {memory_input, kOrigin, kRows * kFileStride},
    }};
    std::array<iree_hal_file_binding_t, 2> targets = {{
        {output_file, kOrigin, kRows * kFileStride},
        {memory_file, kOrigin, kRows * kFileStride},
    }};
    iree_hal_buffer_binding_t buffer = {payload, kOrigin,
                                        kRows * kBufferStride};
    iree_hal_file_transfer_list_t list = {};
    if (indirect) {
      list.type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT;
      list.indirect = {{metadata, 64, kCapacity * 64}, metadata, 0};
    } else {
      list.host = {strided ? 1 : count, rows.data()};
    }
    iree_hal_semaphore_list_t tail = ready;
    iree_status_t status = iree_hal_queue_read(
        transfer_queue_, ready, loaded, {sources.size(), sources.data()},
        {1, &buffer}, &list, nullptr, IREE_HAL_READ_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      tail = loaded;
      if (!invalid) {
        status =
            iree_hal_queue_write(transfer_queue_, loaded, written, {1, &buffer},
                                 {targets.size(), targets.data()}, &list,
                                 nullptr, IREE_HAL_WRITE_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          tail = written;
        }
      }
    }
    if (iree_status_is_ok(status) && !invalid) {
      status = iree_hal_queue_download_linear(
          transfer_queue_, written, finished, payload, 0, actual.data(),
          actual.size(), nullptr);
      if (iree_status_is_ok(status)) {
        tail = finished;
      }
    }
    // A pending file group must not hold up unrelated work on this exact queue.
    const uint32_t pattern = 0x35791ACE;
    uint32_t value = 0;
    SemaphoreList filled(device_, {0}, {1}), downloaded(device_, {0}, {1});
    iree_hal_semaphore_list_t independent_tail =
        iree_hal_semaphore_list_empty();
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_fill_linear(
          transfer_queue_, {}, filled, independent, 0, sizeof(value), &pattern,
          sizeof(pattern), nullptr, 0);
      if (iree_status_is_ok(status)) {
        independent_tail = filled;
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_download_linear(transfer_queue_, filled,
                                              downloaded, independent, 0,
                                              &value, sizeof(value), nullptr);
      if (iree_status_is_ok(status)) {
        independent_tail = downloaded;
      }
    }
    status = iree_status_join(
        status, iree_hal_semaphore_list_wait(independent_tail,
                                             iree_infinite_timeout(), 0));
    if (iree_status_is_ok(status)) {
      EXPECT_EQ(value, pattern);
    }
    sources.fill({});
    targets.fill({});
    rows.fill({});
    list = {};
    buffer = {};
    input_file.reset();
    output_file.reset();
    memory_input.reset();
    memory_file.reset();
    payload.reset();
    metadata.reset();
    IREE_EXPECT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
    status = iree_status_join(
        status, iree_hal_semaphore_list_wait(tail, iree_infinite_timeout(), 0));
    if (invalid) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
    } else {
      IREE_ASSERT_OK(status);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(memory_output, expected_memory);
      iree_io_file_contents_t* output = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          output_path.path_view(), iree_allocator_system(), &output));
      EXPECT_EQ(output->buffer.data_length, expected_file.size());
      if (output->buffer.data_length == expected_file.size()) {
        EXPECT_EQ(memcmp(output->buffer.data, expected_file.data(),
                         expected_file.size()),
                  0);
      }
      iree_io_file_contents_free(output);
    }
  }

  // Independent native I/O service, so completions race Vulkan retirement.
  iree_async_proactor_pool_t* file_pool_ = nullptr;
  // Borrowed from file_pool_ through the final accepted file completion.
  iree_async_proactor_t* file_proactor_ = nullptr;
};

TEST_P(VulkanFiniteFileTest, HostRowsCaptureBindingsAndMetadata) {
  for (size_t count : {0, 1, 32, 64, 65}) {
    RoundTrip(Form::kHostRows, count);
  }
}

TEST_P(VulkanFiniteFileTest, HostStridedRowsAndSlices) {
  for (size_t count : {0, 1, 32, 64, 65}) {
    RoundTrip(Form::kHostStrided, count);
  }
}

TEST_P(VulkanFiniteFileTest, IndirectRowsAcquireOnlyThePublishedPrefix) {
  for (size_t count : {0, 1, 32, 64, 65, 66}) {
    RoundTrip(Form::kIndirectRows, count);
  }
}

TEST_P(VulkanFiniteFileTest, IndirectStridedRowsAndSlices) {
  for (size_t count : {0, 1, 32, 64, 65}) {
    RoundTrip(Form::kIndirectStrided, count);
  }
}

CTS_REGISTER_TEST_SUITE(VulkanFiniteFileTest);

}  // namespace
}  // namespace iree::hal::cts
