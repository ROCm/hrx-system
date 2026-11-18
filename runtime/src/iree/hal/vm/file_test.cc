// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vm/file.h"

#include <array>
#include <utility>

#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_handle.h"
#include "iree/vm/sync.h"

namespace iree::hal::cts {

class FileModuleTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(CtsTestBase::SetUp());
    if (IsSkipped()) {
      return;
    }
    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_ASSERT_OK(iree_hal_vm_register_types(environment, &table));
    IREE_ASSERT_OK(iree_hal_vm_types_resolve(table, &types_));
    iree_vm_environment_free(environment);
    iree_vm_module_t* module = nullptr;
    IREE_ASSERT_OK(iree_hal_vm_file_module_create(
        &types_, transfer_queue_, iree_allocator_system(), &module));
    iree_status_t status = iree_vm_program_create(
        {module, {}}, iree_allocator_system(), &program_);
    iree_vm_module_release(module);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_, {},
                                          iree_allocator_system(), &process_));
  }

  void DropModule() {
    iree_vm_process_release(std::exchange(process_, nullptr));
    iree_vm_program_release(std::exchange(program_, nullptr));
  }

  void TearDown() override {
    DropModule();
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    CtsTestBase::TearDown();
  }

  Ref<iree_hal_file_t> File(iree_byte_span_t contents) {
    iree_io_file_handle_t* handle = nullptr;
    IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, contents,
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    Ref<iree_hal_file_t> file;
    IREE_CHECK_OK(iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, file.out()));
    iree_io_file_handle_release(handle);
    return file;
  }

  Ref<iree_hal_buffer_t> Buffer(const void* data, size_t length) {
    Ref<iree_hal_buffer_t> buffer;
    IREE_CHECK_OK(CreateDeviceBufferWithData(data, length, buffer.out()));
    return buffer;
  }

  Ref<iree_hal_fence_t> Fence(iree_hal_semaphore_t* semaphore) {
    Ref<iree_hal_fence_t> fence;
    IREE_CHECK_OK(iree_hal_fence_create_at(
        semaphore, 1, iree_allocator_system(), fence.out()));
    return fence;
  }

  iree_status_t Transfer(iree_string_view_t name,
                         const std::vector<iree_vm_ref_t>& files,
                         const std::vector<uint64_t>& file_windows,
                         const std::vector<iree_vm_ref_t>& buffers,
                         const std::vector<uint64_t>& buffer_windows,
                         iree_hal_fence_t* wait, iree_hal_buffer_t* records,
                         iree_hal_buffer_t* count,
                         iree_hal_fence_t** out_completion) {
    *out_completion = nullptr;
    const iree_vm_typed_span_t spans[] = {
        {files.data(), files.size(), IREE_VM_SCALAR_TYPE_NONE,
         IREE_VM_SPAN_ACCESS_READ, types_.file},
        {file_windows.data(), file_windows.size(), IREE_VM_SCALAR_TYPE_I64,
         IREE_VM_SPAN_ACCESS_READ, nullptr},
        {buffers.data(), buffers.size(), IREE_VM_SCALAR_TYPE_NONE,
         IREE_VM_SPAN_ACCESS_READ, types_.buffer},
        {buffer_windows.data(), buffer_windows.size(), IREE_VM_SCALAR_TYPE_I64,
         IREE_VM_SPAN_ACCESS_READ, nullptr},
    };
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_span_borrowed(&spans[0]),
        iree_vm_variant_from_span_borrowed(&spans[1]),
        iree_vm_variant_from_span_borrowed(&spans[2]),
        iree_vm_variant_from_span_borrowed(&spans[3]),
        iree_hal_fence_variant_from_ptr_borrowed(&types_, wait),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, records),
        iree_vm_variant_from_i64(0),
        iree_vm_variant_from_i64(iree_hal_buffer_byte_length(records)),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, count),
        iree_vm_variant_from_i64(0),
    };
    iree_vm_function_t function = {};
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("hal.file"), name, &function));
    iree_vm_variant_t result = {};
    iree_status_t status = iree_vm_invoke(
        invocation_, function, iree_vm_variant_span_from_array(arguments),
        {&result, 1});
    if (iree_status_is_ok(status)) {
      status = iree_hal_fence_ptr_from_variant_move(&types_, &result,
                                                    out_completion);
    }
    iree_vm_variant_reset(&result);
    return status;
  }

  iree_vm_ref_t FileRef(iree_hal_file_t* file) {
    return iree_hal_file_ref_from_ptr_borrowed(&types_, file);
  }
  iree_vm_ref_t BufferRef(iree_hal_buffer_t* buffer) {
    return iree_hal_buffer_ref_from_ptr_borrowed(&types_, buffer);
  }

  void Await(iree_hal_fence_t* fence) {
    IREE_ASSERT_OK(iree_hal_fence_wait(fence, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Canonical identities shared by arguments and the native module.
  iree_hal_vm_types_t types_ = {};
  // Linked owner of the file capability.
  iree_vm_program_t* program_ = nullptr;
  // Process carrying the callable lookup context.
  iree_vm_process_t* process_ = nullptr;
  // Backing for synchronous native invocations.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> invocation_storage_ = {};
  // Invocation borrowing invocation_storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_P(FileModuleTest, MultipleWindowsAndDirectionsPreserveUntouchedBytes) {
  std::array<std::array<uint8_t, 64>, 5> contents;
  std::array<Ref<iree_hal_file_t>, 5> files;
  std::vector<iree_vm_ref_t> file_refs;
  std::vector<uint64_t> file_windows;
  for (size_t i = 0; i < files.size(); ++i) {
    for (size_t j = 0; j < contents[i].size(); ++j) {
      contents[i][j] = static_cast<uint8_t>(i * 31 + j);
    }
    files[i] =
        File(iree_make_byte_span(contents[i].data(), contents[i].size()));
    file_refs.push_back(FileRef(files[i]));
    file_windows.insert(file_windows.end(), {8, 48});
  }
  std::array<uint8_t, 80> sentinel;
  sentinel.fill(0xCD);
  auto first = Buffer(sentinel.data(), sentinel.size());
  auto second = Buffer(sentinel.data(), sentinel.size());
  const std::vector<iree_vm_ref_t> buffer_refs = {BufferRef(first),
                                                  BufferRef(second)};
  const std::vector<uint64_t> buffer_windows = {8, 64, 16, 48};
  std::array<iree_hal_file_transfer_record_t, 7> records;
  memset(records.data(), 0xFF, sizeof(records));
  for (size_t i = 0; i < 5; ++i) {
    records[i].words[0] = (uint64_t{i} << 48) | 32;
    records[i].words[1] = i * 5;
    records[i].words[2] = 5;
  }
  records[5].words[0] = 4ull << 48;
  records[5].words[1] = 1ull << 48;
  records[5].words[2] = (1ull << 48) | 3;   // Three bytes per strided row.
  records[5].words[3] = (2ull << 48) | 6;   // Two rows, file stride six.
  records[5].words[4] = 9;                  // Buffer stride nine.
  records[5].words[5] = (2ull << 48) | 16;  // Two slices, file stride sixteen.
  records[5].words[6] = 24;                 // Buffer slice stride twenty-four.
  // The seventh record is outside the live prefix and must not be decoded.
  auto descriptors = Buffer(records.data(), sizeof(records));
  uint64_t live_count = 6;
  auto count = Buffer(&live_count, sizeof(live_count));
  Ref<iree_hal_fence_t> completed;
  IREE_ASSERT_OK(Transfer(IREE_SV("read_indirect"), file_refs, file_windows,
                          buffer_refs, buffer_windows, nullptr, descriptors,
                          count, completed.out()));
  Await(completed);
  auto expected_first = sentinel;
  auto expected_second = sentinel;
  for (size_t i = 0; i < 5; ++i) {
    std::memcpy(expected_first.data() + 8 + i * 5, contents[i].data() + 40, 5);
  }
  for (size_t slice = 0; slice < 2; ++slice) {
    for (size_t row = 0; row < 2; ++row) {
      std::memcpy(expected_second.data() + 16 + slice * 24 + row * 9,
                  contents[4].data() + 8 + slice * 16 + row * 6, 3);
    }
  }
  EXPECT_EQ(ReadBufferBytes(first, 0, sentinel.size()),
            std::vector<uint8_t>(expected_first.begin(), expected_first.end()));
  EXPECT_EQ(
      ReadBufferBytes(second, 0, sentinel.size()),
      std::vector<uint8_t>(expected_second.begin(), expected_second.end()));

  // The same records scatter the payload back into independently bounded files.
  for (auto& file_contents : contents) {
    file_contents.fill(0xAA);
  }
  completed.reset();
  IREE_ASSERT_OK(Transfer(IREE_SV("write_indirect"), file_refs, file_windows,
                          buffer_refs, buffer_windows, nullptr, descriptors,
                          count, completed.out()));
  Await(completed);
  for (size_t i = 0; i < contents.size(); ++i) {
    for (size_t j = 0; j < contents[i].size(); ++j) {
      bool copied = j >= 40 && j < 45;
      if (i == 4) {
        for (size_t slice = 0; slice < 2; ++slice) {
          for (size_t row = 0; row < 2; ++row) {
            size_t begin = 8 + slice * 16 + row * 6;
            copied |= j >= begin && j < begin + 3;
          }
        }
      }
      EXPECT_EQ(contents[i][j],
                copied ? static_cast<uint8_t>(i * 31 + j) : 0xAA);
    }
  }
}

TEST_P(FileModuleTest, CapturesTablesAndResourcesBeforeProducerCompletes) {
  std::array<uint8_t, 16> contents = {1, 2, 3, 4, 5, 6};
  auto file = File(iree_make_byte_span(contents.data(), contents.size()));
  std::array<uint8_t, 16> initial = {};
  auto target = Buffer(initial.data(), initial.size());
  iree_hal_file_transfer_record_t record = {};
  record.words[2] = contents.size();
  auto descriptors = Buffer(&record, sizeof(record));
  uint64_t one = 1;
  auto count = Buffer(&one, sizeof(one));
  Ref<iree_hal_semaphore_t> gate(CreateSemaphore());
  auto wait = Fence(gate);
  Ref<iree_hal_fence_t> completed;
  IREE_ASSERT_OK(Transfer(IREE_SV("read_indirect"), {FileRef(file)}, {0, 16},
                          {BufferRef(target)}, {0, 16}, wait, descriptors,
                          count, completed.out()));
  // Every borrowed array above has already expired. The module and caller's
  // file/metadata owners can disappear before the producer releases the group.
  file.reset();
  descriptors.reset();
  count.reset();
  DropModule();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, iree_hal_fence_query(completed));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  Await(completed);
  EXPECT_EQ(ReadBufferBytes(target, 0, contents.size()),
            std::vector<uint8_t>(contents.begin(), contents.end()));
}

TEST_P(FileModuleTest, EmptyGroupStillWaitsAndPropagatesFailure) {
  iree_hal_file_transfer_record_t record;
  memset(&record, 0xFF, sizeof(record));
  auto descriptors = Buffer(&record, sizeof(record));
  uint64_t zero = 0;
  auto count = Buffer(&zero, sizeof(zero));
  Ref<iree_hal_semaphore_t> gate(CreateSemaphore());
  auto wait = Fence(gate);
  Ref<iree_hal_fence_t> completed;
  IREE_ASSERT_OK(Transfer(IREE_SV("read_indirect"), {}, {}, {}, {}, wait,
                          descriptors, count, completed.out()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, iree_hal_fence_query(completed));
  iree_hal_semaphore_fail(gate, iree_status_from_code(IREE_STATUS_CANCELLED));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_fence_wait(completed, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(FileModuleTest, RejectsMalformedTablesBeforeSubmission) {
  std::array<uint8_t, 16> contents = {};
  auto file = File(iree_make_byte_span(contents.data(), contents.size()));
  auto target = Buffer(contents.data(), contents.size());
  iree_hal_file_transfer_record_t record = {};
  auto descriptors = Buffer(&record, sizeof(record));
  uint64_t zero = 0;
  auto count = Buffer(&zero, sizeof(zero));
  Ref<iree_hal_fence_t> completed;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Transfer(IREE_SV("read_indirect"), {FileRef(file)}, {0},
                                 {BufferRef(target)}, {0, 16}, nullptr,
                                 descriptors, count, completed.out()));
  EXPECT_FALSE(completed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Transfer(IREE_SV("write_indirect"), {FileRef(file)},
                                 {0, 16}, {BufferRef(target)}, {0, 16, 24},
                                 nullptr, descriptors, count, completed.out()));
  EXPECT_FALSE(completed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Transfer(IREE_SV("read_indirect"), {BufferRef(target)},
                                 {0, 16}, {BufferRef(target)}, {0, 16}, nullptr,
                                 descriptors, count, completed.out()));
  EXPECT_FALSE(completed);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(FileModuleTest, {"file_io"}, {});

}  // namespace iree::hal::cts
