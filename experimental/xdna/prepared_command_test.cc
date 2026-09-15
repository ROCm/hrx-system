// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/prepared_command.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "experimental/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/directory.h"
#include "iree/hal/drivers/amd/xdna/image/format.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::ByteSequencePtr;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

static constexpr iree_host_size_t kBindingCount = 3;
static constexpr iree_host_size_t kBufferStorageByteLength = 4096;
static constexpr std::array<iree_device_size_t, kBindingCount>
    kBindingByteLengths = {64, 64, 64};

static ByteSequencePtr LoadMulI32Image() {
  EXPECT_EQ(iree_hal_amd_xdna_test_mul_i32_size(), 1u);
  if (iree_hal_amd_xdna_test_mul_i32_size() != 1u) return {};
  const iree_file_toc_t* file = iree_hal_amd_xdna_test_mul_i32_create();
  const auto* begin = reinterpret_cast<const uint8_t*>(file->data);
  return MakeOwnedByteSequence(std::vector<uint8_t>(begin, begin + file->size));
}

static void ReleaseBufferStorage(void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  ++*static_cast<uint32_t*>(user_data);
}

class XdnaPreparedCommandTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static_assert(kBufferStorageByteLength % IREE_HAL_HEAP_BUFFER_ALIGNMENT ==
                  0);
    queue_family_spec_.name = IREE_SV("test");
    queue_family_spec_.physical_device_affinity = 1;
    queue_family_spec_.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
    iree_hal_queue_family_initialize(/*ordinal=*/7, &queue_family_spec_,
                                     &queue_family_);

    ByteSequencePtr sequence = LoadMulI32Image();
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        IREE_SV("amd.xdna.strix_halo.17f0_11"),
        /*context_column_count=*/1, &target));
    IREE_CHECK_OK(iree_hal_amd_xdna_executable_create(
        &queue_family_, sequence.get(), &target, iree_allocator_system(),
        &executable_));

    for (iree_host_size_t i = 0; i < kBindingCount; ++i) {
      WrapBuffer(i,
                 IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                     IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                     IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                 IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                 IREE_HAL_BUFFER_USAGE_STORAGE);
    }
  }

  void TearDown() override {
    if (prepared_command_ != nullptr) {
      iree_hal_amd_xdna_prepared_command_destroy(prepared_command_);
      prepared_command_ = nullptr;
    }
    for (iree_hal_buffer_t*& buffer : buffers_) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
  }

  void WrapBuffer(iree_host_size_t ordinal, iree_hal_memory_type_t memory_type,
                  iree_hal_memory_access_t allowed_access,
                  iree_hal_buffer_usage_t allowed_usage) {
    iree_hal_buffer_release(buffers_[ordinal]);
    buffers_[ordinal] = nullptr;
    IREE_CHECK_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), memory_type, allowed_access,
        allowed_usage, kBufferStorageByteLength,
        iree_make_byte_span(buffer_storage_[ordinal].data(),
                            buffer_storage_[ordinal].size()),
        (iree_hal_buffer_release_callback_t){
            .fn = ReleaseBufferStorage,
            .user_data = &release_counts_[ordinal],
        },
        iree_allocator_system(), &buffers_[ordinal]));
    bindings_[ordinal] = (iree_hal_amd_xdna_prepared_command_binding_t){
        .buffer_ref = iree_hal_make_buffer_ref(buffers_[ordinal], /*offset=*/0,
                                               kBindingByteLengths[ordinal]),
        .memory = reinterpret_cast<amdf_memory_t*>(uintptr_t{0x1000} +
                                                   ordinal * uintptr_t{0x100}),
        .memory_byte_offset = 64 + ordinal * 512,
        .device_address = 0x100000 + ordinal * 0x1000,
    };
  }

  iree_status_t CreatePrepared(
      iree_host_size_t binding_count,
      const iree_hal_amd_xdna_prepared_command_binding_t* bindings) {
    iree_host_size_t byte_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_prepared_command_query_storage_size(
        executable_, iree_hal_executable_function_from_index(0), 32768,
        &byte_length));
    instructions_.assign(byte_length, 0xCC);
    storage_range_ = {};
    storage_range_.memory = reinterpret_cast<amdf_memory_t*>(uintptr_t{0x1230});
    storage_range_.access_ordinal = 3;
    storage_range_.byte_offset = 65536;
    storage_range_.byte_length = byte_length;
    return iree_hal_amd_xdna_prepared_command_create(
        executable_, iree_hal_executable_function_from_index(0), 32768,
        &storage_range_,
        iree_make_byte_span(instructions_.data(), instructions_.size()),
        binding_count, bindings, iree_allocator_system(), &prepared_command_);
  }

  iree_status_t CreatePrepared() {
    return CreatePrepared(bindings_.size(), bindings_.data());
  }

  void DestroyPrepared() {
    iree_hal_amd_xdna_prepared_command_destroy(prepared_command_);
    prepared_command_ = nullptr;
  }

  // Caller-owned writable instruction storage.
  std::vector<uint8_t> instructions_;
  // Native range identifying the mapped instruction bytes.
  amdf_xdna_kernel_command_t storage_range_ = {};
  // Executable-only family metadata; this fixture provisions no HAL queues.
  iree_hal_queue_family_spec_t queue_family_spec_ = {};
  // HAL queue family borrowed by the executable.
  iree_hal_queue_family_t queue_family_ = {};
  // Qualified executable retained by this fixture.
  iree_hal_executable_t* executable_ = nullptr;
  // Host backing wrapped by the real HAL buffers.
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT)
      std::array<std::array<uint8_t, kBufferStorageByteLength>,
                 kBindingCount> buffer_storage_ = {};
  // Number of final buffer releases observed by the backing owner.
  std::array<uint32_t, kBindingCount> release_counts_ = {};
  // HAL buffer references owned by the fixture until explicitly released.
  std::array<iree_hal_buffer_t*, kBindingCount> buffers_ = {};
  // Resolved caller binding ranges and DMA addresses.
  std::array<iree_hal_amd_xdna_prepared_command_binding_t, kBindingCount>
      bindings_ = {};
  // Cold instance retaining its executable and HAL buffer bindings.
  iree_hal_amd_xdna_prepared_command_t* prepared_command_ = nullptr;
};

TEST_F(XdnaPreparedCommandTest,
       InstantiatesFixedDmaAddressesAndPreservesMetadata) {
  for (iree_host_size_t i = 0; i < bindings_.size(); ++i) {
    bindings_[i].device_address = UINT64_C(0x812345670000) + i * 4096;
  }
  IREE_ASSERT_OK(CreatePrepared());
  iree_hal_amd_xdna_executable_entry_t entry;
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_entry(
      executable_, iree_hal_executable_function_from_index(0), &entry));
  const auto* initialization =
      iree_hal_amd_xdna_prepared_command_initialization(prepared_command_);
  const auto* execution =
      iree_hal_amd_xdna_prepared_command_execution(prepared_command_);
  EXPECT_EQ(initialization->memory, storage_range_.memory);
  EXPECT_EQ(initialization->access_ordinal, storage_range_.access_ordinal);
  EXPECT_EQ(initialization->byte_offset, storage_range_.byte_offset);
  EXPECT_EQ(execution->memory, initialization->memory);
  EXPECT_EQ(execution->access_ordinal, initialization->access_ordinal);
  EXPECT_EQ(execution->byte_offset, storage_range_.byte_offset + 32768);
  EXPECT_EQ(execution->byte_length, entry.native.control.data_length);
  EXPECT_EQ(iree_unaligned_load_le_u32(instructions_.data() + 8), 58u);
  EXPECT_EQ(iree_unaligned_load_le_u32(instructions_.data() + 12),
            initialization->byte_length);
  const size_t control_offset = entry.array.data_length - 16;
  for (iree_host_size_t i = 0; i < entry.native.relocation_count; ++i) {
    const auto& relocation = entry.native.relocations[i];
    const uint64_t address =
        bindings_[relocation.binding_ordinal].device_address;
    const uint32_t source_high = iree_unaligned_load_le_u32(
        entry.native.control.data + relocation.byte_offset + 4);
    for (size_t offset : {control_offset, size_t{32768}}) {
      const uint8_t* field =
          instructions_.data() + offset + relocation.byte_offset;
      EXPECT_EQ(iree_unaligned_load_le_u32(field), uint32_t(address));
      EXPECT_EQ(iree_unaligned_load_le_u32(field + 4),
                (source_high & 0xFFFF0000u) | uint32_t(address >> 32));
    }
  }
  const auto snapshot = instructions_;
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(iree_hal_amd_xdna_prepared_command_execution(prepared_command_),
              execution);
  }
  DestroyPrepared();
  EXPECT_EQ(instructions_, snapshot);
}

TEST_F(XdnaPreparedCommandTest, RetainsEveryBorrowedResource) {
  IREE_ASSERT_OK(CreatePrepared());

  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  for (iree_host_size_t i = 0; i < buffers_.size(); ++i) {
    iree_hal_buffer_release(buffers_[i]);
    buffers_[i] = nullptr;
    EXPECT_EQ(release_counts_[i], 0u);
  }

  DestroyPrepared();
  for (uint32_t release_count : release_counts_) {
    EXPECT_EQ(release_count, 1u);
  }
}

TEST_F(XdnaPreparedCommandTest, RejectsMalformedBindingRanges) {
  auto invalid_bindings = bindings_;
  invalid_bindings[0].buffer_ref.length = kBindingByteLengths[0] - 1;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      CreatePrepared(invalid_bindings.size(), invalid_bindings.data()));

  invalid_bindings = bindings_;
  invalid_bindings[0].buffer_ref.offset = 4;
  invalid_bindings[0].buffer_ref.length = kBindingByteLengths[0];
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      CreatePrepared(invalid_bindings.size(), invalid_bindings.data()));

  invalid_bindings = bindings_;
  ++invalid_bindings[0].device_address;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      CreatePrepared(invalid_bindings.size(), invalid_bindings.data()));

  invalid_bindings = bindings_;
  invalid_bindings[0].memory = nullptr;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      CreatePrepared(invalid_bindings.size(), invalid_bindings.data()));
}

TEST_F(XdnaPreparedCommandTest, RejectsBindingCountMismatch) {
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        CreatePrepared(bindings_.size() - 1, bindings_.data()));
}

TEST_F(XdnaPreparedCommandTest, BoundsUnrestrictedOffsetsByTheActualBuffer) {
  ByteSequencePtr original_sequence = LoadMulI32Image();
  iree_hal_amd_xdna_image_directory_t* raw_directory = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_directory_create(
      original_sequence.get(), iree_allocator_system(), &raw_directory));
  std::unique_ptr<iree_hal_amd_xdna_image_directory_t,
                  decltype(&iree_hal_amd_xdna_image_directory_destroy)>
      directory(raw_directory, iree_hal_amd_xdna_image_directory_destroy);
  uint64_t binding_table_offset = 0;
  for (iree_host_size_t i = 0;
       i <
       iree_hal_amd_xdna_image_directory_program_header_count(directory.get());
       ++i) {
    const auto* header =
        iree_hal_amd_xdna_image_directory_program_header(directory.get(), i);
    if (header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS) {
      binding_table_offset = header->file_range.offset;
    }
  }
  ASSERT_NE(binding_table_offset, 0u);
  const iree_file_toc_t* file = iree_hal_amd_xdna_test_mul_i32_create();
  const auto* begin = reinterpret_cast<const uint8_t*>(file->data);
  std::vector<uint8_t> bytes(begin, begin + file->size);
  iree_byte_span_t record_storage =
      iree_make_byte_span(bytes.data() + binding_table_offset +
                              IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE,
                          IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE);
  iree_hal_amd_xdna_elf_binding_record_t contract;
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_decode_binding_record(
      iree_make_const_byte_span(record_storage.data,
                                record_storage.data_length),
      &contract));
  contract.maximum_byte_offset = UINT64_MAX;
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_elf_encode_binding_record(&contract, record_storage));
  ByteSequencePtr sequence = MakeOwnedByteSequence(bytes);
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), 1, &target));
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_create(
      &queue_family_, sequence.get(), &target, iree_allocator_system(),
      &executable));
  iree_hal_executable_release(executable_);
  executable_ = executable;

  const iree_device_size_t offset =
      kBufferStorageByteLength - kBindingByteLengths[0];
  bindings_[0].buffer_ref.offset = offset;
  bindings_[0].memory_byte_offset += offset;
  bindings_[0].device_address += offset;
  IREE_ASSERT_OK(CreatePrepared());
  DestroyPrepared();

  // The image permits this offset, but the selected range exceeds its buffer.
  ++bindings_[0].buffer_ref.offset;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, CreatePrepared());
  EXPECT_EQ(prepared_command_, nullptr);
  for (uint8_t byte : instructions_) EXPECT_EQ(byte, 0xCC);
}

TEST_F(XdnaPreparedCommandTest, EnforcesBindingAccess) {
  WrapBuffer(/*ordinal=*/2,
             IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                 IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                 IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
             IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_USAGE_STORAGE);
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, CreatePrepared());
}

TEST_F(XdnaPreparedCommandTest, EnforcesBindingUsage) {
  WrapBuffer(/*ordinal=*/2,
             IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                 IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                 IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
             IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
             IREE_HAL_BUFFER_USAGE_STORAGE_READ);
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, CreatePrepared());
}

TEST_F(XdnaPreparedCommandTest, EnforcesDeviceVisibility) {
  WrapBuffer(
      /*ordinal=*/0,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_STORAGE);
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, CreatePrepared());
}

TEST_F(XdnaPreparedCommandTest, RejectsUnrepresentableDmaAddressWithoutWrites) {
  bindings_[0].device_address = UINT64_C(1) << 48;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, CreatePrepared());
  EXPECT_EQ(prepared_command_, nullptr);
  for (uint8_t byte : instructions_) EXPECT_EQ(byte, 0xCC);
  for (iree_host_size_t i = 0; i < buffers_.size(); ++i) {
    iree_hal_buffer_release(buffers_[i]);
    buffers_[i] = nullptr;
    EXPECT_EQ(release_counts_[i], 1u);
  }
}

TEST_F(XdnaPreparedCommandTest, RejectsShortStorageWithoutPublishing) {
  IREE_ASSERT_OK(CreatePrepared());
  DestroyPrepared();
  instructions_.assign(instructions_.size() - 1, 0xCC);
  storage_range_.byte_length = instructions_.size();
  auto* sentinel =
      reinterpret_cast<iree_hal_amd_xdna_prepared_command_t*>(uintptr_t{1});
  auto* output = sentinel;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_amd_xdna_prepared_command_create(
          executable_, iree_hal_executable_function_from_index(0), 32768,
          &storage_range_,
          iree_make_byte_span(instructions_.data(), instructions_.size()),
          bindings_.size(), bindings_.data(), iree_allocator_system(),
          &output));
  EXPECT_EQ(output, sentinel);
  for (uint8_t byte : instructions_) EXPECT_EQ(byte, 0xCC);
}

}  // namespace
