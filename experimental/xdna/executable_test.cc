// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/executable.h"

#include <algorithm>
#include <array>
#include <vector>

#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::StatusCode;
using iree::hal::amd::xdna::testing::ImageFixture;
using iree::hal::amd::xdna::testing::MakeImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

// Models an immutable source whose backing becomes unavailable during loading.
struct FailingSource {
  // Sequence interface borrowing bytes for this scope.
  iree_byte_sequence_t base;
  // Unchanged source bytes.
  std::vector<uint8_t> bytes;
  // Successful reads remaining before the backing fails.
  mutable size_t remaining_reads = SIZE_MAX;
};

void DestroyFailingSource(iree_byte_sequence_t* base) {}

iree_status_t EnumerateFailingSource(
    const iree_byte_sequence_t* base,
    iree_byte_sequence_segment_callback_t callback) {
  const auto& source = *reinterpret_cast<const FailingSource*>(base);
  if (source.remaining_reads == 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "source backing unavailable");
  }
  --source.remaining_reads;
  return callback.fn(
      callback.user_data,
      iree_make_const_byte_span(source.bytes.data(), source.bytes.size()));
}

const iree_byte_sequence_vtable_t kFailingSourceVtable = {
    DestroyFailingSource, EnumerateFailingSource, nullptr};

class XdnaExecutableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto source = MakeOwnedByteSequence(ImageFixture().Build());
    const auto target = MakeImageTarget();
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
        source.get(), &target, iree_allocator_system(), &image_));
    for (size_t i = 0; i < storage_.size(); ++i) {
      const auto allocation = iree_hal_amd_xdna_image_tables_allocation(
          iree_hal_amd_xdna_image_tables(image_), i);
      bytes_[i].assign(allocation.byte_length, 0xCC);
      storage_[i].mapping =
          iree_make_byte_span(bytes_[i].data(), bytes_[i].size());
      storage_[i].memory = reinterpret_cast<amdf_memory_t*>(uintptr_t{16} + i);
      storage_[i].access_ordinal = 3;
      storage_[i].memory_byte_offset = 65536;
      storage_[i].device_address = UINT64_C(0x123456780000) + i * 32768;
    }
    ASSERT_NO_FATAL_FAILURE(
        WrapBinding(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                    IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                    IREE_HAL_BUFFER_USAGE_STORAGE));
  }

  void TearDown() override {
    iree_hal_buffer_release(binding_.buffer_ref.buffer);
    iree_hal_amd_xdna_image_destroy(image_);
  }

  void WrapBinding(iree_hal_memory_type_t type, iree_hal_memory_access_t access,
                   iree_hal_buffer_usage_t usage) {
    iree_hal_buffer_release(binding_.buffer_ref.buffer);
    binding_.buffer_ref.buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), type, access, usage,
        buffer_.size(), iree_make_byte_span(buffer_.data(), buffer_.size()),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &binding_.buffer_ref.buffer));
    binding_.buffer_ref.offset = 0;
    binding_.buffer_ref.length = 16;
    binding_.memory = reinterpret_cast<amdf_memory_t*>(uintptr_t{32});
    binding_.memory_byte_offset = 128;
    binding_.device_address = UINT64_C(0xABCD12340000);
  }

  void ReplaceImage(const ImageFixture& fixture) {
    auto source = MakeOwnedByteSequence(fixture.Build());
    const auto target = MakeImageTarget();
    iree_hal_amd_xdna_image_t* image = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
        source.get(), &target, iree_allocator_system(), &image));
    iree_hal_amd_xdna_image_destroy(image_);
    image_ = image;
  }

  iree_status_t Load() {
    return iree_hal_amd_xdna_executable_load(image_, 0, storage_.size(),
                                             storage_.data());
  }
  iree_status_t Bind() {
    return iree_hal_amd_xdna_executable_bind(image_, 0, storage_.size(),
                                             storage_.data(), 1, &binding_);
  }

  // Admitted image with no native resource ownership.
  iree_hal_amd_xdna_image_t* image_ = nullptr;
  // Caller-owned command and catalog bytes.
  std::array<std::vector<uint8_t>, 2> bytes_;
  // Resolved native ranges borrowing bytes_.
  std::array<iree_hal_amd_xdna_executable_storage_t, 2> storage_ = {};
  // Logical binding storage with room to exercise nonzero offsets.
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) std::array<uint8_t, 128> buffer_ = {};
  // Owning HAL buffer reference and borrowed native address.
  iree_hal_amd_xdna_executable_binding_t binding_ = {};
};

TEST_F(XdnaExecutableTest, LoadsSharedPayloadsAndOnlyExplicitZeroTails) {
  IREE_ASSERT_OK(Load());
  const ImageFixture fixture;
  EXPECT_TRUE(std::equal(bytes_[0].begin() + 32768, bytes_[0].end(),
                         fixture.payloads[1].begin()));
  EXPECT_TRUE(std::all_of(bytes_[0].begin() + 16, bytes_[0].begin() + 32768,
                          [](uint8_t byte) { return byte == 0xCC; }));
  EXPECT_TRUE(std::equal(bytes_[1].begin(), bytes_[1].begin() + 8,
                         fixture.payloads[2].begin()));
  EXPECT_TRUE(std::all_of(bytes_[1].begin() + 8, bytes_[1].end(),
                          [](uint8_t byte) { return byte == 0; }));
  const uint64_t catalog_address = storage_[1].device_address + 4;
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data()),
            (uint32_t)catalog_address | 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 4),
            UINT32_C(0xA5A50000) | (uint32_t)(catalog_address >> 32));
}

TEST_F(XdnaExecutableTest, RebindsWithoutReloadingOrChangingOtherStorage) {
  IREE_ASSERT_OK(Load());
  for (uint64_t address :
       {UINT64_C(0xABCD12340000), UINT64_C(0x123456789000)}) {
    const auto before = bytes_;
    binding_.device_address = address;
    IREE_ASSERT_OK(Bind());
    EXPECT_TRUE(std::equal(bytes_[0].begin(), bytes_[0].begin() + 8,
                           before[0].begin()));
    EXPECT_TRUE(std::equal(bytes_[0].begin() + 16, bytes_[0].end(),
                           before[0].begin() + 16));
    EXPECT_EQ(bytes_[1], before[1]);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 8),
              (uint32_t)(address + 4) | 1u);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 12),
              UINT32_C(0xA5A50000) | (uint32_t)((address + 4) >> 32));
  }
}

TEST_F(XdnaExecutableTest, PreservesAndIgnoresUnusedBindingSlots) {
  ImageFixture fixture;
  fixture.bindings.insert(fixture.bindings.begin(),
                          iree_xdna_elf_binding_record_t{});
  fixture.entries[0].binding_count = 2;
  fixture.relocations[1].source_ordinal = 1;
  ASSERT_NO_FATAL_FAILURE(ReplaceImage(fixture));

  IREE_ASSERT_OK(Load());
  std::array<iree_hal_amd_xdna_executable_binding_t, 2> bindings = {};
  bindings[1] = binding_;
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_bind(
      image_, 0, storage_.size(), storage_.data(), bindings.size(),
      bindings.data()));
  const uint64_t address = binding_.device_address + 4;
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 8),
            (uint32_t)address | 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 12),
            UINT32_C(0xA5A50000) | (uint32_t)(address >> 32));
}

TEST_F(XdnaExecutableTest, ResolvesIndependentInvocationWithContinuation) {
  IREE_ASSERT_OK(Load());
  IREE_ASSERT_OK(Bind());
  const auto before = bytes_;
  // The image names a continuation at offset 32768. Independent invocations
  // must use the establishing range even though that continuation is present.
  amdf_xdna_kernel_command_t command = {};
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_invocation(
      image_, 0, storage_.size(), storage_.data(), &command));
  EXPECT_EQ(command.memory, storage_[0].memory);
  EXPECT_EQ(command.access_ordinal, 3u);
  EXPECT_EQ(command.byte_offset, storage_[0].memory_byte_offset);
  EXPECT_EQ(command.byte_length, 16u);
  EXPECT_EQ(bytes_, before);
}

TEST_F(XdnaExecutableTest,
       ValidatesIndependentInvocationBeforePublishingRange) {
  amdf_xdna_kernel_command_t command = {};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_amd_xdna_executable_query_invocation(image_, 1, storage_.size(),
                                                    storage_.data(), &command));
  EXPECT_EQ(command.memory, nullptr);
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_amd_xdna_executable_query_invocation(
          image_, 0, storage_.size() - 1, storage_.data(), &command));
  EXPECT_EQ(command.memory, nullptr);
  --storage_[0].mapping.data_length;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_amd_xdna_executable_query_invocation(image_, 0, storage_.size(),
                                                    storage_.data(), &command));
  EXPECT_EQ(command.memory, nullptr);
}

TEST_F(XdnaExecutableTest, ChecksAllBackingBeforeLoading) {
  const auto before = bytes_;
  --storage_[1].mapping.data_length;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument, Load());
  EXPECT_EQ(bytes_, before);
  ++storage_[1].mapping.data_length;
  ++storage_[0].device_address;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument, Load());
  EXPECT_EQ(bytes_, before);
}

TEST_F(XdnaExecutableTest, PropagatesSourceFailureAfterPartialLoad) {
  FailingSource source = {};
  source.bytes = ImageFixture().Build();
  iree_byte_sequence_initialize(&kFailingSourceVtable, source.bytes.size(),
                                &source.base);
  const auto target = MakeImageTarget();
  iree_hal_amd_xdna_image_t* image = nullptr;
  iree_status_t status = iree_hal_amd_xdna_image_create(
      &source.base, &target, iree_allocator_system(), &image);
  iree_byte_sequence_release(&source.base);
  std::unique_ptr<iree_hal_amd_xdna_image_t,
                  decltype(&iree_hal_amd_xdna_image_destroy)>
      image_owner(image, iree_hal_amd_xdna_image_destroy);
  IREE_ASSERT_OK(status);
  source.remaining_reads = 1;
  IREE_EXPECT_STATUS_IS(StatusCode::kUnavailable,
                        iree_hal_amd_xdna_executable_load(
                            image, 0, storage_.size(), storage_.data()));
  EXPECT_TRUE(std::all_of(bytes_[0].begin(), bytes_[0].begin() + 16,
                          [](uint8_t byte) { return byte == 0xA5; }));
  EXPECT_TRUE(std::all_of(bytes_[0].begin() + 16, bytes_[0].end(),
                          [](uint8_t byte) { return byte == 0xCC; }));
  EXPECT_TRUE(std::all_of(bytes_[1].begin(), bytes_[1].end(),
                          [](uint8_t byte) { return byte == 0xCC; }));
}

TEST_F(XdnaExecutableTest, ChecksBindingCountAndLogicalRanges) {
  IREE_ASSERT_OK(Load());
  const auto before = bytes_;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_amd_xdna_executable_bind(image_, 0, storage_.size(),
                                        storage_.data(), 0, nullptr));
  binding_.buffer_ref.offset = buffer_.size() - 8;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
  binding_.buffer_ref.offset = 4;
  binding_.buffer_ref.length = IREE_HAL_WHOLE_BUFFER;
  IREE_ASSERT_OK(Bind());
  binding_.buffer_ref.length = 8;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
  binding_.buffer_ref.offset = UINT64_MAX;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
  EXPECT_EQ(bytes_[1], before[1]);
}

TEST_F(XdnaExecutableTest, RejectsInvalidAddressesBeforePatching) {
  IREE_ASSERT_OK(Load());
  const auto before = bytes_;
  for (uint64_t address :
       {UINT64_MAX - 3, UINT64_C(0x1000000000000), UINT64_C(3)}) {
    binding_.device_address = address;
    IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
    EXPECT_EQ(bytes_, before);
  }
  storage_[1].device_address = UINT64_C(0xFFFFFFFFFFFC);
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Load());
  EXPECT_EQ(bytes_, before);
}

TEST_F(XdnaExecutableTest, EnforcesHalAccessUsageAndVisibility) {
  IREE_ASSERT_OK(Load());
  const auto before = bytes_;
  ASSERT_NO_FATAL_FAILURE(WrapBinding(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                                      IREE_HAL_MEMORY_ACCESS_WRITE,
                                      IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, Bind());
  ASSERT_NO_FATAL_FAILURE(WrapBinding(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                                      IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, Bind());
  ASSERT_NO_FATAL_FAILURE(WrapBinding(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
                                      IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, Bind());
  EXPECT_EQ(bytes_, before);
}

}  // namespace
