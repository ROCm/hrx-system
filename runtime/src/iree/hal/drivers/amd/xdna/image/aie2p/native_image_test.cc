// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/native_image.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/transaction.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::Aie2pImageFixtureOptions;
using iree::hal::amd::xdna::testing::Aie2pImageTargetState;
using iree::hal::amd::xdna::testing::BuildAie2pImage;
using iree::hal::amd::xdna::testing::ByteSequencePtr;
using iree::hal::amd::xdna::testing::MakeAie2pImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;
using iree::testing::status::StatusIs;

struct ImageDeleter {
  void operator()(iree_hal_amd_xdna_image_t* image) const {
    iree_hal_amd_xdna_image_destroy(image);
  }
};

using ImagePtr = std::unique_ptr<iree_hal_amd_xdna_image_t, ImageDeleter>;

struct NativeImageDeleter {
  void operator()(iree_hal_amd_xdna_aie2p_native_image_t* image) const {
    iree_hal_amd_xdna_aie2p_native_image_destroy(image);
  }
};

using NativeImagePtr =
    std::unique_ptr<iree_hal_amd_xdna_aie2p_native_image_t, NativeImageDeleter>;

static ImagePtr CreateImage(
    const std::vector<uint8_t>& bytes,
    const iree_hal_amd_xdna_aie2p_target_t* aie2p_target) {
  ByteSequencePtr sequence = MakeOwnedByteSequence(bytes);
  iree_hal_amd_xdna_image_target_t image_target;
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      aie2p_target, &image_target));
  iree_hal_amd_xdna_image_t* image = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_create(
      sequence.get(), &image_target, iree_allocator_system(), &image));
  return ImagePtr(image);
}

static NativeImagePtr CreateNativeImage(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target) {
  iree_hal_amd_xdna_aie2p_native_image_t* native_image = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_native_image_create(
      image, target, iree_allocator_system(), &native_image));
  return NativeImagePtr(native_image);
}

static std::vector<uint8_t> LoadMulI32ImageBytes() {
  EXPECT_EQ(iree_hal_amd_xdna_test_mul_i32_size(), 1u);
  if (iree_hal_amd_xdna_test_mul_i32_size() != 1u) return {};
  const iree_file_toc_t* file = iree_hal_amd_xdna_test_mul_i32_create();
  const auto* data = reinterpret_cast<const uint8_t*>(file->data);
  return std::vector<uint8_t>(data, data + file->size);
}

struct NativeOperation {
  // Native transaction opcode.
  uint8_t opcode;
  // Operation byte offset within the transaction.
  iree_host_size_t byte_offset;
  // Complete operation byte length.
  iree_host_size_t byte_length;
};

static std::vector<NativeOperation> DecodeOperations(
    iree_const_byte_span_t transaction) {
  std::vector<NativeOperation> operations;
  if (transaction.data_length <
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE) {
    ADD_FAILURE() << "native transaction header is truncated";
    return operations;
  }
  const uint32_t expected_operation_count =
      iree_unaligned_load_le_u32(transaction.data + 8);
  if (iree_unaligned_load_le_u32(transaction.data + 12) !=
      transaction.data_length) {
    ADD_FAILURE() << "native transaction byte length is inconsistent";
    return operations;
  }
  iree_host_size_t byte_offset =
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE;
  while (byte_offset < transaction.data_length) {
    const uint8_t opcode = transaction.data[byte_offset];
    iree_host_size_t byte_length = 0;
    switch (opcode) {
      case 0:
        byte_length = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_WRITE32_SIZE;
        break;
      case 1:
        if (transaction.data_length - byte_offset <
            IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE) {
          ADD_FAILURE() << "native block-write header is truncated";
          return {};
        }
        byte_length =
            iree_unaligned_load_le_u32(transaction.data + byte_offset + 12);
        break;
      case 3:
        byte_length =
            IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_MASK_WRITE32_SIZE;
        break;
      case 128:
        byte_length = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_DMA_TASK_WAIT_SIZE;
        break;
      default:
        ADD_FAILURE() << "unknown native opcode " << static_cast<int>(opcode);
        return {};
    }
    if (byte_length == 0 ||
        byte_length > transaction.data_length - byte_offset) {
      ADD_FAILURE() << "native operation exceeds its transaction";
      return {};
    }
    operations.push_back({opcode, byte_offset, byte_length});
    byte_offset += byte_length;
  }
  EXPECT_EQ(byte_offset, transaction.data_length);
  EXPECT_EQ(operations.size(), expected_operation_count);
  return operations;
}

static void ExpectTransactionHeader(iree_const_byte_span_t transaction,
                                    uint32_t operation_count,
                                    uint32_t column_count) {
  ASSERT_GE(transaction.data_length,
            IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE);
  EXPECT_EQ(transaction.data[0], 0u);
  EXPECT_EQ(transaction.data[1], 1u);
  EXPECT_EQ(transaction.data[2], 4u);
  EXPECT_EQ(transaction.data[3], 6u);
  EXPECT_EQ(transaction.data[4], column_count);
  EXPECT_EQ(transaction.data[5], 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(transaction.data + 8), operation_count);
  EXPECT_EQ(iree_unaligned_load_le_u32(transaction.data + 12),
            transaction.data_length);
}

TEST(Aie2pNativeImageTest, LowersAndOwnsSyntheticImage) {
  Aie2pImageTargetState target_state;
  const iree_hal_amd_xdna_aie2p_target_t target =
      MakeAie2pImageTarget(&target_state);
  ImagePtr image = CreateImage(BuildAie2pImage(), &target);
  NativeImagePtr native_image = CreateNativeImage(image.get(), &target);
  image.reset();

  EXPECT_EQ(
      iree_hal_amd_xdna_aie2p_native_image_array_count(native_image.get()), 1u);
  EXPECT_EQ(
      iree_hal_amd_xdna_aie2p_native_image_entry_count(native_image.get()), 1u);

  iree_const_byte_span_t array = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_native_image_query_array_configuration(
      native_image.get(), 0, &array));
  ASSERT_EQ(array.data_length, 64u);
  ExpectTransactionHeader(array, 1, 3);
  EXPECT_EQ(array.data[16], 1u);
  EXPECT_EQ(array.data[20], 1u);
  EXPECT_EQ(array.data[21], 2u);
  EXPECT_EQ(iree_unaligned_load_le_u32(array.data + 24), UINT32_C(0x02220000));
  EXPECT_EQ(iree_unaligned_load_le_u32(array.data + 28), 48u);
  EXPECT_TRUE(std::all_of(array.data + 32, array.data + 64,
                          [](uint8_t value) { return value == 0xA4; }));

  iree_hal_amd_xdna_aie2p_native_entry_t entry = {};
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_native_image_query_entry(
      native_image.get(), 0, &entry));
  EXPECT_EQ(entry.array_ordinal, 0u);
  ASSERT_EQ(entry.control.data_length, 40u);
  ExpectTransactionHeader(entry.control, 1, 3);
  EXPECT_EQ(entry.control.data[16], 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(entry.control.data + 24),
            UINT32_C(0x0001D004));
  EXPECT_EQ(iree_unaligned_load_le_u32(entry.control.data + 28), 24u);
  ASSERT_EQ(entry.relocation_count, 1u);
  EXPECT_EQ(entry.relocations[0].byte_offset, 32u);
  EXPECT_EQ(entry.relocations[0].binding_ordinal, 0u);
  EXPECT_EQ(entry.relocations[0].addend, 0);
  EXPECT_EQ(entry.relocations[0].maximum_value, UINT64_C(0xFFFFFFFFFFFF));

  array = iree_make_const_byte_span(reinterpret_cast<const uint8_t*>(1), 1);
  EXPECT_THAT(
      Status(iree_hal_amd_xdna_aie2p_native_image_query_array_configuration(
          native_image.get(), 1, &array)),
      StatusIs(StatusCode::kOutOfRange));
  EXPECT_EQ(array.data, nullptr);
  EXPECT_EQ(array.data_length, 0u);
}

TEST(Aie2pNativeImageTest, PreservesColdRelocationConstraints) {
  Aie2pImageTargetState target_state;
  const auto target = MakeAie2pImageTarget(&target_state);
  for (int64_t addend : {-64, 128}) {
    Aie2pImageFixtureOptions options;
    options.relocation_addend = addend;
    options.relocation_minimum_value = 256;
    options.relocation_maximum_value = 4096;
    options.relocation_required_alignment = 64;
    ImagePtr image = CreateImage(BuildAie2pImage(options), &target);
    NativeImagePtr native_image = CreateNativeImage(image.get(), &target);
    iree_hal_amd_xdna_aie2p_native_entry_t entry = {};
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_native_image_query_entry(
        native_image.get(), 0, &entry));
    ASSERT_EQ(entry.relocation_count, 1u);
    EXPECT_EQ(entry.relocations[0].addend, addend);
    EXPECT_EQ(entry.relocations[0].minimum_value, 256u);
    EXPECT_EQ(entry.relocations[0].maximum_value, 4096u);
    EXPECT_EQ(entry.relocations[0].required_alignment, 64u);
  }
}

TEST(Aie2pNativeImageTest, RejectsRelocationsOutsideShimAddressFields) {
  Aie2pImageTargetState target_state;
  const auto target = MakeAie2pImageTarget(&target_state);
  // The block writes are structurally valid, but these sites are respectively
  // a descriptor length, a compute-row register and past the shim BD table.
  for (uint32_t address : {0x1D000u, 0x21D004u, 0x1D204u}) {
    Aie2pImageFixtureOptions options;
    options.control_register_address = address;
    ImagePtr image = CreateImage(BuildAie2pImage(options), &target);
    iree_hal_amd_xdna_aie2p_native_image_t* native_image = nullptr;
    EXPECT_THAT(
        Status(iree_hal_amd_xdna_aie2p_native_image_create(
            image.get(), &target, iree_allocator_system(), &native_image)),
        StatusIs(StatusCode::kUnimplemented));
    EXPECT_EQ(native_image, nullptr);
  }
}

TEST(Aie2pNativeImageTest, LowersCanonicalMulI32Image) {
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"),
      /*context_column_count=*/1, &target));
  ImagePtr image = CreateImage(LoadMulI32ImageBytes(), &target);
  NativeImagePtr native_image = CreateNativeImage(image.get(), &target);

  iree_const_byte_span_t array = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_native_image_query_array_configuration(
      native_image.get(), 0, &array));
  const std::vector<NativeOperation> array_operations = DecodeOperations(array);
  ExpectTransactionHeader(array, 50, 1);
  EXPECT_EQ(std::count_if(array_operations.begin(), array_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 0;
                          }),
            30);
  EXPECT_EQ(std::count_if(array_operations.begin(), array_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 1;
                          }),
            7);
  EXPECT_EQ(std::count_if(array_operations.begin(), array_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 3;
                          }),
            13);

  struct TileProgramExpectation {
    // Native context-relative program-memory address.
    uint32_t address;
    // Native block operation length after word padding.
    uint32_t operation_byte_length;
  };
  constexpr TileProgramExpectation kTilePrograms[] = {
      {UINT32_C(0x00220000), 836},
  };
  for (const TileProgramExpectation& expected : kTilePrograms) {
    const auto operation = std::find_if(
        array_operations.begin(), array_operations.end(),
        [&](const NativeOperation& operation) {
          return operation.opcode == 1 &&
                 operation.byte_length == expected.operation_byte_length &&
                 iree_unaligned_load_le_u32(array.data + operation.byte_offset +
                                            8) == expected.address;
        });
    EXPECT_NE(operation, array_operations.end());
  }

  iree_hal_amd_xdna_aie2p_native_entry_t entry = {};
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_native_image_query_entry(
      native_image.get(), 0, &entry));
  const std::vector<NativeOperation> control_operations =
      DecodeOperations(entry.control);
  ExpectTransactionHeader(entry.control, 8, 1);
  EXPECT_EQ(std::count_if(control_operations.begin(), control_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 0;
                          }),
            3);
  EXPECT_EQ(std::count_if(control_operations.begin(), control_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 1;
                          }),
            3);
  EXPECT_EQ(std::count_if(control_operations.begin(), control_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 3;
                          }),
            1);
  EXPECT_EQ(std::count_if(control_operations.begin(), control_operations.end(),
                          [](const NativeOperation& operation) {
                            return operation.opcode == 128;
                          }),
            1);

  constexpr uint64_t kPatchAddresses[] = {
      UINT64_C(0x0001D004),
      UINT64_C(0x0001D024),
      UINT64_C(0x0001D044),
  };
  ASSERT_EQ(entry.relocation_count, std::size(kPatchAddresses));
  for (iree_host_size_t i = 0; i < entry.relocation_count; ++i) {
    const auto& relocation = entry.relocations[i];
    EXPECT_EQ(relocation.binding_ordinal, i);
    EXPECT_EQ(relocation.addend, 0);
    const auto operation = std::find_if(
        control_operations.begin(), control_operations.end(),
        [&](const NativeOperation& operation) {
          return operation.opcode == 1 &&
                 relocation.byte_offset >= operation.byte_offset + 16 &&
                 relocation.byte_offset + 8 <=
                     operation.byte_offset + operation.byte_length;
        });
    ASSERT_NE(operation, control_operations.end());
    EXPECT_EQ(iree_unaligned_load_le_u32(entry.control.data +
                                         operation->byte_offset + 8) +
                  relocation.byte_offset - operation->byte_offset - 16,
              kPatchAddresses[i]);
  }
}

TEST(Aie2pNativeImageTest, RejectsMismatchedTarget) {
  Aie2pImageTargetState target_state;
  iree_hal_amd_xdna_aie2p_target_t target = MakeAie2pImageTarget(&target_state);
  ImagePtr image = CreateImage(BuildAie2pImage(), &target);
  ++target.identity.policy_id;

  auto* native_image =
      reinterpret_cast<iree_hal_amd_xdna_aie2p_native_image_t*>(uintptr_t{1});
  EXPECT_THAT(
      Status(iree_hal_amd_xdna_aie2p_native_image_create(
          image.get(), &target, iree_allocator_system(), &native_image)),
      StatusIs(StatusCode::kFailedPrecondition));
  EXPECT_EQ(native_image, nullptr);
}

}  // namespace
