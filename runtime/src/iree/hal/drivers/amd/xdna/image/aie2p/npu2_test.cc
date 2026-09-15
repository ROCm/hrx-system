// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"

#include <array>
#include <cstdint>
#include <utility>

#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;
using testing::HasSubstr;

struct DeviceProfile {
  // Canonical endpoint key under test.
  const char* target_id;
  // Independent compiler identity expected for that endpoint.
  uint64_t identity;
};

class Aie2pNpu2Test : public ::testing::TestWithParam<DeviceProfile> {
 protected:
  iree_hal_amd_xdna_aie2p_target_t MakeTarget(
      uint16_t context_column_count = 3) {
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(GetParam().target_id), context_column_count,
        &target));
    return target;
  }
};

INSTANTIATE_TEST_SUITE_P(
    Devices, Aie2pNpu2Test,
    ::testing::Values(DeviceProfile{"amd.xdna.strix.17f0_10",
                                    UINT64_C(0x5354524958000001)},
                      DeviceProfile{"amd.xdna.strix_halo.17f0_11",
                                    UINT64_C(0x535848414C4F0001)}));

template <size_t N>
static Status ValidateRecord(const iree_hal_amd_xdna_aie2p_target_t* target,
                             iree_hal_amd_xdna_elf_program_type_t program_type,
                             const std::array<uint8_t, N>& storage) {
  iree_hal_amd_xdna_elf_program_record_header_t record_header;
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_decode_program_record_header(
      iree_make_const_byte_span(
          storage.data(), IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE),
      &record_header));
  iree_hal_amd_xdna_image_target_t image_target;
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      target, &image_target));
  uint32_t referenced_program_header_ordinal = UINT32_MAX;
  return Status(image_target.program_record_validator.fn(
      image_target.program_record_validator.user_data,
      /*program_header_ordinal=*/0, program_type,
      /*program_record_ordinal=*/0, &record_header,
      iree_make_const_byte_span(storage.data(), storage.size()),
      &referenced_program_header_ordinal));
}

static Status ResolveTileMemory(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_elf_tile_destination_t& destination,
    uint32_t virtual_address, uint32_t byte_length,
    iree_hal_amd_xdna_image_tile_placement_t* out_placement) {
  return Status(target->tile_memory_resolver.fn(
      target->tile_memory_resolver.user_data, &destination, virtual_address,
      byte_length, out_placement));
}

static void ExpectStatus(const Status& status, StatusCode expected_code,
                         const char* expected_message_substring) {
  EXPECT_EQ(status.code(), expected_code);
  EXPECT_THAT(status.ToString(), HasSubstr(expected_message_substring));
}

TEST_P(Aie2pNpu2Test, InitializesExactTargetFacts) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  EXPECT_EQ(target.identity.device_profile_revision, 1u);
  EXPECT_EQ(target.identity.device_profile_id, GetParam().identity);
  IREE_EXPECT_OK(iree_hal_amd_xdna_aie2p_target_validate(&target));
  EXPECT_EQ(target.identity.firmware_abi_id, UINT64_C(0x4E5055320006000C));
  EXPECT_EQ(target.identity.policy_id, UINT64_C(0x413250504C414E01));
  EXPECT_EQ(target.supported_capabilities,
            IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES);
  EXPECT_EQ(target.context.column_count, 3u);
  EXPECT_EQ(target.context.row_count, 6u);
  EXPECT_EQ(target.native.transaction.device_generation, 4u);
  EXPECT_EQ(target.native.transaction.memory_tile_row_count, 1u);
  EXPECT_EQ(target.native.register_address.column_shift, 25u);
  EXPECT_EQ(target.native.register_address.row_shift, 20u);
  EXPECT_EQ(target.native.program_memory.host_offset, UINT32_C(0x00020000));
  EXPECT_NE(target.tile_memory_resolver.fn, nullptr);
  EXPECT_NE(target.configuration_register_validator.fn, nullptr);
  EXPECT_NE(target.dma_task_wait_validator.fn, nullptr);
}

TEST_P(Aie2pNpu2Test, RejectsInvalidContextWidths) {
  iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_amd_xdna_aie2p_npu2_target_initialize(
          iree_make_cstring_view(GetParam().target_id), 0, &target));
  EXPECT_EQ(target.context.column_count, 3u);
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_amd_xdna_aie2p_npu2_target_initialize(
          iree_make_cstring_view(GetParam().target_id), 9, &target));
  EXPECT_EQ(target.context.column_count, 3u);
}

TEST_P(Aie2pNpu2Test, RejectsUnknownDeviceWithoutPublishing) {
  iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  IREE_EXPECT_STATUS_IS(StatusCode::kUnimplemented,
                        iree_hal_amd_xdna_aie2p_npu2_target_initialize(
                            IREE_SV("amd.xdna.strix.17f0_ff"), 1, &target));
  EXPECT_EQ(target.identity.device_profile_id, GetParam().identity);
  EXPECT_EQ(target.context.column_count, 3u);
}

TEST_P(Aie2pNpu2Test, ResolvesProgramAndDataMemoryOwners) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  iree_hal_amd_xdna_image_tile_placement_t placement;

  iree_hal_amd_xdna_elf_tile_destination_t destination = {
      /*.column=*/2,
      /*.row=*/4,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM,
      /*.flags=*/0,
  };
  IREE_ASSERT_OK(ResolveTileMemory(&target, destination, 0, 32, &placement));
  EXPECT_EQ(placement.owner_column, 2u);
  EXPECT_EQ(placement.owner_row, 4u);
  EXPECT_EQ(placement.owner_offset, 0u);
  EXPECT_EQ(placement.available_capacity, 16u * 1024u);

  destination = {
      /*.column=*/1,
      /*.row=*/3,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
      /*.flags=*/0,
  };
  IREE_ASSERT_OK(
      ResolveTileMemory(&target, destination, 0x50020, 64, &placement));
  EXPECT_EQ(placement.owner_column, 0u);
  EXPECT_EQ(placement.owner_row, 3u);
  EXPECT_EQ(placement.owner_offset, 0x20u);
  EXPECT_EQ(placement.available_capacity, 64u * 1024u - 0x20u);

  destination = {
      /*.column=*/1,
      /*.row=*/1,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
      /*.flags=*/0,
  };
  IREE_ASSERT_OK(
      ResolveTileMemory(&target, destination, 0x100040, 64, &placement));
  EXPECT_EQ(placement.owner_column, 2u);
  EXPECT_EQ(placement.owner_row, 1u);
  EXPECT_EQ(placement.owner_offset, 0x40u);
  EXPECT_EQ(placement.available_capacity, 512u * 1024u - 0x40u);
}

TEST_P(Aie2pNpu2Test, RejectsUnimplementedTileMemoryWindows) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  iree_hal_amd_xdna_image_tile_placement_t placement;

  const iree_hal_amd_xdna_elf_tile_destination_t first_compute_row = {
      /*.column=*/1,
      /*.row=*/2,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
      /*.flags=*/0,
  };
  EXPECT_THAT(
      ResolveTileMemory(&target, first_compute_row, 0x40000, 4, &placement),
      StatusIs(StatusCode::kOutOfRange));

  const iree_hal_amd_xdna_elf_tile_destination_t west_array_edge = {
      /*.column=*/0,
      /*.row=*/1,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
      /*.flags=*/0,
  };
  EXPECT_THAT(
      ResolveTileMemory(&target, west_array_edge, 0x00000, 4, &placement),
      StatusIs(StatusCode::kOutOfRange));

  const iree_hal_amd_xdna_elf_tile_destination_t shim_program = {
      /*.column=*/0,
      /*.row=*/0,
      /*.memory_space=*/IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM,
      /*.flags=*/0,
  };
  EXPECT_THAT(ResolveTileMemory(&target, shim_program, 0, 4, &placement),
              StatusIs(StatusCode::kOutOfRange));

  iree_hal_amd_xdna_elf_tile_destination_t displaced_program = shim_program;
  displaced_program.row = 2;
  EXPECT_THAT(ResolveTileMemory(&target, displaced_program, 4, 4, &placement),
              StatusIs(StatusCode::kOutOfRange));
}

TEST_P(Aie2pNpu2Test, ValidatesExactRegisterCorpus) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();

  iree_hal_amd_xdna_aie2p_register_write32_t write = {
      /*.address=*/UINT32_C(0x0021D000),
      /*.value=*/UINT32_C(0x00012345),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE>
      write_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
      &write, iree_make_byte_span(write_storage.data(), write_storage.size())));
  IREE_EXPECT_OK(ValidateRecord(
      &target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, write_storage));

  write.value = UINT32_C(0x10000000);
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
      &write, iree_make_byte_span(write_storage.data(), write_storage.size())));
  ExpectStatus(ValidateRecord(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                              write_storage),
               StatusCode::kPermissionDenied, "writes reserved NPU2 bits");

  write.address = UINT32_C(0x00200000);
  write.value = 0;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
      &write, iree_make_byte_span(write_storage.data(), write_storage.size())));
  ExpectStatus(ValidateRecord(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                              write_storage),
               StatusCode::kPermissionDenied, "is not writable");

  iree_hal_amd_xdna_aie2p_register_mask_write32_t mask_write = {
      /*.address=*/UINT32_C(0x0021D000),
      /*.mask=*/UINT32_C(0x0000000F),
      /*.value=*/UINT32_C(0xFFFF000A),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_MASK_WRITE32_RECORD_SIZE>
      mask_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_mask_write32(
      &mask_write,
      iree_make_byte_span(mask_storage.data(), mask_storage.size())));
  IREE_EXPECT_OK(ValidateRecord(
      &target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, mask_storage));

  mask_write.mask = UINT32_C(0x10000000);
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_mask_write32(
      &mask_write,
      iree_make_byte_span(mask_storage.data(), mask_storage.size())));
  ExpectStatus(ValidateRecord(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                              mask_storage),
               StatusCode::kPermissionDenied, "selects reserved NPU2 bits");
}

TEST_P(Aie2pNpu2Test, RejectsRegisterRangesOutsideLogicalContext) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  iree_hal_amd_xdna_aie2p_register_write32_t write = {
      /*.address=*/UINT32_C(0x0621D000),
      /*.value=*/0,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
      &write, iree_make_byte_span(storage.data(), storage.size())));
  ExpectStatus(ValidateRecord(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                              storage),
               StatusCode::kOutOfRange, "exceeds the image context");

  constexpr std::array<uint32_t, 2> kWords = {0, 0};
  std::array<uint8_t, 24> block_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_block_write32(
      UINT32_C(0x000FFFFC), kWords.size(), kWords.data(),
      iree_make_byte_span(block_storage.data(), block_storage.size())));
  ExpectStatus(ValidateRecord(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                              block_storage),
               StatusCode::kOutOfRange, "exceeds the image context");
}

TEST_P(Aie2pNpu2Test, ValidatesDmaTaskWaitChannelsByTileKind) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  iree_hal_amd_xdna_aie2p_dma_task_wait_t wait = {
      /*.column=*/0,
      /*.row=*/1,
      /*.direction=*/IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_MEMORY_TO_STREAM,
      /*.dma_channel=*/5,
      /*.column_count=*/1,
      /*.row_count=*/1,
  };
  IREE_EXPECT_OK(target.dma_task_wait_validator.fn(
      target.dma_task_wait_validator.user_data, &wait));

  wait.row = 2;
  EXPECT_THAT(Status(target.dma_task_wait_validator.fn(
                  target.dma_task_wait_validator.user_data, &wait)),
              StatusIs(StatusCode::kOutOfRange));
  wait.row = 1;
  wait.row_count = 2;
  EXPECT_THAT(Status(target.dma_task_wait_validator.fn(
                  target.dma_task_wait_validator.user_data, &wait)),
              StatusIs(StatusCode::kOutOfRange));
}

}  // namespace
