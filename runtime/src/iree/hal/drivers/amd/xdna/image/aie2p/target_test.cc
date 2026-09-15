// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"

#include <array>
#include <cstdint>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using testing::HasSubstr;

typedef struct validator_state_t {
  // Number of configuration-register operations observed.
  iree_host_size_t configuration_register_call_count;
  // Number of DMA task waits observed.
  iree_host_size_t dma_task_wait_call_count;
  // Terminal code returned by the configuration-register validator.
  iree_status_code_t configuration_register_status_code;
  // Terminal code returned by the DMA task-wait validator.
  iree_status_code_t dma_task_wait_status_code;
  // Last program role supplied for a configuration-register operation.
  iree_hal_amd_xdna_elf_program_type_t last_program_type;
  // Last configuration-register record supplied by the target adapter.
  iree_hal_amd_xdna_aie2p_program_record_t last_configuration_register;
  // Last DMA task wait supplied by the target adapter.
  iree_hal_amd_xdna_aie2p_dma_task_wait_t last_dma_task_wait;
} validator_state_t;

static iree_status_t validate_configuration_register(
    const void* user_data, iree_hal_amd_xdna_elf_program_type_t program_type,
    const iree_hal_amd_xdna_aie2p_program_record_t* record) {
  auto* state = (validator_state_t*)user_data;
  ++state->configuration_register_call_count;
  state->last_program_type = program_type;
  state->last_configuration_register = *record;
  if (state->configuration_register_status_code != IREE_STATUS_OK) {
    return iree_make_status(state->configuration_register_status_code,
                            "test rejected configuration-register operation");
  }
  return iree_ok_status();
}

static iree_status_t validate_dma_task_wait(
    const void* user_data,
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait) {
  auto* state = (validator_state_t*)user_data;
  ++state->dma_task_wait_call_count;
  state->last_dma_task_wait = *wait;
  if (state->dma_task_wait_status_code != IREE_STATUS_OK) {
    return iree_make_status(state->dma_task_wait_status_code,
                            "test rejected DMA task wait");
  }
  return iree_ok_status();
}

static iree_status_t resolve_tile_memory(
    const void* user_data,
    const iree_hal_amd_xdna_elf_tile_destination_t* destination,
    uint32_t virtual_address, uint32_t byte_length,
    iree_hal_amd_xdna_image_tile_placement_t* out_placement) {
  (void)user_data;
  *out_placement = {
      /*.program_header_ordinal=*/0,
      /*.owner_column=*/destination->column,
      /*.owner_row=*/destination->row,
      /*.memory_space=*/destination->memory_space,
      /*.owner_offset=*/virtual_address,
      /*.byte_length=*/byte_length,
      /*.available_capacity=*/byte_length,
  };
  return iree_ok_status();
}

static iree_hal_amd_xdna_aie2p_target_t make_target(validator_state_t* state) {
  return {
      /*.identity=*/
      {
          /*.device_profile_revision=*/7,
          /*.device_profile_id=*/UINT64_C(0x535848414C4F0001),
          /*.firmware_abi_id=*/UINT64_C(0x4E5055320006000C),
          /*.policy_id=*/UINT64_C(0x0123456789ABCDEF),
      },
      /*.supported_capabilities=*/IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES,
      /*.context=*/
      {
          /*.column_count=*/3,
          /*.row_count=*/6,
      },
      /*.native=*/
      {
          /*.transaction=*/
          {
              /*.device_generation=*/4,
              /*.memory_tile_row_count=*/1,
          },
          /*.register_address=*/
          {
              /*.column_shift=*/25,
              /*.row_shift=*/20,
          },
          /*.program_memory=*/
          {
              /*.host_offset=*/UINT32_C(0x00020000),
          },
      },
      /*.tile_memory_resolver=*/
      {
          /*.fn=*/resolve_tile_memory,
          /*.user_data=*/state,
      },
      /*.configuration_register_validator=*/
      {
          /*.fn=*/validate_configuration_register,
          /*.user_data=*/state,
      },
      /*.dma_task_wait_validator=*/
      {
          /*.fn=*/validate_dma_task_wait,
          /*.user_data=*/state,
      },
  };
}

template <size_t N>
static Status validate_record(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_hal_amd_xdna_elf_program_type_t program_type,
    const std::array<uint8_t, N>& storage,
    uint32_t* out_referenced_program_header_ordinal = nullptr) {
  iree_hal_amd_xdna_elf_program_record_header_t header;
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_decode_program_record_header(
      iree_make_const_byte_span(
          storage.data(), IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE),
      &header));
  iree_hal_amd_xdna_image_target_t image_target;
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      target, &image_target));
  uint32_t referenced_program_header_ordinal = UINT32_MAX;
  iree_status_t status = image_target.program_record_validator.fn(
      image_target.program_record_validator.user_data,
      /*program_header_ordinal=*/7, program_type,
      /*program_record_ordinal=*/11, &header,
      iree_make_const_byte_span(storage.data(), storage.size()),
      &referenced_program_header_ordinal);
  if (iree_status_is_ok(status) && out_referenced_program_header_ordinal) {
    *out_referenced_program_header_ordinal = referenced_program_header_ordinal;
  }
  return Status(std::move(status));
}

static void expect_status(const Status& status, StatusCode expected_code,
                          const char* expected_message_substring) {
  EXPECT_EQ(status.code(), expected_code);
  EXPECT_THAT(status.ToString(), HasSubstr(expected_message_substring));
}

TEST(Aie2pTargetTest, ValidatesConfigurationRegisterOperations) {
  validator_state_t state = {};
  const iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_target_validate(&target));

  const iree_hal_amd_xdna_aie2p_register_mask_write32_t value = {
      /*.address=*/UINT32_C(0x0201D000),
      /*.mask=*/UINT32_C(0x0000FFFF),
      /*.value=*/UINT32_C(0x00001234),
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_MASK_WRITE32_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_mask_write32(
      &value, iree_make_byte_span(storage.data(), storage.size())));
  IREE_ASSERT_OK(validate_record(
      &target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL, storage));

  EXPECT_EQ(state.configuration_register_call_count, 1u);
  EXPECT_EQ(state.dma_task_wait_call_count, 0u);
  EXPECT_EQ(state.last_program_type,
            IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL);
  EXPECT_EQ(state.last_configuration_register.type,
            IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32);
  EXPECT_EQ(state.last_configuration_register.value.register_mask_write32.mask,
            value.mask);

  constexpr std::array<uint32_t, 2> kWords = {
      UINT32_C(0x11223344),
      UINT32_C(0xAABBCCDD),
  };
  std::array<uint8_t, 24> block_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_block_write32(
      UINT32_C(0x001D000), kWords.size(), kWords.data(),
      iree_make_byte_span(block_storage.data(), block_storage.size())));
  IREE_ASSERT_OK(validate_record(
      &target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, block_storage));
  EXPECT_EQ(state.configuration_register_call_count, 2u);
  EXPECT_EQ(state.last_program_type, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY);
  EXPECT_EQ(state.last_configuration_register.type,
            IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  uint32_t last_word = 0;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_program_record_read_block_word(
      &state.last_configuration_register, 1, &last_word));
  EXPECT_EQ(last_word, kWords[1]);
}

TEST(Aie2pTargetTest, AcceptsArrayTileProgramLoads) {
  validator_state_t state = {};
  const iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);
  const iree_hal_amd_xdna_aie2p_tile_program_load_t value = {
      /*.program_header_ordinal=*/17,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_tile_program_load(
      &value, iree_make_byte_span(storage.data(), storage.size())));

  uint32_t referenced_program_header_ordinal = UINT32_MAX;
  IREE_ASSERT_OK(validate_record(&target,
                                 IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                                 storage, &referenced_program_header_ordinal));
  EXPECT_EQ(referenced_program_header_ordinal, 17u);
  EXPECT_EQ(state.configuration_register_call_count, 0u);
  EXPECT_EQ(state.dma_task_wait_call_count, 0u);
}

TEST(Aie2pTargetTest, ValidatesContextBoundedControlDmaTaskWaits) {
  validator_state_t state = {};
  const iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);
  const iree_hal_amd_xdna_aie2p_dma_task_wait_t value = {
      /*.column=*/1,
      /*.row=*/0,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.column_count=*/2,
      /*.row_count=*/1,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
      &value, iree_make_byte_span(storage.data(), storage.size())));

  IREE_ASSERT_OK(validate_record(
      &target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL, storage));
  EXPECT_EQ(state.configuration_register_call_count, 0u);
  EXPECT_EQ(state.dma_task_wait_call_count, 1u);
  EXPECT_EQ(state.last_dma_task_wait.column, value.column);
  EXPECT_EQ(state.last_dma_task_wait.column_count, value.column_count);
}

TEST(Aie2pTargetTest, RejectsIncompleteTargetFacts) {
  validator_state_t state = {};
  iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);
  target.identity.policy_id = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.context.column_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.native.transaction.device_generation = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.native.register_address.row_shift =
      target.native.register_address.column_shift;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.native.program_memory.host_offset =
      UINT32_C(1) << target.native.register_address.row_shift;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.tile_memory_resolver.fn = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.configuration_register_validator.fn = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));
  target = make_target(&state);
  target.dma_task_wait_validator.fn = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_validate(&target));

  iree_hal_amd_xdna_image_target_t image_target = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_xdna_aie2p_target_initialize_image_target(
                            &target, &image_target));
  EXPECT_EQ(image_target.program_record_validator.fn, nullptr);
  EXPECT_EQ(image_target.program_record_validator.user_data, nullptr);
}

TEST(Aie2pTargetTest, RejectsRecordsInUnsupportedProgramRoles) {
  validator_state_t state = {};
  const iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);
  const iree_hal_amd_xdna_aie2p_tile_program_load_t load_value = {
      /*.program_header_ordinal=*/17,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE>
      load_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_tile_program_load(
      &load_value,
      iree_make_byte_span(load_storage.data(), load_storage.size())));
  expect_status(
      validate_record(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL,
                      load_storage),
      StatusCode::kInvalidArgument,
      "CONTROL program contains a tile-program load");

  const iree_hal_amd_xdna_aie2p_dma_task_wait_t wait_value = {
      /*.column=*/0,
      /*.row=*/0,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.column_count=*/1,
      /*.row_count=*/1,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      wait_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
      &wait_value,
      iree_make_byte_span(wait_storage.data(), wait_storage.size())));
  expect_status(
      validate_record(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                      wait_storage),
      StatusCode::kInvalidArgument, "ARRAY program contains a DMA task wait");
  EXPECT_EQ(state.configuration_register_call_count, 0u);
  EXPECT_EQ(state.dma_task_wait_call_count, 0u);
}

TEST(Aie2pTargetTest, RejectsDmaTaskWaitOutsideImageContext) {
  validator_state_t state = {};
  const iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);
  const iree_hal_amd_xdna_aie2p_dma_task_wait_t value = {
      /*.column=*/2,
      /*.row=*/5,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_MEMORY_TO_STREAM,
      /*.dma_channel=*/0,
      /*.column_count=*/2,
      /*.row_count=*/1,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
      &value, iree_make_byte_span(storage.data(), storage.size())));

  expect_status(
      validate_record(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL,
                      storage),
      StatusCode::kOutOfRange, "exceeds the image context");
  EXPECT_EQ(state.dma_task_wait_call_count, 0u);
}

TEST(Aie2pTargetTest, PropagatesTargetFactFailures) {
  validator_state_t state = {};
  state.configuration_register_status_code = IREE_STATUS_PERMISSION_DENIED;
  state.dma_task_wait_status_code = IREE_STATUS_UNIMPLEMENTED;
  const iree_hal_amd_xdna_aie2p_target_t target = make_target(&state);

  const iree_hal_amd_xdna_aie2p_register_write32_t write_value = {
      /*.address=*/UINT32_C(0x00100000),
      /*.value=*/0,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE>
      write_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_register_write32(
      &write_value,
      iree_make_byte_span(write_storage.data(), write_storage.size())));
  expect_status(
      validate_record(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                      write_storage),
      StatusCode::kPermissionDenied,
      "rejected configuration-register operation");

  const iree_hal_amd_xdna_aie2p_dma_task_wait_t wait_value = {
      /*.column=*/0,
      /*.row=*/0,
      /*.direction=*/
      IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY,
      /*.dma_channel=*/0,
      /*.column_count=*/1,
      /*.row_count=*/1,
  };
  std::array<uint8_t, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE>
      wait_storage;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
      &wait_value,
      iree_make_byte_span(wait_storage.data(), wait_storage.size())));
  expect_status(
      validate_record(&target, IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL,
                      wait_storage),
      StatusCode::kUnimplemented, "rejected DMA task wait");
}

}  // namespace
