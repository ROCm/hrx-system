// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"

iree_status_t iree_hal_amd_xdna_aie2p_target_validate(
    const iree_hal_amd_xdna_aie2p_target_t* target) {
  if (target == NULL || target->identity.device_profile_revision == 0 ||
      target->identity.device_profile_id == 0 ||
      target->identity.firmware_abi_id == 0 ||
      target->identity.policy_id == 0 ||
      (target->supported_capabilities &
       ~IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES) != 0 ||
      target->context.column_count == 0 ||
      target->context.column_count > UINT8_MAX ||
      target->context.row_count == 0 || target->context.row_count > UINT8_MAX ||
      target->native.transaction.device_generation == 0 ||
      target->native.transaction.memory_tile_row_count == 0 ||
      target->native.transaction.memory_tile_row_count >
          target->context.row_count ||
      target->native.register_address.row_shift == 0 ||
      target->native.register_address.row_shift >=
          target->native.register_address.column_shift ||
      target->native.register_address.column_shift >= 32 ||
      (uint64_t)(target->context.row_count - 1u)
              << target->native.register_address.row_shift >=
          UINT64_C(1) << target->native.register_address.column_shift ||
      (uint64_t)(target->context.column_count - 1u)
              << target->native.register_address.column_shift >
          UINT32_MAX ||
      target->native.program_memory.host_offset == 0 ||
      target->native.program_memory.host_offset % sizeof(uint32_t) != 0 ||
      (uint64_t)target->native.program_memory.host_offset >=
          UINT64_C(1) << target->native.register_address.row_shift ||
      target->tile_memory_resolver.fn == NULL ||
      target->configuration_register_validator.fn == NULL ||
      target->dma_task_wait_validator.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P image target facts are incomplete");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_target_validate_abi_note(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_elf_abi_note_t* abi_note) {
  if (target == NULL || abi_note == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P target ABI validation requires inputs");
  }
  if (abi_note->target_generation !=
          IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P ||
      abi_note->device_profile_revision !=
          target->identity.device_profile_revision ||
      abi_note->device_profile_id != target->identity.device_profile_id ||
      abi_note->firmware_abi_id != target->identity.firmware_abi_id ||
      abi_note->policy_id != target->identity.policy_id ||
      abi_note->context_column_count != target->context.column_count ||
      abi_note->context_row_count != target->context.row_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P image identity or logical context does not match the target");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_abi_note(
    const void* user_data, const iree_hal_amd_xdna_elf_abi_note_t* abi_note) {
  return iree_hal_amd_xdna_aie2p_target_validate_abi_note(
      (const iree_hal_amd_xdna_aie2p_target_t*)user_data, abi_note);
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_dma_task_wait(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait) {
  const uint16_t column_end = (uint16_t)wait->column + wait->column_count;
  const uint16_t row_end = (uint16_t)wait->row + wait->row_count;
  if (column_end > target->context.column_count ||
      row_end > target->context.row_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P DMA task wait exceeds the image context");
  }
  return target->dma_task_wait_validator.fn(
      target->dma_task_wait_validator.user_data, wait);
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_register_range(
    const iree_hal_amd_xdna_aie2p_target_t* target, uint32_t address,
    uint32_t word_count) {
  const uint32_t row_shift = target->native.register_address.row_shift;
  const uint32_t column_shift = target->native.register_address.column_shift;
  const uint32_t tile_offset_mask = (UINT32_C(1) << row_shift) - 1u;
  const uint32_t row_mask = (UINT32_C(1) << (column_shift - row_shift)) - 1u;
  const uint32_t column = address >> column_shift;
  const uint32_t row = (address >> row_shift) & row_mask;
  const uint32_t tile_offset = address & tile_offset_mask;
  const uint64_t tile_end =
      (uint64_t)tile_offset + (uint64_t)word_count * sizeof(uint32_t);
  if (column >= target->context.column_count ||
      row >= target->context.row_count ||
      tile_end > (uint64_t)tile_offset_mask + 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P configuration-register range exceeds the image context");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_program_record(
    const void* user_data, uint32_t program_header_ordinal,
    iree_hal_amd_xdna_elf_program_type_t program_type,
    uint32_t program_record_ordinal,
    const iree_hal_amd_xdna_elf_program_record_header_t* record_header,
    iree_const_byte_span_t record_storage,
    uint32_t* out_referenced_program_header_ordinal) {
  (void)program_header_ordinal;
  (void)program_record_ordinal;
  (void)record_header;
  const iree_hal_amd_xdna_aie2p_target_t* target =
      (const iree_hal_amd_xdna_aie2p_target_t*)user_data;
  if (program_type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY &&
      program_type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P record has an invalid program role");
  }

  iree_hal_amd_xdna_aie2p_program_record_t record;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_decode_program_record(record_storage, &record));
  switch (record.type) {
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32: {
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_register_range(
          target, record.value.register_write32.address, 1));
      return target->configuration_register_validator.fn(
          target->configuration_register_validator.user_data, program_type,
          &record);
    }
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32: {
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_register_range(
          target, record.value.register_mask_write32.address, 1));
      return target->configuration_register_validator.fn(
          target->configuration_register_validator.user_data, program_type,
          &record);
    }
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32: {
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_register_range(
          target, record.value.register_block_write32.address,
          record.value.register_block_write32.word_count));
      return target->configuration_register_validator.fn(
          target->configuration_register_validator.user_data, program_type,
          &record);
    }
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
      if (program_type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P CONTROL program contains a tile-program load");
      }
      *out_referenced_program_header_ordinal =
          record.value.tile_program_load.program_header_ordinal;
      return iree_ok_status();
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      if (program_type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P ARRAY program contains a DMA task wait");
      }
      return iree_hal_amd_xdna_aie2p_validate_dma_task_wait(
          target, &record.value.dma_task_wait);
    default:
      IREE_CHECK_UNREACHABLE("decoded AIE2P program record type");
  }
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_program_relocation(
    const void* user_data, iree_hal_amd_xdna_elf_program_type_t program_type,
    const iree_hal_amd_xdna_image_program_record_t* record,
    uint32_t record_relative_byte_offset,
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation) {
  (void)user_data;
  const uint64_t field_end =
      (uint64_t)record_relative_byte_offset + relocation->field_byte_width;
  if (program_type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL ||
      record->type !=
          IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32 ||
      record_relative_byte_offset <
          IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE ||
      record_relative_byte_offset % sizeof(uint32_t) != 0 ||
      field_end > record->source_range.length) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "AIE2P runtime relocation does not target block-write word data");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_target_initialize_image_target(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_hal_amd_xdna_image_target_t* out_image_target) {
  IREE_ASSERT_ARGUMENT(out_image_target);
  *out_image_target = (iree_hal_amd_xdna_image_target_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_target_validate(target));
  *out_image_target = (iree_hal_amd_xdna_image_target_t){
      .target_flags = IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS,
      .supported_capabilities = target->supported_capabilities,
      .abi_note_validator =
          {
              .fn = iree_hal_amd_xdna_aie2p_validate_abi_note,
              .user_data = target,
          },
      .tile_memory_resolver = target->tile_memory_resolver,
      .program_record_validator =
          {
              .fn = iree_hal_amd_xdna_aie2p_validate_program_record,
              .user_data = target,
          },
      .program_relocation_validator =
          {
              .fn = iree_hal_amd_xdna_aie2p_validate_program_relocation,
              .user_data = target,
          },
  };
  return iree_ok_status();
}
