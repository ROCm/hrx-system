// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/executable_measurement.h"

static void iree_hal_amd_xdna_measurement_write_timer(uint8_t* target,
                                                      uint32_t id) {
  iree_unaligned_store_le_u32(target, 0x83);  // RECORD_TIMER.
  iree_unaligned_store_le_u32(target + 4, 12);
  iree_unaligned_store_le_u32(target + 8, id);
}

iree_status_t iree_hal_amd_xdna_executable_measurement_initialize(
    const iree_xdna_elf_invocation_record_t* invocation, const uint8_t* command,
    uint8_t* target) {
  const uint32_t operation_count = iree_unaligned_load_le_u32(command + 8);
  if (command[0] != 0 || command[1] != 1 ||
      iree_unaligned_load_le_u32(command + 12) != invocation->byte_length ||
      operation_count > UINT32_MAX - 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid XDNA measured command header");
  }
  const uint32_t begin = invocation->measurement.begin;
  const uint32_t end = invocation->measurement.end;
  memcpy(target, command, begin);
  iree_hal_amd_xdna_measurement_write_timer(
      target + begin, IREE_HAL_AMD_XDNA_MEASUREMENT_BEGIN_ID);
  memcpy(target + begin + 12, command + begin, end - begin);
  iree_hal_amd_xdna_measurement_write_timer(
      target + end + 12, IREE_HAL_AMD_XDNA_MEASUREMENT_END_ID);
  memcpy(target + end + 24, command + end, invocation->byte_length - end);
  iree_unaligned_store_le_u32(target + 8, operation_count + 2);
  iree_unaligned_store_le_u32(target + 12, invocation->byte_length + 24);
  return iree_ok_status();
}

iree_byte_span_t iree_hal_amd_xdna_executable_measurement_patch(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_xdna_elf_entry_record_t* entry,
    const iree_xdna_elf_invocation_record_t* invocation, const uint8_t* command,
    uint8_t* target) {
  uint32_t begin = UINT32_MAX, end = 0;
  for (uint32_t i = 0; i < entry->dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry->first_dynamic_relocation + i);
    if (relocation.destination_use != invocation->allocation_use ||
        relocation.byte_offset < invocation->byte_offset ||
        relocation.byte_offset - invocation->byte_offset >=
            invocation->byte_length) {
      continue;
    }
    const uint32_t source_offset =
        relocation.byte_offset - invocation->byte_offset;
    const uint32_t target_offset =
        source_offset +
        (source_offset >= invocation->measurement.begin ? 12 : 0) +
        (source_offset >= invocation->measurement.end ? 12 : 0);
    memcpy(target + target_offset, command + source_offset, 8);
    begin = iree_min(begin, target_offset);
    end = iree_max(end, target_offset + 8);
  }
  return end ? iree_make_byte_span(target + begin, end - begin)
             : iree_byte_span_empty();
}
