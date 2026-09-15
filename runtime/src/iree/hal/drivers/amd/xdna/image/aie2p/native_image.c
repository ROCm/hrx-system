// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/native_image.h"

#include <string.h>

#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/transaction.h"

// Measured storage and scratch requirements for one native transaction.
typedef struct iree_hal_amd_xdna_aie2p_native_measurement_t {
  // Exact complete transaction byte length.
  iree_host_size_t byte_length;
  // Maximum transient source byte length used while encoding.
  iree_host_size_t scratch_byte_length;
  // Number of cold relocation records accompanying the transaction.
  iree_host_size_t relocation_count;
} iree_hal_amd_xdna_aie2p_native_measurement_t;

struct iree_hal_amd_xdna_aie2p_native_image_t {
  // Allocator owning this image and its trailing storage.
  iree_allocator_t host_allocator;
  // Number of entries in |arrays|.
  iree_host_size_t array_count;
  // Native ARRAY transactions in generic image ARRAY order.
  iree_const_byte_span_t* arrays;
  // Number of entries in |entries|.
  iree_host_size_t entry_count;
  // Native entry records in generic image entry order.
  iree_hal_amd_xdna_aie2p_native_entry_t* entries;
};

static iree_status_t iree_hal_amd_xdna_aie2p_native_measurement_add(
    iree_host_size_t operation_byte_length,
    iree_host_size_t scratch_byte_length,
    iree_hal_amd_xdna_aie2p_native_measurement_t* measurement) {
  iree_host_size_t new_byte_length = 0;
  if (!iree_host_size_checked_add(measurement->byte_length,
                                  operation_byte_length, &new_byte_length) ||
      new_byte_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P native transaction exceeds format limits");
  }
  measurement->byte_length = new_byte_length;
  measurement->scratch_byte_length =
      iree_max(measurement->scratch_byte_length, scratch_byte_length);
  return iree_ok_status();
}

static const iree_hal_amd_xdna_image_array_program_t*
iree_hal_amd_xdna_aie2p_native_find_array(
    const iree_hal_amd_xdna_image_t* image, uint32_t program_header_ordinal,
    iree_host_size_t* out_array_ordinal) {
  const iree_host_size_t array_count =
      iree_hal_amd_xdna_image_array_count(image);
  for (iree_host_size_t i = 0; i < array_count; ++i) {
    const iree_hal_amd_xdna_image_array_program_t* array =
        iree_hal_amd_xdna_image_array(image, i);
    if (array->program_header_ordinal == program_header_ordinal) {
      *out_array_ordinal = i;
      return array;
    }
  }
  return NULL;
}

static const iree_hal_amd_xdna_image_control_program_t*
iree_hal_amd_xdna_aie2p_native_find_control(
    const iree_hal_amd_xdna_image_t* image, uint32_t program_header_ordinal) {
  const iree_host_size_t control_count =
      iree_hal_amd_xdna_image_control_count(image);
  for (iree_host_size_t i = 0; i < control_count; ++i) {
    const iree_hal_amd_xdna_image_control_program_t* control =
        iree_hal_amd_xdna_image_control(image, i);
    if (control->program_header_ordinal == program_header_ordinal) {
      return control;
    }
  }
  return NULL;
}

static const iree_hal_amd_xdna_image_tile_placement_t*
iree_hal_amd_xdna_aie2p_native_find_tile_placement(
    const iree_hal_amd_xdna_image_t* image, uint32_t program_header_ordinal) {
  const iree_host_size_t placement_count =
      iree_hal_amd_xdna_image_tile_placement_count(image);
  for (iree_host_size_t i = 0; i < placement_count; ++i) {
    const iree_hal_amd_xdna_image_tile_placement_t* placement =
        iree_hal_amd_xdna_image_tile_placement(image, i);
    if (placement->program_header_ordinal == program_header_ordinal) {
      return placement;
    }
  }
  return NULL;
}

static iree_host_size_t iree_hal_amd_xdna_aie2p_native_first_relocation(
    const iree_hal_amd_xdna_image_t* image, uint32_t program_header_ordinal) {
  iree_host_size_t low = 0;
  iree_host_size_t high = iree_hal_amd_xdna_image_relocation_count(image);
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation =
        iree_hal_amd_xdna_image_relocation(image, middle);
    if (relocation->target_program_header_ordinal < program_header_ordinal) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

static iree_host_size_t iree_hal_amd_xdna_aie2p_native_relocation_end(
    const iree_hal_amd_xdna_image_t* image, uint32_t program_header_ordinal,
    iree_host_size_t first_ordinal) {
  const iree_host_size_t relocation_count =
      iree_hal_amd_xdna_image_relocation_count(image);
  iree_host_size_t ordinal = first_ordinal;
  while (ordinal < relocation_count &&
         iree_hal_amd_xdna_image_relocation(image, ordinal)
                 ->target_program_header_ordinal == program_header_ordinal) {
    ++ordinal;
  }
  return ordinal;
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_measure_tile_load(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_image_program_record_t* record,
    iree_hal_amd_xdna_aie2p_native_measurement_t* measurement) {
  const iree_hal_amd_xdna_image_program_header_t* tile_header =
      iree_hal_amd_xdna_image_program_header(
          image, record->referenced_program_header_ordinal);
  const iree_hal_amd_xdna_image_tile_placement_t* placement =
      iree_hal_amd_xdna_aie2p_native_find_tile_placement(
          image, record->referenced_program_header_ordinal);
  if (tile_header == NULL || placement == NULL ||
      tile_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified AIE2P TILE load has no canonical placement");
  }
  iree_host_size_t padded_byte_length = 0;
  if (!iree_host_size_checked_align(tile_header->memory_size, sizeof(uint32_t),
                                    &padded_byte_length) ||
      padded_byte_length > placement->available_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P TILE program does not fit its native word aperture");
  }
  return iree_hal_amd_xdna_aie2p_native_measurement_add(
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE +
          padded_byte_length,
      padded_byte_length, measurement);
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_measure_record(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_image_program_record_t* record,
    iree_hal_amd_xdna_aie2p_native_measurement_t* measurement) {
  if (record->source_range.length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P image record exceeds host address space");
  }
  const iree_host_size_t source_byte_length =
      (iree_host_size_t)record->source_range.length;
  switch (record->type) {
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      return iree_hal_amd_xdna_aie2p_native_measurement_add(
          IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_WRITE32_SIZE,
          source_byte_length, measurement);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      return iree_hal_amd_xdna_aie2p_native_measurement_add(
          IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_MASK_WRITE32_SIZE,
          source_byte_length, measurement);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
      if (source_byte_length <
          IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "qualified AIE2P block-write record is truncated");
      }
      return iree_hal_amd_xdna_aie2p_native_measurement_add(
          IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE +
              source_byte_length -
              IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE,
          source_byte_length, measurement);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
      return iree_hal_amd_xdna_aie2p_native_measure_tile_load(image, record,
                                                              measurement);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      return iree_hal_amd_xdna_aie2p_native_measurement_add(
          IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_DMA_TASK_WAIT_SIZE,
          source_byte_length, measurement);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "qualified AIE2P record type is unknown");
  }
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_measure_array(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_image_array_program_t* array,
    iree_hal_amd_xdna_aie2p_native_measurement_t* out_measurement) {
  *out_measurement = (iree_hal_amd_xdna_aie2p_native_measurement_t){
      .byte_length = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE,
  };
  const uint64_t record_end =
      (uint64_t)array->first_record_ordinal + array->record_count;
  if (record_end > iree_hal_amd_xdna_image_record_count(image)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified AIE2P ARRAY record range is inconsistent");
  }
  for (uint32_t ordinal = array->first_record_ordinal; ordinal < record_end;
       ++ordinal) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_measure_record(
        image, iree_hal_amd_xdna_image_record(image, ordinal),
        out_measurement));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_measure_control(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_image_control_program_t* control,
    iree_hal_amd_xdna_aie2p_native_measurement_t* out_measurement) {
  *out_measurement = (iree_hal_amd_xdna_aie2p_native_measurement_t){
      .byte_length = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE,
  };
  if (control == NULL) return iree_ok_status();

  const uint64_t record_end =
      (uint64_t)control->first_record_ordinal + control->record_count;
  if (record_end > iree_hal_amd_xdna_image_record_count(image)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified AIE2P CONTROL record range is inconsistent");
  }
  for (uint32_t ordinal = control->first_record_ordinal; ordinal < record_end;
       ++ordinal) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_measure_record(
        image, iree_hal_amd_xdna_image_record(image, ordinal),
        out_measurement));
  }

  const iree_host_size_t first_relocation =
      iree_hal_amd_xdna_aie2p_native_first_relocation(
          image, control->program_header_ordinal);
  const iree_host_size_t relocation_end =
      iree_hal_amd_xdna_aie2p_native_relocation_end(
          image, control->program_header_ordinal, first_relocation);
  out_measurement->relocation_count = relocation_end - first_relocation;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_validate_relocations(
    const iree_hal_amd_xdna_image_t* image) {
  const iree_host_size_t relocation_count =
      iree_hal_amd_xdna_image_relocation_count(image);
  for (iree_host_size_t i = 0; i < relocation_count; ++i) {
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation =
        iree_hal_amd_xdna_image_relocation(image, i);
    const iree_hal_amd_xdna_image_program_header_t* target_program =
        iree_hal_amd_xdna_image_program_header(
            image, relocation->target_program_header_ordinal);
    if (target_program == NULL ||
        target_program->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL ||
        relocation->kind !=
            IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS ||
        relocation->field_byte_width != sizeof(uint64_t)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P transaction 0.1 lowering requires 64-bit CONTROL address "
          "relocations");
    }
  }
  return iree_ok_status();
}

// Cursor used to copy segmented image bytes into bounded scratch storage.
typedef struct iree_hal_amd_xdna_aie2p_native_copy_cursor_t {
  // First unwritten output byte.
  uint8_t* data;
  // Number of writable bytes remaining.
  iree_host_size_t remaining;
} iree_hal_amd_xdna_aie2p_native_copy_cursor_t;

static iree_status_t iree_hal_amd_xdna_aie2p_native_copy_segment(
    void* user_data, iree_const_byte_span_t segment) {
  iree_hal_amd_xdna_aie2p_native_copy_cursor_t* cursor =
      (iree_hal_amd_xdna_aie2p_native_copy_cursor_t*)user_data;
  if (segment.data_length > cursor->remaining) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P image source changed while native bytes were copied");
  }
  memcpy(cursor->data, segment.data, segment.data_length);
  cursor->data += segment.data_length;
  cursor->remaining -= segment.data_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_read_record(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_image_program_record_t* record,
    iree_byte_span_t scratch,
    iree_hal_amd_xdna_aie2p_program_record_t* out_record) {
  if (record->source_range.length > scratch.data_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P native scratch storage is too small");
  }
  const iree_host_size_t byte_length =
      (iree_host_size_t)record->source_range.length;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_read_source_range(
      image, record->source_range,
      iree_make_byte_span(scratch.data, byte_length)));
  return iree_hal_amd_xdna_aie2p_decode_program_record(
      iree_make_const_byte_span(scratch.data, byte_length), out_record);
}

static void iree_hal_amd_xdna_aie2p_native_decode_tile_address(
    const iree_hal_amd_xdna_aie2p_target_t* target, uint32_t address,
    uint8_t* out_column, uint8_t* out_row) {
  const uint32_t row_bit_count = target->native.register_address.column_shift -
                                 target->native.register_address.row_shift;
  const uint32_t row_mask = (UINT32_C(1) << row_bit_count) - 1u;
  *out_column =
      (uint8_t)(address >> target->native.register_address.column_shift);
  *out_row = (uint8_t)((address >> target->native.register_address.row_shift) &
                       row_mask);
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_append_tile_load(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    uint32_t program_header_ordinal, iree_byte_span_t scratch,
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer) {
  const iree_hal_amd_xdna_image_program_header_t* tile_header =
      iree_hal_amd_xdna_image_program_header(image, program_header_ordinal);
  const iree_hal_amd_xdna_image_tile_placement_t* placement =
      iree_hal_amd_xdna_aie2p_native_find_tile_placement(
          image, program_header_ordinal);
  if (tile_header == NULL || placement == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified AIE2P TILE load has no canonical placement");
  }

  iree_host_size_t padded_byte_length = 0;
  if (!iree_host_size_checked_align(tile_header->memory_size, sizeof(uint32_t),
                                    &padded_byte_length) ||
      padded_byte_length > scratch.data_length ||
      padded_byte_length > placement->available_capacity ||
      tile_header->file_range.length > tile_header->memory_size) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified AIE2P TILE program does not fit native storage");
  }
  memset(scratch.data, 0, padded_byte_length);
  iree_hal_amd_xdna_aie2p_native_copy_cursor_t copy_cursor = {
      .data = scratch.data,
      .remaining = (iree_host_size_t)tile_header->file_range.length,
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_enumerate_source_range(
      image, tile_header->file_range,
      (iree_byte_sequence_segment_callback_t){
          .fn = iree_hal_amd_xdna_aie2p_native_copy_segment,
          .user_data = &copy_cursor,
      }));
  if (copy_cursor.remaining != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P image source changed while native bytes were copied");
  }

  iree_host_size_t tile_local_address = 0;
  iree_host_size_t tile_local_end = 0;
  if (!iree_host_size_checked_add(target->native.program_memory.host_offset,
                                  placement->owner_offset,
                                  &tile_local_address) ||
      !iree_host_size_checked_add(tile_local_address, padded_byte_length,
                                  &tile_local_end) ||
      tile_local_end >
          (UINT64_C(1) << target->native.register_address.row_shift)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AIE2P TILE program exceeds its tile-local address field");
  }
  const uint32_t address = ((uint32_t)placement->owner_column
                            << target->native.register_address.column_shift) |
                           ((uint32_t)placement->owner_row
                            << target->native.register_address.row_shift) |
                           (uint32_t)tile_local_address;
  return iree_hal_amd_xdna_aie2p_transaction_writer_append_block_write32(
      writer, address, (uint8_t)placement->owner_column,
      (uint8_t)placement->owner_row,
      iree_make_const_byte_span(scratch.data, padded_byte_length));
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_append_decoded_record(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_aie2p_program_record_t* decoded_record,
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer) {
  switch (decoded_record->type) {
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      return iree_hal_amd_xdna_aie2p_transaction_writer_append_write32(
          writer, decoded_record->value.register_write32.address,
          decoded_record->value.register_write32.value);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      return iree_hal_amd_xdna_aie2p_transaction_writer_append_mask_write32(
          writer, decoded_record->value.register_mask_write32.address,
          decoded_record->value.register_mask_write32.value,
          decoded_record->value.register_mask_write32.mask);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32: {
      const iree_hal_amd_xdna_aie2p_register_block_write32_t* block =
          &decoded_record->value.register_block_write32;
      uint8_t column = 0;
      uint8_t row = 0;
      iree_hal_amd_xdna_aie2p_native_decode_tile_address(target, block->address,
                                                         &column, &row);
      return iree_hal_amd_xdna_aie2p_transaction_writer_append_block_write32(
          writer, block->address, column, row, block->word_data);
    }
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      return iree_hal_amd_xdna_aie2p_transaction_writer_append_dma_task_wait(
          writer, &decoded_record->value.dma_task_wait);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "qualified AIE2P record changed during lowering");
  }
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_append_record(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_image_program_record_t* record,
    iree_byte_span_t scratch,
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer) {
  if (record->type ==
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
    return iree_hal_amd_xdna_aie2p_native_append_tile_load(
        image, target, record->referenced_program_header_ordinal, scratch,
        writer);
  }
  iree_hal_amd_xdna_aie2p_program_record_t decoded_record;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_read_record(
      image, record, scratch, &decoded_record));
  return iree_hal_amd_xdna_aie2p_native_append_decoded_record(
      target, &decoded_record, writer);
}

static iree_hal_amd_xdna_aie2p_transaction_target_t
iree_hal_amd_xdna_aie2p_native_transaction_target(
    const iree_hal_amd_xdna_aie2p_target_t* target) {
  return (iree_hal_amd_xdna_aie2p_transaction_target_t){
      .device_generation = target->native.transaction.device_generation,
      .row_count = (uint8_t)target->context.row_count,
      .column_count = (uint8_t)target->context.column_count,
      .memory_tile_row_count = target->native.transaction.memory_tile_row_count,
  };
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_encode_array(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_image_array_program_t* array,
    iree_byte_span_t scratch, iree_byte_span_t storage) {
  const iree_hal_amd_xdna_aie2p_transaction_target_t transaction_target =
      iree_hal_amd_xdna_aie2p_native_transaction_target(target);
  iree_hal_amd_xdna_aie2p_transaction_writer_t writer;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_initialize(
      &transaction_target, storage, &writer));
  const uint64_t record_end =
      (uint64_t)array->first_record_ordinal + array->record_count;
  for (uint32_t ordinal = array->first_record_ordinal; ordinal < record_end;
       ++ordinal) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_append_record(
        image, target, iree_hal_amd_xdna_image_record(image, ordinal), scratch,
        &writer));
  }
  iree_const_byte_span_t transaction = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_finalize(
      &writer, &transaction));
  return transaction.data_length == storage.data_length
             ? iree_ok_status()
             : iree_make_status(
                   IREE_STATUS_FAILED_PRECONDITION,
                   "AIE2P ARRAY changed after native storage measurement");
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_append_record_relocations(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_elf_entry_record_t* entry,
    const iree_hal_amd_xdna_image_program_header_t* control_header,
    const iree_hal_amd_xdna_image_program_record_t* record,
    const iree_hal_amd_xdna_aie2p_program_record_t* decoded_record,
    iree_host_size_t relocation_end, iree_host_size_t* relocation_ordinal,
    iree_host_size_t native_record_offset,
    iree_hal_amd_xdna_aie2p_native_relocation_t** native_relocation) {
  const uint64_t record_offset =
      record->source_range.offset - control_header->file_range.offset;
  const uint64_t record_end = record_offset + record->source_range.length;
  while (*relocation_ordinal < relocation_end) {
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation =
        iree_hal_amd_xdna_image_relocation(image, *relocation_ordinal);
    if (relocation->target_byte_offset >= record_end) break;
    if (relocation->binding_ordinal < entry->first_binding_ordinal ||
        relocation->binding_ordinal - entry->first_binding_ordinal >=
            entry->binding_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AIE2P CONTROL relocation references another entry's binding");
    }
    const uint64_t word_byte_offset =
        relocation->target_byte_offset - record_offset -
        IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE;
    const uint64_t register_address =
        decoded_record->value.register_block_write32.address + word_byte_offset;
    // AIE2P shim BDs have sixteen 32-byte records. Words 1 and 2 hold the
    // 48-bit byte address; word 2 also contains descriptor configuration.
    const uint32_t column_offset =
        (uint32_t)register_address &
        ((UINT32_C(1) << target->native.register_address.column_shift) - 1);
    if (column_offset < 0x1D004 || column_offset > 0x1D1E4 ||
        (column_offset - 0x1D004) % 0x20 != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P cold address relocation requires a shim DMA address field");
    }
    **native_relocation = (iree_hal_amd_xdna_aie2p_native_relocation_t){
        .byte_offset =
            (uint32_t)(native_record_offset +
                       IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE +
                       word_byte_offset),
        .binding_ordinal =
            relocation->binding_ordinal - entry->first_binding_ordinal,
        .addend = relocation->addend,
        .minimum_value = relocation->minimum_value,
        .maximum_value =
            iree_min(relocation->maximum_value, UINT64_C(0xFFFFFFFFFFFF)),
        .required_alignment = iree_max(relocation->required_alignment, 4),
    };
    ++*native_relocation;
    ++*relocation_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_native_encode_control(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_elf_entry_record_t* entry,
    const iree_hal_amd_xdna_image_control_program_t* control,
    iree_byte_span_t scratch, iree_byte_span_t storage,
    iree_hal_amd_xdna_aie2p_native_relocation_t* native_relocations) {
  const iree_hal_amd_xdna_aie2p_transaction_target_t transaction_target =
      iree_hal_amd_xdna_aie2p_native_transaction_target(target);
  iree_hal_amd_xdna_aie2p_transaction_writer_t writer;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_initialize(
      &transaction_target, storage, &writer));

  iree_host_size_t relocation_ordinal = 0;
  iree_host_size_t relocation_end = 0;
  const iree_hal_amd_xdna_image_program_header_t* control_header = NULL;
  if (control != NULL) {
    control_header = iree_hal_amd_xdna_image_program_header(
        image, control->program_header_ordinal);
    if (control_header == NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "qualified AIE2P CONTROL program header is missing");
    }
    relocation_ordinal = iree_hal_amd_xdna_aie2p_native_first_relocation(
        image, control->program_header_ordinal);
    relocation_end = iree_hal_amd_xdna_aie2p_native_relocation_end(
        image, control->program_header_ordinal, relocation_ordinal);
    const uint64_t record_end =
        (uint64_t)control->first_record_ordinal + control->record_count;
    for (uint32_t ordinal = control->first_record_ordinal; ordinal < record_end;
         ++ordinal) {
      const iree_hal_amd_xdna_image_program_record_t* record =
          iree_hal_amd_xdna_image_record(image, ordinal);
      iree_hal_amd_xdna_aie2p_program_record_t decoded_record;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_read_record(
          image, record, scratch, &decoded_record));
      const iree_host_size_t native_record_offset = writer.byte_offset;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_append_decoded_record(
          target, &decoded_record, &writer));
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_xdna_aie2p_native_append_record_relocations(
              image, target, entry, control_header, record, &decoded_record,
              relocation_end, &relocation_ordinal, native_record_offset,
              &native_relocations));
    }
  }
  if (relocation_ordinal != relocation_end) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified AIE2P CONTROL relocations were not completely lowered");
  }

  iree_const_byte_span_t transaction = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_finalize(
      &writer, &transaction));
  return transaction.data_length == storage.data_length
             ? iree_ok_status()
             : iree_make_status(
                   IREE_STATUS_FAILED_PRECONDITION,
                   "AIE2P CONTROL changed after native storage measurement");
}

iree_status_t iree_hal_amd_xdna_aie2p_native_image_create(
    const iree_hal_amd_xdna_image_t* image,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_aie2p_native_image_t** out_native_image) {
  IREE_ASSERT_ARGUMENT(out_native_image);
  *out_native_image = NULL;
  if (image == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P native image source is required");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_target_validate(target));
  if (iree_hal_amd_xdna_image_target_flags(image) !=
      IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "XDNA image is not an AIE2P executable");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_target_validate_abi_note(
      target, iree_hal_amd_xdna_image_abi_note(image)));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_native_validate_relocations(image));

  const iree_host_size_t array_count =
      iree_hal_amd_xdna_image_array_count(image);
  const iree_host_size_t entry_count =
      iree_hal_amd_xdna_image_entry_count(image);
  iree_host_size_t native_byte_length = 0;
  iree_host_size_t native_relocation_count = 0;
  iree_host_size_t scratch_byte_length = 0;
  for (iree_host_size_t i = 0; i < array_count; ++i) {
    iree_hal_amd_xdna_aie2p_native_measurement_t measurement;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_measure_array(
        image, iree_hal_amd_xdna_image_array(image, i), &measurement));
    if (!iree_host_size_checked_add(native_byte_length, measurement.byte_length,
                                    &native_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P native image storage overflows");
    }
    scratch_byte_length =
        iree_max(scratch_byte_length, measurement.scratch_byte_length);
  }
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_hal_amd_xdna_elf_entry_record_t* entry =
        iree_hal_amd_xdna_image_entry(image, i);
    const iree_hal_amd_xdna_image_control_program_t* control =
        entry->control_program_header_ordinal == UINT32_MAX
            ? NULL
            : iree_hal_amd_xdna_aie2p_native_find_control(
                  image, entry->control_program_header_ordinal);
    if (entry->control_program_header_ordinal != UINT32_MAX &&
        control == NULL) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "qualified AIE2P entry has no CONTROL program");
    }
    iree_hal_amd_xdna_aie2p_native_measurement_t measurement;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_measure_control(
        image, control, &measurement));
    if (!iree_host_size_checked_add(native_relocation_count,
                                    measurement.relocation_count,
                                    &native_relocation_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P native relocation storage overflows");
    }
    if (!iree_host_size_checked_add(native_byte_length, measurement.byte_length,
                                    &native_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P native image storage overflows");
    }
    scratch_byte_length =
        iree_max(scratch_byte_length, measurement.scratch_byte_length);
  }

  iree_host_size_t arrays_offset = 0;
  iree_host_size_t entries_offset = 0;
  iree_host_size_t relocations_offset = 0;
  iree_host_size_t native_bytes_offset = 0;
  iree_host_size_t total_byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_aie2p_native_image_t), &total_byte_length,
      IREE_STRUCT_FIELD_ALIGNED(array_count, iree_const_byte_span_t,
                                iree_alignof(iree_const_byte_span_t),
                                &arrays_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          entry_count, iree_hal_amd_xdna_aie2p_native_entry_t,
          iree_alignof(iree_hal_amd_xdna_aie2p_native_entry_t),
          &entries_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          native_relocation_count, iree_hal_amd_xdna_aie2p_native_relocation_t,
          iree_alignof(iree_hal_amd_xdna_aie2p_native_relocation_t),
          &relocations_offset),
      IREE_STRUCT_FIELD_ALIGNED(native_byte_length, uint8_t,
                                iree_alignof(uint32_t), &native_bytes_offset)));

  iree_hal_amd_xdna_aie2p_native_image_t* native_image = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, total_byte_length, (void**)&native_image);
  if (iree_status_is_ok(status)) {
    *native_image = (iree_hal_amd_xdna_aie2p_native_image_t){
        .host_allocator = host_allocator,
        .array_count = array_count,
        .arrays =
            (iree_const_byte_span_t*)((uint8_t*)native_image + arrays_offset),
        .entry_count = entry_count,
        .entries =
            (iree_hal_amd_xdna_aie2p_native_entry_t*)((uint8_t*)native_image +
                                                      entries_offset),
    };
  }

  iree_byte_span_t scratch = iree_byte_span_empty();
  if (iree_status_is_ok(status) && scratch_byte_length != 0) {
    scratch.data_length = scratch_byte_length;
    status = iree_allocator_malloc_uninitialized(
        host_allocator, scratch.data_length, (void**)&scratch.data);
  }

  iree_host_size_t native_bytes_cursor = native_bytes_offset;
  iree_host_size_t native_relocation_cursor = 0;
  for (iree_host_size_t i = 0; i < array_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_xdna_aie2p_native_measurement_t measurement;
    status = iree_hal_amd_xdna_aie2p_native_measure_array(
        image, iree_hal_amd_xdna_image_array(image, i), &measurement);
    if (!iree_status_is_ok(status)) break;
    iree_byte_span_t storage = iree_make_byte_span(
        (uint8_t*)native_image + native_bytes_cursor, measurement.byte_length);
    status = iree_hal_amd_xdna_aie2p_native_encode_array(
        image, target, iree_hal_amd_xdna_image_array(image, i), scratch,
        storage);
    if (!iree_status_is_ok(status)) break;
    native_image->arrays[i] =
        iree_make_const_byte_span(storage.data, storage.data_length);
    native_bytes_cursor += storage.data_length;
  }

  for (iree_host_size_t i = 0; i < entry_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_xdna_elf_entry_record_t* entry =
        iree_hal_amd_xdna_image_entry(image, i);
    iree_host_size_t array_ordinal = 0;
    if (iree_hal_amd_xdna_aie2p_native_find_array(
            image, entry->array_program_header_ordinal, &array_ordinal) ==
        NULL) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "qualified AIE2P entry has no ARRAY realization");
      break;
    }
    const iree_hal_amd_xdna_image_control_program_t* control =
        entry->control_program_header_ordinal == UINT32_MAX
            ? NULL
            : iree_hal_amd_xdna_aie2p_native_find_control(
                  image, entry->control_program_header_ordinal);
    iree_hal_amd_xdna_aie2p_native_measurement_t measurement;
    status = iree_hal_amd_xdna_aie2p_native_measure_control(image, control,
                                                            &measurement);
    if (!iree_status_is_ok(status)) break;
    iree_byte_span_t storage = iree_make_byte_span(
        (uint8_t*)native_image + native_bytes_cursor, measurement.byte_length);
    iree_hal_amd_xdna_aie2p_native_relocation_t* relocations =
        (iree_hal_amd_xdna_aie2p_native_relocation_t*)((uint8_t*)native_image +
                                                       relocations_offset) +
        native_relocation_cursor;
    status = iree_hal_amd_xdna_aie2p_native_encode_control(
        image, target, entry, control, scratch, storage, relocations);
    if (!iree_status_is_ok(status)) break;
    native_image->entries[i] = (iree_hal_amd_xdna_aie2p_native_entry_t){
        .array_ordinal = (uint32_t)array_ordinal,
        .control = iree_make_const_byte_span(storage.data, storage.data_length),
        .relocation_count = measurement.relocation_count,
        .relocations = relocations,
    };
    native_relocation_cursor += measurement.relocation_count;
    native_bytes_cursor += storage.data_length;
  }

  iree_allocator_free(host_allocator, scratch.data);
  if (iree_status_is_ok(status) &&
      native_bytes_cursor != native_bytes_offset + native_byte_length) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P native image changed after storage measurement");
  }
  if (iree_status_is_ok(status)) {
    *out_native_image = native_image;
  } else {
    iree_hal_amd_xdna_aie2p_native_image_destroy(native_image);
  }
  return status;
}

void iree_hal_amd_xdna_aie2p_native_image_destroy(
    iree_hal_amd_xdna_aie2p_native_image_t* native_image) {
  if (native_image == NULL) return;
  const iree_allocator_t host_allocator = native_image->host_allocator;
  iree_allocator_free(host_allocator, native_image);
}

iree_host_size_t iree_hal_amd_xdna_aie2p_native_image_array_count(
    const iree_hal_amd_xdna_aie2p_native_image_t* native_image) {
  IREE_ASSERT_ARGUMENT(native_image);
  return native_image->array_count;
}

iree_status_t iree_hal_amd_xdna_aie2p_native_image_query_array_configuration(
    const iree_hal_amd_xdna_aie2p_native_image_t* native_image,
    iree_host_size_t ordinal, iree_const_byte_span_t* out_transaction) {
  IREE_ASSERT_ARGUMENT(out_transaction);
  *out_transaction = iree_const_byte_span_empty();
  if (native_image == NULL || ordinal >= native_image->array_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P native ARRAY ordinal is out of range");
  }
  *out_transaction = native_image->arrays[ordinal];
  return iree_ok_status();
}

iree_host_size_t iree_hal_amd_xdna_aie2p_native_image_entry_count(
    const iree_hal_amd_xdna_aie2p_native_image_t* native_image) {
  IREE_ASSERT_ARGUMENT(native_image);
  return native_image->entry_count;
}

iree_status_t iree_hal_amd_xdna_aie2p_native_image_query_entry(
    const iree_hal_amd_xdna_aie2p_native_image_t* native_image,
    iree_host_size_t ordinal,
    iree_hal_amd_xdna_aie2p_native_entry_t* out_entry) {
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = (iree_hal_amd_xdna_aie2p_native_entry_t){0};
  if (native_image == NULL || ordinal >= native_image->entry_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P native entry ordinal is out of range");
  }
  *out_entry = native_image->entries[ordinal];
  return iree_ok_status();
}
