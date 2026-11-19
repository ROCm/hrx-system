// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/image.h"

#include <string.h>

#include "iree/hal/driver/amd/xdna/image/aie_full_elf.h"
#include "iree/hal/driver/amd/xdna/image/directory.h"
#include "iree/hal/driver/amd/xdna/image/validation.h"

struct iree_hal_amd_xdna_image_t {
  // Allocator owning the image and its execution tables.
  iree_allocator_t host_allocator;
  // Retained source of the payload ranges described by the tables.
  iree_byte_sequence_t* source;
  // Admitted execution metadata.
  iree_hal_amd_xdna_image_tables_t tables;
};

static iree_status_t iree_hal_amd_xdna_image_read_native(
    iree_byte_sequence_t* source,
    const iree_hal_amd_xdna_aie2p_target_t* target, iree_allocator_t allocator,
    iree_hal_amd_xdna_image_tables_t* out_tables) {
  iree_hal_amd_xdna_image_directory_t* directory = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_directory_allocate(
      source, allocator, &directory));
  const iree_hal_amd_xdna_image_program_header_t* metadata =
      iree_hal_amd_xdna_image_directory_program_header(directory, 0);
  iree_status_t status = iree_ok_status();
  if (metadata == NULL ||
      metadata->type != IREE_XDNA_ELF_PROGRAM_TYPE_METADATA ||
      metadata->flags != IREE_XDNA_ELF_PROGRAM_FLAG_READ ||
      metadata->virtual_address != 0 || metadata->physical_address != 0 ||
      metadata->memory_size != metadata->file_range.length ||
      metadata->file_range.length < IREE_XDNA_ELF_HEADER_RECORD_SIZE ||
      metadata->file_range.length > IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA header zero must contain bounded metadata");
  }
  const uint32_t load_count =
      (uint32_t)iree_hal_amd_xdna_image_directory_program_header_count(
          directory);
  uint8_t* copy = NULL;
  if (iree_status_is_ok(status)) {
    iree_const_byte_span_t contents;
    if (iree_byte_sequence_try_get_contiguous_span(source, &contents)) {
      contents =
          iree_make_const_byte_span(contents.data + metadata->file_range.offset,
                                    metadata->file_range.length);
    } else {
      status = iree_allocator_malloc_uninitialized(
          allocator, metadata->file_range.length, (void**)&copy);
      if (iree_status_is_ok(status)) {
        status = iree_byte_sequence_read(
            source, metadata->file_range.offset,
            iree_make_byte_span(copy, metadata->file_range.length));
      }
      contents = iree_make_const_byte_span(copy, metadata->file_range.length);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_xdna_image_tables_decode(contents, load_count,
                                                     allocator, out_tables);
    }
  }
  iree_allocator_free(allocator, copy);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate(out_tables, directory, target);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 1; i < load_count; ++i) {
      const iree_hal_amd_xdna_image_program_header_t* load =
          iree_hal_amd_xdna_image_directory_program_header(directory, i);
      out_tables->loads[i] = (iree_hal_amd_xdna_image_load_t){
          .source_offset = (uint32_t)load->file_range.offset,
          .destination_offset = load->virtual_address,
          .file_size = (uint32_t)load->file_range.length,
          .memory_size = load->memory_size,
      };
    }
  }
  iree_hal_amd_xdna_image_directory_free(directory);
  return status;
}

iree_status_t iree_hal_amd_xdna_image_allocate(
    iree_byte_sequence_t* source_sequence,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_allocator_t host_allocator, iree_hal_amd_xdna_image_t** out_image) {
  IREE_ASSERT_ARGUMENT(out_image);
  *out_image = NULL;
  if (target == NULL || target->context.column_count == 0 ||
      target->context.row_count == 0 ||
      !iree_is_power_of_two_uint64(target->instruction_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid XDNA target contract");
  }
  if (!source_sequence) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image source sequence is required");
  }
  uint8_t os_abi = 0;
  IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
      source_sequence, 7 /* EI_OSABI */, iree_make_byte_span(&os_abi, 1)));
  iree_hal_amd_xdna_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*image), (void**)&image));
  image->host_allocator = host_allocator;
  image->source = source_sequence;
  iree_byte_sequence_retain(source_sequence);
  iree_status_t status =
      os_abi == IREE_HAL_AMD_AIE_OS_ABI
          ? iree_hal_amd_xdna_image_read_aie_full_elf(
                source_sequence, target, host_allocator, &image->tables)
          : iree_hal_amd_xdna_image_read_native(source_sequence, target,
                                                host_allocator, &image->tables);
  if (iree_status_is_ok(status)) {
    *out_image = image;
  } else {
    iree_hal_amd_xdna_image_free(image);
  }
  return status;
}

void iree_hal_amd_xdna_image_free(iree_hal_amd_xdna_image_t* image) {
  if (image == NULL) {
    return;
  }
  iree_hal_amd_xdna_image_tables_deinitialize(&image->tables,
                                              image->host_allocator);
  iree_byte_sequence_release(image->source);
  iree_allocator_free(image->host_allocator, image);
}

const iree_hal_amd_xdna_image_tables_t* iree_hal_amd_xdna_image_tables(
    const iree_hal_amd_xdna_image_t* image) {
  return &image->tables;
}

iree_status_t iree_hal_amd_xdna_image_load_allocation(
    const iree_hal_amd_xdna_image_t* image, uint32_t allocation_ordinal,
    uint8_t* target) {
  const iree_xdna_elf_allocation_record_t allocation =
      iree_hal_amd_xdna_image_tables_allocation(&image->tables,
                                                allocation_ordinal);
  for (uint32_t i = 0; i < allocation.load_count; ++i) {
    const iree_hal_amd_xdna_image_load_t* load =
        &image->tables.loads[allocation.first_load + i];
    uint8_t* destination = target + load->destination_offset;
    IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
        image->source, load->source_offset,
        iree_make_byte_span(destination, load->file_size)));
    memset(destination + load->file_size, 0,
           load->memory_size - load->file_size);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_image_find_entry(
    const iree_hal_amd_xdna_image_t* image, iree_string_view_t name,
    uint32_t* out_ordinal) {
  for (uint32_t i = 0; i < image->tables.header.entry_count; ++i) {
    const iree_xdna_elf_entry_record_t entry =
        iree_hal_amd_xdna_image_tables_entry(&image->tables, i);
    if (iree_string_view_equal(name, iree_hal_amd_xdna_image_tables_entry_name(
                                         &image->tables, &entry))) {
      *out_ordinal = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "XDNA entry '%.*s' was not found", (int)name.size,
                          name.data);
}
