// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/image.h"

#include "iree/hal/driver/amd/xdna/image/validation.h"

struct iree_hal_amd_xdna_image_t {
  // Allocator owning this object and its trailing metadata bytes.
  iree_allocator_t host_allocator;
  // Structural directory retaining the immutable source.
  iree_hal_amd_xdna_image_directory_t* directory;
  // Admitted table view over this object's trailing bytes.
  iree_hal_amd_xdna_image_tables_t tables;
};

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
  iree_hal_amd_xdna_image_directory_t* directory = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_directory_allocate(
      source_sequence, host_allocator, &directory));
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
  iree_hal_amd_xdna_image_t* image = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_uninitialized(
        host_allocator, sizeof(*image) + metadata->file_range.length,
        (void**)&image);
  }
  if (iree_status_is_ok(status)) {
    image->host_allocator = host_allocator;
    image->directory = directory;
    status = iree_hal_amd_xdna_image_directory_read_source_range(
        directory, metadata->file_range,
        iree_make_byte_span(image + 1, metadata->file_range.length));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_tables_initialize(
        iree_make_const_byte_span(image + 1, metadata->file_range.length),
        &image->tables);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_xdna_image_validate(&image->tables, directory, target);
  }
  if (iree_status_is_ok(status)) {
    *out_image = image;
  } else {
    iree_allocator_free(host_allocator, image);
    iree_hal_amd_xdna_image_directory_free(directory);
  }
  return status;
}

void iree_hal_amd_xdna_image_free(iree_hal_amd_xdna_image_t* image) {
  if (image == NULL) {
    return;
  }
  iree_hal_amd_xdna_image_directory_free(image->directory);
  iree_allocator_free(image->host_allocator, image);
}

const iree_hal_amd_xdna_image_tables_t* iree_hal_amd_xdna_image_tables(
    const iree_hal_amd_xdna_image_t* image) {
  return &image->tables;
}

const iree_hal_amd_xdna_image_directory_t* iree_hal_amd_xdna_image_directory(
    const iree_hal_amd_xdna_image_t* image) {
  return image->directory;
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
