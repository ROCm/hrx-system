// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/image.h"

struct iree_hal_amd_xdna_image_t {
  // Allocator owning this image object.
  iree_allocator_t host_allocator;
  // Decoded source directory retaining the immutable source byte sequence.
  iree_hal_amd_xdna_image_directory_t* directory;
  // Decoded target-independent metadata tables.
  iree_hal_amd_xdna_image_tables_t* tables;
  // Decoded and target-qualified ARRAY and CONTROL programs.
  iree_hal_amd_xdna_image_programs_t* programs;
  // Whole-image relationship and placement qualification results.
  iree_hal_amd_xdna_image_validation_t* validation;
};

iree_status_t iree_hal_amd_xdna_image_create(
    iree_byte_sequence_t* source_sequence,
    const iree_hal_amd_xdna_image_target_t* target,
    iree_allocator_t host_allocator, iree_hal_amd_xdna_image_t** out_image) {
  IREE_ASSERT_ARGUMENT(out_image);
  *out_image = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_target_validate(target));

  iree_hal_amd_xdna_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*image), (void**)&image));
  *image = (iree_hal_amd_xdna_image_t){
      .host_allocator = host_allocator,
  };

  iree_status_t status = iree_hal_amd_xdna_image_directory_create(
      source_sequence, host_allocator, &image->directory);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_tables_create(
        image->directory, host_allocator, &image->tables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_programs_create(
        image->directory, target->program_record_validator, host_allocator,
        &image->programs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validation_create(
        image->directory, image->tables, image->programs, target,
        host_allocator, &image->validation);
  }

  if (iree_status_is_ok(status)) {
    *out_image = image;
  } else {
    iree_hal_amd_xdna_image_destroy(image);
  }
  return status;
}

void iree_hal_amd_xdna_image_destroy(iree_hal_amd_xdna_image_t* image) {
  if (image == NULL) return;
  const iree_allocator_t host_allocator = image->host_allocator;
  iree_hal_amd_xdna_image_validation_destroy(image->validation);
  iree_hal_amd_xdna_image_programs_destroy(image->programs);
  iree_hal_amd_xdna_image_tables_destroy(image->tables);
  iree_hal_amd_xdna_image_directory_destroy(image->directory);
  iree_allocator_free(host_allocator, image);
}

uint64_t iree_hal_amd_xdna_image_source_length(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_directory_source_length(image->directory);
}

uint32_t iree_hal_amd_xdna_image_target_flags(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_directory_target_flags(image->directory);
}

iree_host_size_t iree_hal_amd_xdna_image_program_header_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_directory_program_header_count(
      image->directory);
}

const iree_hal_amd_xdna_image_program_header_t*
iree_hal_amd_xdna_image_program_header(const iree_hal_amd_xdna_image_t* image,
                                       iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_directory_program_header(image->directory,
                                                          ordinal);
}

const iree_hal_amd_xdna_elf_abi_note_t* iree_hal_amd_xdna_image_abi_note(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_abi_note(image->tables);
}

iree_host_size_t iree_hal_amd_xdna_image_entry_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_entry_count(image->tables);
}

const iree_hal_amd_xdna_elf_entry_record_t* iree_hal_amd_xdna_image_entry(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_entry(image->tables, ordinal);
}

iree_string_view_t iree_hal_amd_xdna_image_entry_name(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_entry_name(image->tables, ordinal);
}

iree_host_size_t iree_hal_amd_xdna_image_binding_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_binding_count(image->tables);
}

const iree_hal_amd_xdna_elf_binding_record_t* iree_hal_amd_xdna_image_binding(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_binding(image->tables, ordinal);
}

iree_host_size_t iree_hal_amd_xdna_image_relocation_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_relocation_count(image->tables);
}

const iree_hal_amd_xdna_elf_relocation_record_t*
iree_hal_amd_xdna_image_relocation(const iree_hal_amd_xdna_image_t* image,
                                   iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_tables_relocation(image->tables, ordinal);
}

iree_host_size_t iree_hal_amd_xdna_image_array_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_programs_array_count(image->programs);
}

const iree_hal_amd_xdna_image_array_program_t* iree_hal_amd_xdna_image_array(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_programs_array(image->programs, ordinal);
}

iree_host_size_t iree_hal_amd_xdna_image_control_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_programs_control_count(image->programs);
}

const iree_hal_amd_xdna_image_control_program_t*
iree_hal_amd_xdna_image_control(const iree_hal_amd_xdna_image_t* image,
                                iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_programs_control(image->programs, ordinal);
}

iree_host_size_t iree_hal_amd_xdna_image_record_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_programs_record_count(image->programs);
}

const iree_hal_amd_xdna_image_program_record_t* iree_hal_amd_xdna_image_record(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_programs_record(image->programs, ordinal);
}

iree_hal_amd_xdna_elf_capabilities_t
iree_hal_amd_xdna_image_structural_capabilities(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_validation_structural_capabilities(
      image->validation);
}

iree_host_size_t iree_hal_amd_xdna_image_tile_placement_count(
    const iree_hal_amd_xdna_image_t* image) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_validation_tile_placement_count(
      image->validation);
}

const iree_hal_amd_xdna_image_tile_placement_t*
iree_hal_amd_xdna_image_tile_placement(const iree_hal_amd_xdna_image_t* image,
                                       iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_validation_tile_placement(image->validation,
                                                           ordinal);
}

iree_status_t iree_hal_amd_xdna_image_enumerate_source_range(
    const iree_hal_amd_xdna_image_t* image,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_sequence_segment_callback_t callback) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_directory_enumerate_source_range(
      image->directory, source_range, callback);
}

iree_status_t iree_hal_amd_xdna_image_read_source_range(
    const iree_hal_amd_xdna_image_t* image,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(image);
  return iree_hal_amd_xdna_image_directory_read_source_range(
      image->directory, source_range, storage);
}
