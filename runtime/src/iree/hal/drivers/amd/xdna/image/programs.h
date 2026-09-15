// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded ARRAY and CONTROL program framing for AMD XDNA images.
//
// This layer decodes target-independent program envelopes and record framing.
// Exact record vocabularies remain target-owned and are accepted through a
// validator callback during transactional construction.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_PROGRAMS_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_PROGRAMS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/directory.h"
#include "iree/hal/drivers/amd/xdna/image/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One structurally decoded ARRAY realization.
typedef struct iree_hal_amd_xdna_image_array_program_t {
  // Program-header ordinal containing this ARRAY payload.
  uint32_t program_header_ordinal;
  // First TILE program-header ordinal used by the realization.
  uint32_t first_tile_program_header_ordinal;
  // Number of consecutive TILE program headers used by the realization.
  uint32_t tile_program_header_count;
  // First row in the flattened program-record table.
  uint32_t first_record_ordinal;
  // Number of framed configuration records.
  uint32_t record_count;
} iree_hal_amd_xdna_image_array_program_t;

// One structurally decoded CONTROL program.
typedef struct iree_hal_amd_xdna_image_control_program_t {
  // Program-header ordinal containing this CONTROL payload.
  uint32_t program_header_ordinal;
  // First row in the flattened program-record table.
  uint32_t first_record_ordinal;
  // Number of framed control records.
  uint32_t record_count;
} iree_hal_amd_xdna_image_control_program_t;

// One framed ARRAY or CONTROL record.
typedef struct iree_hal_amd_xdna_image_program_record_t {
  // Program-header ordinal containing this record.
  uint32_t program_header_ordinal;
  // Dense record ordinal within its containing program.
  uint32_t program_record_ordinal;
  // Target-owned payload ABI record type.
  uint16_t type;
  // Target-owned payload ABI record flags.
  uint16_t flags;
  // Referenced program-header ordinal, or UINT32_MAX when absent.
  uint32_t referenced_program_header_ordinal;
  // Complete serialized record bytes in the image source sequence.
  iree_hal_amd_xdna_image_source_range_t source_range;
} iree_hal_amd_xdna_image_program_record_t;

// Validates one complete target-specific ARRAY or CONTROL record.
//
// Generic framing and source bounds are proven before invocation. The callback
// must reject every type, flag, byte length, or payload field unsupported by
// the selected target. |record_storage| includes the common record header and
// remains valid only for the call. |out_referenced_program_header_ordinal| is
// initialized to UINT32_MAX and may be replaced when the record contains one
// program reference requiring whole-image validation.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amd_xdna_image_program_record_validate_fn_t)(
    const void* user_data, uint32_t program_header_ordinal,
    iree_hal_amd_xdna_elf_program_type_t program_type,
    uint32_t program_record_ordinal,
    const iree_hal_amd_xdna_elf_program_record_header_t* record_header,
    iree_const_byte_span_t record_storage,
    uint32_t* out_referenced_program_header_ordinal);

// Target-specific record validator borrowed during program construction.
typedef struct iree_hal_amd_xdna_image_program_record_validator_t {
  // Function invoked for every framed ARRAY and CONTROL record.
  iree_hal_amd_xdna_image_program_record_validate_fn_t fn;
  // Unowned immutable target facts passed to |fn|.
  const void* user_data;
} iree_hal_amd_xdna_image_program_record_validator_t;

// Immutable decoded program framing independent of its source directory.
typedef struct iree_hal_amd_xdna_image_programs_t
    iree_hal_amd_xdna_image_programs_t;

// Decodes and validates ARRAY and CONTROL programs from |directory|.
//
// The directory and validator are borrowed only for the duration of the call.
// Construction reads at most one bounded program payload at a time and retains
// only compact envelopes and record source ranges. TILE references and all
// cross-component relationships are validated by the whole-image layer.
iree_status_t iree_hal_amd_xdna_image_programs_create(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_hal_amd_xdna_image_program_record_validator_t validator,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_programs_t** out_programs);

// Destroys |programs|.
void iree_hal_amd_xdna_image_programs_destroy(
    iree_hal_amd_xdna_image_programs_t* programs);

// Returns the number of decoded ARRAY realizations.
iree_host_size_t iree_hal_amd_xdna_image_programs_array_count(
    const iree_hal_amd_xdna_image_programs_t* programs);

// Returns the ARRAY realization at |ordinal|, or NULL when out of range.
const iree_hal_amd_xdna_image_array_program_t*
iree_hal_amd_xdna_image_programs_array(
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_host_size_t ordinal);

// Returns the number of decoded CONTROL programs.
iree_host_size_t iree_hal_amd_xdna_image_programs_control_count(
    const iree_hal_amd_xdna_image_programs_t* programs);

// Returns the CONTROL program at |ordinal|, or NULL when out of range.
const iree_hal_amd_xdna_image_control_program_t*
iree_hal_amd_xdna_image_programs_control(
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_host_size_t ordinal);

// Returns the number of framed ARRAY and CONTROL records.
iree_host_size_t iree_hal_amd_xdna_image_programs_record_count(
    const iree_hal_amd_xdna_image_programs_t* programs);

// Returns the flattened program record at |ordinal|, or NULL when out of range.
const iree_hal_amd_xdna_image_program_record_t*
iree_hal_amd_xdna_image_programs_record(
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_host_size_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_PROGRAMS_H_
