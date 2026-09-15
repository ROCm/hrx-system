// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable validated AMD XDNA executable images.
//
// An image owns the complete offline decoding and target qualification result
// while retaining the source byte sequence for later program loading. It is
// immutable after creation and may be queried concurrently. It has no provider
// or device state and issues no device operations.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_IMAGE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_IMAGE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/directory.h"
#include "iree/hal/drivers/amd/xdna/image/programs.h"
#include "iree/hal/drivers/amd/xdna/image/tables.h"
#include "iree/hal/drivers/amd/xdna/image/validation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An immutable decoded and target-qualified executable image.
typedef struct iree_hal_amd_xdna_image_t iree_hal_amd_xdna_image_t;

// Creates an executable image from |source_sequence| for |target|.
//
// Construction is transactional: |out_image| is populated only after every
// directory, metadata, program, and whole-image relationship has been decoded
// and validated. The source sequence and target are borrowed during the call.
// The source sequence is retained on success; the target is not retained.
iree_status_t iree_hal_amd_xdna_image_create(
    iree_byte_sequence_t* source_sequence,
    const iree_hal_amd_xdna_image_target_t* target,
    iree_allocator_t host_allocator, iree_hal_amd_xdna_image_t** out_image);

// Destroys |image| and releases its retained source byte sequence.
void iree_hal_amd_xdna_image_destroy(iree_hal_amd_xdna_image_t* image);

// Returns the complete immutable source byte length.
uint64_t iree_hal_amd_xdna_image_source_length(
    const iree_hal_amd_xdna_image_t* image);

// Returns the target-specific ELF flags qualified during construction.
uint32_t iree_hal_amd_xdna_image_target_flags(
    const iree_hal_amd_xdna_image_t* image);

// Returns the number of decoded program headers.
iree_host_size_t iree_hal_amd_xdna_image_program_header_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the program header at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_image_program_header_t*
iree_hal_amd_xdna_image_program_header(const iree_hal_amd_xdna_image_t* image,
                                       iree_host_size_t ordinal);

// Returns the decoded and target-qualified image ABI note.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_elf_abi_note_t* iree_hal_amd_xdna_image_abi_note(
    const iree_hal_amd_xdna_image_t* image);

// Returns the number of decoded executable entries.
iree_host_size_t iree_hal_amd_xdna_image_entry_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the executable entry at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_elf_entry_record_t* iree_hal_amd_xdna_image_entry(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal);

// Returns the diagnostic name of the executable entry at |ordinal|.
//
// An unnamed or out-of-range entry returns an empty string view. Name storage
// remains valid until |image| is destroyed and is not null terminated.
iree_string_view_t iree_hal_amd_xdna_image_entry_name(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal);

// Returns the number of decoded runtime bindings.
iree_host_size_t iree_hal_amd_xdna_image_binding_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the runtime binding at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_elf_binding_record_t* iree_hal_amd_xdna_image_binding(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal);

// Returns the number of decoded runtime relocations.
iree_host_size_t iree_hal_amd_xdna_image_relocation_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the runtime relocation at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_elf_relocation_record_t*
iree_hal_amd_xdna_image_relocation(const iree_hal_amd_xdna_image_t* image,
                                   iree_host_size_t ordinal);

// Returns the number of decoded ARRAY realizations.
iree_host_size_t iree_hal_amd_xdna_image_array_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the ARRAY realization at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_image_array_program_t* iree_hal_amd_xdna_image_array(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal);

// Returns the number of decoded CONTROL programs.
iree_host_size_t iree_hal_amd_xdna_image_control_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the CONTROL program at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_image_control_program_t*
iree_hal_amd_xdna_image_control(const iree_hal_amd_xdna_image_t* image,
                                iree_host_size_t ordinal);

// Returns the number of framed ARRAY and CONTROL records.
iree_host_size_t iree_hal_amd_xdna_image_record_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the flattened program record at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_image_program_record_t* iree_hal_amd_xdna_image_record(
    const iree_hal_amd_xdna_image_t* image, iree_host_size_t ordinal);

// Returns the capabilities implied by the validated image structure.
iree_hal_amd_xdna_elf_capabilities_t
iree_hal_amd_xdna_image_structural_capabilities(
    const iree_hal_amd_xdna_image_t* image);

// Returns the number of canonical TILE placements.
iree_host_size_t iree_hal_amd_xdna_image_tile_placement_count(
    const iree_hal_amd_xdna_image_t* image);

// Returns the TILE placement at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |image| is destroyed.
const iree_hal_amd_xdna_image_tile_placement_t*
iree_hal_amd_xdna_image_tile_placement(const iree_hal_amd_xdna_image_t* image,
                                       iree_host_size_t ordinal);

// Enumerates the portion of source segments intersecting |source_range|.
//
// Segment boundaries are arbitrary and callers must handle fields split across
// callbacks. Segments remain valid only for the callback duration. An error
// returned by the callback stops enumeration and is propagated unchanged.
iree_status_t iree_hal_amd_xdna_image_enumerate_source_range(
    const iree_hal_amd_xdna_image_t* image,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_sequence_segment_callback_t callback);

// Copies exactly |source_range.length| bytes into |storage|.
//
// Storage must have exactly the requested length. This is intended for bounded
// headers and tables; large payload consumers should enumerate instead.
iree_status_t iree_hal_amd_xdna_image_read_source_range(
    const iree_hal_amd_xdna_image_t* image,
    iree_hal_amd_xdna_image_source_range_t source_range,
    iree_byte_span_t storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_IMAGE_H_
