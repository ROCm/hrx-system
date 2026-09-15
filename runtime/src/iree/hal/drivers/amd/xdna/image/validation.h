// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-image qualification for canonical AMD XDNA executable images.
//
// This layer proves relationships among an already decoded directory, metadata
// tables, and program framing. Target-specific address and record semantics are
// supplied through a borrowed immutable target descriptor.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_VALIDATION_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_VALIDATION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/directory.h"
#include "iree/hal/drivers/amd/xdna/image/programs.h"
#include "iree/hal/drivers/amd/xdna/image/tables.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Canonical physical storage selected for one TILE load action.
typedef struct iree_hal_amd_xdna_image_tile_placement_t {
  // Program-header ordinal defining the load action.
  uint32_t program_header_ordinal;
  // Context-relative column owning the selected storage.
  uint16_t owner_column;
  // Context-relative row owning the selected storage.
  uint16_t owner_row;
  // Canonical storage class within the owner tile.
  iree_hal_amd_xdna_elf_tile_memory_space_t memory_space;
  // Byte offset in the owner's selected storage class.
  uint32_t owner_offset;
  // Number of bytes in the exact placement.
  uint32_t byte_length;
  // Bytes available from |owner_offset| through the selected aperture.
  uint32_t available_capacity;
} iree_hal_amd_xdna_image_tile_placement_t;

// Resolves one exact TILE load range to its canonical physical storage.
//
// Different accessor windows that address the same storage must return the
// same owner coordinate, memory space, and owner-relative offset. Program
// memory must resolve to the accessor tile at owner offset zero. The callback
// must reject coordinates, tile classes, memory spaces, and ranges not
// implemented by the selected target.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amd_xdna_image_tile_memory_resolve_fn_t)(
    const void* user_data,
    const iree_hal_amd_xdna_elf_tile_destination_t* destination,
    uint32_t virtual_address, uint32_t byte_length,
    iree_hal_amd_xdna_image_tile_placement_t* out_placement);

// Immutable tile-memory resolver borrowed during image validation.
typedef struct iree_hal_amd_xdna_image_tile_memory_resolver_t {
  // Function resolving one TILE load action.
  iree_hal_amd_xdna_image_tile_memory_resolve_fn_t fn;
  // Unowned target facts passed to |fn|.
  const void* user_data;
} iree_hal_amd_xdna_image_tile_memory_resolver_t;

// Validates the exact ABI identity and logical geometry against a target.
//
// Generic decoding has already proven the note framing, ABI version,
// capability vocabulary, zero origin, and context-relative coordinate model.
// The callback must enforce target generation, profile revision and identity,
// firmware ABI, formation policy, and context dimensions.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amd_xdna_image_abi_note_validate_fn_t)(
    const void* user_data, const iree_hal_amd_xdna_elf_abi_note_t* abi_note);

// Immutable ABI-note validator borrowed during image validation.
typedef struct iree_hal_amd_xdna_image_abi_note_validator_t {
  // Function validating one locally decoded ABI note.
  iree_hal_amd_xdna_image_abi_note_validate_fn_t fn;
  // Unowned target facts passed to |fn|.
  const void* user_data;
} iree_hal_amd_xdna_image_abi_note_validator_t;

// Validates one runtime relocation against a target-specific record field.
//
// Generic validation proves that |record_relative_byte_offset| and the field
// width name one complete byte range within the record payload rather than its
// common framing. The callback must reject fields that the target record ABI
// does not permit runtime relocation to mutate.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amd_xdna_image_program_relocation_validate_fn_t)(
    const void* user_data, iree_hal_amd_xdna_elf_program_type_t program_type,
    const iree_hal_amd_xdna_image_program_record_t* record,
    uint32_t record_relative_byte_offset,
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation);

// Immutable program-relocation validator borrowed during image validation.
typedef struct iree_hal_amd_xdna_image_program_relocation_validator_t {
  // Function validating one target-specific relocation field.
  iree_hal_amd_xdna_image_program_relocation_validate_fn_t fn;
  // Unowned target facts passed to |fn|.
  const void* user_data;
} iree_hal_amd_xdna_image_program_relocation_validator_t;

// Complete immutable target contract borrowed during image construction.
//
// The descriptor contains no live device or provider object and may be shared
// across concurrent image creation calls. All callback state must remain live
// and immutable for the duration of image construction.
typedef struct iree_hal_amd_xdna_image_target_t {
  // Exact processor-specific ELF flags accepted by the target.
  uint32_t target_flags;
  // Loader capabilities supported for this target.
  iree_hal_amd_xdna_elf_capabilities_t supported_capabilities;
  // Validator for the exact ABI identity and logical geometry.
  iree_hal_amd_xdna_image_abi_note_validator_t abi_note_validator;
  // Resolver for canonical TILE memory placements.
  iree_hal_amd_xdna_image_tile_memory_resolver_t tile_memory_resolver;
  // Validator for target-specific ARRAY and CONTROL records.
  iree_hal_amd_xdna_image_program_record_validator_t program_record_validator;
  // Validator for mutable fields in target-specific program records.
  iree_hal_amd_xdna_image_program_relocation_validator_t
      program_relocation_validator;
} iree_hal_amd_xdna_image_target_t;

// Validates that |target| contains a complete image qualification contract.
iree_status_t iree_hal_amd_xdna_image_target_validate(
    const iree_hal_amd_xdna_image_target_t* target);

// Immutable whole-image qualification results.
typedef struct iree_hal_amd_xdna_image_validation_t
    iree_hal_amd_xdna_image_validation_t;

// Validates all relationships among decoded image components.
//
// All component objects and |target| are borrowed only for the duration of the
// call. On success the returned object owns compact canonical tile placements
// and can outlive every input object.
iree_status_t iree_hal_amd_xdna_image_validation_create(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_programs_t* programs,
    const iree_hal_amd_xdna_image_target_t* target,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_validation_t** out_validation);

// Destroys |validation|.
void iree_hal_amd_xdna_image_validation_destroy(
    iree_hal_amd_xdna_image_validation_t* validation);

// Returns the capabilities implied by the validated image structure.
iree_hal_amd_xdna_elf_capabilities_t
iree_hal_amd_xdna_image_validation_structural_capabilities(
    const iree_hal_amd_xdna_image_validation_t* validation);

// Returns the number of canonical TILE placements.
iree_host_size_t iree_hal_amd_xdna_image_validation_tile_placement_count(
    const iree_hal_amd_xdna_image_validation_t* validation);

// Returns the TILE placement at |ordinal|, or NULL when out of range.
const iree_hal_amd_xdna_image_tile_placement_t*
iree_hal_amd_xdna_image_validation_tile_placement(
    const iree_hal_amd_xdna_image_validation_t* validation,
    iree_host_size_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_VALIDATION_H_
