// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable generic metadata tables decoded from an AMD XDNA image.
//
// This layer validates table-local wire contracts and relationships between
// executable entries, runtime bindings, and relocations. It intentionally does
// not interpret target programs or qualify the image against a device.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TABLES_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TABLES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/directory.h"
#include "iree/hal/drivers/amd/xdna/image/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable decoded generic metadata independent of its source directory.
typedef struct iree_hal_amd_xdna_image_tables_t
    iree_hal_amd_xdna_image_tables_t;

// Decodes and validates generic metadata from |directory|.
//
// The directory is borrowed only for the duration of the call. On success the
// returned object owns compact decoded records and diagnostic entry names and
// can outlive both the directory and its source byte sequence. ARRAY, CONTROL,
// TILE, and target-specific payload semantics are validated by later layers.
iree_status_t iree_hal_amd_xdna_image_tables_create(
    const iree_hal_amd_xdna_image_directory_t* directory,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_tables_t** out_tables);

// Destroys |tables|.
void iree_hal_amd_xdna_image_tables_destroy(
    iree_hal_amd_xdna_image_tables_t* tables);

// Returns the decoded image ABI note.
//
// The returned pointer remains valid until |tables| is destroyed.
const iree_hal_amd_xdna_elf_abi_note_t* iree_hal_amd_xdna_image_tables_abi_note(
    const iree_hal_amd_xdna_image_tables_t* tables);

// Returns the number of decoded executable entries.
iree_host_size_t iree_hal_amd_xdna_image_tables_entry_count(
    const iree_hal_amd_xdna_image_tables_t* tables);

// Returns the executable entry at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |tables| is destroyed.
const iree_hal_amd_xdna_elf_entry_record_t*
iree_hal_amd_xdna_image_tables_entry(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal);

// Returns the diagnostic name of the executable entry at |ordinal|.
//
// An unnamed or out-of-range entry returns an empty string view. Name storage
// remains valid until |tables| is destroyed and is not null terminated.
iree_string_view_t iree_hal_amd_xdna_image_tables_entry_name(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal);

// Returns the number of decoded runtime bindings.
iree_host_size_t iree_hal_amd_xdna_image_tables_binding_count(
    const iree_hal_amd_xdna_image_tables_t* tables);

// Returns the runtime binding at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |tables| is destroyed.
const iree_hal_amd_xdna_elf_binding_record_t*
iree_hal_amd_xdna_image_tables_binding(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal);

// Returns the number of decoded runtime relocations.
iree_host_size_t iree_hal_amd_xdna_image_tables_relocation_count(
    const iree_hal_amd_xdna_image_tables_t* tables);

// Returns the runtime relocation at |ordinal|, or NULL when out of range.
//
// The returned pointer remains valid until |tables| is destroyed.
const iree_hal_amd_xdna_elf_relocation_record_t*
iree_hal_amd_xdna_image_tables_relocation(
    const iree_hal_amd_xdna_image_tables_t* tables, iree_host_size_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TABLES_H_
