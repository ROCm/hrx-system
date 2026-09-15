// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P executable image qualification against immutable target facts.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_TARGET_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_TARGET_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"
#include "iree/hal/drivers/amd/xdna/image/validation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates one decoded configuration-register operation against a target.
//
// The callback receives only REGISTER_WRITE32, REGISTER_MASK_WRITE32, or
// REGISTER_BLOCK_WRITE32 records. It must reject addresses, register ranges,
// masks, or values that are not legal for |program_type| on the selected
// target. Any borrowed block-write word storage remains valid only for the
// call.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amd_xdna_aie2p_configuration_register_validate_fn_t)(
    const void* user_data, iree_hal_amd_xdna_elf_program_type_t program_type,
    const iree_hal_amd_xdna_aie2p_program_record_t* record);

// Immutable configuration-register validator supplied by the target corpus.
typedef struct iree_hal_amd_xdna_aie2p_configuration_register_validator_t {
  // Function validating one decoded register operation.
  iree_hal_amd_xdna_aie2p_configuration_register_validate_fn_t fn;
  // Unowned immutable target facts passed to |fn|.
  const void* user_data;
} iree_hal_amd_xdna_aie2p_configuration_register_validator_t;

// Validates one context-bounded DMA task wait against a target.
//
// Generic qualification proves the complete tile rectangle is within the
// logical context before invocation. The callback must reject tile roles,
// directions, channels, or rectangle shapes that the selected target cannot
// service with firmware task-completion tokens.
typedef iree_status_t(
    IREE_API_PTR* iree_hal_amd_xdna_aie2p_dma_task_wait_validate_fn_t)(
    const void* user_data, const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait);

// Immutable DMA task-wait validator supplied by the target corpus.
typedef struct iree_hal_amd_xdna_aie2p_dma_task_wait_validator_t {
  // Function validating one decoded context-bounded DMA task wait.
  iree_hal_amd_xdna_aie2p_dma_task_wait_validate_fn_t fn;
  // Unowned immutable target facts passed to |fn|.
  const void* user_data;
} iree_hal_amd_xdna_aie2p_dma_task_wait_validator_t;

// Borrowed AIE2P target facts used while qualifying one executable image.
//
// The descriptor is immutable and may be shared across concurrent image
// creation calls. The logical context geometry belongs to the image being
// qualified; target callbacks interpret all addresses and coordinates in that
// context-relative space.
typedef struct iree_hal_amd_xdna_aie2p_target_t {
  // Exact identities used to form executable images for this target.
  struct {
    // Incompatible revision of the resolved device profile.
    uint32_t device_profile_revision;
    // Stable complete device-profile identity.
    uint64_t device_profile_id;
    // Stable firmware and configuration ABI identity.
    uint64_t firmware_abi_id;
    // Stable placement and image-formation policy identity.
    uint64_t policy_id;
  } identity;
  // Loader capabilities supported for this target.
  iree_hal_amd_xdna_elf_capabilities_t supported_capabilities;
  // Logical context-relative geometry required by the image.
  struct {
    // Number of addressable context-relative columns.
    uint16_t column_count;
    // Number of addressable context-relative rows.
    uint16_t row_count;
  } context;
  // Target-native facts required to lower qualified image programs.
  struct {
    // Transaction 0.1 header fields fixed by the AIE generation.
    struct {
      // AIE-RT device-generation value.
      uint8_t device_generation;
      // Number of memory-tile rows in the target generation.
      uint8_t memory_tile_row_count;
    } transaction;
    // Bit layout of context-relative AIE register addresses.
    struct {
      // Bit position of the tile column.
      uint8_t column_shift;
      // Bit position of the tile row.
      uint8_t row_shift;
    } register_address;
    // Host-visible programming aperture for tile program memory.
    struct {
      // Tile-relative byte offset of the program-memory aperture.
      uint32_t host_offset;
    } program_memory;
  } native;
  // Resolver for canonical TILE memory placements.
  iree_hal_amd_xdna_image_tile_memory_resolver_t tile_memory_resolver;
  // Validator for target configuration-register operations.
  iree_hal_amd_xdna_aie2p_configuration_register_validator_t
      configuration_register_validator;
  // Validator for target DMA task waits.
  iree_hal_amd_xdna_aie2p_dma_task_wait_validator_t dma_task_wait_validator;
} iree_hal_amd_xdna_aie2p_target_t;

// Validates that |target| contains a complete AIE2P qualification contract.
iree_status_t iree_hal_amd_xdna_aie2p_target_validate(
    const iree_hal_amd_xdna_aie2p_target_t* target);

// Validates that |abi_note| exactly identifies |target|.
iree_status_t iree_hal_amd_xdna_aie2p_target_validate_abi_note(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_hal_amd_xdna_elf_abi_note_t* abi_note);

// Initializes a generic image target borrowing |target|.
//
// |target| must remain live and immutable while the returned image target is in
// use. The composed target qualifies the exact ABI identity, resolves TILE
// placements, decodes every AIE2P record, and restricts runtime relocations to
// mutable target-defined fields.
iree_status_t iree_hal_amd_xdna_aie2p_target_initialize_image_target(
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_hal_amd_xdna_image_target_t* out_image_target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_TARGET_H_
