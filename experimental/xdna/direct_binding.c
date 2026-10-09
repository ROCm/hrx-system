// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/direct_binding.h"

static iree_hal_memory_access_t iree_xdna_required_binding_access(
    iree_xdna_elf_binding_access_t access) {
  iree_hal_memory_access_t required_access = IREE_HAL_MEMORY_ACCESS_NONE;
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_READ)) {
    required_access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_WRITE)) {
    required_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  }
  return required_access;
}

static iree_hal_buffer_usage_t iree_xdna_required_binding_usage(
    iree_xdna_elf_binding_access_t access) {
  iree_hal_buffer_usage_t required_usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_READ)) {
    required_usage |= IREE_HAL_BUFFER_USAGE_STORAGE_READ;
  }
  if (iree_any_bit_set(access, IREE_XDNA_ELF_BINDING_ACCESS_WRITE)) {
    required_usage |= IREE_HAL_BUFFER_USAGE_STORAGE_WRITE;
  }
  return required_usage;
}

static iree_hal_memory_type_t iree_xdna_required_binding_memory_type(
    const iree_xdna_elf_binding_record_t* contract) {
  iree_hal_memory_type_t required_type = IREE_HAL_MEMORY_TYPE_NONE;
  if (iree_any_bit_set(contract->usage,
                       IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE)) {
    required_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }
  const bool requires_host_visibility =
      contract->address_space == IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST ||
      iree_any_bit_set(contract->usage,
                       IREE_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE);
  if (requires_host_visibility) {
    required_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    if (iree_any_bit_set(contract->usage,
                         IREE_XDNA_ELF_BINDING_USAGE_COHERENT)) {
      required_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
    }
    if (iree_any_bit_set(contract->usage, IREE_XDNA_ELF_BINDING_USAGE_CACHED)) {
      required_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    }
  }
  return required_type;
}

static iree_status_t iree_xdna_validate_binding(
    uint32_t ordinal, const iree_xdna_elf_binding_record_t* contract,
    const iree_hal_amd_xdna_executable_binding_t* binding) {
  if (binding->buffer_ref.reserved || binding->buffer_ref.buffer_slot ||
      !binding->buffer_ref.buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA binding %u is not resolved", ordinal);
  }
  iree_device_size_t byte_offset = 0;
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0,
      iree_hal_buffer_byte_length(binding->buffer_ref.buffer),
      binding->buffer_ref.offset, binding->buffer_ref.length, &byte_offset,
      &byte_length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding->buffer_ref.buffer),
      iree_xdna_required_binding_access(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(binding->buffer_ref.buffer),
      iree_xdna_required_binding_usage(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer_ref.buffer),
      iree_xdna_required_binding_memory_type(contract)));
  if (byte_offset < contract->minimum_byte_offset ||
      byte_offset > contract->maximum_byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u offset %" PRIu64
                            " is outside [%" PRIu64 ", %" PRIu64 "]",
                            ordinal, (uint64_t)byte_offset,
                            contract->minimum_byte_offset,
                            contract->maximum_byte_offset);
  }
  if (byte_length < contract->minimum_byte_length ||
      binding->byte_length != byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u length %" PRIu64
                            " does not satisfy the direct binding contract",
                            ordinal, (uint64_t)byte_length);
  }
  if ((binding->device_address & (contract->minimum_alignment - 1)) != 0 ||
      (byte_length &&
       binding->device_address > UINT64_MAX - (byte_length - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u native range is invalid", ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_xdna_validate_storage(
    const iree_xdna_elf_allocation_record_t* allocation,
    const iree_hal_amd_xdna_executable_storage_t* storage) {
  if (!storage->memory || !storage->mapping.data ||
      storage->mapping.data_length < allocation->byte_length ||
      storage->memory_byte_offset > UINT64_MAX - allocation->byte_length ||
      storage->device_address > UINT64_MAX - allocation->byte_length ||
      (storage->device_address & (allocation->alignment - 1)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA allocation backing has an invalid range or alignment");
  }
  return iree_ok_status();
}

iree_status_t iree_xdna_executable_storage_bind(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    iree_host_size_t binding_count,
    const iree_hal_amd_xdna_executable_binding_t* bindings) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image);
  if (entry_ordinal >= tables->header.entry_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA entry ordinal %u is out of range",
                            entry_ordinal);
  }
  const iree_xdna_elf_entry_record_t entry =
      iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal);
  if (!storage || storage_count != entry.allocation_use_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA entry requires %u backing allocations",
                            entry.allocation_use_count);
  }
  if (binding_count != entry.binding_count || (binding_count && !bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA entry requires %u bindings",
                            entry.binding_count);
  }
  for (uint32_t i = 0; i < entry.allocation_use_count; ++i) {
    const uint32_t ordinal = iree_hal_amd_xdna_image_tables_allocation_use(
        tables, entry.first_allocation_use + i);
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_tables_allocation(tables, ordinal);
    IREE_RETURN_IF_ERROR(iree_xdna_validate_storage(&allocation, &storage[i]));
  }
  for (uint32_t i = 0; i < binding_count; ++i) {
    const iree_xdna_elf_binding_record_t contract =
        iree_hal_amd_xdna_image_tables_binding(tables, entry.first_binding + i);
    if (contract.kind != IREE_XDNA_ELF_BINDING_KIND_NONE) {
      IREE_RETURN_IF_ERROR(
          iree_xdna_validate_binding(i, &contract, &bindings[i]));
    }
  }
  for (uint32_t i = 0; i < entry.dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry.first_dynamic_relocation + i);
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_xdna_executable_storage_validate_relocation(
            &relocation, bindings[relocation.source_ordinal].device_address));
  }
  iree_hal_amd_xdna_executable_storage_patch(image, entry_ordinal, storage,
                                             bindings);
  return iree_ok_status();
}
