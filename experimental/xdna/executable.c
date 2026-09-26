// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/executable.h"

#include <string.h>

static iree_hal_memory_access_t iree_hal_amd_xdna_executable_required_access(
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

static iree_hal_buffer_usage_t iree_hal_amd_xdna_executable_required_usage(
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

static iree_hal_memory_type_t iree_hal_amd_xdna_executable_required_memory_type(
    const iree_xdna_elf_binding_record_t* contract) {
  iree_hal_memory_type_t required_memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  if (iree_any_bit_set(contract->usage,
                       IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE)) {
    required_memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }
  const bool requires_host_visibility =
      contract->address_space == IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST ||
      iree_any_bit_set(contract->usage,
                       IREE_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE);
  if (requires_host_visibility) {
    required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    if (iree_any_bit_set(contract->usage,
                         IREE_XDNA_ELF_BINDING_USAGE_COHERENT)) {
      required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
    }
    if (iree_any_bit_set(contract->usage, IREE_XDNA_ELF_BINDING_USAGE_CACHED)) {
      required_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    }
  }
  return required_memory_type;
}

static iree_status_t iree_hal_amd_xdna_executable_validate_binding(
    uint32_t binding_ordinal, const iree_xdna_elf_binding_record_t* contract,
    const iree_hal_amd_xdna_executable_binding_t* binding) {
  if (binding->buffer_ref.buffer == NULL || binding->memory == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA binding %u is not fully resolved",
                            binding_ordinal);
  }

  iree_device_size_t resource_byte_offset = 0;
  iree_device_size_t resource_byte_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0,
      iree_hal_buffer_byte_length(binding->buffer_ref.buffer),
      binding->buffer_ref.offset, binding->buffer_ref.length,
      &resource_byte_offset, &resource_byte_length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding->buffer_ref.buffer),
      iree_hal_amd_xdna_executable_required_access(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(binding->buffer_ref.buffer),
      iree_hal_amd_xdna_executable_required_usage(contract->access)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer_ref.buffer),
      iree_hal_amd_xdna_executable_required_memory_type(contract)));

  if (resource_byte_offset < contract->minimum_byte_offset ||
      resource_byte_offset > contract->maximum_byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u offset %" PRIu64
                            " is outside [%" PRIu64 ", %" PRIu64 "]",
                            binding_ordinal, (uint64_t)resource_byte_offset,
                            contract->minimum_byte_offset,
                            contract->maximum_byte_offset);
  }
  if (resource_byte_length < contract->minimum_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u length %" PRIu64
                            " is smaller than required %" PRIu64,
                            binding_ordinal, (uint64_t)resource_byte_length,
                            contract->minimum_byte_length);
  }
  if ((binding->device_address & (contract->minimum_alignment - 1)) != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u device address 0x%" PRIx64
                            " does not satisfy %" PRIu64 "-byte alignment",
                            binding_ordinal, binding->device_address,
                            contract->minimum_alignment);
  }
  if ((resource_byte_length != 0 &&
       binding->device_address > UINT64_MAX - (resource_byte_length - 1)) ||
      binding->memory_byte_offset > UINT64_MAX - resource_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding %u native range overflows",
                            binding_ordinal);
  }
  return iree_ok_status();
}

// Qualification has established the field and its binding. Instantiation checks
// only the externally supplied address against the preserved ELF constraints.
static iree_status_t iree_hal_amd_xdna_executable_validate_relocation(
    const iree_xdna_elf_relocation_record_t* relocation,
    uint64_t base_address) {
  const uint64_t displacement = relocation->addend < 0
                                    ? UINT64_C(0) - (uint64_t)relocation->addend
                                    : (uint64_t)relocation->addend;
  if ((relocation->addend < 0 && base_address < displacement) ||
      (relocation->addend >= 0 && base_address > UINT64_MAX - displacement)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA relocated DMA address overflows");
  }
  const uint64_t address = base_address + (uint64_t)relocation->addend;
  if (address < relocation->minimum_value ||
      address > relocation->maximum_value ||
      (address & (relocation->alignment - 1)) != 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA DMA address violates the relocation contract");
  }
  return iree_ok_status();
}

static void iree_hal_amd_xdna_executable_write_address(uint8_t* target,
                                                       uint64_t address) {
  const uint32_t low = iree_unaligned_load_le_u32(target);
  const uint32_t high = iree_unaligned_load_le_u32(target + 4);
  iree_unaligned_store_le_u32(target, (low & 3u) | (uint32_t)address);
  iree_unaligned_store_le_u32(
      target + 4, (high & UINT32_C(0xFFFF0000)) | (uint32_t)(address >> 32));
}

static iree_status_t iree_hal_amd_xdna_executable_entry(
    const iree_hal_amd_xdna_image_t* image, uint32_t ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    iree_xdna_elf_entry_record_t* out_entry) {
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image);
  if (ordinal >= tables->header.entry_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA entry ordinal %u is out of range", ordinal);
  }
  *out_entry = iree_hal_amd_xdna_image_tables_entry(tables, ordinal);
  if (storage == NULL || storage_count != out_entry->allocation_use_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA entry requires %u backing allocations",
                            out_entry->allocation_use_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_validate_storage(
    const iree_xdna_elf_allocation_record_t* allocation,
    const iree_hal_amd_xdna_executable_storage_t* storage) {
  if (storage->memory == NULL || storage->mapping.data == NULL ||
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

static iree_status_t iree_hal_amd_xdna_executable_validate_allocations(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_xdna_elf_entry_record_t* entry,
    const iree_hal_amd_xdna_executable_storage_t* storage) {
  for (uint32_t i = 0; i < entry->allocation_use_count; ++i) {
    const uint32_t ordinal = iree_hal_amd_xdna_image_tables_allocation_use(
        tables, entry->first_allocation_use + i);
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_tables_allocation(tables, ordinal);
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_validate_storage(
        &allocation, &storage[i]));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_load(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage) {
  iree_xdna_elf_entry_record_t entry;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_entry(
      image, entry_ordinal, storage_count, storage, &entry));
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_validate_allocations(
      tables, &entry, storage));
  for (uint32_t i = 0; i < entry.static_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry.first_static_relocation + i);
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_validate_relocation(
        &relocation, storage[relocation.source_ordinal].device_address));
  }
  const iree_hal_amd_xdna_image_directory_t* directory =
      iree_hal_amd_xdna_image_directory(image);
  for (uint32_t i = 0; i < entry.allocation_use_count; ++i) {
    const uint32_t ordinal = iree_hal_amd_xdna_image_tables_allocation_use(
        tables, entry.first_allocation_use + i);
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_tables_allocation(tables, ordinal);
    for (uint32_t j = 0; j < allocation.load_count; ++j) {
      const iree_hal_amd_xdna_image_program_header_t* load =
          iree_hal_amd_xdna_image_directory_program_header(
              directory, allocation.first_load + j);
      uint8_t* destination = storage[i].mapping.data + load->virtual_address;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_directory_read_source_range(
          directory, load->file_range,
          iree_make_byte_span(destination,
                              (iree_host_size_t)load->file_range.length)));
      memset(destination + load->file_range.length, 0,
             load->memory_size - (iree_host_size_t)load->file_range.length);
    }
  }
  for (uint32_t i = 0; i < entry.static_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry.first_static_relocation + i);
    iree_hal_amd_xdna_executable_write_address(
        storage[relocation.destination_use].mapping.data +
            relocation.byte_offset,
        storage[relocation.source_ordinal].device_address +
            (uint64_t)relocation.addend);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_bind(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    iree_host_size_t binding_count,
    const iree_hal_amd_xdna_executable_binding_t* bindings) {
  iree_xdna_elf_entry_record_t entry;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_entry(
      image, entry_ordinal, storage_count, storage, &entry));
  if (binding_count != entry.binding_count ||
      (binding_count != 0 && bindings == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA entry requires %u bindings",
                            entry.binding_count);
  }
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_validate_allocations(
      tables, &entry, storage));
  for (uint32_t i = 0; i < binding_count; ++i) {
    const iree_xdna_elf_binding_record_t contract =
        iree_hal_amd_xdna_image_tables_binding(tables, entry.first_binding + i);
    if (contract.kind == IREE_XDNA_ELF_BINDING_KIND_NONE) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_validate_binding(
        i, &contract, &bindings[i]));
  }
  for (uint32_t i = 0; i < entry.dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry.first_dynamic_relocation + i);
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_validate_relocation(
        &relocation, bindings[relocation.source_ordinal].device_address));
  }
  for (uint32_t i = 0; i < entry.dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry.first_dynamic_relocation + i);
    iree_hal_amd_xdna_executable_write_address(
        storage[relocation.destination_use].mapping.data +
            relocation.byte_offset,
        bindings[relocation.source_ordinal].device_address +
            (uint64_t)relocation.addend);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_query_invocation(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    amdf_xdna_kernel_command_t* out_command) {
  iree_xdna_elf_entry_record_t entry;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_entry(
      image, entry_ordinal, storage_count, storage, &entry));
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image);
  const iree_xdna_elf_invocation_record_t invocation =
      iree_hal_amd_xdna_image_tables_invocation(tables, entry.first_invocation);
  const uint32_t ordinal = iree_hal_amd_xdna_image_tables_allocation_use(
      tables, entry.first_allocation_use + invocation.allocation_use);
  const iree_xdna_elf_allocation_record_t allocation =
      iree_hal_amd_xdna_image_tables_allocation(tables, ordinal);
  const iree_hal_amd_xdna_executable_storage_t* backing =
      &storage[invocation.allocation_use];
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_executable_validate_storage(&allocation, backing));
  *out_command = (amdf_xdna_kernel_command_t){
      .memory = backing->memory,
      .access_ordinal = backing->access_ordinal,
      .byte_offset = backing->memory_byte_offset + invocation.byte_offset,
      .byte_length = invocation.byte_length,
  };
  return iree_ok_status();
}
