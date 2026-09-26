// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/validation.h"

static bool iree_hal_amd_xdna_image_range_fits(uint64_t offset, uint64_t length,
                                               uint64_t capacity) {
  return offset <= capacity && length <= capacity - offset;
}

// Finds initialized coverage in an already validated, destination-ordered load
// range. Binary search bounds admission work when many fixups use one catalog.
static bool iree_hal_amd_xdna_image_range_initialized(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_xdna_elf_allocation_record_t* allocation, uint64_t offset,
    uint64_t length) {
  uint32_t first = allocation->first_load;
  uint32_t end = first + allocation->load_count;
  while (first < end) {
    uint32_t middle = first + (end - first) / 2;
    const iree_hal_amd_xdna_image_program_header_t* load =
        iree_hal_amd_xdna_image_directory_program_header(directory, middle);
    if ((uint64_t)load->virtual_address + load->memory_size <= offset) {
      first = middle + 1;
    } else {
      end = middle;
    }
  }
  uint64_t covered = offset;
  const uint64_t range_end = offset + length;
  const uint32_t load_end = allocation->first_load + allocation->load_count;
  for (uint32_t i = first; i < load_end && covered < range_end; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* load =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    if (load->virtual_address > covered) {
      return false;
    }
    covered = (uint64_t)load->virtual_address + load->memory_size;
  }
  return covered >= range_end;
}

static iree_status_t iree_hal_amd_xdna_image_validate_allocations(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_directory_t* directory) {
  const iree_host_size_t header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  uint32_t next_load = 1;
  for (uint32_t i = 0; i < tables->header.allocation_count; ++i) {
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_tables_allocation(tables, i);
    if ((allocation.domain != IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND &&
         allocation.domain != IREE_XDNA_ELF_ALLOCATION_DOMAIN_DMA) ||
        allocation.flags > IREE_XDNA_ELF_ALLOCATION_FLAG_DEVICE_WRITE ||
        allocation.byte_length == 0 ||
        allocation.byte_length > IREE_HOST_SIZE_MAX ||
        !iree_is_power_of_two_uint64(allocation.alignment) ||
        allocation.first_load != next_load ||
        !iree_hal_amd_xdna_image_range_fits(next_load, allocation.load_count,
                                            header_count)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA allocation %u", i);
    }
    uint64_t previous_end = 0;
    for (uint32_t j = 0; j < allocation.load_count; ++j) {
      const iree_hal_amd_xdna_image_program_header_t* load =
          iree_hal_amd_xdna_image_directory_program_header(directory,
                                                           next_load + j);
      if (load->type != IREE_XDNA_ELF_PROGRAM_TYPE_LOAD ||
          load->flags != IREE_XDNA_ELF_PROGRAM_FLAG_READ ||
          load->physical_address != i || load->memory_size == 0 ||
          load->virtual_address < previous_end ||
          !iree_hal_amd_xdna_image_range_fits(load->virtual_address,
                                              load->memory_size,
                                              allocation.byte_length)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid XDNA load %u", next_load + j);
      }
      previous_end = (uint64_t)load->virtual_address + load->memory_size;
    }
    next_load += allocation.load_count;
  }
  if (next_load != header_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA directory has unowned load headers");
  }
  return iree_ok_status();
}

static iree_xdna_elf_allocation_record_t
iree_hal_amd_xdna_image_entry_allocation(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_xdna_elf_entry_record_t* entry, uint32_t use) {
  return iree_hal_amd_xdna_image_tables_allocation(
      tables, iree_hal_amd_xdna_image_tables_allocation_use(
                  tables, entry->first_allocation_use + use));
}

static iree_status_t iree_hal_amd_xdna_image_validate_relocations(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_xdna_elf_entry_record_t* entry, uint32_t first_relocation,
    uint32_t relocation_count, uint32_t source_count) {
  uint32_t previous_use = 0;
  uint64_t previous_end = 0;
  for (uint32_t i = 0; i < relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(tables, first_relocation + i);
    if (relocation.destination_use >= entry->allocation_use_count ||
        relocation.source_ordinal >= source_count ||
        relocation.kind != IREE_XDNA_ELF_RELOCATION_KIND_SHIM_ADDRESS ||
        relocation.byte_offset % 4 != 0 ||
        relocation.minimum_value > relocation.maximum_value ||
        relocation.maximum_value > UINT64_C(0xFFFFFFFFFFFF) ||
        relocation.alignment < 4 ||
        !iree_is_power_of_two_uint64(relocation.alignment) ||
        relocation.destination_use < previous_use ||
        (relocation.destination_use == previous_use &&
         relocation.byte_offset < previous_end)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA address relocation %u",
                              first_relocation + i);
    }
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_entry_allocation(tables, entry,
                                                 relocation.destination_use);
    if (!iree_hal_amd_xdna_image_range_fits(relocation.byte_offset, 8,
                                            allocation.byte_length) ||
        !iree_hal_amd_xdna_image_range_initialized(directory, &allocation,
                                                   relocation.byte_offset, 8)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA relocation targets uninitialized storage");
    }
    previous_use = relocation.destination_use;
    previous_end = (uint64_t)relocation.byte_offset + 8;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_validate_entry(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    const iree_xdna_elf_entry_record_t* entry) {
  uint32_t previous_allocation = 0;
  for (uint32_t i = 0; i < entry->allocation_use_count; ++i) {
    const uint32_t allocation = iree_hal_amd_xdna_image_tables_allocation_use(
        tables, entry->first_allocation_use + i);
    if (allocation >= tables->header.allocation_count ||
        (i != 0 && allocation <= previous_allocation)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA allocation uses must be unique and ordered");
    }
    previous_allocation = allocation;
  }
  for (uint32_t i = 0; i < entry->binding_count; ++i) {
    const iree_xdna_elf_binding_record_t binding =
        iree_hal_amd_xdna_image_tables_binding(tables,
                                               entry->first_binding + i);
    if (binding.kind == IREE_XDNA_ELF_BINDING_KIND_NONE) {
      if (binding.address_space != IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_NONE ||
          binding.access != 0 || binding.usage != 0 ||
          binding.minimum_byte_length != 0 || binding.minimum_alignment != 0 ||
          binding.minimum_byte_offset != 0 ||
          binding.maximum_byte_offset != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid unused XDNA binding %u", i);
      }
      continue;
    }
    if (binding.kind != IREE_XDNA_ELF_BINDING_KIND_BUFFER ||
        (binding.address_space != IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL &&
         binding.address_space != IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST) ||
        binding.access == 0 ||
        (binding.access & ~IREE_XDNA_ELF_KNOWN_BINDING_ACCESS) != 0 ||
        (binding.usage & ~IREE_XDNA_ELF_KNOWN_BINDING_USAGE) != 0 ||
        binding.minimum_byte_offset > binding.maximum_byte_offset ||
        !iree_is_power_of_two_uint64(binding.minimum_alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA binding %u", i);
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_relocations(
      tables, directory, entry, entry->first_static_relocation,
      entry->static_relocation_count, entry->allocation_use_count));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_relocations(
      tables, directory, entry, entry->first_dynamic_relocation,
      entry->dynamic_relocation_count, entry->binding_count));
  for (uint32_t i = 0; i < entry->static_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry->first_static_relocation + i);
    const iree_xdna_elf_allocation_record_t source =
        iree_hal_amd_xdna_image_entry_allocation(tables, entry,
                                                 relocation.source_ordinal);
    if (source.domain != IREE_XDNA_ELF_ALLOCATION_DOMAIN_DMA) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "shim relocation requires a DMA-domain source");
    }
  }
  uint32_t static_index = 0;
  for (uint32_t i = 0; i < entry->dynamic_relocation_count; ++i) {
    const iree_xdna_elf_relocation_record_t dynamic_relocation =
        iree_hal_amd_xdna_image_tables_relocation(
            tables, entry->first_dynamic_relocation + i);
    const iree_xdna_elf_binding_record_t source_binding =
        iree_hal_amd_xdna_image_tables_binding(
            tables, entry->first_binding + dynamic_relocation.source_ordinal);
    if (source_binding.kind == IREE_XDNA_ELF_BINDING_KIND_NONE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA dynamic relocation sources unused binding %u",
          dynamic_relocation.source_ordinal);
    }
    const iree_xdna_elf_allocation_record_t destination =
        iree_hal_amd_xdna_image_entry_allocation(
            tables, entry, dynamic_relocation.destination_use);
    if (iree_any_bit_set(destination.flags,
                         IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dynamic relocation targets immutable backing");
    }
    // Both slices are sorted; check disjoint ownership in one merge traversal.
    while (static_index < entry->static_relocation_count) {
      const iree_xdna_elf_relocation_record_t static_relocation =
          iree_hal_amd_xdna_image_tables_relocation(
              tables, entry->first_static_relocation + static_index);
      if (static_relocation.destination_use >
          dynamic_relocation.destination_use) {
        break;
      }
      if (static_relocation.destination_use ==
          dynamic_relocation.destination_use) {
        if ((uint64_t)static_relocation.byte_offset >=
            (uint64_t)dynamic_relocation.byte_offset + 8) {
          break;
        }
        if ((uint64_t)static_relocation.byte_offset + 8 >
            dynamic_relocation.byte_offset) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "static and dynamic relocations overlap");
        }
      }
      ++static_index;
    }
  }
  for (uint32_t i = 0; i < entry->invocation_count; ++i) {
    const iree_xdna_elf_invocation_record_t invocation =
        iree_hal_amd_xdna_image_tables_invocation(tables,
                                                  entry->first_invocation + i);
    if (invocation.allocation_use >= entry->allocation_use_count ||
        invocation.next_invocation >= entry->invocation_count ||
        invocation.byte_length == 0 || invocation.byte_length % 4 != 0 ||
        invocation.byte_offset % target->instruction_alignment != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA invocation %u", i);
    }
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_entry_allocation(tables, entry,
                                                 invocation.allocation_use);
    if (allocation.domain != IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND ||
        allocation.alignment < target->instruction_alignment ||
        !iree_hal_amd_xdna_image_range_fits(invocation.byte_offset,
                                            invocation.byte_length,
                                            allocation.byte_length) ||
        !iree_hal_amd_xdna_image_range_initialized(directory, &allocation,
                                                   invocation.byte_offset,
                                                   invocation.byte_length)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA invocation requires initialized command backing");
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_image_validate(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_aie2p_target_t* target) {
  const iree_xdna_elf_header_record_t* header = &tables->header;
  if (iree_hal_amd_xdna_image_directory_target_flags(directory) !=
          IREE_XDNA_ELF_AIE2P_FLAGS ||
      header->target_generation != IREE_XDNA_TARGET_GENERATION_AIE2P ||
      header->native_encoding != IREE_XDNA_ELF_NATIVE_TRANSACTION_0_1 ||
      header->device_profile_revision !=
          target->identity.device_profile_revision ||
      header->device_profile_id != target->identity.device_profile_id ||
      header->firmware_abi_id != target->identity.firmware_abi_id ||
      header->column_count != target->context.column_count ||
      header->row_count != target->context.row_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA image does not match the admitted native context");
  }
  if (header->allocation_count == 0 || header->entry_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image has no executable storage or entries");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_image_validate_allocations(tables, directory));
  uint32_t next_use = 0;
  uint32_t next_binding = 0;
  uint32_t next_relocation = 0;
  uint32_t next_invocation = 0;
  for (uint32_t i = 0; i < header->entry_count; ++i) {
    const iree_xdna_elf_entry_record_t entry =
        iree_hal_amd_xdna_image_tables_entry(tables, i);
    if (entry.name_length == 0 ||
        entry.name_length > IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH ||
        !iree_hal_amd_xdna_image_range_fits(
            entry.name_offset, entry.name_length, header->string_byte_length) ||
        entry.allocation_use_count == 0 || entry.invocation_count == 0 ||
        entry.first_allocation_use != next_use ||
        entry.first_binding != next_binding ||
        entry.first_static_relocation != next_relocation ||
        !iree_hal_amd_xdna_image_range_fits(next_use,
                                            entry.allocation_use_count,
                                            header->allocation_use_count) ||
        !iree_hal_amd_xdna_image_range_fits(next_binding, entry.binding_count,
                                            header->binding_count) ||
        !iree_hal_amd_xdna_image_range_fits(next_relocation,
                                            entry.static_relocation_count,
                                            header->relocation_count) ||
        entry.first_invocation != next_invocation ||
        !iree_hal_amd_xdna_image_range_fits(next_invocation,
                                            entry.invocation_count,
                                            header->invocation_count)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA entry table ranges at %u", i);
    }
    next_relocation += entry.static_relocation_count;
    if (entry.first_dynamic_relocation != next_relocation ||
        !iree_hal_amd_xdna_image_range_fits(next_relocation,
                                            entry.dynamic_relocation_count,
                                            header->relocation_count)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA dynamic relocation range at %u", i);
    }
    next_use += entry.allocation_use_count;
    next_binding += entry.binding_count;
    next_relocation += entry.dynamic_relocation_count;
    next_invocation += entry.invocation_count;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_entry(
        tables, directory, target, &entry));
  }
  if (next_use != header->allocation_use_count ||
      next_binding != header->binding_count ||
      next_relocation != header->relocation_count ||
      next_invocation != header->invocation_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA metadata contains unowned rows");
  }
  return iree_ok_status();
}
