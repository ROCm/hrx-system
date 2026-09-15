// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/validation.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

struct iree_hal_amd_xdna_image_validation_t {
  // Allocator owning this object and its trailing placement table.
  iree_allocator_t host_allocator;
  // Capabilities implied by the complete program directory.
  iree_hal_amd_xdna_elf_capabilities_t structural_capabilities;
  // Number of entries in |tile_placements|.
  iree_host_size_t tile_placement_count;
  // Canonical TILE placements in program-header order.
  iree_hal_amd_xdna_image_tile_placement_t* tile_placements;
};

typedef struct iree_hal_amd_xdna_image_structure_t {
  // Capabilities implied by the complete program directory.
  iree_hal_amd_xdna_elf_capabilities_t capabilities;
  // Number of NOTE program headers.
  iree_host_size_t note_count;
  // Number of ENTRIES program headers.
  iree_host_size_t entry_table_count;
  // Number of BINDINGS program headers.
  iree_host_size_t binding_table_count;
  // Number of RELOCATIONS program headers.
  iree_host_size_t relocation_table_count;
  // Number of TILE program headers.
  iree_host_size_t tile_count;
  // Number of ARRAY program headers.
  iree_host_size_t array_count;
  // Number of CONTROL program headers.
  iree_host_size_t control_count;
  // Number of FIRMWARE program headers.
  iree_host_size_t firmware_count;
} iree_hal_amd_xdna_image_structure_t;

typedef struct iree_hal_amd_xdna_image_placement_index_t {
  // Canonical placement copied for sorting.
  iree_hal_amd_xdna_image_tile_placement_t placement;
} iree_hal_amd_xdna_image_placement_index_t;

typedef struct iree_hal_amd_xdna_image_validation_scratch_t {
  // TILE placements sorted by canonical storage range.
  iree_hal_amd_xdna_image_placement_index_t* placement_indexes;
  // Program-header relationship marks reused across validation phases.
  uint8_t* program_marks;
} iree_hal_amd_xdna_image_validation_scratch_t;

iree_status_t iree_hal_amd_xdna_image_target_validate(
    const iree_hal_amd_xdna_image_target_t* target) {
  if (target == NULL ||
      (target->supported_capabilities &
       ~IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES) != 0 ||
      target->abi_note_validator.fn == NULL ||
      target->tile_memory_resolver.fn == NULL ||
      target->program_record_validator.fn == NULL ||
      target->program_relocation_validator.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image target contract is incomplete");
  }
  return iree_ok_status();
}

static uint8_t iree_hal_amd_xdna_image_program_type_rank(uint32_t type) {
  switch (type) {
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE:
      return 1;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES:
      return 2;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS:
      return 3;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS:
      return 4;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE:
      return 5;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY:
      return 6;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL:
      return 7;
    case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_FIRMWARE:
      return 8;
    default:
      return 0;
  }
}

static bool iree_hal_amd_xdna_image_source_ranges_equal(
    iree_hal_amd_xdna_image_source_range_t lhs,
    iree_hal_amd_xdna_image_source_range_t rhs) {
  return lhs.offset == rhs.offset && lhs.length == rhs.length;
}

static iree_status_t iree_hal_amd_xdna_image_validate_tile_header(
    const iree_hal_amd_xdna_elf_abi_note_t* abi_note,
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    iree_host_size_t program_header_ordinal) {
  const iree_hal_amd_xdna_elf_tile_destination_t destination =
      iree_hal_amd_xdna_elf_unpack_tile_destination(
          program_header->physical_address);
  if (destination.column >= abi_note->context_column_count ||
      destination.row >= abi_note->context_row_count ||
      destination.flags != IREE_HAL_AMD_XDNA_ELF_TILE_KNOWN_FLAGS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA TILE program header %" PRIhsz
                            " has an invalid destination",
                            program_header_ordinal);
  }
  if (program_header->memory_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA TILE placement %" PRIhsz " is empty",
                            program_header_ordinal);
  }
  switch (destination.memory_space) {
    case IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM:
      if (program_header->flags !=
              (IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ |
               IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE) ||
          program_header->file_range.length == 0 ||
          program_header->alignment < 16) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "XDNA TILE program placement %" PRIhsz
                                " has an invalid executable memory contract",
                                program_header_ordinal);
      }
      break;
    case IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA:
      if ((program_header->flags & IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ) ==
              0 ||
          (program_header->flags &
           IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE) != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "XDNA TILE data placement %" PRIhsz
                                " has an invalid data memory contract",
                                program_header_ordinal);
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "XDNA TILE program header %" PRIhsz
                              " selects an unknown memory space",
                              program_header_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_validate_firmware_header(
    const iree_hal_amd_xdna_image_program_header_t* program_header,
    iree_host_size_t program_header_ordinal) {
  if (program_header->flags != IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ ||
      program_header->virtual_address != 0 ||
      program_header->physical_address != 0 ||
      program_header->file_range.length == 0 ||
      program_header->memory_size != program_header->file_range.length ||
      program_header->alignment != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA FIRMWARE program header %" PRIhsz
                            " has an invalid memory contract",
                            program_header_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_validate_source_alias(
    const iree_hal_amd_xdna_image_program_header_t* lhs,
    const iree_hal_amd_xdna_image_program_header_t* rhs) {
  if (lhs->file_range.length == 0 ||
      !iree_hal_amd_xdna_image_source_ranges_equal(lhs->file_range,
                                                   rhs->file_range)) {
    return iree_ok_status();
  }
  const iree_hal_amd_xdna_elf_tile_destination_t lhs_destination =
      iree_hal_amd_xdna_elf_unpack_tile_destination(lhs->physical_address);
  const iree_hal_amd_xdna_elf_tile_destination_t rhs_destination =
      iree_hal_amd_xdna_elf_unpack_tile_destination(rhs->physical_address);
  if (lhs->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE ||
      rhs->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE ||
      lhs->virtual_address != rhs->virtual_address ||
      lhs->memory_size != rhs->memory_size || lhs->flags != rhs->flags ||
      lhs->alignment != rhs->alignment ||
      lhs_destination.memory_space != rhs_destination.memory_space) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA program-header source alias is not an identical TILE payload");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_scan_structure(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_elf_abi_note_t* abi_note,
    const iree_hal_amd_xdna_image_programs_t* programs,
    iree_hal_amd_xdna_image_structure_t* out_structure) {
  *out_structure = (iree_hal_amd_xdna_image_structure_t){0};
  uint8_t previous_rank = 0;
  uint64_t previous_source_offset = 0;
  const iree_hal_amd_xdna_image_program_header_t* previous_header = NULL;
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  for (iree_host_size_t i = 0; i < program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    const uint8_t rank =
        iree_hal_amd_xdna_image_program_type_rank(program_header->type);
    if (rank == 0) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "XDNA program header %" PRIhsz
                              " has unknown type 0x%08" PRIX32,
                              i, program_header->type);
    }
    if (rank < previous_rank || (i != 0 && program_header->file_range.offset <
                                               previous_source_offset)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA program headers are not canonically ordered");
    }
    if (previous_header != NULL) {
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_source_alias(
          previous_header, program_header));
    }
    previous_rank = rank;
    previous_source_offset = program_header->file_range.offset;
    previous_header = program_header;

    switch (program_header->type) {
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE:
        ++out_structure->note_count;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES:
        ++out_structure->entry_table_count;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS:
        ++out_structure->binding_table_count;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS:
        ++out_structure->relocation_table_count;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE: {
        IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_tile_header(
            abi_note, program_header, i));
        ++out_structure->tile_count;
        if (program_header->memory_size > program_header->file_range.length) {
          out_structure->capabilities |=
              IREE_HAL_AMD_XDNA_ELF_CAPABILITY_TILE_ZERO_FILL;
        }
        break;
      }
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY:
        ++out_structure->array_count;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL:
        ++out_structure->control_count;
        break;
      case IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_FIRMWARE: {
        IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_validate_firmware_header(
            program_header, i));
        ++out_structure->firmware_count;
        break;
      }
      default:
        IREE_CHECK_UNREACHABLE("ranked XDNA image program type");
    }
  }

  if (out_structure->note_count != 1 || out_structure->entry_table_count != 1 ||
      out_structure->binding_table_count != 1 ||
      out_structure->relocation_table_count > 1 ||
      out_structure->tile_count == 0 || out_structure->array_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image is missing or duplicates a program "
                            "directory role");
  }
  if (out_structure->array_count !=
          iree_hal_amd_xdna_image_programs_array_count(programs) ||
      out_structure->control_count !=
          iree_hal_amd_xdna_image_programs_control_count(programs)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA program framing does not match its directory");
  }
  if (out_structure->relocation_table_count != 0) {
    out_structure->capabilities |=
        IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS;
  }
  if (out_structure->control_count != 0) {
    out_structure->capabilities |=
        IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS;
  }
  if (out_structure->firmware_count != 0) {
    out_structure->capabilities |=
        IREE_HAL_AMD_XDNA_ELF_CAPABILITY_FIRMWARE_PAYLOADS;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_qualify_target(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_elf_abi_note_t* abi_note,
    iree_hal_amd_xdna_elf_capabilities_t structural_capabilities,
    const iree_hal_amd_xdna_image_target_t* target) {
  if (iree_hal_amd_xdna_image_directory_target_flags(directory) !=
      target->target_flags) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "XDNA image ELF flags do not match the target");
  }
  IREE_RETURN_IF_ERROR(target->abi_note_validator.fn(
      target->abi_note_validator.user_data, abi_note));
  if ((abi_note->required_capabilities & ~target->supported_capabilities) !=
      0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "XDNA image requires capabilities unsupported by the target");
  }
  if (abi_note->required_capabilities != structural_capabilities) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA image capability note disagrees with its program directory");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_resolve_tile_placements(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_elf_abi_note_t* abi_note,
    const iree_hal_amd_xdna_image_target_t* target,
    iree_hal_amd_xdna_image_validation_t* validation) {
  iree_host_size_t placement_ordinal = 0;
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  for (iree_host_size_t i = 0; i < program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    if (program_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE) {
      continue;
    }
    const iree_hal_amd_xdna_elf_tile_destination_t destination =
        iree_hal_amd_xdna_elf_unpack_tile_destination(
            program_header->physical_address);
    iree_hal_amd_xdna_image_tile_placement_t placement = {0};
    IREE_RETURN_IF_ERROR(target->tile_memory_resolver.fn(
        target->tile_memory_resolver.user_data, &destination,
        program_header->virtual_address, program_header->memory_size,
        &placement));
    const uint64_t available_end =
        (uint64_t)placement.owner_offset + placement.available_capacity;
    if (placement.owner_column >= abi_note->context_column_count ||
        placement.owner_row >= abi_note->context_row_count ||
        placement.memory_space != destination.memory_space ||
        placement.byte_length != program_header->memory_size ||
        placement.available_capacity < placement.byte_length ||
        available_end > UINT64_C(1) + UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA tile-memory resolver returned an invalid placement");
    }
    if (destination.memory_space ==
            IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM &&
        (placement.owner_column != destination.column ||
         placement.owner_row != destination.row ||
         placement.owner_offset != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA TILE program placement must begin at its startup address");
    }
    placement.program_header_ordinal = (uint32_t)i;
    validation->tile_placements[placement_ordinal++] = placement;
  }
  if (placement_ordinal != validation->tile_placement_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA resolved TILE placements do not match the directory");
  }
  return iree_ok_status();
}

static int iree_hal_amd_xdna_image_compare_placement_indexes(
    const void* lhs_ptr, const void* rhs_ptr) {
  const iree_hal_amd_xdna_image_tile_placement_t* lhs =
      &((const iree_hal_amd_xdna_image_placement_index_t*)lhs_ptr)->placement;
  const iree_hal_amd_xdna_image_tile_placement_t* rhs =
      &((const iree_hal_amd_xdna_image_placement_index_t*)rhs_ptr)->placement;
  if (lhs->owner_column < rhs->owner_column) return -1;
  if (lhs->owner_column > rhs->owner_column) return 1;
  if (lhs->owner_row < rhs->owner_row) return -1;
  if (lhs->owner_row > rhs->owner_row) return 1;
  if (lhs->memory_space < rhs->memory_space) return -1;
  if (lhs->memory_space > rhs->memory_space) return 1;
  if (lhs->owner_offset < rhs->owner_offset) return -1;
  if (lhs->owner_offset > rhs->owner_offset) return 1;
  if (lhs->byte_length < rhs->byte_length) return -1;
  if (lhs->byte_length > rhs->byte_length) return 1;
  if (lhs->program_header_ordinal < rhs->program_header_ordinal) return -1;
  if (lhs->program_header_ordinal > rhs->program_header_ordinal) return 1;
  return 0;
}

static iree_status_t iree_hal_amd_xdna_image_validate_tile_overlaps(
    const iree_hal_amd_xdna_image_validation_t* validation,
    iree_hal_amd_xdna_image_placement_index_t* placement_indexes) {
  for (iree_host_size_t i = 0; i < validation->tile_placement_count; ++i) {
    placement_indexes[i].placement = validation->tile_placements[i];
  }
  qsort(placement_indexes, validation->tile_placement_count,
        sizeof(*placement_indexes),
        iree_hal_amd_xdna_image_compare_placement_indexes);
  for (iree_host_size_t i = 1; i < validation->tile_placement_count; ++i) {
    const iree_hal_amd_xdna_image_tile_placement_t* lhs =
        &placement_indexes[i - 1].placement;
    const iree_hal_amd_xdna_image_tile_placement_t* rhs =
        &placement_indexes[i].placement;
    if (lhs->owner_column != rhs->owner_column ||
        lhs->owner_row != rhs->owner_row ||
        lhs->memory_space != rhs->memory_space) {
      continue;
    }
    const uint64_t lhs_end = (uint64_t)lhs->owner_offset + lhs->byte_length;
    if (rhs->owner_offset < lhs_end) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA canonical TILE placements %" PRIu32 " and %" PRIu32 " overlap",
          lhs->program_header_ordinal, rhs->program_header_ordinal);
    }
  }
  return iree_ok_status();
}

enum {
  IREE_HAL_AMD_XDNA_PROGRAM_MARK_ARRAY_TILE = 1u << 0,
  IREE_HAL_AMD_XDNA_PROGRAM_MARK_CURRENT_REFERENCE = 1u << 1,
  IREE_HAL_AMD_XDNA_PROGRAM_MARK_ENTRY_ARRAY = 1u << 2,
  IREE_HAL_AMD_XDNA_PROGRAM_MARK_ENTRY_CONTROL = 1u << 3,
};

static iree_status_t iree_hal_amd_xdna_image_validate_array_relationships(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_programs_t* programs,
    uint8_t* program_marks) {
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  const iree_host_size_t record_count =
      iree_hal_amd_xdna_image_programs_record_count(programs);
  const iree_host_size_t array_count =
      iree_hal_amd_xdna_image_programs_array_count(programs);
  for (iree_host_size_t i = 0; i < array_count; ++i) {
    const iree_hal_amd_xdna_image_array_program_t* array =
        iree_hal_amd_xdna_image_programs_array(programs, i);
    uint64_t tile_program_header_end = 0;
    uint64_t record_end = 0;
    if (array->program_header_ordinal >= program_header_count ||
        iree_hal_amd_xdna_image_directory_program_header(
            directory, array->program_header_ordinal)
                ->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY ||
        !iree_checked_add_u64(array->first_tile_program_header_ordinal,
                              array->tile_program_header_count,
                              &tile_program_header_end) ||
        tile_program_header_end > program_header_count ||
        !iree_checked_add_u64(array->first_record_ordinal, array->record_count,
                              &record_end) ||
        record_end > record_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA ARRAY realization references an invalid program range");
    }
    for (uint32_t ordinal = array->first_tile_program_header_ordinal;
         ordinal < tile_program_header_end; ++ordinal) {
      const iree_hal_amd_xdna_image_program_header_t* tile_header =
          iree_hal_amd_xdna_image_directory_program_header(directory, ordinal);
      if (tile_header->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE ||
          (tile_header->flags & IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE) ==
              0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA ARRAY realization references a non-executable TILE");
      }
      program_marks[ordinal] |= IREE_HAL_AMD_XDNA_PROGRAM_MARK_ARRAY_TILE;
      program_marks[ordinal] &=
          (uint8_t)~IREE_HAL_AMD_XDNA_PROGRAM_MARK_CURRENT_REFERENCE;
    }
    for (uint32_t ordinal = array->first_record_ordinal; ordinal < record_end;
         ++ordinal) {
      const iree_hal_amd_xdna_image_program_record_t* record =
          iree_hal_amd_xdna_image_programs_record(programs, ordinal);
      if (record->program_header_ordinal != array->program_header_ordinal) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "XDNA ARRAY records do not match their program framing");
      }
      if (record->referenced_program_header_ordinal == UINT32_MAX) continue;
      const uint32_t reference = record->referenced_program_header_ordinal;
      if (reference < array->first_tile_program_header_ordinal ||
          reference >= tile_program_header_end ||
          (program_marks[reference] &
           IREE_HAL_AMD_XDNA_PROGRAM_MARK_CURRENT_REFERENCE) != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA ARRAY records contain an invalid or duplicate TILE "
            "reference");
      }
      program_marks[reference] |=
          IREE_HAL_AMD_XDNA_PROGRAM_MARK_CURRENT_REFERENCE;
    }
    for (uint32_t ordinal = array->first_tile_program_header_ordinal;
         ordinal < tile_program_header_end; ++ordinal) {
      if ((program_marks[ordinal] &
           IREE_HAL_AMD_XDNA_PROGRAM_MARK_CURRENT_REFERENCE) == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA ARRAY records omit an executable TILE reference");
      }
    }
  }

  for (iree_host_size_t i = 0; i < program_header_count; ++i) {
    const iree_hal_amd_xdna_image_program_header_t* program_header =
        iree_hal_amd_xdna_image_directory_program_header(directory, i);
    if (program_header->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE &&
        (program_header->flags & IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE) !=
            0 &&
        (program_marks[i] & IREE_HAL_AMD_XDNA_PROGRAM_MARK_ARRAY_TILE) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XDNA executable TILE program header %" PRIhsz
                              " is not owned by an ARRAY realization",
                              i);
    }
  }

  const iree_host_size_t control_count =
      iree_hal_amd_xdna_image_programs_control_count(programs);
  for (iree_host_size_t i = 0; i < control_count; ++i) {
    const iree_hal_amd_xdna_image_control_program_t* control =
        iree_hal_amd_xdna_image_programs_control(programs, i);
    const uint64_t record_end =
        (uint64_t)control->first_record_ordinal + control->record_count;
    if (control->program_header_ordinal >= program_header_count ||
        iree_hal_amd_xdna_image_directory_program_header(
            directory, control->program_header_ordinal)
                ->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL ||
        record_end > record_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA CONTROL program references an invalid record range");
    }
    for (uint32_t ordinal = control->first_record_ordinal; ordinal < record_end;
         ++ordinal) {
      const iree_hal_amd_xdna_image_program_record_t* record =
          iree_hal_amd_xdna_image_programs_record(programs, ordinal);
      if (record->program_header_ordinal != control->program_header_ordinal ||
          record->referenced_program_header_ordinal != UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA CONTROL record directory contains an invalid program "
            "reference");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_image_validate_entry_relationships(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_programs_t* programs,
    uint8_t* program_marks) {
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  const iree_host_size_t entry_count =
      iree_hal_amd_xdna_image_tables_entry_count(tables);
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_hal_amd_xdna_elf_entry_record_t* entry =
        iree_hal_amd_xdna_image_tables_entry(tables, i);
    if (entry->array_program_header_ordinal >= program_header_count ||
        iree_hal_amd_xdna_image_directory_program_header(
            directory, entry->array_program_header_ordinal)
                ->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA entry %" PRIhsz " references an invalid ARRAY realization", i);
    }
    program_marks[entry->array_program_header_ordinal] |=
        IREE_HAL_AMD_XDNA_PROGRAM_MARK_ENTRY_ARRAY;
    if (entry->control_program_header_ordinal != UINT32_MAX) {
      if (entry->control_program_header_ordinal >= program_header_count ||
          iree_hal_amd_xdna_image_directory_program_header(
              directory, entry->control_program_header_ordinal)
                  ->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA entry %" PRIhsz " references an invalid CONTROL program", i);
      }
      program_marks[entry->control_program_header_ordinal] |=
          IREE_HAL_AMD_XDNA_PROGRAM_MARK_ENTRY_CONTROL;
    }
  }
  const iree_host_size_t array_count =
      iree_hal_amd_xdna_image_programs_array_count(programs);
  for (iree_host_size_t i = 0; i < array_count; ++i) {
    const iree_hal_amd_xdna_image_array_program_t* array =
        iree_hal_amd_xdna_image_programs_array(programs, i);
    if ((program_marks[array->program_header_ordinal] &
         IREE_HAL_AMD_XDNA_PROGRAM_MARK_ENTRY_ARRAY) == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA ARRAY realization is not referenced by an entry");
    }
  }
  const iree_host_size_t control_count =
      iree_hal_amd_xdna_image_programs_control_count(programs);
  for (iree_host_size_t i = 0; i < control_count; ++i) {
    const iree_hal_amd_xdna_image_control_program_t* control =
        iree_hal_amd_xdna_image_programs_control(programs, i);
    if ((program_marks[control->program_header_ordinal] &
         IREE_HAL_AMD_XDNA_PROGRAM_MARK_ENTRY_CONTROL) == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA CONTROL program is not referenced by an entry");
    }
  }
  return iree_ok_status();
}

static const iree_hal_amd_xdna_image_program_record_t*
iree_hal_amd_xdna_image_find_program_record(
    const iree_hal_amd_xdna_image_programs_t* programs,
    uint32_t program_header_ordinal, uint64_t source_offset) {
  iree_host_size_t low = 0;
  iree_host_size_t high =
      iree_hal_amd_xdna_image_programs_record_count(programs);
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const iree_hal_amd_xdna_image_program_record_t* record =
        iree_hal_amd_xdna_image_programs_record(programs, middle);
    if (record->program_header_ordinal < program_header_ordinal ||
        (record->program_header_ordinal == program_header_ordinal &&
         record->source_range.offset <= source_offset)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low == 0) return NULL;
  const iree_hal_amd_xdna_image_program_record_t* record =
      iree_hal_amd_xdna_image_programs_record(programs, low - 1);
  return record->program_header_ordinal == program_header_ordinal ? record
                                                                  : NULL;
}

static bool iree_hal_amd_xdna_image_relocation_interval_is_satisfiable(
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation) {
  const uint64_t alignment_mask = relocation->required_alignment - 1;
  if (relocation->minimum_value > UINT64_MAX - alignment_mask) return false;
  const uint64_t first_aligned_value =
      (relocation->minimum_value + alignment_mask) & ~alignment_mask;
  return first_aligned_value <= relocation->maximum_value;
}

static iree_status_t iree_hal_amd_xdna_image_validate_relocations(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_programs_t* programs,
    const iree_hal_amd_xdna_image_target_t* target) {
  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  const iree_host_size_t relocation_count =
      iree_hal_amd_xdna_image_tables_relocation_count(tables);
  for (iree_host_size_t i = 0; i < relocation_count; ++i) {
    const iree_hal_amd_xdna_elf_relocation_record_t* relocation =
        iree_hal_amd_xdna_image_tables_relocation(tables, i);
    if (relocation->target_program_header_ordinal >= program_header_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA runtime relocation %" PRIhsz
                              " references an invalid program header",
                              i);
    }
    const iree_hal_amd_xdna_image_program_header_t* target_program =
        iree_hal_amd_xdna_image_directory_program_header(
            directory, relocation->target_program_header_ordinal);
    if (target_program->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE &&
        target_program->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY &&
        target_program->type != IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "XDNA runtime relocation %" PRIhsz
                              " targets an immutable directory payload",
                              i);
    }
    const uint64_t target_end =
        (uint64_t)relocation->target_byte_offset + relocation->field_byte_width;
    if (target_end > target_program->file_range.length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA runtime relocation %" PRIhsz " exceeds its target payload", i);
    }
    if (!iree_hal_amd_xdna_image_relocation_interval_is_satisfiable(
            relocation)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "XDNA runtime relocation %" PRIhsz
          " has no value satisfying its range and alignment",
          i);
    }
    if (target_program->type == IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE) {
      continue;
    }

    const uint64_t source_offset =
        target_program->file_range.offset + relocation->target_byte_offset;
    const iree_hal_amd_xdna_image_program_record_t* record =
        iree_hal_amd_xdna_image_find_program_record(
            programs, relocation->target_program_header_ordinal, source_offset);
    uint64_t record_offset = 0;
    uint64_t record_payload_offset = 0;
    uint64_t record_end = 0;
    bool record_range_is_valid = false;
    if (record != NULL &&
        record->source_range.offset >= target_program->file_range.offset) {
      record_offset =
          record->source_range.offset - target_program->file_range.offset;
      record_range_is_valid =
          iree_checked_add_u64(record_offset,
                               IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE,
                               &record_payload_offset) &&
          iree_checked_add_u64(record_offset, record->source_range.length,
                               &record_end) &&
          record_end <= target_program->file_range.length;
    }
    if (!record_range_is_valid ||
        relocation->target_byte_offset < record_payload_offset ||
        target_end > record_end) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "XDNA runtime relocation %" PRIhsz
                              " targets immutable program framing",
                              i);
    }
    const uint64_t record_relative_byte_offset =
        source_offset - record->source_range.offset;
    if (record_relative_byte_offset > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XDNA runtime relocation record-relative offset overflows");
    }
    IREE_RETURN_IF_ERROR(target->program_relocation_validator.fn(
        target->program_relocation_validator.user_data,
        (iree_hal_amd_xdna_elf_program_type_t)target_program->type, record,
        (uint32_t)record_relative_byte_offset, relocation));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_image_validation_create(
    const iree_hal_amd_xdna_image_directory_t* directory,
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_hal_amd_xdna_image_programs_t* programs,
    const iree_hal_amd_xdna_image_target_t* target,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_image_validation_t** out_validation) {
  IREE_ASSERT_ARGUMENT(out_validation);
  *out_validation = NULL;
  if (directory == NULL || tables == NULL || programs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "decoded XDNA image components are required");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_target_validate(target));

  const iree_hal_amd_xdna_elf_abi_note_t* abi_note =
      iree_hal_amd_xdna_image_tables_abi_note(tables);
  iree_hal_amd_xdna_image_structure_t structure;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_scan_structure(
      directory, abi_note, programs, &structure));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_qualify_target(
      directory, abi_note, structure.capabilities, target));

  iree_host_size_t validation_size = 0;
  iree_host_size_t tile_placements_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_image_validation_t), &validation_size,
      IREE_STRUCT_FIELD_ALIGNED(
          structure.tile_count, iree_hal_amd_xdna_image_tile_placement_t,
          iree_alignof(iree_hal_amd_xdna_image_tile_placement_t),
          &tile_placements_offset)));
  iree_hal_amd_xdna_image_validation_t* validation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, validation_size,
                                             (void**)&validation));
  validation->host_allocator = host_allocator;
  validation->structural_capabilities = structure.capabilities;
  validation->tile_placement_count = structure.tile_count;
  validation->tile_placements =
      (iree_hal_amd_xdna_image_tile_placement_t*)((uint8_t*)validation +
                                                  tile_placements_offset);

  const iree_host_size_t program_header_count =
      iree_hal_amd_xdna_image_directory_program_header_count(directory);
  iree_host_size_t scratch_size = 0;
  iree_host_size_t placement_indexes_offset = 0;
  iree_host_size_t program_marks_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_image_validation_scratch_t), &scratch_size,
      IREE_STRUCT_FIELD_ALIGNED(
          structure.tile_count, iree_hal_amd_xdna_image_placement_index_t,
          iree_alignof(iree_hal_amd_xdna_image_placement_index_t),
          &placement_indexes_offset),
      IREE_STRUCT_FIELD_ALIGNED(program_header_count, uint8_t,
                                iree_alignof(uint8_t), &program_marks_offset));
  iree_hal_amd_xdna_image_validation_scratch_t* scratch = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, scratch_size, (void**)&scratch);
  }
  if (iree_status_is_ok(status)) {
    scratch->placement_indexes =
        (iree_hal_amd_xdna_image_placement_index_t*)((uint8_t*)scratch +
                                                     placement_indexes_offset);
    scratch->program_marks = (uint8_t*)scratch + program_marks_offset;
    memset(scratch->program_marks, 0, program_header_count);
    status = iree_hal_amd_xdna_image_resolve_tile_placements(
        directory, abi_note, target, validation);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate_tile_overlaps(
        validation, scratch->placement_indexes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate_array_relationships(
        directory, programs, scratch->program_marks);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate_entry_relationships(
        directory, tables, programs, scratch->program_marks);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validate_relocations(directory, tables,
                                                          programs, target);
  }
  iree_allocator_free(host_allocator, scratch);

  if (iree_status_is_ok(status)) {
    *out_validation = validation;
  } else {
    iree_allocator_free(host_allocator, validation);
  }
  return status;
}

void iree_hal_amd_xdna_image_validation_destroy(
    iree_hal_amd_xdna_image_validation_t* validation) {
  if (validation == NULL) return;
  iree_allocator_free(validation->host_allocator, validation);
}

iree_hal_amd_xdna_elf_capabilities_t
iree_hal_amd_xdna_image_validation_structural_capabilities(
    const iree_hal_amd_xdna_image_validation_t* validation) {
  IREE_ASSERT_ARGUMENT(validation);
  return validation->structural_capabilities;
}

iree_host_size_t iree_hal_amd_xdna_image_validation_tile_placement_count(
    const iree_hal_amd_xdna_image_validation_t* validation) {
  IREE_ASSERT_ARGUMENT(validation);
  return validation->tile_placement_count;
}

const iree_hal_amd_xdna_image_tile_placement_t*
iree_hal_amd_xdna_image_validation_tile_placement(
    const iree_hal_amd_xdna_image_validation_t* validation,
    iree_host_size_t ordinal) {
  IREE_ASSERT_ARGUMENT(validation);
  return ordinal < validation->tile_placement_count
             ? &validation->tile_placements[ordinal]
             : NULL;
}
