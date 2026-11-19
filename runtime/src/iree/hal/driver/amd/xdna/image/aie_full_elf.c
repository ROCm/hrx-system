// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/aie_full_elf.h"

#include <stdlib.h>
#include <string.h>

#include "iree/hal/driver/amd/xdna/image/aie_sections.h"

// One row per ELF section during admission. A control section's group names
// its public export. PDI sections acquire an allocation only when referenced.
typedef struct iree_hal_amd_aie_resource_t {
  // Kernel portion of the qualified export name.
  iree_string_view_t kernel;
  // Runtime sequence selected by the group's signature symbol.
  iree_string_view_t sequence;
  // Pointer argument count in the kernel signature, including padding slots.
  uint32_t argument_count;
  // Image-wide backing ordinal plus one, or zero for unused sections.
  uint32_t backing;
  // External slot extent, including holes below the highest used slot.
  uint32_t binding_count;
  // Number of address_64 patch sites in this command.
  uint32_t static_count;
  // Number of shim_dma_48 patch sites in this command.
  uint32_t dynamic_count;
  // Offset of this command's patches in destination order.
  uint32_t first_patch;
  // Control section last referencing this PDI while building an entry.
  uint32_t owner;
  // Relative allocation use within that entry.
  uint32_t use;
} iree_hal_amd_aie_resource_t;

typedef struct iree_hal_amd_aie_patch_t {
  // Destination control section ordinal.
  uint32_t command;
  // Immutable configuration section, or zero for an external buffer.
  uint32_t pdi;
  // External argument ordinal for a shim address.
  uint32_t binding;
  // Byte offset of the eight-byte address field in the control section.
  uint32_t offset;
  // Combined program displacement and ELF buffer addend.
  uint64_t displacement;
  // Producer-declared DMA transfer byte length.
  uint32_t transfer_size;
} iree_hal_amd_aie_patch_t;

static bool iree_hal_amd_aie_decimal(iree_string_view_t text,
                                     uint32_t* out_value) {
  uint32_t result = 0;
  if (!text.size) {
    return false;
  }
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    unsigned digit = (unsigned)(text.data[i] - '0');
    if (digit > 9 || result > (UINT32_MAX - digit) / 10) {
      return false;
    }
    result = result * 10 + digit;
  }
  *out_value = result;
  return true;
}

static iree_status_t iree_hal_amd_aie_signature(
    iree_string_view_t signature, iree_hal_amd_aie_resource_t* resource) {
  if (!iree_string_view_consume_prefix(&signature, IREE_SV("_Z"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE kernel signature");
  }
  iree_host_size_t digits = 0;
  while (digits < signature.size && signature.data[digits] >= '0' &&
         signature.data[digits] <= '9') {
    ++digits;
  }
  uint32_t length = 0;
  if (!iree_hal_amd_aie_decimal(iree_string_view_substr(signature, 0, digits),
                                &length) ||
      !length || length >= signature.size - digits) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE kernel name length");
  }
  resource->kernel = iree_string_view_substr(signature, digits, length);
  iree_string_view_t arguments =
      iree_string_view_remove_prefix(signature, digits + length);
  if (iree_string_view_equal(arguments, IREE_SV("v"))) {
    return iree_ok_status();
  }
  while (arguments.size) {
    if (arguments.size < 2 || arguments.data[0] != 'P' ||
        (arguments.data[1] != 'c' && arguments.data[1] != 'i' &&
         arguments.data[1] != 'v')) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported AIE kernel argument encoding");
    }
    ++resource->argument_count;
    arguments = iree_string_view_remove_prefix(arguments, 2);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_aie_group(
    const iree_hal_amd_aie_sections_t* file,
    const iree_hal_amd_aie_section_t* group,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_hal_amd_aie_resource_t* resources,
    iree_xdna_elf_header_record_t* counts) {
  if (group->size < 8 || iree_unaligned_load_le_u32(group->records) != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE ELF group");
  }
  if (group->size != 8) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE group requires auxiliary control or preemption sections");
  }
  const uint32_t member = iree_unaligned_load_le_u32(group->records + 4);
  if (member == 0 || member >= file->count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE group member is out of range");
  }
  const iree_hal_amd_aie_section_t* command = &file->sections[member];
  iree_hal_amd_aie_resource_t* resource = &resources[member];
  if (resource->kernel.size || command->type != IREE_HAL_AMD_AIE_PROGBITS ||
      !iree_string_view_starts_with(command->name, IREE_SV(".ctrltext."))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE group must own a unique controller section");
  }
  if ((command->flags & 0x800) ||
      !iree_is_power_of_two_uint64(command->alignment)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported AIE controller section encoding");
  }
  uint8_t header[16];
  if (command->size < sizeof(header) || command->size % 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "truncated AIE controller transaction");
  }
  IREE_RETURN_IF_ERROR(
      iree_byte_sequence_read(file->source, command->offset,
                              iree_make_byte_span(header, sizeof(header))));
  if (header[0] != 0 || header[1] != 1 ||
      header[2] != 4 /* XAIE_DEV_GEN_AIE2P */ ||
      header[3] != target->context.row_count ||
      header[4] != target->context.column_count || header[5] != 1 ||
      iree_unaligned_load_le_u32(header + 12) != command->size) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE controller transaction does not match the NPU context");
  }
  const uint8_t* object = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_aie_sections_symbol(
      file, group->link, IREE_HAL_AMD_AIE_SYMTAB, group->info,
      1 /* STT_OBJECT */, &resource->sequence, &object));
  const uint8_t* function = NULL;
  iree_string_view_t signature;
  IREE_RETURN_IF_ERROR(iree_hal_amd_aie_sections_symbol(
      file, group->link, IREE_HAL_AMD_AIE_SYMTAB,
      iree_unaligned_load_le_u16(object + 14), 2 /* STT_FUNC */, &signature,
      &function));
  IREE_RETURN_IF_ERROR(iree_hal_amd_aie_signature(signature, resource));
  const iree_host_size_t name_length =
      resource->kernel.size + 1 + resource->sequence.size;
  if (!resource->sequence.size ||
      name_length > IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE export name length");
  }
  for (uint32_t i = 0; i < file->count; ++i) {
    if (i != member &&
        iree_string_view_equal(resources[i].sequence, resource->sequence) &&
        iree_string_view_equal(resources[i].kernel, resource->kernel)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AIE export name");
    }
  }
  resource->backing = ++counts->allocation_count;
  ++counts->entry_count;
  ++counts->invocation_count;
  ++counts->allocation_use_count;
  counts->string_byte_length += (uint32_t)name_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_aie_configuration(
    const iree_hal_amd_aie_section_t* note,
    const iree_hal_amd_xdna_aie2p_target_t* target) {
  if (note->type != IREE_HAL_AMD_AIE_NOTE || note->size < 20 ||
      iree_unaligned_load_le_u32(note->records) != 4 ||
      iree_unaligned_load_le_u32(note->records + 4) != 4 ||
      iree_unaligned_load_le_u32(note->records + 8) != 6 ||
      memcmp(note->records + 12, "XRT", 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE ELF partition note");
  }
  const uint32_t columns = iree_unaligned_load_le_u32(note->records + 16);
  if (columns != target->context.column_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE ELF requires %u columns, context has %u",
                            columns, target->context.column_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_aie_read_patch(
    const iree_hal_amd_aie_sections_t* file, uint32_t symtab,
    const uint8_t* rela, iree_hal_amd_aie_resource_t* resources,
    iree_xdna_elf_header_record_t* counts, iree_hal_amd_aie_patch_t* patch) {
  const uint32_t info = iree_unaligned_load_le_u32(rela + 4);
  const uint32_t kind = info & 0xff;
  const uint32_t addend = iree_unaligned_load_le_u32(rela + 8);
  iree_string_view_t name;
  const uint8_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_aie_sections_symbol(
      file, symtab, IREE_HAL_AMD_AIE_DYNSYM, info >> 8, 1 /* STT_OBJECT */,
      &name, &symbol));
  const uint32_t command = iree_unaligned_load_le_u16(symbol + 14);
  if (command >= file->count || !resources[command].kernel.size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AIE relocation requires an auxiliary section");
  }
  if (kind != 5 /* shim_dma_48 */ && kind != 8 /* address_64 */) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported AIE relocation %u", kind);
  }
  // A shim relocation points to the BD's length word; the address is in the
  // following two words. An address_64 relocation points directly to the field.
  uint64_t offset = iree_unaligned_load_le_u32(rela);
  if (kind == 5) {
    offset += 4;
  }
  if (offset < 16 || offset % 4 || offset + 8 > file->sections[command].size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid AIE relocation field");
  }
  *patch = (iree_hal_amd_aie_patch_t){.command = command,
                                      .offset = (uint32_t)offset};
  if (kind == 8) {
    if (!iree_string_view_starts_with(name, IREE_SV(".pdi.")) || addend) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "expected the firmware base address of an AIE PDI");
    }
    for (uint32_t i = 1; i < file->count; ++i) {
      if (!iree_string_view_equal(file->sections[i].name, name)) {
        continue;
      }
      const iree_hal_amd_aie_section_t* section = &file->sections[i];
      if (patch->pdi || section->type != IREE_HAL_AMD_AIE_PROGBITS ||
          !section->size) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid AIE PDI reference");
      }
      if ((section->flags & 0x800) ||
          !iree_is_power_of_two_uint64(section->alignment)) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported AIE PDI encoding");
      }
      patch->pdi = i;
    }
    if (!patch->pdi) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing AIE PDI section");
    }
    if (!resources[patch->pdi].backing) {
      resources[patch->pdi].backing = ++counts->allocation_count;
    }
    ++resources[command].static_count;
  } else {
    if (!iree_hal_amd_aie_decimal(name, &patch->binding) ||
        patch->binding >= resources[command].argument_count ||
        patch->binding >= UINT16_MAX) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "expected an AIE buffer argument ordinal within "
                              "the kernel signature");
    }
    uint8_t field[8];
    IREE_RETURN_IF_ERROR(iree_byte_sequence_read(
        file->source, file->sections[command].offset + offset,
        iree_make_byte_span(field, sizeof(field))));
    // AIEBU may bake a large buffer displacement into the original BD. Read
    // it once so rebinding always starts from the immutable program offset.
    patch->displacement =
        ((uint64_t)(iree_unaligned_load_le_u32(field + 4) & 0xffffu) << 32) +
        (iree_unaligned_load_le_u32(field) & UINT32_C(0xfffffffc)) + addend;
    if (patch->displacement % 4 || patch->displacement >= (UINT64_C(1) << 48)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "invalid AIE shim DMA displacement");
    }
    patch->transfer_size = iree_unaligned_load_le_u32(symbol + 8);
    resources[command].binding_count =
        iree_max(resources[command].binding_count, patch->binding + 1);
    ++resources[command].dynamic_count;
  }
  return iree_ok_status();
}

static int iree_hal_amd_aie_compare_patches(const void* lhs, const void* rhs) {
  const iree_hal_amd_aie_patch_t* a = lhs;
  const iree_hal_amd_aie_patch_t* b = rhs;
  if (a->command != b->command) {
    return a->command < b->command ? -1 : 1;
  }
  return (a->offset > b->offset) - (a->offset < b->offset);
}

static iree_status_t iree_hal_amd_aie_index_patches(
    const iree_hal_amd_aie_sections_t* file,
    iree_hal_amd_aie_resource_t* resources,
    iree_xdna_elf_header_record_t* counts, iree_hal_amd_aie_patch_t* patches) {
  uint32_t index = 0;
  for (uint32_t i = 0; i < file->count; ++i) {
    const iree_hal_amd_aie_section_t* section = &file->sections[i];
    if (section->type != IREE_HAL_AMD_AIE_RELA) {
      continue;
    }
    for (uint32_t j = 0; j < section->size; j += 12) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_aie_read_patch(file, section->link, section->records + j,
                                      resources, counts, &patches[index++]));
    }
  }
  if (index > 1) {
    qsort(patches, index, sizeof(*patches), iree_hal_amd_aie_compare_patches);
  }
  for (uint32_t i = 0; i < index; ++i) {
    const iree_hal_amd_aie_patch_t* patch = &patches[i];
    if (!i || patch->command != patches[i - 1].command) {
      resources[patch->command].first_patch = i;
    } else if ((uint64_t)patches[i - 1].offset + 8 > patch->offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "overlapping AIE address fields");
    }
    if (patch->pdi && resources[patch->pdi].owner != patch->command) {
      resources[patch->pdi].owner = patch->command;
      ++counts->allocation_use_count;
    }
  }
  uint64_t binding_count = 0;
  for (uint32_t i = 0; i < file->count; ++i) {
    resources[i].owner = 0;
    binding_count += resources[i].binding_count;
  }
  if (binding_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many AIE buffer binding slots");
  }
  counts->binding_count = (uint32_t)binding_count;
  return iree_ok_status();
}

// The section and symbol relationships are resolved. This pass writes dense
// rows in entry order and materializes only referenced sections.
static void iree_hal_amd_aie_build_plan(
    const iree_hal_amd_aie_sections_t* file,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_hal_amd_aie_resource_t* resources,
    const iree_hal_amd_aie_patch_t* patches,
    iree_hal_amd_xdna_image_tables_t* tables) {
  uint32_t next_entry = 0, next_use = 0, next_binding = 0, next_patch = 0,
           next_name = 0;
  for (uint32_t i = 1; i < file->count; ++i) {
    const iree_hal_amd_aie_resource_t* resource = &resources[i];
    if (!resource->backing) {
      continue;
    }
    const uint32_t allocation = resource->backing - 1;
    const iree_hal_amd_aie_section_t* section = &file->sections[i];
    tables->loads[allocation] =
        (iree_hal_amd_xdna_image_load_t){.source_offset = section->offset,
                                         .file_size = section->size,
                                         .memory_size = section->size};
    tables->allocations[allocation] = (iree_xdna_elf_allocation_record_t){
        .domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND,
        .flags = resource->dynamic_count
                     ? 0
                     : IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE,
        .byte_length = section->size,
        .alignment =
            iree_max(section->alignment, target->instruction_alignment),
        .first_load = allocation,
        .load_count = 1};
    if (!resource->kernel.size) {
      continue;
    }
    iree_xdna_elf_entry_record_t* entry = &tables->entries[next_entry];
    const uint32_t name_length =
        (uint32_t)(resource->kernel.size + 1 + resource->sequence.size);
    *entry = (iree_xdna_elf_entry_record_t){
        .name_offset = next_name,
        .name_length = name_length,
        .first_allocation_use = next_use,
        .first_binding = next_binding,
        .binding_count = resource->binding_count,
        .first_static_relocation = next_patch,
        .static_relocation_count = resource->static_count,
        .first_dynamic_relocation = next_patch + resource->static_count,
        .dynamic_relocation_count = resource->dynamic_count,
        .first_invocation = next_entry,
        .invocation_count = 1};
    char* name = tables->strings + next_name;
    memcpy(name, resource->kernel.data, resource->kernel.size);
    name[resource->kernel.size] = ':';
    memcpy(name + resource->kernel.size + 1, resource->sequence.data,
           resource->sequence.size);
    next_name += name_length;
    tables->invocations[next_entry++] =
        (iree_xdna_elf_invocation_record_t){.byte_length = section->size};
    tables->allocation_uses[next_use++] = allocation;
    uint32_t next_static = next_patch,
             next_dynamic = entry->first_dynamic_relocation;
    const uint32_t patch_count =
        resource->static_count + resource->dynamic_count;
    for (uint32_t j = 0; j < patch_count; ++j) {
      const iree_hal_amd_aie_patch_t* patch =
          &patches[resource->first_patch + j];
      if (patch->pdi) {
        iree_hal_amd_aie_resource_t* pdi = &resources[patch->pdi];
        if (pdi->owner != i) {
          pdi->owner = i;
          pdi->use = next_use - entry->first_allocation_use;
          tables->allocation_uses[next_use++] = pdi->backing - 1;
        }
        tables->relocations[next_static++] =
            (iree_xdna_elf_relocation_record_t){
                .byte_offset = patch->offset,
                .source_ordinal = pdi->use,
                .kind = IREE_HAL_AMD_XDNA_RELOCATION_FIRMWARE_ADDRESS_64,
                .maximum_value = UINT64_MAX,
                .alignment = 1};
      } else {
        tables->relocations[next_dynamic++] =
            (iree_xdna_elf_relocation_record_t){
                .byte_offset = patch->offset,
                .source_ordinal = patch->binding,
                .kind = IREE_XDNA_ELF_RELOCATION_KIND_SHIM_ADDRESS,
                .addend = (int64_t)patch->displacement,
                .maximum_value = (UINT64_C(1) << 48) - 1,
                .alignment = 4};
        iree_xdna_elf_binding_record_t* binding =
            &tables->bindings[next_binding + patch->binding];
        // The producer declares a pointer and DMA extent, without direction.
        // Both accesses are required for this unqualified storage pointer.
        *binding = (iree_xdna_elf_binding_record_t){
            .kind = IREE_XDNA_ELF_BINDING_KIND_BUFFER,
            .address_space = IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
            .access = IREE_XDNA_ELF_KNOWN_BINDING_ACCESS,
            .usage = IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
            .minimum_byte_length =
                iree_max(binding->minimum_byte_length,
                         patch->displacement + patch->transfer_size),
            .minimum_alignment = 4,
            .maximum_byte_offset = UINT64_MAX};
      }
    }
    entry->allocation_use_count = next_use - entry->first_allocation_use;
    next_patch += patch_count;
    next_binding += entry->binding_count;
  }
}

static iree_status_t iree_hal_amd_aie_read_groups(
    const iree_hal_amd_aie_sections_t* file,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_hal_amd_aie_resource_t* resources,
    iree_xdna_elf_header_record_t* counts) {
  bool has_partition = false;
  for (uint32_t i = 0; i < file->count; ++i) {
    const iree_hal_amd_aie_section_t* section = &file->sections[i];
    if (iree_string_view_equal(section->name,
                               IREE_SV(".note.xrt.configuration"))) {
      if (has_partition) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate AIE partition note");
      }
      IREE_RETURN_IF_ERROR(iree_hal_amd_aie_configuration(section, target));
      has_partition = true;
    } else if (section->type == IREE_HAL_AMD_AIE_GROUP) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amd_aie_group(file, section, target, resources, counts));
    } else if (section->type == IREE_HAL_AMD_AIE_RELA) {
      counts->relocation_count += section->size / 12;
    }
  }
  if (!has_partition || !counts->entry_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected standalone AIE ELF partition and kernel groups");
  }
  if (counts->relocation_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many AIE relocation records");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_image_read_aie_full_elf(
    iree_byte_sequence_t* source,
    const iree_hal_amd_xdna_aie2p_target_t* target, iree_allocator_t allocator,
    iree_hal_amd_xdna_image_tables_t* out_tables) {
  // The control program is opaque. Its relocation table describes address
  // fields and cannot certify the set of firmware operations it executes.
  if (!target->supports_load_pdi) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "standalone AIE execution requires a native XDNA "
                            "queue with firmware-addressed LOAD_PDI support");
  }
  iree_hal_amd_aie_sections_t file;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_aie_sections_initialize(source, allocator, &file));
  iree_hal_amd_aie_resource_t* resources = NULL;
  iree_hal_amd_aie_patch_t* patches = NULL;
  iree_xdna_elf_header_record_t counts = {
      .target_generation = IREE_XDNA_TARGET_GENERATION_AIE2P,
      .native_encoding = IREE_XDNA_ELF_NATIVE_TRANSACTION_0_1,
      .column_count = target->context.column_count,
      .row_count = target->context.row_count};
  iree_status_t status = iree_allocator_malloc(
      allocator, file.count * sizeof(*resources), (void**)&resources);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_aie_read_groups(&file, target, resources, &counts);
  }
  if (iree_status_is_ok(status) && counts.relocation_count) {
    status = iree_allocator_malloc_uninitialized(
        allocator, counts.relocation_count * sizeof(*patches),
        (void**)&patches);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_aie_index_patches(&file, resources, &counts, patches);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_tables_initialize(
        &counts, counts.allocation_count, allocator, out_tables);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_aie_build_plan(&file, target, resources, patches, out_tables);
  }
  iree_allocator_free(allocator, patches);
  iree_allocator_free(allocator, resources);
  iree_hal_amd_aie_sections_deinitialize(&file);
  return status;
}
