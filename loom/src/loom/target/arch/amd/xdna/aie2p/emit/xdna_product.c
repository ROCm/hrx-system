// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"

#include <string.h>

#include "loom/target/emit/native/elf.h"

enum {
  LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE = 16,
};

typedef struct loom_aie2p_xdna_payloads_t {
  iree_const_byte_span_t abi_note;
  iree_const_byte_span_t entries;
  iree_const_byte_span_t bindings;
  iree_const_byte_span_t relocations;
  iree_const_byte_span_t symbols;
  iree_const_byte_span_t strings;
  loom_xdna_elf_capabilities_t required_capabilities;
  uint16_t partition_column_count;
  uint16_t partition_row_count;
  bool has_relocations;
} loom_aie2p_xdna_payloads_t;

typedef struct loom_aie2p_xdna_entry_layout_t {
  // Source product entry.
  const loom_aie2p_xdna_entry_t* entry;
  // Final ARRAY, CONTROL, and relocation payloads.
  loom_aie2p_encoded_array_program_t encoded_program;
  // Loader capabilities needed by this entry alone.
  loom_xdna_elf_capabilities_t required_capabilities;
  // First entry tile in the flattened product tile table.
  iree_host_size_t first_tile_index;
  // Complete consecutive TILE program-header range length.
  iree_host_size_t tile_segment_count;
  // First TILE program-header ordinal owned by the entry.
  uint32_t first_tile_program_header_ordinal;
  // ARRAY program-header ordinal referenced by the entry table.
  uint32_t array_program_header_ordinal;
  // CONTROL program-header ordinal, or UINT32_MAX when absent.
  uint32_t control_program_header_ordinal;
  // First product-global binding ordinal owned by the entry.
  uint32_t first_binding_ordinal;
  // Candidate section index for the ARRAY payload.
  iree_host_size_t array_payload_index;
  // Candidate section index for the CONTROL payload when present.
  iree_host_size_t control_payload_index;
} loom_aie2p_xdna_entry_layout_t;

static void loom_aie2p_xdna_store_u16(uint8_t* target, uint16_t value) {
  iree_unaligned_store_le_u16(target, value);
}

static void loom_aie2p_xdna_store_u32(uint8_t* target, uint32_t value) {
  iree_unaligned_store_le_u32(target, value);
}

// Returns true when two placed sections can share one ELF file range.
//
// Section names are diagnostic labels and do not participate. Every load-time
// property and byte must match exactly.
static bool loom_aie2p_xdna_sections_identical(
    const loom_native_elf_section_t* lhs,
    const loom_native_elf_section_t* rhs) {
  if (!iree_all_bits_set(lhs->flags, LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC) ||
      !iree_all_bits_set(rhs->flags, LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC) ||
      lhs->type != rhs->type || lhs->flags != rhs->flags ||
      lhs->address != rhs->address || lhs->alignment != rhs->alignment ||
      lhs->entry_size != rhs->entry_size || lhs->link != rhs->link ||
      lhs->info != rhs->info ||
      lhs->contents.data_length != rhs->contents.data_length ||
      lhs->zero_fill_length != rhs->zero_fill_length) {
    return false;
  }
  return lhs->contents.data_length == 0 ||
         memcmp(lhs->contents.data, rhs->contents.data,
                lhs->contents.data_length) == 0;
}

static iree_host_size_t loom_aie2p_xdna_intern_linked_section(
    const loom_native_elf_section_t* section,
    const loom_native_elf_section_t** unique_sections,
    iree_host_size_t* unique_section_count) {
  for (iree_host_size_t i = 0; i < *unique_section_count; ++i) {
    if (loom_aie2p_xdna_sections_identical(section, unique_sections[i])) {
      return i;
    }
  }
  const iree_host_size_t section_index = (*unique_section_count)++;
  unique_sections[section_index] = section;
  return section_index;
}

static iree_status_t loom_aie2p_xdna_allocate_bytes(
    iree_arena_allocator_t* arena, iree_host_size_t byte_length,
    iree_byte_span_t* out_storage) {
  *out_storage = iree_byte_span_empty();
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, byte_length, (void**)&data));
  memset(data, 0, byte_length);
  *out_storage = iree_make_byte_span(data, byte_length);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_measure_partition(
    const loom_aie2p_array_plan_t* plan, uint16_t* out_column_count,
    uint16_t* out_row_count) {
  uint16_t column_count = 0;
#define LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(coordinate_value)         \
  do {                                                                  \
    const loom_xdna_tile_coordinate_t coordinate_ = (coordinate_value); \
    const uint16_t end_ = (uint16_t)coordinate_.column + 1u;            \
    if (end_ > column_count) column_count = end_;                       \
  } while (0)
  for (iree_host_size_t i = 0; i < plan->worker_plan_count; ++i) {
    LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(plan->worker_plans[i].coordinate);
  }
  for (iree_host_size_t i = 0; i < plan->channel_slot_count; ++i) {
    const loom_aie2p_array_channel_slot_t* slot = &plan->channel_slots[i];
    if (slot->sender_storage.owner.column != UINT16_MAX) {
      LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(slot->sender_storage.owner);
    }
    if (slot->receiver_storage.owner.column != UINT16_MAX) {
      LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(slot->receiver_storage.owner);
    }
  }
  for (iree_host_size_t i = 0; i < plan->lock_count; ++i) {
    LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(plan->locks[i].coordinate);
  }
  for (iree_host_size_t i = 0; i < plan->dma_channel_count; ++i) {
    LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(plan->dma_channels[i].coordinate);
  }
  for (iree_host_size_t i = 0; i < plan->route_count; ++i) {
    LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(plan->routes[i].coordinate);
  }
  for (iree_host_size_t i = 0; i < plan->binding_plan_count; ++i) {
    LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE(
        plan->binding_plans[i].shim_coordinate);
  }
#undef LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE
  if (column_count == 0 || column_count > plan->family->column_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P product has invalid partition columns");
  }
  *out_column_count = column_count;
  *out_row_count = plan->family->row_count;
  return iree_ok_status();
}

static loom_xdna_elf_binding_access_t loom_aie2p_xdna_binding_access(
    loom_aie2p_array_binding_access_t access) {
  loom_xdna_elf_binding_access_t result = 0;
  if (access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ ||
      access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ_WRITE) {
    result |= LOOM_XDNA_ELF_BINDING_ACCESS_READ;
  }
  if (access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE ||
      access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ_WRITE) {
    result |= LOOM_XDNA_ELF_BINDING_ACCESS_WRITE;
  }
  return result;
}

static iree_status_t loom_aie2p_xdna_build_binding_records(
    const loom_aie2p_array_plan_t* plan, uint32_t entry_ordinal,
    uint32_t first_binding_ordinal, loom_xdna_elf_binding_record_t* records) {
  if (plan->binding_count == 0) return iree_ok_status();
  memset(records, 0, plan->binding_count * sizeof(*records));
  for (iree_host_size_t i = 0; i < plan->binding_count; ++i) {
    const loom_aie2p_array_binding_t* binding = &plan->bindings[i];
    records[i] = (loom_xdna_elf_binding_record_t){
        .binding_ordinal = first_binding_ordinal + binding->ordinal,
        .entry_ordinal = entry_ordinal,
        .kind = LOOM_XDNA_ELF_BINDING_KIND_BUFFER,
        .address_space = LOOM_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
        .access = loom_aie2p_xdna_binding_access(binding->access),
        .usage = LOOM_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE |
                 LOOM_XDNA_ELF_BINDING_USAGE_COHERENT,
        .minimum_alignment = 1,
    };
  }
  for (iree_host_size_t i = 0; i < plan->binding_plan_count; ++i) {
    const loom_aie2p_array_binding_plan_t* binding = &plan->binding_plans[i];
    IREE_ASSERT_LT(binding->binding_index, plan->binding_count);
    uint64_t minimum_byte_length = 0;
    if (!iree_checked_add_u64(binding->binding_byte_offset,
                              binding->binding_span_byte_length,
                              &minimum_byte_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P binding extent overflows");
    }
    loom_xdna_elf_binding_record_t* record = &records[binding->binding_index];
    record->minimum_byte_length =
        iree_max(record->minimum_byte_length, minimum_byte_length);
    const loom_xdna_tile_facts_t* shim_tile = NULL;
    IREE_RETURN_IF_ERROR(loom_xdna_array_tile_facts(
        plan->family, binding->shim_coordinate, &shim_tile));
    record->minimum_alignment = iree_max(
        record->minimum_alignment, (uint64_t)shim_tile->dma.address_alignment);
  }
  for (iree_host_size_t i = 0; i < plan->binding_count; ++i) {
    IREE_ASSERT_EQ(records[i].binding_ordinal, first_binding_ordinal + i);
    IREE_ASSERT_NE(records[i].minimum_byte_length, 0u);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_encode_entry_table(
    const loom_aie2p_xdna_product_t* product,
    const loom_aie2p_xdna_entry_layout_t* layouts,
    iree_arena_allocator_t* arena, iree_const_byte_span_t* out_payload) {
  iree_host_size_t byte_length = LOOM_XDNA_ELF_TABLE_HEADER_SIZE;
  if (!iree_host_size_checked_add(
          byte_length, product->entry_count * LOOM_XDNA_ELF_ENTRY_RECORD_SIZE,
          &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P entry table exceeds ELF32");
  }
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    if (!iree_host_size_checked_add(byte_length, product->entries[i].name.size,
                                    &byte_length) ||
        product->entries[i].name.size > LOOM_XDNA_ELF_MAX_ENTRY_NAME_LENGTH) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P entry table exceeds ELF32");
    }
  }
  if (byte_length > UINT32_MAX ||
      byte_length > LOOM_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P entry table exceeds ELF32");
  }
  iree_byte_span_t storage;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_bytes(arena, byte_length, &storage));
  const uint32_t names_offset =
      LOOM_XDNA_ELF_TABLE_HEADER_SIZE +
      (uint32_t)product->entry_count * LOOM_XDNA_ELF_ENTRY_RECORD_SIZE;
  const loom_xdna_elf_table_header_t header = {
      .magic = LOOM_XDNA_ELF_ENTRY_TABLE_MAGIC,
      .abi_major = LOOM_XDNA_ELF_TABLE_ABI_MAJOR,
      .abi_minor = LOOM_XDNA_ELF_TABLE_ABI_MINOR,
      .header_size = LOOM_XDNA_ELF_TABLE_HEADER_SIZE,
      .record_size = LOOM_XDNA_ELF_ENTRY_RECORD_SIZE,
      .record_count = (uint32_t)product->entry_count,
      .byte_length = (uint32_t)byte_length,
      .auxiliary_offset = names_offset,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_table_header(
      &header,
      iree_make_byte_span(storage.data, LOOM_XDNA_ELF_TABLE_HEADER_SIZE)));
  uint32_t name_offset = names_offset;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    const loom_aie2p_xdna_entry_layout_t* layout = &layouts[i];
    const loom_xdna_elf_entry_record_t record = {
        .export_ordinal = (uint32_t)i,
        .name_offset = entry->name.size == 0 ? 0 : name_offset,
        .name_length = (uint32_t)entry->name.size,
        .array_program_header_ordinal = layout->array_program_header_ordinal,
        .control_program_header_ordinal =
            layout->control_program_header_ordinal,
        .first_binding_ordinal = layout->first_binding_ordinal,
        .binding_count = (uint32_t)entry->array_plan->binding_count,
        .flags = i == 0 ? LOOM_XDNA_ELF_ENTRY_FLAG_DEFAULT : 0,
        .required_capabilities = layout->required_capabilities,
    };
    IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_entry_record(
        &record,
        iree_make_byte_span(storage.data + LOOM_XDNA_ELF_TABLE_HEADER_SIZE +
                                i * LOOM_XDNA_ELF_ENTRY_RECORD_SIZE,
                            LOOM_XDNA_ELF_ENTRY_RECORD_SIZE)));
    memcpy(storage.data + name_offset, entry->name.data, entry->name.size);
    name_offset += (uint32_t)entry->name.size;
  }
  IREE_ASSERT_EQ(name_offset, byte_length);
  *out_payload = iree_make_const_byte_span(storage.data, storage.data_length);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_encode_binding_table(
    const loom_aie2p_xdna_product_t* product,
    const loom_aie2p_xdna_entry_layout_t* layouts,
    iree_host_size_t binding_count, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_payload) {
  if (binding_count > LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
      binding_count > (LOOM_XDNA_ELF_MAX_METADATA_TABLE_SIZE -
                       LOOM_XDNA_ELF_TABLE_HEADER_SIZE) /
                          LOOM_XDNA_ELF_BINDING_RECORD_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P binding table exceeds ELF32");
  }
  const iree_host_size_t byte_length =
      LOOM_XDNA_ELF_TABLE_HEADER_SIZE +
      binding_count * LOOM_XDNA_ELF_BINDING_RECORD_SIZE;
  iree_byte_span_t storage;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_bytes(arena, byte_length, &storage));
  const loom_xdna_elf_table_header_t header = {
      .magic = LOOM_XDNA_ELF_BINDING_TABLE_MAGIC,
      .abi_major = LOOM_XDNA_ELF_TABLE_ABI_MAJOR,
      .abi_minor = LOOM_XDNA_ELF_TABLE_ABI_MINOR,
      .header_size = LOOM_XDNA_ELF_TABLE_HEADER_SIZE,
      .record_size = LOOM_XDNA_ELF_BINDING_RECORD_SIZE,
      .record_count = (uint32_t)binding_count,
      .byte_length = (uint32_t)byte_length,
      .auxiliary_offset = 0,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_table_header(
      &header,
      iree_make_byte_span(storage.data, LOOM_XDNA_ELF_TABLE_HEADER_SIZE)));
  if (binding_count != 0) {
    loom_xdna_elf_binding_record_t* records = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, binding_count, sizeof(*records), (void**)&records));
    for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_aie2p_xdna_build_binding_records(
          product->entries[i].array_plan, (uint32_t)i,
          layouts[i].first_binding_ordinal,
          records + layouts[i].first_binding_ordinal));
    }
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_binding_record(
          &records[i],
          iree_make_byte_span(storage.data + LOOM_XDNA_ELF_TABLE_HEADER_SIZE +
                                  i * LOOM_XDNA_ELF_BINDING_RECORD_SIZE,
                              LOOM_XDNA_ELF_BINDING_RECORD_SIZE)));
    }
  }
  *out_payload = iree_make_const_byte_span(storage.data, storage.data_length);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_encode_relocation_table(
    const loom_aie2p_xdna_product_t* product,
    const loom_aie2p_xdna_entry_layout_t* layouts,
    iree_host_size_t relocation_count, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_payload) {
  *out_payload = iree_const_byte_span_empty();
  if (relocation_count == 0) return iree_ok_status();
  if (relocation_count > LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
      relocation_count > (LOOM_XDNA_ELF_MAX_METADATA_TABLE_SIZE -
                          LOOM_XDNA_ELF_TABLE_HEADER_SIZE) /
                             LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P relocation table exceeds ELF32");
  }
  const iree_host_size_t byte_length =
      LOOM_XDNA_ELF_TABLE_HEADER_SIZE +
      relocation_count * LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE;
  iree_byte_span_t storage;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_bytes(arena, byte_length, &storage));
  const loom_xdna_elf_table_header_t header = {
      .magic = LOOM_XDNA_ELF_RELOCATION_TABLE_MAGIC,
      .abi_major = LOOM_XDNA_ELF_TABLE_ABI_MAJOR,
      .abi_minor = LOOM_XDNA_ELF_TABLE_ABI_MINOR,
      .header_size = LOOM_XDNA_ELF_TABLE_HEADER_SIZE,
      .record_size = LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE,
      .record_count = (uint32_t)relocation_count,
      .byte_length = (uint32_t)byte_length,
      .auxiliary_offset = 0,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_table_header(
      &header,
      iree_make_byte_span(storage.data, LOOM_XDNA_ELF_TABLE_HEADER_SIZE)));
  iree_host_size_t relocation_index = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_layout_t* layout = &layouts[i];
    for (iree_host_size_t j = 0; j < layout->encoded_program.relocation_count;
         ++j) {
      loom_xdna_elf_relocation_record_t record =
          layout->encoded_program.relocations[j];
      IREE_ASSERT_LT(record.binding_ordinal,
                     layout->entry->array_plan->binding_count);
      record.binding_ordinal += layout->first_binding_ordinal;
      IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_relocation_record(
          &record,
          iree_make_byte_span(
              storage.data + LOOM_XDNA_ELF_TABLE_HEADER_SIZE +
                  relocation_index * LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE,
              LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE)));
      ++relocation_index;
    }
  }
  IREE_ASSERT_EQ(relocation_index, relocation_count);
  *out_payload = iree_make_const_byte_span(storage.data, storage.data_length);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_encode_symbol_tables(
    const loom_aie2p_xdna_tile_t* const* tiles, iree_host_size_t tile_count,
    const iree_host_size_t* code_section_indices, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_symbols, iree_const_byte_span_t* out_strings) {
  iree_host_size_t string_byte_length = 1;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution = tiles[i]->contribution;
    const iree_string_view_t name =
        contribution->object
            .symbols[contribution->realization.entry_symbol_index]
            .name;
    iree_host_size_t name_byte_length = 0;
    if (!iree_host_size_checked_add(name.size, 1, &name_byte_length) ||
        !iree_host_size_checked_add(string_byte_length, name_byte_length,
                                    &string_byte_length) ||
        string_byte_length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P product symbol names exceed ELF32");
    }
  }
  iree_byte_span_t strings;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_bytes(arena, string_byte_length, &strings));
  if (tile_count >
      (IREE_HOST_SIZE_MAX / LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE) - 1u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P product symbol table exceeds host size");
  }
  iree_byte_span_t symbols;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      arena, (tile_count + 1u) * LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE, &symbols));
  iree_host_size_t string_offset = 1;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    const loom_aie2p_leaf_contribution_t* contribution = tiles[i]->contribution;
    const loom_native_object_symbol_t* entry =
        &contribution->object
             .symbols[contribution->realization.entry_symbol_index];
    if (code_section_indices[i] >= UINT16_MAX || entry->size > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P product symbol exceeds ELF32");
    }
    const uint32_t name_offset = (uint32_t)string_offset;
    memcpy(strings.data + string_offset, entry->name.data, entry->name.size);
    string_offset += entry->name.size + 1u;
    uint8_t* record =
        symbols.data + (i + 1u) * LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE;
    loom_aie2p_xdna_store_u32(record + 0, name_offset);
    loom_aie2p_xdna_store_u32(record + 4, tiles[i]->linked_tile->entry_address);
    loom_aie2p_xdna_store_u32(record + 8, (uint32_t)entry->size);
    record[12] = 0x02;  // STB_LOCAL | STT_FUNC.
    record[13] = 0;
    loom_aie2p_xdna_store_u16(record + 14,
                              (uint16_t)(code_section_indices[i] + 1u));
  }
  IREE_ASSERT_EQ(string_offset, string_byte_length);
  *out_symbols = iree_make_const_byte_span(symbols.data, symbols.data_length);
  *out_strings = iree_make_const_byte_span(strings.data, strings.data_length);
  return iree_ok_status();
}

typedef struct loom_aie2p_xdna_inventory_t {
  // Per-entry resolved program-header and payload layout.
  loom_aie2p_xdna_entry_layout_t* layouts;
  // Product tiles flattened in entry and worker order.
  const loom_aie2p_xdna_tile_t** tiles;
  // Number of records in |tiles|.
  iree_host_size_t tile_count;
  // Total linked sections before content interning.
  iree_host_size_t linked_section_count;
  // Total entry-owned TILE program headers.
  iree_host_size_t tile_segment_count;
  // Total ARRAY and CONTROL program headers.
  iree_host_size_t program_segment_count;
  // Total product-global binding records.
  iree_host_size_t binding_count;
  // Total product-global relocation records.
  iree_host_size_t relocation_count;
} loom_aie2p_xdna_inventory_t;

static iree_status_t loom_aie2p_xdna_inventory_product(
    const loom_aie2p_xdna_product_t* product, iree_arena_allocator_t* arena,
    loom_aie2p_xdna_inventory_t* out_inventory) {
  *out_inventory = (loom_aie2p_xdna_inventory_t){0};
  if (product->device_profile == NULL || product->entry_count == 0 ||
      product->entries == NULL ||
      product->entry_count > LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P XDNA product inputs disagree");
  }

  loom_aie2p_xdna_entry_layout_t* layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, product->entry_count, sizeof(*layouts), (void**)&layouts));
  memset(layouts, 0, product->entry_count * sizeof(*layouts));
  const loom_xdna_array_family_t* product_family =
      loom_xdna_device_profile_array_family(product->device_profile);
  iree_host_size_t tile_count = 0;
  iree_host_size_t linked_section_count = 0;
  iree_host_size_t tile_segment_count = 0;
  iree_host_size_t program_segment_count = 0;
  iree_host_size_t binding_count = 0;
  iree_host_size_t relocation_count = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    if (entry->array_plan == NULL || entry->array_program == NULL ||
        entry->tile_count == 0 || entry->tiles == NULL ||
        entry->tile_count != entry->array_plan->worker_plan_count ||
        entry->tile_count != entry->array_program->tile_program_count ||
        entry->array_plan->family != product_family) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AIE2P XDNA product entry inputs disagree");
    }
    loom_aie2p_xdna_entry_layout_t* layout = &layouts[i];
    layout->entry = entry;
    layout->first_tile_index = tile_count;
    if (binding_count > LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AIE2P product binding table exceeds XDNA limits");
    }
    layout->first_binding_ordinal = (uint32_t)binding_count;
    if (entry->array_program->control_record_count != 0) {
      layout->required_capabilities |=
          LOOM_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS;
    }
    if (entry->array_program->relocation_count != 0) {
      layout->required_capabilities |=
          LOOM_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS;
    }

    iree_host_size_t entry_tile_segment_count = 0;
    for (iree_host_size_t j = 0; j < entry->tile_count; ++j) {
      const loom_aie2p_xdna_tile_t* tile = &entry->tiles[j];
      if (tile->contribution == NULL || tile->linked_tile == NULL ||
          tile->linked_tile->entry_section_index >=
              tile->linked_tile->assembly.section_count ||
          !iree_any_bit_set(
              tile->linked_tile->assembly
                  .sections[tile->linked_tile->entry_section_index]
                  .flags,
              LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "AIE2P product tile is incomplete");
      }
      if (!iree_host_size_checked_add(linked_section_count,
                                      tile->linked_tile->assembly.section_count,
                                      &linked_section_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AIE2P product section count overflows");
      }
      for (iree_host_size_t k = 0;
           k < tile->linked_tile->assembly.section_count; ++k) {
        const loom_native_elf_section_t* section =
            &tile->linked_tile->assembly.sections[k];
        if (!iree_any_bit_set(section->flags,
                              LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC)) {
          continue;
        }
        if (!iree_host_size_checked_add(entry_tile_segment_count, 1,
                                        &entry_tile_segment_count)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "AIE2P product segment count overflows");
        }
        if (section->type == LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS) {
          layout->required_capabilities |=
              LOOM_XDNA_ELF_CAPABILITY_TILE_ZERO_FILL;
        }
      }
    }
    layout->tile_segment_count = entry_tile_segment_count;
    if (!iree_host_size_checked_add(tile_count, entry->tile_count,
                                    &tile_count) ||
        !iree_host_size_checked_add(tile_segment_count,
                                    entry_tile_segment_count,
                                    &tile_segment_count) ||
        !iree_host_size_checked_add(
            program_segment_count,
            1u + (entry->array_program->control_record_count != 0),
            &program_segment_count) ||
        !iree_host_size_checked_add(
            binding_count, entry->array_plan->binding_count, &binding_count) ||
        !iree_host_size_checked_add(relocation_count,
                                    entry->array_program->relocation_count,
                                    &relocation_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P product cardinality overflows");
    }
  }

  const iree_host_size_t metadata_segment_count = 3u + (relocation_count != 0);
  iree_host_size_t program_header_count = 0;
  if (binding_count > LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
      relocation_count > LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
      tile_count >= UINT32_MAX || tile_segment_count >= UINT32_MAX ||
      !iree_host_size_checked_add(metadata_segment_count, tile_segment_count,
                                  &program_header_count) ||
      !iree_host_size_checked_add(program_header_count, program_segment_count,
                                  &program_header_count) ||
      program_header_count > LOOM_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P product exceeds XDNA ELF limits");
  }

  const loom_aie2p_xdna_tile_t** tiles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*tiles), (void**)&tiles));
  iree_host_size_t tile_index = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    for (iree_host_size_t j = 0; j < product->entries[i].tile_count; ++j) {
      tiles[tile_index++] = &product->entries[i].tiles[j];
    }
  }
  IREE_ASSERT_EQ(tile_index, tile_count);

  iree_host_size_t tile_program_header_ordinal = metadata_segment_count;
  iree_host_size_t array_program_header_ordinal =
      metadata_segment_count + tile_segment_count;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    loom_aie2p_xdna_entry_layout_t* layout = &layouts[i];
    layout->first_tile_program_header_ordinal =
        (uint32_t)tile_program_header_ordinal;
    tile_program_header_ordinal += layout->tile_segment_count;
    layout->array_program_header_ordinal =
        (uint32_t)array_program_header_ordinal++;
    if (layout->entry->array_program->control_record_count != 0) {
      layout->control_program_header_ordinal =
          (uint32_t)array_program_header_ordinal++;
    } else {
      layout->control_program_header_ordinal = UINT32_MAX;
    }
  }
  IREE_ASSERT_EQ(tile_program_header_ordinal,
                 metadata_segment_count + tile_segment_count);
  IREE_ASSERT_EQ(array_program_header_ordinal, program_header_count);

  *out_inventory = (loom_aie2p_xdna_inventory_t){
      .layouts = layouts,
      .tiles = tiles,
      .tile_count = tile_count,
      .linked_section_count = linked_section_count,
      .tile_segment_count = tile_segment_count,
      .program_segment_count = program_segment_count,
      .binding_count = binding_count,
      .relocation_count = relocation_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_prepare_payloads(
    const loom_aie2p_xdna_product_t* product,
    loom_aie2p_xdna_entry_layout_t* layouts, iree_host_size_t binding_count,
    iree_host_size_t relocation_count, iree_arena_allocator_t* arena,
    loom_aie2p_xdna_payloads_t* out_payloads) {
  *out_payloads = (loom_aie2p_xdna_payloads_t){0};
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    loom_aie2p_xdna_entry_layout_t* layout = &layouts[i];
    uint16_t column_count = 0;
    uint16_t row_count = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_xdna_measure_partition(
        layout->entry->array_plan, &column_count, &row_count));
    out_payloads->partition_column_count =
        iree_max(out_payloads->partition_column_count, column_count);
    out_payloads->partition_row_count =
        iree_max(out_payloads->partition_row_count, row_count);
    out_payloads->required_capabilities |= layout->required_capabilities;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_program_encode(
        layout->entry->array_program, layout->first_tile_program_header_ordinal,
        (uint32_t)layout->tile_segment_count,
        layout->control_program_header_ordinal, arena,
        &layout->encoded_program));
  }
  IREE_RETURN_IF_ERROR(loom_xdna_device_profile_validate_partition(
      product->device_profile, product->device_profile->physical_column_origin,
      out_payloads->partition_column_count));

  out_payloads->has_relocations = relocation_count != 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_encode_relocation_table(
      product, layouts, relocation_count, arena, &out_payloads->relocations));
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_encode_entry_table(
      product, layouts, arena, &out_payloads->entries));
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_encode_binding_table(
      product, layouts, binding_count, arena, &out_payloads->bindings));

  iree_byte_span_t abi_note;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      arena, LOOM_XDNA_ELF_ABI_NOTE_SIZE, &abi_note));
  const loom_xdna_elf_abi_note_t note = {
      .abi_major = LOOM_XDNA_ELF_ABI_MAJOR,
      .abi_minor = LOOM_XDNA_ELF_ABI_MINOR,
      .target_generation = LOOM_XDNA_TARGET_GENERATION_AIE2P,
      .device_profile_revision = product->device_profile->revision,
      .device_profile_id = product->device_profile->identity,
      .firmware_abi_id = product->device_profile->firmware_abi_identity,
      .policy_id = LOOM_AIE2P_XDNA_PRODUCT_POLICY_ID,
      .required_capabilities = out_payloads->required_capabilities,
      .partition_origin_column =
          product->device_profile->physical_column_origin,
      .partition_origin_row = 0,
      .partition_column_count = out_payloads->partition_column_count,
      .partition_row_count = out_payloads->partition_row_count,
      .coordinate_model = LOOM_XDNA_ELF_COORDINATE_MODEL_PARTITION_RELATIVE,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_elf_encode_abi_note(&note, abi_note));
  out_payloads->abi_note =
      iree_make_const_byte_span(abi_note.data, abi_note.data_length);
  return iree_ok_status();
}

static loom_native_elf_segment_t loom_aie2p_xdna_metadata_segment(
    uint32_t type, iree_host_size_t section_index, uint64_t byte_length,
    uint64_t alignment) {
  return (loom_native_elf_segment_t){
      .type = type,
      .flags = LOOM_XDNA_ELF_PROGRAM_FLAG_READ,
      .memory_size = byte_length,
      .first_section = section_index,
      .section_count = 1,
      .alignment = alignment,
  };
}

static iree_status_t loom_aie2p_xdna_tile_segment(
    const loom_native_elf_section_t* section,
    loom_xdna_tile_coordinate_t coordinate,
    loom_xdna_elf_tile_memory_space_t memory_space,
    iree_host_size_t section_index, loom_native_elf_segment_t* out_segment) {
  uint32_t physical_address = 0;
  IREE_RETURN_IF_ERROR(loom_xdna_elf_pack_tile_destination(
      &(loom_xdna_elf_tile_destination_t){
          .column = coordinate.column,
          .row = coordinate.row,
          .memory_space = memory_space,
          .flags = 0,
      },
      &physical_address));
  uint32_t flags = LOOM_XDNA_ELF_PROGRAM_FLAG_READ;
  if (iree_any_bit_set(section->flags, LOOM_NATIVE_ELF_SECTION_FLAG_WRITE)) {
    flags |= LOOM_XDNA_ELF_PROGRAM_FLAG_WRITE;
  }
  if (iree_any_bit_set(section->flags,
                       LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR)) {
    flags |= LOOM_XDNA_ELF_PROGRAM_FLAG_EXECUTE;
  }
  *out_segment = (loom_native_elf_segment_t){
      .type = LOOM_XDNA_ELF_PROGRAM_TYPE_TILE,
      .flags = flags,
      .memory_size = loom_native_elf_section_byte_length(section),
      .first_section = section_index,
      .section_count = 1,
      .virtual_address = section->address,
      .physical_address = physical_address,
      .alignment = section->alignment,
  };
  return iree_ok_status();
}

iree_status_t loom_aie2p_xdna_product_write(
    const loom_aie2p_xdna_product_t* product, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(product);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  loom_aie2p_xdna_inventory_t inventory;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_inventory_product(product, scratch_arena, &inventory));

  iree_host_size_t* tile_section_starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, inventory.tile_count + 1u, sizeof(*tile_section_starts),
      (void**)&tile_section_starts));
  iree_host_size_t linked_section_count = 0;
  for (iree_host_size_t i = 0; i < inventory.tile_count; ++i) {
    tile_section_starts[i] = linked_section_count;
    linked_section_count +=
        inventory.tiles[i]->linked_tile->assembly.section_count;
  }
  IREE_ASSERT_EQ(linked_section_count, inventory.linked_section_count);
  tile_section_starts[inventory.tile_count] = linked_section_count;

  const loom_native_elf_section_t** unique_linked_sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, linked_section_count, sizeof(*unique_linked_sections),
      (void**)&unique_linked_sections));
  iree_host_size_t* linked_section_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, linked_section_count, sizeof(*linked_section_indices),
      (void**)&linked_section_indices));
  iree_host_size_t unique_linked_section_count = 0;
  // Intern entry sections first so their ordering remains worker-stable.
  for (iree_host_size_t i = 0; i < inventory.tile_count; ++i) {
    const loom_aie2p_linked_tile_t* tile = inventory.tiles[i]->linked_tile;
    const iree_host_size_t linked_section_index =
        tile_section_starts[i] + tile->entry_section_index;
    linked_section_indices[linked_section_index] =
        loom_aie2p_xdna_intern_linked_section(
            &tile->assembly.sections[tile->entry_section_index],
            unique_linked_sections, &unique_linked_section_count);
  }
  for (iree_host_size_t i = 0; i < inventory.tile_count; ++i) {
    const loom_aie2p_linked_tile_t* tile = inventory.tiles[i]->linked_tile;
    for (iree_host_size_t j = 0; j < tile->assembly.section_count; ++j) {
      if (j == tile->entry_section_index) continue;
      linked_section_indices[tile_section_starts[i] + j] =
          loom_aie2p_xdna_intern_linked_section(&tile->assembly.sections[j],
                                                unique_linked_sections,
                                                &unique_linked_section_count);
    }
  }

  loom_aie2p_xdna_payloads_t payloads;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_prepare_payloads(
      product, inventory.layouts, inventory.binding_count,
      inventory.relocation_count, scratch_arena, &payloads));

  loom_native_elf_section_t* program_payload_sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, inventory.program_segment_count,
      sizeof(*program_payload_sections), (void**)&program_payload_sections));
  const loom_native_elf_section_t** unique_program_payload_sections = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, inventory.program_segment_count,
                                sizeof(*unique_program_payload_sections),
                                (void**)&unique_program_payload_sections));
  iree_host_size_t* program_payload_section_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, inventory.program_segment_count,
                                sizeof(*program_payload_section_indices),
                                (void**)&program_payload_section_indices));
  iree_host_size_t program_payload_index = 0;
  iree_host_size_t unique_program_payload_section_count = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    loom_aie2p_xdna_entry_layout_t* layout = &inventory.layouts[i];
    layout->array_payload_index = program_payload_index;
    program_payload_sections[program_payload_index] =
        (loom_native_elf_section_t){
            .name = IREE_SV(".xdna.array"),
            .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
            .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
            .alignment = 4,
            .contents = layout->encoded_program.array_payload,
        };
    program_payload_section_indices[program_payload_index] =
        loom_aie2p_xdna_intern_linked_section(
            &program_payload_sections[program_payload_index],
            unique_program_payload_sections,
            &unique_program_payload_section_count);
    ++program_payload_index;
    if (layout->control_program_header_ordinal != UINT32_MAX) {
      layout->control_payload_index = program_payload_index;
      program_payload_sections[program_payload_index] =
          (loom_native_elf_section_t){
              .name = IREE_SV(".xdna.control"),
              .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
              .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
              .alignment = 4,
              .contents = layout->encoded_program.control_payload,
          };
      program_payload_section_indices[program_payload_index] =
          loom_aie2p_xdna_intern_linked_section(
              &program_payload_sections[program_payload_index],
              unique_program_payload_sections,
              &unique_program_payload_section_count);
      ++program_payload_index;
    }
  }
  IREE_ASSERT_EQ(program_payload_index, inventory.program_segment_count);

  const iree_host_size_t metadata_segment_count = 3u + payloads.has_relocations;
  const iree_host_size_t section_count =
      metadata_segment_count + unique_linked_section_count +
      unique_program_payload_section_count + 2u;
  loom_native_elf_section_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, section_count, sizeof(*sections), (void**)&sections));
  iree_host_size_t* code_section_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, inventory.tile_count, sizeof(*code_section_indices),
      (void**)&code_section_indices));

  iree_host_size_t section_index = 0;
  sections[section_index++] = (loom_native_elf_section_t){
      .name = IREE_SV(".note.xdna.abi"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = 4,
      .contents = payloads.abi_note,
  };
  sections[section_index++] = (loom_native_elf_section_t){
      .name = IREE_SV(".xdna.entries"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = 8,
      .contents = payloads.entries,
  };
  sections[section_index++] = (loom_native_elf_section_t){
      .name = IREE_SV(".xdna.bindings"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      .alignment = 8,
      .contents = payloads.bindings,
  };
  if (payloads.has_relocations) {
    sections[section_index++] = (loom_native_elf_section_t){
        .name = IREE_SV(".xdna.relocations"),
        .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
        .flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
        .alignment = 8,
        .contents = payloads.relocations,
    };
  }
  const iree_host_size_t linked_section_base = section_index;
  for (iree_host_size_t i = 0; i < unique_linked_section_count; ++i) {
    sections[section_index++] = *unique_linked_sections[i];
  }
  for (iree_host_size_t i = 0; i < inventory.tile_count; ++i) {
    const loom_aie2p_linked_tile_t* tile = inventory.tiles[i]->linked_tile;
    code_section_indices[i] =
        linked_section_base + linked_section_indices[tile_section_starts[i] +
                                                     tile->entry_section_index];
  }
  const iree_host_size_t program_payload_section_base = section_index;
  for (iree_host_size_t i = 0; i < unique_program_payload_section_count; ++i) {
    sections[section_index++] = *unique_program_payload_sections[i];
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_encode_symbol_tables(
      inventory.tiles, inventory.tile_count, code_section_indices,
      scratch_arena, &payloads.symbols, &payloads.strings));
  const iree_host_size_t string_section_index = section_index + 1u;
  sections[section_index++] = (loom_native_elf_section_t){
      .name = IREE_SV(".symtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
      .alignment = 4,
      .entry_size = LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE,
      .link = (uint32_t)string_section_index + 1u,
      .info = (uint32_t)inventory.tile_count + 1u,
      .contents = payloads.symbols,
  };
  sections[section_index++] = (loom_native_elf_section_t){
      .name = IREE_SV(".strtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .alignment = 1,
      .contents = payloads.strings,
  };
  IREE_ASSERT_EQ(section_index, section_count);

  const iree_host_size_t segment_count = metadata_segment_count +
                                         inventory.tile_segment_count +
                                         inventory.program_segment_count;
  loom_native_elf_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, segment_count, sizeof(*segments), (void**)&segments));
  iree_host_size_t segment_index = 0;
  segments[segment_index++] = loom_aie2p_xdna_metadata_segment(
      LOOM_XDNA_ELF_PROGRAM_TYPE_NOTE, 0, payloads.abi_note.data_length, 4);
  segments[segment_index++] = loom_aie2p_xdna_metadata_segment(
      LOOM_XDNA_ELF_PROGRAM_TYPE_ENTRIES, 1, payloads.entries.data_length, 8);
  segments[segment_index++] = loom_aie2p_xdna_metadata_segment(
      LOOM_XDNA_ELF_PROGRAM_TYPE_BINDINGS, 2, payloads.bindings.data_length, 8);
  if (payloads.has_relocations) {
    segments[segment_index++] = loom_aie2p_xdna_metadata_segment(
        LOOM_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS, 3,
        payloads.relocations.data_length, 8);
  }
  for (iree_host_size_t entry_index = 0; entry_index < product->entry_count;
       ++entry_index) {
    const loom_aie2p_xdna_entry_layout_t* layout =
        &inventory.layouts[entry_index];
    IREE_ASSERT_EQ(segment_index, layout->first_tile_program_header_ordinal);
    const iree_host_size_t tile_end =
        layout->first_tile_index + layout->entry->tile_count;
    for (iree_host_size_t i = layout->first_tile_index; i < tile_end; ++i) {
      const loom_aie2p_linked_tile_t* tile = inventory.tiles[i]->linked_tile;
      const loom_native_elf_section_t* code =
          &tile->assembly.sections[tile->entry_section_index];
      IREE_RETURN_IF_ERROR(loom_aie2p_xdna_tile_segment(
          code, inventory.tiles[i]->coordinate,
          LOOM_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM, code_section_indices[i],
          &segments[segment_index++]));
    }
    for (iree_host_size_t i = layout->first_tile_index; i < tile_end; ++i) {
      const loom_aie2p_linked_tile_t* tile = inventory.tiles[i]->linked_tile;
      for (iree_host_size_t j = 0; j < tile->assembly.section_count; ++j) {
        if (j == tile->entry_section_index ||
            !iree_any_bit_set(tile->assembly.sections[j].flags,
                              LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC)) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_aie2p_xdna_tile_segment(
            &tile->assembly.sections[j], inventory.tiles[i]->coordinate,
            LOOM_XDNA_ELF_TILE_MEMORY_SPACE_DATA,
            linked_section_base +
                linked_section_indices[tile_section_starts[i] + j],
            &segments[segment_index++]));
      }
    }
    IREE_ASSERT_EQ(segment_index, layout->first_tile_program_header_ordinal +
                                      layout->tile_segment_count);
  }
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_layout_t* layout = &inventory.layouts[i];
    IREE_ASSERT_EQ(segment_index, layout->array_program_header_ordinal);
    const iree_host_size_t array_section_index =
        program_payload_section_base +
        program_payload_section_indices[layout->array_payload_index];
    segments[segment_index++] = loom_aie2p_xdna_metadata_segment(
        LOOM_XDNA_ELF_PROGRAM_TYPE_ARRAY, array_section_index,
        layout->encoded_program.array_payload.data_length, 4);
    if (layout->control_program_header_ordinal != UINT32_MAX) {
      IREE_ASSERT_EQ(segment_index, layout->control_program_header_ordinal);
      const iree_host_size_t control_section_index =
          program_payload_section_base +
          program_payload_section_indices[layout->control_payload_index];
      segments[segment_index++] = loom_aie2p_xdna_metadata_segment(
          LOOM_XDNA_ELF_PROGRAM_TYPE_CONTROL, control_section_index,
          layout->encoded_program.control_payload.data_length, 4);
    }
  }
  IREE_ASSERT_EQ(segment_index, segment_count);

  const loom_native_elf32le_file_t file = {
      .type = LOOM_XDNA_ELF_FILE_TYPE_EXEC,
      .machine = LOOM_XDNA_ELF_MACHINE_AIE,
      .os_abi = LOOM_XDNA_ELF_OS_ABI_NONE,
      .abi_version = LOOM_XDNA_ELF_ABI_VERSION_NONE,
      .flags = LOOM_XDNA_ELF_AIE2P_FLAGS,
      .entry = 0,
      .sections = sections,
      .section_count = section_count,
      .segments = segments,
      .segment_count = segment_count,
  };
  return loom_native_elf32le_write_file(&file, stream, scratch_arena);
}
