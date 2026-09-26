// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"

#include <string.h>

#include "iree/schemas/xdna_executable.h"
#include "loom/target/emit/native/elf.h"

enum {
  LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE = 16,
  LOOM_AIE2P_NATIVE_HEADER_SIZE = 16,
};

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

static uint16_t loom_aie2p_xdna_measure_partition(
    const loom_aie2p_array_plan_t* plan) {
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
#undef LOOM_AIE2P_XDNA_ACCUMULATE_COORDINATE
  return column_count;
}

static iree_xdna_elf_binding_access_t loom_aie2p_xdna_binding_access(
    loom_aie2p_array_binding_access_t access) {
  iree_xdna_elf_binding_access_t result = 0;
  if (access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ ||
      access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ_WRITE) {
    result |= IREE_XDNA_ELF_BINDING_ACCESS_READ;
  }
  if (access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE ||
      access == LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ_WRITE) {
    result |= IREE_XDNA_ELF_BINDING_ACCESS_WRITE;
  }
  return result;
}

static void loom_aie2p_xdna_build_binding_records(
    const loom_aie2p_array_plan_t* plan, uint32_t address_alignment,
    iree_xdna_elf_binding_record_t* records) {
  if (plan->binding_slot_count == 0) {
    return;
  }
  memset(records, 0, plan->binding_slot_count * sizeof(*records));
  for (iree_host_size_t i = 0; i < plan->binding_count; ++i) {
    const loom_aie2p_array_binding_t* binding = &plan->bindings[i];
    records[binding->ordinal] = (iree_xdna_elf_binding_record_t){
        .kind = IREE_XDNA_ELF_BINDING_KIND_BUFFER,
        .address_space = IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
        .access = loom_aie2p_xdna_binding_access(binding->access),
        .usage = IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE |
                 IREE_XDNA_ELF_BINDING_USAGE_COHERENT,
        .minimum_alignment = 1,
    };
  }
  for (iree_host_size_t i = 0; i < plan->binding_plan_count; ++i) {
    const loom_aie2p_array_binding_plan_t* binding_plan =
        &plan->binding_plans[i];
    const uint64_t minimum_byte_length = binding_plan->binding_byte_offset +
                                         binding_plan->binding_span_byte_length;
    const uint32_t binding_ordinal =
        plan->bindings[binding_plan->binding_index].ordinal;
    iree_xdna_elf_binding_record_t* record = &records[binding_ordinal];
    record->minimum_byte_length =
        iree_max(record->minimum_byte_length, minimum_byte_length);
    record->minimum_alignment = address_alignment;
  }
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

// A native transaction is emitted without expanding shared code payloads into
// the source buffer. ELF load ranges splice those bytes directly into final
// command backing. Offsets below retain the layout established by measurement.
typedef struct loom_aie2p_xdna_entry_layout_t {
  // First worker in the product's flattened tile table.
  uint32_t first_tile;
  // First global external binding contract.
  uint32_t first_binding;
  // First global dynamic relocation, covering initial and repeat invocations.
  uint32_t first_relocation;
  // Native operation bytes, excluding inline tile code.
  iree_byte_span_t source;
  // Offset of the shared control body in source storage.
  uint32_t control_source_offset;
  // Offset of the control body in initial command backing.
  uint32_t control_destination_offset;
  // Complete initial invocation byte length, including inline code.
  uint32_t initial_byte_length;
  // Aligned beginning of the repeat invocation in command backing.
  uint32_t repeat_offset;
  // Complete repeat invocation byte length.
  uint32_t repeat_byte_length;
  // Native byte offsets of control records relative to their shared body.
  uint32_t* control_record_offsets;
} loom_aie2p_xdna_entry_layout_t;

// Record cardinalities and word extents are established by array programming.
static uint32_t loom_aie2p_xdna_native_record_size(
    const loom_aie2p_program_record_t* record) {
  switch (record->type) {
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      return 24;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      return 28;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
      return 16 +
             (uint32_t)record->value.register_block_write32.word_count * 4u;
    case LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
    case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      return 16;
    default:
      IREE_ASSERT_UNREACHABLE("native program record kind");
      return 0;
  }
}

static void loom_aie2p_xdna_native_header(
    const loom_xdna_device_profile_t* profile,
    const loom_xdna_array_family_t* family, uint16_t columns,
    uint8_t memory_rows, uint32_t operation_count, uint32_t byte_length,
    uint8_t* storage) {
  storage[0] = 0;
  storage[1] = 1;
  storage[2] = profile->transaction_device_generation;
  storage[3] = (uint8_t)family->row_count;
  storage[4] = (uint8_t)columns;
  storage[5] = memory_rows;
  storage[6] = 0;
  storage[7] = 0;
  iree_unaligned_store_le_u32(storage + 8, operation_count);
  iree_unaligned_store_le_u32(storage + 12, byte_length);
}

static void loom_aie2p_xdna_native_block_header(
    const loom_xdna_array_family_t* family, uint32_t address,
    uint32_t byte_length, uint8_t* storage) {
  storage[0] = 1;
  storage[4] = (uint8_t)(address >> family->column_shift);
  const uint32_t row_mask =
      (1u << (family->column_shift - family->row_shift)) - 1u;
  storage[5] = (uint8_t)((address >> family->row_shift) & row_mask);
  iree_unaligned_store_le_u32(storage + 8, address);
  iree_unaligned_store_le_u32(storage + 12, byte_length);
}

// The caller owns one measured, zero-initialized native operation range.
static void loom_aie2p_xdna_native_record(
    const loom_xdna_array_family_t* family,
    const loom_aie2p_program_record_t* record, uint8_t* storage) {
  switch (record->type) {
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      iree_unaligned_store_le_u32(storage + 8,
                                  record->value.register_write32.address);
      iree_unaligned_store_le_u32(storage + 16,
                                  record->value.register_write32.value);
      iree_unaligned_store_le_u32(storage + 20, 24);
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      storage[0] = 3;
      iree_unaligned_store_le_u32(storage + 8,
                                  record->value.register_mask_write32.address);
      iree_unaligned_store_le_u32(storage + 16,
                                  record->value.register_mask_write32.value);
      iree_unaligned_store_le_u32(storage + 20,
                                  record->value.register_mask_write32.mask);
      iree_unaligned_store_le_u32(storage + 24, 28);
      break;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32: {
      const loom_aie2p_program_register_block_write32_t* block =
          &record->value.register_block_write32;
      loom_aie2p_xdna_native_block_header(
          family, block->address, loom_aie2p_xdna_native_record_size(record),
          storage);
      for (iree_host_size_t i = 0; i < block->word_count; ++i) {
        iree_unaligned_store_le_u32(storage + 16 + i * 4, block->words[i]);
      }
      break;
    }
    case LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT: {
      const loom_aie2p_program_dma_task_wait_t* wait =
          &record->value.dma_task_wait;
      storage[0] = 128;
      iree_unaligned_store_le_u32(storage + 4, 16);
      iree_unaligned_store_le_u32(
          storage + 8,
          (uint32_t)(wait->direction ==
                     LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM) |
              ((uint32_t)wait->coordinate.row << 8) |
              ((uint32_t)wait->coordinate.column << 16));
      iree_unaligned_store_le_u32(storage + 12,
                                  ((uint32_t)wait->row_count << 8) |
                                      ((uint32_t)wait->column_count << 16) |
                                      ((uint32_t)wait->dma_channel << 24));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("inline tile loads use linked sections");
  }
}

static iree_status_t loom_aie2p_xdna_measure_entry(
    const loom_aie2p_xdna_entry_t* entry, uint32_t alignment,
    iree_arena_allocator_t* arena, loom_aie2p_xdna_entry_layout_t* layout) {
  const loom_aie2p_array_program_t* program = entry->array_program;
  if (program->control_record_count == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA export requires a finite invocation drain");
  }
  uint64_t array_bytes = LOOM_AIE2P_NATIVE_HEADER_SIZE;
  uint64_t inline_bytes = 0;
  for (iree_host_size_t i = 0; i < program->array_record_count; ++i) {
    const loom_aie2p_program_record_t* record = &program->array_records[i];
    array_bytes += loom_aie2p_xdna_native_record_size(record);
    if (record->type == LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
      const loom_aie2p_linked_tile_t* tile =
          entry->tiles[record->value.tile_program_load.tile_program_index]
              .linked_tile;
      inline_bytes +=
          iree_host_align(tile->assembly.sections[tile->entry_section_index]
                              .contents.data_length,
                          4);
    }
  }
  uint64_t control_bytes = 0;
  for (iree_host_size_t i = 0; i < program->control_record_count; ++i) {
    control_bytes +=
        loom_aie2p_xdna_native_record_size(&program->control_records[i]);
  }
  const uint64_t initial_bytes = array_bytes + inline_bytes + control_bytes;
  const uint64_t repeat_offset = iree_align_uint64(initial_bytes, alignment);
  const uint64_t repeat_bytes = LOOM_AIE2P_NATIVE_HEADER_SIZE + control_bytes;
  if (repeat_offset + repeat_bytes > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "XDNA native command backing exceeds ELF32 offsets");
  }
  layout->control_source_offset = (uint32_t)array_bytes;
  layout->control_destination_offset = (uint32_t)(array_bytes + inline_bytes);
  layout->initial_byte_length = (uint32_t)initial_bytes;
  layout->repeat_offset = (uint32_t)repeat_offset;
  layout->repeat_byte_length = (uint32_t)repeat_bytes;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      arena, (iree_host_size_t)(array_bytes + repeat_bytes), &layout->source));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, program->control_record_count,
                                sizeof(*layout->control_record_offsets),
                                (void**)&layout->control_record_offsets));
  return iree_ok_status();
}

static loom_native_elf_segment_t loom_aie2p_xdna_load(uint32_t allocation,
                                                      uint32_t offset,
                                                      uint32_t size,
                                                      uint32_t section) {
  return (loom_native_elf_segment_t){
      .type = IREE_XDNA_ELF_PROGRAM_TYPE_LOAD,
      .flags = IREE_XDNA_ELF_PROGRAM_FLAG_READ,
      .memory_size = size,
      .first_section = section,
      .section_count = 1,
      .virtual_address = offset,
      .physical_address = allocation,
      .alignment = 1,
  };
}

static uint32_t loom_aie2p_xdna_append_fragment(
    iree_string_view_t name, const uint8_t* data, uint32_t size,
    loom_native_elf_section_t* sections, uint32_t* section_count) {
  const uint32_t ordinal = (*section_count)++;
  sections[ordinal] = (loom_native_elf_section_t){
      .name = name,
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .alignment = 4,
      .contents = iree_make_const_byte_span(data, size),
  };
  return ordinal;
}

static void loom_aie2p_xdna_emit_entry(
    const loom_aie2p_xdna_entry_t* entry, uint32_t allocation,
    const loom_xdna_device_profile_t* profile, uint16_t columns,
    uint8_t memory_rows, uint32_t program_load_base,
    const iree_host_size_t* code_sections,
    loom_aie2p_xdna_entry_layout_t* layout, loom_native_elf_section_t* sections,
    uint32_t* section_count, loom_native_elf_segment_t* segments,
    uint32_t* segment_count) {
  const loom_xdna_array_family_t* family = entry->array_plan->family;
  const loom_aie2p_array_program_t* program = entry->array_program;
  uint8_t* source = layout->source.data;
  loom_aie2p_xdna_native_header(
      profile, family, columns, memory_rows,
      (uint32_t)(program->array_record_count + program->control_record_count),
      layout->initial_byte_length, source);
  uint32_t source_offset = LOOM_AIE2P_NATIVE_HEADER_SIZE;
  uint32_t fragment_start = 0;
  uint32_t destination = 0;
  for (iree_host_size_t i = 0; i < program->array_record_count; ++i) {
    const loom_aie2p_program_record_t* record = &program->array_records[i];
    const uint32_t record_size = loom_aie2p_xdna_native_record_size(record);
    if (record->type == LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
      const uint32_t tile_index =
          record->value.tile_program_load.tile_program_index;
      const loom_aie2p_xdna_tile_t* tile = &entry->tiles[tile_index];
      const loom_native_elf_section_t* code =
          &tile->linked_tile->assembly
               .sections[tile->linked_tile->entry_section_index];
      const uint32_t code_size =
          (uint32_t)iree_host_align(code->contents.data_length, 4);
      const uint32_t address =
          ((uint32_t)tile->coordinate.column << family->column_shift) |
          ((uint32_t)tile->coordinate.row << family->row_shift) |
          (program_load_base + (uint32_t)code->address);
      loom_aie2p_xdna_native_block_header(family, address, 16 + code_size,
                                          source + source_offset);
      source_offset += record_size;
      const uint32_t fragment_size = source_offset - fragment_start;
      const uint32_t fragment = loom_aie2p_xdna_append_fragment(
          IREE_SV(".xdna.command"), source + fragment_start, fragment_size,
          sections, section_count);
      segments[(*segment_count)++] = loom_aie2p_xdna_load(
          allocation, destination, fragment_size, fragment);
      destination += fragment_size;
      segments[(*segment_count)++] =
          loom_aie2p_xdna_load(allocation, destination, code_size,
                               (uint32_t)code_sections[tile_index]);
      destination += code_size;
      fragment_start = source_offset;
    } else {
      loom_aie2p_xdna_native_record(family, record, source + source_offset);
      source_offset += record_size;
    }
  }
  if (source_offset != fragment_start) {
    const uint32_t size = source_offset - fragment_start;
    const uint32_t section = loom_aie2p_xdna_append_fragment(
        IREE_SV(".xdna.command"), source + fragment_start, size, sections,
        section_count);
    segments[(*segment_count)++] =
        loom_aie2p_xdna_load(allocation, destination, size, section);
  }
  uint32_t control_offset = 0;
  for (iree_host_size_t i = 0; i < program->control_record_count; ++i) {
    layout->control_record_offsets[i] = control_offset;
    loom_aie2p_xdna_native_record(
        family, &program->control_records[i],
        source + layout->control_source_offset + control_offset);
    control_offset +=
        loom_aie2p_xdna_native_record_size(&program->control_records[i]);
  }
  const uint32_t control_section = loom_aie2p_xdna_append_fragment(
      IREE_SV(".xdna.command"), source + layout->control_source_offset,
      control_offset, sections, section_count);
  segments[(*segment_count)++] =
      loom_aie2p_xdna_load(allocation, layout->control_destination_offset,
                           control_offset, control_section);
  uint8_t* repeat_header =
      source + layout->control_source_offset + control_offset;
  loom_aie2p_xdna_native_header(profile, family, columns, memory_rows,
                                (uint32_t)program->control_record_count,
                                layout->repeat_byte_length, repeat_header);
  const uint32_t repeat_section = loom_aie2p_xdna_append_fragment(
      IREE_SV(".xdna.command"), repeat_header, LOOM_AIE2P_NATIVE_HEADER_SIZE,
      sections, section_count);
  segments[(*segment_count)++] =
      loom_aie2p_xdna_load(allocation, layout->repeat_offset,
                           LOOM_AIE2P_NATIVE_HEADER_SIZE, repeat_section);
  segments[(*segment_count)++] = loom_aie2p_xdna_load(
      allocation, layout->repeat_offset + LOOM_AIE2P_NATIVE_HEADER_SIZE,
      control_offset, control_section);
}

iree_status_t loom_aie2p_xdna_product_write(
    const loom_aie2p_xdna_product_t* product, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  const loom_xdna_device_profile_t* profile = product->device_profile;
  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(profile);
  if (product->entry_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT / 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA product has too many entries");
  }
  uint32_t program_load_base = 0;
  uint32_t binding_alignment = 1;
  uint8_t memory_rows = 0;
  for (uint8_t i = 0; i < family->tile_count; ++i) {
    const loom_xdna_tile_facts_t* tile = &family->tiles[i];
    switch (tile->kind) {
      case LOOM_XDNA_TILE_KIND_COMPUTE:
        program_load_base = tile->memory.program_load_base;
        break;
      case LOOM_XDNA_TILE_KIND_MEMORY:
        memory_rows = tile->row_count;
        break;
      case LOOM_XDNA_TILE_KIND_SHIM_NOC:
        binding_alignment = tile->dma.address_alignment;
        break;
      default:
        break;
    }
  }

  loom_aie2p_xdna_entry_layout_t* layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, product->entry_count, sizeof(*layouts), (void**)&layouts));
  uint64_t tile_count = 0;
  uint64_t linked_section_count = 0;
  uint64_t binding_count = 0;
  uint64_t relocation_count = 0;
  uint64_t name_length = 0;
  uint16_t columns = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    layouts[i] = (loom_aie2p_xdna_entry_layout_t){
        .first_tile = (uint32_t)tile_count,
        .first_binding = (uint32_t)binding_count,
        .first_relocation = (uint32_t)relocation_count,
    };
    tile_count += entry->tile_count;
    binding_count += entry->array_plan->binding_slot_count;
    relocation_count += 2 * entry->array_program->relocation_count;
    name_length += entry->name.size;
    for (iree_host_size_t j = 0; j < entry->tile_count; ++j) {
      linked_section_count +=
          entry->tiles[j].linked_tile->assembly.section_count;
    }
    columns =
        iree_max(columns, loom_aie2p_xdna_measure_partition(entry->array_plan));
    if (tile_count > IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT ||
        linked_section_count > IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT ||
        binding_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
        relocation_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT ||
        entry->name.size > IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA product exceeds image table limits");
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_xdna_measure_entry(
        entry, profile->limits.instruction_address_alignment, scratch_arena,
        &layouts[i]));
  }
  IREE_RETURN_IF_ERROR(loom_xdna_device_profile_validate_partition(
      profile, profile->physical_column_origin, columns));

  const loom_aie2p_xdna_tile_t** tiles = NULL;
  const loom_native_elf_section_t** unique_sections = NULL;
  iree_host_size_t* code_sections = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, (iree_host_size_t)tile_count,
                                sizeof(*tiles), (void**)&tiles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, (iree_host_size_t)tile_count, sizeof(*code_sections),
      (void**)&code_sections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, (iree_host_size_t)linked_section_count,
      sizeof(*unique_sections), (void**)&unique_sections));
  iree_host_size_t unique_section_count = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    for (iree_host_size_t j = 0; j < entry->tile_count; ++j) {
      const iree_host_size_t index = layouts[i].first_tile + j;
      tiles[index] = &entry->tiles[j];
      const loom_aie2p_linked_tile_t* linked = tiles[index]->linked_tile;
      code_sections[index] =
          1 + loom_aie2p_xdna_intern_linked_section(
                  &linked->assembly.sections[linked->entry_section_index],
                  unique_sections, &unique_section_count);
    }
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    const loom_aie2p_linked_tile_t* linked = tiles[i]->linked_tile;
    for (iree_host_size_t j = 0; j < linked->assembly.section_count; ++j) {
      if (j == linked->entry_section_index) {
        continue;
      }
      loom_aie2p_xdna_intern_linked_section(&linked->assembly.sections[j],
                                            unique_sections,
                                            &unique_section_count);
    }
  }
  const uint64_t section_capacity =
      3 + unique_section_count + tile_count + 4 * product->entry_count;
  const uint64_t segment_capacity =
      1 + 2 * tile_count + 4 * product->entry_count;
  if (section_capacity + 2 > IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT ||
      segment_capacity > IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA product exceeds ELF directory limits");
  }
  loom_native_elf_section_t* sections = NULL;
  loom_native_elf_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, (iree_host_size_t)section_capacity, sizeof(*sections),
      (void**)&sections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, (iree_host_size_t)segment_capacity, sizeof(*segments),
      (void**)&segments));

  const uint32_t entry_count = (uint32_t)product->entry_count;
  const uint32_t allocation_offset = IREE_XDNA_ELF_HEADER_RECORD_SIZE;
  const uint32_t use_offset =
      allocation_offset + entry_count * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE;
  const uint32_t entry_offset = use_offset + entry_count * sizeof(uint32_t);
  const uint32_t binding_offset =
      entry_offset + entry_count * IREE_XDNA_ELF_ENTRY_RECORD_SIZE;
  const uint32_t relocation_offset =
      binding_offset +
      (uint32_t)binding_count * IREE_XDNA_ELF_BINDING_RECORD_SIZE;
  const uint32_t invocation_offset =
      relocation_offset +
      (uint32_t)relocation_count * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE;
  const uint32_t string_offset =
      invocation_offset +
      2 * entry_count * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE;
  if (string_offset + name_length > IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA metadata exceeds its byte limit");
  }
  iree_byte_span_t metadata;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      scratch_arena, string_offset + (iree_host_size_t)name_length, &metadata));
  const iree_xdna_elf_header_record_t header = {
      .magic = IREE_XDNA_ELF_METADATA_MAGIC,
      .version = IREE_XDNA_ELF_METADATA_VERSION,
      .native_encoding = IREE_XDNA_ELF_NATIVE_TRANSACTION_0_1,
      .target_generation = IREE_XDNA_TARGET_GENERATION_AIE2P,
      .device_profile_revision = profile->revision,
      .device_profile_id = profile->identity,
      .firmware_abi_id = profile->firmware_abi_identity,
      .column_count = columns,
      .row_count = family->row_count,
      .allocation_count = entry_count,
      .allocation_use_count = entry_count,
      .entry_count = entry_count,
      .binding_count = (uint32_t)binding_count,
      .relocation_count = (uint32_t)relocation_count,
      .invocation_count = 2 * entry_count,
      .string_byte_length = (uint32_t)name_length,
  };
  iree_xdna_elf_encode_header(&header, metadata.data);
  uint32_t section_count = 0;
  loom_aie2p_xdna_append_fragment(IREE_SV(".xdna.metadata"), metadata.data,
                                  (uint32_t)metadata.data_length, sections,
                                  &section_count);
  for (iree_host_size_t i = 0; i < unique_section_count; ++i) {
    sections[section_count++] = *unique_sections[i];
  }
  uint32_t segment_count = 1;
  segments[0] = (loom_native_elf_segment_t){
      .type = IREE_XDNA_ELF_PROGRAM_TYPE_METADATA,
      .flags = IREE_XDNA_ELF_PROGRAM_FLAG_READ,
      .first_section = 0,
      .section_count = 1,
      .alignment = 1,
  };
  uint32_t name_offset = 0;
  for (uint32_t i = 0; i < entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    loom_aie2p_xdna_entry_layout_t* layout = &layouts[i];
    const uint32_t first_load = segment_count;
    loom_aie2p_xdna_emit_entry(
        entry, i, profile, columns, memory_rows, program_load_base,
        code_sections + layout->first_tile, layout, sections, &section_count,
        segments, &segment_count);
    const iree_xdna_elf_allocation_record_t allocation = {
        .domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND,
        .byte_length =
            (uint64_t)layout->repeat_offset + layout->repeat_byte_length,
        .alignment = profile->limits.instruction_address_alignment,
        .first_load = first_load,
        .load_count = segment_count - first_load,
    };
    iree_xdna_elf_encode_allocation(
        &allocation, metadata.data + allocation_offset +
                         i * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE);
    iree_unaligned_store_le_u32(
        metadata.data + use_offset + i * sizeof(uint32_t), i);
    const iree_xdna_elf_entry_record_t entry_record = {
        .name_offset = name_offset,
        .name_length = (uint32_t)entry->name.size,
        .first_allocation_use = i,
        .allocation_use_count = 1,
        .first_binding = layout->first_binding,
        .binding_count = entry->array_plan->binding_slot_count,
        .first_static_relocation = layout->first_relocation,
        .first_dynamic_relocation = layout->first_relocation,
        .dynamic_relocation_count =
            (uint32_t)(2 * entry->array_program->relocation_count),
        .first_invocation = 2 * i,
        .invocation_count = 2,
    };
    iree_xdna_elf_encode_entry(
        &entry_record,
        metadata.data + entry_offset + i * IREE_XDNA_ELF_ENTRY_RECORD_SIZE);
    memcpy(metadata.data + string_offset + name_offset, entry->name.data,
           entry->name.size);
    name_offset += (uint32_t)entry->name.size;
    iree_xdna_elf_binding_record_t* bindings = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, entry_record.binding_count,
                                  sizeof(*bindings), (void**)&bindings));
    loom_aie2p_xdna_build_binding_records(entry->array_plan, binding_alignment,
                                          bindings);
    for (uint32_t j = 0; j < entry_record.binding_count; ++j) {
      iree_xdna_elf_encode_binding(
          &bindings[j],
          metadata.data + binding_offset +
              (layout->first_binding + j) * IREE_XDNA_ELF_BINDING_RECORD_SIZE);
    }
    for (uint32_t invocation = 0; invocation < 2; ++invocation) {
      const uint32_t body_offset =
          invocation == 0
              ? layout->control_destination_offset
              : layout->repeat_offset + LOOM_AIE2P_NATIVE_HEADER_SIZE;
      for (uint32_t j = 0; j < entry->array_program->relocation_count; ++j) {
        const loom_aie2p_program_relocation_t* source =
            &entry->array_program->relocations[j];
        const iree_xdna_elf_relocation_record_t relocation = {
            .destination_use = 0,
            .source_ordinal = source->binding_ordinal,
            .byte_offset =
                body_offset +
                layout->control_record_offsets[source->target_record_index] +
                16 + source->target_word_index * 4,
            .kind = IREE_XDNA_ELF_RELOCATION_KIND_SHIM_ADDRESS,
            .addend = source->addend,
            .minimum_value = source->minimum_value,
            .maximum_value = source->maximum_value,
            .alignment = source->required_alignment,
        };
        const uint32_t ordinal =
            layout->first_relocation +
            invocation * (uint32_t)entry->array_program->relocation_count + j;
        iree_xdna_elf_encode_relocation(
            &relocation, metadata.data + relocation_offset +
                             ordinal * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE);
      }
      const iree_xdna_elf_invocation_record_t range = {
          .allocation_use = 0,
          .byte_offset = invocation == 0 ? 0 : layout->repeat_offset,
          .byte_length = invocation == 0 ? layout->initial_byte_length
                                         : layout->repeat_byte_length,
          .next_invocation = 1,
      };
      iree_xdna_elf_encode_invocation(
          &range,
          metadata.data + invocation_offset +
              (2 * i + invocation) * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE);
    }
  }
  iree_const_byte_span_t symbols;
  iree_const_byte_span_t strings;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_encode_symbol_tables(
      tiles, (iree_host_size_t)tile_count, code_sections, scratch_arena,
      &symbols, &strings));
  sections[section_count] = (loom_native_elf_section_t){
      .name = IREE_SV(".symtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
      .alignment = 4,
      .entry_size = LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE,
      .link = section_count + 2,
      .info = (uint32_t)tile_count + 1,
      .contents = symbols,
  };
  ++section_count;
  sections[section_count++] = (loom_native_elf_section_t){
      .name = IREE_SV(".strtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .alignment = 1,
      .contents = strings,
  };
  const loom_native_elf32le_file_t file = {
      .type = LOOM_NATIVE_ELF_FILE_TYPE_EXEC,
      .machine = LOOM_NATIVE_ELF_MACHINE_AIE,
      .flags = IREE_XDNA_ELF_AIE2P_FLAGS,
      .sections = sections,
      .section_count = section_count,
      .segments = segments,
      .segment_count = segment_count,
  };
  return loom_native_elf32le_write_file(&file, stream, scratch_arena);
}
