// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"

#include <string.h>

#include "iree/schemas/xdna_executable.h"
#include "loom/target/emit/native/elf_sections.h"

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
    const loom_native_section_t* lhs, const loom_native_section_t* rhs) {
  if (lhs->access == LOOM_NATIVE_SECTION_ACCESS_NONE ||
      rhs->access == LOOM_NATIVE_SECTION_ACCESS_NONE ||
      lhs->storage != rhs->storage || lhs->access != rhs->access ||
      lhs->address != rhs->address || lhs->alignment != rhs->alignment ||
      lhs->contents.data_length != rhs->contents.data_length ||
      lhs->reservation_length != rhs->reservation_length) {
    return false;
  }
  return lhs->contents.data_length == 0 ||
         memcmp(lhs->contents.data, rhs->contents.data,
                lhs->contents.data_length) == 0;
}

static iree_host_size_t loom_aie2p_xdna_intern_linked_section(
    const loom_native_section_t* section,
    const loom_native_section_t** unique_sections,
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

static bool loom_aie2p_xdna_section_requires_load(
    const loom_native_section_t* section) {
  return section->storage == LOOM_NATIVE_SECTION_STORAGE_CONTENTS &&
         section->access != LOOM_NATIVE_SECTION_ACCESS_NONE &&
         section->contents.data_length != 0;
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

// Final product facts and section indices for one linked resident tile.
typedef struct loom_aie2p_xdna_tile_layout_t {
  // Borrowed linked resident tile.
  const loom_aie2p_xdna_tile_t* tile;
  // File section indices in linked assembly-section order.
  iree_host_size_t* file_section_indices;
  // Number of initialized sections loaded by the native transaction.
  uint32_t loadable_section_count;
  // Aligned payload bytes loaded by the native transaction.
  uint32_t inline_byte_length;
} loom_aie2p_xdna_tile_layout_t;
static iree_status_t loom_aie2p_xdna_encode_symbol_tables(
    const loom_aie2p_xdna_tile_layout_t* tiles, iree_host_size_t tile_count,
    iree_host_size_t string_byte_length, iree_arena_allocator_t* arena,
    iree_const_byte_span_t* out_symbols, iree_const_byte_span_t* out_strings) {
  iree_byte_span_t strings;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_bytes(arena, string_byte_length, &strings));
  IREE_ASSERT_LE(tile_count,
                 (IREE_HOST_SIZE_MAX / LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE) - 1u);
  iree_byte_span_t symbols;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      arena, (tile_count + 1u) * LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE, &symbols));
  iree_host_size_t string_offset = 1;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    const loom_aie2p_xdna_tile_t* tile = tiles[i].tile;
    const loom_aie2p_leaf_contribution_t* contribution = tile->contribution;
    const loom_native_object_symbol_t* entry =
        &contribution->object
             .symbols[contribution->realization.entry_symbol_index];
    const iree_host_size_t code_section_index =
        tiles[i].file_section_indices[tile->linked_tile->entry_section_index];
    IREE_ASSERT_LT(code_section_index, UINT16_MAX);
    IREE_ASSERT_LE(entry->size, UINT32_MAX);
    const uint32_t name_offset = (uint32_t)string_offset;
    memcpy(strings.data + string_offset, tile->program_name.data,
           tile->program_name.size);
    string_offset += tile->program_name.size + 1u;
    uint8_t* record =
        symbols.data + (i + 1u) * LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE;
    loom_aie2p_xdna_store_u32(record + 0, name_offset);
    loom_aie2p_xdna_store_u32(record + 4, tile->linked_tile->entry_address);
    loom_aie2p_xdna_store_u32(record + 8, (uint32_t)entry->size);
    record[12] = 0x02;  // STB_LOCAL | STT_FUNC.
    record[13] = 0;
    loom_aie2p_xdna_store_u16(record + 14, (uint16_t)(code_section_index + 1u));
  }
  IREE_ASSERT_EQ(string_offset, string_byte_length);
  *out_symbols = iree_make_const_byte_span(symbols.data, symbols.data_length);
  *out_strings = iree_make_const_byte_span(strings.data, strings.data_length);
  return iree_ok_status();
}

// A native transaction is emitted without expanding shared linked payloads
// into the source buffer. ELF load ranges splice those bytes directly into
// final command backing. Offsets below retain the layout established by
// measurement.
typedef struct loom_aie2p_xdna_entry_layout_t {
  // First worker in the product's flattened tile table.
  uint32_t first_tile;
  // First global external binding contract.
  uint32_t first_binding;
  // First global dynamic relocation, covering initial and repeat invocations.
  uint32_t first_relocation;
  // Native operation bytes, excluding inline linked payloads.
  iree_byte_span_t source;
  // Expanded array initialization operation count.
  uint32_t array_operation_count;
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

struct loom_aie2p_xdna_product_admission_t {
  // Borrowed product whose source-known facts were admitted.
  const loom_aie2p_xdna_product_t* product;
  // Arena receiving retained admission and final image storage.
  iree_arena_allocator_t* arena;
  // Per-entry retained prefixes and final native command layouts.
  loom_aie2p_xdna_entry_layout_t* entry_layouts;
  // Flattened resident tile count across every entry.
  iree_host_size_t tile_count;
  // Aggregate runtime binding-record count.
  uint32_t binding_count;
  // Aggregate runtime relocation-record count.
  uint32_t relocation_count;
  // Aggregate exported entry-name byte length.
  uint32_t entry_name_byte_length;
  // Occupied array-column prefix across all entries.
  uint16_t partition_column_count;
  // Memory-tile row count encoded in native transaction headers.
  uint8_t memory_row_count;
  // Metadata allocation-record table offset.
  uint32_t allocation_offset;
  // Metadata allocation-use table offset.
  uint32_t use_offset;
  // Metadata entry-record table offset.
  uint32_t entry_offset;
  // Metadata binding-record table offset.
  uint32_t binding_offset;
  // Metadata relocation-record table offset.
  uint32_t relocation_offset;
  // Metadata invocation-record table offset.
  uint32_t invocation_offset;
  // Metadata entry-name string table offset.
  uint32_t string_offset;
  // Complete metadata byte length.
  uint32_t metadata_byte_length;
};

struct loom_aie2p_xdna_product_image_t {
  // Final ELF32LE file description.
  loom_native_elf32le_file_t file;
  // Final placement and generated section-name table.
  loom_native_elf_layout_t layout;
};

// Record cardinalities and word extents are established by array programming.
static uint32_t loom_aie2p_xdna_native_record_size(
    const loom_aie2p_program_record_t* record) {
  switch (record->type) {
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      return 24;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      return 28;
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32:
      return 32;
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
    case LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32:
      storage[0] = 4;
      iree_unaligned_store_le_u32(storage + 8,
                                  record->value.register_mask_wait32.address);
      iree_unaligned_store_le_u32(storage + 16,
                                  record->value.register_mask_wait32.value);
      iree_unaligned_store_le_u32(storage + 20,
                                  record->value.register_mask_wait32.mask);
      iree_unaligned_store_le_u32(storage + 24, 32);
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
          storage + 8, (uint32_t)(wait->direction ==
                                  LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM) |
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

static void loom_aie2p_xdna_set_issue(
    loom_aie2p_xdna_product_issue_kind_t kind, uint32_t entry_ordinal,
    uint64_t actual, uint64_t minimum, uint64_t maximum,
    loom_aie2p_xdna_product_issue_t* out_issue) {
  *out_issue = (loom_aie2p_xdna_product_issue_t){
      .kind = kind,
      .entry_ordinal = entry_ordinal,
      .actual = actual,
      .minimum = minimum,
      .maximum = maximum,
  };
}

static loom_aie2p_xdna_product_issue_kind_t loom_aie2p_xdna_measure_entry(
    const loom_aie2p_xdna_entry_t* entry,
    const loom_aie2p_xdna_tile_layout_t* tile_layouts, uint32_t alignment,
    loom_aie2p_xdna_entry_layout_t* layout, uint64_t* out_command_section_count,
    uint64_t* out_segment_count, uint64_t* out_actual) {
  const loom_aie2p_array_program_t* program = entry->array_program;
  uint64_t array_bytes = LOOM_AIE2P_NATIVE_HEADER_SIZE;
  uint64_t inline_bytes = 0;
  uint64_t array_operation_count = 0;
  uint64_t loadable_section_count = 0;
  bool has_trailing_fragment = true;
  for (iree_host_size_t i = 0; i < program->array_record_count; ++i) {
    const loom_aie2p_program_record_t* record = &program->array_records[i];
    if (record->type == LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
      const uint32_t tile_index =
          record->value.tile_program_load.tile_program_index;
      const loom_aie2p_xdna_tile_layout_t* tile_layout =
          &tile_layouts[tile_index];
      array_bytes += LOOM_AIE2P_NATIVE_HEADER_SIZE *
                     (uint64_t)tile_layout->loadable_section_count;
      inline_bytes += tile_layout->inline_byte_length;
      array_operation_count += tile_layout->loadable_section_count;
      loadable_section_count += tile_layout->loadable_section_count;
      if (tile_layout->loadable_section_count != 0) {
        has_trailing_fragment = false;
      }
    } else {
      array_bytes += loom_aie2p_xdna_native_record_size(record);
      ++array_operation_count;
      has_trailing_fragment = true;
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
  const uint64_t native_byte_length = repeat_offset + repeat_bytes;
  if (native_byte_length > UINT32_MAX) {
    *out_actual = native_byte_length;
    return LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NATIVE_COMMAND_BYTE_LENGTH;
  }
  const uint64_t operation_count =
      array_operation_count + program->control_record_count;
  if (operation_count > UINT32_MAX) {
    *out_actual = operation_count;
    return LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NATIVE_OPERATION_COUNT;
  }
  layout->array_operation_count = (uint32_t)array_operation_count;
  layout->control_source_offset = (uint32_t)array_bytes;
  layout->control_destination_offset = (uint32_t)(array_bytes + inline_bytes);
  layout->initial_byte_length = (uint32_t)initial_bytes;
  layout->repeat_offset = (uint32_t)repeat_offset;
  layout->repeat_byte_length = (uint32_t)repeat_bytes;
  const uint64_t has_control = program->control_record_count != 0 ? 1u : 0u;
  *out_command_section_count = loadable_section_count +
                               (has_trailing_fragment ? 1u : 0u) + has_control +
                               1u;
  *out_segment_count = 2u * loadable_section_count +
                       (has_trailing_fragment ? 1u : 0u) + 2u * has_control +
                       1u;
  return LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NONE;
}

static iree_status_t loom_aie2p_xdna_allocate_entry_storage(
    const loom_aie2p_xdna_entry_t* entry, iree_arena_allocator_t* arena,
    loom_aie2p_xdna_entry_layout_t* layout) {
  const uint64_t source_byte_length =
      (uint64_t)layout->control_source_offset + layout->repeat_byte_length;
  IREE_ASSERT_LE(source_byte_length, UINT32_MAX);
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      arena, (iree_host_size_t)source_byte_length, &layout->source));
  const loom_aie2p_array_program_t* program = entry->array_program;
  if (program->control_record_count != 0 || program->measurement.present) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, program->control_record_count + 1,
                                  sizeof(*layout->control_record_offsets),
                                  (void**)&layout->control_record_offsets));
  }
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
    const loom_xdna_device_profile_t* profile,
    const loom_xdna_array_family_t* family, uint16_t columns,
    uint8_t memory_rows, const loom_aie2p_xdna_tile_layout_t* tile_layouts,
    loom_aie2p_xdna_entry_layout_t* layout, loom_native_elf_section_t* sections,
    uint32_t* section_count, loom_native_elf_segment_t* segments,
    uint32_t* segment_count) {
  const loom_aie2p_array_program_t* program = entry->array_program;
  uint8_t* source = layout->source.data;
  loom_aie2p_xdna_native_header(
      profile, family, columns, memory_rows,
      layout->array_operation_count + (uint32_t)program->control_record_count,
      layout->initial_byte_length, source);
  uint32_t source_offset = LOOM_AIE2P_NATIVE_HEADER_SIZE;
  uint32_t fragment_start = 0;
  uint32_t destination = 0;
  for (iree_host_size_t i = 0; i < program->array_record_count; ++i) {
    const loom_aie2p_program_record_t* record = &program->array_records[i];
    if (record->type == LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
      const uint32_t tile_index =
          record->value.tile_program_load.tile_program_index;
      const loom_aie2p_xdna_tile_t* tile = &entry->tiles[tile_index];
      const loom_aie2p_linked_tile_t* linked = tile->linked_tile;
      IREE_ASSERT_EQ(linked->section_placement_count,
                     linked->assembly.section_count);
      const loom_xdna_tile_facts_t* tile_facts =
          loom_xdna_array_tile_facts(family, tile->coordinate);
      for (iree_host_size_t section_index = 0;
           section_index < linked->assembly.section_count; ++section_index) {
        const loom_native_section_t* section =
            &linked->assembly.sections[section_index];
        if (!loom_aie2p_xdna_section_requires_load(section)) {
          continue;
        }
        const uint32_t payload_size =
            (uint32_t)iree_host_align(section->contents.data_length, 4);
        const loom_aie2p_linked_section_placement_t* placement =
            &linked->section_placements[section_index];
        uint32_t local_address = 0;
        switch (placement->memory_space) {
          case LOOM_XDNA_MEMORY_SPACE_PROGRAM:
            local_address =
                tile_facts->memory.program_load_base + placement->owner_offset;
            break;
          case LOOM_XDNA_MEMORY_SPACE_DATA:
            local_address =
                tile_facts->memory.local_base + placement->owner_offset;
            break;
          default:
            IREE_ASSERT_UNREACHABLE("linked section memory space");
        }
        const uint32_t address =
            ((uint32_t)tile->coordinate.column << family->column_shift) |
            ((uint32_t)tile->coordinate.row << family->row_shift) |
            local_address;
        loom_aie2p_xdna_native_block_header(
            family, address, LOOM_AIE2P_NATIVE_HEADER_SIZE + payload_size,
            source + source_offset);
        source_offset += LOOM_AIE2P_NATIVE_HEADER_SIZE;
        const uint32_t fragment_size = source_offset - fragment_start;
        const uint32_t fragment = loom_aie2p_xdna_append_fragment(
            IREE_SV(".xdna.command"), source + fragment_start, fragment_size,
            sections, section_count);
        segments[(*segment_count)++] = loom_aie2p_xdna_load(
            allocation, destination, fragment_size, fragment);
        destination += fragment_size;
        segments[(*segment_count)++] =
            loom_aie2p_xdna_load(allocation, destination, payload_size,
                                 (uint32_t)tile_layouts[tile_index]
                                     .file_section_indices[section_index]);
        destination += payload_size;
        fragment_start = source_offset;
      }
    } else {
      const uint32_t record_size = loom_aie2p_xdna_native_record_size(record);
      loom_aie2p_xdna_native_record(family, record, source + source_offset);
      source_offset += record_size;
    }
  }
  IREE_ASSERT_EQ(source_offset, layout->control_source_offset);
  if (source_offset != fragment_start) {
    const uint32_t size = source_offset - fragment_start;
    const uint32_t section = loom_aie2p_xdna_append_fragment(
        IREE_SV(".xdna.command"), source + fragment_start, size, sections,
        section_count);
    segments[(*segment_count)++] =
        loom_aie2p_xdna_load(allocation, destination, size, section);
    destination += size;
  }
  IREE_ASSERT_EQ(destination, layout->control_destination_offset);
  uint32_t control_offset = 0;
  for (iree_host_size_t i = 0; i < program->control_record_count; ++i) {
    layout->control_record_offsets[i] = control_offset;
    loom_aie2p_xdna_native_record(
        family, &program->control_records[i],
        source + layout->control_source_offset + control_offset);
    control_offset +=
        loom_aie2p_xdna_native_record_size(&program->control_records[i]);
  }
  if (layout->control_record_offsets) {
    layout->control_record_offsets[program->control_record_count] =
        control_offset;
  }
  uint32_t control_section;
  if (program->control_record_count != 0) {
    control_section = loom_aie2p_xdna_append_fragment(
        IREE_SV(".xdna.command"), source + layout->control_source_offset,
        control_offset, sections, section_count);
    segments[(*segment_count)++] =
        loom_aie2p_xdna_load(allocation, layout->control_destination_offset,
                             control_offset, control_section);
  }
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
  if (program->control_record_count != 0) {
    segments[(*segment_count)++] = loom_aie2p_xdna_load(
        allocation, layout->repeat_offset + LOOM_AIE2P_NATIVE_HEADER_SIZE,
        control_offset, control_section);
  }
}

iree_status_t loom_aie2p_xdna_product_admit(
    const loom_aie2p_xdna_product_t* product, iree_arena_allocator_t* arena,
    bool* out_admitted, loom_aie2p_xdna_product_admission_t** out_admission,
    loom_aie2p_xdna_product_issue_t* out_issue) {
  IREE_ASSERT_ARGUMENT(product);
  IREE_ASSERT_ARGUMENT(product->device_profile);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_admitted);
  IREE_ASSERT_ARGUMENT(out_admission);
  IREE_ASSERT_ARGUMENT(out_issue);
  IREE_ASSERT_NE(product->entry_count, 0u);
  *out_admitted = false;
  *out_admission = NULL;
  *out_issue = (loom_aie2p_xdna_product_issue_t){0};

  const uint64_t maximum_entry_count =
      IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT / 2u;
  if (product->entry_count > maximum_entry_count) {
    loom_aie2p_xdna_set_issue(LOOM_AIE2P_XDNA_PRODUCT_ISSUE_ENTRY_COUNT,
                              UINT32_MAX, product->entry_count, 0,
                              maximum_entry_count, out_issue);
    return iree_ok_status();
  }

  loom_aie2p_xdna_product_admission_t* admission = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*admission), (void**)&admission));
  *admission = (loom_aie2p_xdna_product_admission_t){
      .product = product,
      .arena = arena,
      .allocation_offset = IREE_XDNA_ELF_HEADER_RECORD_SIZE,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, product->entry_count, sizeof(*admission->entry_layouts),
      (void**)&admission->entry_layouts));

  const loom_xdna_device_profile_t* profile = product->device_profile;
  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(profile);
  for (uint8_t i = 0; i < family->tile_count; ++i) {
    const loom_xdna_tile_facts_t* tile = &family->tiles[i];
    switch (tile->kind) {
      case LOOM_XDNA_TILE_KIND_COMPUTE:
        break;
      case LOOM_XDNA_TILE_KIND_MEMORY:
        admission->memory_row_count = tile->row_count;
        break;
      default:
        break;
    }
  }

  uint64_t tile_count = 0;
  uint64_t binding_count = 0;
  uint64_t relocation_count = 0;
  uint64_t entry_name_byte_length = 0;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    loom_aie2p_xdna_entry_layout_t* layout = &admission->entry_layouts[i];
    *layout = (loom_aie2p_xdna_entry_layout_t){
        .first_tile = (uint32_t)tile_count,
        .first_binding = (uint32_t)binding_count,
        .first_relocation = (uint32_t)relocation_count,
    };

    if (entry->name.size > IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH) {
      loom_aie2p_xdna_set_issue(
          LOOM_AIE2P_XDNA_PRODUCT_ISSUE_ENTRY_NAME_BYTE_LENGTH, (uint32_t)i,
          entry->name.size, 0, IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH, out_issue);
      return iree_ok_status();
    }
    tile_count += entry->tile_count;
    binding_count += entry->binding_count;
    relocation_count += 2u * (uint64_t)entry->array_program->relocation_count;
    entry_name_byte_length += entry->name.size;
    if (entry->column_count > admission->partition_column_count) {
      admission->partition_column_count = entry->column_count;
    }

    if (binding_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
      loom_aie2p_xdna_set_issue(
          LOOM_AIE2P_XDNA_PRODUCT_ISSUE_BINDING_RECORD_COUNT, (uint32_t)i,
          binding_count, 0, IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT, out_issue);
      return iree_ok_status();
    }
    if (relocation_count > IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT) {
      loom_aie2p_xdna_set_issue(
          LOOM_AIE2P_XDNA_PRODUCT_ISSUE_RELOCATION_RECORD_COUNT, (uint32_t)i,
          relocation_count, 0, IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT, out_issue);
      return iree_ok_status();
    }
  }
  IREE_ASSERT_LT(tile_count, UINT32_MAX);

  const uint64_t entry_count = product->entry_count;
  const uint64_t use_offset =
      admission->allocation_offset +
      entry_count * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE;
  const uint64_t entry_offset = use_offset + entry_count * sizeof(uint32_t);
  const uint64_t binding_offset =
      entry_offset + entry_count * IREE_XDNA_ELF_ENTRY_RECORD_SIZE;
  const uint64_t relocation_offset =
      binding_offset + binding_count * IREE_XDNA_ELF_BINDING_RECORD_SIZE;
  const uint64_t invocation_offset =
      relocation_offset +
      relocation_count * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE;
  const uint64_t string_offset =
      invocation_offset +
      2u * entry_count * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE;
  const uint64_t metadata_byte_length = string_offset + entry_name_byte_length;
  if (metadata_byte_length > IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE) {
    loom_aie2p_xdna_set_issue(
        LOOM_AIE2P_XDNA_PRODUCT_ISSUE_METADATA_BYTE_LENGTH, UINT32_MAX,
        metadata_byte_length, 0, IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE,
        out_issue);
    return iree_ok_status();
  }

  admission->tile_count = (iree_host_size_t)tile_count;
  admission->binding_count = (uint32_t)binding_count;
  admission->relocation_count = (uint32_t)relocation_count;
  admission->entry_name_byte_length = (uint32_t)entry_name_byte_length;
  admission->use_offset = (uint32_t)use_offset;
  admission->entry_offset = (uint32_t)entry_offset;
  admission->binding_offset = (uint32_t)binding_offset;
  admission->relocation_offset = (uint32_t)relocation_offset;
  admission->invocation_offset = (uint32_t)invocation_offset;
  admission->string_offset = (uint32_t)string_offset;
  admission->metadata_byte_length = (uint32_t)metadata_byte_length;
  *out_admission = admission;
  *out_admitted = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_xdna_product_finalize(
    loom_aie2p_xdna_product_admission_t* admission, bool* out_finalized,
    loom_aie2p_xdna_product_image_t** out_image,
    loom_aie2p_xdna_product_issue_t* out_issue) {
  IREE_ASSERT_ARGUMENT(admission);
  IREE_ASSERT_ARGUMENT(out_finalized);
  IREE_ASSERT_ARGUMENT(out_image);
  IREE_ASSERT_ARGUMENT(out_issue);
  *out_finalized = false;
  *out_image = NULL;
  *out_issue = (loom_aie2p_xdna_product_issue_t){0};

  const loom_aie2p_xdna_product_t* product = admission->product;
  iree_arena_allocator_t* arena = admission->arena;
  const loom_xdna_device_profile_t* profile = product->device_profile;
  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(profile);

  loom_aie2p_xdna_tile_layout_t* tile_layouts = NULL;
  if (admission->tile_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, admission->tile_count,
                                                   sizeof(*tile_layouts),
                                                   (void**)&tile_layouts));
  }
  uint64_t linked_section_count = 0;
  uint64_t symbol_string_byte_length = 1;
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    const loom_aie2p_xdna_entry_layout_t* entry_layout =
        &admission->entry_layouts[i];
    IREE_ASSERT(entry->tile_count == 0 || entry->tiles != NULL);
    for (iree_host_size_t j = 0; j < entry->tile_count; ++j) {
      const iree_host_size_t tile_index = entry_layout->first_tile + j;
      loom_aie2p_xdna_tile_layout_t* tile_layout = &tile_layouts[tile_index];
      const loom_aie2p_xdna_tile_t* tile = &entry->tiles[j];
      const loom_aie2p_linked_tile_t* linked = tile->linked_tile;
      IREE_ASSERT_EQ(linked->section_placement_count,
                     linked->assembly.section_count);
      *tile_layout = (loom_aie2p_xdna_tile_layout_t){
          .tile = tile,
      };
      linked_section_count += linked->assembly.section_count;
      uint64_t loadable_section_count = 0;
      uint64_t inline_byte_length = 0;
      for (iree_host_size_t k = 0; k < linked->assembly.section_count; ++k) {
        const loom_native_section_t* section = &linked->assembly.sections[k];
        if (loom_aie2p_xdna_section_requires_load(section)) {
          ++loadable_section_count;
          inline_byte_length +=
              iree_host_align(section->contents.data_length, 4);
        }
      }
      IREE_ASSERT_LE(loadable_section_count, UINT32_MAX);
      IREE_ASSERT_LE(inline_byte_length, UINT32_MAX);
      tile_layout->loadable_section_count = (uint32_t)loadable_section_count;
      tile_layout->inline_byte_length = (uint32_t)inline_byte_length;
      symbol_string_byte_length += tile->program_name.size + 1u;
      if (symbol_string_byte_length > UINT32_MAX) {
        loom_aie2p_xdna_set_issue(
            LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SYMBOL_STRING_BYTE_LENGTH,
            (uint32_t)i, symbol_string_byte_length, 0, UINT32_MAX, out_issue);
        return iree_ok_status();
      }
    }
  }
  IREE_ASSERT_LE(linked_section_count, IREE_HOST_SIZE_MAX);

  uint64_t total_command_section_count = 0;
  uint64_t total_segment_count = 1;  // Metadata segment.
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loom_aie2p_xdna_entry_t* entry = &product->entries[i];
    loom_aie2p_xdna_entry_layout_t* layout = &admission->entry_layouts[i];
    uint64_t entry_command_section_count = 0;
    uint64_t entry_segment_count = 0;
    uint64_t actual = 0;
    const loom_aie2p_xdna_tile_layout_t* entry_tile_layouts =
        entry->tile_count != 0 ? tile_layouts + layout->first_tile : NULL;
    const loom_aie2p_xdna_product_issue_kind_t issue_kind =
        loom_aie2p_xdna_measure_entry(
            entry, entry_tile_layouts,
            profile->limits.instruction_address_alignment, layout,
            &entry_command_section_count, &entry_segment_count, &actual);
    if (issue_kind != LOOM_AIE2P_XDNA_PRODUCT_ISSUE_NONE) {
      loom_aie2p_xdna_set_issue(issue_kind, (uint32_t)i, actual, 0, UINT32_MAX,
                                out_issue);
      return iree_ok_status();
    }
    total_command_section_count += entry_command_section_count;
    total_segment_count += entry_segment_count;
  }
  const uint32_t final_entry_ordinal = (uint32_t)product->entry_count - 1u;
  if (total_segment_count > IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT) {
    loom_aie2p_xdna_set_issue(
        LOOM_AIE2P_XDNA_PRODUCT_ISSUE_PROGRAM_HEADER_COUNT, final_entry_ordinal,
        total_segment_count, 0, IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT,
        out_issue);
    return iree_ok_status();
  }
  const uint64_t minimum_section_header_count =
      5u + total_command_section_count;
  if (minimum_section_header_count > IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT) {
    loom_aie2p_xdna_set_issue(
        LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SECTION_HEADER_COUNT, final_entry_ordinal,
        minimum_section_header_count, 0, IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT,
        out_issue);
    return iree_ok_status();
  }

  iree_host_size_t* linked_section_indices = NULL;
  const loom_native_section_t** unique_sections = NULL;
  if (linked_section_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, (iree_host_size_t)linked_section_count,
        sizeof(*linked_section_indices), (void**)&linked_section_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, (iree_host_size_t)linked_section_count, sizeof(*unique_sections),
        (void**)&unique_sections));
  }
  iree_host_size_t linked_section_offset = 0;
  for (iree_host_size_t i = 0; i < admission->tile_count; ++i) {
    loom_aie2p_xdna_tile_layout_t* tile_layout = &tile_layouts[i];
    tile_layout->file_section_indices =
        linked_section_indices + linked_section_offset;
    linked_section_offset +=
        tile_layout->tile->linked_tile->assembly.section_count;
  }
  IREE_ASSERT_EQ(linked_section_offset, linked_section_count);

  iree_host_size_t unique_section_count = 0;
  for (iree_host_size_t i = 0; i < admission->tile_count; ++i) {
    loom_aie2p_xdna_tile_layout_t* tile_layout = &tile_layouts[i];
    const loom_aie2p_linked_tile_t* linked = tile_layout->tile->linked_tile;
    const iree_host_size_t unique_index = loom_aie2p_xdna_intern_linked_section(
        &linked->assembly.sections[linked->entry_section_index],
        unique_sections, &unique_section_count);
    tile_layout->file_section_indices[linked->entry_section_index] =
        1u + unique_index;
  }
  for (iree_host_size_t i = 0; i < admission->tile_count; ++i) {
    loom_aie2p_xdna_tile_layout_t* tile_layout = &tile_layouts[i];
    const loom_aie2p_linked_tile_t* linked = tile_layout->tile->linked_tile;
    for (iree_host_size_t j = 0; j < linked->assembly.section_count; ++j) {
      if (j == linked->entry_section_index) {
        continue;
      }
      const iree_host_size_t unique_index =
          loom_aie2p_xdna_intern_linked_section(&linked->assembly.sections[j],
                                                unique_sections,
                                                &unique_section_count);
      tile_layout->file_section_indices[j] = 1u + unique_index;
    }
  }
  const uint64_t section_header_count =
      5u + total_command_section_count + unique_section_count;
  if (section_header_count > IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT) {
    loom_aie2p_xdna_set_issue(
        LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SECTION_HEADER_COUNT, UINT32_MAX,
        section_header_count, 0, IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT,
        out_issue);
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_entry_storage(
        &product->entries[i], arena, &admission->entry_layouts[i]));
  }

  const uint64_t section_capacity =
      3u + total_command_section_count + unique_section_count;
  loom_native_elf_section_t* sections = NULL;
  loom_native_elf_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, (iree_host_size_t)section_capacity,
                                sizeof(*sections), (void**)&sections));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, (iree_host_size_t)total_segment_count,
                                sizeof(*segments), (void**)&segments));

  const uint32_t entry_count = (uint32_t)product->entry_count;
  iree_byte_span_t metadata;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_allocate_bytes(
      arena, admission->metadata_byte_length, &metadata));
  const iree_xdna_elf_header_record_t header = {
      .magic = IREE_XDNA_ELF_METADATA_MAGIC,
      .version = IREE_XDNA_ELF_METADATA_VERSION,
      .native_encoding = IREE_XDNA_ELF_NATIVE_TRANSACTION_0_1,
      .target_generation = IREE_XDNA_TARGET_GENERATION_AIE2P,
      .device_profile_revision = profile->revision,
      .device_profile_id = profile->identity,
      .firmware_abi_id = profile->firmware_abi_identity,
      .column_count = admission->partition_column_count,
      .row_count = family->row_count,
      .allocation_count = entry_count,
      .allocation_use_count = entry_count,
      .entry_count = entry_count,
      .binding_count = admission->binding_count,
      .relocation_count = admission->relocation_count,
      .invocation_count = 2 * entry_count,
      .string_byte_length = admission->entry_name_byte_length,
  };
  iree_xdna_elf_encode_header(&header, metadata.data);
  uint32_t section_count = 0;
  loom_aie2p_xdna_append_fragment(IREE_SV(".xdna.metadata"), metadata.data,
                                  (uint32_t)metadata.data_length, sections,
                                  &section_count);
  for (iree_host_size_t i = 0; i < unique_section_count; ++i) {
    sections[section_count++] =
        loom_native_elf_section_from_native(unique_sections[i]);
  }
  uint32_t emitted_segment_count = 1;
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
    loom_aie2p_xdna_entry_layout_t* layout = &admission->entry_layouts[i];
    const uint32_t first_load = emitted_segment_count;
    const loom_aie2p_xdna_tile_layout_t* entry_tile_layouts =
        entry->tile_count != 0 ? tile_layouts + layout->first_tile : NULL;
    loom_aie2p_xdna_emit_entry(
        entry, i, profile, family, admission->partition_column_count,
        admission->memory_row_count, entry_tile_layouts, layout, sections,
        &section_count, segments, &emitted_segment_count);
    const iree_xdna_elf_allocation_record_t allocation = {
        .domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND,
        .byte_length =
            (uint64_t)layout->repeat_offset + layout->repeat_byte_length,
        .alignment = profile->limits.instruction_address_alignment,
        .first_load = first_load,
        .load_count = emitted_segment_count - first_load,
    };
    iree_xdna_elf_encode_allocation(
        &allocation, metadata.data + admission->allocation_offset +
                         i * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE);
    iree_unaligned_store_le_u32(
        metadata.data + admission->use_offset + i * sizeof(uint32_t), i);
    const iree_xdna_elf_entry_record_t entry_record = {
        .name_offset = name_offset,
        .name_length = (uint32_t)entry->name.size,
        .first_allocation_use = i,
        .allocation_use_count = 1,
        .first_binding = layout->first_binding,
        .binding_count = (uint32_t)entry->binding_count,
        .first_static_relocation = layout->first_relocation,
        .first_dynamic_relocation = layout->first_relocation,
        .dynamic_relocation_count =
            (uint32_t)(2 * entry->array_program->relocation_count),
        .first_invocation = 2 * i,
        .invocation_count = 2,
    };
    iree_xdna_elf_encode_entry(&entry_record,
                               metadata.data + admission->entry_offset +
                                   i * IREE_XDNA_ELF_ENTRY_RECORD_SIZE);
    memcpy(metadata.data + admission->string_offset + name_offset,
           entry->name.data, entry->name.size);
    name_offset += (uint32_t)entry->name.size;
    for (uint32_t j = 0; j < entry_record.binding_count; ++j) {
      iree_xdna_elf_encode_binding(
          &entry->bindings[j],
          metadata.data + admission->binding_offset +
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
            &relocation, metadata.data + admission->relocation_offset +
                             ordinal * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE);
      }
      iree_xdna_elf_invocation_record_t range = {
          .allocation_use = 0,
          .byte_offset = invocation == 0 ? 0 : layout->repeat_offset,
          .byte_length = invocation == 0 ? layout->initial_byte_length
                                         : layout->repeat_byte_length,
          .next_invocation = 1,
      };
      if (entry->array_program->measurement.present) {
        const uint32_t control_begin = body_offset - range.byte_offset;
        range.measurement.begin =
            control_begin +
            layout->control_record_offsets[entry->array_program->measurement
                                               .begin_record];
        range.measurement.end =
            control_begin +
            layout->control_record_offsets[entry->array_program->measurement
                                               .end_record];
      }
      iree_xdna_elf_encode_invocation(
          &range,
          metadata.data + admission->invocation_offset +
              (2 * i + invocation) * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE);
    }
  }
  iree_const_byte_span_t symbols;
  iree_const_byte_span_t strings;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_encode_symbol_tables(
      tile_layouts, admission->tile_count,
      (iree_host_size_t)symbol_string_byte_length, arena, &symbols, &strings));
  sections[section_count] = (loom_native_elf_section_t){
      .name = IREE_SV(".symtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB,
      .alignment = 4,
      .entry_size = LOOM_AIE2P_XDNA_ELF32_SYMBOL_SIZE,
      .link = section_count + 2,
      .info = (uint32_t)admission->tile_count + 1u,
      .contents = symbols,
  };
  ++section_count;
  sections[section_count++] = (loom_native_elf_section_t){
      .name = IREE_SV(".strtab"),
      .type = LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB,
      .alignment = 1,
      .contents = strings,
  };
  IREE_ASSERT_EQ(section_count, section_capacity);
  IREE_ASSERT_EQ(emitted_segment_count, total_segment_count);

  uint64_t section_name_byte_length = 1u;
  for (iree_host_size_t i = 0; i < section_count; ++i) {
    section_name_byte_length += sections[i].name.size + 1u;
  }
  section_name_byte_length += IREE_SV(".shstrtab").size + 1u;
  if (section_name_byte_length > UINT32_MAX) {
    loom_aie2p_xdna_set_issue(
        LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SECTION_NAME_BYTE_LENGTH, UINT32_MAX,
        section_name_byte_length, 0, UINT32_MAX, out_issue);
    return iree_ok_status();
  }

  loom_aie2p_xdna_product_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*image), (void**)&image));
  image->file = (loom_native_elf32le_file_t){
      .type = LOOM_NATIVE_ELF_FILE_TYPE_EXEC,
      .machine = LOOM_NATIVE_ELF_MACHINE_AIE,
      .flags = IREE_XDNA_ELF_AIE2P_FLAGS,
      .sections = sections,
      .section_count = section_count,
      .segments = segments,
      .segment_count = emitted_segment_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_native_elf32le_build_layout(&image->file, arena, &image->layout));
  if (image->layout.file_size > UINT32_MAX) {
    loom_aie2p_xdna_set_issue(LOOM_AIE2P_XDNA_PRODUCT_ISSUE_FILE_BYTE_LENGTH,
                              UINT32_MAX, image->layout.file_size, 0,
                              UINT32_MAX, out_issue);
    return iree_ok_status();
  }
  *out_image = image;
  *out_finalized = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_xdna_product_write(
    const loom_aie2p_xdna_product_image_t* image, iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(image);
  IREE_ASSERT_ARGUMENT(stream);
  return loom_native_elf32le_write_file(&image->file, &image->layout, stream);
}
