// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer.h"

#include <string.h>

#include "loom/format/bytecode/writer/body.h"
#include "loom/format/bytecode/writer/catalog.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/format/bytecode/writer/numbering.h"
#include "loom/format/bytecode/writer/symbol.h"
#include "loom/format/bytecode/writer/tables.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/module_record.h"
#include "loom/ops/op_defs.h"

#define LOOM_BYTECODE_DEFAULT_PRODUCER "loom-c"

//===----------------------------------------------------------------------===//
// Top-level writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_validate_module(
    const loom_module_t* module) {
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (needed for op vtables)");
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        bytecode_kind == LOOM_SYMBOL_GLOBAL) {
      if (!symbol->defining_op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GLOBAL symbol %" PRIhsz " has no defining op",
                                i);
      }
      const loom_op_t* op = symbol->defining_op;
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable ||
          !iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
          !vtable->symbol_def ||
          vtable->symbol_def->bytecode_kind != LOOM_SYMBOL_GLOBAL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GLOBAL symbol %" PRIhsz
                                " defining op does not use the GLOBAL "
                                "bytecode payload",
                                i);
      }
      if (op->operand_count != 0 || op->region_count != 0 ||
          op->tied_result_count != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz
            " defining op must not have operands, regions, or tied results",
            i);
      }
      if (op->result_count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz " defining op must have results", i);
      }
      if (op->attribute_count > 0 && !vtable->attr_descriptors) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz
            " defining op has attributes but no descriptors",
            i);
      }
    }
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_EXECUTABLE) ||
        bytecode_kind == LOOM_SYMBOL_EXECUTABLE) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "EXECUTABLE symbols not yet supported");
    }
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD) {
      if (!symbol->defining_op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "RECORD symbol %" PRIhsz " has no defining op",
                                i);
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_validate_record_symbol_op(module, symbol->defining_op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_module(
    const loom_module_t* module, iree_io_stream_t* stream,
    const loom_bytecode_write_options_t* options,
    iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, loom_bytecode_validate_module(module));

  // Check stream capabilities.
  iree_io_stream_mode_t mode = iree_io_stream_mode(stream);
  if (!(mode & IREE_IO_STREAM_MODE_WRITABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not writable");
  }
  if (!(mode & IREE_IO_STREAM_MODE_SEEKABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not seekable (needed for directory "
                            "patching)");
  }

  iree_string_view_t producer = (options && options->producer.size > 0)
                                    ? options->producer
                                    : IREE_SV(LOOM_BYTECODE_DEFAULT_PRODUCER);
  loom_bytecode_location_mode_t location_mode =
      options ? options->location_mode
              : LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS;
  if (location_mode > LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported bytecode location mode %u",
                            (unsigned)location_mode);
  }
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FULL_LOCATIONS bytecode mode requires field span emission");
  }

  // Temporary arena for all working memory. All numbering tables,
  // value maps, and scratch allocations come from here. Deinitialized
  // at the end, returning all blocks to the shared pool in O(1).
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Initialize the page writer.
  loom_bytecode_page_writer_t page_writer;
  loom_bytecode_page_writer_initialize(&page_writer, stream);

  // Initialize numbering context.
  loom_bytecode_numbering_t numbering;
  iree_status_t status =
      loom_bytecode_numbering_initialize(&numbering, module, &arena);
  numbering.location_mode = location_mode;
  numbering.low_repr.environment = options ? options->low_repr_environment
                                           : (loom_low_repr_environment_t){0};

  loom_module_record_plan_t record_plan = {0};
  bool record_plan_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_module_record_plan_initialize(module, &record_plan);
    record_plan_initialized = iree_status_is_ok(status);
  }
  // Pass 1: Number module metadata. Function signatures and bodies are numbered
  // during IR section writing.
  if (iree_status_is_ok(status)) {
    uint32_t unused_id = 0;
    status = loom_bytecode_numbering_intern_module_string(
        &numbering, module->name_id, &unused_id);
  }
  if (iree_status_is_ok(status)) {
    for (loom_symbol_id_t wire_ordinal = 0;
         wire_ordinal < module->symbols.count && iree_status_is_ok(status);
         ++wire_ordinal) {
      const loom_symbol_id_t module_symbol_id =
          loom_bytecode_module_symbol_id(&numbering, wire_ordinal);
      uint32_t unused_id = 0;
      status = loom_bytecode_numbering_intern_module_string(
          &numbering, module->symbols.entries[module_symbol_id].name_id,
          &unused_id);
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < module->encodings.count && iree_status_is_ok(status); ++i) {
      status = loom_bytecode_number_encoding(&numbering, (uint16_t)(i + 1));
    }
  }
  loom_bytecode_symbol_reference_plan_t symbol_reference_plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_symbol_reference_plan_initialize(
        module, &arena, &symbol_reference_plan);
  }

  // File header: magic, version, location mode, module count, producer string.
  iree_string_view_t module_name = module->strings.entries[module->name_id];

  if (iree_status_is_ok(status)) {
    // Magic.
    status = loom_bytecode_page_writer_write(&page_writer, LOOM_BYTECODE_MAGIC,
                                             LOOM_BYTECODE_MAGIC_LENGTH);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer,
                                                LOOM_BYTECODE_FORMAT_VERSION);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer, location_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    1);  // module_count
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u32_le(
        &page_writer, (uint32_t)module_name.size);  // string_pool_length
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // reserved
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_null_terminated_string(
        &page_writer, producer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module directory entry. module_offset and module_length are
  // written as placeholders and patched after all sections are written.
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // name_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    (uint16_t)module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    0);  // module_flags
  }
  // module_offset placeholder: we'll patch this as part of the directory entry.
  iree_host_size_t module_dir_offset_position = 0;
  if (iree_status_is_ok(status)) {
    module_dir_offset_position = page_writer.total_written;
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_length
  }

  // File string pool: module name(s).
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write(&page_writer, module_name.data,
                                             module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module data starts at this offset.
  iree_host_size_t module_start = page_writer.total_written;

  loom_bytecode_section_kind_t section_write_order[LOOM_BYTECODE_SECTION_COUNT];
  iree_host_size_t section_count = 0;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_IR;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_SYMBOLS;
  section_write_order[section_count++] =
      LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_STRINGS;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_SOURCES;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_TYPES;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_ENCODINGS;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_OPS;
  if (location_mode != LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    section_write_order[section_count++] = LOOM_BYTECODE_SECTION_LOCATIONS;
  }
  iree_host_size_t file_header_line_count = 0;
  (void)loom_module_file_header(module, &file_header_line_count);
  if (file_header_line_count > 0) {
    section_write_order[section_count++] = LOOM_BYTECODE_SECTION_SOURCE_TRIVIA;
  }
  loom_bytecode_body_counts_t module_counts = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_count_serialized_bodies(&numbering, &module_counts);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_uvarint(&page_writer, section_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.value_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(
        &page_writer, module_counts.region_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.block_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.op_count);
  }

  // Section directory placeholder — patched after all sections are written.
  iree_host_size_t section_dir_patch_position = page_writer.total_written;
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_zeros(
        &page_writer,
        section_count * sizeof(loom_bytecode_section_dir_entry_t));
  }

  // Track section offsets and lengths.
  uint64_t section_offsets[LOOM_BYTECODE_SECTION_COUNT] = {0};
  uint64_t section_lengths[LOOM_BYTECODE_SECTION_COUNT] = {0};

  // Allocate root-region payload tracking from the arena.
  loom_bytecode_ir_region_list_t* ir_regions = NULL;
  if (iree_status_is_ok(status) && module->symbols.count > 0) {
    status =
        iree_arena_allocate_array(&arena, module->symbols.count,
                                  sizeof(*ir_regions), (void**)&ir_regions);
    if (iree_status_is_ok(status)) {
      memset(ir_regions, 0, module->symbols.count * sizeof(*ir_regions));
    }
  }

  // IR section: independently bounded root regions streamed through the page
  // writer.
  // Written first so the numbering tables grow as entities are encountered.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_IR] =
        page_writer.total_written - module_start;
    status =
        loom_bytecode_write_ir_section(&page_writer, &numbering, ir_regions);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_IR] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_IR];
    }
  }

  // Symbols section: buffered in a string builder because the import/export
  // offset tables at the start reference entry positions that come later.
  // The SYMBOLS section uses a string_builder (which needs realloc, so it
  // can't use the arena). Use the module's context allocator.
  iree_string_builder_t symbols_builder;
  iree_string_builder_initialize(module->context->allocator, &symbols_builder);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_write_symbols_section(&symbols_builder, &numbering,
                                                 ir_regions);
  }
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SYMBOLS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_page_writer_write(
        &page_writer, iree_string_builder_buffer(&symbols_builder),
        iree_string_builder_size(&symbols_builder));
    section_lengths[LOOM_BYTECODE_SECTION_SYMBOLS] =
        iree_string_builder_size(&symbols_builder);
  }
  iree_string_builder_deinitialize(&symbols_builder);

  // Symbol references preserve the direct metadata-only dependency graph.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_symbol_references_section(
        &page_writer, &numbering, &symbol_reference_plan);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES];
    }
  }

  // Strings section: all interned strings from the numbering context.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_STRINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_strings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_STRINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_STRINGS];
    }
  }

  // Sources section: module-local source identifiers (filenames, tags).
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SOURCES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_sources_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SOURCES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SOURCES];
    }
  }

  // Types section: interned type table in topological order.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_TYPES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_types_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_TYPES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_TYPES];
    }
  }

  // Encodings section: kind registry + parameterized instances.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_encodings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_ENCODINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS];
    }
  }

  // Ops section: op kind name registry.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_OPS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_ops_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_OPS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_OPS];
    }
  }

  // Locations section.
  if (iree_status_is_ok(status) &&
      location_mode != LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_locations_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_LOCATIONS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS];
    }
  }

  // Module-owned source presentation.
  if (iree_status_is_ok(status) && file_header_line_count > 0) {
    section_offsets[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_source_trivia_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA];
    }
  }

  // Flush remaining page buffer, then seek back to patch the section
  // directory and module directory with correct offsets and lengths.
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_flush(&page_writer);
  }

  // Patch section directory.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)section_dir_patch_position);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < section_count && iree_status_is_ok(status);
         ++i) {
      loom_bytecode_section_kind_t kind = section_write_order[i];
      uint8_t entry[sizeof(loom_bytecode_section_dir_entry_t)] = {0};
      entry[0] = (uint8_t)kind;
      entry[1] = (uint8_t)((uint16_t)kind >> 8);
      uint64_t section_offset = section_offsets[kind];
      uint64_t section_length = section_lengths[kind];
      for (int byte_index = 0; byte_index < 8; ++byte_index) {
        entry[8 + byte_index] = (uint8_t)(section_offset >> (byte_index * 8));
        entry[16 + byte_index] = (uint8_t)(section_length >> (byte_index * 8));
      }
      status = iree_io_stream_write(stream, sizeof(entry), entry);
    }
  }

  // Patch module directory: module_offset and module_length.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)module_dir_offset_position);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_offset = module_start;
    status = iree_io_stream_write(stream, 8, &module_offset);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_length = page_writer.total_written - module_start;
    status = iree_io_stream_write(stream, 8, &module_length);
  }

  if (iree_status_is_ok(status) && options != NULL) {
    const loom_bytecode_symbol_projection_t* projection =
        &options->symbol_projection;
    for (iree_host_size_t i = 0; i < projection->count; ++i) {
      IREE_ASSERT_LT(projection->module_symbol_ids[i], module->symbols.count);
      projection->wire_symbol_ordinals[i] = loom_bytecode_wire_symbol_ordinal(
          &numbering, projection->module_symbol_ids[i]);
    }
  }

  // All numbering tables, value maps, and IR region lists were arena-allocated.
  // One call returns all blocks to the shared pool.
  if (record_plan_initialized) {
    loom_module_record_plan_deinitialize(&record_plan);
  }
  iree_arena_deinitialize(&arena);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
