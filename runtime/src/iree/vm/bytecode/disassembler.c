// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/disassembler.h"

#include <inttypes.h>
#include <stdarg.h>
#include <string.h>

#include "iree/vm/bytecode/disassembler_native.h"
#include "iree/vm/bytecode/verifier.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/bytecode/wire/module.h"

enum {
  IREE_VM_BYTECODE_DISASSEMBLER_INLINE_BLOCK_COUNT = 16,
};

typedef struct iree_vm_bytecode_disassembler_numeric_value_t {
  // Offset of the canonical spelling in the generated text pool.
  uint16_t name_offset;
  // Encoded selector or flag bit value.
  uint16_t value;
} iree_vm_bytecode_disassembler_numeric_value_t;

typedef struct iree_vm_bytecode_disassembler_numeric_table_t {
  // First entry in |iree_vm_bytecode_disassembler_numeric_values|.
  uint16_t value_base;
  // Number of entries owned by the table.
  uint16_t value_count;
  // Whether values may be combined as a flag set.
  uint8_t is_flags;
  // Whether unknown nonzero values have a canonical numeric spelling.
  uint8_t preserves_unknown;
} iree_vm_bytecode_disassembler_numeric_table_t;

typedef struct iree_vm_bytecode_disassembler_lane_t {
  // Offset of the element-type suffix in the generated text pool.
  uint16_t name_offset;
  // Encoded memory-format selector value.
  uint8_t value;
  // Number of scalar lanes transferred by the format.
  uint8_t lane_count;
} iree_vm_bytecode_disassembler_lane_t;

typedef struct iree_vm_bytecode_disassembler_direct_target_t {
  // Encoded direct-call target selector value.
  uint8_t value;
  // Textual symbol domain selected by |value|.
  uint8_t domain;
} iree_vm_bytecode_disassembler_direct_target_t;

typedef struct iree_vm_bytecode_disassembler_field_t {
  // Offset of the attribute name in the generated text pool.
  uint16_t name_offset;
  // Kind-specific numeric table, symbol domain, or row-table base.
  uint16_t data;
  // Byte offset of the primary field in the instruction record.
  uint8_t primary_offset;
  // Byte offset of the related field, if any.
  uint8_t related_offset;
  // One iree_vm_bytecode_disassembler_field_kind_t value.
  uint8_t kind;
  // Byte width of the primary field.
  uint8_t primary_width;
  // Byte width of the related field, if any.
  uint8_t related_width;
  // Number of scalar elements encoded in the primary field.
  uint8_t element_count;
  // Kind-specific bit offset in the low nibble and length in the high nibble.
  uint8_t bit_range;
  // Number of kind-specific rows beginning at |data|.
  uint8_t data_count;
} iree_vm_bytecode_disassembler_field_t;

typedef struct iree_vm_bytecode_disassembler_instruction_t {
  // First entry in |iree_vm_bytecode_disassembler_fields|.
  uint16_t field_base;
  // First entry in |iree_vm_bytecode_disassembler_lanes|.
  uint16_t lane_base;
  // Fixed encoded instruction record length in bytes.
  uint8_t byte_length;
  // Number of leading fields printed as results.
  uint8_t result_count;
  // Number of fields following results printed as operands.
  uint8_t operand_count;
  // Total number of fields owned by the instruction.
  uint8_t field_count;
  // Byte offset of the mnemonic lane selector, or NO_FIELD.
  uint8_t projection_offset;
  // Number of lane projection rows beginning at |lane_base|.
  uint8_t lane_count;
} iree_vm_bytecode_disassembler_instruction_t;

typedef struct iree_vm_bytecode_disassembler_grammar_t {
  // Byte offset of the grammar in the generated format program.
  uint16_t program_offset;
  // Physical record ordinal consumed by the grammar.
  uint8_t record;
  // One iree_vm_bytecode_disassembler_row_mode_t value.
  uint8_t row_mode;
} iree_vm_bytecode_disassembler_grammar_t;

typedef struct iree_vm_bytecode_disassembler_run_t {
  // Grammar ordinal emitted by the run.
  uint8_t grammar;
  // One iree_vm_bytecode_disassembler_run_kind_t value.
  uint8_t kind;
  // Record ordinal supplying a dynamic run count.
  uint8_t source;
  // Byte offset of a count or matching field in |source|.
  uint8_t field_offset;
  // Byte width of the count or matching field.
  uint8_t field_width;
  // Fixed run count or required matching field value.
  uint16_t value;
} iree_vm_bytecode_disassembler_run_t;

typedef struct iree_vm_bytecode_disassembler_layout_field_t {
  // Byte offset of the field in its associated physical record.
  uint8_t offset;
  // Byte width of the field.
  uint8_t width;
} iree_vm_bytecode_disassembler_layout_field_t;

typedef struct iree_vm_bytecode_disassembler_layout_item_t {
  // Fixed count or alignment, depending on |kind|.
  uint16_t value;
  // First entry in |iree_vm_bytecode_disassembler_layout_fields|.
  uint16_t field_base;
  // One iree_vm_bytecode_disassembler_layout_kind_t value.
  uint8_t kind;
  // Destination record or tail ordinal.
  uint8_t target;
  // Source record ordinal for field-derived extents.
  uint8_t source;
  // Number of fields beginning at |field_base|.
  uint8_t field_count;
  // Signed adjustment applied to a derived record count.
  int8_t adjustment;
} iree_vm_bytecode_disassembler_layout_item_t;

typedef struct iree_vm_bytecode_disassembler_section_t {
  // Offset of the section name in the generated text pool.
  uint16_t name_offset;
  // Encoded section type used for direct lookup.
  uint16_t section_type;
  // First entry in |iree_vm_bytecode_disassembler_runs|.
  uint8_t run_base;
  // First entry in |iree_vm_bytecode_disassembler_layout_items|.
  uint8_t layout_base;
  // Number of grammar runs beginning at |run_base|.
  uint8_t run_count;
  // Number of layout actions beginning at |layout_base|.
  uint8_t layout_count;
} iree_vm_bytecode_disassembler_section_t;

typedef struct iree_vm_bytecode_disassembler_shell_t {
  // Offsets of module and section delimiters in the generated text pool.
  uint16_t literal_offsets[7];
  // Physical image-header record ordinal.
  uint8_t image_record;
  // Physical section-directory record ordinal.
  uint8_t section_record;
  // Byte offset of the Core major revision in the image header.
  uint8_t core_major_offset;
  // Byte width of the Core major revision.
  uint8_t core_major_width;
  // Byte offset of the required Core minor revision in the image header.
  uint8_t core_required_minor_offset;
  // Byte width of the required Core minor revision.
  uint8_t core_required_minor_width;
  // Byte offset of section alignment in a directory row.
  uint8_t section_alignment_offset;
  // Byte width of section alignment.
  uint8_t section_alignment_width;
} iree_vm_bytecode_disassembler_shell_t;

#include "iree/vm/bytecode/disassembler_tables.inl"

typedef struct iree_vm_bytecode_disassembler_record_view_t {
  // First physical row in the verified image.
  const uint8_t* rows;
  // Number of rows available at |rows|.
  uint32_t count;
} iree_vm_bytecode_disassembler_record_view_t;

typedef struct iree_vm_bytecode_disassembler_tail_view_t {
  // First byte of the variable-length tail.
  const uint8_t* data;
  // Tail offset relative to its section payload.
  iree_host_size_t section_offset;
  // Current byte position while emitting sequential tail entries.
  iree_host_size_t cursor;
} iree_vm_bytecode_disassembler_tail_view_t;

typedef struct iree_vm_bytecode_disassembler_reader_t {
  // Verified physical rows indexed by generated record ordinal.
  iree_vm_bytecode_disassembler_record_view_t
      records[IREE_VM_BYTECODE_MODULE_RECORD_COUNT];
  // Next logical row consumed by each append-mode grammar.
  uint32_t record_cursors[IREE_VM_BYTECODE_MODULE_RECORD_COUNT];
  // Verified variable-length section tails indexed by generated tail ordinal.
  iree_vm_bytecode_disassembler_tail_view_t
      tails[IREE_VM_BYTECODE_DISASSEMBLER_TAIL_COUNT];
} iree_vm_bytecode_disassembler_reader_t;

typedef struct iree_vm_bytecode_disassembler_function_t {
  // First byte of the function instruction stream.
  const uint8_t* bytecode;
  // Length of the function instruction stream in bytes.
  uint32_t bytecode_length;
  // First switch-target row owned by the function.
  uint32_t switch_target_base;
  // Number of switch-target rows owned by the function.
  uint32_t switch_target_count;
  // Number of control blocks in the instruction stream.
  uint32_t block_count;
  // Scratch table mapping block ordinals to instruction byte offsets.
  uint32_t* block_offsets;
} iree_vm_bytecode_disassembler_function_t;

typedef struct iree_vm_bytecode_disassembler_t {
  // Caller-provided synchronous text sink.
  iree_vm_bytecode_disassembler_write_callback_t write_callback;
  // Physical views and traversal cursors for the verified image.
  iree_vm_bytecode_disassembler_reader_t reader;
  // Reusable per-function block-offset scratch storage.
  uint32_t* block_offsets;
  // Admitted native declarations and owning function storage descriptions.
  const iree_vm_bytecode_module_layout_t* layout;
  // Allocator for unusually deep native operand traversal.
  iree_allocator_t scratch_allocator;
} iree_vm_bytecode_disassembler_t;

static iree_string_view_t iree_vm_bytecode_disassembler_name_at(
    uint16_t offset) {
  const uint8_t* bstring = iree_vm_bytecode_disassembler_names + offset;
  return iree_make_string_view((const char*)bstring + 1, bstring[0]);
}

static iree_string_view_t iree_vm_bytecode_disassembler_text_at(
    uint16_t offset) {
  const uint8_t* bstring = iree_vm_bytecode_disassembler_strings + offset;
  return iree_make_string_view((const char*)bstring + 1, bstring[0]);
}

static iree_status_t iree_vm_bytecode_disassembler_write(
    iree_vm_bytecode_disassembler_t* disassembler, iree_string_view_t value) {
  if (value.size == 0) {
    return iree_ok_status();
  }
  return disassembler->write_callback.fn(disassembler->write_callback.user_data,
                                         value);
}

static iree_status_t iree_vm_bytecode_disassembler_write_cstring(
    iree_vm_bytecode_disassembler_t* disassembler, const char* value) {
  return iree_vm_bytecode_disassembler_write(disassembler,
                                             iree_make_cstring_view(value));
}

static iree_status_t iree_vm_bytecode_disassembler_write_format(
    iree_vm_bytecode_disassembler_t* disassembler, const char* format, ...) {
  char buffer[128];
  va_list varargs;
  va_start(varargs, format);
  const int length = iree_vsnprintf(buffer, sizeof(buffer), format, varargs);
  va_end(varargs);
  if (IREE_UNLIKELY(length < 0 || (iree_host_size_t)length >= sizeof(buffer))) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "disassembler scalar formatting overflow");
  }
  return iree_vm_bytecode_disassembler_write(
      disassembler, iree_make_string_view(buffer, (iree_host_size_t)length));
}

static uint64_t iree_vm_bytecode_disassembler_load(const uint8_t* data,
                                                   uint8_t width) {
  switch (width) {
    case 1:
      return data[0];
    case 2:
      return iree_unaligned_load_le_u16(data);
    case 4:
      return iree_unaligned_load_le_u32(data);
    case 8:
      return iree_unaligned_load_le_u64(data);
    default:
      IREE_ASSERT_UNREACHABLE("generated disassembler field width is invalid");
      return 0;
  }
}

static int64_t iree_vm_bytecode_disassembler_signed_value(uint64_t value,
                                                          uint8_t width) {
  const uint8_t bit_count = width * 8;
  const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
  if ((value & sign_bit) == 0) {
    return (int64_t)value;
  }
  const uint64_t mask =
      bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
  return -1 - (int64_t)(mask - value);
}

static const uint8_t* iree_vm_bytecode_disassembler_record_at(
    const iree_vm_bytecode_disassembler_reader_t* reader, uint8_t record,
    uint32_t ordinal) {
  return reader->records[record].rows +
         ordinal * iree_vm_bytecode_disassembler_record_lengths[record];
}

static uint64_t iree_vm_bytecode_disassembler_record_field(
    const iree_vm_bytecode_disassembler_reader_t* reader, uint8_t record,
    uint32_t ordinal,
    const iree_vm_bytecode_disassembler_layout_field_t* field) {
  return iree_vm_bytecode_disassembler_load(
      iree_vm_bytecode_disassembler_record_at(reader, record, ordinal) +
          field->offset,
      field->width);
}

static const iree_vm_bytecode_disassembler_section_t*
iree_vm_bytecode_disassembler_find_section(uint16_t section_type) {
  if (section_type >= IREE_ARRAYSIZE(iree_vm_bytecode_disassembler_sections)) {
    return NULL;
  }
  const iree_vm_bytecode_disassembler_section_t* section =
      &iree_vm_bytecode_disassembler_sections[section_type];
  return section->section_type == section_type ? section : NULL;
}

static uint32_t iree_vm_bytecode_disassembler_layout_count(
    const iree_vm_bytecode_disassembler_reader_t* reader,
    const iree_vm_bytecode_disassembler_layout_item_t* item,
    iree_host_size_t remaining_length) {
  const iree_vm_bytecode_disassembler_layout_field_t* fields =
      iree_vm_bytecode_disassembler_layout_fields + item->field_base;
  uint64_t count = 0;
  switch (item->kind) {
    case IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_RECORD_FIXED:
      count = item->value;
      break;
    case IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_RECORD_REMAINING:
      count = remaining_length /
              iree_vm_bytecode_disassembler_record_lengths[item->target];
      break;
    case IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_RECORD_FIELD:
      count = iree_vm_bytecode_disassembler_record_field(reader, item->source,
                                                         0, fields);
      break;
    case IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_RECORD_SUM:
      for (uint32_t i = 0; i < reader->records[item->source].count; ++i) {
        for (uint8_t j = 0; j < item->field_count; ++j) {
          count += iree_vm_bytecode_disassembler_record_field(
              reader, item->source, i, &fields[j]);
        }
      }
      break;
    case IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_RECORD_FINAL_EXTENT:
      if (reader->records[item->source].count != 0) {
        const uint32_t final_ordinal = reader->records[item->source].count - 1;
        count = iree_vm_bytecode_disassembler_record_field(
                    reader, item->source, final_ordinal, &fields[0]) +
                iree_vm_bytecode_disassembler_record_field(
                    reader, item->source, final_ordinal, &fields[1]);
      }
      break;
    default:
      IREE_ASSERT_UNREACHABLE("layout action does not produce a record count");
      break;
  }
  return (uint32_t)((int64_t)count + item->adjustment);
}

static void iree_vm_bytecode_disassembler_map_section(
    iree_const_byte_span_t payload,
    const iree_vm_bytecode_disassembler_section_t* section,
    iree_vm_bytecode_disassembler_reader_t* reader) {
  iree_host_size_t cursor = 0;
  for (uint8_t i = 0; i < section->layout_count; ++i) {
    const iree_vm_bytecode_disassembler_layout_item_t* item =
        &iree_vm_bytecode_disassembler_layout_items[section->layout_base + i];
    if (item->kind <=
        IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_RECORD_FINAL_EXTENT) {
      const uint32_t count = iree_vm_bytecode_disassembler_layout_count(
          reader, item, payload.data_length - cursor);
      reader->records[item->target].rows = payload.data + cursor;
      reader->records[item->target].count = count;
      cursor += (iree_host_size_t)count *
                iree_vm_bytecode_disassembler_record_lengths[item->target];
    } else if (item->kind == IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_ALIGN) {
      cursor = iree_host_align(cursor, item->value);
    } else {
      iree_vm_bytecode_disassembler_tail_view_t* tail =
          &reader->tails[item->target];
      tail->data = payload.data + cursor;
      tail->section_offset = cursor;
      iree_host_size_t tail_length = 0;
      if (item->kind == IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_TAIL_REMAINING) {
        tail_length = payload.data_length - cursor;
      } else {
        const iree_vm_bytecode_disassembler_layout_field_t* fields =
            iree_vm_bytecode_disassembler_layout_fields + item->field_base;
        if (reader->records[item->source].count != 0) {
          const uint32_t final_ordinal =
              reader->records[item->source].count - 1;
          tail_length =
              (iree_host_size_t)(iree_vm_bytecode_disassembler_record_field(
                                     reader, item->source, final_ordinal,
                                     &fields[0]) +
                                 iree_vm_bytecode_disassembler_record_field(
                                     reader, item->source, final_ordinal,
                                     &fields[1]));
        }
      }
      cursor += tail_length;
    }
  }
}

static void iree_vm_bytecode_disassembler_bind_reader(
    iree_const_byte_span_t contents, const iree_vm_bytecode_module_plan_t* plan,
    iree_vm_bytecode_disassembler_reader_t* reader) {
  memset(reader, 0, sizeof(*reader));
  reader->records[iree_vm_bytecode_disassembler_shell.image_record].rows =
      contents.data;
  reader->records[iree_vm_bytecode_disassembler_shell.image_record].count = 1;
  reader->records[iree_vm_bytecode_disassembler_shell.section_record].rows =
      (const uint8_t*)plan->layout.image.sections;
  reader->records[iree_vm_bytecode_disassembler_shell.section_record].count =
      plan->layout.image.section_count;

  iree_host_size_t cursor =
      sizeof(iree_vm_bytecode_v0_image_header_t) +
      (iree_host_size_t)plan->layout.image.section_count *
          sizeof(iree_vm_bytecode_v0_section_directory_row_t);
  for (uint16_t i = 0; i < plan->layout.image.section_count; ++i) {
    const iree_vm_bytecode_v0_section_directory_row_t* row =
        &plan->layout.image.sections[i];
    cursor = iree_host_align(cursor, row->payload_alignment_u32);
    const iree_vm_bytecode_disassembler_section_t* section =
        iree_vm_bytecode_disassembler_find_section(row->section_type_u16);
    iree_vm_bytecode_disassembler_map_section(
        iree_make_const_byte_span(contents.data + cursor,
                                  (iree_host_size_t)row->byte_length_u64),
        section, reader);
    cursor += (iree_host_size_t)row->byte_length_u64;
  }
}

static iree_status_t iree_vm_bytecode_disassembler_indent(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t depth) {
  static const char spaces[] =
      "                                                                ";
  return iree_vm_bytecode_disassembler_write(
      disassembler, iree_make_string_view(spaces, depth * 2));
}

static iree_status_t iree_vm_bytecode_disassembler_symbol_name(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t domain,
    uint32_t ordinal) {
  const iree_string_view_t prefix = iree_vm_bytecode_disassembler_text_at(
      iree_vm_bytecode_disassembler_domain_name_offsets[domain]);
  return iree_vm_bytecode_disassembler_write_format(
      disassembler, "%.*s%" PRIu32, (int)prefix.size, prefix.data, ordinal);
}

static iree_status_t iree_vm_bytecode_disassembler_symbol_reference(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t domain,
    uint32_t ordinal) {
  const iree_string_view_t prefix = iree_vm_bytecode_disassembler_text_at(
      iree_vm_bytecode_disassembler_domain_name_offsets[domain]);
  return iree_vm_bytecode_disassembler_write_format(
      disassembler, "@%.*s%" PRIu32, (int)prefix.size, prefix.data, ordinal);
}

static iree_status_t iree_vm_bytecode_disassembler_block_symbol(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t domain,
    uint32_t ordinal) {
  const iree_string_view_t prefix = iree_vm_bytecode_disassembler_text_at(
      iree_vm_bytecode_disassembler_domain_name_offsets[domain]);
  return iree_vm_bytecode_disassembler_write_format(
      disassembler, "^%.*s%" PRIu32, (int)prefix.size, prefix.data, ordinal);
}

static const iree_vm_bytecode_disassembler_numeric_value_t*
iree_vm_bytecode_disassembler_find_numeric(uint8_t table_ordinal,
                                           uint16_t number) {
  const iree_vm_bytecode_disassembler_numeric_table_t* table =
      &iree_vm_bytecode_disassembler_numeric_tables[table_ordinal];
  for (uint16_t i = 0; i < table->value_count; ++i) {
    const iree_vm_bytecode_disassembler_numeric_value_t* value =
        &iree_vm_bytecode_disassembler_numeric_values[table->value_base + i];
    if (value->value == number) {
      return value;
    }
  }
  return NULL;
}

static iree_status_t iree_vm_bytecode_disassembler_numeric(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t table_ordinal,
    uint16_t number) {
  const iree_vm_bytecode_disassembler_numeric_table_t* table =
      &iree_vm_bytecode_disassembler_numeric_tables[table_ordinal];
  if (!table->is_flags) {
    const iree_vm_bytecode_disassembler_numeric_value_t* value =
        iree_vm_bytecode_disassembler_find_numeric(table_ordinal, number);
    if (value) {
      return iree_vm_bytecode_disassembler_write(
          disassembler,
          iree_vm_bytecode_disassembler_text_at(value->name_offset));
    }
    if (table->preserves_unknown && number != 0) {
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "type(%u)", number);
    }
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "verified numeric value has no text spelling");
  }

  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "["));
  bool has_previous = false;
  for (uint16_t i = 0; i < table->value_count; ++i) {
    const iree_vm_bytecode_disassembler_numeric_value_t* value =
        &iree_vm_bytecode_disassembler_numeric_values[table->value_base + i];
    if (value->value == 0 || (number & value->value) == 0) {
      continue;
    }
    if (has_previous) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
        disassembler,
        iree_vm_bytecode_disassembler_text_at(value->name_offset)));
    number &= (uint16_t)~value->value;
    has_previous = true;
  }
  if (IREE_UNLIKELY(number != 0)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "verified flag value has no text spelling");
  }
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, "]");
}

static iree_status_t iree_vm_bytecode_disassembler_quoted_bytes(
    iree_vm_bytecode_disassembler_t* disassembler,
    iree_const_byte_span_t bytes) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "\""));
  iree_host_size_t run_begin = 0;
  for (iree_host_size_t i = 0; i < bytes.data_length; ++i) {
    const uint8_t value = bytes.data[i];
    const bool needs_escape = value == '\\' || value == '"' || value == '\n' ||
                              value == '\r' || value == '\t' || value < 0x20 ||
                              value == 0x7F;
    if (!needs_escape) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
        disassembler, iree_make_string_view((const char*)bytes.data + run_begin,
                                            i - run_begin)));
    char escape[4];
    iree_host_size_t escape_length = 2;
    escape[0] = '\\';
    if (value == '\\' || value == '"') {
      escape[1] = (char)value;
    } else if (value == '\n') {
      escape[1] = 'n';
    } else if (value == '\r') {
      escape[1] = 'r';
    } else if (value == '\t') {
      escape[1] = 't';
    } else {
      static const char digits[] = "0123456789ABCDEF";
      escape[1] = 'x';
      escape[2] = digits[value >> 4];
      escape[3] = digits[value & 0x0F];
      escape_length = 4;
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
        disassembler, iree_make_string_view(escape, escape_length)));
    run_begin = i + 1;
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler, iree_make_string_view((const char*)bytes.data + run_begin,
                                          bytes.data_length - run_begin)));
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, "\"");
}

static iree_status_t iree_vm_bytecode_disassembler_hex_bytes(
    iree_vm_bytecode_disassembler_t* disassembler,
    iree_const_byte_span_t bytes) {
  static const char digits[] = "0123456789ABCDEF";
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "hex\""));
  while (bytes.data_length != 0) {
    char buffer[128];
    const iree_host_size_t byte_count = iree_min(bytes.data_length, 64);
    for (iree_host_size_t i = 0; i < byte_count; ++i) {
      buffer[i * 2] = digits[bytes.data[i] >> 4];
      buffer[i * 2 + 1] = digits[bytes.data[i] & 0x0F];
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
        disassembler, iree_make_string_view(buffer, byte_count * 2)));
    bytes.data += byte_count;
    bytes.data_length -= byte_count;
  }
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, "\"");
}

static iree_status_t iree_vm_bytecode_disassembler_scalar(
    iree_vm_bytecode_disassembler_t* disassembler, uint64_t value,
    uint8_t width, uint8_t radix) {
  if (radix == IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_HEXADECIMAL) {
    return iree_vm_bytecode_disassembler_write_format(
        disassembler, "0x%0*" PRIX64, width * 2, value);
  }
  return iree_vm_bytecode_disassembler_write_format(disassembler, "%" PRIu64,
                                                    value);
}

static uint64_t iree_vm_bytecode_disassembler_field_element(
    const uint8_t* record, const iree_vm_bytecode_disassembler_field_t* field,
    uint8_t element) {
  const uint8_t element_width = field->primary_width / field->element_count;
  return iree_vm_bytecode_disassembler_load(
      record + field->primary_offset + element * element_width, element_width);
}

static const iree_vm_bytecode_disassembler_lane_t*
iree_vm_bytecode_disassembler_lane(
    const iree_vm_bytecode_disassembler_instruction_t* instruction,
    const uint8_t* record) {
  if (instruction->projection_offset ==
      IREE_VM_BYTECODE_DISASSEMBLER_NO_FIELD) {
    return NULL;
  }
  const uint8_t encoded_value = record[instruction->projection_offset];
  for (uint8_t i = 0; i < instruction->lane_count; ++i) {
    const iree_vm_bytecode_disassembler_lane_t* lane =
        &iree_vm_bytecode_disassembler_lanes[instruction->lane_base + i];
    if (lane->value == encoded_value) {
      return lane;
    }
  }
  IREE_ASSERT_UNREACHABLE("verified memory format has no text projection");
  return NULL;
}

static uint32_t iree_vm_bytecode_disassembler_find_block(
    const iree_vm_bytecode_disassembler_function_t* function,
    uint32_t byte_offset) {
  uint32_t lower = 0;
  uint32_t upper = function->block_count;
  while (lower < upper) {
    const uint32_t middle = lower + (upper - lower) / 2;
    if (function->block_offsets[middle] < byte_offset) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  IREE_ASSERT(lower < function->block_count &&
              function->block_offsets[lower] == byte_offset);
  return lower;
}

static iree_status_t iree_vm_bytecode_disassembler_field_value(
    iree_vm_bytecode_disassembler_t* disassembler,
    const iree_vm_bytecode_disassembler_function_t* function,
    const iree_vm_bytecode_disassembler_lane_t* lane, uint32_t record_end,
    const uint8_t* record, const iree_vm_bytecode_disassembler_field_t* field,
    uint8_t element) {
  uint64_t value =
      iree_vm_bytecode_disassembler_field_element(record, field, element);
  switch (field->kind) {
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_VALUE_REGISTER:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "%%v%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_VALUE_REGISTER_RANGE:
      return iree_vm_bytecode_disassembler_write_format(
          disassembler, "%%v%" PRIu64 ".x%" PRIu64, value,
          iree_vm_bytecode_disassembler_load(record + field->related_offset,
                                             field->related_width));
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_VALUE_REGISTER_FORMAT_RANGE:
      return iree_vm_bytecode_disassembler_write_format(
          disassembler, "%%v%" PRIu64 ".x%u", value, lane->lane_count);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_REF_REGISTER:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "%%r%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_FUNCTION_REGISTER:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "%%f%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_SIGNED: {
      const uint8_t element_width = field->primary_width / field->element_count;
      const int64_t signed_value =
          iree_vm_bytecode_disassembler_signed_value(value, element_width);
      return iree_vm_bytecode_disassembler_write_format(
          disassembler, "%" PRId64, signed_value);
    }
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_UNSIGNED:
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_ORDINAL:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_HEX:
      return iree_vm_bytecode_disassembler_scalar(
          disassembler, value, field->primary_width,
          IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_HEXADECIMAL);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_SELECTOR:
      return iree_vm_bytecode_disassembler_numeric(
          disassembler, (uint8_t)field->data, (uint16_t)value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_PACKED_SELECTOR:
      value = (value >> (field->bit_range & 0x0F)) &
              ((UINT64_C(1) << (field->bit_range >> 4)) - 1);
      return iree_vm_bytecode_disassembler_numeric(
          disassembler, (uint8_t)field->data, (uint16_t)value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_BLOCK_TARGET: {
      const int64_t displacement = iree_vm_bytecode_disassembler_signed_value(
          value, field->primary_width);
      const uint32_t target_offset =
          (uint32_t)((int64_t)record_end + displacement * 4);
      return iree_vm_bytecode_disassembler_block_symbol(
          disassembler, (uint8_t)field->data,
          iree_vm_bytecode_disassembler_find_block(function, target_offset));
    }
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_LOCAL_BYTES:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "#v%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_LOCAL_BYTES_RANGE:
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_LOCAL_BYTES_REPEATED_RANGE:
      return iree_vm_bytecode_disassembler_write_format(
          disassembler, "#v%" PRIu64 ".x%" PRIu64, value,
          iree_vm_bytecode_disassembler_load(record + field->related_offset,
                                             field->related_width));
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_LOCAL_BYTES_FIXED_RANGE:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "#v%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_REF_SLOT:
      return iree_vm_bytecode_disassembler_write_format(disassembler,
                                                        "#r%" PRIu64, value);
    case IREE_VM_BYTECODE_DISASSEMBLER_FIELD_MODULE_SYMBOL:
      return iree_vm_bytecode_disassembler_symbol_reference(
          disassembler, (uint8_t)field->data, (uint32_t)value);
    default:
      IREE_ASSERT_UNREACHABLE("composite field reached scalar printer");
      return iree_ok_status();
  }
}

static iree_status_t iree_vm_bytecode_disassembler_instruction_field(
    iree_vm_bytecode_disassembler_t* disassembler,
    const iree_vm_bytecode_disassembler_function_t* function,
    const iree_vm_bytecode_disassembler_instruction_t* instruction,
    const iree_vm_bytecode_disassembler_lane_t* lane, uint32_t record_end,
    const uint8_t* record, const iree_vm_bytecode_disassembler_field_t* field) {
  if (field->kind == IREE_VM_BYTECODE_DISASSEMBLER_FIELD_COMBINED_HEX) {
    const uint64_t value =
        iree_vm_bytecode_disassembler_load(record + field->primary_offset,
                                           field->primary_width) |
        (iree_vm_bytecode_disassembler_load(record + field->related_offset,
                                            field->related_width)
         << (field->primary_width * 8));
    return iree_vm_bytecode_disassembler_scalar(
        disassembler, value, field->primary_width + field->related_width,
        IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_HEXADECIMAL);
  }
  if (field->kind == IREE_VM_BYTECODE_DISASSEMBLER_FIELD_RODATA_RANGE) {
    return iree_vm_bytecode_disassembler_write_format(
        disassembler, "@rodata%" PRIu64 "+%" PRIu64,
        iree_vm_bytecode_disassembler_load(record + field->primary_offset,
                                           field->primary_width),
        iree_vm_bytecode_disassembler_load(record + field->related_offset,
                                           field->related_width));
  }
  if (field->kind == IREE_VM_BYTECODE_DISASSEMBLER_FIELD_DIRECT_TARGET) {
    const uint8_t selector = (uint8_t)iree_vm_bytecode_disassembler_load(
        record + field->primary_offset, field->primary_width);
    uint8_t domain = 0;
    bool found = false;
    for (uint8_t i = 0; i < field->data_count; ++i) {
      const iree_vm_bytecode_disassembler_direct_target_t* target =
          &iree_vm_bytecode_disassembler_direct_targets[field->data + i];
      if (target->value == selector) {
        domain = target->domain;
        found = true;
        break;
      }
    }
    IREE_ASSERT(found);
    return iree_vm_bytecode_disassembler_symbol_reference(
        disassembler, domain,
        (uint32_t)iree_vm_bytecode_disassembler_load(
            record + field->related_offset, field->related_width));
  }
  if (field->kind == IREE_VM_BYTECODE_DISASSEMBLER_FIELD_SWITCH_SLICE) {
    const uint32_t base = (uint32_t)iree_vm_bytecode_disassembler_load(
        record + field->related_offset, field->related_width);
    const uint32_t ordinal = base - function->switch_target_base;
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "targets("));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_reference(
        disassembler, (uint8_t)field->data, ordinal));
    return iree_vm_bytecode_disassembler_write_format(
        disassembler, ", %" PRIu64 ")",
        iree_vm_bytecode_disassembler_load(record + field->primary_offset,
                                           field->primary_width));
  }

  if (field->element_count > 1) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "["));
  }
  for (uint8_t i = 0; i < field->element_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_field_value(
        disassembler, function, lane, record_end, record, field, i));
  }
  if (field->element_count > 1) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "]"));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_disassembler_instruction(
    iree_vm_bytecode_disassembler_t* disassembler,
    const iree_vm_bytecode_disassembler_function_t* function,
    uint32_t record_offset, const uint8_t* record) {
  const iree_vm_bytecode_disassembler_instruction_t* instruction =
      &iree_vm_bytecode_disassembler_instructions[record[0]];
  const iree_vm_bytecode_disassembler_field_t* fields =
      iree_vm_bytecode_disassembler_fields + instruction->field_base;
  const iree_vm_bytecode_disassembler_lane_t* lane =
      iree_vm_bytecode_disassembler_lane(instruction, record);
  const uint32_t record_end = record_offset + instruction->byte_length;

  for (uint8_t i = 0; i < instruction->result_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_instruction_field(
        disassembler, function, instruction, lane, record_end, record,
        &fields[i]));
  }
  if (instruction->result_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, " = "));
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler, iree_vm_bytecode_disassembler_name_at(
                        iree_vm_bytecode_instruction_name_offsets[record[0]])));
  if (lane) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "."));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
        disassembler,
        iree_vm_bytecode_disassembler_text_at(lane->name_offset)));
  }

  const uint8_t operand_end =
      instruction->result_count + instruction->operand_count;
  for (uint8_t i = instruction->result_count; i < operand_end; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write_cstring(
        disassembler, i == instruction->result_count ? " " : ", "));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_instruction_field(
        disassembler, function, instruction, lane, record_end, record,
        &fields[i]));
  }
  if (operand_end < instruction->field_count) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, " {"));
    for (uint8_t i = operand_end; i < instruction->field_count; ++i) {
      if (i != operand_end) {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_disassembler_write_cstring(disassembler, ", "));
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
          disassembler,
          iree_vm_bytecode_disassembler_text_at(fields[i].name_offset)));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, " = "));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_instruction_field(
          disassembler, function, instruction, lane, record_end, record,
          &fields[i]));
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "}"));
  }
  return iree_ok_status();
}

static void iree_vm_bytecode_disassembler_collect_blocks(
    iree_vm_bytecode_disassembler_function_t* function) {
  uint32_t block_ordinal = 0;
  uint32_t offset = 0;
  while (offset < function->bytecode_length) {
    const uint8_t opcode = function->bytecode[offset];
    if (opcode == IREE_VM_BYTECODE_OPCODE_CONTROL_BLOCK) {
      function->block_offsets[block_ordinal++] = offset;
    }
    offset += iree_vm_bytecode_disassembler_instructions[opcode].byte_length;
  }
  IREE_ASSERT(block_ordinal == function->block_count);
}

static iree_status_t iree_vm_bytecode_disassembler_signature_field(
    iree_vm_bytecode_disassembler_t* disassembler, const uint8_t* descriptor,
    uint8_t kind_offset, uint8_t kind_width, uint8_t type_offset,
    uint8_t type_width, uint8_t kind_table, uint8_t ref_domain,
    uint8_t callable_domain) {
  const uint16_t kind = (uint16_t)iree_vm_bytecode_disassembler_load(
      descriptor + kind_offset, kind_width);
  const iree_vm_bytecode_disassembler_numeric_value_t* spelling =
      iree_vm_bytecode_disassembler_find_numeric(kind_table, kind);
  IREE_ASSERT(spelling);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler,
      iree_vm_bytecode_disassembler_text_at(spelling->name_offset)));
  uint8_t domain = 0;
  if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
    domain = ref_domain;
  } else if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
    domain = callable_domain;
  } else {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "("));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_reference(
      disassembler, domain,
      (uint32_t)iree_vm_bytecode_disassembler_load(descriptor + type_offset,
                                                   type_width)));
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, ")");
}

static iree_status_t iree_vm_bytecode_disassembler_signature_fields(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t descriptor_record,
    uint32_t base, uint32_t count, uint8_t kind_offset, uint8_t kind_width,
    uint8_t type_offset, uint8_t type_width, uint8_t kind_table,
    uint8_t ref_domain, uint8_t callable_domain) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "("));
  for (uint32_t i = 0; i < count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_signature_field(
        disassembler,
        iree_vm_bytecode_disassembler_record_at(&disassembler->reader,
                                                descriptor_record, base + i),
        kind_offset, kind_width, type_offset, type_width, kind_table,
        ref_domain, callable_domain));
  }
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, ")");
}

static iree_status_t iree_vm_bytecode_disassembler_signature(
    iree_vm_bytecode_disassembler_t* disassembler, const uint8_t* row,
    const uint8_t** program_ptr) {
  const uint8_t* program = *program_ptr;
  const uint8_t descriptor_record = *program++;
  const uint8_t kind_table = *program++;
  const uint8_t ref_domain = *program++;
  const uint8_t callable_domain = *program++;
  const uint8_t descriptor_base_offset = *program++;
  const uint8_t descriptor_base_width = *program++;
  uint32_t counts[6];
  for (uint8_t i = 0; i < 6; ++i) {
    const uint8_t count_offset = *program++;
    const uint8_t count_width = *program++;
    counts[i] = (uint32_t)iree_vm_bytecode_disassembler_load(row + count_offset,
                                                             count_width);
  }
  const uint8_t kind_offset = *program++;
  const uint8_t kind_width = *program++;
  const uint8_t type_offset = *program++;
  const uint8_t type_width = *program++;
  *program_ptr = program;

  const uint32_t base = (uint32_t)iree_vm_bytecode_disassembler_load(
      row + descriptor_base_offset, descriptor_base_width);
  const uint32_t argument_count = counts[0] + counts[1] + counts[2];
  const uint32_t result_count = counts[3] + counts[4] + counts[5];
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_signature_fields(
      disassembler, descriptor_record, base, argument_count, kind_offset,
      kind_width, type_offset, type_width, kind_table, ref_domain,
      callable_domain));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, " -> "));
  return iree_vm_bytecode_disassembler_signature_fields(
      disassembler, descriptor_record, base + argument_count, result_count,
      kind_offset, kind_width, type_offset, type_width, kind_table, ref_domain,
      callable_domain);
}

static iree_status_t iree_vm_bytecode_disassembler_native_layout(
    iree_vm_bytecode_disassembler_t* disassembler, const uint8_t* row_bytes,
    const uint8_t** program_ptr) {
  const uint8_t* program = *program_ptr;
  const uint8_t kind_table = *program++;
  const uint8_t ref_domain = *program++;
  const uint8_t layout_domain = *program++;
  *program_ptr = program;
  const iree_vm_bytecode_v0_native_layout_row_t* row =
      (const iree_vm_bytecode_v0_native_layout_row_t*)row_bytes;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "("));
  for (uint32_t i = 0; i < row->field_count_u16; ++i) {
    if (i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, ", "));
    }
    const iree_vm_bytecode_v0_native_field_row_t* field =
        (const iree_vm_bytecode_v0_native_field_row_t*)
            iree_vm_bytecode_disassembler_record_at(
                &disassembler->reader,
                IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_FIELD_ROW,
                row->field_base_u32 + i);
    const uint16_t kind = field->kind_u16 & ~IREE_VM_NATIVE_CALL_FIELD_RESULT;
    if (field->kind_u16 & IREE_VM_NATIVE_CALL_FIELD_RESULT) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, "out "));
    }
    if (kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, "["));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_reference(
          disassembler, layout_domain, field->type_ordinal_u16));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, "]"));
    } else {
      const iree_vm_bytecode_disassembler_numeric_value_t* spelling =
          iree_vm_bytecode_disassembler_find_numeric(kind_table, kind);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
          disassembler,
          iree_vm_bytecode_disassembler_text_at(spelling->name_offset)));
      if (kind == IREE_VM_NATIVE_CALL_FIELD_REF) {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_disassembler_write_cstring(disassembler, "("));
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_reference(
            disassembler, ref_domain, field->type_ordinal_u16));
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_disassembler_write_cstring(disassembler, ")"));
      }
    }
  }
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, ")");
}

static iree_status_t iree_vm_bytecode_disassembler_metadata_value(
    iree_vm_bytecode_disassembler_t* disassembler, uint16_t type,
    iree_const_byte_span_t value) {
  if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8) {
    return iree_vm_bytecode_disassembler_quoted_bytes(disassembler, value);
  }
  if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL) {
    return iree_vm_bytecode_disassembler_write_cstring(
        disassembler, value.data[0] ? "true" : "false");
  }
  if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64) {
    return iree_vm_bytecode_disassembler_write_format(
        disassembler, "%" PRId64,
        iree_vm_bytecode_disassembler_signed_value(
            iree_unaligned_load_le_u64(value.data), 8));
  }
  if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64) {
    return iree_vm_bytecode_disassembler_write_format(
        disassembler, "%" PRIu64, iree_unaligned_load_le_u64(value.data));
  }
  return iree_vm_bytecode_disassembler_hex_bytes(disassembler, value);
}

static iree_status_t iree_vm_bytecode_disassembler_grammar(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t grammar_ordinal,
    uint8_t depth);

static iree_status_t iree_vm_bytecode_disassembler_children(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t grammar_ordinal,
    uint32_t count, uint8_t depth) {
  for (uint32_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_indent(disassembler, depth));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_grammar(
        disassembler, grammar_ordinal, depth));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_disassembler_function_body(
    iree_vm_bytecode_disassembler_t* disassembler, const uint8_t* row,
    uint8_t depth, const uint8_t** program_ptr) {
  const uint8_t* program = *program_ptr;
  const uint8_t tail_ordinal = *program++;
  const uint8_t switch_record = *program++;
  const uint8_t switch_domain = *program++;
  const uint8_t block_domain = *program++;
  uint64_t values[5];
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    const uint8_t offset = *program++;
    const uint8_t width = *program++;
    values[i] = iree_vm_bytecode_disassembler_load(row + offset, width);
  }
  const uint8_t switch_offset = *program++;
  const uint8_t switch_width = *program++;
  *program_ptr = program;

  iree_vm_bytecode_disassembler_function_t function;
  function.bytecode = disassembler->reader.tails[tail_ordinal].data + values[0];
  function.bytecode_length = (uint32_t)values[1];
  function.switch_target_base = (uint32_t)values[2];
  function.switch_target_count = (uint32_t)values[3];
  function.block_count = (uint32_t)values[4];
  function.block_offsets = disassembler->block_offsets;
  iree_vm_bytecode_disassembler_collect_blocks(&function);

  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "{\n"));
  if (function.switch_target_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_indent(disassembler, depth + 1));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write_cstring(
        disassembler, "switch_targets {\n"));
    for (uint32_t i = 0; i < function.switch_target_count; ++i) {
      const uint8_t* target = iree_vm_bytecode_disassembler_record_at(
          &disassembler->reader, switch_record,
          function.switch_target_base + i);
      const uint32_t byte_offset = (uint32_t)iree_vm_bytecode_disassembler_load(
                                       target + switch_offset, switch_width) *
                                   4;
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_indent(disassembler, depth + 2));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_reference(
          disassembler, switch_domain, i));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, " = "));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_block_symbol(
          disassembler, block_domain,
          iree_vm_bytecode_disassembler_find_block(&function, byte_offset)));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, "\n"));
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_indent(disassembler, depth + 1));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "}\n"));
  }

  uint32_t block_ordinal = 0;
  uint32_t offset = 0;
  while (offset < function.bytecode_length) {
    const uint8_t* instruction = function.bytecode + offset;
    if (instruction[0] == IREE_VM_BYTECODE_OPCODE_CONTROL_BLOCK) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_indent(disassembler, depth));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_block_symbol(
          disassembler, block_domain, block_ordinal++));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, ":\n"));
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_indent(disassembler, depth + 1));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_instruction(
        disassembler, &function, offset, instruction));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembler_write_cstring(disassembler, "\n"));
    offset +=
        iree_vm_bytecode_disassembler_instructions[instruction[0]].byte_length;
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_indent(disassembler, depth));
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, "}");
}

static iree_status_t iree_vm_bytecode_disassembler_grammar(
    iree_vm_bytecode_disassembler_t* disassembler, uint8_t grammar_ordinal,
    uint8_t depth) {
  iree_vm_bytecode_disassembler_reader_t* reader = &disassembler->reader;
  const iree_vm_bytecode_disassembler_grammar_t* grammar =
      &iree_vm_bytecode_disassembler_grammars[grammar_ordinal];
  uint32_t row_ordinal = 0;
  if (grammar->row_mode == IREE_VM_BYTECODE_DISASSEMBLER_ROW_APPEND) {
    row_ordinal = reader->record_cursors[grammar->record]++;
  }
  const uint8_t* row = iree_vm_bytecode_disassembler_record_at(
      reader, grammar->record, row_ordinal);
  const uint8_t* program =
      iree_vm_bytecode_disassembler_program + grammar->program_offset;
  for (;;) {
    const uint8_t opcode = *program++;
    if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_END) {
      return iree_ok_status();
    } else if (opcode < 0x80) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
          disassembler, iree_make_string_view((const char*)program,
                                              (iree_host_size_t)opcode)));
      program += opcode;
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_DECLARE) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_name(
          disassembler, *program++, row_ordinal));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_SYMBOL ||
               opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_NULLABLE_SYMBOL) {
      const uint8_t domain = *program++;
      const uint8_t offset = *program++;
      const uint8_t width = *program++;
      const uint64_t ordinal =
          iree_vm_bytecode_disassembler_load(row + offset, width);
      if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_NULLABLE_SYMBOL &&
          ordinal == UINT16_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_disassembler_write_cstring(disassembler, "none"));
      } else {
        if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_NULLABLE_SYMBOL) {
          IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_reference(
              disassembler, domain, (uint32_t)ordinal));
        } else {
          IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_symbol_name(
              disassembler, domain, (uint32_t)ordinal));
        }
      }
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_SCALAR) {
      const uint8_t offset = *program++;
      const uint8_t width = *program++;
      const uint8_t radix = *program++;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_scalar(
          disassembler, iree_vm_bytecode_disassembler_load(row + offset, width),
          width, radix));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_NUMERIC) {
      const uint8_t table = *program++;
      const uint8_t offset = *program++;
      const uint8_t width = *program++;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_numeric(
          disassembler, table,
          (uint16_t)iree_vm_bytecode_disassembler_load(row + offset, width)));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_CONSTANT) {
      ++program;
      const uint8_t width = *program++;
      program += width;
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_BODY) {
      const uint8_t child_grammar = *program++;
      const uint8_t child_record = *program++;
      const uint8_t base_offset = *program++;
      const uint8_t base_width = *program++;
      const uint8_t count_offset = *program++;
      const uint8_t count_width = *program++;
      uint32_t child_base = reader->record_cursors[child_record];
      if (base_offset != IREE_VM_BYTECODE_DISASSEMBLER_NO_FIELD) {
        child_base = (uint32_t)iree_vm_bytecode_disassembler_load(
            row + base_offset, base_width);
        reader->record_cursors[child_record] = child_base;
      }
      uint32_t child_count = 0;
      if (count_offset != IREE_VM_BYTECODE_DISASSEMBLER_NO_FIELD) {
        child_count = (uint32_t)iree_vm_bytecode_disassembler_load(
            row + count_offset, count_width);
      } else {
        const uint32_t parent_count = reader->records[grammar->record].count;
        const uint32_t child_end =
            row_ordinal + 1 < parent_count
                ? (uint32_t)iree_vm_bytecode_disassembler_load(
                      iree_vm_bytecode_disassembler_record_at(
                          reader, grammar->record, row_ordinal + 1) +
                          base_offset,
                      base_width)
                : reader->records[child_record].count;
        child_count = child_end - child_base;
      }
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, "{\n"));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_children(
          disassembler, child_grammar, child_count, depth + 1));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_indent(disassembler, depth));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_write_cstring(disassembler, "}"));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_QUOTED_BLOB) {
      const uint8_t tail = *program++;
      const uint8_t offset_record = *program++;
      const uint8_t offset = *program++;
      const uint8_t width = *program++;
      const uint8_t* begin_row = iree_vm_bytecode_disassembler_record_at(
          reader, offset_record, row_ordinal);
      const uint8_t* end_row = iree_vm_bytecode_disassembler_record_at(
          reader, offset_record, row_ordinal + 1);
      const iree_host_size_t begin =
          (iree_host_size_t)iree_vm_bytecode_disassembler_load(
              begin_row + offset, width);
      const iree_host_size_t end =
          (iree_host_size_t)iree_vm_bytecode_disassembler_load(end_row + offset,
                                                               width);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_quoted_bytes(
          disassembler, iree_make_const_byte_span(
                            reader->tails[tail].data + begin, end - begin)));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_NATIVE_CALL) {
      const iree_vm_bytecode_v0_native_callsite_row_t* callsite =
          (const iree_vm_bytecode_v0_native_callsite_row_t*)row;
      const iree_vm_bytecode_native_call_table_t* calls =
          &disassembler->layout->native_calls;
      iree_vm_native_call_schema_t schema = calls->schema;
      schema.root_ordinal =
          calls->imports[callsite->import_ordinal_u16].layout_ordinal_u16;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_native_call(
          &schema,
          &disassembler->layout->functions.rows[callsite->function_ordinal_u16],
          calls->words + callsite->word_offset_u32,
          disassembler->write_callback, disassembler->scratch_allocator));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_NATIVE_LAYOUT) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_native_layout(
          disassembler, row, &program));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_SIGNATURE) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_disassembler_signature(disassembler, row, &program));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_FUNCTION_BODY) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_function_body(
          disassembler, row, depth, &program));
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_HEX_BLOB) {
      const uint8_t tail_ordinal = *program++;
      const uint8_t length_offset = *program++;
      const uint8_t length_width = *program++;
      const uint8_t alignment_offset = *program++;
      const uint8_t alignment_width = *program++;
      iree_vm_bytecode_disassembler_tail_view_t* tail =
          &reader->tails[tail_ordinal];
      const iree_host_size_t alignment =
          (iree_host_size_t)iree_vm_bytecode_disassembler_load(
              row + alignment_offset, alignment_width);
      const iree_host_size_t absolute_offset =
          iree_host_align(tail->section_offset + tail->cursor, alignment);
      tail->cursor = absolute_offset - tail->section_offset;
      const iree_host_size_t length =
          (iree_host_size_t)iree_vm_bytecode_disassembler_load(
              row + length_offset, length_width);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_hex_bytes(
          disassembler,
          iree_make_const_byte_span(tail->data + tail->cursor, length)));
      tail->cursor += length;
    } else if (opcode == IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_METADATA_VALUE) {
      const uint8_t tail = *program++;
      const uint8_t offset_record = *program++;
      const uint8_t type_offset = *program++;
      const uint8_t type_width = *program++;
      const uint8_t offset = *program++;
      const uint8_t width = *program++;
      const uint8_t* begin_row = iree_vm_bytecode_disassembler_record_at(
          reader, offset_record, row_ordinal);
      const uint8_t* end_row = iree_vm_bytecode_disassembler_record_at(
          reader, offset_record, row_ordinal + 1);
      const iree_host_size_t begin =
          (iree_host_size_t)iree_vm_bytecode_disassembler_load(
              begin_row + offset, width);
      const iree_host_size_t end =
          (iree_host_size_t)iree_vm_bytecode_disassembler_load(end_row + offset,
                                                               width);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_metadata_value(
          disassembler,
          (uint16_t)iree_vm_bytecode_disassembler_load(row + type_offset,
                                                       type_width),
          iree_make_const_byte_span(reader->tails[tail].data + begin,
                                    end - begin)));
    } else {
      IREE_ASSERT_UNREACHABLE("generated disassembler program is invalid");
      return iree_ok_status();
    }
  }
}

static uint32_t iree_vm_bytecode_disassembler_run_count(
    const iree_vm_bytecode_disassembler_reader_t* reader,
    const iree_vm_bytecode_disassembler_run_t* run) {
  if (run->kind == IREE_VM_BYTECODE_DISASSEMBLER_RUN_FIXED) {
    return run->value;
  } else if (run->kind == IREE_VM_BYTECODE_DISASSEMBLER_RUN_RECORD) {
    return reader->records[run->source].count;
  } else if (run->kind == IREE_VM_BYTECODE_DISASSEMBLER_RUN_HEADER_FIELD) {
    return (uint32_t)iree_vm_bytecode_disassembler_load(
        reader->records[run->source].rows + run->field_offset,
        run->field_width);
  }
  uint32_t count = 0;
  const iree_vm_bytecode_disassembler_record_view_t* view =
      &reader->records[run->source];
  for (uint32_t i = reader->record_cursors[run->source]; i < view->count; ++i) {
    const uint8_t* row =
        iree_vm_bytecode_disassembler_record_at(reader, run->source, i);
    if (iree_vm_bytecode_disassembler_load(row + run->field_offset,
                                           run->field_width) != run->value) {
      break;
    }
    ++count;
  }
  return count;
}

static iree_status_t iree_vm_bytecode_disassembler_section(
    iree_vm_bytecode_disassembler_t* disassembler,
    const iree_vm_bytecode_v0_section_directory_row_t* row,
    const iree_vm_bytecode_disassembler_section_t* section) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "  "));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler,
      iree_vm_bytecode_disassembler_text_at(
          iree_vm_bytecode_disassembler_shell.literal_offsets[3])));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler,
      iree_vm_bytecode_disassembler_text_at(section->name_offset)));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler,
      iree_vm_bytecode_disassembler_text_at(
          iree_vm_bytecode_disassembler_shell.literal_offsets[4])));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_scalar(
      disassembler,
      iree_vm_bytecode_disassembler_load(
          (const uint8_t*)row +
              iree_vm_bytecode_disassembler_shell.section_alignment_offset,
          iree_vm_bytecode_disassembler_shell.section_alignment_width),
      iree_vm_bytecode_disassembler_shell.section_alignment_width,
      IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_DECIMAL));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler,
      iree_vm_bytecode_disassembler_text_at(
          iree_vm_bytecode_disassembler_shell.literal_offsets[5])));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "\n"));
  for (uint8_t i = 0; i < section->run_count; ++i) {
    const iree_vm_bytecode_disassembler_run_t* run =
        &iree_vm_bytecode_disassembler_runs[section->run_base + i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_children(
        disassembler, run->grammar,
        iree_vm_bytecode_disassembler_run_count(&disassembler->reader, run),
        2));
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_disassembler_write_cstring(disassembler, "  "));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembler_write(
      disassembler,
      iree_vm_bytecode_disassembler_text_at(
          iree_vm_bytecode_disassembler_shell.literal_offsets[6])));
  return iree_vm_bytecode_disassembler_write_cstring(disassembler, "\n\n");
}

IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_disassembler_instruction_name(uint8_t opcode) {
  return iree_vm_bytecode_disassembler_name_at(
      iree_vm_bytecode_instruction_name_offsets[opcode]);
}

IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_disassembler_module_record_name(uint8_t record_ordinal) {
  if (record_ordinal >= IREE_VM_BYTECODE_MODULE_RECORD_COUNT) {
    return iree_string_view_empty();
  }
  return iree_vm_bytecode_disassembler_name_at(
      iree_vm_bytecode_module_record_name_offsets[record_ordinal]);
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_disassemble_module(
    iree_const_byte_span_t contents,
    iree_vm_bytecode_disassembler_write_callback_t write_callback,
    iree_allocator_t scratch_allocator) {
  IREE_ASSERT_ARGUMENT(write_callback.fn);

  iree_vm_bytecode_module_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_module_structure(
      contents, scratch_allocator, &plan));

  uint32_t
      inline_block_offsets[IREE_VM_BYTECODE_DISASSEMBLER_INLINE_BLOCK_COUNT];
  uint32_t* block_offsets = inline_block_offsets;
  iree_status_t status = iree_ok_status();
  if (plan.layout.functions.maximum_block_count >
      IREE_VM_BYTECODE_DISASSEMBLER_INLINE_BLOCK_COUNT) {
    status = iree_allocator_malloc_array_uninitialized(
        scratch_allocator, plan.layout.functions.maximum_block_count,
        sizeof(*block_offsets), (void**)&block_offsets);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_verify_module_instructions(&plan, block_offsets);
  }
  for (uint16_t i = 0;
       i < plan.layout.image.section_count && iree_status_is_ok(status); ++i) {
    const uint16_t section_type =
        plan.layout.image.sections[i].section_type_u16;
    if (!iree_vm_bytecode_disassembler_find_section(section_type)) {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "section 0x%04" PRIX16
                                " has no canonical text representation",
                                section_type);
    }
  }

  iree_vm_bytecode_disassembler_t disassembler = {0};
  const uint8_t* image = NULL;
  if (iree_status_is_ok(status)) {
    disassembler.write_callback = write_callback;
    disassembler.block_offsets = block_offsets;
    disassembler.layout = &plan.layout;
    disassembler.scratch_allocator = scratch_allocator;
    iree_vm_bytecode_disassembler_bind_reader(contents, &plan,
                                              &disassembler.reader);
    image = disassembler.reader
                .records[iree_vm_bytecode_disassembler_shell.image_record]
                .rows;
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_write(
        &disassembler,
        iree_vm_bytecode_disassembler_text_at(
            iree_vm_bytecode_disassembler_shell.literal_offsets[0]));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_scalar(
        &disassembler,
        iree_vm_bytecode_disassembler_load(
            image + iree_vm_bytecode_disassembler_shell.core_major_offset,
            iree_vm_bytecode_disassembler_shell.core_major_width),
        iree_vm_bytecode_disassembler_shell.core_major_width,
        IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_DECIMAL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_write(
        &disassembler,
        iree_vm_bytecode_disassembler_text_at(
            iree_vm_bytecode_disassembler_shell.literal_offsets[1]));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_scalar(
        &disassembler,
        iree_vm_bytecode_disassembler_load(
            image +
                iree_vm_bytecode_disassembler_shell.core_required_minor_offset,
            iree_vm_bytecode_disassembler_shell.core_required_minor_width),
        iree_vm_bytecode_disassembler_shell.core_required_minor_width,
        IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_DECIMAL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_write(
        &disassembler,
        iree_vm_bytecode_disassembler_text_at(
            iree_vm_bytecode_disassembler_shell.literal_offsets[2]));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_write_cstring(&disassembler, "\n");
  }
  for (uint16_t i = 0;
       i < plan.layout.image.section_count && iree_status_is_ok(status); ++i) {
    const iree_vm_bytecode_v0_section_directory_row_t* row =
        &plan.layout.image.sections[i];
    status = iree_vm_bytecode_disassembler_section(
        &disassembler, row,
        iree_vm_bytecode_disassembler_find_section(row->section_type_u16));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_write(
        &disassembler,
        iree_vm_bytecode_disassembler_text_at(
            iree_vm_bytecode_disassembler_shell.literal_offsets[6]));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_disassembler_write_cstring(&disassembler, "\n");
  }

  if (block_offsets != inline_block_offsets) {
    iree_allocator_free(scratch_allocator, block_offsets);
  }
  return status;
}
