// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/vm/bytecode/assembler_lexer.h"
#include "iree/vm/bytecode/assembler_module.h"
#include "iree/vm/bytecode/assembler_symbols.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/bytecode/wire/module.h"
#include "iree/vm/execution.h"

typedef struct iree_vm_bytecode_assembler_numeric_value_t {
  // Offset of the accepted spelling in the generated text pool.
  uint16_t name_offset;
  // Encoded selector or flag bit value.
  uint16_t value;
} iree_vm_bytecode_assembler_numeric_value_t;

typedef struct iree_vm_bytecode_assembler_numeric_table_t {
  // First entry in |iree_vm_bytecode_assembler_numeric_values|.
  uint16_t value_base;
  // Number of entries owned by the table.
  uint16_t value_count;
  // Whether values may be combined as a flag set.
  uint8_t is_flags;
  // Whether unknown nonzero values use the canonical type(N) spelling.
  uint8_t preserves_unknown;
} iree_vm_bytecode_assembler_numeric_table_t;

typedef struct iree_vm_bytecode_assembler_direct_target_t {
  // Selector value encoded beside the resolved function ordinal.
  uint8_t value;
  // Textual symbol domain searched for the referenced name.
  uint8_t domain;
  // Declaration flag bits participating in target selection.
  uint16_t symbol_flags_mask;
  // Required values of the bits selected by |symbol_flags_mask|.
  uint16_t symbol_flags_value;
} iree_vm_bytecode_assembler_direct_target_t;

typedef struct iree_vm_bytecode_assembler_field_t {
  // Offset of the canonical attribute name in the generated text pool.
  uint16_t name_offset;
  // Kind-specific numeric table, symbol domain, or row-table base.
  uint16_t data;
  // Byte offset of the primary field in the instruction record.
  uint8_t primary_offset;
  // Byte offset of the related field, if any.
  uint8_t related_offset;
  // One generated assembler field-kind value.
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
} iree_vm_bytecode_assembler_field_t;

typedef struct iree_vm_bytecode_assembler_lane_t {
  // Memory-format selector encoded by this projected mnemonic suffix.
  uint8_t value;
  // Number of scalar register lanes named by the suffix.
  uint8_t lane_count;
} iree_vm_bytecode_assembler_lane_t;

typedef struct iree_vm_bytecode_assembler_instruction_t {
  // Offset of the canonical mnemonic in the generated text pool.
  uint16_t name_offset;
  // First entry in |iree_vm_bytecode_assembler_fields|.
  uint16_t field_base;
  // Opcode byte written at the beginning of the instruction record.
  uint8_t opcode;
  // Fixed encoded instruction record length in bytes.
  uint8_t byte_length;
  // Result count in the low nibble and operand count in the high nibble.
  uint8_t result_operand_counts;
  // Number of trailing fields parsed as named attributes.
  uint8_t attribute_count;
  // First entry in |iree_vm_bytecode_assembler_lanes|.
  uint8_t lane_base;
  // Number of lane rows beginning at |lane_base|.
  uint8_t lane_count;
  // Generated control-flow classification used for block validation.
  uint8_t control_flow;
  // One-based generated semantic relationship row, or zero when absent.
  uint8_t relation;
} iree_vm_bytecode_assembler_instruction_t;

typedef struct iree_vm_bytecode_assembler_relation_t {
  // Kind-specific generated predicate payload.
  uint64_t data;
  // Five packed physical field offsets and widths selected by |kind|.
  uint8_t fields[5];
  // Generated relationship kind.
  uint8_t kind;
} iree_vm_bytecode_assembler_relation_t;

typedef struct iree_vm_bytecode_assembler_grammar_t {
  // Byte offset of the grammar in the generated format program.
  uint16_t program_offset;
  // Offset of the leading declaration keyword in the generated text pool.
  uint16_t keyword_offset;
  // Physical record ordinal produced by the grammar.
  uint8_t record;
  // Generated append or singleton row mode.
  uint8_t row_mode;
} iree_vm_bytecode_assembler_grammar_t;

typedef struct iree_vm_bytecode_assembler_run_t {
  // Grammar ordinal accepted by the run.
  uint8_t grammar;
  // Whether exactly one declaration must appear.
  uint8_t is_required_singleton;
} iree_vm_bytecode_assembler_run_t;

typedef struct iree_vm_bytecode_assembler_layout_t {
  // Generated record, alignment, or tail layout action.
  uint8_t kind;
  // Destination record or tail ordinal.
  uint8_t target;
  // Signed adjustment applied to a derived record count.
  int8_t adjustment;
  // Fixed record count or alignment, depending on |kind|.
  uint8_t value;
} iree_vm_bytecode_assembler_layout_t;

typedef struct iree_vm_bytecode_assembler_section_t {
  // Offset of the canonical section name in the generated text pool.
  uint16_t name_offset;
  // Encoded section type written to the directory row.
  uint16_t section_type;
  // Required section flags written to the directory row.
  uint16_t required_flags;
  // First entry in |iree_vm_bytecode_assembler_runs|.
  uint8_t run_base;
  // Number of grammar runs beginning at |run_base|.
  uint8_t run_count;
  // First entry in |iree_vm_bytecode_assembler_layouts|.
  uint8_t layout_base;
  // Number of layout actions beginning at |layout_base|.
  uint8_t layout_count;
} iree_vm_bytecode_assembler_section_t;

typedef struct iree_vm_bytecode_assembler_derivation_t {
  // Physical record containing the derived field.
  uint8_t record;
  // Byte offset of the derived field in its record.
  uint8_t field_offset;
  // Byte width of the derived field.
  uint8_t field_width;
  // Generated record-count, grammar-count, or statistic source kind.
  uint8_t kind;
  // Source record, grammar, or statistic ordinal selected by |kind|.
  uint8_t source;
  // One-based scalar constraint row, or zero when unconstrained.
  uint8_t constraint;
} iree_vm_bytecode_assembler_derivation_t;

typedef struct iree_vm_bytecode_assembler_terminal_offset_t {
  // Tail whose final length is materialized as a sentinel offset.
  uint8_t tail;
  // Physical offset-table record receiving the sentinel row.
  uint8_t record;
} iree_vm_bytecode_assembler_terminal_offset_t;

typedef struct iree_vm_bytecode_assembler_shell_t {
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
  uint8_t core_minor_offset;
  // Byte width of the required Core minor revision.
  uint8_t core_minor_width;
  // Byte offset of section alignment in a directory row.
  uint8_t section_alignment_offset;
  // Byte width of section alignment in a directory row.
  uint8_t section_alignment_width;
} iree_vm_bytecode_assembler_shell_t;

#include "iree/vm/bytecode/assembler_tables.inl"

enum {
  IREE_VM_BYTECODE_ASSEMBLER_BLOCK_SIZE = 32 * 1024,
};

typedef enum iree_vm_bytecode_assembler_blob_syntax_t {
  IREE_VM_BYTECODE_ASSEMBLER_BLOB_QUOTED_STRING = 0,
  IREE_VM_BYTECODE_ASSEMBLER_BLOB_QUOTED_BYTES = 1,
  IREE_VM_BYTECODE_ASSEMBLER_BLOB_HEX = 2,
} iree_vm_bytecode_assembler_blob_syntax_t;

typedef struct iree_vm_bytecode_assembler_summary_t {
  // Number of records authored or implied for each wire record kind.
  uint32_t record_counts[IREE_VM_BYTECODE_ASSEMBLER_RECORD_COUNT];
  // Number of invocations of each logical declaration grammar.
  uint32_t grammar_counts[IREE_VM_BYTECODE_ASSEMBLER_GRAMMAR_COUNT];
  // Derived high-water statistics.
  uint32_t statistics[IREE_VM_BYTECODE_ASSEMBLER_STATISTIC_COUNT];
} iree_vm_bytecode_assembler_summary_t;

typedef struct iree_vm_bytecode_assembler_t {
  // Cursor over the complete source text borrowed for both parses.
  iree_vm_bytecode_assembler_lexer_t lexer;
  // First-pass sizing and derived values.
  iree_vm_bytecode_assembler_summary_t summary;
  // Final seekable output stream, or NULL during the sizing pass.
  iree_io_stream_t* stream;
  // Two-pass source symbol and deferred fixup storage.
  iree_vm_bytecode_assembler_symbols_t symbols;
  // Next row ordinal by wire record kind during the second pass.
  uint32_t record_ordinals[IREE_VM_BYTECODE_ASSEMBLER_RECORD_COUNT];
  // Absolute output offsets of wire record arrays in the active image.
  uint64_t record_offsets[IREE_VM_BYTECODE_ASSEMBLER_RECORD_COUNT];
  // Absolute output offsets of byte tails in the active image.
  uint64_t tail_offsets[IREE_VM_BYTECODE_ASSEMBLER_TAIL_COUNT];
  // Current length of each byte tail while emitting declarations.
  uint64_t tail_lengths[IREE_VM_BYTECODE_ASSEMBLER_TAIL_COUNT];
  // Current function-local symbol scope; zero outside a function.
  uint32_t scope;
  // First canonical section ordinal permitted by strict wire ordering.
  uint16_t section_ordinal;
  // Authored alignment of the active section payload.
  uint32_t section_alignment;
  // Current function bytecode tail-relative offset.
  uint32_t instruction_offset;
  // Current function switch-target table base.
  uint32_t switch_target_base;
  // Number of registers in the active function's value bank.
  uint16_t value_register_count;
  // Number of registers in the active function's ref bank.
  uint16_t ref_register_count;
  // Number of registers in the active function's function bank.
  uint16_t function_register_count;
  // Byte capacity of the active function's local byte storage.
  uint16_t local_byte_length;
  // Cell capacity of the active function's local ref storage.
  uint32_t local_ref_count;
  // Cell capacity of the active function's local function storage.
  uint32_t local_function_count;
  // Maximum alignment required by data authored in the active section.
  uint32_t required_section_alignment;
  // True while the second pass emits final records and tails.
  bool is_emitting;
} iree_vm_bytecode_assembler_t;

static iree_string_view_t iree_vm_bytecode_assembler_string_at(
    uint16_t offset) {
  const uint8_t* value = iree_vm_bytecode_assembler_strings + offset;
  return iree_make_string_view((const char*)value + 1, value[0]);
}

static void iree_vm_bytecode_assembler_skip_space(
    iree_vm_bytecode_assembler_t* assembler) {
  iree_vm_bytecode_assembler_lexer_skip_space(&assembler->lexer);
}

static iree_status_t iree_vm_bytecode_assembler_error(
    iree_vm_bytecode_assembler_t* assembler, const char* message) {
  return iree_vm_bytecode_assembler_lexer_error(&assembler->lexer, message);
}

static bool iree_vm_bytecode_assembler_try_literal(
    iree_vm_bytecode_assembler_t* assembler, iree_string_view_t literal) {
  return iree_vm_bytecode_assembler_lexer_try_literal(&assembler->lexer,
                                                      literal);
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_bytecode_assembler_parse_literal(
    iree_vm_bytecode_assembler_t* assembler, uint16_t literal_offset) {
  if (!iree_vm_bytecode_assembler_try_literal(
          assembler, iree_vm_bytecode_assembler_string_at(literal_offset))) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "expected fixed punctuation");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_name(
    iree_vm_bytecode_assembler_t* assembler, iree_string_view_t* out_name) {
  return iree_vm_bytecode_assembler_lexer_parse_name(&assembler->lexer,
                                                     out_name);
}

static iree_status_t iree_vm_bytecode_assembler_parse_unsigned(
    iree_vm_bytecode_assembler_t* assembler, uint8_t width, uint8_t radix,
    uint64_t* out_value) {
  return iree_vm_bytecode_assembler_lexer_parse_unsigned(
      &assembler->lexer, width, (iree_vm_bytecode_assembler_radix_t)radix,
      out_value);
}

static iree_status_t iree_vm_bytecode_assembler_parse_signed(
    iree_vm_bytecode_assembler_t* assembler, uint8_t width,
    uint64_t* out_bits) {
  return iree_vm_bytecode_assembler_lexer_parse_signed(&assembler->lexer, width,
                                                       out_bits);
}

static bool iree_vm_bytecode_assembler_try_char(
    iree_vm_bytecode_assembler_t* assembler, char expected) {
  iree_vm_bytecode_assembler_skip_space(assembler);
  if (assembler->lexer.cursor == assembler->lexer.end ||
      *assembler->lexer.cursor != expected) {
    return false;
  }
  ++assembler->lexer.cursor;
  return true;
}

static iree_status_t iree_vm_bytecode_assembler_parse_char(
    iree_vm_bytecode_assembler_t* assembler, char expected,
    const char* message) {
  if (!iree_vm_bytecode_assembler_try_char(assembler, expected)) {
    return iree_vm_bytecode_assembler_error(assembler, message);
  }
  return iree_ok_status();
}

static void iree_vm_bytecode_assembler_store(uint8_t* data, uint8_t width,
                                             uint64_t value) {
  switch (width) {
    case 1:
      data[0] = (uint8_t)value;
      break;
    case 2:
      iree_unaligned_store_le_u16(data, (uint16_t)value);
      break;
    case 4:
      iree_unaligned_store_le_u32(data, (uint32_t)value);
      break;
    case 8:
      iree_unaligned_store_le_u64(data, value);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("generated assembler field width is invalid");
      break;
  }
}

static uint64_t iree_vm_bytecode_assembler_load(const uint8_t* data,
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
      IREE_ASSERT_UNREACHABLE("generated assembler field width is invalid");
      return 0;
  }
}

static uint8_t iree_vm_bytecode_assembler_result_count(
    const iree_vm_bytecode_assembler_instruction_t* instruction) {
  return instruction->result_operand_counts & 0x0Fu;
}

static uint8_t iree_vm_bytecode_assembler_operand_count(
    const iree_vm_bytecode_assembler_instruction_t* instruction) {
  return instruction->result_operand_counts >> 4;
}

static bool iree_vm_bytecode_assembler_range_fits(uint64_t base,
                                                  uint64_t length,
                                                  uint64_t capacity) {
  return base <= capacity && length <= capacity - base;
}

static bool iree_vm_bytecode_assembler_validate_scalar(uint64_t value,
                                                       uint8_t constraint) {
  if (constraint == 0) {
    return true;
  }
  const uint8_t ordinal = constraint - 1;
  if (value < iree_vm_bytecode_assembler_scalar_minimums[ordinal] ||
      value > iree_vm_bytecode_assembler_scalar_maximums[ordinal]) {
    return false;
  }
  return (iree_vm_bytecode_assembler_scalar_power_of_two_mask &
          ((uint32_t)1u << ordinal)) == 0 ||
         (value != 0 && (value & (value - 1)) == 0);
}

static uint64_t iree_vm_bytecode_assembler_load_relation_field(
    const uint8_t* record, uint8_t field) {
  const uint8_t offset = field & 0x3Fu;
  const uint8_t width = (uint8_t)1u << (field >> 6);
  return iree_vm_bytecode_assembler_load(record + offset, width);
}

static const iree_vm_bytecode_assembler_numeric_value_t*
iree_vm_bytecode_assembler_find_numeric(uint8_t table_ordinal,
                                        iree_string_view_t name) {
  const iree_vm_bytecode_assembler_numeric_table_t* table =
      &iree_vm_bytecode_assembler_numeric_tables[table_ordinal];
  uint16_t lower = 0;
  uint16_t upper = table->value_count;
  while (lower < upper) {
    const uint16_t middle = lower + (upper - lower) / 2;
    const iree_vm_bytecode_assembler_numeric_value_t* value =
        &iree_vm_bytecode_assembler_numeric_values[table->value_base + middle];
    const int comparison = iree_string_view_compare(
        name, iree_vm_bytecode_assembler_string_at(value->name_offset));
    if (comparison < 0) {
      upper = middle;
    } else if (comparison > 0) {
      lower = middle + 1;
    } else {
      return value;
    }
  }
  return NULL;
}

static iree_status_t iree_vm_bytecode_assembler_parse_numeric_name(
    iree_vm_bytecode_assembler_t* assembler, uint8_t table_ordinal,
    uint16_t* out_value) {
  iree_string_view_t name;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_name(assembler, &name));
  const iree_vm_bytecode_assembler_numeric_value_t* value =
      iree_vm_bytecode_assembler_find_numeric(table_ordinal, name);
  if (!value) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "unknown numeric spelling");
  }
  *out_value = value->value;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_numeric(
    iree_vm_bytecode_assembler_t* assembler, uint8_t table_ordinal,
    uint8_t width, uint16_t* out_value) {
  const iree_vm_bytecode_assembler_numeric_table_t* table =
      &iree_vm_bytecode_assembler_numeric_tables[table_ordinal];
  if (table->is_flags) {
    if (!iree_vm_bytecode_assembler_try_char(assembler, '[')) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "expected '[' before flags");
    }
    uint16_t bits = 0;
    iree_vm_bytecode_assembler_skip_space(assembler);
    while (assembler->lexer.cursor < assembler->lexer.end &&
           *assembler->lexer.cursor != ']') {
      uint16_t value = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_numeric_name(
          assembler, table_ordinal, &value));
      if (value == 0 || (bits & value) != 0) {
        return iree_vm_bytecode_assembler_error(
            assembler, "flag list contains zero or a duplicate flag");
      }
      bits |= value;
      if (!iree_vm_bytecode_assembler_try_char(assembler, ',')) {
        break;
      }
    }
    if (!iree_vm_bytecode_assembler_try_char(assembler, ']')) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "expected ']' after flags");
    }
    *out_value = bits;
    return iree_ok_status();
  }

  const char* saved_cursor = assembler->lexer.cursor;
  iree_string_view_t name;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_name(assembler, &name));
  const iree_vm_bytecode_assembler_numeric_value_t* value =
      iree_vm_bytecode_assembler_find_numeric(table_ordinal, name);
  if (value) {
    *out_value = value->value;
    return iree_ok_status();
  }
  if (!table->preserves_unknown ||
      !iree_string_view_equal(name, IREE_SV("type"))) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "unknown numeric spelling");
  }
  assembler->lexer.cursor = saved_cursor;
  iree_vm_bytecode_assembler_skip_space(assembler);
  assembler->lexer.cursor += 4;
  if (!iree_vm_bytecode_assembler_try_char(assembler, '(')) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "expected '(' after type");
  }
  uint64_t raw_value = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
      assembler, width, IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, &raw_value));
  if (raw_value == 0 || !iree_vm_bytecode_assembler_try_char(assembler, ')')) {
    return iree_vm_bytecode_assembler_error(
        assembler, "unknown numeric type must be nonzero and closed by ')'");
  }
  *out_value = (uint16_t)raw_value;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_seek_write(
    iree_vm_bytecode_assembler_t* assembler, uint64_t offset,
    iree_host_size_t length, const void* data) {
  if (!assembler->is_emitting) {
    return iree_ok_status();
  }
  if (offset > INT64_MAX) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "output exceeds stream range");
  }
  IREE_RETURN_IF_ERROR(iree_io_stream_seek(assembler->stream,
                                           IREE_IO_STREAM_SEEK_SET,
                                           (iree_io_stream_pos_t)offset));
  return iree_io_stream_write(assembler->stream, length, data);
}

static iree_status_t iree_vm_bytecode_assembler_seek(
    iree_vm_bytecode_assembler_t* assembler, uint64_t offset) {
  if (offset > INT64_MAX) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "output exceeds stream range");
  }
  return iree_io_stream_seek(assembler->stream, IREE_IO_STREAM_SEEK_SET,
                             (iree_io_stream_pos_t)offset);
}

static iree_status_t iree_vm_bytecode_assembler_seek_read(
    iree_vm_bytecode_assembler_t* assembler, uint64_t offset,
    iree_host_size_t length, void* data) {
  if (offset > INT64_MAX) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "output exceeds stream range");
  }
  IREE_RETURN_IF_ERROR(iree_io_stream_seek(assembler->stream,
                                           IREE_IO_STREAM_SEEK_SET,
                                           (iree_io_stream_pos_t)offset));
  return iree_io_stream_read(assembler->stream, length, data, NULL);
}

static iree_status_t iree_vm_bytecode_assembler_write_value(
    iree_vm_bytecode_assembler_t* assembler, uint64_t offset, uint8_t width,
    uint64_t value) {
  uint8_t bytes[8];
  iree_vm_bytecode_assembler_store(bytes, width, value);
  return iree_vm_bytecode_assembler_seek_write(assembler, offset, width, bytes);
}

static iree_status_t iree_vm_bytecode_assembler_add_count(
    iree_vm_bytecode_assembler_t* assembler, uint32_t* count,
    uint32_t increment) {
  if (*count > UINT32_MAX - increment) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "declaration count overflows u32");
  }
  *count += increment;
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_value_fits(uint64_t value,
                                                  uint8_t width) {
  return width == 8 || value < (UINT64_C(1) << (width * 8));
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_bytecode_assembler_append_record(
    iree_vm_bytecode_assembler_t* assembler, uint8_t record,
    uint32_t* out_ordinal, uint64_t* out_offset) {
  uint32_t ordinal = 0;
  if (assembler->is_emitting) {
    ordinal = assembler->record_ordinals[record]++;
  } else {
    ordinal = assembler->summary.record_counts[record];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_add_count(
        assembler, &assembler->summary.record_counts[record], 1));
  }
  if (out_ordinal) {
    *out_ordinal = ordinal;
  }
  *out_offset =
      assembler->record_offsets[record] +
      (uint64_t)ordinal * iree_vm_bytecode_assembler_record_lengths[record];
  return iree_ok_status();
}

static uint32_t iree_vm_bytecode_assembler_record_count(
    const iree_vm_bytecode_assembler_t* assembler, uint8_t record) {
  return assembler->is_emitting ? assembler->record_ordinals[record]
                                : assembler->summary.record_counts[record];
}

static void iree_vm_bytecode_assembler_add_symbol(
    iree_vm_bytecode_assembler_t* assembler, uint8_t domain,
    iree_string_view_t name, uint32_t ordinal, uint16_t flags, uint32_t value) {
  const iree_vm_bytecode_assembler_symbol_t symbol = {
      .name = name,
      .scope = assembler->scope,
      .ordinal = ordinal,
      .domain = domain,
      .flags = flags,
      .value = value,
  };
  iree_vm_bytecode_assembler_symbols_add_symbol(&assembler->symbols, &symbol);
}

static void iree_vm_bytecode_assembler_add_fixup(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_fixup_t* fixup) {
  iree_vm_bytecode_assembler_symbols_add_fixup(&assembler->symbols, fixup);
}

typedef struct iree_vm_bytecode_assembler_blob_writer_t {
  // Owning assembler and output stream.
  iree_vm_bytecode_assembler_t* assembler;
  // Absolute output offset of the next decoded byte.
  uint64_t output_offset;
} iree_vm_bytecode_assembler_blob_writer_t;

static iree_status_t iree_vm_bytecode_assembler_write_blob_fragment(
    void* user_data, iree_const_byte_span_t fragment) {
  iree_vm_bytecode_assembler_blob_writer_t* writer =
      (iree_vm_bytecode_assembler_blob_writer_t*)user_data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_write(
      writer->assembler, writer->output_offset, fragment.data_length,
      fragment.data));
  writer->output_offset += fragment.data_length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_blob(
    iree_vm_bytecode_assembler_t* assembler, uint8_t tail, uint32_t alignment,
    iree_vm_bytecode_assembler_blob_syntax_t syntax,
    uint64_t* out_relative_offset, uint64_t* out_length) {
  if (!iree_host_size_is_power_of_two(alignment)) {
    return iree_vm_bytecode_assembler_error(
        assembler, "blob alignment is not a power of two");
  }
  if (alignment > assembler->section_alignment) {
    return iree_vm_bytecode_assembler_error(
        assembler, "blob alignment exceeds its section alignment");
  }
  assembler->required_section_alignment =
      iree_max(assembler->required_section_alignment, alignment);
  uint64_t output_offset = assembler->tail_lengths[tail];
  if (assembler->is_emitting &&
      !iree_checked_add_u64(assembler->tail_offsets[tail], output_offset,
                            &output_offset)) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "blob output offset overflows");
  }
  if (!iree_checked_align_u64(output_offset, alignment, &output_offset)) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "blob alignment overflows");
  }
  uint64_t relative_offset = output_offset;
  if (assembler->is_emitting) {
    relative_offset -= assembler->tail_offsets[tail];
  }
  iree_vm_bytecode_assembler_blob_writer_t writer = {
      .assembler = assembler,
      .output_offset = output_offset,
  };
  iree_vm_bytecode_assembler_lexer_write_fn_t write_fn =
      assembler->is_emitting ? iree_vm_bytecode_assembler_write_blob_fragment
                             : NULL;
  uint64_t length = 0;
  if (syntax == IREE_VM_BYTECODE_ASSEMBLER_BLOB_HEX) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_lexer_parse_hex_bytes(
        &assembler->lexer, write_fn, &writer, &length));
  } else if (syntax == IREE_VM_BYTECODE_ASSEMBLER_BLOB_QUOTED_BYTES) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_lexer_parse_quoted_bytes(
        &assembler->lexer, write_fn, &writer, &length));
  } else {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_lexer_parse_quoted_string(
        &assembler->lexer, write_fn, &writer, &length));
  }
  if (!iree_checked_add_u64(relative_offset, length,
                            &assembler->tail_lengths[tail])) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "blob tail length overflows");
  }
  *out_relative_offset = relative_offset;
  *out_length = length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_metadata_value(
    iree_vm_bytecode_assembler_t* assembler, uint8_t tail, uint16_t type,
    uint64_t* out_relative_offset) {
  if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8) {
    uint64_t length = 0;
    return iree_vm_bytecode_assembler_parse_blob(
        assembler, tail, 1, IREE_VM_BYTECODE_ASSEMBLER_BLOB_QUOTED_BYTES,
        out_relative_offset, &length);
  }
  if (type != IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL &&
      type != IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64 &&
      type != IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64) {
    uint64_t length = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_blob(
        assembler, tail, 1, IREE_VM_BYTECODE_ASSEMBLER_BLOB_HEX,
        out_relative_offset, &length));
    if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64 &&
        length != sizeof(uint64_t)) {
      return iree_vm_bytecode_assembler_error(
          assembler, "f64 metadata must contain exactly eight bytes");
    }
    return iree_ok_status();
  }

  uint8_t bytes[8] = {0};
  iree_host_size_t length = sizeof(bytes);
  if (type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL) {
    iree_string_view_t value;
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_name(assembler, &value));
    length = 1;
    if (iree_string_view_equal(value, IREE_SV("true"))) {
      bytes[0] = 1;
    } else if (!iree_string_view_equal(value, IREE_SV("false"))) {
      return iree_vm_bytecode_assembler_error(
          assembler, "Boolean metadata must be true or false");
    }
  } else {
    uint64_t bits = 0;
    IREE_RETURN_IF_ERROR(
        type == IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64
            ? iree_vm_bytecode_assembler_parse_unsigned(
                  assembler, 8, IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL,
                  &bits)
            : iree_vm_bytecode_assembler_parse_signed(assembler, 8, &bits));
    iree_unaligned_store_le_u64(bytes, bits);
  }

  const uint64_t relative_offset = assembler->tail_lengths[tail];
  uint64_t output_offset = relative_offset;
  if (assembler->is_emitting &&
      !iree_checked_add_u64(assembler->tail_offsets[tail], relative_offset,
                            &output_offset)) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "metadata output offset overflows");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_write(
      assembler, output_offset, length, bytes));
  if (!iree_checked_add_u64(relative_offset, length,
                            &assembler->tail_lengths[tail])) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "metadata tail length overflows");
  }
  *out_relative_offset = relative_offset;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_sigil_name(
    iree_vm_bytecode_assembler_t* assembler, char sigil,
    iree_string_view_t* out_name) {
  if (!iree_vm_bytecode_assembler_try_char(assembler, sigil)) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "expected symbolic reference");
  }
  return iree_vm_bytecode_assembler_parse_name(assembler, out_name);
}

static iree_status_t iree_vm_bytecode_assembler_parse_prefixed_ordinal(
    iree_vm_bytecode_assembler_t* assembler, char sigil, char prefix,
    uint8_t width, uint64_t* out_value) {
  iree_vm_bytecode_assembler_skip_space(assembler);
  if (assembler->lexer.end - assembler->lexer.cursor < 3 ||
      assembler->lexer.cursor[0] != sigil ||
      assembler->lexer.cursor[1] != prefix) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "expected physical ordinal");
  }
  assembler->lexer.cursor += 2;
  return iree_vm_bytecode_assembler_parse_unsigned(
      assembler, width, IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, out_value);
}

static iree_status_t iree_vm_bytecode_assembler_parse_dot_count(
    iree_vm_bytecode_assembler_t* assembler, uint8_t width,
    uint64_t* out_count) {
  iree_vm_bytecode_assembler_skip_space(assembler);
  if (assembler->lexer.end - assembler->lexer.cursor < 2 ||
      assembler->lexer.cursor[0] != '.' || assembler->lexer.cursor[1] != 'x') {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "expected '.x' range count");
  }
  assembler->lexer.cursor += 2;
  return iree_vm_bytecode_assembler_parse_unsigned(
      assembler, width, IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, out_count);
}

static const iree_vm_bytecode_assembler_instruction_t*
iree_vm_bytecode_assembler_find_instruction(iree_string_view_t name) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper =
      IREE_ARRAYSIZE(iree_vm_bytecode_assembler_instructions);
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    const iree_vm_bytecode_assembler_instruction_t* instruction =
        &iree_vm_bytecode_assembler_instructions[middle];
    const int comparison = iree_string_view_compare(
        name, iree_vm_bytecode_assembler_string_at(instruction->name_offset));
    if (comparison < 0) {
      upper = middle;
    } else if (comparison > 0) {
      lower = middle + 1;
    } else {
      return instruction;
    }
  }
  return NULL;
}

static iree_status_t iree_vm_bytecode_assembler_parse_symbol_fixup(
    iree_vm_bytecode_assembler_t* assembler, char sigil, uint8_t domain,
    uint8_t width, uint64_t output_offset,
    iree_vm_bytecode_assembler_fixup_kind_t kind, int8_t related_delta,
    uint8_t related_width, uint32_t base, uint32_t ordinal_base) {
  iree_string_view_t name;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_parse_sigil_name(assembler, sigil, &name));
  const iree_vm_bytecode_assembler_fixup_t fixup = {
      .name = name,
      .output_offset = output_offset,
      .scope = domain == IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_BLOCK ||
                       domain == IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_SWITCH_TARGET
                   ? assembler->scope
                   : 0,
      .domain = domain,
      .width = width,
      .kind = (uint8_t)kind,
      .related_delta = related_delta,
      .related_width = related_width,
      .base = base,
      .ordinal_base = ordinal_base,
  };
  iree_vm_bytecode_assembler_add_fixup(assembler, &fixup);
  return iree_ok_status();
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_bytecode_assembler_parse_instruction_field_value(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_instruction_t* instruction,
    const iree_vm_bytecode_assembler_field_t* field, uint8_t element,
    uint64_t record_offset, uint8_t* record) {
  const uint8_t element_width = field->primary_width / field->element_count;
  uint8_t* primary = record + field->primary_offset + element * element_width;
  const uint64_t primary_output_offset =
      record_offset + field->primary_offset + element * element_width;
  uint64_t value = 0;
  switch (field->kind) {
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '%', 'v', element_width, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER_RANGE:
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER_FORMAT_RANGE: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '%', 'v', element_width, &value));
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_dot_count(
          assembler, field->related_width, &count));
      if (field->kind ==
          IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER_RANGE) {
        iree_vm_bytecode_assembler_store(record + field->related_offset,
                                         field->related_width, count);
      } else {
        const iree_vm_bytecode_assembler_lane_t* lane = NULL;
        for (uint8_t i = 0; i < instruction->lane_count; ++i) {
          const iree_vm_bytecode_assembler_lane_t* candidate =
              &iree_vm_bytecode_assembler_lanes[instruction->lane_base + i];
          if (candidate->lane_count == count) {
            lane = candidate;
            break;
          }
        }
        if (!lane) {
          return iree_vm_bytecode_assembler_error(
              assembler, "register range is invalid for the mnemonic");
        }
        iree_vm_bytecode_assembler_store(record + field->related_offset,
                                         field->related_width, lane->value);
      }
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_REF_REGISTER: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '%', 'r', element_width, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_FUNCTION_REGISTER: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '%', 'f', element_width, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_SIGNED: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_signed(
          assembler, element_width, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_UNSIGNED:
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_ORDINAL: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
          assembler, element_width, IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL,
          &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_HEX: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
          assembler, element_width,
          IREE_VM_BYTECODE_ASSEMBLER_SCALAR_HEXADECIMAL, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_COMBINED_HEX: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
          assembler, field->primary_width + field->related_width,
          IREE_VM_BYTECODE_ASSEMBLER_SCALAR_HEXADECIMAL, &value));
      const uint8_t low_bits = field->primary_width * 8;
      iree_vm_bytecode_assembler_store(record + field->related_offset,
                                       field->related_width, value >> low_bits);
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_SELECTOR: {
      uint16_t numeric = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_numeric(
          assembler, (uint8_t)field->data, element_width, &numeric));
      value = numeric;
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_PACKED_SELECTOR: {
      uint16_t numeric = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_numeric(
          assembler, (uint8_t)field->data, element_width, &numeric));
      const uint8_t bit_offset = field->bit_range & 0x0F;
      const uint8_t bit_length = field->bit_range >> 4;
      const uint64_t mask = (UINT64_C(1) << bit_length) - 1;
      value = iree_vm_bytecode_assembler_load(primary, element_width);
      value |= ((uint64_t)numeric & mask) << bit_offset;
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_BLOCK_TARGET:
      return iree_vm_bytecode_assembler_parse_symbol_fixup(
          assembler, '^', IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_BLOCK,
          element_width, primary_output_offset,
          IREE_VM_BYTECODE_ASSEMBLER_FIXUP_BLOCK_DISPLACEMENT, 0, 0,
          assembler->instruction_offset + instruction->byte_length, 0);
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES:
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES_FIXED_RANGE: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '#', 'v', element_width, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES_RANGE:
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES_REPEATED_RANGE: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '#', 'v', element_width, &value));
      uint64_t count = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_dot_count(
          assembler, field->related_width, &count));
      iree_vm_bytecode_assembler_store(record + field->related_offset,
                                       field->related_width, count);
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_REF_SLOT: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_prefixed_ordinal(
          assembler, '#', 'r', element_width, &value));
      break;
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_MODULE_SYMBOL:
      if (field->data == IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_GLOBAL_VALUE) {
        iree_string_view_t name;
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_parse_sigil_name(assembler, '@', &name));
        if (name.size < 3 || name.data[0] != 'g' || name.data[1] != 'v') {
          return iree_vm_bytecode_assembler_error(
              assembler, "value global must use its physical gv ordinal");
        }
        iree_vm_bytecode_assembler_lexer_t ordinal_lexer = {
            .source = name,
            .cursor = name.data + 2,
            .end = name.data + name.size,
        };
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_lexer_parse_unsigned(
            &ordinal_lexer, element_width,
            IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, &value));
        if (ordinal_lexer.cursor != ordinal_lexer.end) {
          return iree_vm_bytecode_assembler_error(
              assembler, "value global ordinal has trailing characters");
        }
        const iree_vm_bytecode_assembler_fixup_t fixup = {
            .output_offset = primary_output_offset,
            .value = (uint32_t)value,
            .domain = (uint8_t)field->data,
            .width = element_width,
            .kind = IREE_VM_BYTECODE_ASSEMBLER_FIXUP_GLOBAL_PARTITION,
            .ordinal_base = field->data_count - 1,
        };
        iree_vm_bytecode_assembler_add_fixup(assembler, &fixup);
        return iree_ok_status();
      }
      if (field->data_count ==
          IREE_VM_BYTECODE_ASSEMBLER_FIELD_DATA_OPTIONAL_IMPORT) {
        return iree_vm_bytecode_assembler_parse_symbol_fixup(
            assembler, '@', (uint8_t)field->data, element_width,
            primary_output_offset,
            IREE_VM_BYTECODE_ASSEMBLER_FIXUP_OPTIONAL_IMPORT, 0, 0, 0, 0);
      }
      return iree_vm_bytecode_assembler_parse_symbol_fixup(
          assembler, '@', (uint8_t)field->data, element_width,
          primary_output_offset,
          field->data_count ? IREE_VM_BYTECODE_ASSEMBLER_FIXUP_GLOBAL_PARTITION
                            : IREE_VM_BYTECODE_ASSEMBLER_FIXUP_ORDINAL,
          0, 0, 0, field->data_count ? field->data_count - 1 : 0);
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_RODATA_RANGE: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol_fixup(
          assembler, '@', (uint8_t)field->data, element_width,
          primary_output_offset, IREE_VM_BYTECODE_ASSEMBLER_FIXUP_ORDINAL, 0, 0,
          0, 0));
      if (!iree_vm_bytecode_assembler_try_char(assembler, '+')) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected rodata byte offset");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
          assembler, field->related_width,
          IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, &value));
      iree_vm_bytecode_assembler_store(record + field->related_offset,
                                       field->related_width, value);
      return iree_ok_status();
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_DIRECT_TARGET: {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol_fixup(
          assembler, '@', IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_DIRECT_TARGET,
          field->related_width, record_offset + field->related_offset,
          IREE_VM_BYTECODE_ASSEMBLER_FIXUP_DIRECT_TARGET,
          (int8_t)((int)field->primary_offset - (int)field->related_offset),
          field->primary_width, field->data_count, field->data));
      return iree_ok_status();
    }
    case IREE_VM_BYTECODE_ASSEMBLER_FIELD_SWITCH_SLICE: {
      if (!iree_vm_bytecode_assembler_try_literal(assembler,
                                                  IREE_SV("targets("))) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected switch target slice");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol_fixup(
          assembler, '@', (uint8_t)field->data, field->related_width,
          record_offset + field->related_offset,
          IREE_VM_BYTECODE_ASSEMBLER_FIXUP_SWITCH_SLICE,
          (int8_t)((int)field->primary_offset - (int)field->related_offset),
          field->primary_width, 0, assembler->switch_target_base));
      if (!iree_vm_bytecode_assembler_try_char(assembler, ',')) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected switch slice count");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
          assembler, element_width, IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL,
          &value));
      if (!iree_vm_bytecode_assembler_try_char(assembler, ')')) {
        return iree_vm_bytecode_assembler_error(
            assembler, "expected ')' after switch slice");
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("generated assembler field kind is invalid");
      return iree_ok_status();
  }
  iree_vm_bytecode_assembler_store(primary, element_width, value);
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_instruction_field(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_instruction_t* instruction,
    const iree_vm_bytecode_assembler_field_t* field, uint64_t record_offset,
    uint8_t* record) {
  if (field->element_count > 1) {
    if (!iree_vm_bytecode_assembler_try_char(assembler, '[')) {
      return iree_vm_bytecode_assembler_error(
          assembler, "expected '[' before field array");
    }
  }
  for (uint8_t i = 0; i < field->element_count; ++i) {
    if (i != 0) {
      if (!iree_vm_bytecode_assembler_try_char(assembler, ',')) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected array separator");
      }
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_instruction_field_value(
            assembler, instruction, field, i, record_offset, record));
  }
  if (field->element_count > 1) {
    if (!iree_vm_bytecode_assembler_try_char(assembler, ']')) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "expected ']' after field array");
    }
  }
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_validate_instruction_fields(
    const iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_instruction_t* instruction,
    const uint8_t* record) {
  const uint8_t field_count =
      iree_vm_bytecode_assembler_result_count(instruction) +
      iree_vm_bytecode_assembler_operand_count(instruction) +
      instruction->attribute_count;
  const iree_vm_bytecode_assembler_field_t* fields =
      iree_vm_bytecode_assembler_fields + instruction->field_base;
  for (uint8_t i = 0; i < field_count; ++i) {
    const iree_vm_bytecode_assembler_field_t* field = &fields[i];
    const uint8_t element_width = field->primary_width / field->element_count;
    for (uint8_t j = 0; j < field->element_count; ++j) {
      const uint64_t value = iree_vm_bytecode_assembler_load(
          record + field->primary_offset + j * element_width, element_width);
      switch (field->kind) {
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER:
          if (value >= assembler->value_register_count) {
            return false;
          }
          break;
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER_RANGE: {
          const uint64_t count = iree_vm_bytecode_assembler_load(
              record + field->related_offset, field->related_width);
          if (count == 0 ||
              !iree_vm_bytecode_assembler_range_fits(
                  value, count, assembler->value_register_count)) {
            return false;
          }
          break;
        }
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_VALUE_REGISTER_FORMAT_RANGE: {
          const uint64_t format = iree_vm_bytecode_assembler_load(
              record + field->related_offset, field->related_width);
          const uint64_t count = format < 16 ? UINT64_C(1) << (format & 3) : 0;
          if (count == 0 ||
              !iree_vm_bytecode_assembler_range_fits(
                  value, count, assembler->value_register_count)) {
            return false;
          }
          break;
        }
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_REF_REGISTER:
          if (value >= assembler->ref_register_count) {
            return false;
          }
          break;
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_FUNCTION_REGISTER:
          if (value >= assembler->function_register_count) {
            return false;
          }
          break;
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_UNSIGNED:
          if (!iree_vm_bytecode_assembler_validate_scalar(
                  value, (uint8_t)field->data)) {
            return false;
          }
          break;
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_ORDINAL:
          if (field->data_count ==
                  IREE_VM_BYTECODE_ASSEMBLER_FIELD_DATA_FUNCTION_LOCAL &&
              value >= assembler->local_function_count) {
            return false;
          }
          break;
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES: {
          const uint64_t format = iree_vm_bytecode_assembler_load(
              record + field->related_offset, field->related_width);
          const uint64_t length = format < 16
                                      ? (UINT64_C(1) << (format >> 2)) *
                                            (UINT64_C(1) << (format & 3))
                                      : 0;
          if (length == 0 || !iree_vm_bytecode_assembler_range_fits(
                                 value, length, assembler->local_byte_length)) {
            return false;
          }
          break;
        }
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES_RANGE: {
          const uint64_t length = iree_vm_bytecode_assembler_load(
              record + field->related_offset, field->related_width);
          if (!iree_vm_bytecode_assembler_range_fits(
                  value, length, assembler->local_byte_length)) {
            return false;
          }
          break;
        }
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES_REPEATED_RANGE: {
          const uint64_t count = iree_vm_bytecode_assembler_load(
              record + field->related_offset, field->related_width);
          const uint64_t length = count * field->data;
          if ((value & (field->data_count - 1)) != 0 ||
              !iree_vm_bytecode_assembler_range_fits(
                  value, length, assembler->local_byte_length)) {
            return false;
          }
          break;
        }
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_LOCAL_BYTES_FIXED_RANGE:
          if ((value & (field->data_count - 1)) != 0 ||
              !iree_vm_bytecode_assembler_range_fits(
                  value, field->data, assembler->local_byte_length)) {
            return false;
          }
          break;
        case IREE_VM_BYTECODE_ASSEMBLER_FIELD_REF_SLOT:
          if (value >= assembler->local_ref_count) {
            return false;
          }
          break;
        default:
          break;
      }
    }
  }
  return true;
}

static bool iree_vm_bytecode_assembler_validate_immediate_relation(
    const iree_vm_bytecode_assembler_relation_t* relation,
    const uint8_t* record) {
  if (relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_FIELDS_DISTINCT) {
    return iree_vm_bytecode_assembler_load_relation_field(
               record, relation->fields[0]) !=
           iree_vm_bytecode_assembler_load_relation_field(record,
                                                          relation->fields[1]);
  }
  if (relation->kind ==
      IREE_VM_BYTECODE_ASSEMBLER_RELATION_INTEGER_BITSTREAM_SHAPE) {
    const uint64_t field_width = iree_vm_bytecode_assembler_load_relation_field(
        record, relation->fields[0]);
    const uint64_t source_count =
        iree_vm_bytecode_assembler_load_relation_field(record,
                                                       relation->fields[1]);
    const uint64_t result_count =
        iree_vm_bytecode_assembler_load_relation_field(record,
                                                       relation->fields[2]);
    const uint64_t source_width =
        iree_vm_bytecode_assembler_load_relation_field(record,
                                                       relation->fields[3]);
    const uint64_t result_width =
        iree_vm_bytecode_assembler_load_relation_field(record,
                                                       relation->fields[4]);
    const bool is_pack = (relation->data & 1) != 0;
    const uint64_t maximum_bits = relation->data >> 1;
    const uint64_t carrier_width = is_pack ? source_width : result_width;
    const uint64_t source_bits =
        source_count * (is_pack ? field_width : source_width);
    const uint64_t result_bits =
        result_count * (is_pack ? result_width : field_width);
    return field_width <= carrier_width && source_bits == result_bits &&
           source_bits <= maximum_bits;
  }
  if (relation->kind ==
      IREE_VM_BYTECODE_ASSEMBLER_RELATION_PACKED_SELECTOR_PAIRS) {
    const uint64_t packed = iree_vm_bytecode_assembler_load_relation_field(
        record, relation->fields[0]);
    const uint16_t components = (uint16_t)(relation->data >> 48);
    const uint8_t first_offset = components & 0x0F;
    const uint8_t first_length = (components >> 4) & 0x0F;
    const uint8_t second_offset = (components >> 8) & 0x0F;
    const uint8_t second_length = components >> 12;
    const uint64_t first =
        (packed >> first_offset) & ((UINT64_C(1) << first_length) - 1);
    const uint64_t second =
        (packed >> second_offset) & ((UINT64_C(1) << second_length) - 1);
    const uint64_t pair = first | (second << first_length);
    const uint64_t pair_mask = relation->data & ((UINT64_C(1) << 48) - 1);
    return pair < 48 && (pair_mask & (UINT64_C(1) << pair)) != 0;
  }
  return true;
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_bytecode_assembler_parse_instruction(
    iree_vm_bytecode_assembler_t* assembler, uint8_t tail_ordinal,
    uint8_t* out_control_flow) {
  iree_vm_bytecode_assembler_skip_space(assembler);
  const char* instruction_begin = assembler->lexer.cursor;
  const char* mnemonic_begin = instruction_begin;
  if (mnemonic_begin < assembler->lexer.end && *mnemonic_begin == '%') {
    while (mnemonic_begin < assembler->lexer.end && *mnemonic_begin != '=') {
      ++mnemonic_begin;
    }
    if (mnemonic_begin == assembler->lexer.end) {
      return iree_vm_bytecode_assembler_error(
          assembler, "instruction results have no mnemonic");
    }
    ++mnemonic_begin;
  }
  iree_vm_bytecode_assembler_lexer_t mnemonic_lexer = assembler->lexer;
  mnemonic_lexer.cursor = mnemonic_begin;
  iree_string_view_t mnemonic;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_lexer_parse_name(&mnemonic_lexer, &mnemonic));
  const iree_vm_bytecode_assembler_instruction_t* instruction =
      iree_vm_bytecode_assembler_find_instruction(mnemonic);
  if (!instruction) {
    assembler->lexer.cursor = mnemonic_lexer.cursor;
    return iree_vm_bytecode_assembler_error(assembler,
                                            "unknown instruction mnemonic");
  }

  uint8_t record[IREE_VM_BYTECODE_ASSEMBLER_MAX_RECORD_SIZE] = {0};
  record[0] = instruction->opcode;
  const uint64_t record_output_offset = assembler->tail_offsets[tail_ordinal] +
                                        assembler->tail_lengths[tail_ordinal];
  const iree_vm_bytecode_assembler_field_t* fields =
      iree_vm_bytecode_assembler_fields + instruction->field_base;
  const uint8_t result_count =
      iree_vm_bytecode_assembler_result_count(instruction);
  const uint8_t operand_count =
      iree_vm_bytecode_assembler_operand_count(instruction);

  assembler->lexer.cursor = instruction_begin;
  for (uint8_t i = 0; i < result_count; ++i) {
    if (i != 0) {
      if (!iree_vm_bytecode_assembler_try_char(assembler, ',')) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected result separator");
      }
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_instruction_field(
        assembler, instruction, &fields[i], record_output_offset, record));
  }
  if (result_count != 0) {
    if (!iree_vm_bytecode_assembler_try_char(assembler, '=')) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "expected '=' after results");
    }
  }
  iree_string_view_t parsed_mnemonic;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_parse_name(assembler, &parsed_mnemonic));

  const uint8_t operand_end = result_count + operand_count;
  for (uint8_t i = result_count; i < operand_end; ++i) {
    if (i != result_count) {
      if (!iree_vm_bytecode_assembler_try_char(assembler, ',')) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected operand separator");
      }
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_instruction_field(
        assembler, instruction, &fields[i], record_output_offset, record));
  }
  if (instruction->attribute_count != 0) {
    if (!iree_vm_bytecode_assembler_try_char(assembler, '{')) {
      return iree_vm_bytecode_assembler_error(
          assembler, "expected instruction attributes");
    }
    for (uint8_t i = operand_end;
         i < operand_end + instruction->attribute_count; ++i) {
      if (i != operand_end) {
        if (!iree_vm_bytecode_assembler_try_char(assembler, ',')) {
          return iree_vm_bytecode_assembler_error(
              assembler, "expected attribute separator");
        }
      }
      iree_string_view_t attribute_name;
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_parse_name(assembler, &attribute_name));
      if (!iree_string_view_equal(
              attribute_name,
              iree_vm_bytecode_assembler_string_at(fields[i].name_offset))) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "unexpected attribute name");
      }
      if (!iree_vm_bytecode_assembler_try_char(assembler, '=')) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "expected attribute value");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_instruction_field(
          assembler, instruction, &fields[i], record_output_offset, record));
    }
    if (!iree_vm_bytecode_assembler_try_char(assembler, '}')) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "expected '}' after attributes");
    }
  }

  if (!assembler->is_emitting) {
    if (!iree_vm_bytecode_assembler_validate_instruction_fields(
            assembler, instruction, record)) {
      assembler->lexer.cursor = instruction_begin;
      return iree_vm_bytecode_assembler_error(
          assembler, "instruction field violates its constraints");
    }
    if (instruction->relation != 0) {
      const iree_vm_bytecode_assembler_relation_t* relation =
          &iree_vm_bytecode_assembler_relations[instruction->relation - 1];
      if (!iree_vm_bytecode_assembler_validate_immediate_relation(relation,
                                                                  record)) {
        assembler->lexer.cursor = instruction_begin;
        return iree_vm_bytecode_assembler_error(
            assembler, "instruction fields violate their relationship");
      }
    }
  }
  if (instruction->relation != 0) {
    const iree_vm_bytecode_assembler_relation_t* relation =
        &iree_vm_bytecode_assembler_relations[instruction->relation - 1];
    if (relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_ABI_SLOT ||
        relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_RODATA_OFFSET ||
        relation->kind ==
            IREE_VM_BYTECODE_ASSEMBLER_RELATION_RODATA_STATIC_OFFSET ||
        relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_CALL ||
        relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_CALL_INDIRECT ||
        relation->kind ==
            IREE_VM_BYTECODE_ASSEMBLER_RELATION_FUNCTION_ADDRESS) {
      const iree_vm_bytecode_assembler_fixup_t fixup = {
          .name = mnemonic,
          .output_offset = record_output_offset,
          .scope = assembler->scope,
          .width = instruction->byte_length,
          .kind = IREE_VM_BYTECODE_ASSEMBLER_FIXUP_RELATION,
          .ordinal_base = instruction->relation - 1,
      };
      iree_vm_bytecode_assembler_add_fixup(assembler, &fixup);
    }
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_write(
      assembler, record_output_offset, instruction->byte_length, record));
  if (assembler->tail_lengths[tail_ordinal] >
      UINT32_MAX - instruction->byte_length) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "function bytecode exceeds u32");
  }
  assembler->tail_lengths[tail_ordinal] += instruction->byte_length;
  assembler->instruction_offset += instruction->byte_length;
  *out_control_flow = instruction->control_flow;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_signature_fields(
    iree_vm_bytecode_assembler_t* assembler, uint8_t descriptor_record,
    uint8_t kind_table, uint8_t ref_domain, uint8_t callable_domain,
    uint8_t kind_offset, uint8_t kind_width, uint8_t type_offset,
    uint8_t type_width, uint32_t counts[3]) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
      assembler, '(', "expected '(' before signature fields"));
  if (iree_vm_bytecode_assembler_try_char(assembler, ')')) {
    return iree_ok_status();
  }
  for (;;) {
    uint16_t kind = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_numeric(
        assembler, kind_table, kind_width, &kind));
    if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_INVALID) {
      return iree_vm_bytecode_assembler_error(
          assembler, "invalid is not a signature field kind");
    }

    uint32_t descriptor_ordinal = 0;
    uint64_t descriptor_offset = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_record(
        assembler, descriptor_record, &descriptor_ordinal, &descriptor_offset));
    uint8_t descriptor[IREE_VM_BYTECODE_ASSEMBLER_MAX_RECORD_SIZE] = {0};
    iree_vm_bytecode_assembler_store(descriptor + kind_offset, kind_width,
                                     kind);
    if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF ||
        kind == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
          assembler, '(', "expected '(' before signature type"));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_symbol_fixup(
          assembler, '@',
          kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF ? ref_domain
                                                      : callable_domain,
          type_width, descriptor_offset + type_offset,
          IREE_VM_BYTECODE_ASSEMBLER_FIXUP_ORDINAL, 0, 0, 0, 0));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
          assembler, ')', "expected ')' after signature type"));
    }
    uint32_t count_ordinal = 0;
    if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
      count_ordinal = 1;
    } else if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      count_ordinal = 2;
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_add_count(
        assembler, &counts[count_ordinal], 1));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_write(
        assembler, descriptor_offset,
        iree_vm_bytecode_assembler_record_lengths[descriptor_record],
        descriptor));

    if (iree_vm_bytecode_assembler_try_char(assembler, ')')) {
      break;
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
        assembler, ',', "expected signature field separator"));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_signature(
    iree_vm_bytecode_assembler_t* assembler, uint32_t descriptor_base,
    uint8_t* row, const uint8_t** program_ptr) {
  const uint8_t* program = *program_ptr;
  const uint8_t descriptor_record = *program++;
  const uint8_t kind_table = *program++;
  const uint8_t ref_domain = *program++;
  const uint8_t callable_domain = *program++;
  const uint8_t descriptor_base_offset = *program++;
  const uint8_t descriptor_base_width = *program++;
  uint8_t count_offsets[6];
  uint8_t count_widths[6];
  for (uint8_t i = 0; i < 6; ++i) {
    count_offsets[i] = *program++;
    count_widths[i] = *program++;
  }
  const uint8_t kind_offset = *program++;
  const uint8_t kind_width = *program++;
  const uint8_t type_offset = *program++;
  const uint8_t type_width = *program++;
  *program_ptr = program;

  iree_vm_bytecode_assembler_store(row + descriptor_base_offset,
                                   descriptor_base_width, descriptor_base);
  uint32_t counts[6] = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_signature_fields(
      assembler, descriptor_record, kind_table, ref_domain, callable_domain,
      kind_offset, kind_width, type_offset, type_width, &counts[0]));
  if (!iree_vm_bytecode_assembler_try_literal(assembler, IREE_SV("->"))) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "expected signature result arrow");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_signature_fields(
      assembler, descriptor_record, kind_table, ref_domain, callable_domain,
      kind_offset, kind_width, type_offset, type_width, &counts[3]));
  if ((uint64_t)counts[0] + counts[1] + counts[2] > UINT16_MAX ||
      (uint64_t)counts[3] + counts[4] + counts[5] > UINT16_MAX) {
    return iree_vm_bytecode_assembler_error(
        assembler, "signature side exceeds the logical field limit");
  }
  for (uint8_t i = 0; i < 6; ++i) {
    if (!iree_vm_bytecode_assembler_value_fits(counts[i], count_widths[i])) {
      return iree_vm_bytecode_assembler_error(
          assembler, "signature field count does not fit its record");
    }
    iree_vm_bytecode_assembler_store(row + count_offsets[i], count_widths[i],
                                     counts[i]);
  }
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_control_flow_is_terminal(
    uint8_t control_flow) {
  return control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_RETURN ||
         control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_YIELD ||
         control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_BRANCH ||
         control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_FAIL;
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_bytecode_assembler_parse_function_body(
    iree_vm_bytecode_assembler_t* assembler, uint32_t function_ordinal,
    uint8_t* row, const uint8_t** program_ptr) {
  const uint8_t* program = *program_ptr;
  const uint8_t tail_ordinal = *program++;
  const uint8_t switch_record = *program++;
  const uint8_t switch_domain = *program++;
  const uint8_t block_domain = *program++;
  uint8_t value_offsets[5];
  uint8_t value_widths[5];
  uint8_t value_constraints[5];
  for (uint8_t i = 0; i < 5; ++i) {
    value_offsets[i] = *program++;
    value_widths[i] = *program++;
    value_constraints[i] = *program++;
  }
  const uint8_t switch_offset = *program++;
  const uint8_t switch_width = *program++;
  *program_ptr = program;

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
      assembler, '{', "expected '{' before function body"));
  const uint32_t saved_scope = assembler->scope;
  assembler->scope = function_ordinal + 1;
  assembler->local_byte_length = (uint16_t)iree_vm_bytecode_assembler_load(
      row + offsetof(iree_vm_bytecode_v0_function_row_t, local_byte_length_u16),
      sizeof(uint16_t));
  assembler->value_register_count = (uint16_t)iree_vm_bytecode_assembler_load(
      row + offsetof(iree_vm_bytecode_v0_function_row_t,
                     value_register_count_u16),
      sizeof(uint16_t));
  assembler->ref_register_count = (uint16_t)iree_vm_bytecode_assembler_load(
      row +
          offsetof(iree_vm_bytecode_v0_function_row_t, ref_register_count_u16),
      sizeof(uint16_t));
  assembler->function_register_count =
      (uint16_t)iree_vm_bytecode_assembler_load(
          row + offsetof(iree_vm_bytecode_v0_function_row_t,
                         function_register_count_u16),
          sizeof(uint16_t));
  assembler->local_ref_count = (uint32_t)iree_vm_bytecode_assembler_load(
      row + offsetof(iree_vm_bytecode_v0_function_row_t, local_ref_count_u32),
      sizeof(uint32_t));
  assembler->local_function_count = (uint32_t)iree_vm_bytecode_assembler_load(
      row + offsetof(iree_vm_bytecode_v0_function_row_t,
                     local_function_count_u32),
      sizeof(uint32_t));
  const uint64_t bytecode_base = assembler->tail_lengths[tail_ordinal];
  assembler->instruction_offset = 0;
  const uint32_t switch_base =
      iree_vm_bytecode_assembler_record_count(assembler, switch_record);
  assembler->switch_target_base = switch_base;
  const iree_host_size_t fixup_base = assembler->symbols.fixup_count;

  if (iree_vm_bytecode_assembler_try_literal(assembler,
                                             IREE_SV("switch_targets"))) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
        assembler, '{', "expected '{' before switch targets"));
    while (!iree_vm_bytecode_assembler_try_char(assembler, '}')) {
      iree_string_view_t target_name;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_sigil_name(
          assembler, '@', &target_name));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
          assembler, '=', "expected '=' in switch target"));
      iree_string_view_t block_name;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_sigil_name(
          assembler, '^', &block_name));

      uint32_t target_ordinal = 0;
      uint64_t target_offset = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_record(
          assembler, switch_record, &target_ordinal, &target_offset));
      iree_vm_bytecode_assembler_add_symbol(assembler, switch_domain,
                                            target_name, target_ordinal, 0, 0);
      const iree_vm_bytecode_assembler_fixup_t fixup = {
          .name = block_name,
          .output_offset = target_offset + switch_offset,
          .scope = assembler->scope,
          .domain = block_domain,
          .width = switch_width,
          .kind = IREE_VM_BYTECODE_ASSEMBLER_FIXUP_SWITCH_TARGET,
      };
      iree_vm_bytecode_assembler_add_fixup(assembler, &fixup);
    }
  }
  const uint32_t switch_end =
      iree_vm_bytecode_assembler_record_count(assembler, switch_record);

  uint32_t block_count = 0;
  uint8_t final_control_flow = IREE_VM_BYTECODE_ASSEMBLER_CONTROL_SEQUENTIAL;
  bool has_call = false;
  bool has_native_call = false;
  while (!iree_vm_bytecode_assembler_try_char(assembler, '}')) {
    iree_string_view_t block_name;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_sigil_name(
        assembler, '^', &block_name));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
        assembler, ':', "expected ':' after block name"));
    iree_vm_bytecode_assembler_add_symbol(assembler, block_domain, block_name,
                                          block_count, 0,
                                          assembler->instruction_offset);
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_add_count(assembler, &block_count, 1));

    bool is_first_instruction = true;
    for (;;) {
      iree_vm_bytecode_assembler_skip_space(assembler);
      if (assembler->lexer.cursor == assembler->lexer.end) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "unterminated function body");
      }
      const char next = *assembler->lexer.cursor;
      if (next == '^' || next == '}') {
        break;
      }
      uint8_t control_flow = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_instruction(
          assembler, tail_ordinal, &control_flow));
      if (is_first_instruction &&
          control_flow != IREE_VM_BYTECODE_ASSEMBLER_CONTROL_BLOCK) {
        return iree_vm_bytecode_assembler_error(
            assembler, "a labeled block must begin with control.block");
      }
      if (!is_first_instruction &&
          control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_BLOCK) {
        return iree_vm_bytecode_assembler_error(
            assembler, "control.block requires a source label");
      }
      is_first_instruction = false;
      has_call |= control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_CALL;
      has_native_call |=
          control_flow == IREE_VM_BYTECODE_ASSEMBLER_CONTROL_NATIVE_CALL;
      final_control_flow = control_flow;
    }
    if (is_first_instruction) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "block has no instructions");
    }
  }
  if (block_count == 0) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "function has no control blocks");
  }
  if (!iree_vm_bytecode_assembler_control_flow_is_terminal(
          final_control_flow)) {
    return iree_vm_bytecode_assembler_error(
        assembler, "function falls through past its final instruction");
  }

  const uint16_t function_flags = (uint16_t)iree_vm_bytecode_assembler_load(
      row + offsetof(iree_vm_bytecode_v0_function_row_t, flags_u16),
      sizeof(uint16_t));
  if (has_call != iree_any_bit_set(function_flags,
                                   IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)) {
    return iree_vm_bytecode_assembler_error(
        assembler, "function call summary does not match its instructions");
  }
  if (has_native_call !=
      iree_any_bit_set(function_flags,
                       IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_NATIVE_CALL)) {
    return iree_vm_bytecode_assembler_error(
        assembler,
        "function native call summary does not match its instructions");
  }

  const uint64_t values[5] = {
      bytecode_base, assembler->instruction_offset,
      switch_base,   switch_end - switch_base,
      block_count,
  };
  for (uint8_t i = 0; i < 5; ++i) {
    if (!iree_vm_bytecode_assembler_value_fits(values[i], value_widths[i]) ||
        !iree_vm_bytecode_assembler_validate_scalar(values[i],
                                                    value_constraints[i])) {
      return iree_vm_bytecode_assembler_error(
          assembler, "function extent does not fit its record");
    }
    iree_vm_bytecode_assembler_store(row + value_offsets[i], value_widths[i],
                                     values[i]);
  }
  if (!assembler->is_emitting) {
    const uint8_t statistic =
        IREE_VM_BYTECODE_ASSEMBLER_STATISTIC_MAXIMUM_BLOCK_COUNT;
    assembler->summary.statistics[statistic] =
        iree_max(assembler->summary.statistics[statistic], block_count);
  } else {
    for (iree_host_size_t i = fixup_base; i < assembler->symbols.fixup_count;
         ++i) {
      iree_vm_bytecode_assembler_fixup_t* fixup = &assembler->symbols.fixups[i];
      if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_SWITCH_SLICE) {
        fixup->base = switch_base;
        fixup->value = switch_end;
      }
    }
  }
  assembler->scope = saved_scope;
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_grammar_matches(
    const iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_grammar_t* grammar) {
  iree_vm_bytecode_assembler_lexer_t lexer = assembler->lexer;
  return iree_vm_bytecode_assembler_lexer_try_literal(
      &lexer, iree_vm_bytecode_assembler_string_at(grammar->keyword_offset));
}

static iree_status_t iree_vm_bytecode_assembler_parse_grammar(
    iree_vm_bytecode_assembler_t* assembler, uint8_t grammar_ordinal);

static iree_status_t iree_vm_bytecode_assembler_parse_children(
    iree_vm_bytecode_assembler_t* assembler, uint8_t grammar_ordinal) {
  const iree_vm_bytecode_assembler_grammar_t* grammar =
      &iree_vm_bytecode_assembler_grammars[grammar_ordinal];
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
      assembler, '{', "expected '{' before nested declarations"));
  while (!iree_vm_bytecode_assembler_try_char(assembler, '}')) {
    if (!iree_vm_bytecode_assembler_grammar_matches(assembler, grammar)) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "unexpected nested declaration");
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_parse_grammar(assembler, grammar_ordinal));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_grammar(
    iree_vm_bytecode_assembler_t* assembler, uint8_t grammar_ordinal) {
  const iree_vm_bytecode_assembler_grammar_t* grammar =
      &iree_vm_bytecode_assembler_grammars[grammar_ordinal];
  uint32_t row_ordinal = 0;
  uint64_t row_offset = assembler->record_offsets[grammar->record];
  uint8_t row[IREE_VM_BYTECODE_ASSEMBLER_MAX_RECORD_SIZE] = {0};
  if (grammar->row_mode == IREE_VM_BYTECODE_ASSEMBLER_ROW_APPEND) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_record(
        assembler, grammar->record, &row_ordinal, &row_offset));
  } else if (assembler->is_emitting) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_read(
        assembler, row_offset,
        iree_vm_bytecode_assembler_record_lengths[grammar->record], row));
  }
  if (!assembler->is_emitting) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_add_count(
        assembler, &assembler->summary.grammar_counts[grammar_ordinal], 1));
  }

  iree_string_view_t declaration_name = iree_string_view_empty();
  uint8_t declaration_domain = IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD;
  uint8_t declaration_flags_offset = 0;
  uint8_t declaration_flags_width = 0;
  const uint8_t* program =
      iree_vm_bytecode_assembler_program + grammar->program_offset;
  for (;;) {
    const uint8_t opcode = *program++;
    if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_END) {
      break;
    }
    if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_LITERAL) {
      const uint16_t literal_offset = iree_unaligned_load_le_u16(program);
      program += 2;
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_parse_literal(assembler, literal_offset));
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_DECLARE ||
               opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_DECLARE_FLAGS) {
      declaration_domain = *program++;
      if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_DECLARE_FLAGS) {
        declaration_flags_offset = *program++;
        declaration_flags_width = *program++;
      }
      if (row_ordinal > UINT16_MAX) {
        return iree_vm_bytecode_assembler_error(
            assembler, "declaration ordinal exceeds u16");
      }
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_parse_name(assembler, &declaration_name));
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_SYMBOL ||
               opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_NULLABLE_SYMBOL) {
      const uint8_t domain = *program++;
      const uint8_t field_offset = *program++;
      const uint8_t field_width = *program++;
      if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_NULLABLE_SYMBOL &&
          iree_vm_bytecode_assembler_try_literal(assembler, IREE_SV("none"))) {
        iree_vm_bytecode_assembler_store(
            row + field_offset, field_width,
            field_width == 8 ? UINT64_MAX
                             : (UINT64_C(1) << (field_width * 8)) - 1);
      } else {
        if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_NULLABLE_SYMBOL) {
          IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_char(
              assembler, '@', "expected '@' before nullable symbol"));
        }
        iree_string_view_t name;
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_parse_name(assembler, &name));
        const iree_vm_bytecode_assembler_fixup_t fixup = {
            .name = name,
            .output_offset = row_offset + field_offset,
            .domain = domain,
            .width = field_width,
            .kind = IREE_VM_BYTECODE_ASSEMBLER_FIXUP_ORDINAL,
        };
        iree_vm_bytecode_assembler_add_fixup(assembler, &fixup);
      }
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_SCALAR) {
      const uint8_t field_offset = *program++;
      const uint8_t field_width = *program++;
      const uint8_t constraint = *program++;
      const uint8_t radix = *program++;
      uint64_t value = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
          assembler, field_width, radix, &value));
      if (!iree_vm_bytecode_assembler_validate_scalar(value, constraint)) {
        return iree_vm_bytecode_assembler_error(
            assembler, "scalar field violates its constraints");
      }
      iree_vm_bytecode_assembler_store(row + field_offset, field_width, value);
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_NUMERIC) {
      const uint8_t table_ordinal = *program++;
      const uint8_t field_offset = *program++;
      const uint8_t field_width = *program++;
      const uint8_t constraint = *program++;
      uint16_t value = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_numeric(
          assembler, table_ordinal, field_width, &value));
      if (!iree_vm_bytecode_assembler_validate_scalar(value, constraint)) {
        return iree_vm_bytecode_assembler_error(
            assembler, "numeric field violates its constraints");
      }
      iree_vm_bytecode_assembler_store(row + field_offset, field_width, value);
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_CONSTANT) {
      const uint8_t field_offset = *program++;
      const uint8_t field_width = *program++;
      memcpy(row + field_offset, program, field_width);
      program += field_width;
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_BODY) {
      const uint8_t child_grammar = *program++;
      const uint8_t child_record = *program++;
      const uint8_t base_offset = *program++;
      const uint8_t base_width = *program++;
      const uint8_t base_constraint = *program++;
      const uint8_t count_offset = *program++;
      const uint8_t count_width = *program++;
      const uint8_t count_constraint = *program++;
      const uint32_t child_base =
          iree_vm_bytecode_assembler_record_count(assembler, child_record);
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_parse_children(assembler, child_grammar));
      const uint32_t child_count =
          iree_vm_bytecode_assembler_record_count(assembler, child_record) -
          child_base;
      if ((base_offset != IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD &&
           (!iree_vm_bytecode_assembler_value_fits(child_base, base_width) ||
            !iree_vm_bytecode_assembler_validate_scalar(child_base,
                                                        base_constraint))) ||
          (count_offset != IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD &&
           (!iree_vm_bytecode_assembler_value_fits(child_count, count_width) ||
            !iree_vm_bytecode_assembler_validate_scalar(child_count,
                                                        count_constraint)))) {
        return iree_vm_bytecode_assembler_error(
            assembler, "nested declaration range exceeds its record");
      }
      if (base_offset != IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD) {
        iree_vm_bytecode_assembler_store(row + base_offset, base_width,
                                         child_base);
      }
      if (count_offset != IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD) {
        iree_vm_bytecode_assembler_store(row + count_offset, count_width,
                                         child_count);
      }
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_QUOTED_BLOB) {
      const uint8_t tail = *program++;
      ++program;
      const uint8_t field_offset = *program++;
      const uint8_t field_width = *program++;
      uint64_t relative_offset = 0;
      uint64_t length = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_blob(
          assembler, tail, 1, IREE_VM_BYTECODE_ASSEMBLER_BLOB_QUOTED_STRING,
          &relative_offset, &length));
      if (!iree_vm_bytecode_assembler_value_fits(relative_offset,
                                                 field_width)) {
        return iree_vm_bytecode_assembler_error(
            assembler, "quoted byte offset exceeds its record");
      }
      iree_vm_bytecode_assembler_store(row + field_offset, field_width,
                                       relative_offset);
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_SIGNATURE) {
      const uint8_t descriptor_record = program[0];
      const uint32_t descriptor_base =
          iree_vm_bytecode_assembler_record_count(assembler, descriptor_record);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_signature(
          assembler, descriptor_base, row, &program));
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_FUNCTION_BODY) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_function_body(
          assembler, row_ordinal, row, &program));
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_HEX_BLOB) {
      const uint8_t tail = *program++;
      const uint8_t length_offset = *program++;
      const uint8_t length_width = *program++;
      const uint8_t alignment_offset = *program++;
      const uint8_t alignment_width = *program++;
      const uint64_t alignment = iree_vm_bytecode_assembler_load(
          row + alignment_offset, alignment_width);
      if (alignment > UINT32_MAX) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "blob alignment exceeds u32");
      }
      uint64_t relative_offset = 0;
      uint64_t length = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_blob(
          assembler, tail, (uint32_t)alignment,
          IREE_VM_BYTECODE_ASSEMBLER_BLOB_HEX, &relative_offset, &length));
      if (!iree_vm_bytecode_assembler_value_fits(length, length_width)) {
        return iree_vm_bytecode_assembler_error(
            assembler, "blob length exceeds its record");
      }
      iree_vm_bytecode_assembler_store(row + length_offset, length_width,
                                       length);
    } else if (opcode == IREE_VM_BYTECODE_ASSEMBLER_FORMAT_METADATA_VALUE) {
      const uint8_t tail = *program++;
      const uint8_t offset_record = *program++;
      const uint8_t type_offset = *program++;
      const uint8_t type_width = *program++;
      const uint8_t offset_field = *program++;
      const uint8_t offset_width = *program++;
      uint64_t offset_row = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_record(
          assembler, offset_record, NULL, &offset_row));
      uint64_t relative_offset = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_metadata_value(
          assembler, tail,
          (uint16_t)iree_vm_bytecode_assembler_load(row + type_offset,
                                                    type_width),
          &relative_offset));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_write_value(
          assembler, offset_row + offset_field, offset_width, relative_offset));
    } else {
      IREE_ASSERT_UNREACHABLE("generated assembler grammar opcode is invalid");
      return iree_ok_status();
    }
  }

  if (declaration_domain != IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD) {
    const uint16_t flags =
        declaration_flags_width
            ? (uint16_t)iree_vm_bytecode_assembler_load(
                  row + declaration_flags_offset, declaration_flags_width)
            : 0;
    iree_vm_bytecode_assembler_add_symbol(
        assembler, declaration_domain, declaration_name, row_ordinal, flags, 0);
  }
  return iree_vm_bytecode_assembler_seek_write(
      assembler, row_offset,
      iree_vm_bytecode_assembler_record_lengths[grammar->record], row);
}

static iree_status_t iree_vm_bytecode_assembler_parse_runs(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_section_t* section) {
  for (uint16_t i = 0; i < section->run_count; ++i) {
    const iree_vm_bytecode_assembler_run_t* run =
        &iree_vm_bytecode_assembler_runs[section->run_base + i];
    const iree_vm_bytecode_assembler_grammar_t* grammar =
        &iree_vm_bytecode_assembler_grammars[run->grammar];
    if (run->is_required_singleton) {
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_parse_grammar(assembler, run->grammar));
    } else {
      while (iree_vm_bytecode_assembler_grammar_matches(assembler, grammar)) {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_assembler_parse_grammar(assembler, run->grammar));
      }
    }
  }
  iree_vm_bytecode_assembler_skip_space(assembler);
  if (assembler->lexer.cursor == assembler->lexer.end ||
      *assembler->lexer.cursor != '}') {
    return iree_vm_bytecode_assembler_error(
        assembler, "unexpected declaration in module section");
  }
  return iree_ok_status();
}

static uint16_t iree_vm_bytecode_assembler_find_section(
    iree_string_view_t name) {
  for (uint16_t i = 0; i < IREE_ARRAYSIZE(iree_vm_bytecode_assembler_sections);
       ++i) {
    if (iree_string_view_equal(
            name, iree_vm_bytecode_assembler_string_at(
                      iree_vm_bytecode_assembler_sections[i].name_offset))) {
      return i;
    }
  }
  return UINT16_MAX;
}

static iree_status_t iree_vm_bytecode_assembler_layout_section(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_section_t* section,
    uint64_t section_offset) {
  uint64_t cursor = section_offset;
  for (uint16_t i = 0; i < section->layout_count; ++i) {
    const iree_vm_bytecode_assembler_layout_t* item =
        &iree_vm_bytecode_assembler_layouts[section->layout_base + i];
    if (item->kind == IREE_VM_BYTECODE_ASSEMBLER_LAYOUT_RECORD) {
      const int64_t count =
          item->value
              ? item->value
              : (int64_t)assembler->summary.record_counts[item->target] +
                    item->adjustment;
      if (count < 0) {
        return iree_vm_bytecode_assembler_error(
            assembler, "section record count is invalid");
      }
      assembler->record_offsets[item->target] = cursor;
      uint64_t byte_length = 0;
      if (!iree_checked_mul_u64(
              (uint64_t)count,
              iree_vm_bytecode_assembler_record_lengths[item->target],
              &byte_length) ||
          !iree_checked_add_u64(cursor, byte_length, &cursor)) {
        return iree_vm_bytecode_assembler_error(
            assembler, "section record layout overflows");
      }
    } else if (item->kind == IREE_VM_BYTECODE_ASSEMBLER_LAYOUT_ALIGN) {
      if (!iree_checked_align_u64(cursor, item->value, &cursor)) {
        return iree_vm_bytecode_assembler_error(assembler,
                                                "section alignment overflows");
      }
    } else {
      assembler->tail_offsets[item->target] = cursor;
      assembler->tail_lengths[item->target] = 0;
    }
  }
  return iree_vm_bytecode_assembler_seek(assembler, cursor);
}

static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_vm_bytecode_assembler_parse_section(
    iree_vm_bytecode_assembler_t* assembler) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[3]));
  iree_string_view_t name;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_name(assembler, &name));
  const uint16_t section_ordinal =
      iree_vm_bytecode_assembler_find_section(name);
  if (section_ordinal == UINT16_MAX) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "section has no assembly grammar");
  }
  if (section_ordinal < assembler->section_ordinal) {
    return iree_vm_bytecode_assembler_error(
        assembler, "sections are not in canonical wire order");
  }
  assembler->section_ordinal = section_ordinal + 1;
  const iree_vm_bytecode_assembler_section_t* section =
      &iree_vm_bytecode_assembler_sections[section_ordinal];

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[4]));
  uint64_t alignment = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
      assembler, iree_vm_bytecode_assembler_shell.section_alignment_width,
      IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, &alignment));
  if (alignment < IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT ||
      !iree_host_size_is_power_of_two((iree_host_size_t)alignment)) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "section alignment is invalid");
  }
  assembler->section_alignment = (uint32_t)alignment;
  assembler->required_section_alignment =
      IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[5]));

  uint64_t directory_offset = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_append_record(
      assembler, iree_vm_bytecode_assembler_shell.section_record, NULL,
      &directory_offset));
  uint64_t section_offset = 0;
  if (assembler->is_emitting) {
    section_offset = (uint64_t)iree_io_stream_length(assembler->stream);
    if (!iree_checked_align_u64(section_offset, alignment, &section_offset)) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "section offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_layout_section(
        assembler, section, section_offset));
  }

  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_parse_runs(assembler, section));
  if (section->section_type == IREE_VM_BYTECODE_SECTION_RODATA &&
      alignment != assembler->required_section_alignment) {
    return iree_vm_bytecode_assembler_error(
        assembler, "rodata section alignment does not match its blocks");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[6]));
  if (assembler->is_emitting) {
    const uint64_t section_end =
        (uint64_t)iree_io_stream_length(assembler->stream);
    if (section_end == section_offset) {
      return iree_vm_bytecode_assembler_error(assembler,
                                              "module section is empty");
    }
    const iree_vm_bytecode_v0_section_directory_row_t directory = {
        .section_type_u16 = section->section_type,
        .section_flags_u16 = section->required_flags,
        .payload_alignment_u32 = (uint32_t)alignment,
        .byte_length_u64 = section_end - section_offset,
    };
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_write(
        assembler, directory_offset, sizeof(directory), &directory));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_parse_module(
    iree_vm_bytecode_assembler_t* assembler) {
  assembler->section_ordinal = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[0]));
  uint64_t core_major = 0;
  uint64_t core_minor = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
      assembler, iree_vm_bytecode_assembler_shell.core_major_width,
      IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, &core_major));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[1]));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_unsigned(
      assembler, iree_vm_bytecode_assembler_shell.core_minor_width,
      IREE_VM_BYTECODE_ASSEMBLER_SCALAR_DECIMAL, &core_minor));
  if (core_major != IREE_VM_BYTECODE_CORE_MAJOR ||
      core_minor > IREE_VM_BYTECODE_CORE_MINOR) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "unsupported VM Core version");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_literal(
      assembler, iree_vm_bytecode_assembler_shell.literal_offsets[2]));

  iree_vm_bytecode_v0_image_header_t header = {0};
  memcpy(header.magic_u8, IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES,
         IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH);
  header.core_major_u16 = (uint16_t)core_major;
  header.core_required_minor_u16 = (uint16_t)core_minor;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_write(
      assembler,
      assembler->record_offsets[iree_vm_bytecode_assembler_shell.image_record],
      sizeof(header), &header));

  while (!iree_vm_bytecode_assembler_try_literal(
      assembler, iree_vm_bytecode_assembler_string_at(
                     iree_vm_bytecode_assembler_shell.literal_offsets[6]))) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_parse_section(assembler));
  }
  iree_vm_bytecode_assembler_skip_space(assembler);
  if (assembler->lexer.cursor != assembler->lexer.end) {
    return iree_vm_bytecode_assembler_error(assembler,
                                            "text follows VM module");
  }
  return iree_ok_status();
}

static uint64_t iree_vm_bytecode_assembler_related_offset(
    const iree_vm_bytecode_assembler_fixup_t* fixup) {
  const int64_t offset = (int64_t)fixup->output_offset + fixup->related_delta;
  IREE_ASSERT_GE(offset, 0);
  return (uint64_t)offset;
}

static iree_status_t iree_vm_bytecode_assembler_read_value(
    iree_vm_bytecode_assembler_t* assembler, uint64_t offset, uint8_t width,
    uint64_t* out_value) {
  uint8_t bytes[8];
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_seek_read(assembler, offset, width, bytes));
  *out_value = iree_vm_bytecode_assembler_load(bytes, width);
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_read_record(
    iree_vm_bytecode_assembler_t* assembler, uint8_t record, uint32_t ordinal,
    void* out_record) {
  const uint64_t offset =
      assembler->record_offsets[record] +
      (uint64_t)ordinal * iree_vm_bytecode_assembler_record_lengths[record];
  return iree_vm_bytecode_assembler_seek_read(
      assembler, offset, iree_vm_bytecode_assembler_record_lengths[record],
      out_record);
}

static uint16_t iree_vm_bytecode_assembler_direct_count(uint16_t argument_count,
                                                        uint16_t result_count) {
  return iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                  iree_max(argument_count, result_count));
}

static uint32_t iree_vm_bytecode_assembler_overflow_count(uint16_t count) {
  return count > IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? (uint32_t)count - IREE_VM_CALL_DIRECT_REGISTER_COUNT
             : 0;
}

static iree_status_t iree_vm_bytecode_assembler_read_callable(
    iree_vm_bytecode_assembler_t* assembler, uint16_t callable_ordinal,
    iree_vm_bytecode_v0_callable_type_row_t* out_callable) {
  return iree_vm_bytecode_assembler_read_record(
      assembler, IREE_VM_BYTECODE_MODULE_RECORD_CALLABLE_TYPE_ROW,
      callable_ordinal, out_callable);
}

static iree_status_t iree_vm_bytecode_assembler_read_signature(
    iree_vm_bytecode_assembler_t* assembler, uint16_t signature_ordinal,
    iree_vm_bytecode_v0_signature_row_t* out_signature) {
  return iree_vm_bytecode_assembler_read_record(
      assembler, IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_ROW,
      signature_ordinal, out_signature);
}

static iree_status_t iree_vm_bytecode_assembler_read_target_callable(
    iree_vm_bytecode_assembler_t* assembler, uint8_t target_kind,
    uint16_t target_ordinal,
    iree_vm_bytecode_v0_callable_type_row_t* out_callable,
    bool* out_may_yield) {
  if (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
    iree_vm_bytecode_v0_function_row_t function;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_record(
        assembler, IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW, target_ordinal,
        &function));
    if (out_may_yield) {
      *out_may_yield = iree_any_bit_set(
          function.flags_u16, IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD);
    }
    return iree_vm_bytecode_assembler_read_callable(
        assembler, function.callable_type_ordinal_u16, out_callable);
  }

  iree_vm_bytecode_v0_import_entry_row_t import;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_record(
      assembler, IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_ENTRY_ROW,
      target_ordinal, &import));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_callable(
      assembler, import.callable_type_ordinal_u16, out_callable));
  if (out_may_yield) {
    *out_may_yield = iree_any_bit_set(
        out_callable->flags_u16, IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD);
  }
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_callable_is_compatible(
    const iree_vm_bytecode_v0_callable_type_row_t* source,
    bool source_may_yield,
    const iree_vm_bytecode_v0_callable_type_row_t* destination) {
  return source->signature_ordinal_u16 == destination->signature_ordinal_u16 &&
         (!source_may_yield ||
          iree_any_bit_set(destination->flags_u16,
                           IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD));
}

static iree_status_t iree_vm_bytecode_assembler_validate_call_packet(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_callable_type_row_t* callable,
    uint16_t direct_ref_move_mask, bool* out_valid) {
  iree_vm_bytecode_v0_signature_row_t signature;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_signature(
      assembler, callable->signature_ordinal_u16, &signature));
  bool valid = function->value_register_count_u16 >=
                   iree_vm_bytecode_assembler_direct_count(
                       signature.argument_value_count_u16,
                       signature.result_value_count_u16) &&
               function->ref_register_count_u16 >=
                   iree_vm_bytecode_assembler_direct_count(
                       signature.argument_ref_count_u16,
                       signature.result_ref_count_u16) &&
               function->function_register_count_u16 >=
                   iree_vm_bytecode_assembler_direct_count(
                       signature.argument_function_count_u16,
                       signature.result_function_count_u16);
  const uint16_t direct_ref_argument_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature.argument_ref_count_u16);
  const uint16_t valid_ref_move_mask =
      direct_ref_argument_count == IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? UINT16_MAX
          : (uint16_t)((1u << direct_ref_argument_count) - 1u);
  valid &= (direct_ref_move_mask & ~valid_ref_move_mask) == 0;
  const uint32_t required_local_bytes =
      sizeof(uint64_t) * (iree_vm_bytecode_assembler_overflow_count(
                              signature.argument_value_count_u16) +
                          iree_vm_bytecode_assembler_overflow_count(
                              signature.result_value_count_u16));
  const uint32_t required_local_refs =
      iree_vm_bytecode_assembler_overflow_count(
          signature.argument_ref_count_u16) +
      iree_vm_bytecode_assembler_overflow_count(signature.result_ref_count_u16);
  const uint32_t required_local_functions =
      iree_vm_bytecode_assembler_overflow_count(
          signature.argument_function_count_u16) +
      iree_vm_bytecode_assembler_overflow_count(
          signature.result_function_count_u16);
  valid &= required_local_bytes <= function->local_byte_length_u16 &&
           required_local_refs <= function->local_ref_count_u32 &&
           required_local_functions <= function->local_function_count_u32;
  *out_valid = valid;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_resolve_relation(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_fixup_t* fixup) {
  const iree_vm_bytecode_assembler_relation_t* relation =
      &iree_vm_bytecode_assembler_relations[fixup->ordinal_base];
  uint8_t record[IREE_VM_BYTECODE_ASSEMBLER_MAX_RECORD_SIZE];
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_seek_read(
      assembler, fixup->output_offset, fixup->width, record));
  iree_vm_bytecode_v0_function_row_t function;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_record(
      assembler, IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW, fixup->scope - 1,
      &function));

  bool valid = false;
  if (relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_ABI_SLOT) {
    iree_vm_bytecode_v0_callable_type_row_t callable;
    iree_vm_bytecode_v0_signature_row_t signature;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_callable(
        assembler, function.callable_type_ordinal_u16, &callable));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_signature(
        assembler, callable.signature_ordinal_u16, &signature));
    const uint16_t count =
        iree_unaligned_load_le_u16((const uint8_t*)&signature + relation->data);
    const uint64_t slot = iree_vm_bytecode_assembler_load_relation_field(
        record, relation->fields[0]);
    valid = slot < iree_vm_bytecode_assembler_overflow_count(count);
  } else if (relation->kind ==
                 IREE_VM_BYTECODE_ASSEMBLER_RELATION_RODATA_OFFSET ||
             relation->kind ==
                 IREE_VM_BYTECODE_ASSEMBLER_RELATION_RODATA_STATIC_OFFSET) {
    const uint64_t offset = iree_vm_bytecode_assembler_load_relation_field(
        record, relation->fields[0]);
    const uint32_t rodata_ordinal =
        (uint32_t)iree_vm_bytecode_assembler_load_relation_field(
            record, relation->fields[1]);
    iree_vm_bytecode_v0_rodata_block_descriptor_t rodata;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_record(
        assembler, IREE_VM_BYTECODE_MODULE_RECORD_RODATA_BLOCK_DESCRIPTOR,
        rodata_ordinal, &rodata));
    if (relation->kind ==
        IREE_VM_BYTECODE_ASSEMBLER_RELATION_RODATA_STATIC_OFFSET) {
      const uint64_t length = iree_vm_bytecode_assembler_load_relation_field(
          record, relation->fields[2]);
      valid = iree_vm_bytecode_assembler_range_fits(offset, length,
                                                    rodata.byte_length_u64);
    } else {
      valid = offset <= rodata.byte_length_u64;
    }
  } else if (relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_CALL ||
             relation->kind ==
                 IREE_VM_BYTECODE_ASSEMBLER_RELATION_CALL_INDIRECT) {
    iree_vm_bytecode_v0_callable_type_row_t callable;
    if (relation->kind == IREE_VM_BYTECODE_ASSEMBLER_RELATION_CALL) {
      const uint8_t target_kind =
          (uint8_t)iree_vm_bytecode_assembler_load_relation_field(
              record, relation->fields[0]);
      const uint16_t target_ordinal =
          (uint16_t)iree_vm_bytecode_assembler_load_relation_field(
              record, relation->fields[1]);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_target_callable(
          assembler, target_kind, target_ordinal, &callable, NULL));
    } else {
      const uint16_t callable_ordinal =
          (uint16_t)iree_vm_bytecode_assembler_load_relation_field(
              record, relation->fields[1]);
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_callable(
          assembler, callable_ordinal, &callable));
    }
    const uint16_t direct_ref_move_mask =
        (uint16_t)iree_vm_bytecode_assembler_load_relation_field(
            record, relation->fields[2]);
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_validate_call_packet(
        assembler, &function, &callable, direct_ref_move_mask, &valid));
  } else if (relation->kind ==
             IREE_VM_BYTECODE_ASSEMBLER_RELATION_FUNCTION_ADDRESS) {
    const uint8_t target_kind =
        (uint8_t)iree_vm_bytecode_assembler_load_relation_field(
            record, relation->fields[0]);
    const uint16_t target_ordinal =
        (uint16_t)iree_vm_bytecode_assembler_load_relation_field(
            record, relation->fields[1]);
    const uint16_t destination_ordinal =
        (uint16_t)iree_vm_bytecode_assembler_load_relation_field(
            record, relation->fields[2]);
    iree_vm_bytecode_v0_callable_type_row_t source;
    iree_vm_bytecode_v0_callable_type_row_t destination;
    bool source_may_yield = false;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_target_callable(
        assembler, target_kind, target_ordinal, &source, &source_may_yield));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_callable(
        assembler, destination_ordinal, &destination));
    valid = iree_vm_bytecode_assembler_callable_is_compatible(
        &source, source_may_yield, &destination);
  } else {
    IREE_ASSERT_UNREACHABLE("generated deferred relation kind is invalid");
    return iree_ok_status();
  }
  if (!valid) {
    assembler->lexer.cursor = fixup->name.data;
    return iree_vm_bytecode_assembler_error(
        assembler, "instruction fields violate their relationship");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_resolve_direct_target(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_fixup_t* fixup) {
  const iree_vm_bytecode_assembler_symbol_t* selected = NULL;
  uint8_t selector = 0;
  for (uint32_t i = 0; i < fixup->base; ++i) {
    const iree_vm_bytecode_assembler_direct_target_t* candidate =
        &iree_vm_bytecode_assembler_direct_targets[fixup->ordinal_base + i];
    const iree_vm_bytecode_assembler_symbol_t* symbol =
        iree_vm_bytecode_assembler_symbols_lookup(
            &assembler->symbols, 0, candidate->domain, fixup->name);
    if (!symbol || (symbol->flags & candidate->symbol_flags_mask) !=
                       candidate->symbol_flags_value) {
      continue;
    }
    if (selected) {
      assembler->lexer.cursor = fixup->name.data;
      return iree_vm_bytecode_assembler_error(
          assembler, "direct target symbol is ambiguous");
    }
    selected = symbol;
    selector = candidate->value;
  }
  if (!selected ||
      !iree_vm_bytecode_assembler_value_fits(selected->ordinal, fixup->width)) {
    assembler->lexer.cursor = fixup->name.data;
    return iree_vm_bytecode_assembler_error(
        assembler, "direct target symbol cannot be resolved");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_write_value(
      assembler, fixup->output_offset, fixup->width, selected->ordinal));
  return iree_vm_bytecode_assembler_write_value(
      assembler, iree_vm_bytecode_assembler_related_offset(fixup),
      fixup->related_width, selector);
}

static iree_status_t iree_vm_bytecode_assembler_resolve_fixup(
    iree_vm_bytecode_assembler_t* assembler,
    const iree_vm_bytecode_assembler_fixup_t* fixup) {
  if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_DIRECT_TARGET) {
    return iree_vm_bytecode_assembler_resolve_direct_target(assembler, fixup);
  }
  if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_GLOBAL_PARTITION &&
      fixup->domain == IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_GLOBAL_VALUE) {
    const uint64_t globals_offset =
        assembler
            ->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_GLOBALS_HEADER];
    if (globals_offset == UINT64_MAX) {
      return iree_vm_bytecode_assembler_error(
          assembler, "value global requires a globals section");
    }
    const uint32_t partition =
        iree_vm_bytecode_assembler_global_partitions[fixup->ordinal_base];
    const uint32_t upper_offset = partition & 0xFFFFu;
    const uint32_t lower_offset_plus_one = partition >> 16;
    uint64_t upper = 0;
    uint64_t lower = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_value(
        assembler, globals_offset + upper_offset, sizeof(uint32_t), &upper));
    if (lower_offset_plus_one != 0) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_value(
          assembler, globals_offset + lower_offset_plus_one - 1,
          sizeof(uint32_t), &lower));
    }
    if (fixup->value < lower || fixup->value >= upper) {
      return iree_vm_bytecode_assembler_error(
          assembler, "value global ordinal is out of range");
    }
    return iree_vm_bytecode_assembler_write_value(
        assembler, fixup->output_offset, fixup->width, fixup->value);
  }

  const iree_vm_bytecode_assembler_symbol_t* symbol =
      iree_vm_bytecode_assembler_symbols_lookup(
          &assembler->symbols, fixup->scope, fixup->domain, fixup->name);
  if (!symbol) {
    assembler->lexer.cursor = fixup->name.data;
    return iree_vm_bytecode_assembler_error(assembler,
                                            "symbol cannot be resolved");
  }

  uint64_t value = symbol->ordinal;
  if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_OPTIONAL_IMPORT) {
    if (!iree_any_bit_set(symbol->flags,
                          IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL)) {
      assembler->lexer.cursor = fixup->name.data;
      return iree_vm_bytecode_assembler_error(
          assembler, "import reference requires an optional declaration");
    }
  } else if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_GLOBAL_PARTITION) {
    const uint64_t globals_offset =
        assembler
            ->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_GLOBALS_HEADER];
    const uint32_t partition =
        iree_vm_bytecode_assembler_global_partitions[fixup->ordinal_base];
    const uint32_t upper_offset = partition & 0xFFFFu;
    const uint32_t lower_offset_plus_one = partition >> 16;
    uint64_t upper = 0;
    uint64_t lower = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_value(
        assembler, globals_offset + upper_offset, sizeof(uint32_t), &upper));
    if (lower_offset_plus_one != 0) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_value(
          assembler, globals_offset + lower_offset_plus_one - 1,
          sizeof(uint32_t), &lower));
    }
    if (value < lower || value >= upper) {
      assembler->lexer.cursor = fixup->name.data;
      return iree_vm_bytecode_assembler_error(
          assembler, "global reference is outside its storage partition");
    }
  } else if (fixup->kind ==
             IREE_VM_BYTECODE_ASSEMBLER_FIXUP_BLOCK_DISPLACEMENT) {
    const int64_t byte_displacement =
        (int64_t)symbol->value - (int64_t)fixup->base;
    if ((byte_displacement & 3) != 0) {
      return iree_vm_bytecode_assembler_error(
          assembler, "branch target is not word aligned");
    }
    const int64_t word_displacement = byte_displacement / 4;
    const uint8_t bit_count = fixup->width * 8;
    const int64_t minimum = -(INT64_C(1) << (bit_count - 1));
    const int64_t maximum = (INT64_C(1) << (bit_count - 1)) - 1;
    if (word_displacement < minimum || word_displacement > maximum) {
      assembler->lexer.cursor = fixup->name.data;
      return iree_vm_bytecode_assembler_error(
          assembler, "branch target exceeds the selected instruction form");
    }
    return iree_vm_bytecode_assembler_write_value(
        assembler, fixup->output_offset, fixup->width,
        (uint64_t)word_displacement);
  } else if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_SWITCH_TARGET) {
    if ((symbol->value & 3) != 0) {
      return iree_vm_bytecode_assembler_error(
          assembler, "switch target is not word aligned");
    }
    value = symbol->value / 4;
  } else if (fixup->kind == IREE_VM_BYTECODE_ASSEMBLER_FIXUP_SWITCH_SLICE) {
    uint64_t count = 0;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_read_value(
        assembler, iree_vm_bytecode_assembler_related_offset(fixup),
        fixup->related_width, &count));
    if (symbol->ordinal < fixup->base || symbol->ordinal >= fixup->value ||
        count > fixup->value - symbol->ordinal) {
      return iree_vm_bytecode_assembler_error(
          assembler, "switch target slice is outside its function");
    }
    value = symbol->ordinal;
  }
  if (!iree_vm_bytecode_assembler_value_fits(value, fixup->width)) {
    assembler->lexer.cursor = fixup->name.data;
    return iree_vm_bytecode_assembler_error(
        assembler, "resolved symbol ordinal exceeds its encoded field");
  }
  return iree_vm_bytecode_assembler_write_value(assembler, fixup->output_offset,
                                                fixup->width, value);
}

static iree_status_t iree_vm_bytecode_assembler_resolve_fixups(
    iree_vm_bytecode_assembler_t* assembler) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_symbols_prepare(&assembler->symbols));
  for (iree_host_size_t i = 0; i < assembler->symbols.fixup_count; ++i) {
    if (assembler->symbols.fixups[i].kind ==
        IREE_VM_BYTECODE_ASSEMBLER_FIXUP_RELATION) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_resolve_fixup(
        assembler, &assembler->symbols.fixups[i]));
  }
  for (iree_host_size_t i = 0; i < assembler->symbols.fixup_count; ++i) {
    if (assembler->symbols.fixups[i].kind !=
        IREE_VM_BYTECODE_ASSEMBLER_FIXUP_RELATION) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_resolve_relation(
        assembler, &assembler->symbols.fixups[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_finalize(
    iree_vm_bytecode_assembler_t* assembler) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_vm_bytecode_assembler_derivations); ++i) {
    const iree_vm_bytecode_assembler_derivation_t* derivation =
        &iree_vm_bytecode_assembler_derivations[i];
    if (assembler->record_offsets[derivation->record] == UINT64_MAX) {
      continue;
    }
    uint64_t value = 0;
    if (derivation->kind ==
        IREE_VM_BYTECODE_ASSEMBLER_DERIVATION_RECORD_COUNT) {
      value = assembler->summary.record_counts[derivation->source];
    } else if (derivation->kind ==
               IREE_VM_BYTECODE_ASSEMBLER_DERIVATION_GRAMMAR_COUNT) {
      value = assembler->summary.grammar_counts[derivation->source];
    } else {
      value = assembler->summary.statistics[derivation->source];
    }
    if (!iree_vm_bytecode_assembler_value_fits(value,
                                               derivation->field_width) ||
        !iree_vm_bytecode_assembler_validate_scalar(value,
                                                    derivation->constraint)) {
      return iree_vm_bytecode_assembler_error(
          assembler, "derived value exceeds its encoded field");
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_write_value(
        assembler,
        assembler->record_offsets[derivation->record] +
            derivation->field_offset,
        derivation->field_width, value));
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_vm_bytecode_assembler_terminal_offsets); ++i) {
    const iree_vm_bytecode_assembler_terminal_offset_t* terminal =
        &iree_vm_bytecode_assembler_terminal_offsets[i];
    if (assembler->record_offsets[terminal->record] == UINT64_MAX) {
      continue;
    }
    const uint64_t row_offset =
        assembler->record_offsets[terminal->record] +
        (uint64_t)assembler->summary.record_counts[terminal->record] *
            iree_vm_bytecode_assembler_record_lengths[terminal->record];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_write_value(
        assembler, row_offset,
        iree_vm_bytecode_assembler_record_lengths[terminal->record],
        assembler->tail_lengths[terminal->tail]));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_assemble_module(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_byte_sequence_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = NULL;
  if (!source.data && source.size != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM assembly source is invalid");
  }
  if (!iree_unicode_utf8_validate(source)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM assembly source is not valid UTF-8");
  }

  iree_vm_bytecode_assembler_t assembler = {0};
  iree_vm_bytecode_assembler_symbols_initialize_counting(&assembler.symbols);
  iree_vm_bytecode_assembler_lexer_initialize(source, &assembler.lexer);
  iree_status_t status = iree_vm_bytecode_assembler_parse_module(&assembler);

  iree_host_size_t storage_size = 0;
  iree_host_size_t symbol_offset = 0;
  iree_host_size_t fixup_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        0, &storage_size,
        IREE_STRUCT_FIELD_ALIGNED(
            assembler.symbols.symbol_count, iree_vm_bytecode_assembler_symbol_t,
            iree_alignof(iree_vm_bytecode_assembler_symbol_t), &symbol_offset),
        IREE_STRUCT_FIELD_ALIGNED(
            assembler.symbols.fixup_count, iree_vm_bytecode_assembler_fixup_t,
            iree_alignof(iree_vm_bytecode_assembler_fixup_t), &fixup_offset));
  }
  uint8_t* storage = NULL;
  if (iree_status_is_ok(status) && storage_size != 0) {
    status = iree_allocator_malloc_uninitialized(host_allocator, storage_size,
                                                 (void**)&storage);
  }

  if (iree_status_is_ok(status)) {
    iree_vm_bytecode_assembler_symbols_initialize_recording(
        assembler.symbols.symbol_count,
        storage
            ? (iree_vm_bytecode_assembler_symbol_t*)(storage + symbol_offset)
            : NULL,
        assembler.symbols.fixup_count,
        storage ? (iree_vm_bytecode_assembler_fixup_t*)(storage + fixup_offset)
                : NULL,
        &assembler.symbols);
    status = iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_RESIZABLE | IREE_IO_STREAM_MODE_SEEKABLE,
        IREE_VM_BYTECODE_ASSEMBLER_BLOCK_SIZE, host_allocator,
        &assembler.stream);
  }

  if (iree_status_is_ok(status)) {
    memset(assembler.record_ordinals, 0, sizeof(assembler.record_ordinals));
    memset(assembler.record_offsets, 0xFF, sizeof(assembler.record_offsets));
    memset(assembler.tail_offsets, 0xFF, sizeof(assembler.tail_offsets));
    memset(assembler.tail_lengths, 0, sizeof(assembler.tail_lengths));
    assembler.record_offsets[iree_vm_bytecode_assembler_shell.image_record] = 0;
    assembler.record_offsets[iree_vm_bytecode_assembler_shell.section_record] =
        sizeof(iree_vm_bytecode_v0_image_header_t);
    const uint64_t directory_end =
        sizeof(iree_vm_bytecode_v0_image_header_t) +
        (uint64_t)
                assembler.summary.record_counts[iree_vm_bytecode_assembler_shell
                                                    .section_record] *
            sizeof(iree_vm_bytecode_v0_section_directory_row_t);
    assembler.is_emitting = true;
    iree_vm_bytecode_assembler_lexer_initialize(source, &assembler.lexer);
    status = iree_vm_bytecode_assembler_seek(&assembler, directory_end);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_parse_module(&assembler);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_resolve_fixups(&assembler);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_assembler_finalize(&assembler);
  }
  if (iree_status_is_ok(status)) {
    const iree_vm_bytecode_assembler_module_view_t view = {
        .stream = assembler.stream,
        .record_offsets = assembler.record_offsets,
        .record_counts = assembler.summary.record_counts,
        .string_data_offset =
            assembler.tail_offsets[IREE_VM_BYTECODE_ASSEMBLER_TAIL_STRINGS],
    };
    status = iree_vm_bytecode_assembler_module_validate(&view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(assembler.stream, out_contents);
  }

  iree_io_stream_release(assembler.stream);
  iree_allocator_free(host_allocator, storage);
  return status;
}
