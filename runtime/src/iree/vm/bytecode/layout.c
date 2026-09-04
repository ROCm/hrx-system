// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/layout.h"

#include <string.h>

#include "iree/vm/bytecode/layout/data.inl"

#if !defined(IREE_ENDIANNESS_LITTLE)
#error "IREE VM bytecode version zero requires a little-endian host"
#endif

typedef struct iree_vm_bytecode_cursor_t {
  // Exact bytes being consumed.
  iree_const_byte_span_t span;
  // Next unconsumed byte offset.
  iree_host_size_t offset;
  // Diagnostic name of the containing structure.
  const char* name;
} iree_vm_bytecode_cursor_t;

// Returns the required flags from a packed known-section descriptor.
static uint16_t iree_vm_bytecode_section_required_flags(uint32_t descriptor) {
  return (uint16_t)descriptor;
}

// Returns the introducing Core minor from a packed known-section descriptor.
static uint16_t iree_vm_bytecode_section_since_minor(uint32_t descriptor) {
  return (uint16_t)(descriptor >> 16);
}

static bool iree_vm_bytecode_bytes_are_zero(const uint8_t* data,
                                            iree_host_size_t length) {
  for (iree_host_size_t i = 0; i < length; ++i) {
    if (data[i] != 0) return false;
  }
  return true;
}

static iree_status_t iree_vm_bytecode_cursor_take(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t length,
    iree_host_size_t alignment, iree_const_byte_span_t* out_span) {
  if (!iree_host_size_has_alignment(cursor->offset, alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s offset %" PRIhsz " is not %" PRIhsz
                            "-byte aligned",
                            cursor->name, cursor->offset, alignment);
  }
  if (cursor->offset > cursor->span.data_length ||
      length > cursor->span.data_length - cursor->offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is truncated at byte %" PRIhsz, cursor->name,
                            cursor->offset);
  }
  *out_span =
      iree_make_const_byte_span(cursor->span.data + cursor->offset, length);
  cursor->offset += length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_cursor_take_array(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t count,
    iree_host_size_t element_size, iree_host_size_t alignment,
    iree_const_byte_span_t* out_span) {
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_mul(count, element_size, &length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s array byte length overflows", cursor->name);
  }
  return iree_vm_bytecode_cursor_take(cursor, length, alignment, out_span);
}

static iree_status_t iree_vm_bytecode_cursor_take_remaining_array(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t element_size,
    iree_host_size_t alignment, iree_const_byte_span_t* out_span,
    uint32_t* out_count) {
  const iree_host_size_t remaining_length =
      cursor->span.data_length - cursor->offset;
  if (remaining_length % element_size != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s rows do not consume the section", cursor->name);
  }
  const iree_host_size_t count = remaining_length / element_size;
  if (count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s row count exceeds uint32_t", cursor->name);
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      cursor, count, element_size, alignment, out_span));
  *out_count = (uint32_t)count;
  return iree_ok_status();
}

#define IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(cursor, count, type, out_data)   \
  do {                                                                      \
    iree_const_byte_span_t array_span;                                      \
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(                \
        (cursor), (count), sizeof(type), iree_alignof(type), &array_span)); \
    *(out_data) = (const type*)array_span.data;                             \
  } while (0)

#define IREE_VM_BYTECODE_CURSOR_TAKE_REMAINING_ARRAY(cursor, type, out_data, \
                                                     out_count)              \
  do {                                                                       \
    iree_const_byte_span_t array_span;                                       \
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_remaining_array(       \
        (cursor), sizeof(type), iree_alignof(type), &array_span,             \
        (out_count)));                                                       \
    *(out_data) = (const type*)array_span.data;                              \
  } while (0)

static iree_status_t iree_vm_bytecode_cursor_align(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t alignment) {
  if (!iree_host_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s alignment is not a power of two", cursor->name);
  }
  iree_host_size_t aligned_offset = 0;
  if (!iree_host_size_checked_align(cursor->offset, alignment,
                                    &aligned_offset) ||
      aligned_offset > cursor->span.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s alignment overflows its extent", cursor->name);
  }
  const iree_host_size_t padding_length = aligned_offset - cursor->offset;
  if (!iree_vm_bytecode_bytes_are_zero(cursor->span.data + cursor->offset,
                                       padding_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s has nonzero alignment padding", cursor->name);
  }
  cursor->offset = aligned_offset;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_cursor_finish(
    const iree_vm_bytecode_cursor_t* cursor) {
  if (cursor->offset != cursor->span.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s has %" PRIhsz " trailing bytes", cursor->name,
                            cursor->span.data_length - cursor->offset);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_requirements(
    iree_const_byte_span_t span,
    iree_vm_bytecode_requirement_table_t* out_table) {
  if (span.data_length % sizeof(iree_vm_bytecode_v0_requirement_row_t) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Requirements rows do not consume the section");
  }
  const iree_host_size_t count =
      span.data_length / sizeof(iree_vm_bytecode_v0_requirement_row_t);
  if (count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Requirements row count exceeds uint16_t");
  }
  out_table->rows = (const iree_vm_bytecode_v0_requirement_row_t*)span.data;
  out_table->count = (uint16_t)count;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_strings(
    iree_const_byte_span_t span, iree_vm_bytecode_string_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Strings section"};
  const iree_vm_bytecode_v0_strings_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_strings_header_t, &header);
  iree_host_size_t offset_count = 0;
  if (!iree_host_size_checked_add(header->string_count_u32, 1, &offset_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strings offset count overflows host size");
  }
  const iree_vm_bytecode_v0_string_offset_t* offsets = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, offset_count, iree_vm_bytecode_v0_string_offset_t, &offsets);
  const iree_host_size_t data_length = span.data_length - cursor.offset;
  if (data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strings byte tail exceeds uint32_t");
  }
  out_table->offsets = offsets;
  out_table->data = span.data + cursor.offset;
  out_table->count = header->string_count_u32;
  out_table->data_length = (uint32_t)data_length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_ref_types(
    iree_const_byte_span_t span, iree_vm_bytecode_ref_type_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "RefTypes section"};
  const iree_vm_bytecode_v0_ref_types_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_ref_types_header_t, &header);
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->group_count_u32,
                                     iree_vm_bytecode_v0_ref_type_group_row_t,
                                     &out_table->groups);
  IREE_VM_BYTECODE_CURSOR_TAKE_REMAINING_ARRAY(
      &cursor, iree_vm_bytecode_v0_ref_type_entry_row_t, &out_table->entries,
      &out_table->entry_count);
  out_table->group_count = header->group_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_signatures(
    iree_const_byte_span_t span,
    iree_vm_bytecode_signature_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Signatures section"};
  const iree_vm_bytecode_v0_signatures_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_signatures_header_t, &header);
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->signature_count_u32,
                                     iree_vm_bytecode_v0_signature_row_t,
                                     &out_table->rows);
  IREE_VM_BYTECODE_CURSOR_TAKE_REMAINING_ARRAY(
      &cursor, iree_vm_bytecode_v0_signature_descriptor_row_t,
      &out_table->descriptors, &out_table->descriptor_count);
  out_table->count = header->signature_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_callable_types(
    iree_const_byte_span_t span,
    const iree_vm_bytecode_signature_table_t* signatures,
    iree_vm_bytecode_callable_type_table_t* out_table,
    iree_vm_module_callable_field_counts_t* out_fields) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "CallableTypes section"};
  const iree_vm_bytecode_v0_callable_types_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_callable_types_header_t, &header);
  const iree_vm_bytecode_v0_callable_type_row_t* rows = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->callable_type_count_u32,
                                     iree_vm_bytecode_v0_callable_type_row_t,
                                     &rows);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  iree_vm_module_callable_field_counts_t fields = {0};
  for (uint32_t i = 0; i < header->callable_type_count_u32; ++i) {
    if (rows[i].signature_ordinal_u16 >= signatures->count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callable type signature is out of range");
    }
    const iree_vm_bytecode_v0_signature_row_t* signature =
        &signatures->rows[rows[i].signature_ordinal_u16];
    const iree_host_size_t value_count =
        (iree_host_size_t)signature->argument_value_count_u16 +
        signature->result_value_count_u16;
    const iree_host_size_t ref_count =
        (iree_host_size_t)signature->argument_ref_count_u16 +
        signature->result_ref_count_u16;
    const iree_host_size_t function_count =
        (iree_host_size_t)signature->argument_function_count_u16 +
        signature->result_function_count_u16;
    if (!iree_host_size_checked_add(fields.value_count, value_count,
                                    &fields.value_count) ||
        !iree_host_size_checked_add(fields.ref_count, ref_count,
                                    &fields.ref_count) ||
        !iree_host_size_checked_add(fields.function_count, function_count,
                                    &fields.function_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "aggregate callable signature fields overflow host size");
    }
  }
  out_table->rows = rows;
  out_table->count = header->callable_type_count_u32;
  *out_fields = fields;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_imports(
    iree_const_byte_span_t span, iree_vm_bytecode_import_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Imports section"};
  const iree_vm_bytecode_v0_imports_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_imports_header_t, &header);
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->group_count_u32,
                                     iree_vm_bytecode_v0_import_group_row_t,
                                     &out_table->groups);
  IREE_VM_BYTECODE_CURSOR_TAKE_REMAINING_ARRAY(
      &cursor, iree_vm_bytecode_v0_import_entry_row_t, &out_table->entries,
      &out_table->entry_count);
  out_table->group_count = header->group_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_exports(
    iree_const_byte_span_t span, iree_vm_bytecode_export_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Exports section"};
  const iree_vm_bytecode_v0_exports_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_exports_header_t, &header);
  const iree_vm_bytecode_v0_export_row_t* rows = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->export_count_u32,
                                     iree_vm_bytecode_v0_export_row_t, &rows);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  out_table->rows = rows;
  out_table->count = header->export_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_functions(
    iree_const_byte_span_t span, iree_vm_bytecode_function_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Functions section"};
  const iree_vm_bytecode_v0_functions_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_functions_header_t, &header);
  const iree_vm_bytecode_v0_function_row_t* rows = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->function_count_u32,
                                     iree_vm_bytecode_v0_function_row_t, &rows);

  uint64_t switch_target_count = 0;
  uint64_t bytecode_length = 0;
  if (header->function_count_u32 != 0) {
    const iree_vm_bytecode_v0_function_row_t* final_row =
        &rows[header->function_count_u32 - 1];
    switch_target_count = (uint64_t)final_row->switch_target_base_u32 +
                          final_row->switch_target_entry_count_u32;
    bytecode_length = (uint64_t)final_row->bytecode_offset_u32 +
                      final_row->bytecode_length_u32;
  }
  if (switch_target_count > UINT32_MAX || bytecode_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Functions tail extents overflow uint32_t");
  }
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, (iree_host_size_t)switch_target_count,
      iree_vm_bytecode_v0_switch_target_entry_t, &switch_targets);
  iree_const_byte_span_t bytecode_data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, (iree_host_size_t)bytecode_length, 4, &bytecode_data));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  out_table->rows = rows;
  out_table->switch_targets = switch_targets;
  out_table->bytecode_data = bytecode_data.data;
  out_table->count = header->function_count_u32;
  out_table->switch_target_count = (uint32_t)switch_target_count;
  out_table->bytecode_length = (uint32_t)bytecode_length;
  out_table->maximum_block_count = header->maximum_block_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_constants(
    iree_const_byte_span_t span, iree_vm_bytecode_constant_table_t* out_table) {
  if (span.data_length % sizeof(iree_vm_bytecode_v0_constant_cell_t) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Constants cells do not consume the section");
  }
  const iree_host_size_t count =
      span.data_length / sizeof(iree_vm_bytecode_v0_constant_cell_t);
  if (count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Constants cell count exceeds uint32_t");
  }
  out_table->cells = (const iree_vm_bytecode_v0_constant_cell_t*)span.data;
  out_table->count = (uint32_t)count;
  return iree_ok_status();
}

static iree_host_size_t iree_vm_bytecode_bit_word_count(uint32_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static iree_status_t iree_vm_bytecode_plan_process_layout(
    const iree_vm_bytecode_v0_globals_header_t* header,
    iree_vm_bytecode_process_layout_t* out_layout) {
  iree_vm_bytecode_process_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_vm_bytecode_process_header_t), &layout.total_size,
      IREE_STRUCT_FIELD(header->value_count_u32, uint64_t,
                        &layout.values_offset),
      IREE_STRUCT_FIELD(header->ref_count_u32, iree_vm_ref_t,
                        &layout.refs_offset),
      IREE_STRUCT_FIELD(header->function_count_u32, iree_vm_function_ref_t,
                        &layout.functions_offset),
      IREE_STRUCT_FIELD(
          iree_vm_bytecode_bit_word_count(header->immutable_value_count_u32),
          uint64_t, &layout.value_set_bits_offset),
      IREE_STRUCT_FIELD(
          iree_vm_bytecode_bit_word_count(header->immutable_ref_count_u32),
          uint64_t, &layout.ref_set_bits_offset),
      IREE_STRUCT_FIELD(
          iree_vm_bytecode_bit_word_count(header->immutable_function_count_u32),
          uint64_t, &layout.function_set_bits_offset)));
  *out_layout = layout;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_globals(
    iree_const_byte_span_t span, iree_vm_bytecode_global_table_t* out_table,
    iree_vm_bytecode_process_layout_t* out_process_layout) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Globals section"};
  const iree_vm_bytecode_v0_globals_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_globals_header_t, &header);
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* refs = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, header->ref_count_u32,
      iree_vm_bytecode_v0_global_ref_descriptor_row_t, &refs);
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* functions = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, header->function_count_u32,
      iree_vm_bytecode_v0_global_function_descriptor_row_t, &functions);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  out_table->header = header;
  out_table->refs = refs;
  out_table->functions = functions;
  return iree_vm_bytecode_plan_process_layout(header, out_process_layout);
}

static iree_status_t iree_vm_bytecode_map_rodata(
    iree_const_byte_span_t span, uint32_t payload_alignment,
    iree_vm_bytecode_rodata_table_t* out_table,
    iree_vm_bytecode_rodata_storage_plan_t* out_storage_plan) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Rodata section"};
  const iree_vm_bytecode_v0_rodata_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_rodata_header_t, &header);
  const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptors = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, header->block_count_u32,
      iree_vm_bytecode_v0_rodata_block_descriptor_t, &descriptors);

  const iree_host_size_t blocks_offset = cursor.offset;
  uint32_t maximum_alignment = IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT;
  iree_vm_bytecode_rodata_storage_plan_t storage_plan = {0};
  for (uint32_t i = 0; i < header->block_count_u32; ++i) {
    const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptor =
        &descriptors[i];
    const uint32_t alignment = descriptor->minimum_alignment_u32;
    if (!iree_host_size_is_power_of_two(alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rodata block alignment is not a power of two");
    }
    maximum_alignment = iree_max(maximum_alignment, alignment);
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_align(&cursor, alignment));
    if (descriptor->byte_length_u64 > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rodata block is not host-addressable");
    }
    iree_const_byte_span_t block;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
        &cursor, (iree_host_size_t)descriptor->byte_length_u64, alignment,
        &block));
    if (!iree_host_ptr_has_alignment(block.data, alignment)) {
      if (!iree_host_size_checked_align(storage_plan.copy_length, alignment,
                                        &storage_plan.copy_length) ||
          !iree_host_size_checked_add(
              storage_plan.copy_length,
              (iree_host_size_t)descriptor->byte_length_u64,
              &storage_plan.copy_length)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rodata fallback storage overflows host size");
      }
      storage_plan.copy_alignment =
          iree_max(storage_plan.copy_alignment, (iree_host_size_t)alignment);
    }
  }
  if (payload_alignment != maximum_alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Rodata section alignment does not match its blocks");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  out_table->descriptors = descriptors;
  out_table->section_begin = span.data;
  out_table->count = header->block_count_u32;
  out_table->blocks_offset = blocks_offset;
  *out_storage_plan = storage_plan;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_presentation(
    iree_const_byte_span_t span,
    iree_vm_bytecode_presentation_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Presentation section"};
  const iree_vm_bytecode_v0_presentation_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_presentation_header_t, &header);
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, header->entry_count_u32,
      iree_vm_bytecode_v0_presentation_entry_row_t, &out_table->entries);
  IREE_VM_BYTECODE_CURSOR_TAKE_REMAINING_ARRAY(
      &cursor, iree_vm_bytecode_v0_presentation_field_row_t, &out_table->fields,
      &out_table->field_count);
  out_table->entry_count = header->entry_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_map_metadata(
    iree_const_byte_span_t span, iree_vm_bytecode_metadata_table_t* out_table) {
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Metadata section"};
  const iree_vm_bytecode_v0_metadata_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_metadata_header_t, &header);
  const iree_vm_bytecode_v0_metadata_scope_row_t* import_scopes = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->import_scope_count_u32,
                                     iree_vm_bytecode_v0_metadata_scope_row_t,
                                     &import_scopes);
  const iree_vm_bytecode_v0_metadata_scope_row_t* export_scopes = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->export_scope_count_u32,
                                     iree_vm_bytecode_v0_metadata_scope_row_t,
                                     &export_scopes);
  const iree_vm_bytecode_v0_metadata_entry_row_t* entries = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(&cursor, header->total_entry_count_u32,
                                     iree_vm_bytecode_v0_metadata_entry_row_t,
                                     &entries);
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_cursor_align(&cursor, iree_alignof(uint64_t)));
  iree_host_size_t offset_count = 0;
  if (!iree_host_size_checked_add(header->total_entry_count_u32, 1,
                                  &offset_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata offset count overflows host size");
  }
  const iree_vm_bytecode_v0_metadata_value_offset_t* value_offsets = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, offset_count, iree_vm_bytecode_v0_metadata_value_offset_t,
      &value_offsets);
  out_table->header = header;
  out_table->import_scopes = import_scopes;
  out_table->export_scopes = export_scopes;
  out_table->entries = entries;
  out_table->value_offsets = value_offsets;
  out_table->value_data = span.data + cursor.offset;
  out_table->value_data_length = span.data_length - cursor.offset;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_module_plan_build(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan) {
  if (!contents.data || contents.data_length == 0 ||
      !iree_host_ptr_has_alignment(contents.data,
                                   IREE_VM_BYTECODE_IMAGE_ALIGNMENT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bytecode image must be nonempty and eight-byte aligned");
  }

  memset(out_plan, 0, sizeof(*out_plan));
  iree_vm_bytecode_cursor_t cursor = {contents, 0, "bytecode image"};
  const iree_vm_bytecode_v0_image_header_t* header = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, 1, iree_vm_bytecode_v0_image_header_t, &header);
  if (memcmp(header->magic_u8, IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES,
             IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode image magic is invalid");
  }
  if (header->core_major_u16 != IREE_VM_BYTECODE_CORE_MAJOR) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "bytecode Core major %u is unsupported",
                            (unsigned)header->core_major_u16);
  }
  if (header->core_required_minor_u16 > IREE_VM_BYTECODE_CORE_MINOR) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "bytecode Core minor %u is newer than runtime minor %d",
        (unsigned)header->core_required_minor_u16, IREE_VM_BYTECODE_CORE_MINOR);
  }
  if (header->zero_padding_u16 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode image header padding is nonzero");
  }

  const iree_vm_bytecode_v0_section_directory_row_t* rows = NULL;
  IREE_VM_BYTECODE_CURSOR_TAKE_ARRAY(
      &cursor, header->section_count_u16,
      iree_vm_bytecode_v0_section_directory_row_t, &rows);
  out_plan->layout.image.header = header;
  out_plan->layout.image.sections = rows;
  out_plan->layout.image.section_count = header->section_count_u16;

  uint16_t previous_type = 0;
  for (uint16_t i = 0; i < header->section_count_u16; ++i) {
    const iree_vm_bytecode_v0_section_directory_row_t* row = &rows[i];
    if (row->section_type_u16 == 0 ||
        (i != 0 && row->section_type_u16 <= previous_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "section types must be nonzero and strictly increasing");
    }
    if (row->payload_alignment_u32 < IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT ||
        !iree_host_size_is_power_of_two(row->payload_alignment_u32)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "section 0x%04" PRIx16
                              " has invalid payload alignment",
                              row->section_type_u16);
    }

    const bool is_known =
        row->section_type_u16 <= IREE_VM_BYTECODE_SECTION_METADATA;
    if (is_known) {
      const uint32_t descriptor =
          iree_vm_bytecode_section_layouts[row->section_type_u16];
      if (iree_vm_bytecode_section_since_minor(descriptor) >
              header->core_required_minor_u16 ||
          row->section_flags_u16 !=
              iree_vm_bytecode_section_required_flags(descriptor) ||
          row->byte_length_u64 == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "known section 0x%04" PRIx16
                                " has invalid version, flags, or length",
                                row->section_type_u16);
      }
    } else if (row->section_flags_u16 !=
               IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "unknown section 0x%04" PRIx16
                              " is not skippable",
                              row->section_type_u16);
    }

    const uint8_t authority = (uint8_t)(row->section_type_u16 >> 8);
    if (authority != 0 && (authority < 0xF0 || authority > 0xFD)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "section 0x%04" PRIx16
                              " has an invalid authority",
                              row->section_type_u16);
    }
    if (authority != 0) {
      out_plan->layout.image.extension_pages |=
          (uint16_t)(1u << (authority - 0xF0));
    }

    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_cursor_align(&cursor, row->payload_alignment_u32));
    if (row->byte_length_u64 > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "section 0x%04" PRIx16 " is not host-addressable",
                              row->section_type_u16);
    }
    iree_const_byte_span_t payload;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
        &cursor, (iree_host_size_t)row->byte_length_u64,
        row->payload_alignment_u32, &payload));
#define IREE_VM_BYTECODE_MAP_SECTION(section, call) \
  case section: {                                   \
    IREE_RETURN_IF_ERROR(call);                     \
    break;                                          \
  }
    switch (row->section_type_u16) {
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_REQUIREMENTS,
          iree_vm_bytecode_map_requirements(payload,
                                            &out_plan->layout.requirements));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_STRINGS,
          iree_vm_bytecode_map_strings(payload, &out_plan->layout.strings));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_REF_TYPES,
          iree_vm_bytecode_map_ref_types(payload, &out_plan->layout.ref_types));
      IREE_VM_BYTECODE_MAP_SECTION(IREE_VM_BYTECODE_SECTION_SIGNATURES,
                                   iree_vm_bytecode_map_signatures(
                                       payload, &out_plan->layout.signatures));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES,
          iree_vm_bytecode_map_callable_types(
              payload, &out_plan->layout.signatures,
              &out_plan->layout.callable_types, &out_plan->callable_fields));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_IMPORTS,
          iree_vm_bytecode_map_imports(payload, &out_plan->layout.imports));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_EXPORTS,
          iree_vm_bytecode_map_exports(payload, &out_plan->layout.exports));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_FUNCTIONS,
          iree_vm_bytecode_map_functions(payload, &out_plan->layout.functions));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_CONSTANTS,
          iree_vm_bytecode_map_constants(payload, &out_plan->layout.constants));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_GLOBALS,
          iree_vm_bytecode_map_globals(payload, &out_plan->layout.globals,
                                       &out_plan->process_layout));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_RODATA,
          iree_vm_bytecode_map_rodata(payload, row->payload_alignment_u32,
                                      &out_plan->layout.rodata,
                                      &out_plan->rodata_storage));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_PRESENTATION,
          iree_vm_bytecode_map_presentation(payload,
                                            &out_plan->layout.presentation));
      IREE_VM_BYTECODE_MAP_SECTION(
          IREE_VM_BYTECODE_SECTION_METADATA,
          iree_vm_bytecode_map_metadata(payload, &out_plan->layout.metadata));
      default:
        break;
    }
#undef IREE_VM_BYTECODE_MAP_SECTION
    previous_type = row->section_type_u16;
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  return iree_ok_status();
}
