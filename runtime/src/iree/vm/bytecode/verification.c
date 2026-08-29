// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/wire/hal/opcodes.h"

#if !defined(IREE_ENDIANNESS_LITTLE)
#error "IREE VM bytecode version zero requires a little-endian host"
#endif

static_assert(sizeof(iree_vm_bytecode_v0_signature_descriptor_row_t) ==
                  sizeof(iree_vm_module_signature_type_t),
              "wire and provider signature leaves must have equal size");
static_assert(iree_alignof(iree_vm_bytecode_v0_signature_descriptor_row_t) ==
                  iree_alignof(iree_vm_module_signature_type_t),
              "wire and provider signature leaves must have equal alignment");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_INVALID,
                           IREE_VM_MODULE_SIGNATURE_TYPE_KIND_INVALID,
                           "wire and provider invalid kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_I8,
                           IREE_VM_SCALAR_TYPE_I8,
                           "wire and provider i8 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_I16,
                           IREE_VM_SCALAR_TYPE_I16,
                           "wire and provider i16 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_I32,
                           IREE_VM_SCALAR_TYPE_I32,
                           "wire and provider i32 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_I64,
                           IREE_VM_SCALAR_TYPE_I64,
                           "wire and provider i64 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN,
                           IREE_VM_SCALAR_TYPE_F8E4M3FN,
                           "wire and provider f8e4m3fn kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2,
                           IREE_VM_SCALAR_TYPE_F8E5M2,
                           "wire and provider f8e5m2 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_F16,
                           IREE_VM_SCALAR_TYPE_F16,
                           "wire and provider f16 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_BF16,
                           IREE_VM_SCALAR_TYPE_BF16,
                           "wire and provider bf16 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_F32,
                           IREE_VM_SCALAR_TYPE_F32,
                           "wire and provider f32 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_F64,
                           IREE_VM_SCALAR_TYPE_F64,
                           "wire and provider f64 kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_REF,
                           IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF,
                           "wire and provider ref kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION,
                           IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION,
                           "wire and provider function kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL,
                           IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL,
                           "wire and provider import flags must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD,
                           IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
                           "wire and provider callable flags must agree");
IREE_STATIC_ASSERT_ENUM_EQ(
    IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT,
    IREE_VM_MODULE_DECLARATION_KIND_IMPORT,
    "wire and provider import presentation kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(
    IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_EXPORT,
    IREE_VM_MODULE_DECLARATION_KIND_EXPORT,
    "wire and provider export presentation kinds must agree");
IREE_STATIC_ASSERT_ENUM_EQ(
    IREE_VM_BYTECODE_METADATA_VALUE_TYPE_INVALID,
    IREE_VM_METADATA_VALUE_TYPE_INVALID,
    "wire and provider invalid metadata types must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL,
                           IREE_VM_METADATA_VALUE_TYPE_BOOL,
                           "wire and provider bool metadata types must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64,
                           IREE_VM_METADATA_VALUE_TYPE_I64,
                           "wire and provider i64 metadata types must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64,
                           IREE_VM_METADATA_VALUE_TYPE_U64,
                           "wire and provider u64 metadata types must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64,
                           IREE_VM_METADATA_VALUE_TYPE_F64,
                           "wire and provider f64 metadata types must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8,
                           IREE_VM_METADATA_VALUE_TYPE_UTF8,
                           "wire and provider UTF-8 metadata types must agree");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES,
                           IREE_VM_METADATA_VALUE_TYPE_BYTES,
                           "wire and provider bytes metadata types must agree");

enum iree_vm_bytecode_string_ordinal_flag_bits_e {
  IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONE = 0u,
  IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NULLABLE = 1u << 0,
  IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY = 1u << 1,
};
typedef uint32_t iree_vm_bytecode_string_ordinal_flags_t;

typedef struct iree_vm_bytecode_cursor_t {
  // Exact bytes being consumed.
  iree_const_byte_span_t span;
  // Next unconsumed byte offset.
  iree_host_size_t offset;
  // Diagnostic name of the containing structure.
  const char* name;
} iree_vm_bytecode_cursor_t;

typedef struct iree_vm_bytecode_section_map_t {
  // Fixed image header at byte zero.
  const iree_vm_bytecode_v0_image_header_t* header;
  // Number of section directory rows.
  uint16_t section_count;
  // Strictly type-sorted section directory rows.
  const iree_vm_bytecode_v0_section_directory_row_t* rows;
  // Known section payloads indexed by their architectural type ID.
  iree_const_byte_span_t spans[IREE_VM_BYTECODE_SECTION_METADATA + 1];
  // Known section payload alignments indexed by architectural type ID.
  uint32_t payload_alignments[IREE_VM_BYTECODE_SECTION_METADATA + 1];
  // Architectural extension pages owning at least one section.
  uint16_t extension_section_pages;
} iree_vm_bytecode_section_map_t;

static bool iree_vm_bytecode_bytes_are_zero(const uint8_t* data,
                                            iree_host_size_t length) {
  for (iree_host_size_t i = 0; i < length; ++i) {
    if (data[i] != 0) return false;
  }
  return true;
}

static iree_status_t iree_vm_bytecode_cursor_take(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t length,
    iree_host_size_t alignment, const void** out_data) {
  if (!iree_host_size_has_alignment(cursor->offset, alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s offset %" PRIhsz " is not aligned to %" PRIhsz
                            " bytes",
                            cursor->name, cursor->offset, alignment);
  }
  if (cursor->offset > cursor->span.data_length ||
      length > cursor->span.data_length - cursor->offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s is truncated at byte %" PRIhsz, cursor->name,
                            cursor->offset);
  }
  *out_data = cursor->span.data + cursor->offset;
  cursor->offset += length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_cursor_take_array(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t count,
    iree_host_size_t element_size, iree_host_size_t alignment,
    const void** out_data) {
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_mul(count, element_size, &length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s array byte length overflows", cursor->name);
  }
  return iree_vm_bytecode_cursor_take(cursor, length, alignment, out_data);
}

static iree_status_t iree_vm_bytecode_cursor_align(
    iree_vm_bytecode_cursor_t* cursor, iree_host_size_t alignment) {
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

static bool iree_vm_bytecode_known_section_flags(uint16_t section_type,
                                                 uint16_t* out_required_flags) {
  switch (section_type) {
    case IREE_VM_BYTECODE_SECTION_REQUIREMENTS:
    case IREE_VM_BYTECODE_SECTION_STRINGS:
    case IREE_VM_BYTECODE_SECTION_REF_TYPES:
    case IREE_VM_BYTECODE_SECTION_SIGNATURES:
    case IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES:
    case IREE_VM_BYTECODE_SECTION_IMPORTS:
    case IREE_VM_BYTECODE_SECTION_EXPORTS:
    case IREE_VM_BYTECODE_SECTION_FUNCTIONS:
    case IREE_VM_BYTECODE_SECTION_CONSTANTS:
    case IREE_VM_BYTECODE_SECTION_GLOBALS:
    case IREE_VM_BYTECODE_SECTION_RODATA:
      *out_required_flags = 0;
      return true;
    case IREE_VM_BYTECODE_SECTION_PRESENTATION:
    case IREE_VM_BYTECODE_SECTION_METADATA:
      *out_required_flags = IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE;
      return true;
    default:
      return false;
  }
}

static iree_status_t iree_vm_bytecode_verify_envelope(
    iree_const_byte_span_t contents,
    iree_vm_bytecode_section_map_t* out_sections) {
  iree_vm_bytecode_cursor_t cursor = {contents, 0, "bytecode image"};
  const iree_vm_bytecode_v0_image_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (memcmp(header->magic_u8, IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES,
             IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode image magic is invalid");
  }
  if (header->core_major_u16 != IREE_VM_BYTECODE_CORE_MAJOR ||
      header->core_required_minor_u16 > IREE_VM_BYTECODE_CORE_MINOR) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "bytecode Core version %" PRIu16 ".%" PRIu16
        " is not supported by runtime version %d.%d",
        header->core_major_u16, header->core_required_minor_u16,
        IREE_VM_BYTECODE_CORE_MAJOR, IREE_VM_BYTECODE_CORE_MINOR);
  }
  if (header->zero_padding_u16 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode image header padding is nonzero");
  }

  const iree_vm_bytecode_v0_section_directory_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->section_count_u16, sizeof(*rows), iree_alignof(*rows),
      (const void**)&rows));

  memset(out_sections, 0, sizeof(*out_sections));
  out_sections->header = header;
  out_sections->section_count = header->section_count_u16;
  out_sections->rows = rows;
  uint16_t previous_type = 0;
  for (uint16_t i = 0; i < header->section_count_u16; ++i) {
    const iree_vm_bytecode_v0_section_directory_row_t* row = &rows[i];
    if (row->section_type_u16 == 0 ||
        (i != 0 && row->section_type_u16 <= previous_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "section directory types must be strictly increasing and nonzero");
    }
    if (row->payload_alignment_u32 < IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT ||
        !iree_host_size_is_power_of_two(row->payload_alignment_u32)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "section 0x%04" PRIx16 " has invalid payload alignment %" PRIu32,
          row->section_type_u16, row->payload_alignment_u32);
    }
    uint16_t required_flags = 0;
    const bool is_known = iree_vm_bytecode_known_section_flags(
        row->section_type_u16, &required_flags);
    if ((is_known && row->section_flags_u16 != required_flags) ||
        (!is_known &&
         row->section_flags_u16 != IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE)) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "section 0x%04" PRIx16
                              " has unsupported flags 0x%04" PRIx16,
                              row->section_type_u16, row->section_flags_u16);
    }
    if (is_known && row->byte_length_u64 == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "known section 0x%04" PRIx16 " is empty",
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
      out_sections->extension_section_pages |=
          (uint16_t)(1u << (authority - 0xF0));
    }

    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_cursor_align(&cursor, row->payload_alignment_u32));
    if (row->byte_length_u64 > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "section 0x%04" PRIx16 " is not host-addressable",
                              row->section_type_u16);
    }
    const uint8_t* payload = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
        &cursor, (iree_host_size_t)row->byte_length_u64,
        row->payload_alignment_u32, (const void**)&payload));
    if (is_known) {
      out_sections->spans[row->section_type_u16] = iree_make_const_byte_span(
          payload, (iree_host_size_t)row->byte_length_u64);
      out_sections->payload_alignments[row->section_type_u16] =
          row->payload_alignment_u32;
    }
    previous_type = row->section_type_u16;
  }
  return iree_vm_bytecode_cursor_finish(&cursor);
}

static iree_status_t iree_vm_bytecode_verify_requirements(
    const iree_vm_bytecode_section_map_t* sections,
    iree_vm_bytecode_module_layout_t* layout) {
  const iree_const_byte_span_t span =
      sections->spans[IREE_VM_BYTECODE_SECTION_REQUIREMENTS];
  if (iree_const_byte_span_is_empty(span)) {
    if (sections->extension_section_pages != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "extension-owned sections require a Requirements declaration");
    }
    return iree_ok_status();
  }
  if (span.data_length % sizeof(iree_vm_bytecode_v0_requirement_row_t) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Requirements rows do not consume the section");
  }
  const iree_host_size_t count =
      span.data_length / sizeof(iree_vm_bytecode_v0_requirement_row_t);
  const iree_vm_bytecode_v0_requirement_row_t* rows =
      (const iree_vm_bytecode_v0_requirement_row_t*)span.data;
  uint16_t previous_page = 0;
  uint16_t declared_pages = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (rows[i].page_id_u16 < 0xF0 || rows[i].page_id_u16 > 0xFD ||
        (i != 0 && rows[i].page_id_u16 <= previous_page)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Requirements page IDs must be unique sorted extension pages");
    }
    declared_pages |= (uint16_t)(1u << (rows[i].page_id_u16 - 0xF0));
    previous_page = rows[i].page_id_u16;
  }
  if ((sections->extension_section_pages & ~declared_pages) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "extension-owned section has no exact Requirements declaration");
  }
  layout->requirements.count = (uint16_t)count;
  layout->requirements.rows = rows;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_string_ordinal(
    const iree_vm_bytecode_string_table_t* table, uint16_t ordinal,
    iree_vm_bytecode_string_ordinal_flags_t flags, const char* label) {
  if (iree_any_bit_set(flags, IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NULLABLE) &&
      ordinal == UINT16_MAX) {
    return iree_ok_status();
  }
  if (ordinal >= table->count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s string ordinal is out of range", label);
  }
  if (iree_any_bit_set(flags, IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY) &&
      iree_vm_bytecode_string_at(table, ordinal).size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s string must be nonempty", label);
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_strings(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Strings section"};
  const iree_vm_bytecode_v0_strings_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->string_count_u32 == 0 || header->string_count_u32 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strings count must be in [1, 65535]");
  }
  const iree_vm_bytecode_v0_string_offset_t* offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, (iree_host_size_t)header->string_count_u32 + 1, sizeof(*offsets),
      iree_alignof(*offsets), (const void**)&offsets));
  const iree_host_size_t data_length = span.data_length - cursor.offset;
  if (data_length > UINT32_MAX || offsets[0] != 0 ||
      offsets[header->string_count_u32] != data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strings offsets do not cover the exact byte tail");
  }
  const uint8_t* data = span.data + cursor.offset;
  for (uint32_t i = 0; i < header->string_count_u32; ++i) {
    if (offsets[i] > offsets[i + 1] || offsets[i + 1] > data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Strings offsets are not monotonic");
    }
    const iree_string_view_t value = iree_make_string_view(
        (const char*)data + offsets[i], offsets[i + 1] - offsets[i]);
    if (iree_string_view_find_char(value, '\0', 0) != IREE_STRING_VIEW_NPOS ||
        !iree_unicode_utf8_validate(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "string %" PRIu32 " is not NUL-free valid UTF-8",
                              i);
    }
  }
  layout->strings.count = header->string_count_u32;
  layout->strings.offsets = offsets;
  layout->strings.data = data;
  layout->strings.data_length = (uint32_t)data_length;
  cursor.offset = span.data_length;
  return iree_vm_bytecode_cursor_finish(&cursor);
}

static iree_status_t iree_vm_bytecode_verify_ref_types(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "RefTypes section"};
  const iree_vm_bytecode_v0_ref_types_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->group_count_u32 == 0 || header->group_count_u32 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RefTypes group count must be in [1, 65535]");
  }
  const iree_vm_bytecode_v0_ref_type_group_row_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->group_count_u32, sizeof(*groups), iree_alignof(*groups),
      (const void**)&groups));
  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < header->group_count_u32; ++i) {
    if (groups[i].zero_padding_u16 != 0 || groups[i].entry_count_u32 == 0 ||
        groups[i].entry_count_u32 > 65536u - entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RefTypes group counts or padding are invalid");
    }
    entry_count += groups[i].entry_count_u32;
  }
  const iree_vm_bytecode_v0_ref_type_entry_row_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, entry_count, sizeof(*entries), iree_alignof(*entries),
      (const void**)&entries));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  uint32_t entry_base = 0;
  iree_string_view_t previous_namespace = iree_string_view_empty();
  for (uint32_t group_i = 0; group_i < header->group_count_u32; ++group_i) {
    const iree_vm_bytecode_v0_ref_type_group_row_t* group = &groups[group_i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
        &layout->strings, group->namespace_string_u16,
        IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY, "ref-type namespace"));
    const iree_string_view_t namespace_name = iree_vm_bytecode_string_at(
        &layout->strings, group->namespace_string_u16);
    if (group_i != 0 &&
        iree_string_view_compare(previous_namespace, namespace_name) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "RefTypes namespaces must be strictly byte-sorted");
    }
    iree_string_view_t previous_name = iree_string_view_empty();
    for (uint32_t entry_i = 0; entry_i < group->entry_count_u32; ++entry_i) {
      const iree_vm_bytecode_v0_ref_type_entry_row_t* entry =
          &entries[entry_base + entry_i];
      if (entry->required_flags_u16 != 0) {
        return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                                "ref type has unsupported required flags");
      }
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
          &layout->strings, entry->type_name_string_u16,
          IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY, "ref-type name"));
      const iree_string_view_t name = iree_vm_bytecode_string_at(
          &layout->strings, entry->type_name_string_u16);
      if (entry_i != 0 && iree_string_view_compare(previous_name, name) >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "RefTypes names must be strictly byte-sorted within a namespace");
      }
      previous_name = name;
    }
    entry_base += group->entry_count_u32;
    previous_namespace = namespace_name;
  }
  layout->ref_types.group_count = header->group_count_u32;
  layout->ref_types.groups = groups;
  layout->ref_types.entry_count = entry_count;
  layout->ref_types.entries = entries;
  return iree_ok_status();
}

static bool iree_vm_bytecode_signature_kind_is_scalar(uint16_t kind) {
  return kind >= IREE_VM_BYTECODE_SIGNATURE_KIND_I8 &&
         kind <= IREE_VM_BYTECODE_SIGNATURE_KIND_F64;
}

static iree_status_t iree_vm_bytecode_verify_signature_descriptors(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors,
    uint32_t count, uint16_t expected_value_count, uint16_t expected_ref_count,
    uint16_t expected_function_count) {
  uint32_t value_count = 0;
  uint32_t ref_count = 0;
  uint32_t function_count = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
        &descriptors[i];
    if (iree_vm_bytecode_signature_kind_is_scalar(descriptor->kind_u16)) {
      if (descriptor->type_ordinal_u16 != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "scalar signature descriptor has a type ordinal");
      }
      ++value_count;
    } else if (descriptor->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
      if (descriptor->type_ordinal_u16 >= layout->ref_types.entry_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "signature ref-type ordinal is out of range");
      }
      ++ref_count;
    } else if (descriptor->kind_u16 ==
               IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
      ++function_count;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signature descriptor kind is invalid");
    }
  }
  if (value_count != expected_value_count || ref_count != expected_ref_count ||
      function_count != expected_function_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "signature bank counts do not match source-ordered descriptors");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_signatures(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Signatures section"};
  const iree_vm_bytecode_v0_signatures_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->signature_count_u32 == 0 ||
      header->signature_count_u32 > 65536u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Signatures count must be in [1, 65536]");
  }
  const iree_vm_bytecode_v0_signature_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->signature_count_u32, sizeof(*rows), iree_alignof(*rows),
      (const void**)&rows));
  uint32_t descriptor_count = 0;
  for (uint32_t i = 0; i < header->signature_count_u32; ++i) {
    const iree_vm_bytecode_v0_signature_row_t* row = &rows[i];
    const uint32_t argument_count =
        iree_vm_bytecode_signature_argument_count(row);
    const uint32_t result_count = iree_vm_bytecode_signature_result_count(row);
    if (argument_count > UINT16_MAX || result_count > UINT16_MAX ||
        row->descriptor_base_u32 != descriptor_count ||
        argument_count > UINT32_MAX - descriptor_count ||
        result_count > UINT32_MAX - descriptor_count - argument_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signature descriptor prefix is invalid");
    }
    descriptor_count += argument_count + result_count;
  }
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, descriptor_count, sizeof(*descriptors),
      iree_alignof(*descriptors), (const void**)&descriptors));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  layout->signatures.count = header->signature_count_u32;
  layout->signatures.rows = rows;
  layout->signatures.descriptor_count = descriptor_count;
  layout->signatures.descriptors = descriptors;
  for (uint32_t i = 0; i < header->signature_count_u32; ++i) {
    const iree_vm_bytecode_v0_signature_row_t* row = &rows[i];
    const iree_vm_bytecode_v0_signature_descriptor_row_t* signature =
        descriptors + row->descriptor_base_u32;
    const uint32_t argument_count =
        iree_vm_bytecode_signature_argument_count(row);
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_signature_descriptors(
        layout, signature, argument_count, row->argument_value_count_u16,
        row->argument_ref_count_u16, row->argument_function_count_u16));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_signature_descriptors(
        layout, signature + argument_count,
        iree_vm_bytecode_signature_result_count(row),
        row->result_value_count_u16, row->result_ref_count_u16,
        row->result_function_count_u16));
  }
  return iree_ok_status();
}

static int iree_vm_bytecode_compare_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static int iree_vm_bytecode_compare_signature_descriptors(
    const iree_vm_bytecode_v0_signature_descriptor_row_t* lhs,
    const iree_vm_bytecode_v0_signature_descriptor_row_t* rhs, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    int comparison =
        iree_vm_bytecode_compare_u32(lhs[i].kind_u16, rhs[i].kind_u16);
    if (comparison != 0) return comparison;
    comparison = iree_vm_bytecode_compare_u32(lhs[i].type_ordinal_u16,
                                              rhs[i].type_ordinal_u16);
    if (comparison != 0) return comparison;
  }
  return 0;
}

static int iree_vm_bytecode_compare_callable_types(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_callable_type_row_t* lhs,
    const iree_vm_bytecode_v0_callable_type_row_t* rhs) {
  int comparison = iree_vm_bytecode_compare_u32(lhs->nesting_depth_u16,
                                                rhs->nesting_depth_u16);
  if (comparison != 0) return comparison;

  const iree_vm_bytecode_v0_signature_row_t* lhs_signature =
      &layout->signatures.rows[lhs->signature_ordinal_u16];
  const iree_vm_bytecode_v0_signature_row_t* rhs_signature =
      &layout->signatures.rows[rhs->signature_ordinal_u16];
  const uint32_t lhs_argument_count =
      iree_vm_bytecode_signature_argument_count(lhs_signature);
  const uint32_t rhs_argument_count =
      iree_vm_bytecode_signature_argument_count(rhs_signature);
  comparison =
      iree_vm_bytecode_compare_u32(lhs_argument_count, rhs_argument_count);
  if (comparison != 0) return comparison;
  const iree_vm_bytecode_v0_signature_descriptor_row_t* lhs_descriptors =
      iree_vm_bytecode_signature_descriptors(&layout->signatures,
                                             lhs->signature_ordinal_u16);
  const iree_vm_bytecode_v0_signature_descriptor_row_t* rhs_descriptors =
      iree_vm_bytecode_signature_descriptors(&layout->signatures,
                                             rhs->signature_ordinal_u16);
  comparison = iree_vm_bytecode_compare_signature_descriptors(
      lhs_descriptors, rhs_descriptors, lhs_argument_count);
  if (comparison != 0) return comparison;
  const uint32_t lhs_result_count =
      iree_vm_bytecode_signature_result_count(lhs_signature);
  const uint32_t rhs_result_count =
      iree_vm_bytecode_signature_result_count(rhs_signature);
  comparison = iree_vm_bytecode_compare_u32(lhs_result_count, rhs_result_count);
  if (comparison != 0) return comparison;
  comparison = iree_vm_bytecode_compare_signature_descriptors(
      lhs_descriptors + lhs_argument_count,
      rhs_descriptors + rhs_argument_count, lhs_result_count);
  if (comparison != 0) return comparison;
  return iree_vm_bytecode_compare_u32(lhs->flags_u16, rhs->flags_u16);
}

static iree_status_t iree_vm_bytecode_verify_callable_types(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) {
    for (uint32_t i = 0; i < layout->signatures.descriptor_count; ++i) {
      if (layout->signatures.descriptors[i].kind_u16 ==
          IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function signature type requires CallableTypes");
      }
    }
    return iree_ok_status();
  }
  iree_vm_bytecode_cursor_t cursor = {span, 0, "CallableTypes section"};
  const iree_vm_bytecode_v0_callable_types_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->callable_type_count_u32 == 0 ||
      header->callable_type_count_u32 > 65536u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CallableTypes count must be in [1, 65536]");
  }
  const iree_vm_bytecode_v0_callable_type_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->callable_type_count_u32, sizeof(*rows),
      iree_alignof(*rows), (const void**)&rows));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  layout->callable_types.count = header->callable_type_count_u32;
  layout->callable_types.rows = rows;

  for (uint32_t i = 0; i < header->callable_type_count_u32; ++i) {
    const iree_vm_bytecode_v0_callable_type_row_t* row = &rows[i];
    if (row->signature_ordinal_u16 >= layout->signatures.count ||
        row->reserved_u16 != 0 ||
        (row->flags_u16 & ~IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD) !=
            0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callable type row is invalid");
    }
    const iree_vm_bytecode_v0_signature_row_t* signature_row =
        &layout->signatures.rows[row->signature_ordinal_u16];
    const uint32_t descriptor_count =
        iree_vm_bytecode_signature_argument_count(signature_row) +
        iree_vm_bytecode_signature_result_count(signature_row);
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
        iree_vm_bytecode_signature_descriptors(&layout->signatures,
                                               row->signature_ordinal_u16);
    uint32_t expected_depth = 0;
    for (uint32_t j = 0; j < descriptor_count; ++j) {
      if (descriptors[j].kind_u16 != IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
        continue;
      }
      const uint16_t child_ordinal = descriptors[j].type_ordinal_u16;
      if (child_ordinal >= i) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "callable function types must be topologically ordered");
      }
      if (rows[child_ordinal].nesting_depth_u16 == UINT16_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "callable type nesting depth overflows u16");
      }
      expected_depth = iree_max(
          expected_depth, (uint32_t)rows[child_ordinal].nesting_depth_u16 + 1);
    }
    if (row->nesting_depth_u16 != expected_depth) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callable type nesting depth is not canonical");
    }
    if (i != 0 && iree_vm_bytecode_compare_callable_types(layout, &rows[i - 1],
                                                          row) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "callable types must be unique and strictly ordered");
    }
  }
  for (uint32_t i = 0; i < layout->signatures.descriptor_count; ++i) {
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptor =
        &layout->signatures.descriptors[i];
    if (descriptor->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION &&
        descriptor->type_ordinal_u16 >= header->callable_type_count_u32) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "signature callable-type ordinal is out of range");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_imports(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Imports section"};
  const iree_vm_bytecode_v0_imports_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->group_count_u32 == 0 || header->group_count_u32 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Imports group count must be in [1, 65535]");
  }
  const iree_vm_bytecode_v0_import_group_row_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->group_count_u32, sizeof(*groups), iree_alignof(*groups),
      (const void**)&groups));
  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < header->group_count_u32; ++i) {
    if (groups[i].zero_padding_u16 != 0 || groups[i].entry_count_u32 == 0 ||
        groups[i].entry_count_u32 > 65536u - entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Imports group counts or padding are invalid");
    }
    entry_count += groups[i].entry_count_u32;
  }
  const iree_vm_bytecode_v0_import_entry_row_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, entry_count, sizeof(*entries), iree_alignof(*entries),
      (const void**)&entries));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  uint32_t entry_base = 0;
  iree_string_view_t previous_module = iree_string_view_empty();
  for (uint32_t group_i = 0; group_i < header->group_count_u32; ++group_i) {
    const iree_vm_bytecode_v0_import_group_row_t* group = &groups[group_i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
        &layout->strings, group->module_name_string_u16,
        IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY, "import module name"));
    const iree_string_view_t module_name = iree_vm_bytecode_string_at(
        &layout->strings, group->module_name_string_u16);
    if (group_i != 0 &&
        iree_string_view_compare(previous_module, module_name) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Imports groups must be strictly byte-sorted");
    }
    iree_string_view_t previous_symbol = iree_string_view_empty();
    uint16_t previous_callable = 0;
    for (uint32_t entry_i = 0; entry_i < group->entry_count_u32; ++entry_i) {
      const iree_vm_bytecode_v0_import_entry_row_t* entry =
          &entries[entry_base + entry_i];
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
          &layout->strings, entry->symbol_name_string_u16,
          IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY, "import symbol name"));
      if (entry->callable_type_ordinal_u16 >= layout->callable_types.count ||
          (entry->flags_u16 & ~IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL) != 0 ||
          entry->zero_padding_u16 != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "import entry fields are invalid");
      }
      const iree_string_view_t symbol = iree_vm_bytecode_string_at(
          &layout->strings, entry->symbol_name_string_u16);
      if (entry_i != 0) {
        const int comparison =
            iree_string_view_compare(previous_symbol, symbol);
        if (comparison > 0 ||
            (comparison == 0 &&
             entry->callable_type_ordinal_u16 <= previous_callable)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Imports entries must be strictly sorted by symbol and "
              "callable type");
        }
      }
      previous_symbol = symbol;
      previous_callable = entry->callable_type_ordinal_u16;
    }
    entry_base += group->entry_count_u32;
    previous_module = module_name;
  }
  layout->imports.group_count = header->group_count_u32;
  layout->imports.groups = groups;
  layout->imports.entry_count = entry_count;
  layout->imports.entries = entries;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_exports(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Exports section"};
  const iree_vm_bytecode_v0_exports_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->export_count_u32 == 0 || header->export_count_u32 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Exports count must be in [1, 65535]");
  }
  const iree_vm_bytecode_v0_export_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->export_count_u32, sizeof(*rows), iree_alignof(*rows),
      (const void**)&rows));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  iree_string_view_t previous_name = iree_string_view_empty();
  for (uint32_t i = 0; i < header->export_count_u32; ++i) {
    const iree_vm_bytecode_v0_export_row_t* row = &rows[i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
        &layout->strings, row->name_string_u16,
        IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY, "export name"));
    if (row->callable_type_ordinal_u16 >= layout->callable_types.count ||
        row->zero_padding_u16 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "export row fields are invalid");
    }
    const iree_string_view_t name =
        iree_vm_bytecode_string_at(&layout->strings, row->name_string_u16);
    if (i != 0 && iree_string_view_compare(previous_name, name) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Exports names must be strictly byte-sorted");
    }
    previous_name = name;
  }
  layout->exports.count = header->export_count_u32;
  layout->exports.rows = rows;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_function_signature_structure(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* row) {
  if (row->callable_type_ordinal_u16 >= layout->callable_types.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function callable type ordinal is out of range");
  }
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      iree_vm_bytecode_function_callable_type(layout, row);
  const iree_vm_bytecode_callable_type_flags_t expected_callable_flags =
      iree_any_bit_set(row->flags_u16, IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD)
          ? IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD
          : 0;
  if (callable_type->flags_u16 != expected_callable_flags) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function behavior does not match its canonical callable type");
  }
  const iree_vm_bytecode_v0_signature_row_t* signature =
      iree_vm_bytecode_function_signature(layout, row);
  const uint16_t required_value_registers =
      iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
               iree_max(signature->argument_value_count_u16,
                        signature->result_value_count_u16));
  const uint16_t required_ref_registers =
      iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
               iree_max(signature->argument_ref_count_u16,
                        signature->result_ref_count_u16));
  const uint16_t required_function_registers =
      iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
               iree_max(signature->argument_function_count_u16,
                        signature->result_function_count_u16));
  if (row->value_register_count_u16 < required_value_registers ||
      row->ref_register_count_u16 < required_ref_registers ||
      row->function_register_count_u16 < required_function_registers) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function register banks do not cover their direct signature prefixes");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_functions(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Functions section"};
  const iree_vm_bytecode_v0_functions_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->function_count_u32 == 0 || header->function_count_u32 > 65536u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Functions count must be in [1, 65536]");
  }
  const iree_vm_bytecode_v0_function_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->function_count_u32, sizeof(*rows), iree_alignof(*rows),
      (const void**)&rows));
  uint32_t switch_target_count = 0;
  uint32_t bytecode_length = 0;
  for (uint32_t i = 0; i < header->function_count_u32; ++i) {
    const iree_vm_bytecode_v0_function_row_t* row = &rows[i];
    if ((row->flags_u16 & ~(IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD |
                            IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)) != 0 ||
        row->bytecode_length_u32 == 0 || row->bytecode_length_u32 % 4 != 0 ||
        row->bytecode_length_u32 > INT32_MAX ||
        row->switch_target_base_u32 != switch_target_count ||
        row->bytecode_offset_u32 != bytecode_length ||
        row->switch_target_entry_count_u32 > UINT32_MAX - switch_target_count ||
        row->bytecode_length_u32 > UINT32_MAX - bytecode_length ||
        row->block_count_u32 == 0 || row->block_count_u32 > 65536u ||
        row->block_count_u32 > row->bytecode_length_u32 / 4u ||
        row->value_register_count_u16 > 256 ||
        row->ref_register_count_u16 > 256 ||
        row->function_register_count_u16 > 256 ||
        row->local_ref_count_u32 > (uint32_t)UINT16_MAX + 1u ||
        row->local_function_count_u32 > (uint32_t)UINT16_MAX + 1u ||
        row->reserved_u32[0] != 0 || row->reserved_u32[1] != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function row %" PRIu32 " is invalid", i);
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_verify_function_signature_structure(layout, row));
    switch_target_count += row->switch_target_entry_count_u32;
    bytecode_length += row->bytecode_length_u32;
  }
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, switch_target_count, sizeof(*switch_targets),
      iree_alignof(*switch_targets), (const void**)&switch_targets));
  const uint8_t* bytecode_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, bytecode_length, 4, (const void**)&bytecode_data));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  layout->functions.count = header->function_count_u32;
  layout->functions.rows = rows;
  layout->functions.switch_target_count = switch_target_count;
  layout->functions.switch_targets = switch_targets;
  layout->functions.bytecode_data = bytecode_data;
  layout->functions.bytecode_length = bytecode_length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_constants(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  if (span.data_length % sizeof(iree_vm_bytecode_v0_constant_cell_t) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Constants cells do not consume the section");
  }
  const iree_host_size_t count =
      span.data_length / sizeof(iree_vm_bytecode_v0_constant_cell_t);
  if (count == 0 || count > 65536u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Constants count must be in [1, 65536]");
  }
  layout->constants.count = (uint32_t)count;
  layout->constants.cells =
      (const iree_vm_bytecode_v0_constant_cell_t*)span.data;
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
      sizeof(iree_vm_bytecode_process_state_t), &layout.total_size,
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

static iree_status_t iree_vm_bytecode_verify_globals(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout,
    iree_vm_bytecode_process_layout_t* out_process_layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Globals section"};
  const iree_vm_bytecode_v0_globals_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if ((header->value_count_u32 == 0 && header->ref_count_u32 == 0 &&
       header->function_count_u32 == 0) ||
      header->value_count_u32 > 65536u || header->ref_count_u32 > 65536u ||
      header->function_count_u32 > 65536u ||
      header->immutable_value_count_u32 > header->value_count_u32 ||
      header->immutable_ref_count_u32 > header->ref_count_u32 ||
      header->immutable_function_count_u32 > header->function_count_u32 ||
      header->reserved_u32[0] != 0 || header->reserved_u32[1] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Globals counts or reserved fields are invalid");
  }
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* refs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->ref_count_u32, sizeof(*refs), iree_alignof(*refs),
      (const void**)&refs));
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* functions = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->function_count_u32, sizeof(*functions),
      iree_alignof(*functions), (const void**)&functions));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  for (uint32_t i = 0; i < header->ref_count_u32; ++i) {
    if (refs[i].ref_type_ordinal_u16 >= layout->ref_types.entry_count ||
        (refs[i].flags_u16 & ~IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ref-global descriptor is invalid");
    }
  }
  for (uint32_t i = 0; i < header->function_count_u32; ++i) {
    if (functions[i].callable_type_ordinal_u16 >=
            layout->callable_types.count ||
        (functions[i].flags_u16 &
         ~IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE) != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function-global descriptor is invalid");
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_plan_process_layout(header, out_process_layout));
  layout->globals.header = header;
  layout->globals.refs = refs;
  layout->globals.functions = functions;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_rodata(
    iree_const_byte_span_t span, uint32_t payload_alignment,
    iree_vm_bytecode_module_layout_t* layout,
    iree_vm_bytecode_rodata_storage_plan_t* out_storage_plan) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Rodata section"};
  const iree_vm_bytecode_v0_rodata_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->block_count_u32 == 0 || header->block_count_u32 > 65536u ||
      header->zero_padding_u32 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Rodata header is invalid");
  }
  const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->block_count_u32, sizeof(*descriptors),
      iree_alignof(*descriptors), (const void**)&descriptors));
  uint32_t maximum_alignment = IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT;
  for (uint32_t i = 0; i < header->block_count_u32; ++i) {
    const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptor =
        &descriptors[i];
    if (!iree_host_size_is_power_of_two(descriptor->minimum_alignment_u32) ||
        descriptor->reserved_u32 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rodata block descriptor is invalid");
    }
    maximum_alignment =
        iree_max(maximum_alignment, descriptor->minimum_alignment_u32);
  }
  if (payload_alignment != maximum_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Rodata payload alignment %" PRIu32
                            " does not match maximum block alignment %" PRIu32,
                            payload_alignment, maximum_alignment);
  }

  iree_vm_bytecode_rodata_storage_plan_t storage_plan = {0};
  const iree_host_size_t blocks_offset = cursor.offset;
  for (uint32_t i = 0; i < header->block_count_u32; ++i) {
    const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptor =
        &descriptors[i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_align(
        &cursor, descriptor->minimum_alignment_u32));
    if (descriptor->byte_length_u64 > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rodata block is not host-addressable");
    }
    const uint8_t* block = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
        &cursor, (iree_host_size_t)descriptor->byte_length_u64,
        descriptor->minimum_alignment_u32, (const void**)&block));
    if (!iree_host_ptr_has_alignment(block,
                                     descriptor->minimum_alignment_u32)) {
      if (!iree_host_size_checked_align(storage_plan.copy_length,
                                        descriptor->minimum_alignment_u32,
                                        &storage_plan.copy_length) ||
          !iree_host_size_checked_add(
              storage_plan.copy_length,
              (iree_host_size_t)descriptor->byte_length_u64,
              &storage_plan.copy_length)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rodata fallback storage overflows host size");
      }
      storage_plan.copy_alignment =
          iree_max(storage_plan.copy_alignment,
                   (iree_host_size_t)descriptor->minimum_alignment_u32);
    }
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  layout->rodata.count = header->block_count_u32;
  layout->rodata.descriptors = descriptors;
  layout->rodata.section_begin = span.data;
  layout->rodata.blocks_offset = blocks_offset;
  *out_storage_plan = storage_plan;
  return iree_ok_status();
}

static const iree_vm_bytecode_v0_callable_type_row_t*
iree_vm_bytecode_presentation_callable(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_presentation_entry_row_t* entry) {
  uint16_t callable_type_ordinal = 0;
  if (entry->declaration_kind_u16 ==
      IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT) {
    callable_type_ordinal =
        layout->imports.entries[entry->declaration_ordinal_u16]
            .callable_type_ordinal_u16;
  } else {
    callable_type_ordinal = layout->exports.rows[entry->declaration_ordinal_u16]
                                .callable_type_ordinal_u16;
  }
  return &layout->callable_types.rows[callable_type_ordinal];
}

static iree_status_t iree_vm_bytecode_verify_presentation(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Presentation section"};
  const iree_vm_bytecode_v0_presentation_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->entry_count_u32 == 0 || header->entry_count_u32 > 131071u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Presentation entry count must be in [1, 131071]");
  }
  const iree_vm_bytecode_v0_presentation_entry_row_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->entry_count_u32, sizeof(*entries),
      iree_alignof(*entries), (const void**)&entries));

  uint32_t field_count = 0;
  uint16_t previous_kind = 0;
  uint16_t previous_ordinal = 0;
  for (uint32_t i = 0; i < header->entry_count_u32; ++i) {
    const iree_vm_bytecode_v0_presentation_entry_row_t* entry = &entries[i];
    uint32_t declaration_count = 0;
    switch (entry->declaration_kind_u16) {
      case IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT:
        declaration_count = layout->imports.entry_count;
        break;
      case IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_EXPORT:
        declaration_count = layout->exports.count;
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Presentation declaration kind is invalid");
    }
    if (entry->declaration_ordinal_u16 >= declaration_count ||
        (i != 0 && (entry->declaration_kind_u16 < previous_kind ||
                    (entry->declaration_kind_u16 == previous_kind &&
                     entry->declaration_ordinal_u16 <= previous_ordinal)))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Presentation declarations must be valid and strictly ordered");
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
        &layout->strings, entry->documentation_string_u16,
        IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NULLABLE,
        "presentation documentation"));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
        &layout->strings, entry->authored_type_string_u16,
        IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NULLABLE,
        "presentation authored type"));

    const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
        iree_vm_bytecode_presentation_callable(layout, entry);
    const iree_vm_bytecode_v0_signature_row_t* signature =
        &layout->signatures.rows[callable_type->signature_ordinal_u16];
    const uint32_t declaration_field_count =
        iree_vm_bytecode_signature_argument_count(signature) +
        iree_vm_bytecode_signature_result_count(signature);
    if (entry->field_base_u32 != field_count ||
        declaration_field_count > UINT32_MAX - field_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Presentation field prefix is invalid");
    }
    field_count += declaration_field_count;
    previous_kind = entry->declaration_kind_u16;
    previous_ordinal = entry->declaration_ordinal_u16;
  }

  const iree_vm_bytecode_v0_presentation_field_row_t* fields = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, field_count, sizeof(*fields), iree_alignof(*fields),
      (const void**)&fields));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  for (uint32_t i = 0; i < header->entry_count_u32; ++i) {
    const iree_vm_bytecode_v0_presentation_entry_row_t* entry = &entries[i];
    const uint32_t field_end = i + 1 < header->entry_count_u32
                                   ? entries[i + 1].field_base_u32
                                   : field_count;
    bool has_value = entry->documentation_string_u16 != UINT16_MAX ||
                     entry->authored_type_string_u16 != UINT16_MAX;
    for (uint32_t field_i = entry->field_base_u32; field_i < field_end;
         ++field_i) {
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
          &layout->strings, fields[field_i].name_string_u16,
          IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NULLABLE,
          "presentation field name"));
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
          &layout->strings, fields[field_i].authored_type_string_u16,
          IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NULLABLE,
          "presentation field authored type"));
      has_value |= fields[field_i].name_string_u16 != UINT16_MAX ||
                   fields[field_i].authored_type_string_u16 != UINT16_MAX;
    }
    if (!has_value) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Presentation entry has no authored information");
    }
  }

  layout->presentation.entry_count = header->entry_count_u32;
  layout->presentation.entries = entries;
  layout->presentation.field_count = field_count;
  layout->presentation.fields = fields;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_metadata_value(
    uint16_t value_type, iree_const_byte_span_t value) {
  switch (value_type) {
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_INVALID:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Metadata value type is invalid");
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL:
      if (value.data_length != 1 || value.data[0] > 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "BOOL metadata must be exactly one canonical 0/1 byte");
      }
      break;
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64:
      if (value.data_length != 8) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "64-bit metadata values must contain exactly eight bytes");
      }
      break;
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8:
      if (!iree_unicode_utf8_validate(iree_make_string_view(
              (const char*)value.data, value.data_length))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "UTF8 metadata contains invalid UTF-8");
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_metadata_entry_range(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_metadata_entry_row_t* entries,
    const iree_vm_bytecode_v0_metadata_value_offset_t* value_offsets,
    const uint8_t* value_data, uint32_t entry_base, uint32_t entry_count) {
  iree_string_view_t previous_key = iree_string_view_empty();
  for (uint32_t i = 0; i < entry_count; ++i) {
    const uint32_t entry_ordinal = entry_base + i;
    const iree_vm_bytecode_v0_metadata_entry_row_t* entry =
        &entries[entry_ordinal];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_string_ordinal(
        &layout->strings, entry->key_string_u16,
        IREE_VM_BYTECODE_STRING_ORDINAL_FLAG_NONEMPTY, "metadata key"));
    const iree_string_view_t key =
        iree_vm_bytecode_string_at(&layout->strings, entry->key_string_u16);
    if (i != 0 && iree_string_view_compare(previous_key, key) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Metadata keys must be strictly sorted within each scope");
    }
    const iree_host_size_t value_begin =
        (iree_host_size_t)value_offsets[entry_ordinal];
    const iree_host_size_t value_end =
        (iree_host_size_t)value_offsets[entry_ordinal + 1];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_value(
        entry->value_type_u16,
        iree_make_const_byte_span(value_data + value_begin,
                                  value_end - value_begin)));
    previous_key = key;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_metadata_scopes(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_metadata_scope_row_t* scopes,
    uint32_t scope_count, uint32_t declaration_count, uint32_t* inout_base) {
  uint16_t previous_ordinal = 0;
  for (uint32_t i = 0; i < scope_count; ++i) {
    const iree_vm_bytecode_v0_metadata_scope_row_t* scope = &scopes[i];
    if (scope->declaration_ordinal_u16 >= declaration_count ||
        scope->entry_count_u16 == 0 || scope->entry_base_u32 != *inout_base ||
        (i != 0 && scope->declaration_ordinal_u16 <= previous_ordinal) ||
        scope->entry_count_u16 >
            layout->metadata.header->total_entry_count_u32 - *inout_base) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Metadata declaration scopes are invalid or out of order");
    }
    *inout_base += scope->entry_count_u16;
    previous_ordinal = scope->declaration_ordinal_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_metadata(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Metadata section"};
  const iree_vm_bytecode_v0_metadata_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(*header), iree_alignof(*header), (const void**)&header));
  if (header->total_entry_count_u32 == 0 ||
      header->module_entry_count_u32 > header->total_entry_count_u32 ||
      header->import_scope_count_u32 > layout->imports.entry_count ||
      header->export_scope_count_u32 > layout->exports.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata header counts are invalid");
  }
  const iree_vm_bytecode_v0_metadata_scope_row_t* import_scopes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->import_scope_count_u32, sizeof(*import_scopes),
      iree_alignof(*import_scopes), (const void**)&import_scopes));
  const iree_vm_bytecode_v0_metadata_scope_row_t* export_scopes = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->export_scope_count_u32, sizeof(*export_scopes),
      iree_alignof(*export_scopes), (const void**)&export_scopes));
  const iree_vm_bytecode_v0_metadata_entry_row_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->total_entry_count_u32, sizeof(*entries),
      iree_alignof(*entries), (const void**)&entries));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_cursor_align(&cursor, iree_alignof(uint64_t)));
  iree_host_size_t value_offset_count = 0;
  if (!iree_host_size_checked_add(
          (iree_host_size_t)header->total_entry_count_u32, 1,
          &value_offset_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata offset count overflows");
  }
  const iree_vm_bytecode_v0_metadata_value_offset_t* value_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, value_offset_count, sizeof(*value_offsets),
      iree_alignof(*value_offsets), (const void**)&value_offsets));
  const uint8_t* value_data = span.data + cursor.offset;
  const iree_host_size_t value_data_length = span.data_length - cursor.offset;
  cursor.offset = span.data_length;

  if (value_offsets[0] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata value offsets must begin at zero");
  }
  for (uint32_t i = 0; i < header->total_entry_count_u32; ++i) {
    if (value_offsets[i] > value_offsets[i + 1] ||
        value_offsets[i + 1] > value_data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Metadata value offsets are invalid");
    }
  }
  if (value_offsets[header->total_entry_count_u32] != value_data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metadata value offsets do not consume the value bytes");
  }

  layout->metadata.header = header;
  layout->metadata.import_scopes = import_scopes;
  layout->metadata.export_scopes = export_scopes;
  layout->metadata.entries = entries;
  layout->metadata.value_offsets = value_offsets;
  layout->metadata.value_data = value_data;
  layout->metadata.value_data_length = value_data_length;

  uint32_t entry_base = header->module_entry_count_u32;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_scopes(
      layout, import_scopes, header->import_scope_count_u32,
      layout->imports.entry_count, &entry_base));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_scopes(
      layout, export_scopes, header->export_scope_count_u32,
      layout->exports.count, &entry_base));
  if (entry_base != header->total_entry_count_u32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metadata scopes do not partition the complete entry table");
  }

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_entry_range(
      layout, entries, value_offsets, value_data, 0,
      header->module_entry_count_u32));
  for (uint32_t i = 0; i < header->import_scope_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_entry_range(
        layout, entries, value_offsets, value_data,
        import_scopes[i].entry_base_u32, import_scopes[i].entry_count_u16));
  }
  for (uint32_t i = 0; i < header->export_scope_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_entry_range(
        layout, entries, value_offsets, value_data,
        export_scopes[i].entry_base_u32, export_scopes[i].entry_count_u16));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_exports_against_functions(
    const iree_vm_bytecode_module_layout_t* layout) {
  for (uint32_t i = 0; i < layout->exports.count; ++i) {
    const iree_vm_bytecode_v0_export_row_t* export_row =
        &layout->exports.rows[i];
    if (export_row->function_ordinal_u16 >= layout->functions.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "export function ordinal is out of range");
    }
    const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
        &layout->callable_types.rows[export_row->callable_type_ordinal_u16];
    const iree_vm_bytecode_v0_function_row_t* function =
        &layout->functions.rows[export_row->function_ordinal_u16];
    const iree_vm_bytecode_v0_callable_type_row_t* function_callable_type =
        iree_vm_bytecode_function_callable_type(layout, function);
    if (callable_type->signature_ordinal_u16 !=
            function_callable_type->signature_ordinal_u16 ||
        (iree_any_bit_set(function->flags_u16,
                          IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD) &&
         !iree_any_bit_set(callable_type->flags_u16,
                           IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "export callable contract does not match its function");
    }
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_module_verify_structure(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  iree_vm_bytecode_section_map_t sections = {0};
  iree_vm_bytecode_module_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_envelope(contents, &sections));
  plan.layout.image.header = sections.header;
  plan.layout.image.section_count = sections.section_count;
  plan.layout.image.sections = sections.rows;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_verify_requirements(&sections, &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_strings(
      sections.spans[IREE_VM_BYTECODE_SECTION_STRINGS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_types(
      sections.spans[IREE_VM_BYTECODE_SECTION_REF_TYPES], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_signatures(
      sections.spans[IREE_VM_BYTECODE_SECTION_SIGNATURES], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_callable_types(
      sections.spans[IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_imports(
      sections.spans[IREE_VM_BYTECODE_SECTION_IMPORTS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_exports(
      sections.spans[IREE_VM_BYTECODE_SECTION_EXPORTS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_functions(
      sections.spans[IREE_VM_BYTECODE_SECTION_FUNCTIONS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_constants(
      sections.spans[IREE_VM_BYTECODE_SECTION_CONSTANTS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_globals(
      sections.spans[IREE_VM_BYTECODE_SECTION_GLOBALS], &plan.layout,
      &plan.process_layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_rodata(
      sections.spans[IREE_VM_BYTECODE_SECTION_RODATA],
      sections.payload_alignments[IREE_VM_BYTECODE_SECTION_RODATA],
      &plan.layout, &plan.rodata_storage_plan));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_presentation(
      sections.spans[IREE_VM_BYTECODE_SECTION_PRESENTATION], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata(
      sections.spans[IREE_VM_BYTECODE_SECTION_METADATA], &plan.layout));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_verify_exports_against_functions(&plan.layout));
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_module_verify_inspectable(
    const iree_vm_bytecode_module_plan_t* plan) {
  for (uint16_t i = 0; i < plan->layout.requirements.count; ++i) {
    const iree_vm_bytecode_v0_requirement_row_t* requirement =
        &plan->layout.requirements.rows[i];
    if (requirement->page_id_u16 != IREE_VM_ISA_PAGE_HAL) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "extension page 0x%02" PRIx16
                              " is unknown to inspection tooling",
                              requirement->page_id_u16);
    }
    if (requirement->major_u16 != IREE_VM_ISA_HAL_MAJOR ||
        requirement->required_minor_u16 > IREE_VM_ISA_HAL_MINOR) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "bytecode HAL version %" PRIu16 ".%" PRIu16
                              " is not supported by inspection version %d.%d",
                              requirement->major_u16,
                              requirement->required_minor_u16,
                              IREE_VM_ISA_HAL_MAJOR, IREE_VM_ISA_HAL_MINOR);
    }
  }
  return iree_ok_status();
}
