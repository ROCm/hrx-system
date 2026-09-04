// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verifier.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/vm/bytecode/verifier_data.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/execution.h"

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

typedef struct iree_vm_bytecode_section_map_t {
  // Fixed image header at byte zero.
  const iree_vm_bytecode_v0_image_header_t* header;
  // Strictly type-sorted section directory rows.
  const iree_vm_bytecode_v0_section_directory_row_t* rows;
  // Number of rows in |rows|.
  uint16_t count;
  // Known Core section payloads indexed by section type.
  iree_const_byte_span_t spans[IREE_VM_BYTECODE_SECTION_METADATA + 1];
  // Known payload alignments indexed by section type.
  uint32_t alignments[IREE_VM_BYTECODE_SECTION_METADATA + 1];
  // Extension authorities owning at least one section.
  uint16_t extension_pages;
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

static bool iree_vm_bytecode_verify_signature_descriptor(
    const iree_vm_bytecode_module_layout_t* layout, uint16_t kind,
    uint16_t type_ordinal) {
  if (kind >= IREE_VM_BYTECODE_SIGNATURE_KIND_I8 &&
      kind <= IREE_VM_BYTECODE_SIGNATURE_KIND_F64) {
    return type_ordinal == 0;
  } else if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
    return type_ordinal < layout->ref_types.entry_count;
  } else if (kind == IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
    // Callable types are parsed after signatures and checked as one complete
    // topologically ordered domain.
    return true;
  }
  return false;
}

static iree_status_t iree_vm_bytecode_verify_record(
    iree_vm_bytecode_module_record_ordinal_t record_ordinal,
    const uint8_t* record, const iree_vm_bytecode_module_layout_t* layout) {
  switch (record_ordinal) {
#include "iree/vm/bytecode/verifier/module_cases.inl"
    default:
      IREE_ASSERT_UNREACHABLE("generated module-record verifier is incomplete");
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "module record %u violates field constraints",
                          (unsigned)record_ordinal);
}

static iree_status_t iree_vm_bytecode_verify_envelope(
    iree_const_byte_span_t contents,
    iree_vm_bytecode_section_map_t* out_sections) {
  if (!contents.data || contents.data_length == 0 ||
      !iree_host_ptr_has_alignment(contents.data,
                                   IREE_VM_BYTECODE_IMAGE_ALIGNMENT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bytecode image must be nonempty and eight-byte aligned");
  }
  iree_vm_bytecode_cursor_t cursor = {contents, 0, "bytecode image"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_image_header_t),
      iree_alignof(iree_vm_bytecode_v0_image_header_t), &slice));
  const iree_vm_bytecode_v0_image_header_t* header =
      (const iree_vm_bytecode_v0_image_header_t*)slice.data;
  iree_vm_bytecode_module_layout_t empty_layout = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_IMAGE_HEADER, (const uint8_t*)header,
      &empty_layout));

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->section_count_u16,
      sizeof(iree_vm_bytecode_v0_section_directory_row_t),
      iree_alignof(iree_vm_bytecode_v0_section_directory_row_t), &slice));
  const iree_vm_bytecode_v0_section_directory_row_t* rows =
      (const iree_vm_bytecode_v0_section_directory_row_t*)slice.data;

  iree_vm_bytecode_section_map_t sections = {0};
  sections.header = header;
  sections.rows = rows;
  sections.count = header->section_count_u16;
  uint16_t previous_type = 0;
  for (uint16_t i = 0; i < header->section_count_u16; ++i) {
    const iree_vm_bytecode_v0_section_directory_row_t* row = &rows[i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_SECTION_DIRECTORY_ROW,
        (const uint8_t*)row, &empty_layout));
    if (row->section_type_u16 == 0 ||
        (i != 0 && row->section_type_u16 <= previous_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "section types must be nonzero and strictly increasing");
    }
    const bool is_known =
        row->section_type_u16 <= IREE_VM_BYTECODE_SECTION_METADATA;
    if (is_known) {
      const uint32_t section_descriptor =
          iree_vm_bytecode_module_section_verification[row->section_type_u16];
      if (iree_vm_bytecode_section_verification_since_minor(
              section_descriptor) > header->core_required_minor_u16 ||
          row->section_flags_u16 !=
              iree_vm_bytecode_section_verification_required_flags(
                  section_descriptor)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "known section 0x%04" PRIx16
                                " has invalid version or flags",
                                row->section_type_u16);
      }
      if (row->byte_length_u64 == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "known section 0x%04" PRIx16 " is empty",
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
      sections.extension_pages |= (uint16_t)(1u << (authority - 0xF0));
    }

    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_cursor_align(&cursor, row->payload_alignment_u32));
    if (row->byte_length_u64 > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "section 0x%04" PRIx16 " is not host-addressable",
                              row->section_type_u16);
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
        &cursor, (iree_host_size_t)row->byte_length_u64,
        row->payload_alignment_u32, &slice));
    if (is_known) {
      sections.spans[row->section_type_u16] = iree_make_const_byte_span(
          slice.data, (iree_host_size_t)row->byte_length_u64);
      sections.alignments[row->section_type_u16] = row->payload_alignment_u32;
    }
    previous_type = row->section_type_u16;
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  *out_sections = sections;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_requirements(
    const iree_vm_bytecode_section_map_t* sections,
    iree_vm_bytecode_module_layout_t* layout) {
  const iree_const_byte_span_t span =
      sections->spans[IREE_VM_BYTECODE_SECTION_REQUIREMENTS];
  if (iree_const_byte_span_is_empty(span)) {
    if (sections->extension_pages != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "extension sections require a Requirements section");
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
  iree_vm_bytecode_module_layout_t empty_layout = {0};
  uint16_t previous_page = 0;
  uint16_t declared_pages = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_REQUIREMENT_ROW,
        (const uint8_t*)&rows[i], &empty_layout));
    if (i != 0 && rows[i].page_id_u16 <= previous_page) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Requirements pages must be strictly increasing");
    }
    declared_pages |= (uint16_t)(1u << (rows[i].page_id_u16 - 0xF0));
    previous_page = rows[i].page_id_u16;
  }
  if ((sections->extension_pages & ~declared_pages) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "extension section has no Requirements declaration");
  }
  layout->requirements.rows = rows;
  layout->requirements.count = (uint16_t)count;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_strings(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Strings section"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_strings_header_t),
      iree_alignof(iree_vm_bytecode_v0_strings_header_t), &slice));
  const iree_vm_bytecode_v0_strings_header_t* header =
      (const iree_vm_bytecode_v0_strings_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_STRINGS_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, (iree_host_size_t)header->string_count_u32 + 1,
      sizeof(iree_vm_bytecode_v0_string_offset_t),
      iree_alignof(iree_vm_bytecode_v0_string_offset_t), &slice));
  const iree_vm_bytecode_v0_string_offset_t* offsets =
      (const iree_vm_bytecode_v0_string_offset_t*)slice.data;
  const iree_host_size_t data_length = span.data_length - cursor.offset;
  if (data_length > UINT32_MAX || offsets[0].byte_offset_u32 != 0 ||
      offsets[header->string_count_u32].byte_offset_u32 != data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strings offsets do not cover the byte tail");
  }
  const uint8_t* data = span.data + cursor.offset;
  for (uint32_t i = 0; i < header->string_count_u32; ++i) {
    const uint32_t begin = offsets[i].byte_offset_u32;
    const uint32_t end = offsets[i + 1].byte_offset_u32;
    if (begin > end || end > data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Strings offsets are not monotonic");
    }
    const iree_string_view_t value =
        iree_make_string_view((const char*)data + begin, end - begin);
    if (iree_string_view_find_char(value, '\0', 0) != IREE_STRING_VIEW_NPOS ||
        !iree_unicode_utf8_validate(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "string %" PRIu32 " is not NUL-free valid UTF-8",
                              i);
    }
  }
  layout->strings.offsets = offsets;
  layout->strings.data = data;
  layout->strings.count = header->string_count_u32;
  layout->strings.data_length = (uint32_t)data_length;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_ref_types(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "RefTypes section"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_ref_types_header_t),
      iree_alignof(iree_vm_bytecode_v0_ref_types_header_t), &slice));
  const iree_vm_bytecode_v0_ref_types_header_t* header =
      (const iree_vm_bytecode_v0_ref_types_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPES_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->group_count_u32,
      sizeof(iree_vm_bytecode_v0_ref_type_group_row_t),
      iree_alignof(iree_vm_bytecode_v0_ref_type_group_row_t), &slice));
  const iree_vm_bytecode_v0_ref_type_group_row_t* groups =
      (const iree_vm_bytecode_v0_ref_type_group_row_t*)slice.data;
  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < header->group_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_GROUP_ROW,
        (const uint8_t*)&groups[i], layout));
    if (groups[i].entry_count_u32 > 65536u - entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RefTypes entry count exceeds 65536");
    }
    entry_count += groups[i].entry_count_u32;
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, entry_count, sizeof(iree_vm_bytecode_v0_ref_type_entry_row_t),
      iree_alignof(iree_vm_bytecode_v0_ref_type_entry_row_t), &slice));
  const iree_vm_bytecode_v0_ref_type_entry_row_t* entries =
      (const iree_vm_bytecode_v0_ref_type_entry_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  uint32_t entry_base = 0;
  iree_string_view_t previous_namespace = iree_string_view_empty();
  for (uint32_t group_i = 0; group_i < header->group_count_u32; ++group_i) {
    const iree_vm_bytecode_v0_ref_type_group_row_t* group = &groups[group_i];
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
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
          IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_ENTRY_ROW,
          (const uint8_t*)entry, layout));
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
  layout->ref_types.groups = groups;
  layout->ref_types.entries = entries;
  layout->ref_types.group_count = header->group_count_u32;
  layout->ref_types.entry_count = entry_count;
  return iree_ok_status();
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
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_DESCRIPTOR_ROW,
        (const uint8_t*)&descriptors[i], layout));
    if (descriptors[i].kind_u16 >= IREE_VM_BYTECODE_SIGNATURE_KIND_I8 &&
        descriptors[i].kind_u16 <= IREE_VM_BYTECODE_SIGNATURE_KIND_F64) {
      ++value_count;
    } else if (descriptors[i].kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
      ++ref_count;
    } else {
      ++function_count;
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
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_signatures_header_t),
      iree_alignof(iree_vm_bytecode_v0_signatures_header_t), &slice));
  const iree_vm_bytecode_v0_signatures_header_t* header =
      (const iree_vm_bytecode_v0_signatures_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURES_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->signature_count_u32,
      sizeof(iree_vm_bytecode_v0_signature_row_t),
      iree_alignof(iree_vm_bytecode_v0_signature_row_t), &slice));
  const iree_vm_bytecode_v0_signature_row_t* rows =
      (const iree_vm_bytecode_v0_signature_row_t*)slice.data;
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
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, descriptor_count,
      sizeof(iree_vm_bytecode_v0_signature_descriptor_row_t),
      iree_alignof(iree_vm_bytecode_v0_signature_descriptor_row_t), &slice));
  const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
      (const iree_vm_bytecode_v0_signature_descriptor_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  layout->signatures.rows = rows;
  layout->signatures.descriptors = descriptors;
  layout->signatures.count = header->signature_count_u32;
  layout->signatures.descriptor_count = descriptor_count;
  for (uint32_t i = 0; i < header->signature_count_u32; ++i) {
    const iree_vm_bytecode_v0_signature_row_t* row = &rows[i];
    const uint32_t argument_count =
        iree_vm_bytecode_signature_argument_count(row);
    const iree_vm_bytecode_v0_signature_descriptor_row_t* signature =
        descriptors + row->descriptor_base_u32;
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
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_callable_types_header_t),
      iree_alignof(iree_vm_bytecode_v0_callable_types_header_t), &slice));
  const iree_vm_bytecode_v0_callable_types_header_t* header =
      (const iree_vm_bytecode_v0_callable_types_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_CALLABLE_TYPES_HEADER,
      (const uint8_t*)header, layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->callable_type_count_u32,
      sizeof(iree_vm_bytecode_v0_callable_type_row_t),
      iree_alignof(iree_vm_bytecode_v0_callable_type_row_t), &slice));
  const iree_vm_bytecode_v0_callable_type_row_t* rows =
      (const iree_vm_bytecode_v0_callable_type_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  layout->callable_types.rows = rows;
  layout->callable_types.count = header->callable_type_count_u32;
  for (uint32_t i = 0; i < header->callable_type_count_u32; ++i) {
    const iree_vm_bytecode_v0_callable_type_row_t* row = &rows[i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_CALLABLE_TYPE_ROW, (const uint8_t*)row,
        layout));
    const iree_vm_bytecode_v0_signature_row_t* signature =
        &layout->signatures.rows[row->signature_ordinal_u16];
    const uint32_t descriptor_count =
        iree_vm_bytecode_signature_argument_count(signature) +
        iree_vm_bytecode_signature_result_count(signature);
    const iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
        iree_vm_bytecode_signature_descriptors(&layout->signatures,
                                               row->signature_ordinal_u16);
    uint32_t expected_depth = 0;
    for (uint32_t j = 0; j < descriptor_count; ++j) {
      if (descriptors[j].kind_u16 != IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
        continue;
      }
      const uint16_t child_ordinal = descriptors[j].type_ordinal_u16;
      if (child_ordinal >= i ||
          rows[child_ordinal].nesting_depth_u16 == UINT16_MAX) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "callable function types are not topologically ordered");
      }
      expected_depth = iree_max(
          expected_depth, (uint32_t)rows[child_ordinal].nesting_depth_u16 + 1);
    }
    if (row->nesting_depth_u16 != expected_depth) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callable nesting depth is not canonical");
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
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_imports_header_t),
      iree_alignof(iree_vm_bytecode_v0_imports_header_t), &slice));
  const iree_vm_bytecode_v0_imports_header_t* header =
      (const iree_vm_bytecode_v0_imports_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_IMPORTS_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->group_count_u32,
      sizeof(iree_vm_bytecode_v0_import_group_row_t),
      iree_alignof(iree_vm_bytecode_v0_import_group_row_t), &slice));
  const iree_vm_bytecode_v0_import_group_row_t* groups =
      (const iree_vm_bytecode_v0_import_group_row_t*)slice.data;
  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < header->group_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_GROUP_ROW,
        (const uint8_t*)&groups[i], layout));
    if (groups[i].entry_count_u32 > 65536u - entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Imports entry count exceeds 65536");
    }
    entry_count += groups[i].entry_count_u32;
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, entry_count, sizeof(iree_vm_bytecode_v0_import_entry_row_t),
      iree_alignof(iree_vm_bytecode_v0_import_entry_row_t), &slice));
  const iree_vm_bytecode_v0_import_entry_row_t* entries =
      (const iree_vm_bytecode_v0_import_entry_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));

  uint32_t entry_base = 0;
  iree_string_view_t previous_module = iree_string_view_empty();
  for (uint32_t group_i = 0; group_i < header->group_count_u32; ++group_i) {
    const iree_vm_bytecode_v0_import_group_row_t* group = &groups[group_i];
    const iree_string_view_t module_name = iree_vm_bytecode_string_at(
        &layout->strings, group->module_name_string_u16);
    if (group_i != 0 &&
        iree_string_view_compare(previous_module, module_name) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Imports groups must be strictly byte-sorted");
    }
    iree_string_view_t previous_symbol = iree_string_view_empty();
    uint16_t previous_callable_type = 0;
    for (uint32_t entry_i = 0; entry_i < group->entry_count_u32; ++entry_i) {
      const iree_vm_bytecode_v0_import_entry_row_t* entry =
          &entries[entry_base + entry_i];
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
          IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_ENTRY_ROW,
          (const uint8_t*)entry, layout));
      const iree_string_view_t symbol_name = iree_vm_bytecode_string_at(
          &layout->strings, entry->symbol_name_string_u16);
      if (entry_i != 0) {
        const int comparison =
            iree_string_view_compare(previous_symbol, symbol_name);
        if (comparison > 0 ||
            (comparison == 0 &&
             entry->callable_type_ordinal_u16 <= previous_callable_type)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Imports entries must be sorted by symbol and callable type");
        }
      }
      previous_symbol = symbol_name;
      previous_callable_type = entry->callable_type_ordinal_u16;
    }
    entry_base += group->entry_count_u32;
    previous_module = module_name;
  }
  layout->imports.groups = groups;
  layout->imports.entries = entries;
  layout->imports.group_count = header->group_count_u32;
  layout->imports.entry_count = entry_count;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_exports(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Exports section"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_exports_header_t),
      iree_alignof(iree_vm_bytecode_v0_exports_header_t), &slice));
  const iree_vm_bytecode_v0_exports_header_t* header =
      (const iree_vm_bytecode_v0_exports_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_EXPORTS_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->export_count_u32,
      sizeof(iree_vm_bytecode_v0_export_row_t),
      iree_alignof(iree_vm_bytecode_v0_export_row_t), &slice));
  const iree_vm_bytecode_v0_export_row_t* rows =
      (const iree_vm_bytecode_v0_export_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  iree_string_view_t previous_name = iree_string_view_empty();
  for (uint32_t i = 0; i < header->export_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_EXPORT_ROW, (const uint8_t*)&rows[i],
        layout));
    const iree_string_view_t name =
        iree_vm_bytecode_string_at(&layout->strings, rows[i].name_string_u16);
    if (i != 0 && iree_string_view_compare(previous_name, name) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Exports names must be unique and strictly byte-sorted");
    }
    previous_name = name;
  }
  layout->exports.rows = rows;
  layout->exports.count = header->export_count_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_function_signature(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* row) {
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      iree_vm_bytecode_function_callable_type(layout, row);
  if (iree_any_bit_set(row->flags_u16,
                       IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD) &&
      !iree_any_bit_set(callable_type->flags_u16,
                        IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function behavior does not match its callable type");
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
        "function register banks do not cover its signature prefixes");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_functions(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout,
    uint32_t* out_maximum_block_count) {
  if (iree_const_byte_span_is_empty(span)) {
    *out_maximum_block_count = 0;
    return iree_ok_status();
  }
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Functions section"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_functions_header_t),
      iree_alignof(iree_vm_bytecode_v0_functions_header_t), &slice));
  const iree_vm_bytecode_v0_functions_header_t* header =
      (const iree_vm_bytecode_v0_functions_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_FUNCTIONS_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->function_count_u32,
      sizeof(iree_vm_bytecode_v0_function_row_t),
      iree_alignof(iree_vm_bytecode_v0_function_row_t), &slice));
  const iree_vm_bytecode_v0_function_row_t* rows =
      (const iree_vm_bytecode_v0_function_row_t*)slice.data;

  layout->functions.rows = rows;
  layout->functions.count = header->function_count_u32;
  uint32_t switch_target_count = 0;
  uint32_t bytecode_length = 0;
  uint32_t maximum_block_count = 0;
  for (uint32_t i = 0; i < header->function_count_u32; ++i) {
    const iree_vm_bytecode_v0_function_row_t* row = &rows[i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW, (const uint8_t*)row,
        layout));
    if (row->bytecode_length_u32 % 4 != 0 ||
        row->switch_target_base_u32 != switch_target_count ||
        row->bytecode_offset_u32 != bytecode_length ||
        row->switch_target_entry_count_u32 > UINT32_MAX - switch_target_count ||
        row->bytecode_length_u32 > UINT32_MAX - bytecode_length ||
        row->block_count_u32 > row->bytecode_length_u32 / 4u) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function row %" PRIu32 " has invalid canonical extents", i);
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_verify_function_signature(layout, row));
    switch_target_count += row->switch_target_entry_count_u32;
    bytecode_length += row->bytecode_length_u32;
    maximum_block_count = iree_max(maximum_block_count, row->block_count_u32);
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, switch_target_count,
      sizeof(iree_vm_bytecode_v0_switch_target_entry_t),
      iree_alignof(iree_vm_bytecode_v0_switch_target_entry_t), &slice));
  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets =
      (const iree_vm_bytecode_v0_switch_target_entry_t*)slice.data;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_cursor_take(&cursor, bytecode_length, 4, &slice));
  const uint8_t* bytecode_data = slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  layout->functions.switch_targets = switch_targets;
  layout->functions.bytecode_data = bytecode_data;
  layout->functions.switch_target_count = switch_target_count;
  layout->functions.bytecode_length = bytecode_length;
  *out_maximum_block_count = maximum_block_count;
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
  if (count > 65536u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Constants count exceeds 65536");
  }
  layout->constants.cells =
      (const iree_vm_bytecode_v0_constant_cell_t*)span.data;
  layout->constants.count = (uint32_t)count;
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

static iree_status_t iree_vm_bytecode_verify_globals(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout,
    iree_vm_bytecode_process_layout_t* out_process_layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Globals section"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_globals_header_t),
      iree_alignof(iree_vm_bytecode_v0_globals_header_t), &slice));
  const iree_vm_bytecode_v0_globals_header_t* header =
      (const iree_vm_bytecode_v0_globals_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_GLOBALS_HEADER, (const uint8_t*)header,
      layout));
  if ((header->value_count_u32 == 0 && header->ref_count_u32 == 0 &&
       header->function_count_u32 == 0) ||
      header->immutable_value_count_u32 > header->value_count_u32 ||
      header->immutable_ref_count_u32 > header->ref_count_u32 ||
      header->immutable_function_count_u32 > header->function_count_u32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Globals counts or immutable prefixes are invalid");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->ref_count_u32,
      sizeof(iree_vm_bytecode_v0_global_ref_descriptor_row_t),
      iree_alignof(iree_vm_bytecode_v0_global_ref_descriptor_row_t), &slice));
  const iree_vm_bytecode_v0_global_ref_descriptor_row_t* refs =
      (const iree_vm_bytecode_v0_global_ref_descriptor_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->function_count_u32,
      sizeof(iree_vm_bytecode_v0_global_function_descriptor_row_t),
      iree_alignof(iree_vm_bytecode_v0_global_function_descriptor_row_t),
      &slice));
  const iree_vm_bytecode_v0_global_function_descriptor_row_t* functions =
      (const iree_vm_bytecode_v0_global_function_descriptor_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_finish(&cursor));
  for (uint32_t i = 0; i < header->ref_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_GLOBAL_REF_DESCRIPTOR_ROW,
        (const uint8_t*)&refs[i], layout));
  }
  for (uint32_t i = 0; i < header->function_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_GLOBAL_FUNCTION_DESCRIPTOR_ROW,
        (const uint8_t*)&functions[i], layout));
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
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_rodata_header_t),
      iree_alignof(iree_vm_bytecode_v0_rodata_header_t), &slice));
  const iree_vm_bytecode_v0_rodata_header_t* header =
      (const iree_vm_bytecode_v0_rodata_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_RODATA_HEADER, (const uint8_t*)header,
      layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->block_count_u32,
      sizeof(iree_vm_bytecode_v0_rodata_block_descriptor_t),
      iree_alignof(iree_vm_bytecode_v0_rodata_block_descriptor_t), &slice));
  const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptors =
      (const iree_vm_bytecode_v0_rodata_block_descriptor_t*)slice.data;
  uint32_t maximum_alignment = IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT;
  for (uint32_t i = 0; i < header->block_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_RODATA_BLOCK_DESCRIPTOR,
        (const uint8_t*)&descriptors[i], layout));
    maximum_alignment =
        iree_max(maximum_alignment, descriptors[i].minimum_alignment_u32);
  }
  if (payload_alignment != maximum_alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Rodata section alignment does not match its blocks");
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
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
        &cursor, (iree_host_size_t)descriptor->byte_length_u64,
        descriptor->minimum_alignment_u32, &slice));
    if (!iree_host_ptr_has_alignment(slice.data,
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
  layout->rodata.descriptors = descriptors;
  layout->rodata.section_begin = span.data;
  layout->rodata.count = header->block_count_u32;
  layout->rodata.blocks_offset = blocks_offset;
  *out_storage_plan = storage_plan;
  return iree_ok_status();
}

static const iree_vm_bytecode_v0_callable_type_row_t*
iree_vm_bytecode_presentation_callable(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_presentation_entry_row_t* entry) {
  const uint16_t callable_type_ordinal =
      entry->declaration_kind_u16 ==
              IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT
          ? layout->imports.entries[entry->declaration_ordinal_u16]
                .callable_type_ordinal_u16
          : layout->exports.rows[entry->declaration_ordinal_u16]
                .callable_type_ordinal_u16;
  return &layout->callable_types.rows[callable_type_ordinal];
}

static iree_status_t iree_vm_bytecode_verify_presentation(
    iree_const_byte_span_t span, iree_vm_bytecode_module_layout_t* layout) {
  if (iree_const_byte_span_is_empty(span)) return iree_ok_status();
  iree_vm_bytecode_cursor_t cursor = {span, 0, "Presentation section"};
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_presentation_header_t),
      iree_alignof(iree_vm_bytecode_v0_presentation_header_t), &slice));
  const iree_vm_bytecode_v0_presentation_header_t* header =
      (const iree_vm_bytecode_v0_presentation_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_HEADER,
      (const uint8_t*)header, layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->entry_count_u32,
      sizeof(iree_vm_bytecode_v0_presentation_entry_row_t),
      iree_alignof(iree_vm_bytecode_v0_presentation_entry_row_t), &slice));
  const iree_vm_bytecode_v0_presentation_entry_row_t* entries =
      (const iree_vm_bytecode_v0_presentation_entry_row_t*)slice.data;

  uint32_t field_count = 0;
  uint16_t previous_kind = 0;
  uint16_t previous_ordinal = 0;
  for (uint32_t i = 0; i < header->entry_count_u32; ++i) {
    const iree_vm_bytecode_v0_presentation_entry_row_t* entry = &entries[i];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_ENTRY_ROW,
        (const uint8_t*)entry, layout));
    uint32_t declaration_count = 0;
    if (entry->declaration_kind_u16 ==
        IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT) {
      declaration_count = layout->imports.entry_count;
    } else {
      declaration_count = layout->exports.count;
    }
    if (entry->declaration_ordinal_u16 >= declaration_count ||
        (i != 0 && (entry->declaration_kind_u16 < previous_kind ||
                    (entry->declaration_kind_u16 == previous_kind &&
                     entry->declaration_ordinal_u16 <= previous_ordinal)))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Presentation declarations are invalid or out of order");
    }
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
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, field_count,
      sizeof(iree_vm_bytecode_v0_presentation_field_row_t),
      iree_alignof(iree_vm_bytecode_v0_presentation_field_row_t), &slice));
  const iree_vm_bytecode_v0_presentation_field_row_t* fields =
      (const iree_vm_bytecode_v0_presentation_field_row_t*)slice.data;
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
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
          IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_FIELD_ROW,
          (const uint8_t*)&fields[field_i], layout));
      has_value |= fields[field_i].name_string_u16 != UINT16_MAX ||
                   fields[field_i].authored_type_string_u16 != UINT16_MAX;
    }
    if (!has_value) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Presentation entry has no authored information");
    }
  }
  layout->presentation.entries = entries;
  layout->presentation.fields = fields;
  layout->presentation.entry_count = header->entry_count_u32;
  layout->presentation.field_count = field_count;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_metadata_value(
    uint16_t value_type, iree_const_byte_span_t value) {
  switch (value_type) {
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL:
      if (value.data_length == 1 && value.data[0] <= 1) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "BOOL metadata must be one canonical zero or one byte");
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64:
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64:
      if (value.data_length == 8) return iree_ok_status();
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "64-bit metadata value must contain exactly eight bytes");
    case IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8:
      if (!iree_unicode_utf8_validate(iree_make_string_view(
              (const char*)value.data, value.data_length))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "UTF8 metadata is not valid UTF-8");
      }
      return iree_ok_status();
    default:
      // BYTES and unknown nonzero types are opaque spans.
      return iree_ok_status();
  }
}

static iree_status_t iree_vm_bytecode_verify_metadata_entries(
    const iree_vm_bytecode_module_layout_t* layout, uint32_t entry_base,
    uint32_t entry_count) {
  iree_string_view_t previous_key = iree_string_view_empty();
  for (uint32_t i = 0; i < entry_count; ++i) {
    const uint32_t entry_ordinal = entry_base + i;
    const iree_vm_bytecode_v0_metadata_entry_row_t* entry =
        &layout->metadata.entries[entry_ordinal];
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_METADATA_ENTRY_ROW,
        (const uint8_t*)entry, layout));
    const iree_string_view_t key =
        iree_vm_bytecode_string_at(&layout->strings, entry->key_string_u16);
    if (i != 0 && iree_string_view_compare(previous_key, key) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Metadata keys must be unique and sorted within each scope");
    }
    const iree_host_size_t value_begin =
        (iree_host_size_t)layout->metadata.value_offsets[entry_ordinal]
            .byte_offset_u64;
    const iree_host_size_t value_end =
        (iree_host_size_t)layout->metadata.value_offsets[entry_ordinal + 1]
            .byte_offset_u64;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_value(
        entry->value_type_u16,
        iree_make_const_byte_span(layout->metadata.value_data + value_begin,
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
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_METADATA_SCOPE_ROW,
        (const uint8_t*)scope, layout));
    if (scope->declaration_ordinal_u16 >= declaration_count ||
        scope->entry_base_u32 != *inout_base ||
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
  iree_const_byte_span_t slice = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take(
      &cursor, sizeof(iree_vm_bytecode_v0_metadata_header_t),
      iree_alignof(iree_vm_bytecode_v0_metadata_header_t), &slice));
  const iree_vm_bytecode_v0_metadata_header_t* header =
      (const iree_vm_bytecode_v0_metadata_header_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_METADATA_HEADER, (const uint8_t*)header,
      layout));
  if (header->module_entry_count_u32 > header->total_entry_count_u32 ||
      header->import_scope_count_u32 > layout->imports.entry_count ||
      header->export_scope_count_u32 > layout->exports.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata header counts are invalid");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->import_scope_count_u32,
      sizeof(iree_vm_bytecode_v0_metadata_scope_row_t),
      iree_alignof(iree_vm_bytecode_v0_metadata_scope_row_t), &slice));
  const iree_vm_bytecode_v0_metadata_scope_row_t* import_scopes =
      (const iree_vm_bytecode_v0_metadata_scope_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->export_scope_count_u32,
      sizeof(iree_vm_bytecode_v0_metadata_scope_row_t),
      iree_alignof(iree_vm_bytecode_v0_metadata_scope_row_t), &slice));
  const iree_vm_bytecode_v0_metadata_scope_row_t* export_scopes =
      (const iree_vm_bytecode_v0_metadata_scope_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, header->total_entry_count_u32,
      sizeof(iree_vm_bytecode_v0_metadata_entry_row_t),
      iree_alignof(iree_vm_bytecode_v0_metadata_entry_row_t), &slice));
  const iree_vm_bytecode_v0_metadata_entry_row_t* entries =
      (const iree_vm_bytecode_v0_metadata_entry_row_t*)slice.data;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_cursor_align(&cursor, iree_alignof(uint64_t)));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_cursor_take_array(
      &cursor, (iree_host_size_t)header->total_entry_count_u32 + 1,
      sizeof(iree_vm_bytecode_v0_metadata_value_offset_t),
      iree_alignof(iree_vm_bytecode_v0_metadata_value_offset_t), &slice));
  const iree_vm_bytecode_v0_metadata_value_offset_t* value_offsets =
      (const iree_vm_bytecode_v0_metadata_value_offset_t*)slice.data;
  const uint8_t* value_data = span.data + cursor.offset;
  const iree_host_size_t value_data_length = span.data_length - cursor.offset;
  if (value_offsets[0].byte_offset_u64 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata value offsets must begin at zero");
  }
  for (uint32_t i = 0; i < header->total_entry_count_u32; ++i) {
    const uint64_t begin = value_offsets[i].byte_offset_u64;
    const uint64_t end = value_offsets[i + 1].byte_offset_u64;
    if (begin > end || end > value_data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Metadata value offsets are invalid");
    }
  }
  if (value_offsets[header->total_entry_count_u32].byte_offset_u64 !=
      value_data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metadata value offsets do not consume the byte tail");
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
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_entries(
      layout, 0, header->module_entry_count_u32));
  for (uint32_t i = 0; i < header->import_scope_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_entries(
        layout, import_scopes[i].entry_base_u32,
        import_scopes[i].entry_count_u16));
  }
  for (uint32_t i = 0; i < header->export_scope_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata_entries(
        layout, export_scopes[i].entry_base_u32,
        export_scopes[i].entry_count_u16));
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
    const iree_vm_bytecode_v0_callable_type_row_t* export_type =
        &layout->callable_types.rows[export_row->callable_type_ordinal_u16];
    const iree_vm_bytecode_v0_function_row_t* function =
        &layout->functions.rows[export_row->function_ordinal_u16];
    const iree_vm_bytecode_v0_callable_type_row_t* function_type =
        iree_vm_bytecode_function_callable_type(layout, function);
    if (export_type->signature_ordinal_u16 !=
            function_type->signature_ordinal_u16 ||
        (iree_any_bit_set(function->flags_u16,
                          IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD) &&
         !iree_any_bit_set(export_type->flags_u16,
                           IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "export callable contract does not match its function");
    }
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_module_structure(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan) {
  iree_vm_bytecode_section_map_t sections = {0};
  iree_vm_bytecode_module_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_envelope(contents, &sections));
  plan.layout.image.header = sections.header;
  plan.layout.image.sections = sections.rows;
  plan.layout.image.section_count = sections.count;
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
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_functions(
      sections.spans[IREE_VM_BYTECODE_SECTION_FUNCTIONS], &plan.layout,
      &plan.maximum_block_count));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_exports(
      sections.spans[IREE_VM_BYTECODE_SECTION_EXPORTS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_constants(
      sections.spans[IREE_VM_BYTECODE_SECTION_CONSTANTS], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_globals(
      sections.spans[IREE_VM_BYTECODE_SECTION_GLOBALS], &plan.layout,
      &plan.process_layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_rodata(
      sections.spans[IREE_VM_BYTECODE_SECTION_RODATA],
      sections.alignments[IREE_VM_BYTECODE_SECTION_RODATA], &plan.layout,
      &plan.rodata_storage));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_presentation(
      sections.spans[IREE_VM_BYTECODE_SECTION_PRESENTATION], &plan.layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata(
      sections.spans[IREE_VM_BYTECODE_SECTION_METADATA], &plan.layout));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_verify_exports_against_functions(&plan.layout));
  *out_plan = plan;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Instruction verification
//===----------------------------------------------------------------------===//

typedef struct iree_vm_bytecode_instruction_context_t {
  // Complete mapped module layout.
  const iree_vm_bytecode_module_layout_t* layout;
  // Current function declaration.
  const iree_vm_bytecode_v0_function_row_t* function;
  // Current function signature.
  const iree_vm_bytecode_v0_signature_row_t* signature;
  // Decoded control.block byte offsets in ascending order.
  const uint32_t* block_offsets;
  // Accumulated atomic-carrier requirements.
  iree_vm_bytecode_atomic_carrier_bits_t* required_atomic_carrier_bits;
} iree_vm_bytecode_instruction_context_t;

static bool iree_vm_bytecode_range_fits_u32(uint32_t base, uint32_t length,
                                            uint32_t limit) {
  return base <= limit && length <= limit - base;
}

static bool iree_vm_bytecode_range_fits_u64(uint64_t base, uint64_t length,
                                            uint64_t limit) {
  return base <= limit && length <= limit - base;
}

static bool iree_vm_bytecode_verify_global_ordinal(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t ordinal,
    uint32_t packed_extent) {
  if (!context->layout->globals.header) return false;
  const uint32_t upper_offset = packed_extent & 0xFFFFu;
  const uint32_t lower_offset_plus_one = packed_extent >> 16;
  const uint8_t* globals = (const uint8_t*)context->layout->globals.header;
  const uint32_t lower =
      lower_offset_plus_one == 0
          ? 0
          : iree_unaligned_load_le_u32(globals + lower_offset_plus_one - 1u);
  const uint32_t upper = iree_unaligned_load_le_u32(globals + upper_offset);
  return ordinal >= lower && ordinal < upper;
}

static bool iree_vm_bytecode_verify_local_memory_format_range(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t base,
    uint32_t format) {
  const uint32_t length =
      format < 16 ? (1u << (format >> 2)) * (1u << (format & 3u)) : 0;
  return length != 0 &&
         iree_vm_bytecode_range_fits_u32(
             base, length, context->function->local_byte_length_u16);
}

static bool iree_vm_bytecode_verify_optional_import(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t ordinal) {
  return ordinal < context->layout->imports.entry_count &&
         iree_any_bit_set(context->layout->imports.entries[ordinal].flags_u16,
                          IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL);
}

static bool iree_vm_bytecode_verify_rodata_offset(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t ordinal,
    uint32_t offset) {
  return ordinal < context->layout->rodata.count &&
         offset <= context->layout->rodata.descriptors[ordinal].byte_length_u64;
}

static bool iree_vm_bytecode_verify_rodata_static_offset(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t ordinal,
    uint32_t offset, uint32_t length) {
  return ordinal < context->layout->rodata.count &&
         iree_vm_bytecode_range_fits_u64(
             offset, length,
             context->layout->rodata.descriptors[ordinal].byte_length_u64);
}

static bool iree_vm_bytecode_verify_integer_bitstream_shape(
    uint32_t field_width, uint32_t source_count, uint32_t result_count,
    uint32_t source_width, uint32_t result_width, bool is_pack,
    uint32_t maximum_bits) {
  const bool source_width_valid = source_width == 8 || source_width == 16 ||
                                  source_width == 32 || source_width == 64;
  const bool result_width_valid = result_width == 8 || result_width == 16 ||
                                  result_width == 32 || result_width == 64;
  const uint32_t source_bits =
      source_count * (is_pack ? field_width : source_width);
  const uint32_t result_bits =
      result_count * (is_pack ? result_width : field_width);
  const uint32_t carrier_width = is_pack ? source_width : result_width;
  return source_count != 0 && result_count != 0 && field_width != 0 &&
         field_width <= carrier_width && source_width_valid &&
         result_width_valid && source_bits == result_bits &&
         source_bits <= maximum_bits;
}

static bool iree_vm_bytecode_verify_value_register_format_range(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t base,
    uint32_t format, uint32_t maximum_lane_count) {
  const uint32_t lane_count = format < 16 ? 1u << (format & 3u) : 0;
  return lane_count != 0 && lane_count <= maximum_lane_count &&
         iree_vm_bytecode_range_fits_u32(
             base, lane_count, context->function->value_register_count_u16);
}

static bool iree_vm_bytecode_function_has_block_at(
    const iree_vm_bytecode_instruction_context_t* context,
    uint32_t target_offset) {
  uint32_t lower = 0;
  uint32_t upper = context->function->block_count_u32;
  while (lower < upper) {
    const uint32_t middle = lower + (upper - lower) / 2u;
    const uint32_t block_offset = context->block_offsets[middle];
    if (target_offset < block_offset) {
      upper = middle;
    } else if (target_offset > block_offset) {
      lower = middle + 1u;
    } else {
      return true;
    }
  }
  return false;
}

static bool iree_vm_bytecode_verify_control_target(
    const iree_vm_bytecode_instruction_context_t* context,
    uint32_t record_offset, uint8_t record_length, int32_t target_word_delta) {
  const int64_t target_offset =
      (int64_t)record_offset + record_length + (int64_t)target_word_delta * 4;
  return target_offset >= 0 &&
         target_offset < context->function->bytecode_length_u32 &&
         (target_offset & 3) == 0 &&
         iree_vm_bytecode_function_has_block_at(context,
                                                (uint32_t)target_offset);
}

static const iree_vm_bytecode_v0_callable_type_row_t*
iree_vm_bytecode_callable_type_at(
    const iree_vm_bytecode_instruction_context_t* context, uint32_t ordinal) {
  return ordinal < context->layout->callable_types.count
             ? &context->layout->callable_types.rows[ordinal]
             : NULL;
}

static uint16_t iree_vm_bytecode_direct_count(uint16_t argument_count,
                                              uint16_t result_count) {
  return iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                  iree_max(argument_count, result_count));
}

static uint32_t iree_vm_bytecode_overflow_count(uint16_t count) {
  return count > IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? (uint32_t)count - IREE_VM_CALL_DIRECT_REGISTER_COUNT
             : 0;
}

static bool iree_vm_bytecode_verify_call_packet(
    const iree_vm_bytecode_instruction_context_t* context,
    const iree_vm_bytecode_v0_callable_type_row_t* callable_type,
    uint16_t direct_ref_move_mask) {
  if (!callable_type) return false;
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &context->layout->signatures.rows[callable_type->signature_ordinal_u16];
  if (context->function->value_register_count_u16 <
          iree_vm_bytecode_direct_count(signature->argument_value_count_u16,
                                        signature->result_value_count_u16) ||
      context->function->ref_register_count_u16 <
          iree_vm_bytecode_direct_count(signature->argument_ref_count_u16,
                                        signature->result_ref_count_u16) ||
      context->function->function_register_count_u16 <
          iree_vm_bytecode_direct_count(signature->argument_function_count_u16,
                                        signature->result_function_count_u16)) {
    return false;
  }

  const uint16_t direct_ref_argument_count = iree_min(
      IREE_VM_CALL_DIRECT_REGISTER_COUNT, signature->argument_ref_count_u16);
  const uint16_t valid_ref_move_mask =
      direct_ref_argument_count == IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? UINT16_MAX
          : (uint16_t)((1u << direct_ref_argument_count) - 1u);
  if ((direct_ref_move_mask & ~valid_ref_move_mask) != 0) return false;

  const uint32_t required_local_bytes =
      sizeof(uint64_t) *
      (iree_vm_bytecode_overflow_count(signature->argument_value_count_u16) +
       iree_vm_bytecode_overflow_count(signature->result_value_count_u16));
  if (required_local_bytes > context->function->local_byte_length_u16) {
    return false;
  }

  const uint32_t required_local_refs =
      iree_vm_bytecode_overflow_count(signature->argument_ref_count_u16) +
      iree_vm_bytecode_overflow_count(signature->result_ref_count_u16);
  if (required_local_refs > context->function->local_ref_count_u32) {
    return false;
  }

  const uint32_t required_local_functions =
      iree_vm_bytecode_overflow_count(signature->argument_function_count_u16) +
      iree_vm_bytecode_overflow_count(signature->result_function_count_u16);
  return required_local_functions <=
         context->function->local_function_count_u32;
}

static const iree_vm_bytecode_v0_callable_type_row_t*
iree_vm_bytecode_direct_target_callable(
    const iree_vm_bytecode_instruction_context_t* context, uint8_t target_kind,
    uint16_t target_ordinal) {
  if (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
    if (target_ordinal >= context->layout->functions.count) return NULL;
    return iree_vm_bytecode_function_callable_type(
        context->layout, &context->layout->functions.rows[target_ordinal]);
  }
  if (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_REQUIRED_IMPORT ||
      target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_OPTIONAL_IMPORT) {
    if (target_ordinal >= context->layout->imports.entry_count) return NULL;
    const iree_vm_bytecode_v0_import_entry_row_t* import =
        &context->layout->imports.entries[target_ordinal];
    const bool is_optional = iree_any_bit_set(
        import->flags_u16, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL);
    if (is_optional !=
        (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_OPTIONAL_IMPORT)) {
      return NULL;
    }
    return &context->layout->callable_types
                .rows[import->callable_type_ordinal_u16];
  }
  return NULL;
}

static bool iree_vm_bytecode_callable_type_is_compatible(
    const iree_vm_bytecode_v0_callable_type_row_t* source,
    bool source_may_yield,
    const iree_vm_bytecode_v0_callable_type_row_t* destination) {
  return source->signature_ordinal_u16 == destination->signature_ordinal_u16 &&
         (!source_may_yield ||
          iree_any_bit_set(destination->flags_u16,
                           IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD));
}

static bool iree_vm_bytecode_verify_function_address(
    const iree_vm_bytecode_instruction_context_t* context, uint8_t target_kind,
    uint16_t target_ordinal, uint16_t callable_type_ordinal) {
  const iree_vm_bytecode_v0_callable_type_row_t* destination =
      iree_vm_bytecode_callable_type_at(context, callable_type_ordinal);
  if (!destination) return false;

  if (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
    if (target_ordinal >= context->layout->functions.count) return false;
    const iree_vm_bytecode_v0_function_row_t* target =
        &context->layout->functions.rows[target_ordinal];
    const iree_vm_bytecode_v0_callable_type_row_t* source =
        iree_vm_bytecode_function_callable_type(context->layout, target);
    return iree_vm_bytecode_callable_type_is_compatible(
        source,
        iree_any_bit_set(target->flags_u16,
                         IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD),
        destination);
  }
  if (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_REQUIRED_IMPORT ||
      target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_OPTIONAL_IMPORT) {
    if (target_ordinal >= context->layout->imports.entry_count) return false;
    const iree_vm_bytecode_v0_import_entry_row_t* import =
        &context->layout->imports.entries[target_ordinal];
    const bool is_optional = iree_any_bit_set(
        import->flags_u16, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL);
    if (is_optional !=
        (target_kind == IREE_VM_BYTECODE_CONTROL_CALL_TARGET_OPTIONAL_IMPORT)) {
      return false;
    }
    const iree_vm_bytecode_v0_callable_type_row_t* source =
        &context->layout->callable_types
             .rows[import->callable_type_ordinal_u16];
    return iree_vm_bytecode_callable_type_is_compatible(
        source,
        iree_any_bit_set(source->flags_u16,
                         IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD),
        destination);
  }
  return false;
}

static bool iree_vm_bytecode_verify_direct_call(
    const iree_vm_bytecode_instruction_context_t* context, uint8_t target_kind,
    uint16_t target_ordinal, uint16_t direct_ref_move_mask) {
  return iree_vm_bytecode_verify_call_packet(
      context,
      iree_vm_bytecode_direct_target_callable(context, target_kind,
                                              target_ordinal),
      direct_ref_move_mask);
}

static bool iree_vm_bytecode_verify_indirect_call(
    const iree_vm_bytecode_instruction_context_t* context,
    uint16_t callable_type_ordinal, uint16_t direct_ref_move_mask) {
  return iree_vm_bytecode_verify_call_packet(
      context,
      iree_vm_bytecode_callable_type_at(context, callable_type_ordinal),
      direct_ref_move_mask);
}

static bool iree_vm_bytecode_verify_instruction(
    const uint8_t* record, uint32_t record_offset, uint8_t record_length,
    uint32_t descriptor,
    const iree_vm_bytecode_instruction_context_t* context) {
  switch (iree_vm_bytecode_instruction_verification_shape_ordinal(descriptor)) {
#include "iree/vm/bytecode/verifier/instruction_cases.inl"
    default:
      IREE_ASSERT_UNREACHABLE("generated instruction verifier is incomplete");
      return false;
  }
}

static bool iree_vm_bytecode_control_flow_is_terminal(
    iree_vm_bytecode_control_flow_t control_flow) {
  return control_flow == IREE_VM_BYTECODE_CONTROL_FLOW_RETURN ||
         control_flow == IREE_VM_BYTECODE_CONTROL_FLOW_YIELD ||
         control_flow == IREE_VM_BYTECODE_CONTROL_FLOW_BRANCH ||
         control_flow == IREE_VM_BYTECODE_CONTROL_FLOW_FAIL;
}

static iree_status_t iree_vm_bytecode_verify_function_instructions(
    const iree_vm_bytecode_module_layout_t* layout,
    const iree_vm_bytecode_v0_function_row_t* function,
    uint32_t function_ordinal, uint32_t* block_offsets,
    iree_vm_bytecode_atomic_carrier_bits_t* required_atomic_carrier_bits) {
  const iree_const_byte_span_t bytecode =
      iree_vm_bytecode_function_data(&layout->functions, function);
  uint32_t block_count = 0;
  uint32_t record_offset = 0;
  iree_vm_bytecode_control_flow_t final_control_flow =
      IREE_VM_BYTECODE_CONTROL_FLOW_INVALID;
  bool has_call = false;
  while (record_offset < bytecode.data_length) {
    const uint8_t opcode = bytecode.data[record_offset];
    const uint32_t descriptor =
        iree_vm_bytecode_instruction_verification[opcode];
    if (descriptor == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32
                              " has unknown opcode 0x%02" PRIx8
                              " at byte %" PRIu32,
                              function_ordinal, opcode, record_offset);
    }
    const uint8_t record_length =
        iree_vm_bytecode_verification_byte_length(descriptor);
    if (record_length > bytecode.data_length - record_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32 " opcode 0x%02" PRIx8
                              " at byte %" PRIu32 " is truncated",
                              function_ordinal, opcode, record_offset);
    }
    const iree_vm_bytecode_control_flow_t control_flow =
        iree_vm_bytecode_instruction_verification_control_flow(descriptor);
    if (record_offset == 0 &&
        control_flow != IREE_VM_BYTECODE_CONTROL_FLOW_BLOCK) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32
                              " does not begin with control.block",
                              function_ordinal);
    }
    if (control_flow == IREE_VM_BYTECODE_CONTROL_FLOW_BLOCK) {
      if (block_count >= function->block_count_u32) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "function %" PRIu32
                                " has more control.block records than declared",
                                function_ordinal);
      }
      block_offsets[block_count++] = record_offset;
    }
    has_call |= control_flow == IREE_VM_BYTECODE_CONTROL_FLOW_CALL;
    final_control_flow = control_flow;
    record_offset += record_length;
  }

  if (block_count != function->block_count_u32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function %" PRIu32
        " control.block count does not match its declaration",
        function_ordinal);
  }
  if (has_call != iree_any_bit_set(function->flags_u16,
                                   IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function %" PRIu32
                            " call summary does not match its records",
                            function_ordinal);
  }
  if (!iree_vm_bytecode_control_flow_is_terminal(final_control_flow)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function %" PRIu32
                            " falls through past its final record",
                            function_ordinal);
  }

  const iree_vm_bytecode_instruction_context_t context = {
      .layout = layout,
      .function = function,
      .signature = iree_vm_bytecode_function_signature(layout, function),
      .block_offsets = block_offsets,
      .required_atomic_carrier_bits = required_atomic_carrier_bits,
  };
  record_offset = 0;
  while (record_offset < bytecode.data_length) {
    const uint8_t* record = bytecode.data + record_offset;
    const uint32_t descriptor =
        iree_vm_bytecode_instruction_verification[record[0]];
    const uint8_t record_length =
        iree_vm_bytecode_verification_byte_length(descriptor);
    if (!iree_vm_bytecode_verify_instruction(
            record, record_offset, record_length, descriptor, &context)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32 " opcode 0x%02" PRIx8
                              " at byte %" PRIu32
                              " violates instruction constraints",
                              function_ordinal, record[0], record_offset);
    }
    record_offset += record_length;
  }

  const iree_vm_bytecode_v0_switch_target_entry_t* switch_targets =
      iree_vm_bytecode_function_switch_targets(&layout->functions, function);
  for (uint32_t i = 0; i < function->switch_target_entry_count_u32; ++i) {
    const uint32_t word_offset = switch_targets[i].target_word_offset_u32;
    if (word_offset > UINT32_MAX / 4u ||
        !iree_vm_bytecode_function_has_block_at(&context, word_offset * 4u)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32 " switch target %" PRIu32
                              " does not name a control.block",
                              function_ordinal, i);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_module_instructions(
    iree_vm_bytecode_module_plan_t* plan, uint32_t* block_offsets) {
  iree_vm_bytecode_atomic_carrier_bits_t required_atomic_carrier_bits = 0;
  for (uint32_t i = 0; i < plan->layout.functions.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_instructions(
        &plan->layout, &plan->layout.functions.rows[i], i, block_offsets,
        &required_atomic_carrier_bits));
  }
  plan->required_atomic_carrier_bits = required_atomic_carrier_bits;
  return iree_ok_status();
}
