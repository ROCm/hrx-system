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

static bool iree_vm_bytecode_bytes_are_zero(const uint8_t* data,
                                            iree_host_size_t length) {
  for (iree_host_size_t i = 0; i < length; ++i) {
    if (data[i] != 0) return false;
  }
  return true;
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

static iree_status_t iree_vm_bytecode_verify_requirements(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (layout->requirements.count == 0) {
    if (layout->image.extension_pages != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "extension sections require a Requirements section");
    }
    return iree_ok_status();
  }
  const iree_vm_bytecode_v0_requirement_row_t* rows = layout->requirements.rows;
  iree_vm_bytecode_module_layout_t empty_layout = {0};
  uint16_t previous_page = 0;
  uint16_t declared_pages = 0;
  for (uint16_t i = 0; i < layout->requirements.count; ++i) {
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
  if ((layout->image.extension_pages & ~declared_pages) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "extension section has no Requirements declaration");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_strings(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->strings.offsets) return iree_ok_status();
  const iree_vm_bytecode_v0_strings_header_t* header =
      (const iree_vm_bytecode_v0_strings_header_t*)((const uint8_t*)layout
                                                        ->strings.offsets -
                                                    sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_STRINGS_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_string_offset_t* offsets = layout->strings.offsets;
  const uint32_t data_length = layout->strings.data_length;
  if (offsets[0].byte_offset_u32 != 0 ||
      offsets[layout->strings.count].byte_offset_u32 != data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strings offsets do not cover the byte tail");
  }
  for (uint32_t i = 0; i < layout->strings.count; ++i) {
    const uint32_t begin = offsets[i].byte_offset_u32;
    const uint32_t end = offsets[i + 1].byte_offset_u32;
    if (begin > end || end > data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Strings offsets are not monotonic");
    }
    const iree_string_view_t value = iree_make_string_view(
        (const char*)layout->strings.data + begin, end - begin);
    if (iree_string_view_find_char(value, '\0', 0) != IREE_STRING_VIEW_NPOS ||
        !iree_unicode_utf8_validate(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "string %" PRIu32 " is not NUL-free valid UTF-8",
                              i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_ref_types(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->ref_types.groups) return iree_ok_status();
  const iree_vm_bytecode_v0_ref_types_header_t* header =
      (const iree_vm_bytecode_v0_ref_types_header_t*)((const uint8_t*)layout
                                                          ->ref_types.groups -
                                                      sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPES_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_ref_type_group_row_t* groups =
      layout->ref_types.groups;
  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < layout->ref_types.group_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_GROUP_ROW,
        (const uint8_t*)&groups[i], layout));
    if (groups[i].entry_count_u32 > 65536u - entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RefTypes entry count exceeds 65536");
    }
    entry_count += groups[i].entry_count_u32;
  }
  if (entry_count != layout->ref_types.entry_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RefTypes groups do not partition the entry table");
  }

  uint32_t entry_base = 0;
  iree_string_view_t previous_namespace = iree_string_view_empty();
  for (uint32_t group_i = 0; group_i < layout->ref_types.group_count;
       ++group_i) {
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
          &layout->ref_types.entries[entry_base + entry_i];
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
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->signatures.rows) return iree_ok_status();
  const iree_vm_bytecode_v0_signatures_header_t* header =
      (const iree_vm_bytecode_v0_signatures_header_t*)((const uint8_t*)layout
                                                           ->signatures.rows -
                                                       sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURES_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_signature_row_t* rows = layout->signatures.rows;
  uint32_t descriptor_count = 0;
  for (uint32_t i = 0; i < layout->signatures.count; ++i) {
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
  if (descriptor_count != layout->signatures.descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Signatures do not partition the descriptor table");
  }
  for (uint32_t i = 0; i < layout->signatures.count; ++i) {
    const iree_vm_bytecode_v0_signature_row_t* row = &rows[i];
    const uint32_t argument_count =
        iree_vm_bytecode_signature_argument_count(row);
    const iree_vm_bytecode_v0_signature_descriptor_row_t* signature =
        layout->signatures.descriptors + row->descriptor_base_u32;
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
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->callable_types.rows) {
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
  const iree_vm_bytecode_v0_callable_types_header_t* header =
      (const iree_vm_bytecode_v0_callable_types_header_t*)((const uint8_t*)layout
                                                               ->callable_types
                                                               .rows -
                                                           sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_CALLABLE_TYPES_HEADER,
      (const uint8_t*)header, layout));
  const iree_vm_bytecode_v0_callable_type_row_t* rows =
      layout->callable_types.rows;
  for (uint32_t i = 0; i < layout->callable_types.count; ++i) {
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
        descriptor->type_ordinal_u16 >= layout->callable_types.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "signature callable-type ordinal is out of range");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_imports(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->imports.groups) return iree_ok_status();
  const iree_vm_bytecode_v0_imports_header_t* header =
      (const iree_vm_bytecode_v0_imports_header_t*)((const uint8_t*)
                                                        layout->imports.groups -
                                                    sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_IMPORTS_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_import_group_row_t* groups = layout->imports.groups;
  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < layout->imports.group_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_GROUP_ROW,
        (const uint8_t*)&groups[i], layout));
    if (groups[i].entry_count_u32 > 65536u - entry_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Imports entry count exceeds 65536");
    }
    entry_count += groups[i].entry_count_u32;
  }
  if (entry_count != layout->imports.entry_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Imports groups do not partition the entry table");
  }

  uint32_t entry_base = 0;
  iree_string_view_t previous_module = iree_string_view_empty();
  for (uint32_t group_i = 0; group_i < layout->imports.group_count; ++group_i) {
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
          &layout->imports.entries[entry_base + entry_i];
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
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_exports(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->exports.rows) return iree_ok_status();
  const iree_vm_bytecode_v0_exports_header_t* header =
      (const iree_vm_bytecode_v0_exports_header_t*)((const uint8_t*)
                                                        layout->exports.rows -
                                                    sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_EXPORTS_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_export_row_t* rows = layout->exports.rows;
  iree_string_view_t previous_name = iree_string_view_empty();
  for (uint32_t i = 0; i < layout->exports.count; ++i) {
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
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->functions.rows) return iree_ok_status();
  const iree_vm_bytecode_v0_functions_header_t* header =
      (const iree_vm_bytecode_v0_functions_header_t*)((const uint8_t*)layout
                                                          ->functions.rows -
                                                      sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_FUNCTIONS_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_function_row_t* rows = layout->functions.rows;

  uint32_t switch_target_count = 0;
  uint32_t bytecode_length = 0;
  uint32_t observed_maximum_block_count = 0;
  for (uint32_t i = 0; i < layout->functions.count; ++i) {
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
    observed_maximum_block_count =
        iree_max(observed_maximum_block_count, row->block_count_u32);
  }
  if (header->maximum_block_count_u32 != observed_maximum_block_count ||
      switch_target_count != layout->functions.switch_target_count ||
      bytecode_length != layout->functions.bytecode_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Functions header or tail extents do not match its rows");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_constants(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (layout->constants.count > 65536u) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Constants count exceeds 65536");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_globals(
    const iree_vm_bytecode_module_layout_t* layout) {
  const iree_vm_bytecode_v0_globals_header_t* header = layout->globals.header;
  if (!header) return iree_ok_status();
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
  for (uint32_t i = 0; i < header->ref_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_GLOBAL_REF_DESCRIPTOR_ROW,
        (const uint8_t*)&layout->globals.refs[i], layout));
  }
  for (uint32_t i = 0; i < header->function_count_u32; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_GLOBAL_FUNCTION_DESCRIPTOR_ROW,
        (const uint8_t*)&layout->globals.functions[i], layout));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_verify_rodata(
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->rodata.descriptors) return iree_ok_status();
  const iree_vm_bytecode_v0_rodata_header_t* header =
      (const iree_vm_bytecode_v0_rodata_header_t*)layout->rodata.section_begin;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_RODATA_HEADER, (const uint8_t*)header,
      layout));
  const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptors =
      layout->rodata.descriptors;
  for (uint32_t i = 0; i < layout->rodata.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
        IREE_VM_BYTECODE_MODULE_RECORD_RODATA_BLOCK_DESCRIPTOR,
        (const uint8_t*)&descriptors[i], layout));
  }
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
    const iree_vm_bytecode_module_layout_t* layout) {
  if (!layout->presentation.entries) return iree_ok_status();
  const iree_vm_bytecode_v0_presentation_header_t* header =
      (const iree_vm_bytecode_v0_presentation_header_t*)((const uint8_t*)layout
                                                             ->presentation
                                                             .entries -
                                                         sizeof(*header));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_HEADER,
      (const uint8_t*)header, layout));
  const iree_vm_bytecode_v0_presentation_entry_row_t* entries =
      layout->presentation.entries;

  uint32_t field_count = 0;
  uint16_t previous_kind = 0;
  uint16_t previous_ordinal = 0;
  for (uint32_t i = 0; i < layout->presentation.entry_count; ++i) {
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
  if (field_count != layout->presentation.field_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Presentation entries do not partition the field table");
  }
  const iree_vm_bytecode_v0_presentation_field_row_t* fields =
      layout->presentation.fields;

  for (uint32_t i = 0; i < layout->presentation.entry_count; ++i) {
    const iree_vm_bytecode_v0_presentation_entry_row_t* entry = &entries[i];
    const uint32_t field_end = i + 1 < layout->presentation.entry_count
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
    const iree_vm_bytecode_module_layout_t* layout) {
  const iree_vm_bytecode_v0_metadata_header_t* header = layout->metadata.header;
  if (!header) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_record(
      IREE_VM_BYTECODE_MODULE_RECORD_METADATA_HEADER, (const uint8_t*)header,
      layout));
  if (header->module_entry_count_u32 > header->total_entry_count_u32 ||
      header->import_scope_count_u32 > layout->imports.entry_count ||
      header->export_scope_count_u32 > layout->exports.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metadata header counts are invalid");
  }
  const iree_vm_bytecode_v0_metadata_scope_row_t* import_scopes =
      layout->metadata.import_scopes;
  const iree_vm_bytecode_v0_metadata_scope_row_t* export_scopes =
      layout->metadata.export_scopes;
  const iree_vm_bytecode_v0_metadata_value_offset_t* value_offsets =
      layout->metadata.value_offsets;
  const iree_host_size_t value_data_length = layout->metadata.value_data_length;
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

iree_status_t iree_vm_bytecode_verify_module_layout(
    const iree_vm_bytecode_module_plan_t* plan) {
  const iree_vm_bytecode_module_layout_t* layout = &plan->layout;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_requirements(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_strings(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_ref_types(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_signatures(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_callable_types(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_imports(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_functions(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_exports(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_constants(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_globals(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_rodata(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_presentation(layout));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_metadata(layout));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_verify_exports_against_functions(layout));
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_module_structure(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan) {
  iree_vm_bytecode_module_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_plan_build(contents, &plan));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_module_layout(&plan));
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
    const iree_vm_bytecode_instruction_context_t* context) {
  switch (record[0]) {
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
    uint32_t function_ordinal, uint32_t* block_offsets) {
  const iree_const_byte_span_t bytecode =
      iree_vm_bytecode_function_data(&layout->functions, function);
  uint32_t block_count = 0;
  uint32_t record_offset = 0;
  iree_vm_bytecode_control_flow_t final_control_flow =
      IREE_VM_BYTECODE_CONTROL_FLOW_INVALID;
  bool has_call = false;
  while (record_offset < bytecode.data_length) {
    const uint8_t opcode = bytecode.data[record_offset];
    const uint16_t descriptor =
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
  };
  record_offset = 0;
  while (record_offset < bytecode.data_length) {
    const uint8_t* record = bytecode.data + record_offset;
    const uint16_t descriptor =
        iree_vm_bytecode_instruction_verification[record[0]];
    const uint8_t record_length =
        iree_vm_bytecode_verification_byte_length(descriptor);
    if (!iree_vm_bytecode_verify_instruction(record, record_offset,
                                             record_length, &context)) {
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
    const iree_vm_bytecode_module_plan_t* plan, uint32_t* block_offsets) {
  for (uint32_t i = 0; i < plan->layout.functions.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_function_instructions(
        &plan->layout, &plan->layout.functions.rows[i], i, block_offsets));
  }
  return iree_ok_status();
}
