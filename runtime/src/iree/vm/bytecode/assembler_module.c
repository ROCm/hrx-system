// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler_module.h"

#include <string.h>

#include "iree/vm/execution.h"

static iree_status_t iree_vm_bytecode_assembler_module_error(
    const char* message) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "VM assembly: %s",
                          message);
}

static iree_status_t iree_vm_bytecode_assembler_module_read(
    const iree_vm_bytecode_assembler_module_view_t* view, uint64_t offset,
    iree_host_size_t length, void* buffer) {
  IREE_RETURN_IF_ERROR(iree_io_stream_seek(
      view->stream, IREE_IO_STREAM_SEEK_SET, (iree_io_stream_pos_t)offset));
  return iree_io_stream_read(view->stream, length, buffer, NULL);
}

static iree_status_t iree_vm_bytecode_assembler_module_read_record(
    const iree_vm_bytecode_assembler_module_view_t* view,
    iree_vm_bytecode_module_record_ordinal_t record_ordinal,
    uint32_t row_ordinal, iree_host_size_t row_length, void* out_row) {
  const uint64_t offset =
      view->record_offsets[record_ordinal] + (uint64_t)row_ordinal * row_length;
  return iree_vm_bytecode_assembler_module_read(view, offset, row_length,
                                                out_row);
}

#define IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(view, record, ordinal, out_row) \
  iree_vm_bytecode_assembler_module_read_record(view, record, ordinal,         \
                                                sizeof(*(out_row)), out_row)

static iree_status_t iree_vm_bytecode_assembler_module_string_range(
    const iree_vm_bytecode_assembler_module_view_t* view, uint16_t ordinal,
    uint32_t* out_begin, uint32_t* out_end) {
  iree_vm_bytecode_v0_string_offset_t offsets[2];
  IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
      view, IREE_VM_BYTECODE_MODULE_RECORD_STRING_OFFSET, ordinal,
      &offsets[0]));
  IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
      view, IREE_VM_BYTECODE_MODULE_RECORD_STRING_OFFSET, ordinal + 1,
      &offsets[1]));
  *out_begin = offsets[0].byte_offset_u32;
  *out_end = offsets[1].byte_offset_u32;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_require_nonempty_string(
    const iree_vm_bytecode_assembler_module_view_t* view, uint16_t ordinal) {
  uint32_t begin = 0;
  uint32_t end = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_string_range(
      view, ordinal, &begin, &end));
  return begin != end ? iree_ok_status()
                      : iree_vm_bytecode_assembler_module_error(
                            "an identity string is empty");
}

static iree_status_t iree_vm_bytecode_assembler_module_compare_strings(
    const iree_vm_bytecode_assembler_module_view_t* view, uint16_t lhs_ordinal,
    uint16_t rhs_ordinal, int* out_comparison) {
  uint32_t lhs_begin = 0;
  uint32_t lhs_end = 0;
  uint32_t rhs_begin = 0;
  uint32_t rhs_end = 0;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_string_range(
      view, lhs_ordinal, &lhs_begin, &lhs_end));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_string_range(
      view, rhs_ordinal, &rhs_begin, &rhs_end));

  const uint32_t lhs_length = lhs_end - lhs_begin;
  const uint32_t rhs_length = rhs_end - rhs_begin;
  const uint32_t common_length = iree_min(lhs_length, rhs_length);
  uint8_t lhs_buffer[64];
  uint8_t rhs_buffer[64];
  uint32_t compared = 0;
  while (compared < common_length) {
    const iree_host_size_t fragment_length =
        iree_min((uint32_t)sizeof(lhs_buffer), common_length - compared);
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read(
        view, view->string_data_offset + lhs_begin + compared, fragment_length,
        lhs_buffer));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read(
        view, view->string_data_offset + rhs_begin + compared, fragment_length,
        rhs_buffer));
    const int comparison = memcmp(lhs_buffer, rhs_buffer, fragment_length);
    if (comparison != 0) {
      *out_comparison = comparison;
      return iree_ok_status();
    }
    compared += (uint32_t)fragment_length;
  }
  *out_comparison = lhs_length < rhs_length ? -1 : lhs_length > rhs_length;
  return iree_ok_status();
}

static int iree_vm_bytecode_assembler_module_compare_u32(uint32_t lhs,
                                                         uint32_t rhs) {
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static uint32_t iree_vm_bytecode_assembler_module_signature_argument_count(
    const iree_vm_bytecode_v0_signature_row_t* signature) {
  return (uint32_t)signature->argument_value_count_u16 +
         signature->argument_ref_count_u16 +
         signature->argument_function_count_u16;
}

static uint32_t iree_vm_bytecode_assembler_module_signature_result_count(
    const iree_vm_bytecode_v0_signature_row_t* signature) {
  return (uint32_t)signature->result_value_count_u16 +
         signature->result_ref_count_u16 + signature->result_function_count_u16;
}

static iree_status_t iree_vm_bytecode_assembler_module_read_callable(
    const iree_vm_bytecode_assembler_module_view_t* view, uint16_t ordinal,
    iree_vm_bytecode_v0_callable_type_row_t* out_callable) {
  return IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
      view, IREE_VM_BYTECODE_MODULE_RECORD_CALLABLE_TYPE_ROW, ordinal,
      out_callable);
}

static iree_status_t iree_vm_bytecode_assembler_module_read_signature(
    const iree_vm_bytecode_assembler_module_view_t* view, uint16_t ordinal,
    iree_vm_bytecode_v0_signature_row_t* out_signature) {
  return IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
      view, IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_ROW, ordinal,
      out_signature);
}

static iree_status_t
iree_vm_bytecode_assembler_module_compare_signature_descriptors(
    const iree_vm_bytecode_assembler_module_view_t* view, uint32_t lhs_base,
    uint32_t rhs_base, uint32_t count, int* out_comparison) {
  for (uint32_t i = 0; i < count; ++i) {
    iree_vm_bytecode_v0_signature_descriptor_row_t lhs;
    iree_vm_bytecode_v0_signature_descriptor_row_t rhs;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_DESCRIPTOR_ROW,
        lhs_base + i, &lhs));
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_DESCRIPTOR_ROW,
        rhs_base + i, &rhs));
    int comparison = iree_vm_bytecode_assembler_module_compare_u32(
        lhs.kind_u16, rhs.kind_u16);
    if (comparison == 0) {
      comparison = iree_vm_bytecode_assembler_module_compare_u32(
          lhs.type_ordinal_u16, rhs.type_ordinal_u16);
    }
    if (comparison != 0) {
      *out_comparison = comparison;
      return iree_ok_status();
    }
  }
  *out_comparison = 0;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_compare_callables(
    const iree_vm_bytecode_assembler_module_view_t* view,
    const iree_vm_bytecode_v0_callable_type_row_t* lhs,
    const iree_vm_bytecode_v0_callable_type_row_t* rhs,
    bool* out_same_signature, int* out_comparison) {
  *out_same_signature = false;
  int comparison = iree_vm_bytecode_assembler_module_compare_u32(
      lhs->nesting_depth_u16, rhs->nesting_depth_u16);
  if (comparison != 0) {
    *out_comparison = comparison;
    return iree_ok_status();
  }

  iree_vm_bytecode_v0_signature_row_t lhs_signature;
  iree_vm_bytecode_v0_signature_row_t rhs_signature;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_signature(
      view, lhs->signature_ordinal_u16, &lhs_signature));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_signature(
      view, rhs->signature_ordinal_u16, &rhs_signature));
  const uint32_t lhs_argument_count =
      iree_vm_bytecode_assembler_module_signature_argument_count(
          &lhs_signature);
  const uint32_t rhs_argument_count =
      iree_vm_bytecode_assembler_module_signature_argument_count(
          &rhs_signature);
  comparison = iree_vm_bytecode_assembler_module_compare_u32(
      lhs_argument_count, rhs_argument_count);
  if (comparison == 0) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_compare_signature_descriptors(
            view, lhs_signature.descriptor_base_u32,
            rhs_signature.descriptor_base_u32, lhs_argument_count,
            &comparison));
  }
  const uint32_t lhs_result_count =
      iree_vm_bytecode_assembler_module_signature_result_count(&lhs_signature);
  const uint32_t rhs_result_count =
      iree_vm_bytecode_assembler_module_signature_result_count(&rhs_signature);
  if (comparison == 0) {
    comparison = iree_vm_bytecode_assembler_module_compare_u32(
        lhs_result_count, rhs_result_count);
  }
  if (comparison == 0) {
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_compare_signature_descriptors(
            view, lhs_signature.descriptor_base_u32 + lhs_argument_count,
            rhs_signature.descriptor_base_u32 + rhs_argument_count,
            lhs_result_count, &comparison));
  }
  if (comparison == 0) {
    *out_same_signature = true;
    comparison = iree_vm_bytecode_assembler_module_compare_u32(lhs->flags_u16,
                                                               rhs->flags_u16);
  }
  *out_comparison = comparison;
  return iree_ok_status();
}

static bool iree_vm_bytecode_assembler_module_callable_is_compatible(
    const iree_vm_bytecode_v0_callable_type_row_t* source,
    bool source_may_yield,
    const iree_vm_bytecode_v0_callable_type_row_t* destination) {
  return source->signature_ordinal_u16 == destination->signature_ordinal_u16 &&
         (!source_may_yield ||
          iree_any_bit_set(destination->flags_u16,
                           IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD));
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_requirements(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  uint16_t previous_page = 0;
  const uint32_t count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_REQUIREMENT_ROW];
  for (uint32_t i = 0; i < count; ++i) {
    iree_vm_bytecode_v0_requirement_row_t row;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_REQUIREMENT_ROW, i, &row));
    if (i != 0 && row.page_id_u16 <= previous_page) {
      return iree_vm_bytecode_assembler_module_error(
          "requirement pages are not strictly ordered");
    }
    previous_page = row.page_id_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_ref_types(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  uint32_t entry_base = 0;
  uint16_t previous_namespace = 0;
  const uint32_t group_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_GROUP_ROW];
  for (uint32_t group_i = 0; group_i < group_count; ++group_i) {
    iree_vm_bytecode_v0_ref_type_group_row_t group;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_GROUP_ROW, group_i,
        &group));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_require_nonempty_string(
            view, group.namespace_string_u16));
    if (group_i != 0) {
      int comparison = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_strings(
          view, previous_namespace, group.namespace_string_u16, &comparison));
      if (comparison >= 0) {
        return iree_vm_bytecode_assembler_module_error(
            "reference type namespaces are not strictly ordered");
      }
    }

    uint16_t previous_name = 0;
    for (uint32_t entry_i = 0; entry_i < group.entry_count_u32; ++entry_i) {
      iree_vm_bytecode_v0_ref_type_entry_row_t entry;
      IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_ENTRY_ROW,
          entry_base + entry_i, &entry));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_module_require_nonempty_string(
              view, entry.type_name_string_u16));
      if (entry_i != 0) {
        int comparison = 0;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_strings(
            view, previous_name, entry.type_name_string_u16, &comparison));
        if (comparison >= 0) {
          return iree_vm_bytecode_assembler_module_error(
              "reference type names are not strictly ordered");
        }
      }
      previous_name = entry.type_name_string_u16;
    }
    entry_base += group.entry_count_u32;
    previous_namespace = group.namespace_string_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_callables(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  const uint32_t callable_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_CALLABLE_TYPE_ROW];
  iree_vm_bytecode_v0_callable_type_row_t previous = {0};
  for (uint32_t i = 0; i < callable_count; ++i) {
    iree_vm_bytecode_v0_callable_type_row_t callable;
    iree_vm_bytecode_v0_signature_row_t signature;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_callable(
        view, (uint16_t)i, &callable));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_signature(
        view, callable.signature_ordinal_u16, &signature));
    const uint32_t descriptor_count =
        iree_vm_bytecode_assembler_module_signature_argument_count(&signature) +
        iree_vm_bytecode_assembler_module_signature_result_count(&signature);
    uint32_t expected_depth = 0;
    for (uint32_t descriptor_i = 0; descriptor_i < descriptor_count;
         ++descriptor_i) {
      iree_vm_bytecode_v0_signature_descriptor_row_t descriptor;
      IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_DESCRIPTOR_ROW,
          signature.descriptor_base_u32 + descriptor_i, &descriptor));
      if (descriptor.kind_u16 != IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION) {
        continue;
      }
      if (descriptor.type_ordinal_u16 >= i) {
        return iree_vm_bytecode_assembler_module_error(
            "callable types are not topologically ordered");
      }
      iree_vm_bytecode_v0_callable_type_row_t child;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_callable(
          view, descriptor.type_ordinal_u16, &child));
      expected_depth =
          iree_max(expected_depth, (uint32_t)child.nesting_depth_u16 + 1);
    }
    if (callable.nesting_depth_u16 != expected_depth) {
      return iree_vm_bytecode_assembler_module_error(
          "callable nesting depth is not canonical");
    }
    if (i != 0) {
      bool has_same_signature = false;
      int comparison = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_callables(
          view, &previous, &callable, &has_same_signature, &comparison));
      if (comparison >= 0) {
        return iree_vm_bytecode_assembler_module_error(
            "callable types are not unique and strictly ordered");
      }
      if (has_same_signature &&
          previous.signature_ordinal_u16 != callable.signature_ordinal_u16) {
        return iree_vm_bytecode_assembler_module_error(
            "equal callable signatures do not share one signature ordinal");
      }
    }
    previous = callable;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_imports(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  uint32_t entry_base = 0;
  uint16_t previous_module = 0;
  const uint32_t group_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_GROUP_ROW];
  for (uint32_t group_i = 0; group_i < group_count; ++group_i) {
    iree_vm_bytecode_v0_import_group_row_t group;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_GROUP_ROW, group_i,
        &group));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_require_nonempty_string(
            view, group.module_name_string_u16));
    if (group_i != 0) {
      int comparison = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_strings(
          view, previous_module, group.module_name_string_u16, &comparison));
      if (comparison >= 0) {
        return iree_vm_bytecode_assembler_module_error(
            "import groups are not strictly ordered");
      }
    }

    uint16_t previous_symbol = 0;
    uint16_t previous_callable = 0;
    for (uint32_t entry_i = 0; entry_i < group.entry_count_u32; ++entry_i) {
      iree_vm_bytecode_v0_import_entry_row_t entry;
      IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_ENTRY_ROW,
          entry_base + entry_i, &entry));
      IREE_RETURN_IF_ERROR(
          iree_vm_bytecode_assembler_module_require_nonempty_string(
              view, entry.symbol_name_string_u16));
      if (entry_i != 0) {
        int comparison = 0;
        IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_strings(
            view, previous_symbol, entry.symbol_name_string_u16, &comparison));
        if (comparison > 0 ||
            (comparison == 0 &&
             entry.callable_type_ordinal_u16 <= previous_callable)) {
          return iree_vm_bytecode_assembler_module_error(
              "import entries are not strictly ordered");
        }
      }
      previous_symbol = entry.symbol_name_string_u16;
      previous_callable = entry.callable_type_ordinal_u16;
    }
    entry_base += group.entry_count_u32;
    previous_module = group.module_name_string_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_functions(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  const uint32_t function_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW];
  for (uint32_t i = 0; i < function_count; ++i) {
    iree_vm_bytecode_v0_function_row_t function;
    iree_vm_bytecode_v0_callable_type_row_t callable;
    iree_vm_bytecode_v0_signature_row_t signature;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW, i, &function));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_callable(
        view, function.callable_type_ordinal_u16, &callable));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_signature(
        view, callable.signature_ordinal_u16, &signature));
    if ((iree_any_bit_set(function.flags_u16,
                          IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD) &&
         !iree_any_bit_set(callable.flags_u16,
                           IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD)) ||
        function.value_register_count_u16 <
            iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                     iree_max(signature.argument_value_count_u16,
                              signature.result_value_count_u16)) ||
        function.ref_register_count_u16 <
            iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                     iree_max(signature.argument_ref_count_u16,
                              signature.result_ref_count_u16)) ||
        function.function_register_count_u16 <
            iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                     iree_max(signature.argument_function_count_u16,
                              signature.result_function_count_u16))) {
      return iree_vm_bytecode_assembler_module_error(
          "a function frame does not implement its callable contract");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_globals(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  if (view->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_GLOBALS_HEADER] ==
      UINT64_MAX) {
    return iree_ok_status();
  }
  iree_vm_bytecode_v0_globals_header_t header;
  IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
      view, IREE_VM_BYTECODE_MODULE_RECORD_GLOBALS_HEADER, 0, &header));
  if ((header.value_count_u32 == 0 && header.ref_count_u32 == 0 &&
       header.function_count_u32 == 0) ||
      header.immutable_value_count_u32 > header.value_count_u32 ||
      header.immutable_ref_count_u32 > header.ref_count_u32 ||
      header.immutable_function_count_u32 > header.function_count_u32) {
    return iree_vm_bytecode_assembler_module_error(
        "global counts or immutable prefixes are invalid");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_exports(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  uint16_t previous_name = 0;
  const uint32_t export_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_EXPORT_ROW];
  for (uint32_t i = 0; i < export_count; ++i) {
    iree_vm_bytecode_v0_export_row_t export_row;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_EXPORT_ROW, i, &export_row));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_require_nonempty_string(
            view, export_row.name_string_u16));
    if (i != 0) {
      int comparison = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_strings(
          view, previous_name, export_row.name_string_u16, &comparison));
      if (comparison >= 0) {
        return iree_vm_bytecode_assembler_module_error(
            "exports are not unique and strictly ordered");
      }
    }

    iree_vm_bytecode_v0_callable_type_row_t export_callable;
    iree_vm_bytecode_v0_function_row_t function;
    iree_vm_bytecode_v0_callable_type_row_t function_callable;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_callable(
        view, export_row.callable_type_ordinal_u16, &export_callable));
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW,
        export_row.function_ordinal_u16, &function));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_callable(
        view, function.callable_type_ordinal_u16, &function_callable));
    if (!iree_vm_bytecode_assembler_module_callable_is_compatible(
            &function_callable,
            iree_any_bit_set(function.flags_u16,
                             IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD),
            &export_callable)) {
      return iree_vm_bytecode_assembler_module_error(
          "an export callable contract does not match its function");
    }
    previous_name = export_row.name_string_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_presentation(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  uint32_t field_base = 0;
  uint16_t previous_kind = 0;
  uint16_t previous_ordinal = 0;
  const uint32_t entry_count =
      view->record_counts
          [IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_ENTRY_ROW];
  const uint32_t field_count =
      view->record_counts
          [IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_FIELD_ROW];
  for (uint32_t i = 0; i < entry_count; ++i) {
    iree_vm_bytecode_v0_presentation_entry_row_t entry;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_ENTRY_ROW, i,
        &entry));
    if (i != 0 && (entry.declaration_kind_u16 < previous_kind ||
                   (entry.declaration_kind_u16 == previous_kind &&
                    entry.declaration_ordinal_u16 <= previous_ordinal))) {
      return iree_vm_bytecode_assembler_module_error(
          "presentation declarations are not strictly ordered");
    }

    uint16_t callable_ordinal = 0;
    if (entry.declaration_kind_u16 ==
        IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT) {
      iree_vm_bytecode_v0_import_entry_row_t import;
      IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_IMPORT_ENTRY_ROW,
          entry.declaration_ordinal_u16, &import));
      callable_ordinal = import.callable_type_ordinal_u16;
    } else {
      iree_vm_bytecode_v0_export_row_t export_row;
      IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_EXPORT_ROW,
          entry.declaration_ordinal_u16, &export_row));
      callable_ordinal = export_row.callable_type_ordinal_u16;
    }
    iree_vm_bytecode_v0_callable_type_row_t callable;
    iree_vm_bytecode_v0_signature_row_t signature;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_callable(
        view, callable_ordinal, &callable));
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_read_signature(
        view, callable.signature_ordinal_u16, &signature));
    const uint32_t declaration_field_count =
        iree_vm_bytecode_assembler_module_signature_argument_count(&signature) +
        iree_vm_bytecode_assembler_module_signature_result_count(&signature);
    if (entry.field_base_u32 != field_base || field_base > field_count ||
        declaration_field_count > field_count - field_base) {
      return iree_vm_bytecode_assembler_module_error(
          "presentation fields do not match their declaration");
    }
    bool has_value = entry.documentation_string_u16 != UINT16_MAX ||
                     entry.authored_type_string_u16 != UINT16_MAX;
    for (uint32_t field_i = 0; field_i < declaration_field_count; ++field_i) {
      iree_vm_bytecode_v0_presentation_field_row_t field;
      IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_PRESENTATION_FIELD_ROW,
          field_base + field_i, &field));
      has_value |= field.name_string_u16 != UINT16_MAX ||
                   field.authored_type_string_u16 != UINT16_MAX;
    }
    if (!has_value) {
      return iree_vm_bytecode_assembler_module_error(
          "a presentation entry has no authored information");
    }
    field_base += declaration_field_count;
    previous_kind = entry.declaration_kind_u16;
    previous_ordinal = entry.declaration_ordinal_u16;
  }
  return field_base == field_count
             ? iree_ok_status()
             : iree_vm_bytecode_assembler_module_error(
                   "presentation entries do not consume their fields");
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_metadata_range(
    const iree_vm_bytecode_assembler_module_view_t* view, uint32_t entry_base,
    uint32_t entry_count) {
  uint16_t previous_key = 0;
  for (uint32_t i = 0; i < entry_count; ++i) {
    iree_vm_bytecode_v0_metadata_entry_row_t entry;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_METADATA_ENTRY_ROW, entry_base + i,
        &entry));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_require_nonempty_string(
            view, entry.key_string_u16));
    if (i != 0) {
      int comparison = 0;
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_module_compare_strings(
          view, previous_key, entry.key_string_u16, &comparison));
      if (comparison >= 0) {
        return iree_vm_bytecode_assembler_module_error(
            "metadata keys are not unique and strictly ordered");
      }
    }
    previous_key = entry.key_string_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_metadata_scopes(
    const iree_vm_bytecode_assembler_module_view_t* view, uint32_t scope_base,
    uint32_t scope_count) {
  uint16_t previous_ordinal = 0;
  for (uint32_t i = 0; i < scope_count; ++i) {
    iree_vm_bytecode_v0_metadata_scope_row_t scope;
    IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_METADATA_SCOPE_ROW, scope_base + i,
        &scope));
    if (i != 0 && scope.declaration_ordinal_u16 <= previous_ordinal) {
      return iree_vm_bytecode_assembler_module_error(
          "metadata declaration scopes are not strictly ordered");
    }
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_assembler_module_validate_metadata_range(
            view, scope.entry_base_u32, scope.entry_count_u16));
    previous_ordinal = scope.declaration_ordinal_u16;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_module_validate_metadata(
    const iree_vm_bytecode_assembler_module_view_t* view) {
  if (view->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_METADATA_HEADER] ==
      UINT64_MAX) {
    return iree_ok_status();
  }
  iree_vm_bytecode_v0_metadata_header_t header;
  IREE_RETURN_IF_ERROR(IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
      view, IREE_VM_BYTECODE_MODULE_RECORD_METADATA_HEADER, 0, &header));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_metadata_range(
          view, 0, header.module_entry_count_u32));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_metadata_scopes(
          view, 0, header.import_scope_count_u32));
  return iree_vm_bytecode_assembler_module_validate_metadata_scopes(
      view, header.import_scope_count_u32, header.export_scope_count_u32);
}

static iree_status_t iree_vm_bytecode_assembler_module_native_calls(
    const iree_vm_bytecode_assembler_module_view_t* view,
    iree_allocator_t scratch_allocator) {
  if (view->record_offsets
          [IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_CALLS_HEADER] == UINT64_MAX) {
    return iree_ok_status();
  }
  const uint32_t layout_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_LAYOUT_ROW];
  const uint32_t field_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_FIELD_ROW];
  const uint32_t word_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_CALL_WORD];
  const uint32_t import_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_IMPORT_ROW];
  const uint32_t callsite_count =
      view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_CALLSITE_ROW];
  if (!layout_count || layout_count > 65536 || !import_count ||
      import_count > 65536 || !callsite_count) {
    return iree_vm_bytecode_assembler_module_error(
        "native calls require nonempty layout, import and callsite tables");
  }
  iree_host_size_t storage_size = 0, layouts_offset = 0, fields_offset = 0,
                   words_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(layout_count, iree_vm_native_call_layout_t,
                        &layouts_offset),
      IREE_STRUCT_FIELD(field_count, iree_vm_native_call_field_t,
                        &fields_offset),
      IREE_STRUCT_FIELD(word_count, uint32_t, &words_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      scratch_allocator, storage_size, (void**)&storage));
  iree_vm_native_call_layout_t* layouts =
      (iree_vm_native_call_layout_t*)(storage + layouts_offset);
  const iree_vm_native_call_schema_t schema = {
      .layouts = layouts,
      .fields = (const iree_vm_native_call_field_t*)(storage + fields_offset),
      .layout_count = layout_count,
      .field_count = field_count,
  };
  const uint32_t* words = (const uint32_t*)(storage + words_offset);
  iree_status_t status = iree_vm_bytecode_assembler_module_read(
      view,
      view->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_LAYOUT_ROW],
      layout_count * sizeof(*layouts), layouts);
  if (iree_status_is_ok(status) && field_count) {
    status = iree_vm_bytecode_assembler_module_read(
        view,
        view->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_FIELD_ROW],
        field_count * sizeof(*schema.fields), storage + fields_offset);
  }
  if (iree_status_is_ok(status) && word_count) {
    status = iree_vm_bytecode_assembler_module_read(
        view,
        view->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_CALL_WORD],
        word_count * sizeof(*words), storage + words_offset);
  }
  for (uint32_t i = 0; i < layout_count && iree_status_is_ok(status); ++i) {
    iree_vm_native_call_layout_t* row = &layouts[i];
    for (uint32_t j = 0; j < row->field_count && iree_status_is_ok(status);
         ++j) {
      const iree_vm_native_call_field_t field =
          schema.fields[row->field_base + j];
      if (field.kind != IREE_VM_NATIVE_CALL_FIELD_RANGE) {
        continue;
      }
      if (field.type_ordinal >= i) {
        status = iree_vm_bytecode_assembler_module_error(
            "native range must name an earlier layout");
      } else {
        row->depth =
            iree_max(row->depth, 1u + layouts[field.type_ordinal].depth);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_native_call_schema_validate(
        &schema,
        view->record_counts[IREE_VM_BYTECODE_MODULE_RECORD_REF_TYPE_ENTRY_ROW]);
  }
  for (uint32_t i = 0; i < callsite_count && iree_status_is_ok(status); ++i) {
    iree_vm_bytecode_v0_native_callsite_row_t row;
    iree_vm_bytecode_v0_native_import_row_t import;
    iree_vm_bytecode_v0_function_row_t function;
    status = IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
        view, IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_CALLSITE_ROW, i, &row);
    if (iree_status_is_ok(status)) {
      status = IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_IMPORT_ROW,
          row.import_ordinal_u16, &import);
    }
    if (iree_status_is_ok(status)) {
      status = IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ(
          view, IREE_VM_BYTECODE_MODULE_RECORD_FUNCTION_ROW,
          row.function_ordinal_u16, &function);
    }
    if (iree_status_is_ok(status)) {
      iree_vm_native_callsite_t callsite = {
          .schema = schema,
          .record = words + row.word_offset_u32,
          .word_count = row.word_count_u32,
          .storage = {.value_count = function.value_register_count_u16,
                      .ref_count = function.ref_register_count_u16 +
                                   function.local_ref_count_u32,
                      .local_byte_length = function.local_byte_length_u16},
      };
      callsite.schema.root_ordinal = import.layout_ordinal_u16;
      status = iree_vm_native_call_verify(&callsite, scratch_allocator);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(
        view->stream, IREE_IO_STREAM_SEEK_SET,
        view->record_offsets[IREE_VM_BYTECODE_MODULE_RECORD_NATIVE_LAYOUT_ROW]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_write(view->stream, layout_count * sizeof(*layouts),
                                  layouts);
  }
  iree_allocator_free(scratch_allocator, storage);
  return status;
}

iree_status_t iree_vm_bytecode_assembler_module_finalize(
    const iree_vm_bytecode_assembler_module_view_t* view,
    iree_allocator_t scratch_allocator) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_requirements(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_ref_types(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_callables(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_imports(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_functions(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_globals(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_exports(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_presentation(view));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_module_validate_metadata(view));
  return iree_vm_bytecode_assembler_module_native_calls(view,
                                                        scratch_allocator);
}

#undef IREE_VM_BYTECODE_ASSEMBLER_MODULE_READ
