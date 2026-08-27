// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/symbol_schema.h"

#include "loom/error/error_catalog.h"

iree_status_t loom_bytecode_symbol_validate_string_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view, uint64_t string_id,
    iree_string_view_t field_name, uint64_t offset,
    iree_string_view_t* out_string) {
  if (string_id >= module_view->strings.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(field_name),
        loom_param_u64(string_id),
        loom_param_u64(module_view->strings.count),
    };
    return loom_bytecode_reader_emit_error(decoder, LOOM_ERR_BYTECODE_010,
                                           params, IREE_ARRAYSIZE(params),
                                           offset, 0);
  }
  *out_string = module_view->strings.values[string_id];
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbol_validate_type_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view, uint64_t type_id,
    uint64_t offset) {
  if (type_id >= module_view->types.count) {
    return loom_bytecode_reader_emit_table_ref(
        decoder, IREE_SV("TYPES"), type_id, module_view->types.count, offset);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbol_resolve_op_ref(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reader_module_view_t* module_view,
    uint64_t op_table_index_plus1, uint64_t offset,
    const loom_op_vtable_t** out_vtable) {
  if (op_table_index_plus1 == 0 ||
      op_table_index_plus1 > module_view->ops.count) {
    return loom_bytecode_reader_emit_table_ref(decoder, IREE_SV("OPS"),
                                               op_table_index_plus1,
                                               module_view->ops.count, offset);
  }
  *out_vtable = module_view->ops.values[op_table_index_plus1 - 1];
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbol_validate_func_enum(
    loom_bytecode_reader_decoder_t* decoder, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint8_t attr_index,
    iree_string_view_t field_name, uint8_t value, uint64_t offset) {
  if (value == 0) {
    return iree_ok_status();
  }
  if (!vtable || attr_index == LOOM_ATTR_INDEX_NONE ||
      attr_index >= vtable->attribute_count || !vtable->attr_descriptors) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        field_name, offset,
        IREE_SV("function_metadata_field_is_not_supported_by_op"));
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_ENUM) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        field_name, offset,
        IREE_SV("function_metadata_field_must_target_an_enum_attr"));
  }
  if (!iree_any_bit_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM) &&
      !loom_attr_descriptor_has_enum_case(descriptor, value)) {
    return loom_bytecode_reader_emit_enum_value(
        decoder, field_name, value,
        loom_attr_descriptor_enum_case_span(descriptor), offset);
  }
  return iree_ok_status();
}

uint8_t loom_bytecode_symbol_find_op_attr_index_by_name(
    const loom_op_vtable_t* vtable, iree_string_view_t name) {
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (iree_string_view_equal(
            loom_attr_descriptor_name(&vtable->attr_descriptors[i]), name)) {
      return i;
    }
  }
  return LOOM_ATTR_INDEX_NONE;
}

bool loom_bytecode_symbol_attr_is_identity(const loom_op_vtable_t* vtable,
                                           uint8_t attr_index) {
  return vtable && vtable->symbol_def &&
         attr_index == vtable->symbol_def->name_attr_index;
}

bool loom_bytecode_symbol_func_metadata_attr_is_shared(
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    uint8_t attr_index) {
  if (loom_bytecode_symbol_attr_is_identity(vtable, attr_index)) {
    return true;
  }
  iree_string_view_t name =
      loom_attr_descriptor_name(&vtable->attr_descriptors[attr_index]);
  if (iree_string_view_equal(name, IREE_SV("import_module")) ||
      iree_string_view_equal(name, IREE_SV("import_symbol")) ||
      attr_index == func_like->visibility_attr_index ||
      attr_index == func_like->cc_attr_index ||
      attr_index == func_like->purity_attr_index ||
      attr_index == func_like->predicates_attr_index) {
    return true;
  }
  if (attr_index == func_like->template_family_attr_index ||
      attr_index == func_like->priority_attr_index) {
    return true;
  }
  return false;
}

uint8_t loom_bytecode_symbol_find_identity_attr_index(
    const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->attr_descriptors) {
    return LOOM_ATTR_INDEX_NONE;
  }
  if (vtable->symbol_def) {
    uint8_t attr_index = vtable->symbol_def->name_attr_index;
    if (attr_index < vtable->attribute_count &&
        vtable->attr_descriptors[attr_index].attr_kind == LOOM_ATTR_SYMBOL) {
      return attr_index;
    }
    return LOOM_ATTR_INDEX_NONE;
  }
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (vtable->attr_descriptors[i].attr_kind == LOOM_ATTR_SYMBOL) {
      return i;
    }
  }
  return LOOM_ATTR_INDEX_NONE;
}

iree_status_t loom_bytecode_symbol_validate_global_vtable(
    loom_bytecode_reader_decoder_t* decoder, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint64_t op_ref_offset) {
  if (!iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def ||
      vtable->symbol_def->bytecode_kind != LOOM_SYMBOL_GLOBAL) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("global-payload symbol defining op must use the GLOBAL "
                "bytecode payload"));
  }
  if (loom_op_vtable_operand_descriptor_count(vtable) != 0 ||
      vtable->region_count != 0 ||
      iree_any_bit_set(
          vtable->vtable_flags,
          LOOM_OP_VTABLE_VARIADIC_OPERANDS | LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("global_symbol_defining_op_must_not_have_operands_or_regions"));
  }
  if (loom_bytecode_symbol_find_identity_attr_index(vtable) ==
      LOOM_ATTR_INDEX_NONE) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("global_symbol_defining_op_must_declare_a_symbol_attribute"));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_symbol_validate_record_vtable(
    loom_bytecode_reader_decoder_t* decoder, uint64_t symbol_index,
    const loom_op_vtable_t* vtable, uint64_t op_ref_offset) {
  if (!iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def ||
      !loom_symbol_definition_implements(vtable->symbol_def,
                                         LOOM_SYMBOL_INTERFACE_RECORD)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record_symbol_defining_op_must_define_a_record_symbol"));
  }
  if (loom_op_vtable_operand_descriptor_count(vtable) != 0 ||
      vtable->fixed_result_count != 0 ||
      iree_any_bit_set(
          vtable->vtable_flags,
          LOOM_OP_VTABLE_VARIADIC_OPERANDS | LOOM_OP_VTABLE_VARIADIC_RESULTS)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record_symbol_defining_op_must_not_have_operands_or_results"));
  }
  if (vtable->region_count > 1 ||
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record symbol defining op must declare at most one fixed "
                "region"));
  }
  if (loom_bytecode_symbol_find_identity_attr_index(vtable) ==
      LOOM_ATTR_INDEX_NONE) {
    return loom_bytecode_reader_emit_invalid_field(
        decoder, IREE_SV("SYMBOLS"), IREE_SV("symbol"), symbol_index,
        IREE_SV("def_op_table_index_plus1"), op_ref_offset,
        IREE_SV("record_symbol_defining_op_must_declare_a_symbol_attribute"));
  }
  return iree_ok_status();
}
