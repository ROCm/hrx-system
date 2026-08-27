// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/symbol.h"

#include <string.h>

#include "loom/format/bytecode/writer/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Symbol metadata and dependency facets
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_symbol_kind_byte(loom_symbol_kind_t kind,
                                                    uint8_t* out_byte) {
  switch (kind) {
    case LOOM_SYMBOL_FUNC_DEF:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_DEF;
      return iree_ok_status();
    case LOOM_SYMBOL_FUNC_DECL:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_DECL;
      return iree_ok_status();
    case LOOM_SYMBOL_TEMPLATE_DECL:
      *out_byte = LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL;
      return iree_ok_status();
    case LOOM_SYMBOL_TEMPLATE_DEF:
      *out_byte = LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF;
      return iree_ok_status();
    case LOOM_SYMBOL_TEMPLATE_UKERNEL:
      *out_byte = LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL;
      return iree_ok_status();
    case LOOM_SYMBOL_GLOBAL:
      *out_byte = LOOM_BYTECODE_SYMBOL_GLOBAL;
      return iree_ok_status();
    case LOOM_SYMBOL_EXECUTABLE:
      *out_byte = LOOM_BYTECODE_SYMBOL_EXECUTABLE;
      return iree_ok_status();
    case LOOM_SYMBOL_RECORD:
      *out_byte = LOOM_BYTECODE_SYMBOL_RECORD;
      return iree_ok_status();
    case LOOM_SYMBOL_NONE:
      break;
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown symbol kind %u", (unsigned)kind);
}

// Writes the function-like metadata fields for a single symbol entry.
// Called from loom_bytecode_write_symbols_section for each function-like
// symbol that has a defining op.
static bool loom_bytecode_func_metadata_attr_is_shared(
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    uint8_t attr_index) {
  if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) return true;
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

static iree_status_t loom_bytecode_write_func_payload_attrs(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_value_numbering_t* signature_numbering) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, func_like.op);
  const loom_attribute_t* attrs = loom_op_attrs(func_like.op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < func_like.op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        func_like.op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_func_metadata_attr_is_shared(
                       vtable, func_like.vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_attr_count));
  for (uint8_t i = 0; i < func_like.op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        func_like.op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_func_metadata_attr_is_shared(
                        vtable, func_like.vtable, i)) {
      continue;
    }
    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(
        builder, numbering, signature_numbering, attrs[i], descriptor));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_region_payload_references(
    iree_string_builder_t* builder,
    const loom_bytecode_ir_region_list_t* region_list) {
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, region_list->count));
  for (uint8_t i = 0; i < region_list->count; ++i) {
    const loom_bytecode_ir_region_payload_t* payload = &region_list->values[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, payload->region_index));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, payload->offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u32_le(builder, payload->length));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_func_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_value_numbering_t* signature_numbering,
    const loom_bytecode_ir_region_list_t* region_list) {
  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_op(
      numbering, func_like.op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, func_like.op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_source_trivia(
      builder,
      iree_any_bit_set(func_like.op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_u8(builder, loom_func_like_cc(func_like)));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_u8(builder, loom_func_like_purity(func_like)));

  loom_value_slice_t workload_args =
      loom_kernel_workload_arg_ids(module, func_like.op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  uint16_t result_count = func_like.op->result_count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, workload_args.count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, arg_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, result_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
      signature_numbering,
      (iree_host_size_t)workload_args.count + arg_count + result_count));

  for (uint16_t i = 0; i < workload_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, workload_args.values[i]));
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, arg_ids[i]));
  }
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  for (uint16_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, result_ids[i]));
  }

  // Kernel workload and ordinary FuncLike argument value definitions.
  for (uint16_t i = 0; i < workload_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_value_def(
        builder, numbering, signature_numbering,
        loom_module_value(module, workload_args.values[i])));
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_value_def(builder, numbering, signature_numbering,
                                     loom_module_value(module, arg_ids[i])));
  }

  // Result value definitions with tied info.
  const loom_tied_result_t* tied_results = loom_op_tied_results(func_like.op);
  uint16_t tied_result_count = func_like.op->tied_result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    bool is_tied = false;
    uint16_t tied_operand_index = 0;
    for (uint16_t t = 0; t < tied_result_count; ++t) {
      if (tied_results[t].result_index == i) {
        is_tied = true;
        tied_operand_index = tied_results[t].operand_index;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, is_tied ? 1 : 0));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_value_def(builder, numbering, signature_numbering,
                                     loom_module_value(module, result_ids[i])));
    if (is_tied) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, tied_operand_index));
    }
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, tied_result_count));

  // Predicates.
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(func_like, &predicate_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, predicate_count));
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->arg_count));
    for (uint8_t arg_index = 0; arg_index < predicate->arg_count; ++arg_index) {
      uint8_t tag = predicate->arg_tags[arg_index];
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, tag));
      switch (tag) {
        case LOOM_PRED_ARG_VALUE: {
          uint32_t value_number = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
              signature_numbering, (loom_value_id_t)predicate->args[arg_index],
              &value_number));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_emit_uvarint(builder, value_number));
          break;
        }
        case LOOM_PRED_ARG_CONST: {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_emit_svarint(builder, predicate->args[arg_index]));
          break;
        }
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown predicate arg tag %d", (int)tag);
      }
    }
  }

  // Template provider metadata references the family by module-local symbol
  // ordinal. This preserves symbol identity across private families and lets
  // selected materialization project the reference without string lookup.
  loom_symbol_ref_t template_family = loom_func_like_template_family(func_like);
  if (func_like.vtable->template_family_attr_index != LOOM_ATTR_INDEX_NONE) {
    if (!loom_symbol_ref_is_valid(template_family) ||
        template_family.module_id != 0 ||
        template_family.symbol_id >= module->symbols.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template provider symbol must reference a module-local family");
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(
        builder, loom_bytecode_wire_symbol_ordinal(numbering,
                                                   template_family.symbol_id)));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(
        builder, (uint64_t)loom_func_like_priority(func_like)));
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_write_func_payload_attrs(
      builder, numbering, module, func_like, signature_numbering));

  return loom_bytecode_write_region_payload_references(builder, region_list);
}

static iree_status_t loom_bytecode_write_global_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, const loom_op_t* op,
    loom_bytecode_value_numbering_t* value_numbering) {
  loom_bytecode_global_value_list_t local_values = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_collect_global_values(
      numbering->arena, module, op, &local_values));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_number_global(numbering, op, &local_values));
  IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
      value_numbering, local_values.count));

  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_source_trivia(
      builder, iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, op->result_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, local_values.count));
  for (iree_host_size_t i = 0; i < local_values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        value_numbering, local_values.values[i]));
  }
  for (iree_host_size_t i = 0; i < local_values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_value_def(
        builder, numbering, value_numbering,
        loom_module_value(module, local_values.values[i])));
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(
        builder, numbering, value_numbering, attrs[i], descriptor));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_record_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, const loom_op_t* op,
    const loom_bytecode_ir_region_list_t* region_list) {
  IREE_RETURN_IF_ERROR(loom_bytecode_validate_record_symbol_op(module, op));
  IREE_RETURN_IF_ERROR(loom_bytecode_number_record(numbering, op));

  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_source_trivia(
      builder, iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(builder, numbering, NULL,
                                                       attrs[i], descriptor));
  }

  return loom_bytecode_write_region_payload_references(builder, region_list);
}

static loom_attribute_t loom_bytecode_find_op_attr_by_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    iree_string_view_t name) {
  if (!vtable || !vtable->attr_descriptors) return loom_attr_absent();
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (iree_string_view_equal(
            loom_attr_descriptor_name(&vtable->attr_descriptors[i]), name)) {
      return attrs[i];
    }
  }
  return loom_attr_absent();
}

static iree_status_t loom_bytecode_find_string_attr_by_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    iree_string_view_t name, loom_string_id_t* out_string_id) {
  loom_attribute_t attr = loom_bytecode_find_op_attr_by_name(vtable, op, name);
  if (loom_attr_is_absent(attr)) {
    *out_string_id = LOOM_STRING_ID_INVALID;
    return iree_ok_status();
  }
  if (attr.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function symbol attribute %.*s must be a string",
                            (int)name.size, name.data);
  }
  *out_string_id = loom_attr_as_string_id(attr);
  return iree_ok_status();
}

typedef struct loom_bytecode_symbol_linkage_t {
  // True when the symbol has public source visibility.
  bool is_public;
  // True when the symbol is available for cross-module static linkage.
  bool is_export;
  // True when the symbol resolves to another module.
  bool is_import;
  // True when the source symbol name was explicitly authored.
  bool has_import_symbol;
  // Source module for an imported symbol.
  loom_string_id_t import_module_id;
  // Source symbol name for an imported symbol.
  loom_string_id_t import_symbol_id;
} loom_bytecode_symbol_linkage_t;

static bool loom_bytecode_symbol_has_visibility_attr(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!symbol->defining_op) return false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, symbol->defining_op);
  if (!vtable || !vtable->attr_descriptors) return false;
  const loom_attribute_t* attrs = loom_op_const_attrs(symbol->defining_op);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (!iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                IREE_SV("visibility"))) {
      continue;
    }
    if (descriptor->attr_kind != LOOM_ATTR_ENUM ||
        i >= symbol->defining_op->attribute_count) {
      return false;
    }
    return loom_attr_as_enum(attrs[i]) != 0;
  }
  return false;
}

static iree_status_t loom_bytecode_symbol_linkage(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_bytecode_symbol_linkage_t* out_linkage) {
  *out_linkage = (loom_bytecode_symbol_linkage_t){
      .is_public = iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC) ||
                   loom_bytecode_symbol_has_visibility_attr(module, symbol),
      .is_export = false,
      .is_import = false,
      .has_import_symbol = false,
      .import_module_id = LOOM_STRING_ID_INVALID,
      .import_symbol_id = LOOM_STRING_ID_INVALID,
  };
  out_linkage->is_export = out_linkage->is_public;
  if (!symbol->defining_op) return iree_ok_status();

  loom_func_like_t func_like = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func_like)) return iree_ok_status();
  if (loom_func_like_export_symbol(func_like) != LOOM_STRING_ID_INVALID) {
    out_linkage->is_export = true;
  }

  const loom_op_vtable_t* op_vtable = loom_op_vtable(module, func_like.op);
  IREE_RETURN_IF_ERROR(loom_bytecode_find_string_attr_by_name(
      op_vtable, func_like.op, IREE_SV("import_module"),
      &out_linkage->import_module_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_find_string_attr_by_name(
      op_vtable, func_like.op, IREE_SV("import_symbol"),
      &out_linkage->import_symbol_id));
  out_linkage->has_import_symbol =
      out_linkage->import_symbol_id != LOOM_STRING_ID_INVALID;

  if (out_linkage->import_module_id == LOOM_STRING_ID_INVALID) {
    if (out_linkage->import_symbol_id != LOOM_STRING_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function symbol import_symbol requires import_module");
    }
    return iree_ok_status();
  }

  out_linkage->is_import = true;
  out_linkage->is_export = false;
  if (out_linkage->import_symbol_id == LOOM_STRING_ID_INVALID) {
    out_linkage->import_symbol_id = symbol->name_id;
  }
  return iree_ok_status();
}

// Writes the SYMBOLS section into a string builder (for offset table patching).
iree_status_t loom_bytecode_write_symbols_section(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_ir_region_list_t* ir_regions) {
  const loom_module_t* module = numbering->module;

  // Classify symbols.
  uint32_t import_count = 0;
  uint32_t export_count = 0;
  iree_host_size_t root_region_payload_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    loom_bytecode_symbol_linkage_t linkage;
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_linkage(
        module, &module->symbols.entries[i], &linkage));
    if (linkage.is_import) {
      ++import_count;
    } else if (linkage.is_export) {
      ++export_count;
    }
    root_region_payload_count += ir_regions[i].count;
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, module->symbols.count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, import_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, export_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, root_region_payload_count));

  // Reserve import/export offset tables (patched after writing entries).
  iree_host_size_t import_table_offset = iree_string_builder_size(builder);
  for (uint32_t i = 0; i < import_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, 0));
  }
  iree_host_size_t export_table_offset = iree_string_builder_size(builder);
  for (uint32_t i = 0; i < export_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, 0));
  }

  iree_host_size_t entries_start = iree_string_builder_size(builder);
  uint32_t import_index = 0;
  uint32_t export_index = 0;

  for (loom_symbol_id_t wire_ordinal = 0; wire_ordinal < module->symbols.count;
       ++wire_ordinal) {
    const loom_symbol_id_t module_symbol_id =
        loom_bytecode_module_symbol_id(numbering, wire_ordinal);
    const loom_symbol_t* symbol = &module->symbols.entries[module_symbol_id];
    loom_bytecode_symbol_linkage_t linkage;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_linkage(module, symbol, &linkage));
    loom_symbol_kind_t metadata_kind = loom_symbol_bytecode_kind(symbol);
    bool has_function_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(metadata_kind);
    bool has_global_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        metadata_kind == LOOM_SYMBOL_GLOBAL;
    bool has_record_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        metadata_kind == LOOM_SYMBOL_RECORD;
    uint64_t entry_offset = iree_string_builder_size(builder) - entries_start;

    // Track import/export offsets.
    if (linkage.is_import) {
      loom_bytecode_patch_u64_le(
          builder, import_table_offset + (iree_host_size_t)import_index * 8,
          entry_offset);
      ++import_index;
    } else if (linkage.is_export) {
      loom_bytecode_patch_u64_le(
          builder, export_table_offset + (iree_host_size_t)export_index * 8,
          entry_offset);
      ++export_index;
    }

    // Name.
    uint32_t name_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, symbol->name_id, &name_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, name_writer_id));

    // Kind.
    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_kind_byte(
        loom_symbol_bytecode_kind(symbol), &kind_byte));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, kind_byte));

    // Visibility.
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(
        builder, linkage.is_public ? LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC
                                   : LOOM_BYTECODE_SYMBOL_VISIBILITY_PRIVATE));

    // Flags.
    uint16_t bytecode_flags =
        linkage.is_public ? LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC : 0;
    if (linkage.is_export) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_EXPORT;
    }
    if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_RETAIN)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_RETAIN;
    }
    if (loom_symbol_definition_is_declaration(symbol->definition)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION;
    }
    if (loom_symbol_definition_is_test_only(symbol->definition)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY;
    }
    if (linkage.is_import) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_IMPORT;
      if (linkage.has_import_symbol) {
        bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL;
      }
    }
    if (has_function_metadata && symbol->defining_op) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(func_like) &&
          func_like.vtable->predicates_attr_index != LOOM_ATTR_INDEX_NONE &&
          !loom_attr_is_absent(loom_op_const_attrs(
              func_like.op)[func_like.vtable->predicates_attr_index])) {
        bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES;
      }
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u16_le(builder, bytecode_flags));
    if (linkage.is_import) {
      uint32_t import_module_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, linkage.import_module_id, &import_module_string_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, import_module_string_id));
      uint32_t import_symbol_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, linkage.import_symbol_id, &import_symbol_string_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, import_symbol_string_id));
    }

    // Function metadata.
    if (has_function_metadata && symbol->defining_op) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(func_like)) {
        loom_bytecode_value_numbering_t signature_numbering;
        loom_bytecode_value_numbering_initialize(&signature_numbering, module,
                                                 numbering->arena);
        IREE_RETURN_IF_ERROR(loom_bytecode_write_func_metadata(
            builder, numbering, module, func_like, &signature_numbering,
            &ir_regions[module_symbol_id]));
      }
    } else if (has_global_metadata && symbol->defining_op) {
      loom_bytecode_value_numbering_t signature_numbering;
      loom_bytecode_value_numbering_initialize(&signature_numbering, module,
                                               numbering->arena);
      IREE_RETURN_IF_ERROR(loom_bytecode_write_global_metadata(
          builder, numbering, module, symbol->defining_op,
          &signature_numbering));
    } else if (has_record_metadata && symbol->defining_op) {
      IREE_RETURN_IF_ERROR(loom_bytecode_write_record_metadata(
          builder, numbering, module, symbol->defining_op,
          &ir_regions[module_symbol_id]));
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Symbol reference index
//===----------------------------------------------------------------------===//

// Prepared analysis and aggregate counts written to SYMBOL_REFERENCES.
static uint32_t loom_bytecode_count_dependency_occurrences(
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t first_occurrence_id) {
  uint32_t dependency_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id = first_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table->occurrences[occurrence_id];
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      ++dependency_count;
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return dependency_count;
}

iree_status_t loom_bytecode_symbol_reference_plan_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_symbol_reference_plan_t* out_plan) {
  *out_plan = (loom_bytecode_symbol_reference_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(module, arena, &out_plan->table));

  for (iree_host_size_t i = 0; i < out_plan->table.occurrence_count; ++i) {
    if (loom_symbol_reference_occurrence_is_dependency(
            &out_plan->table.occurrences[i])) {
      ++out_plan->dependency_count;
    }
  }
  out_plan->module_dependency_count =
      loom_bytecode_count_dependency_occurrences(
          &out_plan->table, out_plan->table.first_module_occurrence_id);

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_dependency_row(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering,
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t first_occurrence_id,
    uint32_t dependency_count) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, dependency_count));
  loom_symbol_reference_occurrence_id_t occurrence_id = first_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table->occurrences[occurrence_id];
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, occurrence->source_root_region_index_plus_one));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, loom_bytecode_wire_symbol_ordinal(
                           numbering, occurrence->target_symbol_id)));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, occurrence->target_interfaces));
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_symbol_references_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering,
    const loom_bytecode_symbol_reference_plan_t* plan) {
  const loom_symbol_reference_table_t* table = &plan->table;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, table->symbol_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, plan->dependency_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, table->template_demands.count));
  IREE_RETURN_IF_ERROR(loom_bytecode_write_dependency_row(
      page_writer, numbering, table, table->first_module_occurrence_id,
      plan->module_dependency_count));
  for (loom_symbol_id_t wire_ordinal = 0; wire_ordinal < table->symbol_count;
       ++wire_ordinal) {
    const loom_symbol_id_t module_symbol_id =
        loom_bytecode_module_symbol_id(numbering, wire_ordinal);
    const loom_symbol_reference_symbol_occurrences_t* symbol =
        &table->symbols[module_symbol_id];
    const uint32_t dependency_count =
        loom_bytecode_count_dependency_occurrences(
            table, symbol->first_outgoing_occurrence_id);
    IREE_RETURN_IF_ERROR(loom_bytecode_write_dependency_row(
        page_writer, numbering, table, symbol->first_outgoing_occurrence_id,
        dependency_count));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, symbol->template_demand_count));
    loom_template_demand_id_t demand_id = symbol->first_template_demand_id;
    while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID) {
      const loom_template_demand_t* demand =
          &table->template_demands.values[demand_id];
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, demand->source_root_region_index_plus_one));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, loom_bytecode_wire_symbol_ordinal(
                           numbering, demand->family_symbol_id)));
      demand_id = demand->next_source_demand_id;
    }
  }
  return iree_ok_status();
}
