// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/low_asm.h"

#include <inttypes.h>

#include "loom/format/text/printer/atoms.h"
#include "loom/format/text/printer/regions.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static iree_status_t loom_print_low_asm_lookup_descriptor_set(
    loom_print_context_t* ctx,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (!loom_text_low_asm_environment_supports_printing(
          &ctx->low_asm_environment)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "low asm region printing requires a descriptor-backed print "
        "environment");
  }
  if (iree_string_view_is_empty(ctx->low_asm_descriptor_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm region printing requires a descriptor-set key");
  }
  IREE_RETURN_IF_ERROR(ctx->low_asm_environment.vtable->lookup_descriptor_set(
      ctx->low_asm_environment.state, ctx->low_asm_descriptor_set_key,
      out_descriptor_set));
  if (*out_descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "low asm descriptor set '%.*s' was not found",
                            (int)ctx->low_asm_descriptor_set_key.size,
                            ctx->low_asm_descriptor_set_key.data);
  }
  return iree_ok_status();
}

bool loom_print_low_asm_is_requested(loom_print_context_t* ctx) {
  return !iree_string_view_is_empty(ctx->low_asm_descriptor_set_key);
}

static bool loom_print_low_asm_allows_canonical_structural_op(
    loom_print_context_t* ctx, const loom_op_t* op) {
  iree_string_view_t op_name = loom_op_name(ctx->module, op);
  return iree_string_view_equal(op_name, IREE_SV("low.br")) ||
         iree_string_view_equal(op_name, IREE_SV("low.cond_br")) ||
         iree_string_view_equal(op_name, IREE_SV("low.func.call")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.yield")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.if")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.for"));
}

static iree_status_t loom_print_low_asm_region_preflight(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    bool entry_args_declared_by_parent, bool* out_available);

static iree_status_t loom_print_low_asm_preflight_canonical_structural_op(
    loom_print_context_t* ctx,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    const loom_op_t* op, bool* out_available) {
  iree_string_view_t op_name = loom_op_name(ctx->module, op);
  if (iree_string_view_equal(op_name, IREE_SV("low.scf.if"))) {
    if (op->region_count < 1 || loom_op_regions(op)[0] == NULL) {
      *out_available = false;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, loom_op_regions(op)[0], descriptor_set,
        /*entry_args_declared_by_parent=*/false, out_available));
    if (!*out_available) {
      return iree_ok_status();
    }
    if (op->region_count > 1 && loom_op_regions(op)[1] != NULL) {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
          ctx, loom_op_regions(op)[1], descriptor_set,
          /*entry_args_declared_by_parent=*/false, out_available));
    }
  } else if (iree_string_view_equal(op_name, IREE_SV("low.scf.for"))) {
    if (op->region_count < 1 || loom_op_regions(op)[0] == NULL) {
      *out_available = false;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, loom_op_regions(op)[0], descriptor_set,
        /*entry_args_declared_by_parent=*/true, out_available));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_describe_operation(
    loom_print_context_t* ctx,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    const loom_op_t* op, loom_text_low_asm_statement_t* out_statement) {
  return ctx->low_asm_environment.vtable->describe_operation(
      ctx->low_asm_environment.state, descriptor_set, ctx->module, op,
      out_statement);
}

static iree_status_t loom_print_low_asm_attr_name(
    const loom_module_t* module, const loom_named_attr_t* attr,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (attr->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm immediate attribute name is out of range");
  }
  *out_name = module->strings.entries[attr->name_id];
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_find_immediate_attr(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate,
    const loom_named_attr_t** out_attr) {
  *out_immediate = (loom_text_low_asm_immediate_descriptor_t){0};
  *out_attr = NULL;
  IREE_RETURN_IF_ERROR(ctx->low_asm_environment.vtable->immediate_descriptor(
      ctx->low_asm_environment.state, &statement->packet, immediate_index,
      out_immediate));
  for (iree_host_size_t i = 0; i < statement->attributes.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_print_low_asm_attr_name(
        ctx->module, &statement->attributes.entries[i], &attr_name));
    if (iree_string_view_equal(attr_name, out_immediate->field_name)) {
      *out_attr = &statement->attributes.entries[i];
      return iree_ok_status();
    }
  }
  if (out_immediate->has_default_value) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "low asm packet '%.*s' is missing immediate "
      "attribute '%.*s'",
      (int)statement->packet.opcode_key.size, statement->packet.opcode_key.data,
      (int)out_immediate->field_name.size, out_immediate->field_name.data);
}

static bool loom_print_low_asm_immediate_attr_is_default(
    const loom_text_low_asm_immediate_descriptor_t* immediate,
    const loom_named_attr_t* attr) {
  if (!immediate->has_default_value || !attr) {
    return false;
  }
  switch ((loom_attr_kind_t)attr->value.kind) {
    case LOOM_ATTR_I64:
      return attr->value.i64 == immediate->default_value;
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
      return (int64_t)attr->value.raw == immediate->default_value;
    default:
      return false;
  }
}

static bool loom_print_low_asm_should_print_immediate_attr(
    const loom_text_low_asm_immediate_descriptor_t* immediate,
    const loom_named_attr_t* attr) {
  return attr != NULL &&
         !loom_print_low_asm_immediate_attr_is_default(immediate, attr);
}

static iree_status_t loom_print_low_asm_result_types_require_annotation(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement,
    bool* out_required) {
  *out_required = false;
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    loom_value_id_t result = statement->results[i];
    bool annotation_required = false;
    iree_string_view_t diagnostic_detail = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        ctx->low_asm_environment.vtable->result_type_annotation_required(
            ctx->low_asm_environment.state, &statement->packet,
            statement->operands, statement->operand_count, i, ctx->module,
            ctx->module->values.entries[result].type, &annotation_required,
            &diagnostic_detail));
    if (!iree_string_view_is_empty(diagnostic_detail)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' result %u cannot be "
                              "printed: %.*s",
                              (int)statement->packet.opcode_key.size,
                              statement->packet.opcode_key.data, i,
                              (int)diagnostic_detail.size,
                              diagnostic_detail.data);
    }
    *out_required |= annotation_required;
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_region_preflight(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    bool entry_args_declared_by_parent, bool* out_available) {
  *out_available = true;
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (block_index == 0 && block->arg_count != 0 &&
        !entry_args_declared_by_parent) {
      *out_available = false;
      return iree_ok_status();
    }
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
      loom_text_low_asm_statement_t statement = {0};
      IREE_RETURN_IF_ERROR(loom_print_low_asm_describe_operation(
          ctx, descriptor_set, current_op, &statement));
      if (statement.kind == LOOM_TEXT_LOW_ASM_STATEMENT_UNKNOWN) {
        if (!loom_print_low_asm_allows_canonical_structural_op(ctx,
                                                               current_op)) {
          *out_available = false;
          return iree_ok_status();
        }
        IREE_RETURN_IF_ERROR(
            loom_print_low_asm_preflight_canonical_structural_op(
                ctx, descriptor_set, current_op, out_available));
        if (!*out_available) {
          return iree_ok_status();
        }
        continue;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_result_list(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
        ctx, statement->results[i],
        loom_print_field_ref(LOOM_PRINT_FIELD_RESULT, i)));
  }
  if (statement->result_count > 0) {
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "=", false));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_value_list(
    loom_print_context_t* ctx, const loom_value_id_t* values,
    uint16_t value_count, loom_print_field_kind_t field_kind) {
  for (uint16_t i = 0; i < value_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
        ctx, values[i], loom_print_field_ref(field_kind, i)));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_named_immediates(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement,
    uint16_t immediate_start, uint16_t immediate_end) {
  bool has_printed_immediate = false;
  for (uint16_t i = immediate_start; i < immediate_end; ++i) {
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    const loom_named_attr_t* attr = NULL;
    IREE_RETURN_IF_ERROR(loom_print_low_asm_find_immediate_attr(
        ctx, statement, i, &immediate, &attr));
    if (loom_print_low_asm_should_print_immediate_attr(&immediate, attr)) {
      has_printed_immediate = true;
      break;
    }
  }
  if (!has_printed_immediate) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
  iree_host_size_t printed_count = 0;
  for (uint16_t i = immediate_start; i < immediate_end; ++i) {
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    const loom_named_attr_t* attr = NULL;
    IREE_RETURN_IF_ERROR(loom_print_low_asm_find_immediate_attr(
        ctx, statement, i, &immediate, &attr));
    if (!loom_print_low_asm_should_print_immediate_attr(&immediate, attr)) {
      continue;
    }
    if (printed_count > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write(ctx->stream, immediate.spelling));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(
        loom_print_attr(ctx->stream, &attr->value, ctx->module, NULL));
    ++printed_count;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  loom_print_did_write(ctx);
  if (statement->has_immediate_attribute_field) {
    loom_print_report_field(
        ctx,
        loom_print_field_ref(LOOM_PRINT_FIELD_ATTR,
                             statement->immediate_attribute_field_index),
        start, ctx->stream->offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_positional_immediates(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  for (uint16_t i = 0; i < statement->packet.asm_immediate_count; ++i) {
    if (i > 0 || statement->operand_count > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    const loom_named_attr_t* attr = NULL;
    IREE_RETURN_IF_ERROR(loom_print_low_asm_find_immediate_attr(
        ctx, statement, i, &immediate, &attr));
    if (attr == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm packet '%.*s' cannot omit positional immediate '%.*s'",
          (int)statement->packet.opcode_key.size,
          statement->packet.opcode_key.data, (int)immediate.field_name.size,
          immediate.field_name.data);
    }
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    iree_host_size_t start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(
        loom_print_attr(ctx->stream, &attr->value, ctx->module, NULL));
    loom_print_did_write(ctx);
    if (statement->has_immediate_attribute_field) {
      loom_print_report_field(
          ctx,
          loom_print_field_ref(LOOM_PRINT_FIELD_ATTR,
                               statement->immediate_attribute_field_index),
          start, ctx->stream->offset);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_immediates(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  if (statement->packet.immediate_count == 0) {
    return iree_ok_status();
  }
  if (statement->packet.has_named_immediates) {
    return loom_print_low_asm_named_immediates(
        ctx, statement, /*immediate_start=*/0,
        statement->packet.immediate_count);
  }
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_positional_immediates(ctx, statement));
  if (statement->packet.asm_immediate_count >=
      statement->packet.immediate_count) {
    return iree_ok_status();
  }
  return loom_print_low_asm_named_immediates(
      ctx, statement, statement->packet.asm_immediate_count,
      statement->packet.immediate_count);
}

static iree_status_t loom_print_low_asm_result_type_annotation(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  bool annotation_required = false;
  IREE_RETURN_IF_ERROR(loom_print_low_asm_result_types_require_annotation(
      ctx, statement, &annotation_required));
  if (!annotation_required) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(loom_print_type(
        ctx, ctx->module->values.entries[statement->results[i]].type));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_packet(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_low_asm_result_list(ctx, statement));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, statement->packet.mnemonic, false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_immediates(ctx, statement));
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_result_type_annotation(ctx, statement));
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_return(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "return", false));
  return loom_print_low_asm_value_list(ctx, statement->operands,
                                       statement->operand_count,
                                       LOOM_PRINT_FIELD_OPERAND);
}

static iree_status_t loom_print_low_asm_structural_attr_dict(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  if (statement->structural_attribute_count == 0) {
    return iree_ok_status();
  }
  bool wrote_dict = false;
  for (uint8_t i = 0; i < statement->structural_attribute_count; ++i) {
    const loom_text_low_asm_structural_attribute_t* attr =
        &statement->structural_attributes[i];
    if (attr->descriptor &&
        loom_attr_descriptor_elides_value(attr->descriptor, attr->value)) {
      continue;
    }
    if (!wrote_dict) {
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
      wrote_dict = true;
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, attr->name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(loom_print_attr(ctx->stream, attr->value, ctx->module,
                                         attr->descriptor));
  }
  if (wrote_dict) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_named_attr_dict(
    loom_print_context_t* ctx, loom_named_attr_slice_t attrs) {
  if (attrs.count == 0) {
    return iree_ok_status();
  }
  loom_attribute_t attr =
      loom_make_canonical_attr_dict(attrs.entries, attrs.count);
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_attr(ctx->stream, &attr, ctx->module, NULL));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural_result_type(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_result_value_type(ctx, statement->results[0]));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural_operand_types(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", false));
  for (uint16_t i = 0; i < statement->operand_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, statement->operands[i]));
    loom_print_did_write(ctx);
  }
  return loom_print_emit_cstr(ctx, ")", true);
}

static iree_status_t loom_print_low_asm_structural_resource(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "resource", false));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, statement->structural_key, true));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_structural_attr_dict(ctx, statement));
  return loom_print_low_asm_structural_result_type(ctx, statement);
}

static iree_status_t loom_print_low_asm_structural_live_in(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "live_in", false));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, statement->structural_key, true));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_named_attr_dict(ctx, statement->attributes));
  return loom_print_low_asm_structural_result_type(ctx, statement);
}

static iree_status_t loom_print_low_asm_structural_concat(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "concat", false));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ")", true));
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_structural_operand_types(ctx, statement));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "->", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_result_value_type(ctx, statement->results[0]));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural_slice(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "slice", false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "[", true));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      ctx->stream, "%" PRId64, statement->structural_offset));
  loom_print_did_write(ctx);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "]", true));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, statement->operands[0]));
  loom_print_did_write(ctx);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "->", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_result_value_type(ctx, statement->results[0]));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural_copy(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "copy", false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, statement->operands[0]));
  loom_print_did_write(ctx);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "->", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_result_value_type(ctx, statement->results[0]));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural_storage_reserve(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "storage", false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_structural_attr_dict(ctx, statement));
  return loom_print_low_asm_structural_result_type(ctx, statement);
}

static iree_status_t loom_print_low_asm_structural_storage_address(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "storage_address", false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_structural_attr_dict(ctx, statement));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, statement->operands[0]));
  loom_print_did_write(ctx);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "->", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_result_value_type(ctx, statement->results[0]));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural_storage_view(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "storage_view", false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_structural_attr_dict(ctx, statement));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, statement->operands[0]));
  loom_print_did_write(ctx);
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "->", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_result_value_type(ctx, statement->results[0]));
  loom_print_did_write(ctx);
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_structural(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_low_asm_result_list(ctx, statement));
  switch (statement->structural_kind) {
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE:
      return loom_print_low_asm_structural_resource(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN:
      return loom_print_low_asm_structural_live_in(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT:
      return loom_print_low_asm_structural_concat(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE:
      return loom_print_low_asm_structural_slice(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE:
      return loom_print_low_asm_structural_storage_reserve(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS:
      return loom_print_low_asm_structural_storage_address(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW:
      return loom_print_low_asm_structural_storage_view(ctx, statement);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY:
      return loom_print_low_asm_structural_copy(ctx, statement);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low asm structural kind %u",
                              (uint32_t)statement->structural_kind);
  }
}

static iree_status_t loom_print_low_asm_statement(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  ctx->has_previous_token = false;
  ctx->glue_next = false;
  ctx->last_char = 0;

  switch (statement->kind) {
    case LOOM_TEXT_LOW_ASM_STATEMENT_PACKET: {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_packet(ctx, statement));
      break;
    }
    case LOOM_TEXT_LOW_ASM_STATEMENT_RETURN: {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_return(ctx, statement));
      break;
    }
    case LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL: {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_structural(ctx, statement));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low asm statement kind %u",
                              (uint32_t)statement->kind);
  }

  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_LOCATIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_print_location(ctx->stream, ctx->module, statement->location));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    bool entry_args_declared_by_parent) {
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    const bool entry_block = block_index == 0;
    const bool block_args_declared_by_parent =
        entry_block && entry_args_declared_by_parent;
    const bool needs_label =
        loom_print_block_has_label(ctx, block) ||
        loom_print_block_needs_synthetic_label(ctx, region, block) ||
        (block->arg_count != 0 && !block_args_declared_by_parent);
    if (needs_label) {
      IREE_RETURN_IF_ERROR(loom_print_block_label_line_with_options(
          ctx, region, block, !block_args_declared_by_parent));
    }
    const loom_op_t* last_live_op = block->last_op;
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
      if (current_op == last_live_op &&
          loom_print_should_elide_implicit_terminator(region_descriptor,
                                                      current_op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
      loom_text_low_asm_statement_t statement = {0};
      IREE_RETURN_IF_ERROR(loom_print_low_asm_describe_operation(
          ctx, descriptor_set, current_op, &statement));
      if (statement.kind == LOOM_TEXT_LOW_ASM_STATEMENT_UNKNOWN) {
        if (loom_print_low_asm_allows_canonical_structural_op(ctx,
                                                              current_op)) {
          IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
          IREE_RETURN_IF_ERROR(loom_print_op(ctx, current_op));
          continue;
        }
        iree_string_view_t op_name = loom_op_name(ctx->module, current_op);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm region contains unsupported op '%.*s'",
                                (int)op_name.size, op_name.data);
      }
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_print_low_asm_statement(ctx, &statement));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_prepare_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set,
    bool* out_available) {
  (void)region_descriptor;
  *out_descriptor_set = NULL;
  *out_available = true;
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_lookup_descriptor_set(ctx, out_descriptor_set));

  if (!iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, region, *out_descriptor_set, entry_args_declared_by_parent,
        out_available));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_region_with_descriptor_set(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    iree_string_view_t descriptor_set_key, bool entry_args_declared_by_parent,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    bool print_descriptor_set_prefix) {
  const loom_text_low_asm_descriptor_set_t* previous_descriptor_set =
      ctx->low_register_descriptor_set;
  const uint16_t previous_depth = ctx->low_asm_region_depth;
  if (previous_depth == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low asm region nesting exceeds uint16_t range");
  }
  ctx->low_register_descriptor_set = descriptor_set;
  ctx->low_asm_region_depth = (uint16_t)(previous_depth + 1);
  iree_status_t status = iree_ok_status();
  if (print_descriptor_set_prefix) {
    status = loom_print_emit_cstr(ctx, "asm", false);
    if (iree_status_is_ok(status)) {
      status = loom_print_emit_cstr(ctx, "<", true);
    }
    if (iree_status_is_ok(status)) {
      status = loom_print_emit(ctx, descriptor_set_key, true);
    }
    if (iree_status_is_ok(status)) {
      status = loom_print_emit_cstr(ctx, ">", true);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_print_space_if_needed(ctx);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    status = loom_output_stream_write_cstring(ctx->stream, "{ ... }");
  } else if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(ctx->stream, "{\n");
    ++ctx->indent;
    if (iree_status_is_ok(status)) {
      status = loom_print_low_asm_region_body(ctx, region, region_descriptor,
                                              descriptor_set,
                                              entry_args_declared_by_parent);
    }
    --ctx->indent;
    if (iree_status_is_ok(status)) {
      status = loom_print_indent(ctx);
    }
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write_char(ctx->stream, '}');
    }
  }
  ctx->low_register_descriptor_set = previous_descriptor_set;
  ctx->low_asm_region_depth = previous_depth;
  IREE_RETURN_IF_ERROR(status);
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

iree_status_t loom_print_low_asm_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent) {
  const loom_text_low_asm_descriptor_set_t* descriptor_set = NULL;
  bool available = false;
  IREE_RETURN_IF_ERROR(loom_print_low_asm_prepare_region(
      ctx, region, region_descriptor, entry_args_declared_by_parent,
      &descriptor_set, &available));
  if (!available) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "region has no lossless low asm spelling for descriptor set '%.*s'",
        (int)ctx->low_asm_descriptor_set_key.size,
        ctx->low_asm_descriptor_set_key.data);
  }
  return loom_print_low_asm_region_with_descriptor_set(
      ctx, region, region_descriptor, ctx->low_asm_descriptor_set_key,
      entry_args_declared_by_parent, descriptor_set,
      /*print_descriptor_set_prefix=*/true);
}

iree_status_t loom_print_low_asm_optional_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent, bool* out_printed) {
  *out_printed = false;
  const loom_text_low_asm_descriptor_set_t* descriptor_set = NULL;
  bool available = false;
  IREE_RETURN_IF_ERROR(loom_print_low_asm_prepare_region(
      ctx, region, region_descriptor, entry_args_declared_by_parent,
      &descriptor_set, &available));
  if (!available) {
    return iree_ok_status();
  }
  *out_printed = true;
  return loom_print_low_asm_region_with_descriptor_set(
      ctx, region, region_descriptor, ctx->low_asm_descriptor_set_key,
      entry_args_declared_by_parent, descriptor_set,
      /*print_descriptor_set_prefix=*/ctx->low_asm_region_depth == 0);
}
