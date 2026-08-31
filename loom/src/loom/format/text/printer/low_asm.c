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

typedef enum loom_print_low_asm_preflight_failure_kind_e {
  LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_NONE = 0,
  LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_ENTRY_ARGS = 1,
  LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_OPERATION = 2,
} loom_print_low_asm_preflight_failure_kind_t;

typedef enum loom_print_low_asm_prefix_e {
  // Nested region inheriting its parent's representation contract.
  LOOM_PRINT_LOW_ASM_PREFIX_NONE = 0,
  // Function body whose representation contract is carried by the function.
  LOOM_PRINT_LOW_ASM_PREFIX_MARKER = 1,
} loom_print_low_asm_prefix_t;

typedef struct loom_print_low_asm_preflight_failure_t {
  // Reason the region cannot be printed as lossless low asm.
  loom_print_low_asm_preflight_failure_kind_t kind;
  // Region block ordinal containing the failure.
  uint16_t block_index;
  // Canonical Loom operation name such as `low.op`.
  iree_string_view_t operation_name;
  // Descriptor key when the operation is `low.op` or `low.const`.
  iree_string_view_t packet_descriptor_key;
  // Number of SSA results on the failed operation.
  uint16_t result_count;
  // Number of SSA operands on the failed operation.
  uint16_t operand_count;
} loom_print_low_asm_preflight_failure_t;

static bool loom_print_low_asm_preserves_source(
    const loom_print_context_t* ctx) {
  return iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_PRESERVE_LOW_ASM);
}

static bool loom_print_low_asm_source_is_marked(const loom_region_t* region) {
  return region && iree_any_bit_set(region->source_flags,
                                    LOOM_REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM);
}

bool loom_print_low_asm_is_requested(loom_print_context_t* ctx,
                                     const loom_region_t* region) {
  if (ctx->low_repr.descriptor_set == NULL) return false;
  if (ctx->low_asm_region_depth != 0) return true;
  if (loom_print_low_asm_preserves_source(ctx)) {
    return loom_print_low_asm_source_is_marked(region);
  }
  return iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_PREFER_LOW_ASM |
                                          LOOM_TEXT_PRINT_REQUIRE_LOW_ASM);
}

bool loom_print_low_asm_uses_marker(loom_print_context_t* ctx,
                                    const loom_region_t* region) {
  if (loom_print_low_asm_preserves_source(ctx)) {
    return loom_print_low_asm_source_is_marked(region);
  }
  return ctx->low_asm_region_depth == 0;
}

static bool loom_print_low_asm_is_required(loom_print_context_t* ctx,
                                           const loom_region_t* region) {
  if (loom_print_low_asm_preserves_source(ctx)) {
    return ctx->low_asm_region_depth != 0 ||
           loom_print_low_asm_source_is_marked(region);
  }
  return iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_REQUIRE_LOW_ASM);
}

static bool loom_print_low_asm_allows_canonical_op(loom_print_context_t* ctx,
                                                   const loom_op_t* op) {
  if (iree_any_bit_set(loom_op_effective_traits(ctx->module, op),
                       LOOM_TRAIT_HINT | LOOM_TRAIT_COMPILE_TIME_ONLY)) {
    return true;
  }
  iree_string_view_t op_name = loom_op_name(ctx->module, op);
  return iree_string_view_equal(op_name, IREE_SV("low.br")) ||
         iree_string_view_equal(op_name, IREE_SV("low.cond_br")) ||
         iree_string_view_equal(op_name, IREE_SV("low.func.call")) ||
         iree_string_view_equal(op_name, IREE_SV("low.reload")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.condition")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.yield")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.if")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.for")) ||
         iree_string_view_equal(op_name, IREE_SV("low.scf.while")) ||
         iree_string_view_equal(op_name, IREE_SV("low.spill"));
}

static iree_status_t loom_print_low_asm_region_preflight(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    bool entry_args_declared_by_parent,
    loom_print_low_asm_preflight_failure_t* out_failure, bool* out_available);

static iree_string_view_t loom_print_low_asm_packet_descriptor_key(
    loom_print_context_t* ctx,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    const loom_op_t* op) {
  iree_string_view_t op_name = loom_op_name(ctx->module, op);
  if (!iree_string_view_equal(op_name, IREE_SV("low.op")) &&
      !iree_string_view_equal(op_name, IREE_SV("low.const"))) {
    return iree_string_view_empty();
  }
  if (op->attribute_count == 0) {
    return iree_string_view_empty();
  }
  const loom_attribute_t descriptor_attr = loom_op_const_attrs(op)[0];
  if (descriptor_attr.kind != LOOM_ATTR_SCOPED_ENUM) {
    return iree_string_view_empty();
  }
  return loom_low_repr_descriptor_key(
      &ctx->low_asm_environment.low_repr, descriptor_set,
      loom_attr_as_scoped_enum(descriptor_attr));
}

static void loom_print_low_asm_record_entry_args_failure(
    loom_print_low_asm_preflight_failure_t* out_failure, uint16_t block_index) {
  if (out_failure->kind != LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_NONE) {
    return;
  }
  *out_failure = (loom_print_low_asm_preflight_failure_t){
      .kind = LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_ENTRY_ARGS,
      .block_index = block_index,
  };
}

static void loom_print_low_asm_record_operation_failure(
    loom_print_context_t* ctx,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    loom_print_low_asm_preflight_failure_t* out_failure, uint16_t block_index,
    const loom_op_t* op) {
  if (out_failure->kind != LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_NONE) {
    return;
  }
  *out_failure = (loom_print_low_asm_preflight_failure_t){
      .kind = LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_OPERATION,
      .block_index = block_index,
      .operation_name = loom_op_name(ctx->module, op),
      .packet_descriptor_key =
          loom_print_low_asm_packet_descriptor_key(ctx, descriptor_set, op),
      .result_count = op->result_count,
      .operand_count = op->operand_count,
  };
}

static iree_status_t loom_print_low_asm_preflight_canonical_structural_op(
    loom_print_context_t* ctx,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    uint16_t block_index, const loom_op_t* op,
    loom_print_low_asm_preflight_failure_t* out_failure, bool* out_available) {
  iree_string_view_t op_name = loom_op_name(ctx->module, op);
  if (iree_string_view_equal(op_name, IREE_SV("low.scf.if"))) {
    if (op->region_count < 1 || loom_op_regions(op)[0] == NULL) {
      *out_available = false;
      loom_print_low_asm_record_operation_failure(ctx, descriptor_set,
                                                  out_failure, block_index, op);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, loom_op_regions(op)[0], descriptor_set,
        /*entry_args_declared_by_parent=*/false, out_failure, out_available));
    if (!*out_available) {
      return iree_ok_status();
    }
    if (op->region_count > 1 && loom_op_regions(op)[1] != NULL) {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
          ctx, loom_op_regions(op)[1], descriptor_set,
          /*entry_args_declared_by_parent=*/false, out_failure, out_available));
    }
  } else if (iree_string_view_equal(op_name, IREE_SV("low.scf.for"))) {
    if (op->region_count < 1 || loom_op_regions(op)[0] == NULL) {
      *out_available = false;
      loom_print_low_asm_record_operation_failure(ctx, descriptor_set,
                                                  out_failure, block_index, op);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, loom_op_regions(op)[0], descriptor_set,
        /*entry_args_declared_by_parent=*/true, out_failure, out_available));
  } else if (iree_string_view_equal(op_name, IREE_SV("low.scf.while"))) {
    if (op->region_count < 2 || loom_op_regions(op)[0] == NULL ||
        loom_op_regions(op)[1] == NULL) {
      *out_available = false;
      loom_print_low_asm_record_operation_failure(ctx, descriptor_set,
                                                  out_failure, block_index, op);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, loom_op_regions(op)[0], descriptor_set,
        /*entry_args_declared_by_parent=*/true, out_failure, out_available));
    if (!*out_available) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, loom_op_regions(op)[1], descriptor_set,
        /*entry_args_declared_by_parent=*/true, out_failure, out_available));
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
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "low asm packet '%.*s' is missing immediate "
                          "attribute '%.*s'",
                          (int)statement->packet.descriptor_key.size,
                          statement->packet.descriptor_key.data,
                          (int)out_immediate->field_name.size,
                          out_immediate->field_name.data);
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
            loom_module_value_type(ctx->module, result), &annotation_required,
            &diagnostic_detail));
    if (!iree_string_view_is_empty(diagnostic_detail)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' result %u cannot be "
                              "printed: %.*s",
                              (int)statement->packet.descriptor_key.size,
                              statement->packet.descriptor_key.data, i,
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
    bool entry_args_declared_by_parent,
    loom_print_low_asm_preflight_failure_t* out_failure, bool* out_available) {
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
      loom_print_low_asm_record_entry_args_failure(out_failure, block_index);
      return iree_ok_status();
    }
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
      loom_text_low_asm_statement_t statement = {0};
      IREE_RETURN_IF_ERROR(loom_print_low_asm_describe_operation(
          ctx, descriptor_set, current_op, &statement));
      if (statement.kind == LOOM_TEXT_LOW_ASM_STATEMENT_UNKNOWN) {
        if (!loom_print_low_asm_allows_canonical_op(ctx, current_op)) {
          *out_available = false;
          loom_print_low_asm_record_operation_failure(
              ctx, descriptor_set, out_failure, block_index, current_op);
          return iree_ok_status();
        }
        IREE_RETURN_IF_ERROR(
            loom_print_low_asm_preflight_canonical_structural_op(
                ctx, descriptor_set, block_index, current_op, out_failure,
                out_available));
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

static iree_status_t loom_print_low_asm_value_span(
    loom_print_context_t* ctx, const loom_value_id_t* values,
    uint16_t value_start, uint16_t value_count,
    loom_print_field_kind_t field_kind) {
  for (uint16_t i = 0; i < value_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    const uint16_t value_index = value_start + i;
    IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
        ctx, values[value_index],
        loom_print_field_ref(field_kind, value_index)));
  }
  return iree_ok_status();
}

static void loom_print_low_asm_operand_segment_delimiters(
    loom_text_low_asm_operand_segment_delimiter_t delimiter,
    const char** out_open, const char** out_close) {
  switch (delimiter) {
    case LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_ANGLE:
      *out_open = "<";
      *out_close = ">";
      return;
    case LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_SQUARE:
      *out_open = "[";
      *out_close = "]";
      return;
    case LOOM_TEXT_LOW_ASM_OPERAND_SEGMENT_DELIMITER_PAREN:
      *out_open = "(";
      *out_close = ")";
      return;
    default:
      IREE_ASSERT_UNREACHABLE("validated low asm operand segment delimiter");
      *out_open = "";
      *out_close = "";
      return;
  }
}

static iree_status_t loom_print_low_asm_operand_segments(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  uint16_t operand_offset = 0;
  for (uint16_t segment_index = 0;
       segment_index < statement->packet.operand_segment_count;
       ++segment_index) {
    loom_text_low_asm_operand_segment_descriptor_t segment = {0};
    IREE_RETURN_IF_ERROR(
        ctx->low_asm_environment.vtable->operand_segment_descriptor(
            ctx->low_asm_environment.state, &statement->packet, segment_index,
            &segment));
    const uint16_t segment_operand_count =
        segment.is_variadic ? statement->operand_count - operand_offset
                            : segment.fixed_operand_count;
    const char* open = NULL;
    const char* close = NULL;
    loom_print_low_asm_operand_segment_delimiters(segment.delimiter, &open,
                                                  &close);
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, open, true));
    loom_print_set_glue(ctx);
    IREE_RETURN_IF_ERROR(loom_print_low_asm_value_span(
        ctx, statement->operands, operand_offset, segment_operand_count,
        LOOM_PRINT_FIELD_OPERAND));
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, close, true));
    operand_offset += segment_operand_count;
  }
  IREE_ASSERT_EQ(operand_offset, statement->operand_count);
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
    IREE_RETURN_IF_ERROR(loom_print_attr(ctx, &attr->value, NULL));
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
          (int)statement->packet.descriptor_key.size,
          statement->packet.descriptor_key.data, (int)immediate.field_name.size,
          immediate.field_name.data);
    }
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    iree_host_size_t start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(loom_print_attr(ctx, &attr->value, NULL));
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
        ctx, loom_module_value_type(ctx->module, statement->results[i])));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_packet(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_low_asm_result_list(ctx, statement));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, statement->packet.mnemonic, false));
  if (statement->packet.operand_segment_count != 0) {
    IREE_RETURN_IF_ERROR(loom_print_low_asm_operand_segments(ctx, statement));
  } else {
    IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(
        ctx, statement->operands, statement->operand_count,
        LOOM_PRINT_FIELD_OPERAND));
  }
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
    IREE_RETURN_IF_ERROR(loom_print_attr(ctx, attr->value, attr->descriptor));
  }
  if (wrote_dict) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_named_attr_dict(
    loom_print_context_t* ctx,
    loom_text_low_asm_structural_build_flags_t build_flags,
    loom_named_attr_slice_t attrs) {
  if (!iree_any_bit_set(
          build_flags,
          LOOM_TEXT_LOW_ASM_STRUCTURAL_BUILD_FLAG_HAS_ATTRIBUTES)) {
    return iree_ok_status();
  }
  loom_attribute_t attr =
      loom_make_canonical_attr_dict(attrs.entries, attrs.count);
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_print_attr(ctx, &attr, NULL));
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
  if (statement->operand_count != 0) {
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "extent", false));
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
    IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(
        ctx, statement->operands, statement->operand_count,
        LOOM_PRINT_FIELD_OPERAND));
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ")", true));
  }
  IREE_RETURN_IF_ERROR(loom_print_low_asm_structural_attr_dict(ctx, statement));
  return loom_print_low_asm_structural_result_type(ctx, statement);
}

static iree_status_t loom_print_low_asm_structural_live_in(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "live_in", false));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, statement->structural_key, true));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_named_attr_dict(
      ctx, statement->structural_build_flags, statement->attributes));
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

static iree_status_t loom_print_low_asm_structural_transfer(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(
      ctx,
      statement->structural_kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_MOVE ? "move"
                                                                      : "copy",
      false));
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
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_MOVE:
      return loom_print_low_asm_structural_transfer(ctx, statement);
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
      if (iree_any_bit_set(block->flags, LOOM_BLOCK_FLAG_LEADING_BLANK_LINE)) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
      }
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
      if (iree_any_bit_set(current_op->flags,
                           LOOM_OP_FLAG_LEADING_BLANK_LINE)) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
      }
      IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
      loom_text_low_asm_statement_t statement = {0};
      IREE_RETURN_IF_ERROR(loom_print_low_asm_describe_operation(
          ctx, descriptor_set, current_op, &statement));
      if (statement.kind == LOOM_TEXT_LOW_ASM_STATEMENT_UNKNOWN) {
        if (loom_print_low_asm_allows_canonical_op(ctx, current_op)) {
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
    loom_text_low_repr_context_t* out_low_repr,
    loom_print_low_asm_preflight_failure_t* out_failure, bool* out_available) {
  (void)region_descriptor;
  *out_failure = (loom_print_low_asm_preflight_failure_t){0};
  *out_available = true;
  *out_low_repr = ctx->low_repr;

  if (!iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, region, out_low_repr->descriptor_set,
        entry_args_declared_by_parent, out_failure, out_available));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_make_unavailable_status(
    iree_string_view_t repr_contract,
    const loom_print_low_asm_preflight_failure_t* failure) {
  switch (failure->kind) {
    case LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_ENTRY_ARGS:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "region has no lossless low asm spelling for representation "
          "contract '%.*s': entry block %u has arguments that are not declared "
          "by the parent",
          (int)repr_contract.size, repr_contract.data, failure->block_index);
    case LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_OPERATION:
      if (!iree_string_view_is_empty(failure->packet_descriptor_key)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "region has no lossless low asm spelling for representation "
            "contract "
            "'%.*s': operation '%.*s' with packet descriptor '%.*s' in block "
            "%u "
            "has no matching low asm packet form (%u results, %u operands)",
            (int)repr_contract.size, repr_contract.data,
            (int)failure->operation_name.size, failure->operation_name.data,
            (int)failure->packet_descriptor_key.size,
            failure->packet_descriptor_key.data, failure->block_index,
            failure->result_count, failure->operand_count);
      }
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "region has no lossless low asm spelling for representation "
          "contract '%.*s': operation '%.*s' in block %u has no matching low "
          "asm form (%u results, %u operands)",
          (int)repr_contract.size, repr_contract.data,
          (int)failure->operation_name.size, failure->operation_name.data,
          failure->block_index, failure->result_count, failure->operand_count);
    case LOOM_PRINT_LOW_ASM_PREFLIGHT_FAILURE_NONE:
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "region has no lossless low asm spelling for representation contract "
          "'%.*s'",
          (int)repr_contract.size, repr_contract.data);
  }
}

static iree_status_t loom_print_low_asm_region_with_repr(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent, loom_text_low_repr_context_t low_repr,
    loom_print_low_asm_prefix_t prefix) {
  const loom_text_low_repr_context_t previous_low_repr = ctx->low_repr;
  const uint16_t previous_depth = ctx->low_asm_region_depth;
  if (previous_depth == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low asm region nesting exceeds uint16_t range");
  }
  ctx->low_repr = low_repr;
  ctx->low_asm_region_depth = (uint16_t)(previous_depth + 1);
  iree_status_t status = iree_ok_status();
  switch (prefix) {
    case LOOM_PRINT_LOW_ASM_PREFIX_NONE:
      break;
    case LOOM_PRINT_LOW_ASM_PREFIX_MARKER:
      status = loom_print_emit_cstr(ctx, "asm", false);
      break;
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
                                              low_repr.descriptor_set,
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
  ctx->low_repr = previous_low_repr;
  ctx->low_asm_region_depth = previous_depth;
  IREE_RETURN_IF_ERROR(status);
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

iree_status_t loom_print_low_asm_optional_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent, bool* out_printed) {
  *out_printed = false;
  loom_text_low_repr_context_t low_repr = {0};
  loom_print_low_asm_preflight_failure_t failure = {0};
  bool available = false;
  iree_status_t status = loom_print_low_asm_prepare_region(
      ctx, region, region_descriptor, entry_args_declared_by_parent, &low_repr,
      &failure, &available);
  if (!iree_status_is_ok(status)) {
    // A preferred spelling is opportunistic. Descriptor-backed description
    // rejects malformed canonical packet shapes with a user-input status; the
    // generic printer can still represent those operations losslessly for
    // diagnostic fixtures. Explicit or required asm keeps the strict failure.
    if (!loom_print_low_asm_is_required(ctx, region) &&
        (iree_status_is_invalid_argument(status) ||
         iree_status_is_out_of_range(status))) {
      iree_status_free(status);
      return iree_ok_status();
    }
    return status;
  }
  if (!available) {
    if (loom_print_low_asm_is_required(ctx, region)) {
      return loom_print_low_asm_make_unavailable_status(low_repr.contract_key,
                                                        &failure);
    }
    return iree_ok_status();
  }
  *out_printed = true;
  const loom_print_low_asm_prefix_t prefix =
      loom_print_low_asm_uses_marker(ctx, region)
          ? LOOM_PRINT_LOW_ASM_PREFIX_MARKER
          : LOOM_PRINT_LOW_ASM_PREFIX_NONE;
  return loom_print_low_asm_region_with_repr(ctx, region, region_descriptor,
                                             entry_args_declared_by_parent,
                                             low_repr, prefix);
}
