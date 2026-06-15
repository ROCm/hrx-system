// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/parser.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/aliases.h"
#include "loom/format/text/parser/context.h"
#include "loom/format/text/parser/diagnostics.h"
#include "loom/format/text/parser/format.h"
#include "loom/format/text/parser/locations.h"
#include "loom/format/text/parser/low_asm.h"
#include "loom/format/text/parser/pipeline.h"
#include "loom/format/text/parser/regions.h"
#include "loom/format/text/parser/scope.h"
#include "loom/format/text/parser/types.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Alias table
//===----------------------------------------------------------------------===//

iree_status_t loom_alias_table_add(loom_alias_table_t* table,
                                   iree_arena_allocator_t* arena,
                                   iree_string_view_t name,
                                   uint16_t encoding_id) {
  if (table->count >= table->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, table->count, 8, sizeof(loom_alias_entry_t), &table->capacity,
        (void**)&table->entries));
  }
  table->entries[table->count].name = name;
  table->entries[table->count].encoding_id = encoding_id;
  ++table->count;
  return iree_ok_status();
}

uint16_t loom_alias_table_lookup(const loom_alias_table_t* table,
                                 iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    if (iree_string_view_equal(table->entries[i].name, name)) {
      return table->entries[i].encoding_id;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Error recovery
//===----------------------------------------------------------------------===//

void loom_parser_sync_to_newline(loom_parser_t* parser) {
  // The tokenizer doesn't expose "at start of line" directly, but
  // we can advance until we find an op-starting token or EOF.
  // A simple heuristic: advance until we see a token at column 1 or
  // whose kind is SSA_VALUE (result name), OP_NAME, LOOM_TOKEN_ERROR
  // (lexical error at the next sibling op), BLOCK_LABEL, RBRACE (end of
  // region), or EOF.
  for (;;) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    if (token.kind == LOOM_TOKEN_EOF) {
      break;
    }
    if (token.kind == LOOM_TOKEN_RBRACE) {
      break;
    }
    // If token is at column 1 (or 3+ for indented ops), it's likely
    // the start of a new op. We use the heuristic that SSA_VALUE at
    // position <= the next line start means we've synced.
    if (token.kind == LOOM_TOKEN_SSA_VALUE && token.column <= 2) {
      break;
    }
    if (token.kind == LOOM_TOKEN_OP_NAME && token.column <= 2) {
      break;
    }
    if (token.kind == LOOM_TOKEN_ERROR && token.column <= 2) {
      break;
    }
    if (token.kind == LOOM_TOKEN_BLOCK_LABEL) {
      break;
    }
    loom_tokenizer_next(&parser->tokenizer);
  }
}

void loom_parser_sync_to_brace(loom_parser_t* parser) {
  int depth = 1;
  for (;;) {
    loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
    if (token.kind == LOOM_TOKEN_EOF) {
      break;
    }
    if (token.kind == LOOM_TOKEN_LBRACE) {
      ++depth;
    }
    if (token.kind == LOOM_TOKEN_RBRACE) {
      --depth;
      if (depth <= 0) {
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Keyword matching
//===----------------------------------------------------------------------===//

// Returns the token kind that a keyword would produce as a standalone
// token, or LOOM_TOKEN_BARE_IDENT for text keywords (import, where, etc.).
// Used by the optional group presence check to peek for punctuation
// keywords like ',' or '->' without assuming BARE_IDENT.
loom_token_kind_t loom_keyword_token_kind(uint16_t keyword_id) {
  switch (keyword_id) {
    case LOOM_KW_COMMA:
      return LOOM_TOKEN_COMMA;
    case LOOM_KW_COLON:
      return LOOM_TOKEN_COLON;
    case LOOM_KW_ARROW:
      return LOOM_TOKEN_ARROW;
    case LOOM_KW_EQUALS:
      return LOOM_TOKEN_EQUALS;
    case LOOM_KW_LPAREN:
      return LOOM_TOKEN_LPAREN;
    case LOOM_KW_RPAREN:
      return LOOM_TOKEN_RPAREN;
    case LOOM_KW_LBRACKET:
      return LOOM_TOKEN_LBRACKET;
    case LOOM_KW_RBRACKET:
      return LOOM_TOKEN_RBRACKET;
    case LOOM_KW_LBRACE:
      return LOOM_TOKEN_LBRACE;
    case LOOM_KW_RBRACE:
      return LOOM_TOKEN_RBRACE;
    default:
      return LOOM_TOKEN_BARE_IDENT;
  }
}

iree_status_t loom_parse_keyword(loom_parser_t* parser, uint16_t keyword_id) {
  switch (keyword_id) {
    case LOOM_KW_COMMA:
      return loom_parser_expect(parser, LOOM_TOKEN_COMMA, NULL);
    case LOOM_KW_COLON:
      return loom_parser_expect(parser, LOOM_TOKEN_COLON, NULL);
    case LOOM_KW_ARROW:
      return loom_parser_expect(parser, LOOM_TOKEN_ARROW, NULL);
    case LOOM_KW_EQUALS:
      return loom_parser_expect(parser, LOOM_TOKEN_EQUALS, NULL);
    case LOOM_KW_LPAREN:
      return loom_parser_expect(parser, LOOM_TOKEN_LPAREN, NULL);
    case LOOM_KW_RPAREN:
      return loom_parser_expect(parser, LOOM_TOKEN_RPAREN, NULL);
    case LOOM_KW_LBRACKET:
      return loom_parser_expect(parser, LOOM_TOKEN_LBRACKET, NULL);
    case LOOM_KW_RBRACKET:
      return loom_parser_expect(parser, LOOM_TOKEN_RBRACKET, NULL);
    case LOOM_KW_LBRACE:
      return loom_parser_expect(parser, LOOM_TOKEN_LBRACE, NULL);
    case LOOM_KW_RBRACE:
      return loom_parser_expect(parser, LOOM_TOKEN_RBRACE, NULL);
    default: {
      // Text keyword; match as BARE_IDENT.
      loom_bstring_t keyword_bstring =
          loom_keyword_bstring((loom_keyword_id_t)keyword_id);
      if (!keyword_bstring) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "keyword id %u is out of range", keyword_id);
      }
      iree_string_view_t expected = loom_bstring_view(keyword_bstring);
      if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer, expected)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek, expected);
      }
      return iree_ok_status();
    }
  }
}

//===----------------------------------------------------------------------===//
// Op finalization from accumulator to loom_op_t.
//===----------------------------------------------------------------------===//

static void loom_parser_set_region_parent_op(loom_region_t* region,
                                             loom_op_t* parent_op) {
  if (!region) {
    return;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) { op->parent_op = parent_op; }
  }
}

static loom_block_t* loom_parser_find_block_by_label(
    const loom_parser_t* parser, loom_region_t* region,
    iree_string_view_t label) {
  if (!region) {
    return NULL;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    if (block->label_id == LOOM_STRING_ID_INVALID ||
        block->label_id >= parser->module->strings.count) {
      continue;
    }
    if (iree_string_view_equal(parser->module->strings.entries[block->label_id],
                               label)) {
      return block;
    }
  }
  return NULL;
}

static iree_status_t loom_parser_add_pending_successor_ref(
    loom_parser_t* parser, loom_region_t* region, loom_op_t* op,
    uint8_t successor_index, loom_token_t label_token) {
  loom_parser_pending_successor_refs_t* pending =
      &parser->pending_successor_refs;
  if (pending->count >= pending->capacity) {
    iree_host_size_t capacity = pending->capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, pending->count, pending->count + 1,
        sizeof(loom_parser_pending_successor_ref_t), &capacity,
        (void**)&pending->entries));
    pending->capacity = capacity;
  }
  pending->entries[pending->count++] = (loom_parser_pending_successor_ref_t){
      .region = region,
      .op = op,
      .label_token = label_token,
      .successor_index = successor_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_parser_resolve_pending_successor_refs(
    loom_parser_t* parser, loom_region_t* region,
    iree_host_size_t pending_start) {
  loom_parser_pending_successor_refs_t* pending =
      &parser->pending_successor_refs;
  for (iree_host_size_t i = pending_start; i < pending->count; ++i) {
    loom_parser_pending_successor_ref_t ref = pending->entries[i];
    if (ref.region != region) {
      continue;
    }
    loom_block_t* target =
        loom_parser_find_block_by_label(parser, region, ref.label_token.text);
    if (!target) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(ref.label_token.text),
      };
      return loom_parser_emit(parser, LOOM_ERR_PARSE_032, params,
                              IREE_ARRAYSIZE(params), ref.label_token);
    }
    loom_op_successors(ref.op)[ref.successor_index] = target;
  }
  pending->count = pending_start;
  return iree_ok_status();
}

static bool loom_parser_type_is_known(loom_type_t type) {
  return type.header != 0;
}

static bool loom_parser_resolve_operand_segment_value(
    const loom_parsed_op_t* parsed, uint8_t field_index,
    loom_value_id_t* out_value_id) {
  if (field_index >= parsed->operand_segment_count) return false;
  uint32_t start = 0;
  for (uint8_t i = 0; i < field_index; ++i) {
    start += parsed->operand_segment_counts[i];
  }
  if (start >= parsed->operand_count ||
      parsed->operand_segment_counts[field_index] == 0) {
    return false;
  }
  *out_value_id = parsed->operand_ids[start];
  return true;
}

static bool loom_parser_resolve_field_type(const loom_module_t* module,
                                           const loom_op_vtable_t* vtable,
                                           const loom_parsed_op_t* parsed,
                                           loom_field_ref_t ref,
                                           loom_type_t* out_type) {
  const uint8_t index = LOOM_FIELD_REF_INDEX(ref);
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  switch (LOOM_FIELD_REF_CATEGORY(ref)) {
    case LOOM_FIELD_OPERAND:
      if (loom_op_vtable_has_segmented_operands(vtable)) {
        if (!loom_parser_resolve_operand_segment_value(parsed, index,
                                                       &value_id)) {
          return false;
        }
        break;
      }
      if (index >= parsed->operand_count) {
        return false;
      }
      value_id = parsed->operand_ids[index];
      break;
    case LOOM_FIELD_RESULT:
      if (index >= parsed->result_count) {
        return false;
      }
      value_id = parsed->result_ids[index];
      break;
    default:
      return false;
  }
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_parser_type_is_known(type)) {
    return false;
  }
  *out_type = type;
  return true;
}

static iree_status_t loom_parser_try_infer_same_type_result(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_parsed_op_t* parsed, uint16_t result_index) {
  const loom_field_ref_t target_ref =
      LOOM_FIELD_REF(LOOM_FIELD_RESULT, result_index);
  for (uint16_t constraint_index = 0;
       constraint_index < vtable->constraint_count; ++constraint_index) {
    const loom_constraint_t* constraint =
        &vtable->constraints[constraint_index];
    if (constraint->relation != LOOM_RELATION_PAIRWISE_EQ ||
        constraint->property != LOOM_PROPERTY_TYPE) {
      continue;
    }

    bool mentions_target = false;
    for (uint8_t arg_index = 0; arg_index < constraint->arg_count;
         ++arg_index) {
      if (constraint->args[arg_index] == target_ref) {
        mentions_target = true;
        break;
      }
    }
    if (!mentions_target) {
      continue;
    }

    for (uint8_t arg_index = 0; arg_index < constraint->arg_count;
         ++arg_index) {
      const loom_field_ref_t source_ref = constraint->args[arg_index];
      if (source_ref == target_ref) {
        continue;
      }
      loom_type_t inferred_type = {0};
      if (!loom_parser_resolve_field_type(parser->module, vtable, parsed,
                                          source_ref, &inferred_type)) {
        continue;
      }
      return loom_module_set_value_type(
          parser->module, parsed->result_ids[result_index], inferred_type);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_finalize_op(
    loom_parser_t* parser, loom_op_kind_t kind, const loom_op_vtable_t* vtable,
    loom_parsed_op_t* parsed, loom_location_id_t location, loom_op_t** out_op) {
  // Allocate and insert the op. The attribute storage must use the
  // vtable's declared count, not the parser's high-water-mark. The
  // vtable defines the storage layout: the printer and typed accessors
  // index by field_index up to vtable->attribute_count. If the parser
  // only set index 0 of a 5-attribute op, the op still needs 5 slots
  // allocated. The allocation is zero-filled, so unset optional
  // attributes at higher indices read as zero (LOOM_ATTR_NONE).
  loom_op_t* op = NULL;
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (operand_segment_count > 0) {
    if (parsed->successor_count > 0) {
      IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op_with_successors(
          &parser->builder, kind, parsed->operand_count,
          parsed->operand_segment_counts, parsed->operand_segment_count,
          parsed->result_count, parsed->successor_count, parsed->region_count,
          parsed->tied_result_count, vtable->attribute_count, location, &op));
    } else {
      IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
          &parser->builder, kind, parsed->operand_count,
          parsed->operand_segment_counts, parsed->operand_segment_count,
          parsed->result_count, parsed->region_count, parsed->tied_result_count,
          vtable->attribute_count, location, &op));
    }
  } else if (parsed->successor_count > 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
        &parser->builder, kind, parsed->operand_count, parsed->result_count,
        parsed->successor_count, parsed->region_count,
        parsed->tied_result_count, vtable->attribute_count, location, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        &parser->builder, kind, parsed->operand_count, parsed->result_count,
        parsed->region_count, parsed->tied_result_count,
        vtable->attribute_count, location, &op));
  }

  // Copy operands.
  if (parsed->operand_count > 0) {
    memcpy(loom_op_operands(op), parsed->operand_ids,
           parsed->operand_count * sizeof(loom_value_id_t));
  }

  // Copy successors and record labels that need region-end resolution. Forward
  // references intentionally remain NULL until the enclosing region has parsed
  // every block label.
  if (parsed->successor_count > 0) {
    memcpy(loom_op_successors(op), parsed->successors,
           parsed->successor_count * sizeof(loom_block_t*));
    loom_region_t* successor_region =
        op->parent_block ? op->parent_block->parent_region : NULL;
    if (successor_region) {
      successor_region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    }
    for (uint8_t i = 0; i < parsed->successor_count; ++i) {
      if (parsed->successor_label_tokens[i].kind == LOOM_TOKEN_BLOCK_LABEL) {
        IREE_RETURN_IF_ERROR(loom_parser_add_pending_successor_ref(
            parser, successor_region, op, i,
            parsed->successor_label_tokens[i]));
      }
    }
  }

  // Fill in implicit result types from result descriptors and equality
  // constraints. When the format doesn't include a RESULT_TYPE element for a
  // result, the value's type is NONE. SameType constraints can recover
  // pass-through result types from typed operands, and fixed-type constraints
  // (e.g., I1 for comparison results) provide concrete singleton types.
  if (vtable->result_descriptors) {
    for (uint16_t i = 0;
         i < parsed->result_count && i < vtable->fixed_result_count; ++i) {
      loom_value_t* value =
          &parser->module->values.entries[parsed->result_ids[i]];
      if (value->type.header != 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_parser_try_infer_same_type_result(parser, vtable, parsed, i));
      if (value->type.header != 0) {
        continue;
      }
      if (vtable->result_descriptors[i].type_constraint ==
          LOOM_TYPE_CONSTRAINT_I1) {
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_type(parser->module, parsed->result_ids[i],
                                       loom_type_scalar(LOOM_SCALAR_TYPE_I1)));
      }
    }
  }

  // Copy result value IDs into the op. Body-op results become visible in the
  // current lexical scope here; symbol-definition results are signature values
  // only and must not leak into the module scope or their function body.
  // Values already have their types and names set during LHS parsing and the
  // format walk.
  loom_value_id_t* result_slots = loom_op_results(op);
  for (uint16_t i = 0; i < parsed->result_count; ++i) {
    loom_value_id_t value_id = parsed->result_ids[i];
    result_slots[i] = value_id;
    if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
      continue;
    }
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID &&
        name_id < parser->module->strings.count) {
      bool duplicate = false;
      IREE_RETURN_IF_ERROR(loom_parser_scope_define(
          parser->scope, &parser->parser_arena, name_id, value_id, &duplicate));
      if (duplicate) {
        return loom_parser_emit_duplicate_value_name(
            parser, parsed->result_name_tokens[i]);
      }
    }
  }

  // Copy regions.
  if (parsed->region_count > 0) {
    memcpy(loom_op_regions(op), parsed->regions,
           parsed->region_count * sizeof(loom_region_t*));
    for (uint8_t i = 0; i < parsed->region_count; ++i) {
      loom_parser_set_region_parent_op(parsed->regions[i], op);
    }
  }

  // Copy tied results.
  if (parsed->tied_result_count > 0) {
    memcpy(loom_op_tied_results(op), parsed->tied_results,
           parsed->tied_result_count * sizeof(loom_tied_result_t));
  }

  // Copy attributes.
  if (parsed->attribute_count > 0) {
    memcpy(loom_op_attrs(op), parsed->attributes,
           parsed->attribute_count * sizeof(loom_attribute_t));
  }

  // Set instance flags.
  op->instance_flags = parsed->instance_flags;

  // Attribute- or instance-flag-dependent traits become part of the op at the
  // parse construction boundary. Later use/def rebuilds must not recompute
  // semantic traits.
  loom_op_refresh_effective_traits(parser->module, op);

  // Link symbol-defining ops incrementally so the symbol table has
  // valid defining_op pointers throughout parsing. Use-def chains
  // are built in batch by loom_module_compute_uses after parsing.
  if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    loom_module_link_symbol_defining_op(parser->module, op, vtable);
  }

  *out_op = op;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_op_into(loom_parser_t* parser,
                                        loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;

  // Capture the start position of the op for source location tracking.
  // This is the first token: either a result name or the op name.
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  const iree_string_view_t* comments = NULL;
  iree_host_size_t comment_count = 0;
  loom_tokenizer_take_pending_comments(&parser->tokenizer, &comments,
                                       &comment_count);

  // Parse LHS result names: %a, %b = op.name ...
  // Or just: op.name ... (for ops with no results).
  if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    // Collect result names until we see '='.
    for (;;) {
      if (parsed->result_count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
        break;
      }
      loom_token_t name_token = loom_tokenizer_next(&parser->tokenizer);
      loom_type_t none_type = {0};
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_define_value(parser->module, none_type, &value_id));
      if (name_token.text.size > 0) {
        loom_string_id_t name_id = 0;
        IREE_RETURN_IF_ERROR(loom_module_intern_string(
            parser->module, name_token.text, &name_id));
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_name(parser->module, value_id, name_id));
      }
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_result(
          parsed, &parser->parser_arena, value_id, name_token));
    }
    // Expect '='.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'='"));
    }
  }

  // Parse op name.
  loom_token_t op_name_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_OP_NAME, &op_name_token);

  // Look up the op vtable.
  loom_op_kind_t kind;
  const loom_op_vtable_t* vtable = loom_context_lookup_op_by_name(
      parser->context, op_name_token.text, &kind);
  if (!vtable) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_006, params,
                            IREE_ARRAYSIZE(params), op_name_token);
  }

  if (loom_op_vtable_has_segmented_operands(vtable)) {
    IREE_RETURN_IF_ERROR(loom_parsed_op_prepare_operand_segments(
        parsed, &parser->parser_arena,
        loom_op_vtable_operand_segment_count(vtable)));
  }

  uint16_t pending_func_arg_start = parser->pending_func_args.count;
  const loom_text_low_asm_descriptor_set_t* previous_descriptor_set =
      parser->low_register_descriptor_set;

  bool is_symbol_definition =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
  if (is_symbol_definition && parsed->result_count > 0) {
    return loom_parser_emit_result_count_mismatch(parser, vtable, op_name_token,
                                                  /*expected_count=*/0,
                                                  parsed->result_count);
  }

  // Walk the format elements.
  bool func_args_consumed_by_region = false;
  iree_status_t walk_status = loom_parser_walk_format(
      parser, vtable, op_name_token, parsed, pending_func_arg_start,
      &func_args_consumed_by_region);
  parser->low_register_descriptor_set = previous_descriptor_set;
  IREE_RETURN_IF_ERROR(walk_status);

  if (parser->error_count > errors_before) {
    loom_parser_definition_scope_discard(parser);
  }

  // If FUNC_ARGS produced signature args but no body REGION consumed them
  // (e.g., func.decl, func.ukernel), these are declaration signature args.
  // Store their value IDs as op operands so FuncArgs printing and func-like
  // verification can recover the signature.
  if (parser->error_count == errors_before && func_args_consumed_by_region) {
    loom_parser_pending_block_args_truncate(&parser->pending_func_args,
                                            pending_func_arg_start);
  } else if (parser->error_count == errors_before &&
             parser->pending_func_args.count > pending_func_arg_start) {
    for (uint16_t i = pending_func_arg_start;
         i < parser->pending_func_args.count; ++i) {
      uint16_t operand_index = parsed->operand_count;
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
          parsed, &parser->parser_arena,
          parser->pending_func_args.entries[i].value_id));
      loom_token_t name_token = parser->pending_func_args.entries[i].name_token;
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
          parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
          operand_index, name_token, name_token.line, name_token.end_column));
    }
    loom_parser_pending_block_args_truncate(&parser->pending_func_args,
                                            pending_func_arg_start);
  } else if (parser->error_count > 0) {
    loom_parser_pending_block_args_truncate(&parser->pending_func_args,
                                            pending_func_arg_start);
    loom_parser_pending_block_args_clear(&parser->pending_block_args);
  }

  bool has_variadic_results =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
  if (!is_symbol_definition && !has_variadic_results &&
      parser->error_count == errors_before &&
      parsed->result_count != vtable->fixed_result_count) {
    IREE_RETURN_IF_ERROR(loom_parser_emit_result_count_mismatch(
        parser, vtable, op_name_token, vtable->fixed_result_count,
        parsed->result_count));
  }

  // Build the implicit parser-source fallback first, then let an explicit
  // trailing loc(...) annotation override it when present.
  loom_location_id_t fallback_location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id != LOOM_SOURCE_ID_INVALID) {
    loom_location_entry_t entry = loom_location_file_range(
        parser->source_id, (uint16_t)start_token.line,
        (uint16_t)start_token.column,
        (uint16_t)parser->tokenizer.consumed_end_line,
        (uint16_t)parser->tokenizer.consumed_end_column);
    IREE_RETURN_IF_ERROR(
        loom_module_add_location(parser->module, entry, &fallback_location));
  }
  loom_location_id_t location = fallback_location;
  IREE_RETURN_IF_ERROR(
      loom_parse_optional_op_location(parser, location, &location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  if (location == fallback_location &&
      fallback_location != LOOM_LOCATION_UNKNOWN &&
      parsed->field_span_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_location_field_spans(
        parser->module, fallback_location, parsed->field_spans,
        parsed->field_span_count));
  }

  // Finalize the op.
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_finalize_op(parser, kind, vtable, parsed, location, &op));
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

iree_status_t loom_parse_op(loom_parser_t* parser) {
  loom_parsed_op_t* parsed = NULL;
  IREE_RETURN_IF_ERROR(loom_parser_acquire_parsed_op(parser, &parsed));
  iree_status_t status = loom_parse_op_into(parser, parsed);
  loom_parser_result_scope_reset(&parser->result_scope);
  loom_parser_release_parsed_op(parser, parsed);
  return status;
}

//===----------------------------------------------------------------------===//
// Block and region parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_block_body(loom_parser_t* parser,
                                           loom_block_t* block) {
  // Set the builder's insertion point to this block.
  loom_builder_set_block(&parser->builder, block);

  // Parse ops until we hit '}' (end of region) or EOF.
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
    if (loom_parser_at_error_limit(parser)) {
      break;
    }
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_op(parser));
    if (parser->error_count > errors_before) {
      loom_parser_sync_to_newline(parser);
    }
  }
  return iree_ok_status();
}

static bool loom_parser_block_has_explicit_terminator(
    loom_parser_t* parser, const loom_block_t* block) {
  if (block->op_count == 0) {
    return false;
  }
  const loom_op_t* last_op = loom_block_const_last_op(block);
  const loom_op_vtable_t* last_vtable =
      loom_context_resolve_op(parser->context, last_op->kind);
  return last_vtable &&
         iree_any_bit_set(last_vtable->traits, LOOM_TRAIT_TERMINATOR);
}

iree_status_t loom_parser_append_implicit_terminator(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_block_t* block) {
  if (region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN ||
      loom_parser_block_has_explicit_terminator(parser, block)) {
    return iree_ok_status();
  }

  loom_builder_set_block(&parser->builder, block);
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id != LOOM_SOURCE_ID_INVALID) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    if (token.kind != LOOM_TOKEN_EOF && token.kind != LOOM_TOKEN_NONE) {
      loom_location_entry_t entry = loom_location_file_range(
          parser->source_id, (uint16_t)token.line, (uint16_t)token.column,
          (uint16_t)token.line, (uint16_t)token.end_column);
      IREE_RETURN_IF_ERROR(
          loom_module_add_location(parser->module, entry, &location));
    }
  }
  loom_op_t* terminator = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &parser->builder, region_descriptor->implicit_terminator,
      /*operand_count=*/0, /*result_count=*/0, /*region_count=*/0,
      /*tied_result_count=*/0, /*attribute_count=*/0, location, &terminator));
  return loom_builder_finalize_op(&parser->builder, terminator);
}

iree_status_t loom_parser_seed_region_entry_block(loom_parser_t* parser,
                                                  loom_region_t* region) {
  if (parser->pending_block_args.count > 0) {
    loom_block_t* entry_block = loom_region_entry_block(region);
    for (uint16_t i = 0; i < parser->pending_block_args.count; ++i) {
      const loom_parser_pending_block_arg_t pending_arg =
          parser->pending_block_args.entries[i];
      LOOM_PARSE_DEFINE_VALUE_NAME(parser, pending_arg.name_token,
                                   pending_arg.value_id);
      IREE_RETURN_IF_ERROR(loom_block_add_arg(parser->module, entry_block,
                                              pending_arg.value_id));
    }
    loom_parser_pending_block_args_clear(&parser->pending_block_args);
  }
  return iree_ok_status();
}

iree_status_t loom_parser_parse_optional_block_label(loom_parser_t* parser,
                                                     loom_region_t* region,
                                                     loom_block_t* block,
                                                     bool* out_present) {
  *out_present = false;
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
    return iree_ok_status();
  }

  const iree_string_view_t* comments = NULL;
  iree_host_size_t comment_count = 0;
  loom_tokenizer_take_pending_comments(&parser->tokenizer, &comments,
                                       &comment_count);
  loom_token_t label_token = loom_tokenizer_next(&parser->tokenizer);
  loom_string_id_t label_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, label_token.text, &label_id));
  if (loom_parser_find_block_by_label(parser, region, label_token.text)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(label_token.text),
    };
    return loom_parser_emit(parser, LOOM_ERR_PARSE_033, params,
                            IREE_ARRAYSIZE(params), label_token);
  }
  block->label_id = label_id;
  IREE_RETURN_IF_ERROR(loom_module_attach_block_comments(
      parser->module, block, comments, comment_count));

  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
           !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
      if (block->arg_count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      loom_token_t arg_name_token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &arg_name_token);
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
      loom_type_t arg_type = {0};
      IREE_RETURN_IF_ERROR(
          loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &arg_type));

      loom_value_id_t arg_value_id = 0;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &parser->builder, block, arg_type, &arg_value_id));
      LOOM_PARSE_DEFINE_VALUE_NAME(parser, arg_name_token, arg_value_id);
    }
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  *out_present = true;
  return iree_ok_status();
}

static iree_status_t loom_parse_region_body(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t* region, const void* user_data,
    bool* out_region_end_consumed) {
  *out_region_end_consumed = false;
  (void)user_data;

  // Seed the entry block with pending block args from FUNC_ARGS, BINDING_LIST,
  // or implicit region operands. Values are already defined in the module;
  // region-local names are defined in the child scope here.
  IREE_RETURN_IF_ERROR(loom_parser_seed_region_entry_block(parser, region));

  // Parse blocks. The entry block may not have an explicit label.
  bool first_block = true;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) {
      break;
    }

    loom_block_t* block = NULL;
    if (first_block) {
      block = loom_region_entry_block(region);
      first_block = false;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(parser->module, region, &block));
    }

    bool has_label = false;
    IREE_RETURN_IF_ERROR(loom_parser_parse_optional_block_label(
        parser, region, block, &has_label));

    const uint32_t block_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_block_body(parser, block));
    if (parser->error_count == block_errors_before) {
      IREE_RETURN_IF_ERROR(loom_parser_append_implicit_terminator(
          parser, region_descriptor, block));
    }
  }

  if (first_block && !loom_parser_at_error_limit(parser)) {
    IREE_RETURN_IF_ERROR(loom_parser_append_implicit_terminator(
        parser, region_descriptor, loom_region_entry_block(region)));
  }

  loom_tokenizer_discard_pending_comments(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  *out_region_end_consumed = true;
  return iree_ok_status();
}

iree_status_t loom_parse_braced_region_with_body(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_parse_region_body_callback_t body, loom_region_t** out_region) {
  uint32_t errors_before = parser->error_count;

  // Expect '{'.
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'{'"));
  }

  // Push a new lexical scope for the region. Pending FUNC_ARGS and
  // BINDING_LIST names are defined in this child scope when seeding the entry
  // block, so signature/result names from the parent op's Scope(...) do not
  // leak into the body.
  loom_parser_scope_t* outer_scope = parser->scope;
  IREE_RETURN_IF_ERROR(
      loom_parser_scope_push(parser, outer_scope, &parser->scope));

  // Allocate the region with a single entry block initially.
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(parser->module, 1, &region));

  // Save and set builder insertion point.
  loom_builder_ip_t saved_ip = loom_builder_save(&parser->builder);

  iree_host_size_t pending_successor_start =
      parser->pending_successor_refs.count;
  bool region_end_consumed = false;
  iree_status_t status = body.fn(parser, region_descriptor, region,
                                 body.user_data, &region_end_consumed);
  if (iree_status_is_ok(status) && parser->error_count == errors_before) {
    status = loom_parser_resolve_pending_successor_refs(
        parser, region, pending_successor_start);
  }
  if (parser->error_count > errors_before && !region_end_consumed &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    loom_parser_sync_to_brace(parser);
  }
  loom_parser_pending_block_args_clear(&parser->pending_block_args);
  if (parser->error_count > errors_before || !iree_status_is_ok(status)) {
    parser->pending_successor_refs.count = pending_successor_start;
  }

  // Restore scope and insertion point.
  loom_parser_scope_pop(parser);
  loom_builder_restore(&parser->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  *out_region = region;
  return iree_ok_status();
}

static iree_status_t loom_parse_braced_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region) {
  loom_parse_region_body_callback_t body = {
      .fn = loom_parse_region_body,
  };
  return loom_parse_braced_region_with_body(parser, region_descriptor, body,
                                            out_region);
}

iree_status_t loom_parse_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region) {
  return loom_parse_braced_region(parser, region_descriptor, out_region);
}

iree_status_t loom_parse_region_with_syntax(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_syntax_t syntax, loom_region_t** out_region) {
  switch (syntax) {
    case LOOM_REGION_SYNTAX_DEFAULT: {
      return loom_parse_braced_region(parser, region_descriptor, out_region);
    }
    case LOOM_REGION_SYNTAX_TEST_DO: {
      IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, LOOM_KW_DO));
      return loom_parse_braced_region(parser, region_descriptor, out_region);
    }
    case LOOM_REGION_SYNTAX_LOW_ASM: {
      return loom_parse_low_asm_prefixed_region(parser, region_descriptor,
                                                out_region);
    }
    case LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL: {
      if (loom_tokenizer_at_keyword(&parser->tokenizer, IREE_SV("asm"))) {
        return loom_parse_low_asm_prefixed_region(parser, region_descriptor,
                                                  out_region);
      }
      if (parser->low_asm_region_depth != 0) {
        return loom_parse_low_asm_inherited_region(parser, region_descriptor,
                                                   out_region);
      }
      return loom_parse_braced_region(parser, region_descriptor, out_region);
    }
    case LOOM_REGION_SYNTAX_PIPELINE: {
      if (loom_tokenizer_at_keyword(&parser->tokenizer, IREE_SV("pipeline"))) {
        return loom_parse_pipeline_prefixed_region(parser, region_descriptor,
                                                   out_region);
      }
      return loom_parse_braced_region(parser, region_descriptor, out_region);
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported region syntax %u", (uint32_t)syntax);
  }
}

//===----------------------------------------------------------------------===//
// Module parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_module_body(loom_parser_t* parser) {
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) {
      break;
    }

    // Check for attribute aliases: #alias = #encoding<params>.
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_HASH_ATTR)) {
      loom_tokenizer_discard_pending_comments(&parser->tokenizer);
      loom_token_t alias_token = loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
        uint32_t errors_before = parser->error_count;

        if (loom_context_lookup_encoding_vtable(parser->context,
                                                alias_token.text)) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(
                  IREE_SV("alias name shadows a registered encoding family")),
          };
          IREE_RETURN_IF_ERROR(loom_parser_emit(parser, LOOM_ERR_PARSE_014,
                                                params, IREE_ARRAYSIZE(params),
                                                alias_token));
          loom_parser_sync_to_newline(parser);
          continue;
        }
        if (loom_alias_table_lookup(&parser->aliases, alias_token.text) != 0) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(IREE_SV("duplicate encoding alias name")),
          };
          IREE_RETURN_IF_ERROR(loom_parser_emit(parser, LOOM_ERR_PARSE_014,
                                                params, IREE_ARRAYSIZE(params),
                                                alias_token));
          loom_parser_sync_to_newline(parser);
          continue;
        }

        loom_string_id_t alias_name_id = 0;
        IREE_RETURN_IF_ERROR(loom_module_intern_string(
            parser->module, alias_token.text, &alias_name_id));

        uint16_t encoding_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_parse_static_encoding(parser, alias_name_id, &encoding_id));
        if (parser->error_count > errors_before) {
          loom_parser_sync_to_newline(parser);
          continue;
        }

        if (encoding_id != 0) {
          IREE_RETURN_IF_ERROR(
              loom_alias_table_add(&parser->aliases, &parser->parser_arena,
                                   alias_token.text, encoding_id));
        }
        continue;
      }
      // Not an alias; this is an error.
      IREE_RETURN_IF_ERROR(loom_parser_emit_token_text_error(
          parser, LOOM_ERR_PARSE_014, alias_token));
      loom_parser_sync_to_newline(parser);
      continue;
    }

    // Parse a top-level op (function definition, etc.).
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_op(parser));
    if (parser->error_count > errors_before) {
      loom_parser_sync_to_newline(parser);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_text_parse(iree_string_view_t source,
                              iree_string_view_t filename,
                              loom_context_t* context,
                              iree_arena_block_pool_t* block_pool,
                              const loom_text_parse_options_t* options,
                              loom_module_t** out_module) {
  *out_module = NULL;

  // Allocate the module using the context's host allocator.
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(context, IREE_SV(""), block_pool,
                                            NULL, context->allocator, &module));

  // Register the source filename for location tracking. This interns the string
  // into the module so locations can reference it by module-local ID.
  iree_status_t status = iree_ok_status();
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  if (!iree_string_view_is_empty(filename)) {
    status = loom_module_register_source(module, filename, &source_id);
  }
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    return status;
  }

  // Initialize the parser.
  loom_parser_scope_t root_scope = {0};
  loom_parser_t parser = {
      .context = context,
      .module = module,
      .filename = filename,
      .source = source,
      .source_id = source_id,
      .cached_location =
          {
              .source_name = filename,
              .source_id = source_id,
          },
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .low_asm_environment = options ? options->low_asm_environment
                                     : (loom_text_low_asm_environment_t){0},
      .max_errors = options ? options->max_errors : 0,
      .scope = &root_scope,
      .definition_scope =
          {
              .pop_at = UINT16_MAX,
          },
  };
  if (parser.max_errors == 0) {
    parser.max_errors = 20;
  }
  iree_arena_initialize(block_pool, &parser.parser_arena);
  loom_tokenizer_initialize(source, filename, &parser.parser_arena,
                            &parser.tokenizer);
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &parser.builder);

  // Parse the module body.
  status = loom_parse_module_body(&parser);

  // Check for tokenizer scan errors.
  if (iree_status_is_ok(status)) {
    status = loom_tokenizer_consume_status(&parser.tokenizer);
  }

  // Resolve any module-level successor labels after the top-level block has
  // been fully parsed. Nested region references resolve at each region exit.
  if (iree_status_is_ok(status) && parser.error_count == 0) {
    status = loom_parser_resolve_pending_successor_refs(&parser, module->body,
                                                        /*pending_start=*/0);
  }

  // Build use-def chains in one pass.
  if (iree_status_is_ok(status) && parser.error_count == 0) {
    status = loom_module_compute_uses(module);
  }

  // Cleanup.
  loom_tokenizer_deinitialize(&parser.tokenizer);
  iree_arena_deinitialize(&parser.parser_arena);

  // Infrastructure failures propagate directly.
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    return status;
  }

  // Parse errors were emitted as diagnostics but no infrastructure failure.
  // The caller checks *out_module; NULL means parse errors occurred.
  if (parser.error_count > 0) {
    loom_module_free(module);
    return iree_ok_status();
  }

  *out_module = module;
  return iree_ok_status();
}
