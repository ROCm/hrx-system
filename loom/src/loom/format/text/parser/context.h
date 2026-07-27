// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_CONTEXT_H_
#define LOOM_FORMAT_TEXT_PARSER_CONTEXT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/low_asm.h"
#include "loom/format/text/parser/accumulator.h"
#include "loom/format/text/parser/aliases.h"
#include "loom/format/text/parser/parser.h"
#include "loom/format/text/parser/scope.h"
#include "loom/format/text/parser/types.h"
#include "loom/format/text/tokenizer.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_parser_t {
  loom_tokenizer_t tokenizer;
  loom_module_t* module;
  loom_context_t* context;
  iree_arena_allocator_t parser_arena;
  loom_builder_t builder;
  loom_parser_scope_t* scope;

  // Parser-owned reusable child lexical scope frames. The root module scope is
  // stack-owned in loom_text_parse and is not part of this free list.
  loom_parser_scope_t* scope_free_list;

  // Reusable result-name overlay while parsing result type annotations.
  loom_parser_result_scope_t result_scope;

  // Parser-owned reusable op scratch frames.
  loom_parsed_op_t* parsed_op_free_list;

  // Parser-owned reusable encoding-parameter scratch frames.
  loom_parser_encoding_params_t* encoding_params_free_list;

  // Parser-owned reusable type-list scratch frames.
  loom_parser_type_list_t* type_list_free_list;

  loom_alias_table_t aliases;
  loom_symbol_map_t symbol_lookup;
  loom_diagnostic_sink_t diagnostic_sink;

  // Parse-time environment for low.asm regions.
  loom_text_low_asm_environment_t low_asm_environment;

  // Descriptor-set context used for parsing target-low register types.
  const loom_text_low_asm_descriptor_set_t* low_register_descriptor_set;

  // Nesting depth of active descriptor-backed low asm region bodies.
  uint16_t low_asm_region_depth;

  uint32_t error_count;
  uint32_t max_errors;
  iree_string_view_t filename;
  iree_string_view_t source;
  loom_source_id_t source_id;

  // One-entry lookaside for repeated loc("same_file":...) annotations.
  struct {
    iree_string_view_t source_name;
    loom_source_id_t source_id;
  } cached_location;

  // Pending signature arguments from FUNC_ARGS. Bodyful func-like ops attach
  // these to their declared body region; bodyless declarations store them as
  // operands.
  loom_parser_pending_block_args_t pending_func_args;

  // Pending block arguments from BLOCK_ARGS, BINDING_LIST, or implicit region
  // operands such as loop IVs.
  loom_parser_pending_block_args_t pending_block_args;

  // Pending CFG successor labels awaiting the end of their enclosing region.
  loom_parser_pending_successor_refs_t pending_successor_refs;

  // Placeholder values created by ARG-mode type parsing inside Scope(...).
  loom_parser_unresolved_placeholders_t unresolved_placeholders;

  // One active Scope(...) declaration wrapper.
  loom_parser_definition_scope_t definition_scope;
} loom_parser_t;

// Returns true while parsing the body of the one active Scope(...) declaration
// wrapper.
static inline bool loom_parser_in_definition_scope(
    const loom_parser_t* parser) {
  return parser->definition_scope.pop_at != UINT16_MAX;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_CONTEXT_H_
