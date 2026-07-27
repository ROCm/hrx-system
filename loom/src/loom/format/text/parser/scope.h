// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_SCOPE_H_
#define LOOM_FORMAT_TEXT_PARSER_SCOPE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/tokenizer.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_parser_t loom_parser_t;

// Placeholder SSA value created by ARG-mode type parsing inside the current
// Scope(...). |resolved| tracks whether a later declaration in the same scope
// bound that placeholder name, and |name_token| is retained so scope-exit
// diagnostics can point at the original forward reference without storing
// source locations on every lexical scope entry.
typedef struct loom_parser_unresolved_placeholder_t {
  loom_value_id_t value_id;
  loom_token_t name_token;
  bool resolved;
} loom_parser_unresolved_placeholder_t;

// Growable scratch list of signature/global placeholders created by ARG-mode
// type parsing. Entries are appended as placeholders are created and truncated
// when the one active Scope(...) exits.
typedef struct loom_parser_unresolved_placeholders_t {
  loom_parser_unresolved_placeholder_t* entries;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_parser_unresolved_placeholders_t;

enum loom_parser_definition_scope_flag_bits_e {
  // ARG-mode placeholders in this Scope(...) are declarations as soon as type
  // parsing infers an index/encoding binding type from first use. Global
  // definitions need this because `%dim`/`%enc` names appear only inside the
  // result type and optional predicates, not as later `%name: type` binders.
  LOOM_PARSER_DEFINITION_SCOPE_FLAG_RESOLVE_PLACEHOLDERS_FROM_USE = 1u << 0,
};
typedef uint8_t loom_parser_definition_scope_flags_t;

// Tracks the one active Scope(...) declaration scope in an op format. Nested
// Scope(...) is intentionally unsupported by the DSL, so the C parser only
// needs one placeholder pop index and one format-element pop point.
typedef struct loom_parser_definition_scope_t {
  iree_host_size_t placeholder_start;
  // UINT16_MAX when no Scope(...) is active.
  uint16_t pop_at;
  loom_parser_definition_scope_flags_t flags;
} loom_parser_definition_scope_t;

// Open-addressed hash table entry keyed by interned string ID.
// Empty slots have name_id == LOOM_STRING_ID_INVALID.
typedef struct loom_parser_scope_entry_t {
  loom_string_id_t name_id;
  loom_value_id_t value_id;
} loom_parser_scope_entry_t;

// A name scope for SSA value resolution. Inner scopes see outer names via the
// parent chain; outer scopes cannot see inner names.
typedef struct loom_parser_scope_t {
  // Parser-owned scopes returned to parser->scope_free_list after pop.
  struct loom_parser_scope_t* next_free;
  struct loom_parser_scope_t* parent;
  loom_parser_scope_entry_t* entries;
  // Power of 2. 0 = not yet allocated.
  iree_host_size_t capacity;
  iree_host_size_t count;
} loom_parser_scope_t;

#define LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES 16

// Parser-owned scratch overlay for op result names while parsing result type
// annotations. Small result sets use inline linear scan; larger sets spill to
// a retained hash table with a per-op active capacity.
typedef struct loom_parser_result_scope_t {
  loom_parser_scope_entry_t
      inline_entries[LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES];
  loom_parser_scope_entry_t* hash_entries;
  // Active hash slots for this op.
  iree_host_size_t hash_capacity;
  // Retained backing allocation.
  iree_host_size_t hash_storage_capacity;
  uint16_t count;
} loom_parser_result_scope_t;

// Defines a name in this scope. Sets |*out_duplicate| to true if the name is
// already defined in this scope (not a parent).
iree_status_t loom_parser_scope_define(loom_parser_scope_t* scope,
                                       iree_arena_allocator_t* arena,
                                       loom_string_id_t name_id,
                                       loom_value_id_t value_id,
                                       bool* out_duplicate);

// Looks up a name in this scope and parent chain.
loom_value_id_t loom_parser_scope_lookup(const loom_parser_scope_t* scope,
                                         loom_string_id_t name_id);

// Looks up a name in only this scope, not parents.
loom_value_id_t loom_parser_scope_lookup_local(const loom_parser_scope_t* scope,
                                               loom_string_id_t name_id);

// Pushes a parser-owned child scope with |parent| as its lexical parent.
iree_status_t loom_parser_scope_push(loom_parser_t* parser,
                                     loom_parser_scope_t* parent,
                                     loom_parser_scope_t** out_scope);

// Pops |parser->scope| to its parent.
void loom_parser_scope_pop(loom_parser_t* parser);

// Reserves result-scope storage for |entry_count| named results.
iree_status_t loom_parser_result_scope_prepare(
    loom_parser_result_scope_t* scope, iree_host_size_t entry_count,
    iree_arena_allocator_t* arena);

// Drops the active result overlay.
void loom_parser_result_scope_reset(loom_parser_result_scope_t* scope);

// Defines a result name in the active result overlay.
iree_status_t loom_parser_result_scope_define(loom_parser_result_scope_t* scope,
                                              loom_string_id_t name_id,
                                              loom_value_id_t value_id,
                                              bool* out_duplicate);

// Resolves a value ID from the active result overlay, then lexical scope.
loom_value_id_t loom_parser_lookup_value(const loom_parser_t* parser,
                                         loom_string_id_t name_id);

// Interns |name_token.text|, stores it on |value_id|, and defines the value.
iree_status_t loom_parser_define_value_name(loom_parser_t* parser,
                                            loom_token_t name_token,
                                            loom_value_id_t value_id);

// Defines a typed SSA value in the current scope and returns its value ID.
iree_status_t loom_parser_define_value(loom_parser_t* parser,
                                       loom_token_t name_token,
                                       loom_type_t type,
                                       loom_value_id_t* out_value_id);

// Emits ERR_PARSE_002 for a duplicate SSA value or block argument name.
iree_status_t loom_parser_emit_duplicate_value_name(loom_parser_t* parser,
                                                    loom_token_t name_token);

// Resolves an SSA name token against the current scope chain.
iree_status_t loom_parser_resolve_value(loom_parser_t* parser,
                                        loom_token_t name_token,
                                        loom_value_id_t* out_value_id);

// Enters a one-level declaration scope that should pop after format element
// `pop_at`.
iree_status_t loom_parser_definition_scope_push(
    loom_parser_t* parser, uint16_t pop_at,
    loom_parser_definition_scope_flags_t flags);

// Pops the active declaration scope when |format_index| reaches pop_at.
iree_status_t loom_parser_definition_scope_pop_if_needed(loom_parser_t* parser,
                                                         uint16_t format_index);

// Discards the active declaration scope without diagnostics.
void loom_parser_definition_scope_discard(loom_parser_t* parser);

// Tracks a newly created ARG-mode placeholder.
iree_status_t loom_parser_add_unresolved_placeholder(loom_parser_t* parser,
                                                     loom_value_id_t value_id,
                                                     loom_token_t name_token);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_SCOPE_H_
