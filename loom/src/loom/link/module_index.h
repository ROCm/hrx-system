// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Provider-backed module index for linker planning.
//
// The index is the cold planning layer above bytecode/text/materialized inputs.
// It records lightweight symbol identity, provenance, and direct reference
// facts without cloning IR so the planner can compute an exact symbol closure
// before materialization.

#ifndef LOOM_LINK_MODULE_INDEX_H_
#define LOOM_LINK_MODULE_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/text/parser.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel used when no provider/module/symbol ordinal is present.
#define LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL IREE_HOST_SIZE_MAX

// Dense ordinal assigned to one implementation-contract key.
typedef uint32_t loom_link_contract_ordinal_t;
#define LOOM_LINK_CONTRACT_ORDINAL_INVALID \
  ((loom_link_contract_ordinal_t)UINT32_MAX)

typedef struct loom_link_module_index_t loom_link_module_index_t;

typedef enum loom_link_provider_kind_e {
  // Provider wraps one caller-owned in-memory module.
  LOOM_LINK_PROVIDER_MATERIALIZED = 0,
  // Provider wraps one validated bytecode file index.
  LOOM_LINK_PROVIDER_BYTECODE = 1,
  // Provider owns one text-parsed in-memory module.
  LOOM_LINK_PROVIDER_TEXT = 2,
} loom_link_provider_kind_t;

typedef enum loom_link_provider_role_e {
  // Primary input providers selected before library providers.
  LOOM_LINK_PROVIDER_ROLE_INPUT = 0,
  // Library providers searched after primary inputs.
  LOOM_LINK_PROVIDER_ROLE_LIBRARY = 1,
} loom_link_provider_role_t;

typedef enum loom_link_symbol_identity_e {
  // Symbol identity is scoped to one provider-local module.
  LOOM_LINK_SYMBOL_IDENTITY_PRIVATE = 0,
  // Symbol identity is resolved by name across providers.
  LOOM_LINK_SYMBOL_IDENTITY_GLOBAL = 1,
} loom_link_symbol_identity_t;

enum loom_link_symbol_flag_bits_e {
  // Symbol is publicly visible in its source module.
  LOOM_LINK_SYMBOL_FLAG_PUBLIC = 1u << 0,
  // Symbol is an import declaration supplied by another provider/module.
  LOOM_LINK_SYMBOL_FLAG_IMPORT = 1u << 1,
  // Symbol is exported from its provider/module.
  LOOM_LINK_SYMBOL_FLAG_EXPORT = 1u << 2,
  // Symbol is declaration-like and may be superseded by a concrete definition.
  LOOM_LINK_SYMBOL_FLAG_DECLARATION = 1u << 3,
  // Symbol has materializable IR owned by its provider.
  LOOM_LINK_SYMBOL_FLAG_HAS_BODY = 1u << 4,
  // Symbol implements the config symbol interface.
  LOOM_LINK_SYMBOL_FLAG_CONFIG = 1u << 5,
  // Symbol exists only for test or benchmark tooling.
  LOOM_LINK_SYMBOL_FLAG_TEST_ONLY = 1u << 6,
};
typedef uint32_t loom_link_symbol_flags_t;

// Options applied when adding a provider to an index.
typedef struct loom_link_module_index_add_options_t {
  // Stable provider label for diagnostics and deterministic private naming.
  iree_string_view_t provider_name;
  // Provider precedence role. Zero defaults to INPUT.
  loom_link_provider_role_t role;
} loom_link_module_index_add_options_t;

// Indexed provider record.
typedef struct loom_link_module_index_provider_t {
  // Index-wide provider ordinal.
  iree_host_size_t ordinal;
  // Source representation kind.
  loom_link_provider_kind_t kind;
  // Search precedence role.
  loom_link_provider_role_t role;
  // Arena-owned provider label.
  iree_string_view_t name;
  // First module ordinal owned by this provider.
  iree_host_size_t module_start_ordinal;
  // Number of modules owned by this provider.
  iree_host_size_t module_count;
  // Validated source backing present only for BYTECODE providers.
  struct {
    // Borrowed complete bytecode file contents.
    iree_const_byte_span_t contents;
    // Arena-owned source filename used for materialization diagnostics.
    iree_string_view_t filename;
    // Validated file index whose child arrays are owned by the index arena.
    loom_bytecode_file_metadata_t metadata;
  } bytecode;
} loom_link_module_index_provider_t;

// Indexed module record.
typedef struct loom_link_module_index_module_t {
  // Index-wide module ordinal.
  iree_host_size_t ordinal;
  // Provider that owns this module.
  iree_host_size_t provider_ordinal;
  // Provider-local module ordinal.
  iree_host_size_t provider_module_ordinal;
  // Borrowed module name.
  iree_string_view_t name;
  // Materialized module pointer when already available.
  const loom_module_t* materialized_module;
  // True when the index owns materialized_module and frees it on destroy.
  bool owns_materialized_module;
  // First symbol ordinal owned by this module.
  iree_host_size_t symbol_start_ordinal;
  // Number of symbols owned by this module.
  iree_host_size_t symbol_count;
  // Direct dependency occurrences owned by this module.
  struct {
    // Number of module-root occurrences at the start of values.
    uint32_t root_count;
    // Total number of occurrences.
    iree_host_size_t count;
    // Module-local target symbol ordinals in deterministic occurrence order.
    // Targets may repeat within a source row.
    const uint32_t* values;
  } dependencies;
  // Abstract implementation-contract demand occurrences owned by this module.
  struct {
    // Total number of occurrences.
    iree_host_size_t count;
    // Dense index contract ordinals in deterministic demand order. Ordinals
    // may repeat within a source row.
    const loom_link_contract_ordinal_t* values;
  } contract_demands;
} loom_link_module_index_module_t;

// Indexed module-local symbol record.
typedef struct loom_link_module_index_symbol_t {
  // Index-wide symbol ordinal.
  iree_host_size_t ordinal;
  // Module that owns this symbol.
  iree_host_size_t module_ordinal;
  // Module-local symbol ordinal.
  iree_host_size_t module_symbol_ordinal;
  // Borrowed module-local symbol name without an '@' sigil.
  iree_string_view_t name;
  // Canonical in-memory symbol kind.
  loom_symbol_kind_t kind;
  // Implementation-contract group for func.template/func.ukernel providers,
  // or LOOM_LINK_CONTRACT_ORDINAL_INVALID.
  loom_link_contract_ordinal_t provider_contract_ordinal;
  // Link identity class.
  loom_link_symbol_identity_t identity;
  // Linker-index symbol flags.
  loom_link_symbol_flags_t flags;
  // Slice in the owning module's flat dependency occurrence array.
  struct {
    // First occurrence index.
    uint32_t first;
    // Number of occurrences.
    uint32_t count;
  } dependencies;
  // Slice in the owning module's flat contract-demand array.
  struct {
    // First demand index.
    uint32_t first;
    // Number of demands.
    uint32_t count;
  } contract_demands;
  // Intrusive index-owned chains.
  struct {
    // Next index symbol with the same name, or INVALID_ORDINAL.
    iree_host_size_t same_name_ordinal;
    // Next provider symbol for provider_contract_ordinal, or INVALID_ORDINAL.
    iree_host_size_t contract_provider_ordinal;
  } next;
} loom_link_module_index_symbol_t;

// One unique abstract implementation-contract key and its provider chain.
typedef struct loom_link_module_index_contract_t {
  // Dense index-wide contract ordinal.
  loom_link_contract_ordinal_t ordinal;
  // Borrowed implementation-contract key.
  iree_string_view_t name;
  // Provider-symbol chain in stable index order.
  struct {
    // First symbol ordinal, or INVALID_ORDINAL.
    iree_host_size_t first_symbol_ordinal;
    // Last symbol ordinal, or INVALID_ORDINAL.
    iree_host_size_t last_symbol_ordinal;
  } providers;
} loom_link_module_index_contract_t;

// Creates an empty module index over |context|.
//
// |block_pool| is used for transient text/bytecode parsing scratch and for
// text-provider module storage. The caller must keep materialized modules and
// bytecode buffers alive until the index is released.
iree_status_t loom_link_module_index_create(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_link_module_index_t** out_index);

// Releases |index| and any text-provider modules it owns.
void loom_link_module_index_free(loom_link_module_index_t* index);

// Adds one caller-owned materialized module to |index|.
iree_status_t loom_link_module_index_add_materialized(
    loom_link_module_index_t* index, const loom_module_t* module,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal);

// Adds one bytecode file to |index| using metadata-only validation.
//
// The index borrows |bytecode| through its lifetime. |filename| and any
// provider name supplied in |options| are copied into index-owned storage.
iree_status_t loom_link_module_index_add_bytecode(
    loom_link_module_index_t* index, iree_const_byte_span_t bytecode,
    iree_string_view_t filename,
    const loom_bytecode_index_options_t* index_options,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal);

// Parses and adds one text module to |index|.
//
// Text is the cold path: the provider owns the parsed module and indexes its
// materialized symbol table through the same records used by caller-owned IR.
iree_status_t loom_link_module_index_add_text(
    loom_link_module_index_t* index, iree_string_view_t source,
    iree_string_view_t filename, const loom_text_parse_options_t* parse_options,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal);

// Returns the number of indexed providers.
iree_host_size_t loom_link_module_index_provider_count(
    const loom_link_module_index_t* index);

// Returns provider |ordinal|, or NULL if out of range.
const loom_link_module_index_provider_t* loom_link_module_index_provider_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal);

// Returns the number of indexed modules.
iree_host_size_t loom_link_module_index_module_count(
    const loom_link_module_index_t* index);

// Returns module |ordinal|, or NULL if out of range.
const loom_link_module_index_module_t* loom_link_module_index_module_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal);

// Returns the number of indexed symbols.
iree_host_size_t loom_link_module_index_symbol_count(
    const loom_link_module_index_t* index);

// Returns symbol |ordinal|, or NULL if out of range.
const loom_link_module_index_symbol_t* loom_link_module_index_symbol_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal);

// Returns the provider that supplied |symbol|, or NULL if its ordinal is stale.
const loom_link_module_index_provider_t* loom_link_module_index_symbol_provider(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Returns the module that owns |symbol|, or NULL if its ordinal is stale.
const loom_link_module_index_module_t* loom_link_module_index_symbol_module(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Looks up the selected global-identity symbol named |name|.
//
// Names may be passed with or without a leading '@'. INPUT providers shadow
// LIBRARY providers; ties are resolved by provider insertion order.
const loom_link_module_index_symbol_t* loom_link_module_index_lookup_global(
    const loom_link_module_index_t* index, iree_string_view_t name);

// Returns the first indexed symbol named |name| across all identity classes.
const loom_link_module_index_symbol_t* loom_link_module_index_lookup_name(
    const loom_link_module_index_t* index, iree_string_view_t name);

// Returns the next same-name symbol for |symbol|, or NULL if none exists.
const loom_link_module_index_symbol_t* loom_link_module_index_next_same_name(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Returns the next global duplicate for |symbol| in selected-first order, or
// NULL if none exists. Begin enumeration with the result of
// loom_link_module_index_lookup_global and pass each returned duplicate back to
// continue until NULL.
const loom_link_module_index_symbol_t*
loom_link_module_index_next_global_duplicate(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Looks up a private symbol by provider-local module and name.
//
// Names may be passed with or without a leading '@'. Global-identity symbols
// are intentionally ignored by this lookup.
const loom_link_module_index_symbol_t* loom_link_module_index_lookup_private(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module, iree_string_view_t name);

// Returns the number of unique implementation-contract keys.
iree_host_size_t loom_link_module_index_contract_count(
    const loom_link_module_index_t* index);

// Returns implementation contract |ordinal|, or NULL if out of range.
const loom_link_module_index_contract_t* loom_link_module_index_contract_at(
    const loom_link_module_index_t* index,
    loom_link_contract_ordinal_t ordinal);

// Returns a status that names the two provider locations for a duplicate
// global symbol. This is a diagnostic helper for planner conflict reporting.
iree_status_t loom_link_module_index_duplicate_global_status(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* selected,
    const loom_link_module_index_symbol_t* duplicate);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_MODULE_INDEX_H_
