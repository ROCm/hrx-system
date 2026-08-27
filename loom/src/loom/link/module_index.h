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
#include "loom/link/symbol_facet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel used when no provider/module/symbol ordinal is present.
#define LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL IREE_HOST_SIZE_MAX

// Dense ordinal assigned to one template-family key.
typedef uint32_t loom_link_template_family_ordinal_t;
#define LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID \
  ((loom_link_template_family_ordinal_t)UINT32_MAX)

typedef struct loom_link_module_index_t loom_link_module_index_t;

// Borrowed index-wide symbol ordinal list.
typedef struct loom_link_module_index_symbol_ordinal_list_t {
  // Index-wide symbol ordinals in stable provider order.
  const iree_host_size_t* values;
  // Number of symbol ordinals.
  iree_host_size_t count;
} loom_link_module_index_symbol_ordinal_list_t;

typedef enum loom_link_provider_kind_e {
  // Provider wraps one caller-owned in-memory module.
  LOOM_LINK_PROVIDER_MATERIALIZED = 0,
  // Provider wraps one validated bytecode file index.
  LOOM_LINK_PROVIDER_BYTECODE = 1,
  // Provider owns one text-parsed in-memory module.
  LOOM_LINK_PROVIDER_TEXT = 2,
} loom_link_provider_kind_t;

typedef enum loom_link_provider_role_e {
  // Direct sources jointly owned by the module being assembled.
  LOOM_LINK_PROVIDER_ROLE_INPUT = 0,
  // Separate libraries contributing exported exact definitions and templates.
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
  // Symbol is an external runtime import declaration.
  LOOM_LINK_SYMBOL_FLAG_IMPORT = 1u << 1,
  // Symbol is exported from its provider/module.
  LOOM_LINK_SYMBOL_FLAG_EXPORT = 1u << 2,
  // Symbol declares an externally supplied object or compile-time contract.
  LOOM_LINK_SYMBOL_FLAG_DECLARATION = 1u << 3,
  // Symbol is a concrete definition owned by its provider.
  LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION = 1u << 4,
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
  // Provider linkage role. Zero defaults to INPUT.
  loom_link_provider_role_t role;
} loom_link_module_index_add_options_t;

// Indexed provider record.
typedef struct loom_link_module_index_provider_t {
  // Index-wide provider ordinal.
  iree_host_size_t ordinal;
  // Source representation kind.
  loom_link_provider_kind_t kind;
  // Linkage ownership role.
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
    // Target interface constraints parallel to values. Zero accepts any
    // target interface.
    const loom_symbol_interface_flags_t* target_interfaces;
    // Source root region indices plus one parallel to values. Zero identifies
    // a source symbol contract or a module-root occurrence.
    const uint8_t* source_root_region_indices_plus_one;
  } dependencies;
  // Abstract template-family demand occurrences owned by this module.
  struct {
    // Total number of occurrences.
    iree_host_size_t count;
    // Dense index family ordinals in deterministic demand order. Ordinals may
    // repeat within a source row.
    const loom_link_template_family_ordinal_t* values;
    // Source root region indices plus one parallel to values.
    const uint8_t* source_root_region_indices_plus_one;
  } template_demands;
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
  // Dense family identity for template family symbols and providers, or
  // INVALID when this symbol has no template-family role.
  loom_link_template_family_ordinal_t template_family_ordinal;
  // Link identity class.
  loom_link_symbol_identity_t identity;
  // Linker-index symbol flags.
  loom_link_symbol_flags_t flags;
  // Independently selectable structural projections of this symbol.
  struct {
    // Schema projected from the defining operation during source indexing.
    loom_link_symbol_facet_schema_t schema;
    // Index-wide ordinal of the first semantic facet. Remaining facets follow
    // in the order returned by loom_link_module_index_symbol_facet_kind_at().
    iree_host_size_t start_ordinal;
  } facets;
  // Slice in the owning module's flat dependency occurrence array.
  struct {
    // First occurrence index.
    uint32_t first;
    // Number of occurrences.
    uint32_t count;
  } dependencies;
  // Slice in the owning module's flat template-demand array.
  struct {
    // First demand index.
    uint32_t first;
    // Number of demands.
    uint32_t count;
  } template_demands;
  // Intrusive index-owned chains.
  struct {
    // Next index symbol with the same name, or INVALID_ORDINAL.
    iree_host_size_t same_name_ordinal;
    // Next concrete provider for template_family_ordinal, or invalid.
    iree_host_size_t template_provider_ordinal;
  } next;
} loom_link_module_index_symbol_t;

// One unique template-family symbol identity and its provider chain.
typedef struct loom_link_module_index_template_family_t {
  // Dense index-wide family ordinal.
  loom_link_template_family_ordinal_t ordinal;
  // Indexed symbol carrying this family's private or global identity.
  iree_host_size_t identity_symbol_ordinal;
  // Borrowed family symbol name for diagnostics.
  iree_string_view_t name;
  // Provider-symbol chain in stable index order.
  struct {
    // First symbol ordinal, or INVALID_ORDINAL.
    iree_host_size_t first_symbol_ordinal;
    // Last symbol ordinal, or INVALID_ORDINAL.
    iree_host_size_t last_symbol_ordinal;
    // Number of provider symbols in the chain.
    iree_host_size_t count;
  } providers;
} loom_link_module_index_template_family_t;

// Allocates an empty module index over |context|.
//
// |block_pool| is used for transient text/bytecode parsing scratch and for
// text-provider module storage. The caller must keep materialized modules and
// bytecode buffers alive until the index is released.
iree_status_t loom_link_module_index_allocate(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_link_module_index_t** out_index);

// Frees |index| and any text-provider modules it owns.
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

// Returns the number of index-wide structural symbol facets.
iree_host_size_t loom_link_module_index_facet_count(
    const loom_link_module_index_t* index);

// Returns symbol |ordinal|, or NULL if out of range.
const loom_link_module_index_symbol_t* loom_link_module_index_symbol_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal);

// Returns semantic facet |ordinal| from |symbol|'s compact schema, or INVALID
// when ordinal is out of range.
loom_link_symbol_facet_kind_t loom_link_module_index_symbol_facet_kind_at(
    const loom_link_module_index_symbol_t* symbol, uint8_t ordinal);

// Classifies a physical source root as one semantic facet, or INVALID when the
// source root is outside the schema. Zero names the symbol contract; positive
// values name one-based root-region slots from materialized/text/bytecode
// provider metadata.
loom_link_symbol_facet_kind_t
loom_link_module_index_symbol_source_root_facet_kind(
    const loom_link_module_index_symbol_t* symbol,
    uint8_t source_root_region_index_plus_one);

// Returns the index-wide ordinal for semantic |kind|, or INVALID when |symbol|
// does not expose that facet.
iree_host_size_t loom_link_module_index_symbol_facet_ordinal(
    const loom_link_module_index_symbol_t* symbol,
    loom_link_symbol_facet_kind_t kind);

// Returns exported symbols owned by INPUT providers.
//
// The list is projected during provider indexing. Querying it performs no
// provider, module, or symbol traversal.
loom_link_module_index_symbol_ordinal_list_t
loom_link_module_index_input_exports(const loom_link_module_index_t* index);

// Returns the provider that supplied |symbol|, or NULL if its ordinal is stale.
const loom_link_module_index_provider_t* loom_link_module_index_symbol_provider(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Returns the module that owns |symbol|, or NULL if its ordinal is stale.
const loom_link_module_index_module_t* loom_link_module_index_symbol_module(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Looks up the first global-identity symbol named |name| in canonical
// enumeration order.
//
// Names may be passed with or without a leading '@'. INPUT providers enumerate
// before LIBRARY providers and ties use provider insertion order. This order
// does not resolve duplicate definitions; linkage validates uniqueness.
const loom_link_module_index_symbol_t* loom_link_module_index_lookup_global(
    const loom_link_module_index_t* index, iree_string_view_t name);

// Returns the first indexed symbol named |name| across all identity classes.
const loom_link_module_index_symbol_t* loom_link_module_index_lookup_name(
    const loom_link_module_index_t* index, iree_string_view_t name);

// Returns the next same-name symbol for |symbol|, or NULL if none exists.
const loom_link_module_index_symbol_t* loom_link_module_index_next_same_name(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol);

// Returns the next global duplicate for |symbol| in canonical order, or NULL
// if none exists. Begin enumeration with the result of
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

// Returns the number of unique template-family identities.
iree_host_size_t loom_link_module_index_template_family_count(
    const loom_link_module_index_t* index);

// Returns template family |ordinal|, or NULL if out of range.
const loom_link_module_index_template_family_t*
loom_link_module_index_template_family_at(
    const loom_link_module_index_t* index,
    loom_link_template_family_ordinal_t ordinal);

// Annotates |status| with the indexed provider and module locations of two
// conflicting global definitions.
iree_status_t loom_link_module_index_annotate_global_collision(
    iree_status_t status, const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* selected,
    const loom_link_module_index_symbol_t* duplicate);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_MODULE_INDEX_H_
