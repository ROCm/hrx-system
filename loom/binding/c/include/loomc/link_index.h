// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_INDEX_H_
#define LOOMC_LINK_INDEX_H_

#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/result.h"

/// @file
/// Reusable link indexes for deterministic provider discovery.
///
/// A frozen link index records provider identity and symbol metadata so link
/// operations can resolve live roots without rebuilding reusable library
/// indexes. Providers may be immutable source handles or already materialized
/// modules. Builders reserve deterministic provider slots, fill them in any
/// order, and finish into an immutable, shareable index whose provider order is
/// independent of scheduling order.
///
/// Source and module handles are retained by the builder and transferred to a
/// successful frozen index. Materialized modules therefore enter recursive
/// composition without a serialization round trip. A module must not be
/// mutated while retained by a builder or frozen index. The retained module
/// also retains its workspace, so the caller may release its module and
/// workspace references after adding the provider.
///
/// INPUT providers jointly own requester definitions and may contribute output
/// roots. LIBRARY providers contribute exported resolution candidates. A
/// public library definition selected only as a dependency becomes private in
/// the linked product; provider visibility never implicitly re-exports it.
///
/// Index construction scans provider metadata once. Later selective operations
/// share the frozen index and materialize only the selected roots, facets, and
/// reachable dependencies instead of rebuilding the complete provider catalog.
///
/// @par Example
/// Retain a frozen library index when handing it to an asynchronous worker:
///
/// @code{.c}
/// void schedule_worker(loomc_link_index_t* library_index) {
///   loomc_link_index_retain(library_index);
///   // `enqueue_work` is application-owned scheduling policy.
///   enqueue_work(library_index);
/// }
///
/// void worker_main(loomc_link_index_t* library_index) {
///   // Use library_index with a per-worker workspace and prepared linker.
///   loomc_link_index_release(library_index);
/// }
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Mutable builder used to construct a frozen link index.
///
/// @thread_safety
/// Reserving slots and finishing a builder require exclusive access. Distinct
/// already-reserved slots may be filled concurrently when each slot is filled
/// by at most one thread and no thread calls finish until all fills have
/// joined.
typedef struct loomc_link_index_builder_t loomc_link_index_builder_t;

/// Frozen immutable link index.
///
/// @thread_safety
/// Frozen indexes are immutable and may be shared by many linker invocations
/// across many threads.
typedef struct loomc_link_index_t loomc_link_index_t;

/// Linkage role assigned to a provider in a link index.
typedef enum loomc_link_provider_role_e {
  /// Direct sources jointly owned by the module being assembled.
  LOOMC_LINK_PROVIDER_ROLE_INPUT = 0,

  /// Separate libraries contributing exported exact definitions and templates.
  LOOMC_LINK_PROVIDER_ROLE_LIBRARY = 1,
} loomc_link_provider_role_t;

/// Representation kind recorded for an indexed provider.
typedef enum loomc_link_provider_kind_e {
  /// Provider kind is unknown.
  LOOMC_LINK_PROVIDER_KIND_UNKNOWN = 0,

  /// Provider was indexed from text `.loom` source.
  LOOMC_LINK_PROVIDER_KIND_TEXT = 1,

  /// Provider was indexed from bytecode `.loombc` source.
  LOOMC_LINK_PROVIDER_KIND_BYTECODE = 2,

  /// Provider was indexed from an already materialized module.
  LOOMC_LINK_PROVIDER_KIND_MATERIALIZED = 3,
} loomc_link_provider_kind_t;

/// Linker-facing symbol identity class.
typedef enum loomc_link_symbol_identity_e {
  /// Symbol identity is scoped to one provider-local module.
  LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE = 0,

  /// Symbol identity is resolved by name across providers.
  LOOMC_LINK_SYMBOL_IDENTITY_GLOBAL = 1,
} loomc_link_symbol_identity_t;

/// Linker-facing symbol kind.
typedef enum loomc_link_symbol_kind_e {
  /// Symbol kind is unknown.
  LOOMC_LINK_SYMBOL_KIND_UNKNOWN = 0,

  /// Function definition with a body.
  LOOMC_LINK_SYMBOL_KIND_FUNCTION_DEFINITION = 1,

  /// Function declaration or import.
  LOOMC_LINK_SYMBOL_KIND_FUNCTION_DECLARATION = 2,

  /// Function template.
  LOOMC_LINK_SYMBOL_KIND_FUNCTION_TEMPLATE = 3,

  /// Microkernel implementation.
  LOOMC_LINK_SYMBOL_KIND_FUNCTION_UKERNEL = 4,

  /// Global value or configuration symbol.
  LOOMC_LINK_SYMBOL_KIND_GLOBAL = 5,

  /// Executable/package symbol.
  LOOMC_LINK_SYMBOL_KIND_EXECUTABLE = 6,

  /// Record symbol.
  LOOMC_LINK_SYMBOL_KIND_RECORD = 7,
} loomc_link_symbol_kind_t;

/// Link-index symbol flag bit values.
typedef enum loomc_link_symbol_flag_bits_e {
  /// Symbol is publicly visible in its source module.
  LOOMC_LINK_SYMBOL_FLAG_PUBLIC = 1u << 0,

  /// Symbol is an import declaration supplied by another provider/module.
  LOOMC_LINK_SYMBOL_FLAG_IMPORT = 1u << 1,

  /// Symbol is exported from its provider/module.
  LOOMC_LINK_SYMBOL_FLAG_EXPORT = 1u << 2,

  /// Symbol is declaration-like and may be superseded by a concrete definition.
  LOOMC_LINK_SYMBOL_FLAG_DECLARATION = 1u << 3,

  /// Symbol is a concrete definition owned by its provider.
  LOOMC_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION = 1u << 4,

  /// Symbol implements the config symbol interface.
  LOOMC_LINK_SYMBOL_FLAG_CONFIG = 1u << 5,

  /// Symbol exists only for test or benchmark tooling.
  LOOMC_LINK_SYMBOL_FLAG_TEST_ONLY = 1u << 6,

  /// Source symbol is explicitly preserved by ordinary symbol pruning.
  LOOMC_LINK_SYMBOL_FLAG_RETAIN = 1u << 7,
} loomc_link_symbol_flag_bits_t;

/// Bitmask of `loomc_link_symbol_flag_bits_t` values.
typedef uint32_t loomc_link_symbol_flags_t;

/// Link-index builder creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LINK_INDEX_BUILDER_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_link_index_builder_options_t)`, and fill the requested fields.
typedef struct loomc_link_index_builder_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_LINK_INDEX_BUILDER_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future builder options.
  const void* next;

  /// Total bytes acquired from the host allocator per persistent frozen-index
  /// block. Internal tracking metadata consumes part of this capacity.
  loomc_host_size_t block_size;
} loomc_link_index_builder_options_t;

/// Deterministic provider slot reserved by a link-index builder.
///
/// Slots are stable within a builder. Parallel workers fill distinct slots so
/// provider order and diagnostics do not depend on fill scheduling order.
typedef struct loomc_link_index_provider_slot_t {
  /// Deterministic provider slot ordinal reserved by the builder.
  loomc_host_size_t ordinal;
} loomc_link_index_provider_slot_t;

/// Provider options used when adding sources or modules to a link index.
///
/// @lifetime
/// String views are borrowed for the duration of the builder call that consumes
/// this descriptor unless that call explicitly documents a copy.
typedef struct loomc_link_index_provider_options_t {
  /// Stable provider label for diagnostics and private-name determinism.
  loomc_string_view_t provider_name;

  /// Provider linkage role.
  loomc_link_provider_role_t role;

} loomc_link_index_provider_options_t;

/// Indexed provider metadata.
///
/// @lifetime
/// String views are borrowed from the frozen index and remain valid until the
/// index is released.
typedef struct loomc_link_index_provider_t {
  /// Index-wide provider ordinal.
  loomc_host_size_t ordinal;

  /// Source representation kind.
  loomc_link_provider_kind_t kind;

  /// Provider linkage role.
  loomc_link_provider_role_t role;

  /// Provider label used for diagnostics.
  loomc_string_view_t name;

  /// First module ordinal owned by this provider.
  loomc_host_size_t module_start_ordinal;

  /// Number of modules owned by this provider.
  loomc_host_size_t module_count;
} loomc_link_index_provider_t;

/// Indexed module metadata.
///
/// @lifetime
/// String views are borrowed from the frozen index and remain valid until the
/// index is released.
typedef struct loomc_link_index_module_t {
  /// Index-wide module ordinal.
  loomc_host_size_t ordinal;

  /// Provider that owns this module.
  loomc_host_size_t provider_ordinal;

  /// Provider-local module ordinal.
  loomc_host_size_t provider_module_ordinal;

  /// Module name.
  loomc_string_view_t name;

  /// First symbol ordinal owned by this module.
  loomc_host_size_t symbol_start_ordinal;

  /// Number of symbols owned by this module.
  loomc_host_size_t symbol_count;
} loomc_link_index_module_t;

/// Indexed symbol metadata.
///
/// @lifetime
/// String views are borrowed from the frozen index and remain valid until the
/// index is released.
typedef struct loomc_link_index_symbol_t {
  /// Index-wide symbol ordinal.
  loomc_host_size_t ordinal;

  /// Provider that supplied this symbol.
  loomc_host_size_t provider_ordinal;

  /// Module that owns this symbol.
  loomc_host_size_t module_ordinal;

  /// Provider-local module ordinal.
  loomc_host_size_t provider_module_ordinal;

  /// Module-local symbol ordinal.
  loomc_host_size_t module_symbol_ordinal;

  /// Module-local symbol name without an `@` sigil.
  loomc_string_view_t name;

  /// Linker-facing symbol kind.
  loomc_link_symbol_kind_t kind;

  /// Link identity class.
  loomc_link_symbol_identity_t identity;

  /// Link-index symbol flags.
  loomc_link_symbol_flags_t flags;
} loomc_link_index_symbol_t;

/// Creates an empty link-index builder.
///
/// @param context Reusable Loom API context retained by the builder.
/// @param options Builder options, or `NULL` for defaults.
/// @param allocator Host allocator used for builder and frozen-index storage.
/// @param out_builder Receives the mutable builder on success.
/// @return OK when the builder was created.
///
/// @ownership
/// The caller owns the returned builder and releases it with
/// `loomc_link_index_builder_release`.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_create(
    loomc_context_t* context, const loomc_link_index_builder_options_t* options,
    loomc_allocator_t allocator, loomc_link_index_builder_t** out_builder);

/// Releases a link-index builder.
///
/// @param builder Builder to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// No other thread may be reserving, filling, or finishing `builder`.
LOOMC_API_EXPORT void loomc_link_index_builder_release(
    loomc_link_index_builder_t* builder);

/// Reserves a deterministic provider slot.
///
/// @param builder Builder to mutate.
/// @param options Provider options copied by this call, or `NULL` for defaults.
/// @param out_slot Receives the reserved provider slot.
/// @return OK when the slot was reserved.
///
/// @thread_safety
/// Reserving slots requires exclusive access to `builder`.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_reserve_provider_slot(
    loomc_link_index_builder_t* builder,
    const loomc_link_index_provider_options_t* options,
    loomc_link_index_provider_slot_t* out_slot);

/// Fills an already reserved provider slot with a source.
///
/// @param builder Builder that owns `slot`.
/// @param slot Provider slot previously returned by
/// `loomc_link_index_builder_reserve_provider_slot`.
/// @param source Source retained by the builder on success.
/// @return OK when the slot was filled.
///
/// @thread_safety
/// Distinct reserved slots may be filled concurrently when each slot is filled
/// by at most one thread and no thread calls reserve or finish concurrently.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_fill_source_slot(
    loomc_link_index_builder_t* builder, loomc_link_index_provider_slot_t slot,
    loomc_source_t* source);

/// Fills an already reserved provider slot with a materialized module.
///
/// @param builder Builder that owns `slot`.
/// @param slot Provider slot previously returned by
/// `loomc_link_index_builder_reserve_provider_slot`.
/// @param module Module retained by the builder on success.
/// @return OK when the slot was filled.
///
/// @lifetime
/// The caller must not mutate `module` until the builder and any frozen index
/// produced from it have released their retained references.
///
/// @thread_safety
/// Distinct reserved slots may be filled concurrently when each slot is filled
/// by at most one thread and no thread calls reserve or finish concurrently.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_fill_module_slot(
    loomc_link_index_builder_t* builder, loomc_link_index_provider_slot_t slot,
    loomc_module_t* module);

/// Reserves and fills the next provider slot with a source.
///
/// @param builder Builder to mutate.
/// @param source Source retained by the builder on success.
/// @param options Provider options copied by this call, or `NULL` for defaults.
/// @param out_slot Receives the provider slot, or `NULL` if not needed.
/// @return OK when the source was queued for indexing.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_add_source(
    loomc_link_index_builder_t* builder, loomc_source_t* source,
    const loomc_link_index_provider_options_t* options,
    loomc_link_index_provider_slot_t* out_slot);

/// Reserves and fills the next provider slot with a materialized module.
///
/// @param builder Builder to mutate.
/// @param module Module retained by the builder on success.
/// @param options Provider options copied by this call, or `NULL` for defaults.
/// @param out_slot Receives the provider slot, or `NULL` if not needed.
/// @return OK when the module was queued for indexing.
///
/// @lifetime
/// The caller must not mutate `module` until the builder and any frozen index
/// produced from it have released their retained references.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_add_module(
    loomc_link_index_builder_t* builder, loomc_module_t* module,
    const loomc_link_index_provider_options_t* options,
    loomc_link_index_provider_slot_t* out_slot);

/// Finishes a builder into a frozen immutable link index.
///
/// @param builder Builder to finish. The builder becomes finished after this
/// call returns OK, whether the operation result succeeded or failed.
/// @param out_link_index Receives one retained frozen index when the result
/// state is succeeded.
/// @param out_result Receives one retained result describing indexing outcome.
/// @return OK when a result was produced. Non-OK status means API or
/// infrastructure failure prevented result production.
///
/// @ownership
/// Finishing does not consume `builder`; the caller still releases it with
/// `loomc_link_index_builder_release` after the call returns. The caller owns
/// `out_link_index` on successful result state and releases it with
/// `loomc_link_index_release`. The caller owns `out_result` and releases it
/// with `loomc_result_release`.
///
/// @error_contract
/// Invalid Loom text, malformed bytecode, empty reserved slots, and unsupported
/// source formats produce OK status plus a failed result with diagnostics.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_finish(
    loomc_link_index_builder_t* builder, loomc_link_index_t** out_link_index,
    loomc_result_t** out_result);

/// Retains a frozen link index for another owner.
///
/// @param link_index Link index to retain.
///
/// @thread_safety
/// Retain/release operations for frozen indexes are intended to be safe from
/// multiple threads.
LOOMC_API_EXPORT void loomc_link_index_retain(loomc_link_index_t* link_index);

/// Releases a frozen link index from one owner.
///
/// @param link_index Link index to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations for frozen indexes are intended to be safe from
/// multiple threads. No mutation occurs after the index is frozen.
LOOMC_API_EXPORT void loomc_link_index_release(loomc_link_index_t* link_index);

/// Returns the number of providers in a frozen index.
///
/// @param link_index Index to inspect.
/// @return Provider count.
LOOMC_API_EXPORT loomc_host_size_t
loomc_link_index_provider_count(const loomc_link_index_t* link_index);

/// Returns provider metadata by ordinal.
///
/// @param link_index Index to inspect.
/// @param ordinal Zero-based provider ordinal.
/// @param out_provider Receives provider metadata.
/// @return True when `ordinal` was valid.
LOOMC_API_EXPORT bool loomc_link_index_provider_at(
    const loomc_link_index_t* link_index, loomc_host_size_t ordinal,
    loomc_link_index_provider_t* out_provider);

/// Returns the number of modules in a frozen index.
///
/// @param link_index Index to inspect.
/// @return Module count.
LOOMC_API_EXPORT loomc_host_size_t
loomc_link_index_module_count(const loomc_link_index_t* link_index);

/// Returns module metadata by ordinal.
///
/// @param link_index Index to inspect.
/// @param ordinal Zero-based module ordinal.
/// @param out_module Receives module metadata.
/// @return True when `ordinal` was valid.
LOOMC_API_EXPORT bool loomc_link_index_module_at(
    const loomc_link_index_t* link_index, loomc_host_size_t ordinal,
    loomc_link_index_module_t* out_module);

/// Returns the number of symbols in a frozen index.
///
/// @param link_index Index to inspect.
/// @return Symbol count.
LOOMC_API_EXPORT loomc_host_size_t
loomc_link_index_symbol_count(const loomc_link_index_t* link_index);

/// Returns symbol metadata by ordinal.
///
/// @param link_index Index to inspect.
/// @param ordinal Zero-based symbol ordinal.
/// @param out_symbol Receives symbol metadata.
/// @return True when `ordinal` was valid.
LOOMC_API_EXPORT bool loomc_link_index_symbol_at(
    const loomc_link_index_t* link_index, loomc_host_size_t ordinal,
    loomc_link_index_symbol_t* out_symbol);

/// Looks up the first global symbol by name in canonical enumeration order.
///
/// INPUT providers enumerate before LIBRARY providers. Ties preserve
/// provider-slot order. This order does not resolve duplicate definitions;
/// linkage validates uniqueness.
///
/// @param link_index Index to inspect.
/// @param name Symbol name with or without a leading `@`.
/// @param out_symbol Receives symbol metadata.
/// @return True when a global symbol named `name` was found.
LOOMC_API_EXPORT bool loomc_link_index_lookup_global(
    const loomc_link_index_t* link_index, loomc_string_view_t name,
    loomc_link_index_symbol_t* out_symbol);

/// Returns the next duplicate global symbol in canonical order.
///
/// Begin enumeration with a symbol returned by
/// `loomc_link_index_lookup_global` and pass each returned duplicate back to
/// continue until this returns false.
///
/// @param link_index Index to inspect.
/// @param symbol Symbol previously returned from this index.
/// @param out_symbol Receives duplicate symbol metadata.
/// @return True when another duplicate exists.
LOOMC_API_EXPORT bool loomc_link_index_next_global_duplicate(
    const loomc_link_index_t* link_index,
    const loomc_link_index_symbol_t* symbol,
    loomc_link_index_symbol_t* out_symbol);

/// Looks up a private symbol within an indexed module.
///
/// @param link_index Index to inspect.
/// @param module Module metadata previously returned from this index.
/// @param name Symbol name with or without a leading `@`.
/// @param out_symbol Receives private symbol metadata.
/// @return True when a private symbol named `name` was found in `module`.
LOOMC_API_EXPORT bool loomc_link_index_lookup_private(
    const loomc_link_index_t* link_index,
    const loomc_link_index_module_t* module, loomc_string_view_t name,
    loomc_link_index_symbol_t* out_symbol);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LINK_INDEX_H_
