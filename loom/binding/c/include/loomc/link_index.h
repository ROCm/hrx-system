// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_INDEX_H_
#define LOOMC_LINK_INDEX_H_

#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/result.h"

/// @file
/// Reusable link indexes for deterministic library and source discovery.
///
/// A frozen link index records source/provider identity and symbol metadata so
/// link operations can resolve live roots without rebuilding reusable library
/// indexes. Builders reserve deterministic source slots, accept source handles
/// in any slot order, and finish into an immutable, shareable index whose
/// provider order is independent of scheduling order.
///
/// The first indexing contract is intentionally explicit about lifetimes:
/// source handles are retained by the builder, and a frozen index owns the
/// persistent storage needed for provider/module/symbol metadata. Resetting a
/// caller workspace never invalidates a frozen link index.
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

/// Search role assigned to a source provider in a link index.
typedef enum loomc_link_provider_role_e {
  /// Primary input providers are selected before libraries.
  LOOMC_LINK_PROVIDER_ROLE_INPUT = 0,

  /// Library providers are searched after primary inputs.
  LOOMC_LINK_PROVIDER_ROLE_LIBRARY = 1,
} loomc_link_provider_role_t;

/// Source representation kind recorded for an indexed provider.
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

  /// Symbol has materializable IR owned by its provider.
  LOOMC_LINK_SYMBOL_FLAG_HAS_BODY = 1u << 4,

  /// Symbol implements the config symbol interface.
  LOOMC_LINK_SYMBOL_FLAG_CONFIG = 1u << 5,

  /// Symbol is a check.case correctness harness.
  LOOMC_LINK_SYMBOL_FLAG_CHECK_CASE = 1u << 6,

  /// Symbol is a check.benchmark policy record.
  LOOMC_LINK_SYMBOL_FLAG_CHECK_BENCHMARK = 1u << 7,
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

  /// Arena block size used for persistent frozen-index metadata.
  loomc_host_size_t block_size;
} loomc_link_index_builder_options_t;

/// Deterministic source slot reserved by a link-index builder.
///
/// Slots are stable within a builder. Parallel workers fill distinct slots so
/// provider order and diagnostics do not depend on fill scheduling order.
typedef struct loomc_link_index_source_slot_t {
  /// Deterministic provider slot ordinal reserved by the builder.
  loomc_host_size_t ordinal;
} loomc_link_index_source_slot_t;

/// Source provider options used when adding sources to a link index.
///
/// @lifetime
/// String views are borrowed for the duration of the builder call that consumes
/// this descriptor unless that call explicitly documents a copy.
typedef struct loomc_link_index_source_options_t {
  /// Stable provider label for diagnostics and private-name determinism.
  loomc_string_view_t provider_name;

  /// Provider search precedence role.
  loomc_link_provider_role_t role;
} loomc_link_index_source_options_t;

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

  /// Provider search precedence role.
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

/// Reserves a deterministic source slot.
///
/// @param builder Builder to mutate.
/// @param options Provider options copied by this call, or `NULL` for defaults.
/// @param out_slot Receives the reserved source slot.
/// @return OK when the slot was reserved.
///
/// @thread_safety
/// Reserving slots requires exclusive access to `builder`.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_reserve_source_slot(
    loomc_link_index_builder_t* builder,
    const loomc_link_index_source_options_t* options,
    loomc_link_index_source_slot_t* out_slot);

/// Fills an already reserved source slot.
///
/// @param builder Builder that owns `slot`.
/// @param slot Source slot previously returned by
/// `loomc_link_index_builder_reserve_source_slot`.
/// @param source Source retained by the builder on success.
/// @return OK when the slot was filled.
///
/// @thread_safety
/// Distinct reserved slots may be filled concurrently when each slot is filled
/// by at most one thread and no thread calls reserve or finish concurrently.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_fill_source_slot(
    loomc_link_index_builder_t* builder, loomc_link_index_source_slot_t slot,
    loomc_source_t* source);

/// Reserves and fills the next source slot.
///
/// @param builder Builder to mutate.
/// @param source Source retained by the builder on success.
/// @param options Provider options copied by this call, or `NULL` for defaults.
/// @param out_slot Receives the source slot, or `NULL` if not needed.
/// @return OK when the source was queued for indexing.
LOOMC_API_EXPORT loomc_status_t loomc_link_index_builder_add_source(
    loomc_link_index_builder_t* builder, loomc_source_t* source,
    const loomc_link_index_source_options_t* options,
    loomc_link_index_source_slot_t* out_slot);

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

/// Looks up the selected global symbol by name.
///
/// @param link_index Index to inspect.
/// @param name Symbol name with or without a leading `@`.
/// @param out_symbol Receives selected symbol metadata.
/// @return True when a global symbol named `name` was found.
LOOMC_API_EXPORT bool loomc_link_index_lookup_global(
    const loomc_link_index_t* link_index, loomc_string_view_t name,
    loomc_link_index_symbol_t* out_symbol);

/// Returns the next duplicate global symbol for `symbol`.
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
