// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_PROGRAM_H_
#define LOOMC_TARGET_CMD_PROGRAM_H_

#include "loomc/artifact.h"
#include "loomc/config.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

/// @file
/// Portable command-program products and independent kernel requests.
///
/// Command product construction selects command roots from a reusable frozen
/// link index, lowers their host-side orchestration to portable command bytes,
/// and optionally publishes independently compilable kernel source modules.
/// The operation never prints or reparses a module. Exact root ordinals let a
/// high-throughput embedding resolve human-readable names once at ingress and
/// use indexed identity throughout planning.
///
/// Without a request sink, construction leaves kernel implementation facets
/// unopened. With a sink, it classifies only surviving source-backed launch
/// sites and publishes one ordinary module per live semantic class. Published
/// modules transfer independently and are never retained by the command
/// product.
///
/// @par Example
/// Build one command product from an already resolved root:
///
/// @code{.c}
/// loomc_cmd_program_product_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
///     .structure_size = sizeof(options),
///     .link_index = link_index,
///     .root_symbol_ordinals = &root_symbol_ordinal,
///     .root_symbol_count = 1,
/// };
/// loomc_cmd_program_product_t* product = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_cmd_program_product_build(
///     workspace, &options, loomc_allocator_system(), &product, &result);
/// if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
///   loomc_cmd_program_t program;
///   if (loomc_cmd_program_product_program_at(product, 0, &program)) {
///     // Load or cache program.artifact.contents.
///   }
/// }
/// loomc_result_release(result);
/// loomc_cmd_program_product_release(product);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Stable format name for serialized portable command-program bytes.
#define LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM "loom-command"

/// Immutable command-program product.
///
/// Products own their serialized program bytes, root projections, and copied
/// symbol names. They retain no source module, link index, workspace, compiler
/// plan, or analysis state.
///
/// @thread_safety
/// Products are immutable after construction. Retained handles may be shared
/// across threads.
typedef struct loomc_cmd_program_product_t loomc_cmd_program_product_t;

/// Command-product operation flag bits.
typedef enum loomc_cmd_program_product_flag_bits_e {
  /// Select exported command definitions from INPUT providers in addition to
  /// explicitly supplied roots.
  LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS = 1u << 0,
} loomc_cmd_program_product_flag_bits_t;

/// Bitmask of `loomc_cmd_program_product_flag_bits_t` values.
typedef uint32_t loomc_cmd_program_product_flags_t;

/// One independently compilable kernel source request.
///
/// The request module is ordinary Loom IR and may be passed directly to the
/// module query, serialization, compilation, and emission APIs. It contains
/// one exported kernel root specialized for every launch site represented by
/// this semantic class.
typedef struct loomc_cmd_kernel_request_t {
  /// Product-local executable-entry requirement satisfied by this request.
  uint32_t entry_requirement_ordinal;

  /// Exported kernel root in `module`, without a leading `@`.
  ///
  /// This view borrows from `module` and remains valid until that module is
  /// released or mutated.
  loomc_string_view_t root_symbol;

  /// Number of command-program launch sites represented by this request.
  loomc_host_size_t member_count;

  /// Independently owned ordinary Loom source module.
  loomc_module_t* module;
} loomc_cmd_kernel_request_t;

/// Accepts ownership of one kernel source request at callback entry.
///
/// @param user_data Callback-owned state.
/// @param request Request whose `module` reference transfers to the callback.
/// @return OK to continue publication or a non-OK status to terminate the
/// parent operation.
///
/// @ownership
/// The callback owns `request.module` at entry, including when it returns a
/// non-OK status. It must release or transfer that module reference.
///
/// @par Transaction
/// Publication is provisional until the parent operation returns OK with a
/// succeeded result. A non-OK status or failed result cancels the parent
/// product; the embedding remains responsible for requests it accepted before
/// cancellation.
typedef loomc_status_t(LOOMC_API_PTR* loomc_cmd_kernel_request_publish_fn_t)(
    void* user_data, loomc_cmd_kernel_request_t request);

/// Optional destination for streamed kernel source requests.
typedef struct loomc_cmd_kernel_request_sink_t {
  /// Callback accepting ownership of each published request.
  loomc_cmd_kernel_request_publish_fn_t publish;

  /// Opaque value passed to `publish`.
  void* user_data;
} loomc_cmd_kernel_request_sink_t;

/// Command-product construction options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_cmd_program_product_options_t)`, and fill the requested
/// fields.
typedef struct loomc_cmd_program_product_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Optional invocation extensions such as
  /// `loomc_target_specialization_options_t`.
  const void* next;

  /// Frozen provider index containing command and kernel sources.
  loomc_link_index_t* link_index;

  /// Index-wide symbol ordinals of explicit command roots.
  ///
  /// Every ordinal must expose a command implementation. Root order is
  /// retained in the returned product, including repeated explicit roots.
  /// Automatically selected INPUT exports already named here are not appended
  /// a second time.
  const loomc_host_size_t* root_symbol_ordinals;

  /// Number of entries in `root_symbol_ordinals`.
  loomc_host_size_t root_symbol_count;

  /// Product construction flags.
  loomc_cmd_program_product_flags_t flags;

  /// Per-invocation config bindings applied during selective materialization.
  loomc_config_options_t config;

  /// Optional sink enabling source-backed kernel request publication.
  ///
  /// A NULL callback leaves kernel implementation bodies unopened and returns
  /// only executable-entry requirements.
  loomc_cmd_kernel_request_sink_t kernel_request_sink;
} loomc_cmd_program_product_options_t;

/// One serialized command root and its executable-entry projection.
///
/// @lifetime
/// Every view borrows from the product that returned this record and remains
/// valid until that product is released.
typedef struct loomc_cmd_program_t {
  /// Root symbol without a leading `@`.
  loomc_string_view_t symbol;

  /// Portable command-program artifact.
  ///
  /// The artifact identifier is `symbol`, its kind is
  /// `LOOMC_ARTIFACT_KIND_EXECUTABLE`, and its format is
  /// `LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM`.
  loomc_artifact_t artifact;

  /// Product-local entry requirement ordinals in root-local slot order.
  const uint32_t* entry_requirement_ordinals;

  /// Number of entries in `entry_requirement_ordinals`.
  loomc_host_size_t entry_requirement_count;
} loomc_cmd_program_t;

/// One product-wide executable-entry binding requirement.
///
/// @lifetime
/// `symbol` borrows from the product that returned this record and remains
/// valid until that product is released.
typedef struct loomc_cmd_entry_requirement_t {
  /// Logical kernel entry symbol without a leading `@`.
  loomc_string_view_t symbol;

  /// True when the operation published an ordinary Loom source request for
  /// this exact semantic class.
  bool has_source_request;
} loomc_cmd_entry_requirement_t;

/// Builds portable command programs and optional kernel source requests.
///
/// @param workspace Invocation-local compiler workspace.
/// @param options Product selection and specialization options.
/// @param allocator Host allocator used for returned product and result state.
/// @param out_product Receives one retained product when the result succeeds;
/// receives NULL when the result fails.
/// @param out_result Receives one retained result for the operation.
/// @return OK when the operation ran to a result. Non-OK statuses represent
/// structural API misuse, callback failure, or infrastructure failure.
///
/// @ownership
/// The caller owns `out_product` when non-NULL and releases it with
/// `loomc_cmd_program_product_release`. The caller owns `out_result` on an OK
/// return and releases it with `loomc_result_release`. Each request module is
/// transferred independently to `options->kernel_request_sink`.
///
/// @lifetime
/// The operation borrows `options`, its arrays, `link_index`, and `workspace`
/// synchronously. The returned product borrows none of them. Request modules
/// retain `workspace` through their normal module ownership contract.
///
/// @thread_safety
/// Calls may share a frozen link index. Each concurrent call requires a
/// distinct workspace. Callback synchronization is owned by the embedding.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_product_build(
    loomc_workspace_t* workspace,
    const loomc_cmd_program_product_options_t* options,
    loomc_allocator_t allocator, loomc_cmd_program_product_t** out_product,
    loomc_result_t** out_result);

/// Retains a command product for another owner.
///
/// @param product Product to retain.
LOOMC_API_EXPORT void loomc_cmd_program_product_retain(
    loomc_cmd_program_product_t* product);

/// Releases a command product.
///
/// @param product Product to release. Passing NULL is allowed.
LOOMC_API_EXPORT void loomc_cmd_program_product_release(
    loomc_cmd_program_product_t* product);

/// Returns the number of serialized command roots in `product`.
///
/// @param product Product to query.
/// @return Number of serialized command roots.
LOOMC_API_EXPORT loomc_host_size_t loomc_cmd_program_product_program_count(
    const loomc_cmd_program_product_t* product);

/// Returns serialized command-root metadata by product-local ordinal.
///
/// @param product Product to query.
/// @param ordinal Product-local root ordinal.
/// @param out_program Receives the borrowed root record when found.
/// @return True when `ordinal` was valid and `out_program` was populated.
LOOMC_API_EXPORT bool loomc_cmd_program_product_program_at(
    const loomc_cmd_program_product_t* product, loomc_host_size_t ordinal,
    loomc_cmd_program_t* out_program);

/// Returns the number of product-wide executable-entry requirements.
///
/// @param product Product to query.
/// @return Number of product-wide executable-entry requirements.
LOOMC_API_EXPORT loomc_host_size_t
loomc_cmd_program_product_entry_requirement_count(
    const loomc_cmd_program_product_t* product);

/// Returns an executable-entry requirement by product-local ordinal.
///
/// @param product Product to query.
/// @param ordinal Product-local entry-requirement ordinal.
/// @param out_requirement Receives the borrowed requirement when found.
/// @return True when `ordinal` was valid and `out_requirement` was populated.
LOOMC_API_EXPORT bool loomc_cmd_program_product_entry_requirement_at(
    const loomc_cmd_program_product_t* product, loomc_host_size_t ordinal,
    loomc_cmd_entry_requirement_t* out_requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_PROGRAM_H_
