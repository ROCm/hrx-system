// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_PROGRAM_H_
#define LOOMC_TARGET_CMD_PROGRAM_H_

#include "loomc/artifact.h"
#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/product.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

/// @file
/// Portable command-program products and independent kernel requests.
///
/// Command product construction selects command roots from a reusable frozen
/// link index, lowers their host-side orchestration to portable command bytes,
/// and optionally publishes independently compilable kernel requests.
/// The operation never prints or reparses a module. Exact root ordinals let a
/// high-throughput embedding resolve human-readable names once at ingress and
/// use indexed identity throughout planning.
///
/// Without a request sink, construction leaves kernel implementation facets
/// unopened. With a sink, it classifies only surviving source-backed launch
/// sites and publishes one executable-entry kernel request per live semantic
/// class. Published requests require the kernel product descriptor and never
/// request a host launch companion because the command program already owns
/// physical launch geometry. They transfer independently and are never
/// retained by the product.
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
/// loomc_product_t* product = NULL;
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
/// loomc_product_release(product);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Stable format name for serialized portable command-program bytes.
#define LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM "loom-command"

/// Command-product operation flag bits.
typedef enum loomc_cmd_program_product_flag_bits_e {
  /// Select exported command definitions from INPUT providers in addition to
  /// explicitly supplied roots.
  LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS = 1u << 0,
} loomc_cmd_program_product_flag_bits_t;

/// Bitmask of `loomc_cmd_program_product_flag_bits_t` values.
typedef uint32_t loomc_cmd_program_product_flags_t;

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

  /// Optional sink enabling source-backed request publication.
  ///
  /// A NULL callback leaves kernel implementation bodies unopened and returns
  /// only executable-entry requirements.
  loomc_request_sink_t request_sink;
} loomc_cmd_program_product_options_t;

/// Command-product options for an ordinary bytecode request.
///
/// The request source is appended to an invocation-local module-index overlay
/// as its only INPUT provider. An optional frozen library supplies dependencies
/// without being traversed or copied. Request roots select command programs by
/// exact bytecode-local identity and preserve their order and duplicates in the
/// returned product.
typedef struct loomc_cmd_program_request_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_REQUEST_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Optional invocation extensions such as
  /// `loomc_target_specialization_options_t`.
  const void* next;

  /// Optional frozen library-only index used to resolve request declarations.
  /// Passing NULL plans against the request source alone.
  const loomc_link_index_t* library_index;

  /// Per-invocation config bindings applied during selective materialization.
  loomc_config_options_t config;

  /// Optional sink enabling source-backed kernel request publication.
  ///
  /// A NULL callback leaves kernel implementation bodies unopened and returns
  /// only executable-entry requirements.
  loomc_request_sink_t request_sink;
} loomc_cmd_program_request_options_t;

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
  const loomc_requirement_ordinal_t* entry_requirement_ordinals;

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
} loomc_cmd_entry_requirement_t;

/// Builds portable command programs and optional kernel source requests.
///
/// The returned product has the command descriptor, one executable artifact
/// and exported root per selected command root, and one requirement per
/// product-wide executable-entry binding. Command-specific queries expose the
/// root symbols, root-to-requirement projections, and requirement symbols.
/// The product owns all serialized bytes and copied names and retains no source
/// module, link index, workspace, compiler plan, or analysis state.
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
/// `loomc_product_release`. The caller owns `out_result` on an OK return and
/// releases it with `loomc_result_release`. Each request is transferred
/// independently to `options->request_sink`.
///
/// @lifetime
/// The operation borrows `options`, its arrays, `link_index`, and `workspace`
/// synchronously. The returned product and published requests borrow none of
/// them.
///
/// @thread_safety
/// Calls may share a frozen link index. Each concurrent call requires a
/// distinct workspace. Callback synchronization is owned by the embedding.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_product_build(
    loomc_workspace_t* workspace,
    const loomc_cmd_program_product_options_t* options,
    loomc_allocator_t allocator, loomc_product_t** out_product,
    loomc_result_t** out_result);

/// Builds portable command programs from an ordinary bytecode request.
///
/// The operation indexes only `request`, maps its exact roots into an overlay
/// over `options->library_index`, and directly plans the command product. It
/// does not print, reparse names, or serialize an intermediate linked request.
/// The returned product preserves request-root order as export order. Published
/// kernel requests carry provisional bindings to its executable-entry
/// requirements.
///
/// @param context Context used to decode the request and shared with the
/// optional library index.
/// @param workspace Invocation-local compiler workspace.
/// @param request Immutable request requiring a command-program product.
/// @param options Product selection and specialization options, or NULL.
/// @param allocator Host allocator used for returned product and result state.
/// @param out_product Receives one retained product when the result succeeds;
/// receives NULL when the result fails.
/// @param out_result Receives one retained result for the operation.
/// @return OK when the operation ran to a result. Non-OK statuses represent
/// structural API misuse, callback failure, or infrastructure failure.
///
/// @ownership
/// The caller owns `out_product` when non-NULL and releases it with
/// `loomc_product_release`. The caller owns `out_result` on an OK return and
/// releases it with `loomc_result_release`. Each request is transferred
/// independently to `options->request_sink`.
///
/// @lifetime
/// The operation borrows `context`, `workspace`, `request`, `options`, and the
/// optional library index synchronously. The product and published requests
/// retain none of them.
///
/// @thread_safety
/// Calls may share a context, immutable request, and frozen library index. Each
/// concurrent call requires a distinct workspace. Callback synchronization is
/// owned by the embedding.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_product_build_request(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_request_t* request,
    const loomc_cmd_program_request_options_t* options,
    loomc_allocator_t allocator, loomc_product_t** out_product,
    loomc_result_t** out_result);

/// Returns the process-local command-product descriptor.
///
/// The returned pointer may be compared with `loomc_product_descriptor` to
/// identify products before using command-specific projections.
///
/// @return Process-lifetime command-product descriptor.
LOOMC_API_EXPORT const loomc_product_descriptor_t*
loomc_cmd_program_product_descriptor(void);

/// Returns the number of serialized command roots in `product`.
///
/// @param product Product to query.
/// @return Number of serialized command roots.
LOOMC_API_EXPORT loomc_host_size_t
loomc_cmd_program_product_program_count(const loomc_product_t* product);

/// Returns serialized command-root metadata by product-local ordinal.
///
/// @param product Product to query.
/// @param ordinal Product-local root ordinal.
/// @param out_program Receives the borrowed root record when found.
/// @return True when `ordinal` was valid and `out_program` was populated.
LOOMC_API_EXPORT bool loomc_cmd_program_product_program_at(
    const loomc_product_t* product, loomc_host_size_t ordinal,
    loomc_cmd_program_t* out_program);

/// Returns the number of product-wide executable-entry requirements.
///
/// @param product Product to query.
/// @return Number of product-wide executable-entry requirements.
LOOMC_API_EXPORT loomc_host_size_t
loomc_cmd_program_product_entry_requirement_count(
    const loomc_product_t* product);

/// Returns an executable-entry requirement by product-local ordinal.
///
/// @param product Product to query.
/// @param ordinal Product-local entry-requirement ordinal.
/// @param out_requirement Receives the borrowed requirement when found.
/// @return True when `ordinal` was valid and `out_requirement` was populated.
LOOMC_API_EXPORT bool loomc_cmd_program_product_entry_requirement_at(
    const loomc_product_t* product, loomc_host_size_t ordinal,
    loomc_cmd_entry_requirement_t* out_requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_PROGRAM_H_
