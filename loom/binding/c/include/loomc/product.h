// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PRODUCT_H_
#define LOOMC_PRODUCT_H_

#include "loomc/source.h"

/// @file
/// Durable requests published while constructing compiler products.
///
/// A request is an independently compilable ordinary Loom bytecode source.
/// It owns the exact source roots selected by its producer and the provisional
/// bindings from requirements in the parent product to those roots. Requests
/// retain no mutable module, workspace, link plan, or compiler analysis state.
///
/// The defining operations of the requested roots provide the extensible
/// compilation route after the source is indexed. No closed target-kind value
/// or filename convention participates in routing.

#ifdef __cplusplus
extern "C" {
#endif

/// Parent-product-local requirement ordinal.
typedef uint32_t loomc_requirement_ordinal_t;

/// Invalid parent-product requirement ordinal.
#define LOOMC_REQUIREMENT_ORDINAL_INVALID UINT32_MAX

/// Request-local root ordinal.
typedef uint32_t loomc_request_root_ordinal_t;

/// Invalid request-local root ordinal.
#define LOOMC_REQUEST_ROOT_ORDINAL_INVALID UINT32_MAX

/// Durable address of one root in a request's bytecode source.
///
/// Ordinals are source-local and are never compared across different bytecode
/// contents.
typedef struct loomc_request_root_t {
  /// Bytecode module ordinal containing the root.
  uint32_t module_ordinal;

  /// Bytecode SYMBOLS ordinal naming the root.
  uint32_t symbol_ordinal;
} loomc_request_root_t;

/// Provisional association between a parent requirement and a request root.
///
/// One request may bind several requirements, and several bindings may name
/// the same root. Bindings become part of the parent product only when the
/// parent operation succeeds.
typedef struct loomc_request_binding_t {
  /// Requirement ordinal in the parent product.
  loomc_requirement_ordinal_t requirement_ordinal;

  /// Root ordinal in the request.
  loomc_request_root_ordinal_t root_ordinal;
} loomc_request_binding_t;

/// Immutable independently compilable product request.
///
/// @thread_safety
/// Requests are immutable after publication. Retained handles may be queued,
/// cached, and shared across threads independently of the producer lifetime.
typedef struct loomc_request_t loomc_request_t;

/// Accepts ownership of one published request at callback entry.
///
/// @param user_data Callback-owned state.
/// @param request Request reference transferred to the callback.
/// @return OK to continue publication or a non-OK status to terminate the
/// parent operation.
///
/// @ownership
/// The callback owns `request` at entry, including when it returns a non-OK
/// status. It must release or transfer that reference.
///
/// @par Transaction
/// Publication is provisional until the parent operation returns OK with a
/// succeeded result. Accepted requests remain receiver-owned and independently
/// usable when the parent operation is cancelled, but their parent bindings
/// must not be committed.
typedef loomc_status_t(LOOMC_API_PTR* loomc_request_publish_fn_t)(
    void* user_data, loomc_request_t* request);

/// Optional destination for streamed product requests.
typedef struct loomc_request_sink_t {
  /// Callback accepting ownership of each published request.
  loomc_request_publish_fn_t publish;

  /// Opaque value passed to `publish`.
  void* user_data;
} loomc_request_sink_t;

/// Retains a request for another owner.
///
/// @param request Request to retain. Passing NULL is allowed.
LOOMC_API_EXPORT void loomc_request_retain(loomc_request_t* request);

/// Releases a request.
///
/// @param request Request to release. Passing NULL is allowed.
LOOMC_API_EXPORT void loomc_request_release(loomc_request_t* request);

/// Returns the immutable bytecode source owned by `request`.
///
/// @param request Request to inspect.
/// @return Borrowed bytecode source, or NULL for a NULL request.
///
/// @lifetime
/// The source remains valid while `request` is retained. Callers that need an
/// independent source reference retain it with `loomc_source_retain`.
LOOMC_API_EXPORT loomc_source_t* loomc_request_source(
    const loomc_request_t* request);

/// Returns the number of canonical roots in `request`.
///
/// @param request Request to inspect, or NULL.
/// @return Root count, or zero for a NULL request.
LOOMC_API_EXPORT loomc_host_size_t
loomc_request_root_count(const loomc_request_t* request);

/// Returns a request root by request-local ordinal.
///
/// @param request Request to inspect.
/// @param ordinal Request-local root ordinal.
/// @param out_root Receives the root when found.
/// @return True when `ordinal` was valid and `out_root` was populated.
LOOMC_API_EXPORT bool loomc_request_root_at(
    const loomc_request_t* request, loomc_request_root_ordinal_t ordinal,
    loomc_request_root_t* out_root);

/// Returns the number of provisional parent bindings in `request`.
///
/// @param request Request to inspect, or NULL.
/// @return Binding count, or zero for a NULL request.
LOOMC_API_EXPORT loomc_host_size_t
loomc_request_binding_count(const loomc_request_t* request);

/// Returns a provisional parent binding by ordinal.
///
/// @param request Request to inspect.
/// @param ordinal Zero-based binding ordinal.
/// @param out_binding Receives the binding when found.
/// @return True when `ordinal` was valid and `out_binding` was populated.
LOOMC_API_EXPORT bool loomc_request_binding_at(
    const loomc_request_t* request, loomc_host_size_t ordinal,
    loomc_request_binding_t* out_binding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PRODUCT_H_
