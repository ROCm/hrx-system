// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PRODUCT_H_
#define LOOMC_PRODUCT_H_

#include "loomc/artifact.h"
#include "loomc/source.h"

/// @file
/// Immutable compiler products and their recursively published requests.
///
/// A product is the immutable value returned by a successful compiler
/// operation. It owns that operation's artifacts and exposes the two ordinal
/// spaces needed for composition: exported roots that may satisfy a parent
/// binding and requirements that must be satisfied by child products or an
/// embedding. Product-specific APIs project richer metadata from the same
/// handle without changing its ownership model.
///
/// `loomc_result_t` and `loomc_product_t` deliberately answer different
/// questions. A result reports whether an operation succeeded and carries its
/// diagnostics. A product exists only on success and carries the immutable
/// value produced by the operation.
///
/// A request is an independently compilable ordinary Loom bytecode source.
/// It owns the exact source roots selected by its producer and the provisional
/// bindings from requirements in the parent product to those roots. Requests
/// retain no mutable module, workspace, link plan, or compiler analysis state.
///
/// Every request names the process-local product descriptor it expects a
/// successful pipeline to produce. This is the open compilation route used by
/// hosts: command programs, compiled modules, and future product families
/// add descriptors without extending a central kind enumeration. Root
/// operations remain the durable source contract and are validated by the
/// selected product operation. Symbol strings, filenames, and target names do
/// not participate in routing.

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque process-local identity for one product representation.
///
/// Descriptor pointers are stable for the lifetime of the process and may be
/// compared by identity. They are never serialized. A request uses the
/// descriptor to name its required successful representation while its source
/// roots retain the durable operation contracts compiled by that route.
typedef struct loomc_product_descriptor_t loomc_product_descriptor_t;

/// Immutable successful compiler product.
///
/// Product artifacts, exported roots, and requirements use product-local
/// ordinals. An operation-specific API defines the metadata and meaning of
/// each root and requirement. A request binding associates one requirement
/// ordinal in a parent product with one root ordinal in a child request; the
/// resulting child product preserves that request-root order as its exported
/// root order.
///
/// @thread_safety
/// Products are immutable after construction. Retained handles may be queued,
/// cached, and shared across threads independently of the producing compiler
/// invocation.
typedef struct loomc_product_t loomc_product_t;

/// Retains a product for another owner.
///
/// @param product Product to retain. Passing NULL is allowed.
LOOMC_API_EXPORT void loomc_product_retain(loomc_product_t* product);

/// Releases a product.
///
/// @param product Product to release. Passing NULL is allowed.
LOOMC_API_EXPORT void loomc_product_release(loomc_product_t* product);

/// Returns the process-local descriptor identifying `product`.
///
/// @param product Product to inspect, or NULL.
/// @return Product descriptor, or NULL for a NULL product.
LOOMC_API_EXPORT const loomc_product_descriptor_t* loomc_product_descriptor(
    const loomc_product_t* product);

/// Returns the number of artifacts owned by `product`.
///
/// @param product Product to inspect, or NULL.
/// @return Artifact count, or zero for a NULL product.
LOOMC_API_EXPORT loomc_host_size_t
loomc_product_artifact_count(const loomc_product_t* product);

/// Returns an artifact by product-local ordinal.
///
/// @param product Product to inspect.
/// @param ordinal Zero-based artifact ordinal.
/// @return Borrowed artifact view, or NULL when `ordinal` is out of range.
///
/// @lifetime
/// The returned view and its strings remain valid while `product` is retained.
/// Callers that need independent byte ownership retain `artifact->contents`.
LOOMC_API_EXPORT const loomc_artifact_t* loomc_product_artifact_at(
    const loomc_product_t* product, loomc_host_size_t ordinal);

/// Returns the number of roots exported by `product`.
///
/// Export ordinals preserve the root order of the request or direct build that
/// produced the product. Product-specific APIs expose root metadata.
///
/// @param product Product to inspect, or NULL.
/// @return Exported-root count, or zero for a NULL product.
LOOMC_API_EXPORT loomc_host_size_t
loomc_product_export_count(const loomc_product_t* product);

/// Returns the number of unresolved requirements in `product`.
///
/// Requirements are addressed by `loomc_requirement_ordinal_t`. Their typed
/// contracts are exposed by product-specific APIs; source requests carry the
/// provisional child bindings needed by a composing host.
///
/// @param product Product to inspect, or NULL.
/// @return Requirement count, or zero for a NULL product.
LOOMC_API_EXPORT loomc_host_size_t
loomc_product_requirement_count(const loomc_product_t* product);

/// Parent-product-local requirement ordinal.
typedef uint32_t loomc_requirement_ordinal_t;

/// Invalid parent-product requirement ordinal.
#define LOOMC_REQUIREMENT_ORDINAL_INVALID UINT32_MAX

/// Request-local root ordinal.
typedef uint32_t loomc_request_root_ordinal_t;

/// Invalid request-local root ordinal.
#define LOOMC_REQUEST_ROOT_ORDINAL_INVALID UINT32_MAX

/// Product-descriptor-local semantic goal for one request root.
///
/// Goal values are interpreted only by the product operation named by the
/// request descriptor. Zero is the default goal for every descriptor. Product
/// packages define any additional values in their own public headers.
typedef uint32_t loomc_request_root_goal_t;

/// Default root goal defined by every product descriptor.
#define LOOMC_REQUEST_ROOT_GOAL_DEFAULT 0u

/// Durable identity of one root in a request's bytecode source.
///
/// The source ordinals address the defining operation while `goal` identifies
/// the semantic value requested from the selected product operation. Ordinals
/// are source-local and are never compared across different bytecode contents.
/// Goal values are meaningful only with the request's product descriptor.
typedef struct loomc_request_root_t {
  /// Bytecode module ordinal containing the root.
  uint32_t module_ordinal;

  /// Bytecode SYMBOLS ordinal naming the root.
  uint32_t symbol_ordinal;

  /// Product-descriptor-local semantic goal for the root.
  loomc_request_root_goal_t goal;

  /// Reserved for future use and must be zero.
  uint32_t reserved;
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

/// Creates an immutable request over ordinary Loom bytecode.
///
/// @param product_descriptor Process-lifetime descriptor for the required
/// successful product representation.
/// @param source Bytecode source retained by the request.
/// @param roots Source-local roots copied in product export order. Reserved
/// fields must be zero. Goal values are validated by the selected product
/// operation.
/// @param root_count Number of entries in `roots`; must be nonzero.
/// @param bindings Optional parent-requirement bindings copied into the
/// request. Entries must have strictly increasing requirement ordinals and
/// refer to valid request-local roots.
/// @param binding_count Number of entries in `bindings`.
/// @param allocator Host allocator used for request and metadata storage.
/// @param out_request Receives one retained request.
/// @return OK when the immutable request was created.
///
/// Root addresses are validated against the bytecode source by the operation
/// that consumes the request. This constructor performs no source indexing or
/// module scan, so a caller may resolve human-readable names once through a
/// link index and form many requests by exact ordinal.
///
/// @ownership
/// The caller retains `source` and owns the returned request. Release the
/// request with `loomc_request_release`.
///
/// @thread_safety
/// Creation is local to the caller. The returned request is immutable and may
/// be retained and shared across threads.
LOOMC_API_EXPORT loomc_status_t loomc_request_create(
    const loomc_product_descriptor_t* product_descriptor,
    loomc_source_t* source, const loomc_request_root_t* roots,
    loomc_host_size_t root_count, const loomc_request_binding_t* bindings,
    loomc_host_size_t binding_count, loomc_allocator_t allocator,
    loomc_request_t** out_request);

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

/// Returns the required product representation for `request`.
///
/// The descriptor is part of the immutable request contract. A pipeline
/// selected for the request must produce a product with this descriptor on
/// success. Descriptor identity is process-local and is never serialized into
/// the request source.
///
/// @param request Request to inspect, or NULL.
/// @return Required product descriptor, or NULL for a NULL request.
LOOMC_API_EXPORT const loomc_product_descriptor_t*
loomc_request_product_descriptor(const loomc_request_t* request);

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
