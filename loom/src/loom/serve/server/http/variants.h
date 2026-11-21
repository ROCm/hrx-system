// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_HTTP_VARIANTS_H_
#define LOOM_SERVE_SERVER_HTTP_VARIANTS_H_

#include "loom/serve/server/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_http_variants_t loom_serve_http_variants_t;

// Cold management owner, serialized with routing and service advancement.
// Registry and transport outlive it. Observers retain identities, not model
// pointers; accepted selection callbacks own independent completion storage.
iree_status_t loom_serve_http_variants_allocate(
    loom_serve_http_server_t* server, loom_serve_registry_t* registry,
    iree_allocator_t host_allocator, loom_serve_http_variants_t** out_variants);

// Requires no pending observers or callbacks. NULL is accepted.
void loom_serve_http_variants_free(loom_serve_http_variants_t* variants);
bool loom_serve_http_variants_pending(
    const loom_serve_http_variants_t* variants);
iree_host_size_t loom_serve_http_variants_claims(
    const loom_serve_http_variants_t* variants);

// Recognizes variant management. On GET, X-Loom-Model selects inventory at
// the path also used by the catalog for a deployment named "variants".
bool loom_serve_http_variants_handles(const loom_serve_http_request_t* request);

// Consumes the request claim by responding or retaining an asynchronous
// observer. allow_wait reflects shared transport credit, including other
// management readers. Errors from public input/model policy are responses;
// only response allocation failures terminate the serving owner.
// Staging accepts JSON locators or a multipart source package (see package.h).
// An accepted service retains its input independently of the transport claim.
iree_status_t loom_serve_http_variants_accept(
    loom_serve_http_variants_t* variants,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, bool allow_wait,
    iree_string_builder_t* scratch);

// Runs after this slot's service advance and before registry retirement.
// Copies terminal observations to transport without extending model lifetime.
// Disconnected compilation readers retire immediately; accepted selection
// callbacks still retire normally. Returns errors only from response storage.
iree_status_t loom_serve_http_variants_advance(
    loom_serve_http_variants_t* variants, iree_host_size_t model_index,
    bool stopping, iree_string_builder_t* scratch, bool* out_progress);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_HTTP_VARIANTS_H_
