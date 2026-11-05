// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_HTTP_PACKAGE_H_
#define LOOM_SERVE_SERVER_HTTP_PACKAGE_H_

#include "loom/runtime/program/package.h"
#include "loom/serve/transport/http/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decodes either ordinary JSON (no package) or multipart/form-data containing
// exactly one 'request' JSON part and repeated 'file' parts. A file's filename
// is its portable package path. The request filename defaults to request.json
// and establishes the document directory; payload file names are never local
// filesystem destinations. Parts may arrive in any order. JSON bytes remain
// borrowed from request; the returned package independently owns all files.
// Multipart fields are already parsed and retained by the HTTP request owner.
//
// Uses the transport's body, MIME-header and part budgets, with no implicit
// larger limit or checkpoint upload path. Failure publishes no package.
// Content-Type may be absent for ordinary JSON clients; unsupported explicit
// types reject.
iree_status_t loom_serve_http_package_parse(
    const loom_serve_http_request_t* request, iree_allocator_t host_allocator,
    iree_string_view_t* out_json, loom_runtime_package_t** out_package);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_HTTP_PACKAGE_H_
