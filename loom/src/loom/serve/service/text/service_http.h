// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_HTTP_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_HTTP_H_

#include "loom/serve/service/service.h"
#include "loom/serve/service/text/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adapts native text/sequence admission and observation to the HTTP registry.
// Borrows the service without allocating another owner or progress loop.
loom_serve_service_t loom_serve_text_http_interface(
    loom_serve_text_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_HTTP_H_
