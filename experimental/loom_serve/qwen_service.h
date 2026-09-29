// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SERVICE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SERVICE_H_

#include "experimental/loom_serve/http_server.h"
#include "experimental/loom_serve/qwen_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs one application owner until transport shutdown or model failure. Model
// and transport are borrowed. Each turn advances at most one prefill chunk or
// decode step per ready row, with bounded copied SSE output between stages.
// X-Loom-Session selects retained state; idle rows are an LRU prefix cache, not
// durable sessions. Busy named sessions reject concurrent requests. Untagged
// requests always replay. Peer cancellation discards its checkpoint at a
// completed stage boundary. Return relinquishes every connection view before
// the caller destroys transport/model. Capacities match the created model.
iree_status_t loom_serve_qwen_service_run(loom_serve_qwen_model_t* model,
                                          loom_serve_http_server_t* server,
                                          iree_host_size_t row_count,
                                          iree_host_size_t chunk_size,
                                          iree_host_size_t default_max_tokens,
                                          iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_QWEN_SERVICE_H_
