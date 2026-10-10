// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_SIGNALS_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_SIGNALS_H_

#include "iree/async/util/proactor_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Application-owned interrupt/termination subscriptions on shared I/O threads.
// Independent of HTTP listeners and model lifetimes.
typedef struct loom_serve_signals_t loom_serve_signals_t;

// Installs |callback| on pool entry zero and joins registration. Called only
// from outer application startup, after iree_async_signal_block_default has
// been called before thread creation. The callback runs on the poll owner and
// only publishes shutdown intent; it never waits or destroys shared resources.
// Its storage remains valid until destroy returns.
iree_status_t loom_serve_signals_create(iree_async_proactor_pool_t* pool,
                                        iree_async_signal_callback_t callback,
                                        loom_serve_signals_t** out_signals,
                                        iree_allocator_t allocator);

// Unsubscribes on the poll owner and joins its final callback access before
// releasing progress ownership. Outer application teardown only; NULL is safe.
// The shared I/O owner must remain capable of accepting the teardown handoff.
void loom_serve_signals_destroy(loom_serve_signals_t* signals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_SIGNALS_H_
