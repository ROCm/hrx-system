// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_BINDING_H_
#define LOOM_SERVE_SERVER_REALTIME_BINDING_H_

#include "loom/serve/server/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copied deployment identity. The configuration supplying name outlives the
// session; no model, service, parameter, or executable pointer is retained.
typedef struct loom_serve_realtime_binding_t {
  // Stable registry ordinal, reused only with a different incarnation.
  iree_host_size_t index;
  // Registration lifetime; zero denotes an immutable borrowed service.
  uint64_t incarnation;
  // Configuration-owned deployment selector, independent of registry storage.
  iree_string_view_t name;
} loom_serve_realtime_binding_t;

// Resolves metadata without preparing or activating the deployment. Native
// admission still owns readiness, queue pressure, and source capabilities.
// kinds uses model-kind bit positions, as native registry admission does.
iree_status_t loom_serve_realtime_binding_resolve(
    loom_serve_registry_t* registry, iree_string_view_t name, uint32_t kinds,
    loom_serve_realtime_binding_t* out_binding);

// Checks the copied identity before another admission or checkpoint operation.
// An accepted stream already owns its result independently of this check.
bool loom_serve_realtime_binding_present(
    loom_serve_registry_t* registry,
    const loom_serve_realtime_binding_t* binding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_BINDING_H_
