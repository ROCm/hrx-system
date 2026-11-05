// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_HTTP_H_
#define LOOM_SERVE_SERVICE_FINITE_HTTP_H_

#include "loom/serve/service/service.h"
#include "loom/service/finite/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adds HTTP routing and request delivery to the real finite service. Copies
// borrowed capabilities without allocating adapter state. The service remains
// owned by its registration; accepted connections retain their input through
// the same source/CPU retirement as native requests.
loom_serve_service_t loom_serve_finite_http_interface(
    loom_service_finite_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_FINITE_HTTP_H_
