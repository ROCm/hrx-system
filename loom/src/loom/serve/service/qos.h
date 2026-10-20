// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_QOS_H_
#define LOOM_SERVE_SERVICE_QOS_H_

#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/transport/http/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reads the inference request's X-Loom-QoS extension: background or realtime.
// An absent header selects background. Unknown or empty values are malformed
// input. This class follows the request through preparation and execution;
// it neither changes source compatibility nor promises a latency bound.
iree_status_t loom_serve_qos_parse(const loom_serve_http_request_t* request,
                                   loom_serve_execution_qos_t* out_qos);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_QOS_H_
