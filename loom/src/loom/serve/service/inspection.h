// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_INSPECTION_H_
#define LOOM_SERVE_SERVICE_INSPECTION_H_

#include "loom/service/inspection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_http_inspection_kind_e {
  LOOM_SERVE_HTTP_INSPECT_REQUESTS,
  LOOM_SERVE_HTTP_INSPECT_SESSIONS,
} loom_serve_http_inspection_kind_t;

// Parses a nonempty decimal owner-table offset without signs or wrapping.
bool loom_serve_http_inspection_offset(iree_string_view_t value,
                                       iree_host_size_t* out_offset);

// Appends data and next_offset JSON members, each with a leading comma. The
// caller already selected a present capability and owns the model through this
// call. Examines at most 64 slots including holes; next_offset is null once the
// table end is observed. No query or record survives the call. |now| is shared
// by every age in this page. Failure can only come from output allocation.
iree_status_t loom_serve_http_inspection_format(
    const loom_service_inspection_t* inspection, void* self,
    loom_serve_http_inspection_kind_t kind, iree_host_size_t offset,
    iree_time_t now, iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_INSPECTION_H_
