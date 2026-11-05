// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TELEMETRY_RECORD_H_
#define LOOM_SERVE_TELEMETRY_RECORD_H_

#include "loom/runtime/telemetry/record.h"
#include "loom/service/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consumer-side JSON formatting of an owned record and its absolute position.
// Model code may define additional static event schemas without HTTP coupling.
iree_status_t loom_serve_telemetry_record_format(
    uint64_t position, const loom_runtime_telemetry_record_t* record,
    iree_string_builder_t* output);
iree_status_t loom_serve_telemetry_counters_format(
    const loom_service_telemetry_counters_t* counters,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_TELEMETRY_RECORD_H_
