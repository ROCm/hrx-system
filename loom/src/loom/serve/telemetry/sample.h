// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TELEMETRY_SAMPLE_H_
#define LOOM_SERVE_TELEMETRY_SAMPLE_H_

#include "loom/runtime/telemetry/sample.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends one self-describing topic object. Integer values are decimal strings
// so JavaScript clients can difference counters without losing low bits.
// This is consumer-side projection; producers have no JSON dependency.
iree_status_t loom_serve_telemetry_sample_write_json(
    const loom_runtime_telemetry_topic_t* topic, const uint64_t* values,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_TELEMETRY_SAMPLE_H_
