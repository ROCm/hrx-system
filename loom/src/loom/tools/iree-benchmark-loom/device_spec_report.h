// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL device-spec JSON projection for iree-benchmark-loom reports.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_DEVICE_SPEC_REPORT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_DEVICE_SPEC_REPORT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes a compact immutable HAL device spec summary as a JSON object.
iree_status_t iree_benchmark_loom_write_device_spec_json(
    const iree_hal_device_spec_t* device_spec, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_DEVICE_SPEC_REPORT_H_
