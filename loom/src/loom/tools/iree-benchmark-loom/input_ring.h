// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dispatch benchmark input-ring sizing policy.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_INPUT_RING_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_INPUT_RING_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the number of physical binding sets in a dispatch input ring.
//
// A nonzero |requested_count| is exact. Automatic selection otherwise uses
// enough whole binding sets to approach |requested_min_bytes| without
// exceeding |dispatches_per_batch|. This bounds automatic rotation to one
// batch and avoids duplicating a binding set that already meets the byte
// target. The returned count is always at least one.
iree_host_size_t iree_benchmark_loom_input_ring_select_count(
    iree_host_size_t requested_count, uint64_t requested_min_bytes,
    uint64_t binding_set_bytes, iree_host_size_t dispatches_per_batch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_INPUT_RING_H_
