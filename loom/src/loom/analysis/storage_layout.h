// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Checked sequential byte layout for storage requirements.

#ifndef LOOM_ANALYSIS_STORAGE_LAYOUT_H_
#define LOOM_ANALYSIS_STORAGE_LAYOUT_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends one byte range to a sequential storage layout.
//
// |byte_alignment| must be a nonzero power of two. The range begins at the
// first aligned byte at or after |*inout_byte_extent|. On success the range
// offset is returned in |out_byte_offset| when non-NULL and
// |inout_byte_extent| advances past the range. Overflow leaves the input
// extent unchanged and returns false. The owning verifier or planner diagnoses
// overflow with its source operation; arithmetic does not produce statuses.
bool loom_storage_layout_append(uint64_t byte_length, uint64_t byte_alignment,
                                uint64_t* inout_byte_extent,
                                uint64_t* out_byte_offset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_STORAGE_LAYOUT_H_
