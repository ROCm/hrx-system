// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable integer identities derived from textual keys.
//
// Stable IDs are used when text syntax needs an ergonomic symbolic spelling but
// compiled or serialized forms need a compact durable identity. The hash is
// deterministic, independent of table ordering, and masked into the positive
// signed 63-bit range so it can travel through existing i64 attribute paths.

#ifndef LOOM_UTIL_STABLE_ID_H_
#define LOOM_UTIL_STABLE_ID_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for an absent stable ID. Valid stable IDs are never zero.
#define LOOM_STABLE_ID_NONE UINT64_C(0)

// Returns a stable non-zero 63-bit identity for |key|.
uint64_t loom_stable_id_from_string(iree_string_view_t key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_STABLE_ID_H_
