// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_STORAGE_BOUNDS_H_
#define LOOM_SERVE_SERVICE_TEXT_STORAGE_BOUNDS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// A necessary occupancy constraint in one source allocation. Coordinates and
// minimum use physical mapping units, not token or block IDs. A full interval
// is fixed; a smaller minimum permits any placement within the interval.
typedef struct loom_serve_text_storage_bound_t {
  // Independent source allocation identity.
  iree_host_size_t allocation;
  // Inclusive beginning of the possible placement domain.
  uint64_t begin;
  // Exclusive end of the nonempty placement domain.
  uint64_t end;
  // Required occupied units, at most end - begin.
  uint64_t minimum;
} loom_serve_text_storage_bound_t;

// Returns a necessary, not sufficient, storage bound. Sorts caller-owned
// scratch in place. Disjoint domains add; overlapping movable domains use their
// maximum requirement, while fixed intervals contribute their unique union.
// Thus aliases, fragmentation, and uncertain placement never inflate the bound.
uint64_t loom_serve_text_storage_bound(iree_host_size_t count,
                                       loom_serve_text_storage_bound_t* bounds);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_STORAGE_BOUNDS_H_
