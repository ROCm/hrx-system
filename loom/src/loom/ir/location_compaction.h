// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IR_LOCATION_COMPACTION_H_
#define LOOM_IR_LOCATION_COMPACTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Removes location rows that are not reachable from live operations.
//
// Fused and tagged child locations reachable from an operation location are
// retained transitively. Retained rows preserve their relative order while
// operation and child location IDs are rewritten to the compact table. Source
// table entries are unchanged. The operation walk and location graph traversal
// are linear in the live IR and location-table sizes.
//
// This invalidates previously retained location IDs and location-row pointers.
// |scratch_arena| is used only for temporary reachability and remap storage.
iree_status_t loom_module_compact_locations(
    loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_removed_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_LOCATION_COMPACTION_H_
