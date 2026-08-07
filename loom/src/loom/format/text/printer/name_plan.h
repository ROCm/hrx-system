// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_NAME_PLAN_H_
#define LOOM_FORMAT_TEXT_PRINTER_NAME_PLAN_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_print_name_resolution_t loom_print_name_resolution_t;

// Immutable canonical SSA name resolutions for one module print.
//
// Construction indexes explicit names by parser scope and resolves all
// duplicate and generated-name collisions once. Emission is then an O(1)
// value-id lookup. The plan owns only the compact resolutions in a scoped
// arena backed by the module block pool; its larger construction index is
// discarded from that arena before initialization returns.
typedef struct loom_print_name_plan_t {
  // Canonical resolution for each module value ID.
  // NULL when the module has no explicit value names.
  loom_print_name_resolution_t* resolutions;
  // Print-scoped arena owning resolutions.
  iree_arena_allocator_t arena;
} loom_print_name_plan_t;

// Builds canonical SSA name resolutions for |module|.
iree_status_t loom_print_name_plan_initialize(const loom_module_t* module,
                                              loom_print_name_plan_t* out_plan);

// Releases storage owned by |plan|.
void loom_print_name_plan_deinitialize(loom_print_name_plan_t* plan);

// Prints |value_id|'s canonical SSA reference.
//
// A NULL |plan| resolves the single reference directly. This bounded fallback
// supports standalone type and attribute printing; complete module and
// operation printers always provide a plan.
iree_status_t loom_print_name_plan_write_value_ref(
    const loom_print_name_plan_t* plan, loom_output_stream_t* stream,
    const loom_module_t* module, loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_NAME_PLAN_H_
