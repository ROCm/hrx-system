// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_LINK_TEMPLATE_INPUT_BINDING_H_
#define LOOM_LINK_TEMPLATE_INPUT_BINDING_H_

#include "loom/link/plan_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_link_template_input_module_t
    loom_link_template_input_module_t;

// Encoding identity correspondence for one materialized caller. The immutable
// source index can outlive this query; no caller identity is stored in it.
typedef struct loom_link_template_input_binding_t {
  // Frozen source modules owning signature-coordinate input contracts.
  const loom_link_module_index_t* index;
  // Caller module and the linker-owned source symbol correspondence.
  const loom_link_plan_materialization_t* materialization;
  // Query-owned storage, live until every returned provider summary retires.
  iree_arena_allocator_t* arena;
  // Lazy module directory and reached-only readers, owned by this query.
  loom_link_template_input_module_t** modules;
} loom_link_template_input_binding_t;

// Binds a source contract's static identities once per distinct retained row.
// Empty contracts and dimensions-only contracts borrow source storage directly.
iree_status_t loom_link_template_input_bind(
    loom_link_template_input_binding_t* binding,
    const loom_link_module_index_symbol_t* source,
    const loom_input_contract_t** out_contract);

// Releases all decoder-owned temporary storage. Returned contracts and imported
// encoding definitions remain owned by the query arena and caller module.
void loom_link_template_input_binding_deinitialize(
    loom_link_template_input_binding_t* binding);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_TEMPLATE_INPUT_BINDING_H_
