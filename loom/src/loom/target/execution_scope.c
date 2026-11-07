// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/execution_scope.h"

#include "loom/ops/target/facts.h"
#include "loom/target/facts_builder.h"
#include "loom/target/provider.h"

iree_status_t loom_target_execution_scope_resolve(
    loom_target_execution_scope_table_t* table,
    const loom_symbol_reference_table_t* references,
    loom_symbol_reference_execution_scope_id_t scope,
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_target_environment_t* environment,
    loom_target_execution_scope_t inherited, iree_arena_allocator_t* arena,
    const loom_target_execution_scope_t** out_scope) {
  if (table->values == NULL) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        iree_arena_allocator(arena), references->execution_targets.count,
        sizeof(*table->values), (void**)&table->values));
  }
  loom_target_execution_scope_t** slot = &table->values[scope - 1];
  if (*slot == NULL) {
    loom_target_execution_scope_t target = inherited;
    const loom_symbol_ref_t witness =
        references->execution_targets.values[scope - 1];
    if (loom_symbol_ref_is_valid(witness)) {
      const loom_symbol_facts_base_t* base_facts = NULL;
      IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
          symbol_facts, module, witness, &base_facts));
      const loom_target_symbol_facts_t* facts =
          loom_target_symbol_facts_cast(base_facts);
      target = (loom_target_execution_scope_t){
          .witness = witness,
          .facts = facts ? facts->projection : NULL,
      };
    }
    const loom_target_facts_t* source = target.facts;
    IREE_RETURN_IF_ERROR(
        loom_target_facts_builder_project_worker(source, arena, &target.facts));
    if (source != target.facts) {
      target.witness = loom_symbol_ref_null();
    }
    if (environment && target.facts && !target.provider) {
      target.provider = loom_target_environment_lookup_fact_provider(
          environment, target.facts->fact_type);
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(**slot), (void**)slot));
    **slot = target;
  }
  *out_scope = *slot;
  return iree_ok_status();
}
