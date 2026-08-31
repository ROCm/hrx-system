// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/root_function_versions.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/target/facts.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"

typedef struct loom_target_root_context_t {
  // Stable exact target requirement shared by roots in this context.
  const loom_target_facts_t* target_requirement_facts;

  // Provider and exact facts used to compile roots in this context.
  loom_resolved_target_t resolved_target;

  // Compilation-local identity shared by roots in this context.
  loom_target_context_ordinal_t target_context_ordinal;
} loom_target_root_context_t;

static iree_status_t loom_target_root_function_versions_next_context_ordinal(
    const loom_function_version_list_t* versions,
    loom_target_context_ordinal_t* out_ordinal) {
  iree_host_size_t next_ordinal = 0;
  for (iree_host_size_t i = 0; i < versions->count; ++i) {
    const loom_target_function_version_t* version =
        loom_target_function_version_const_cast(versions->values[i]);
    if (version == NULL) continue;
    const iree_host_size_t candidate =
        (iree_host_size_t)version->target_context_ordinal + 1;
    if (candidate > next_ordinal) next_ordinal = candidate;
  }
  if (next_ordinal >= LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target roots exceed %u invocation contexts",
                            (unsigned)LOOM_TARGET_CONTEXT_ORDINAL_INVALID);
  }
  *out_ordinal = (loom_target_context_ordinal_t)next_ordinal;
  return iree_ok_status();
}

static void loom_target_root_function_versions_seed_contexts(
    const loom_module_t* module,
    const loom_target_function_version_snapshot_t* version_snapshot,
    loom_target_root_context_t* contexts_by_target_symbol) {
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    const loom_target_function_version_t* version =
        loom_target_function_version_snapshot_at(version_snapshot, symbol_id);
    if (version == NULL || !version->authored_target_is_exact) continue;
    const loom_symbol_ref_t target_ref =
        loom_func_like_target(version->base.function);
    if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
        target_ref.symbol_id >= module->symbols.count) {
      continue;
    }
    loom_target_root_context_t* context =
        &contexts_by_target_symbol[target_ref.symbol_id];
    if (context->resolved_target.facts != NULL) continue;
    *context = (loom_target_root_context_t){
        .target_requirement_facts = version->target_requirement_facts,
        .resolved_target = version->resolved_target,
        .target_context_ordinal = version->target_context_ordinal,
    };
  }
}

static iree_status_t loom_target_root_function_versions_prepare_context(
    const loom_target_environment_t* environment, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table, loom_symbol_ref_t target_ref,
    loom_target_context_ordinal_t* next_context_ordinal,
    iree_arena_allocator_t* version_arena,
    loom_target_root_context_t* out_context) {
  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, target_ref, &target_base_facts));
  const loom_target_symbol_facts_t* target_symbol_facts =
      loom_target_symbol_facts_cast(target_base_facts);
  if (target_symbol_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target root requires an exact authored or specialized target");
  }
  const loom_target_provider_t* provider =
      loom_target_environment_lookup_fact_provider(
          environment, target_symbol_facts->projection->fact_type);
  if (provider == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target root uses a target family not linked into the environment");
  }
  if (*next_context_ordinal >= LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target roots exceed %u invocation contexts",
                            (unsigned)LOOM_TARGET_CONTEXT_ORDINAL_INVALID);
  }

  loom_target_facts_t* stable_target_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
      target_symbol_facts->projection, version_arena, &stable_target_facts));
  *out_context = (loom_target_root_context_t){
      .target_requirement_facts = stable_target_facts,
      .resolved_target =
          {
              .provider = provider,
              .facts = stable_target_facts,
          },
      .target_context_ordinal = (*next_context_ordinal)++,
  };
  return iree_ok_status();
}

iree_status_t loom_target_root_function_versions_prepare(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_symbol_id_t* root_symbol_ids, iree_host_size_t root_count,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_function_version_owner_t* version_owner,
    loom_function_version_ordinal_t* out_root_version_ordinals,
    uint32_t* out_error_count) {
  IREE_ASSERT(root_count == 0 || root_symbol_ids != NULL);
  IREE_ASSERT(root_count == 0 || out_root_version_ordinals != NULL);
  *out_error_count = 0;
  if (root_count == 0) return iree_ok_status();
  memset(out_root_version_ordinals, 0xFF,
         root_count * sizeof(*out_root_version_ordinals));

  loom_target_function_version_snapshot_t version_snapshot = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, loom_function_version_owner_list(version_owner), scratch_arena,
      &version_snapshot));

  loom_target_root_context_t* contexts_by_target_symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, module->symbols.count, sizeof(*contexts_by_target_symbol),
      (void**)&contexts_by_target_symbol));
  memset(contexts_by_target_symbol, 0,
         module->symbols.count * sizeof(*contexts_by_target_symbol));
  loom_target_root_function_versions_seed_contexts(module, &version_snapshot,
                                                   contexts_by_target_symbol);

  loom_target_context_ordinal_t next_context_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_target_root_function_versions_next_context_ordinal(
      loom_function_version_owner_list(version_owner), &next_context_ordinal));
  iree_host_size_t required_version_capacity = 0;
  if (!iree_host_size_checked_add(version_owner->list.count, root_count,
                                  &required_version_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target root version count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_function_version_owner_reserve(
      version_owner, required_version_capacity));
  loom_target_function_version_t* new_versions = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(version_owner->arena, root_count,
                                sizeof(*new_versions), (void**)&new_versions));
  iree_host_size_t new_version_count = 0;
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, scratch_arena);

  for (iree_host_size_t i = 0; i < root_count; ++i) {
    const loom_symbol_id_t root_symbol_id = root_symbol_ids[i];
    if (root_symbol_id >= module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target root symbol %u is out of range",
                              (unsigned)root_symbol_id);
    }
    loom_function_version_t* existing_version =
        loom_target_function_version_snapshot_handle_at(&version_snapshot,
                                                        root_symbol_id);
    if (existing_version != NULL) {
      if (loom_target_function_version_cast(existing_version) == NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "target root has an unsupported function-version representation");
      }
      out_root_version_ordinals[i] =
          loom_target_function_version_snapshot_ordinal_at(&version_snapshot,
                                                           root_symbol_id);
      continue;
    }

    const loom_symbol_facts_base_t* function_base_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
        &fact_table, module, root_symbol_id, &function_base_facts));
    const loom_func_symbol_facts_t* function_facts =
        loom_func_symbol_facts_cast(function_base_facts);
    if (function_facts == NULL || !function_facts->has_body ||
        !function_facts->exports) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target root must name an exported function definition");
    }
    const loom_symbol_ref_t target_ref = function_facts->target_symbol;
    if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
        target_ref.symbol_id >= module->symbols.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target root requires an exact authored or specialized target");
    }

    loom_target_root_context_t* context =
        &contexts_by_target_symbol[target_ref.symbol_id];
    if (context->resolved_target.facts == NULL) {
      IREE_RETURN_IF_ERROR(loom_target_root_function_versions_prepare_context(
          environment, module, &fact_table, target_ref, &next_context_ordinal,
          version_owner->arena, context));
    }
    const loom_symbol_t* target_symbol =
        &module->symbols.entries[target_ref.symbol_id];
    const iree_string_view_t target_name =
        module->strings.entries[target_symbol->name_id];
    bool contract_valid = false;
    const loom_target_facts_t* function_target_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_facts(
        module, function_facts, target_name, context->resolved_target.facts,
        diagnostic_emitter, version_owner->arena, &contract_valid,
        &function_target_facts));
    if (!contract_valid) {
      if (*out_error_count != UINT32_MAX) ++*out_error_count;
      continue;
    }

    loom_target_function_version_t* version =
        &new_versions[new_version_count++];
    *version = (loom_target_function_version_t){
        .base =
            {
                .type = &loom_target_function_version_type,
                .function = loom_func_like_cast(
                    module,
                    module->symbols.entries[root_symbol_id].defining_op),
            },
        .authored_target_name = target_name,
        .target_requirement_facts = context->target_requirement_facts,
        .resolved_target = context->resolved_target,
        .target_context_ordinal = context->target_context_ordinal,
        .target_binding_source = LOOM_TARGET_BINDING_SOURCE_AUTHORED,
        .authored_target_is_exact = true,
        .function_target_facts = function_target_facts,
    };
    if (version_owner->list.count >= LOOM_FUNCTION_VERSION_ORDINAL_INVALID) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "target root function-version count exceeds the ordinal domain");
    }
    const loom_function_version_ordinal_t version_ordinal =
        (loom_function_version_ordinal_t)version_owner->list.count;
    IREE_RETURN_IF_ERROR(
        loom_function_version_owner_append(version_owner, &version->base));
    version_snapshot.version_handles_by_symbol[root_symbol_id] = &version->base;
    version_snapshot.version_ordinals_by_symbol[root_symbol_id] =
        version_ordinal;
    out_root_version_ordinals[i] = version_ordinal;
  }
  return iree_ok_status();
}
