// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/module_sealing.h"

#include <stdio.h>
#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/link/linker.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/facts.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"

typedef struct loom_target_sealing_group_t {
  // Exact provider-owned target facts shared by this group.
  loom_resolved_target_t resolved_target;

  // Exact ordinary target definition in the sealed module.
  loom_symbol_ref_t target_ref;
} loom_target_sealing_group_t;

typedef struct loom_target_sealing_plan_t {
  // Target versions reconciled against the source module symbol table.
  loom_target_function_version_snapshot_t version_snapshot;

  // Distinct resolved targets in source function-symbol order.
  loom_target_sealing_group_t* groups;

  // Number of initialized entries in |groups|.
  iree_host_size_t group_count;

  // Group index for each source symbol, or IREE_HOST_SIZE_MAX when unversioned.
  iree_host_size_t* group_indices_by_symbol;
} loom_target_sealing_plan_t;

static iree_string_view_t loom_target_sealing_module_name(
    const loom_module_t* module) {
  if (module->name_id == LOOM_STRING_ID_INVALID) return IREE_SV("module");
  return module->strings.entries[module->name_id];
}

static iree_status_t loom_target_sealing_validate_resolved_target(
    const loom_resolved_target_t* resolved_target) {
  if (resolved_target->provider == NULL || resolved_target->facts == NULL ||
      resolved_target->facts->fact_type == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target function version has no resolved target");
  }
  const loom_target_provider_t* provider = resolved_target->provider;
  if (provider->profile_type == NULL ||
      provider->profile_type->fact_type != resolved_target->facts->fact_type) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "resolved target fact type does not match its owning provider");
  }
  if (provider->materialize_definition == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target provider family '%.*s' cannot materialize target definitions",
        (int)provider->profile_type->name.size,
        provider->profile_type->name.data);
  }
  return iree_ok_status();
}

static iree_host_size_t loom_target_sealing_find_group(
    const loom_target_sealing_plan_t* plan,
    const loom_resolved_target_t* resolved_target) {
  for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
    const loom_resolved_target_t* existing = &plan->groups[i].resolved_target;
    if (existing->provider == resolved_target->provider &&
        loom_target_facts_are_equivalent(existing->facts,
                                         resolved_target->facts)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_target_sealing_plan_build(
    const loom_module_t* source_module,
    const loom_function_version_list_t* function_versions,
    iree_arena_allocator_t* arena, loom_target_sealing_plan_t* out_plan) {
  *out_plan = (loom_target_sealing_plan_t){0};
  if (function_versions != NULL) {
    for (iree_host_size_t i = 0; i < function_versions->count; ++i) {
      if (loom_target_function_version_const_cast(
              function_versions->values[i]) == NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "function version %zu has no target sealing representation", i);
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      source_module, function_versions, arena, &out_plan->version_snapshot));

  if (source_module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, source_module->symbols.count,
                                  sizeof(*out_plan->group_indices_by_symbol),
                                  (void**)&out_plan->group_indices_by_symbol));
    for (iree_host_size_t i = 0; i < source_module->symbols.count; ++i) {
      out_plan->group_indices_by_symbol[i] = IREE_HOST_SIZE_MAX;
    }
  }
  if (function_versions == NULL || function_versions->count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function_versions->count, sizeof(*out_plan->groups),
      (void**)&out_plan->groups));

  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const loom_target_function_version_t* function_version =
        loom_target_function_version_snapshot_at(&out_plan->version_snapshot,
                                                 symbol_id);
    if (function_version == NULL) continue;
    IREE_RETURN_IF_ERROR(loom_target_sealing_validate_resolved_target(
        &function_version->resolved_target));

    iree_host_size_t group_index = loom_target_sealing_find_group(
        out_plan, &function_version->resolved_target);
    if (group_index == IREE_HOST_SIZE_MAX) {
      group_index = out_plan->group_count++;
      out_plan->groups[group_index] = (loom_target_sealing_group_t){
          .resolved_target = function_version->resolved_target,
          .target_ref = loom_symbol_ref_null(),
      };
    }
    out_plan->group_indices_by_symbol[symbol_id] = group_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_target_sealing_clone_source(
    const loom_module_t* source_module, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_module_t** out_module) {
  const loom_module_t* source_modules[] = {source_module};
  const loom_link_options_t options = {
      .module_name = loom_target_sealing_module_name(source_module),
  };
  return loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules), &options, block_pool,
      allocator, out_module);
}

static iree_status_t loom_target_sealing_index_symbols(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_map_t* out_symbol_map) {
  *out_symbol_map = (loom_symbol_map_t){0};
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    const loom_string_id_t name_id = module->symbols.entries[symbol_id].name_id;
    uint16_t indexed_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_symbol_map_find_or_insert(
        out_symbol_map, arena, name_id, symbol_id, &indexed_symbol_id));
    if (indexed_symbol_id != symbol_id) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "sealed module contains duplicate symbol names");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_sealing_reuse_definitions(
    loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_target_sealing_plan_t* plan) {
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (symbol->definition == NULL ||
        symbol->definition->fact_domain != &loom_target_symbol_fact_domain) {
      continue;
    }
    const loom_symbol_facts_base_t* base_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                       symbol_id, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    if (target_facts == NULL) continue;

    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_target_sealing_group_t* group = &plan->groups[i];
      if (loom_symbol_ref_is_valid(group->target_ref)) continue;
      if (loom_target_facts_are_equivalent(group->resolved_target.facts,
                                           target_facts->projection)) {
        group->target_ref = (loom_symbol_ref_t){
            .module_id = 0,
            .symbol_id = symbol_id,
        };
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_sealing_add_target_symbol(
    loom_module_t* module, loom_symbol_map_t* symbol_map,
    iree_arena_allocator_t* arena, iree_host_size_t group_index,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  for (iree_host_size_t collision_ordinal = 0;; ++collision_ordinal) {
    char name_buffer[96];
    const int name_length =
        collision_ordinal == 0
            ? snprintf(name_buffer, sizeof(name_buffer),
                       "__loom_sealed_target_%zu", group_index)
            : snprintf(name_buffer, sizeof(name_buffer),
                       "__loom_sealed_target_%zu_%zu", group_index,
                       collision_ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "sealed target symbol name is too long");
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module,
        iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
        &name_id));
    if (loom_symbol_map_find(symbol_map, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    IREE_RETURN_IF_ERROR(
        loom_symbol_map_insert(symbol_map, arena, name_id, symbol_id));
    *out_target_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }
}

static iree_status_t loom_target_sealing_materialize_definitions(
    loom_module_t* module, loom_symbol_map_t* symbol_map,
    loom_symbol_fact_table_t* fact_table, iree_arena_allocator_t* arena,
    loom_target_sealing_plan_t* plan) {
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
    loom_target_sealing_group_t* group = &plan->groups[i];
    if (loom_symbol_ref_is_valid(group->target_ref)) continue;

    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_target_sealing_add_target_symbol(
        module, symbol_map, arena, i, &target_ref));
    loom_op_t* target_op = NULL;
    IREE_RETURN_IF_ERROR(
        group->resolved_target.provider->materialize_definition(
            &builder, &group->resolved_target, target_ref,
            LOOM_LOCATION_UNKNOWN, &target_op));
    if (target_op == NULL ||
        module->symbols.entries[target_ref.symbol_id].defining_op !=
            target_op) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target provider did not materialize the requested definition");
    }

    const loom_symbol_facts_base_t* base_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
        fact_table, module, target_ref.symbol_id, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    if (target_facts == NULL ||
        !loom_target_facts_are_equivalent(group->resolved_target.facts,
                                          target_facts->projection)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "materialized target definition did not reproject to resolved facts");
    }
    group->target_ref = target_ref;
  }
  return iree_ok_status();
}

static iree_status_t loom_target_sealing_find_cloned_function(
    const loom_module_t* source_module, loom_symbol_id_t source_symbol_id,
    loom_module_t* sealed_module, const loom_symbol_map_t* sealed_symbol_map,
    loom_func_like_t* out_function) {
  *out_function = (loom_func_like_t){0};
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_symbol_id];
  const iree_string_view_t source_name =
      source_module->strings.entries[source_symbol->name_id];
  const loom_string_id_t sealed_name_id =
      loom_module_lookup_string(sealed_module, source_name);
  const loom_symbol_id_t sealed_symbol_id =
      loom_symbol_map_find(sealed_symbol_map, sealed_name_id);
  if (sealed_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sealed module clone did not preserve function symbol '@%.*s'",
        (int)source_name.size, source_name.data);
  }
  const loom_symbol_t* sealed_symbol =
      &sealed_module->symbols.entries[sealed_symbol_id];
  const loom_func_like_t function =
      loom_func_like_cast(sealed_module, sealed_symbol->defining_op);
  if (!loom_func_like_isa(function) ||
      function.vtable->target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sealed symbol '@%.*s' is not a target-assignable function",
        (int)source_name.size, source_name.data);
  }
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_target_sealing_bind_functions(
    const loom_module_t* source_module, loom_module_t* sealed_module,
    const loom_symbol_map_t* sealed_symbol_map,
    const loom_target_sealing_plan_t* plan) {
  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const iree_host_size_t group_index =
        plan->group_indices_by_symbol[symbol_id];
    if (group_index == IREE_HOST_SIZE_MAX) continue;
    loom_func_like_t sealed_function = {0};
    IREE_RETURN_IF_ERROR(loom_target_sealing_find_cloned_function(
        source_module, symbol_id, sealed_module, sealed_symbol_map,
        &sealed_function));
    loom_func_like_set_target(sealed_module, sealed_function,
                              plan->groups[group_index].target_ref);
  }
  return iree_ok_status();
}

iree_status_t loom_target_module_seal(
    const loom_module_t* source_module,
    const loom_function_version_list_t* function_versions,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_sealed_module) {
  if (out_sealed_module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_sealed_module must not be NULL");
  }
  *out_sealed_module = NULL;
  if (source_module == NULL || block_pool == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source_module and block_pool must not be NULL");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  loom_target_sealing_plan_t plan = {0};
  loom_module_t* sealed_module = NULL;
  loom_symbol_map_t sealed_symbol_map = {0};
  loom_symbol_fact_table_t target_fact_table;
  loom_symbol_fact_table_initialize(&target_fact_table, &scratch_arena);

  iree_status_t status = loom_target_sealing_plan_build(
      source_module, function_versions, &scratch_arena, &plan);
  if (iree_status_is_ok(status)) {
    status = loom_target_sealing_clone_source(source_module, block_pool,
                                              allocator, &sealed_module);
  }
  if (iree_status_is_ok(status) && plan.group_count > 0) {
    status = loom_target_sealing_index_symbols(sealed_module, &scratch_arena,
                                               &sealed_symbol_map);
  }
  if (iree_status_is_ok(status) && plan.group_count > 0) {
    status = loom_target_sealing_reuse_definitions(sealed_module,
                                                   &target_fact_table, &plan);
  }
  if (iree_status_is_ok(status) && plan.group_count > 0) {
    status = loom_target_sealing_materialize_definitions(
        sealed_module, &sealed_symbol_map, &target_fact_table, &scratch_arena,
        &plan);
  }
  if (iree_status_is_ok(status) && plan.group_count > 0) {
    status = loom_target_sealing_bind_functions(source_module, sealed_module,
                                                &sealed_symbol_map, &plan);
  }
  if (iree_status_is_ok(status)) {
    *out_sealed_module = sealed_module;
    sealed_module = NULL;
  }

  loom_module_free(sealed_module);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
