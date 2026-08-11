// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/specialization.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/provider.h"

typedef struct loom_target_resolved_specialization_t {
  // Authored function represented by the compiler version.
  loom_func_like_t function;

  // Stable function name string ID used for diagnostics.
  loom_string_id_t function_name_id;

  // Structured profile borrowed from the request.
  const loom_target_profile_t* target_profile;

  // Provider owning |target_profile| and every fact derived from it.
  const loom_target_provider_t* target_provider;

  // Profile facts projected once for all requests sharing |target_profile|.
  const loom_target_facts_t* projected_profile_facts;

  // Function facts projected at specialization construction.
  const loom_func_symbol_facts_t* function_facts;

  // Authored target witness name, or empty for a targetless function.
  iree_string_view_t authored_target_name;

  // Authored target facts when projected by the target witness, or NULL.
  const loom_target_symbol_facts_t* authored_target_facts;

  // Profile facts refined by the authored target requirement but not by the
  // function-local ABI/export contract.
  const loom_target_facts_t* target_context_facts;

  // Compiler-owned target-refined function version.
  loom_target_function_version_t* version;
} loom_target_resolved_specialization_t;

static iree_string_view_t loom_target_specialization_normalize_function_name(
    iree_string_view_t function_name) {
  function_name = iree_string_view_trim(function_name);
  (void)iree_string_view_consume_prefix_char(&function_name, '@');
  return function_name;
}

static iree_status_t loom_target_specialization_lookup_target(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_ref_t target_ref, iree_string_view_t* out_target_name,
    const loom_target_symbol_facts_t** out_target_facts) {
  *out_target_name = iree_string_view_empty();
  *out_target_facts = NULL;
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, target_ref, &base_facts));
  const loom_symbol_t* target_symbol =
      &module->symbols.entries[target_ref.symbol_id];
  *out_target_name = module->strings.entries[target_symbol->name_id];
  *out_target_facts = loom_target_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_target_specialization_resolve_function(
    loom_module_t* module, iree_string_view_t requested_name,
    iree_host_size_t* request_ordinals, loom_symbol_fact_table_t* fact_table,
    loom_target_resolved_specialization_t* out_specialization) {
  const iree_string_view_t function_name =
      loom_target_specialization_normalize_function_name(requested_name);
  if (iree_string_view_is_empty(function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target specialization function name is empty");
  }

  const loom_string_id_t function_name_id =
      loom_module_lookup_string(module, function_name);
  if (function_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target specialization function '@%.*s' does not exist",
        (int)function_name.size, function_name.data);
  }
  const loom_symbol_id_t function_symbol_id =
      loom_module_find_symbol(module, function_name_id);
  if (function_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target specialization function '@%.*s' does not exist",
        (int)function_name.size, function_name.data);
  }
  if (request_ordinals[function_name_id] != IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target specialization function '@%.*s' was requested more than once",
        (int)function_name.size, function_name.data);
  }

  loom_symbol_t* function_symbol = &module->symbols.entries[function_symbol_id];
  loom_func_like_t function =
      loom_func_like_cast(module, function_symbol->defining_op);
  if (!loom_func_like_isa(function) ||
      function.vtable->target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target specialization symbol '@%.*s' is not a target-assignable "
        "function",
        (int)function_name.size, function_name.data);
  }

  const loom_symbol_facts_base_t* function_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
      fact_table, module, function_symbol_id, &function_base_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(function_base_facts);
  if (function_facts == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "verified target-assignable function has no indexed function facts");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_symbol_ref_t authored_target_ref = function_facts->target_symbol;
  iree_string_view_t authored_target_name = iree_string_view_empty();
  const loom_target_symbol_facts_t* authored_target_facts = NULL;
  if (loom_symbol_ref_is_valid(authored_target_ref)) {
    IREE_RETURN_IF_ERROR(loom_target_specialization_lookup_target(
        module, fact_table, authored_target_ref, &authored_target_name,
        &authored_target_facts));
  }

  *out_specialization = (loom_target_resolved_specialization_t){
      .function = function,
      .function_name_id = function_name_id,
      .function_facts = function_facts,
      .authored_target_name = authored_target_name,
      .authored_target_facts = authored_target_facts,
  };
  return iree_ok_status();
}

static iree_status_t loom_target_specialization_emit_conflict(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_module_t* module,
    const loom_target_resolved_specialization_t* specialization,
    iree_string_view_t effective_target_name) {
  const iree_string_view_t function_name =
      module->strings.entries[specialization->function_name_id];
  const loom_diagnostic_param_t params[] = {
      loom_param_string(function_name),
      loom_param_string(specialization->authored_target_name),
      loom_param_string(effective_target_name),
  };
  const loom_diagnostic_emission_t emission = {
      .op = specialization->function.op,
      .error = LOOM_ERR_TARGET_052,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_target_specialization_prepare_versions(
    const loom_module_t* module,
    loom_target_resolved_specialization_t* specializations,
    iree_host_size_t specialization_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    uint32_t* out_error_count) {
  *out_error_count = 0;

  for (iree_host_size_t i = 0; i < specialization_count; ++i) {
    loom_target_resolved_specialization_t* specialization = &specializations[i];
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_target_resolved_specialization_t* prior = &specializations[j];
      if (prior->target_profile == specialization->target_profile) {
        specialization->projected_profile_facts =
            prior->projected_profile_facts;
        break;
      }
    }
    if (specialization->projected_profile_facts == NULL) {
      loom_target_facts_t* projected_profile_facts = NULL;
      IREE_RETURN_IF_ERROR(loom_target_profile_project_facts(
          specialization->target_profile, arena, &projected_profile_facts));
      specialization->projected_profile_facts = projected_profile_facts;
    }
  }

  for (iree_host_size_t i = 0; i < specialization_count; ++i) {
    loom_target_resolved_specialization_t* specialization = &specializations[i];
    const loom_target_facts_t* projected_profile_facts =
        specialization->projected_profile_facts;
    if (specialization->authored_target_facts != NULL &&
        !loom_target_facts_satisfy_specialization_requirement(
            projected_profile_facts,
            specialization->authored_target_facts->projection)) {
      IREE_RETURN_IF_ERROR(loom_target_specialization_emit_conflict(
          diagnostic_emitter, module, specialization,
          loom_target_facts_identity_name(projected_profile_facts)));
      if (*out_error_count != UINT32_MAX) {
        ++*out_error_count;
      }
      continue;
    }

    const loom_target_facts_t* target_context_facts = projected_profile_facts;
    if (specialization->authored_target_facts != NULL) {
      for (iree_host_size_t j = 0; j < i; ++j) {
        const loom_target_resolved_specialization_t* prior =
            &specializations[j];
        if (prior->projected_profile_facts == projected_profile_facts &&
            prior->authored_target_facts ==
                specialization->authored_target_facts &&
            prior->target_context_facts != NULL) {
          target_context_facts = prior->target_context_facts;
          break;
        }
      }
      if (target_context_facts == projected_profile_facts) {
        loom_target_facts_t* refined_context_facts = NULL;
        IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
            projected_profile_facts, arena, &refined_context_facts));
        loom_target_facts_builder_apply_requirement(
            specialization->authored_target_facts->projection,
            refined_context_facts);
        target_context_facts = refined_context_facts;
      }
    }
    specialization->target_context_facts = target_context_facts;

    const iree_string_view_t target_name =
        !iree_string_view_is_empty(specialization->authored_target_name)
            ? specialization->authored_target_name
            : loom_target_facts_identity_name(target_context_facts);
    bool contract_valid = false;
    const loom_target_facts_t* function_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_facts(
        module, specialization->function_facts, target_name,
        target_context_facts, diagnostic_emitter, arena, &contract_valid,
        &function_facts));
    if (!contract_valid) {
      if (*out_error_count != UINT32_MAX) {
        ++*out_error_count;
      }
      continue;
    }

    *specialization->version = (loom_target_function_version_t){
        .base =
            {
                .type = &loom_target_function_version_type,
                .function = specialization->function,
            },
        .authored_target_name = specialization->authored_target_name,
        .authored_target_facts =
            specialization->authored_target_facts != NULL
                ? specialization->authored_target_facts->projection
                : NULL,
        .target_provider = specialization->target_provider,
        .target_context_facts = target_context_facts,
        .effective_target_facts = function_facts,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_target_specialize_functions(
    const loom_target_environment_t* environment, loom_module_t* module,
    loom_target_specialization_request_list_t requests,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_target_specialization_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_target_specialization_result_t){0};
  loom_function_version_owner_initialize(arena, &out_result->function_versions);
  if (requests.count == 0) {
    return iree_ok_status();
  }
  if (requests.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target specialization request count is nonzero but values is NULL");
  }

  loom_target_resolved_specialization_t* specializations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, requests.count,
                                                 sizeof(*specializations),
                                                 (void**)&specializations));
  iree_host_size_t* request_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->strings.count,
                                                 sizeof(*request_ordinals),
                                                 (void**)&request_ordinals));
  for (iree_host_size_t i = 0; i < module->strings.count; ++i) {
    request_ordinals[i] = IREE_HOST_SIZE_MAX;
  }
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);

  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    const loom_target_specialization_request_t* request = &requests.values[i];
    if (request->target_profile == NULL ||
        request->target_profile->type == NULL ||
        request->target_profile->target_bundle == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target specialization request %" PRIhsz
                              " has no complete target profile",
                              i);
    }
    const loom_target_provider_t* target_provider =
        loom_target_environment_lookup_profile_provider(
            environment, request->target_profile->type);
    if (target_provider == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target specialization request %" PRIhsz
          " uses profile family '%.*s' not linked into the target environment",
          i, (int)request->target_profile->type->name.size,
          request->target_profile->type->name.data);
    }
    IREE_RETURN_IF_ERROR(loom_target_specialization_resolve_function(
        module, request->function_name, request_ordinals, &fact_table,
        &specializations[i]));
    request_ordinals[specializations[i].function_name_id] = i;
    specializations[i].target_profile = request->target_profile;
    specializations[i].target_provider = target_provider;
  }

  loom_target_function_version_t* target_versions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, requests.count,
                                                 sizeof(*target_versions),
                                                 (void**)&target_versions));
  IREE_RETURN_IF_ERROR(loom_function_version_owner_reserve(
      &out_result->function_versions, requests.count));
  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    specializations[i].version = &target_versions[i];
  }

  IREE_RETURN_IF_ERROR(loom_target_specialization_prepare_versions(
      module, specializations, requests.count, diagnostic_emitter, arena,
      &out_result->error_count));
  if (out_result->error_count != 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_function_version_owner_append(
        &out_result->function_versions, &target_versions[i].base));
  }
  return iree_ok_status();
}
