// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/specialization.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/provider.h"

typedef struct loom_target_resolved_specialization_t {
  // Function receiving the effective target.
  loom_func_like_t function;

  // Stable function name string ID used by the supplemental context.
  loom_string_id_t function_name_id;

  // Structured profile borrowed from the request.
  const loom_target_profile_t* target_profile;

  // Authored target requirement, or NULL for a targetless function.
  const loom_op_t* authored_target_op;

  // Exact target record materialized from |target_profile|.
  loom_symbol_ref_t effective_target_ref;
} loom_target_resolved_specialization_t;

static iree_string_view_t loom_target_specialization_normalize_function_name(
    iree_string_view_t function_name) {
  function_name = iree_string_view_trim(function_name);
  (void)iree_string_view_consume_prefix_char(&function_name, '@');
  return function_name;
}

static iree_status_t loom_target_specialization_resolve_function(
    loom_module_t* module, iree_string_view_t requested_name,
    iree_host_size_t* request_ordinals,
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

  const loom_symbol_ref_t authored_target_ref = loom_func_like_target(function);
  const loom_op_t* authored_target_op = NULL;
  if (loom_symbol_ref_is_valid(authored_target_ref)) {
    authored_target_op =
        module->symbols.entries[authored_target_ref.symbol_id].defining_op;
  }

  *out_specialization = (loom_target_resolved_specialization_t){
      .function = function,
      .function_name_id = function_name_id,
      .authored_target_op = authored_target_op,
  };
  return iree_ok_status();
}

static iree_status_t loom_target_specialization_lookup_target(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_ref_t target_ref, loom_target_record_view_t* out_target) {
  *out_target = (loom_target_record_view_t){0};
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, target_ref, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "verified function target has no indexed target facts");
    IREE_BUILTIN_UNREACHABLE();
  }
  *out_target = loom_target_record_view_make(module, target_facts);
  return iree_ok_status();
}

static iree_status_t loom_target_specialization_emit_conflict(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_module_t* module,
    const loom_target_resolved_specialization_t* specialization,
    loom_target_record_view_t authored_target,
    loom_target_record_view_t effective_target) {
  const iree_string_view_t function_name =
      module->strings.entries[specialization->function_name_id];
  const loom_diagnostic_param_t params[] = {
      loom_param_string(function_name),
      loom_param_string(authored_target.facts->name),
      loom_param_string(effective_target.facts->name),
  };
  const loom_diagnostic_emission_t emission = {
      .op = specialization->function.op,
      .error = LOOM_ERR_TARGET_052,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_target_specialization_validate_requirements(
    const loom_target_environment_t* environment, const loom_module_t* module,
    loom_target_resolved_specialization_t* specializations,
    iree_host_size_t specialization_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    uint32_t* out_error_count) {
  *out_error_count = 0;
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);

  for (iree_host_size_t i = 0; i < specialization_count; ++i) {
    const loom_target_resolved_specialization_t* specialization =
        &specializations[i];
    const loom_symbol_ref_t authored_target_ref =
        loom_func_like_target(specialization->function);
    if (!loom_symbol_ref_is_valid(authored_target_ref)) {
      continue;
    }

    loom_target_record_view_t effective_target = {0};
    IREE_RETURN_IF_ERROR(loom_target_specialization_lookup_target(
        module, &fact_table, specialization->effective_target_ref,
        &effective_target));
    loom_target_record_view_t authored_target = {0};
    IREE_RETURN_IF_ERROR(loom_target_specialization_lookup_target(
        module, &fact_table, authored_target_ref, &authored_target));
    if (loom_target_satisfies_requirement(environment, effective_target,
                                          authored_target)) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_target_specialization_emit_conflict(
        diagnostic_emitter, module, specialization, authored_target,
        effective_target));
    if (*out_error_count != UINT32_MAX) {
      ++*out_error_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_specialization_bind_functions(
    loom_module_t* module,
    const loom_target_resolved_specialization_t* specializations,
    iree_host_size_t specialization_count, iree_arena_allocator_t* arena) {
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < specialization_count && iree_status_is_ok(status); ++i) {
    const loom_target_resolved_specialization_t* specialization =
        &specializations[i];
    status = loom_rewriter_set_attr(
        &rewriter, specialization->function.op,
        specialization->function.vtable->target_attr_index,
        loom_attr_symbol(specialization->effective_target_ref));
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
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
    IREE_RETURN_IF_ERROR(loom_target_specialization_resolve_function(
        module, request->function_name, request_ordinals, &specializations[i]));
    request_ordinals[specializations[i].function_name_id] = i;
    specializations[i].target_profile = request->target_profile;
  }

  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_target_environment_materialize_effective_target(
        environment, module, specializations[i].target_profile,
        specializations[i].authored_target_op,
        &specializations[i].effective_target_ref));
    IREE_ASSERT(
        loom_symbol_ref_is_valid(specializations[i].effective_target_ref));
  }

  IREE_RETURN_IF_ERROR(loom_target_specialization_validate_requirements(
      environment, module, specializations, requests.count, diagnostic_emitter,
      arena, &out_result->error_count));
  if (out_result->error_count != 0) {
    return iree_ok_status();
  }

  const loom_target_profile_t** profiles_by_function_name_id = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->strings.count, sizeof(*profiles_by_function_name_id),
      (void**)&profiles_by_function_name_id));
  memset(profiles_by_function_name_id, 0,
         module->strings.count * sizeof(*profiles_by_function_name_id));
  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    profiles_by_function_name_id[specializations[i].function_name_id] =
        specializations[i].target_profile;
  }

  IREE_RETURN_IF_ERROR(loom_target_specialization_bind_functions(
      module, specializations, requests.count, arena));
  out_result->context = (loom_target_specialization_context_t){
      .profiles_by_function_name_id = profiles_by_function_name_id,
      .profile_capacity = module->strings.count,
  };
  return iree_ok_status();
}

const loom_target_profile_t* loom_target_specialization_context_lookup(
    const loom_target_specialization_context_t* context,
    const loom_module_t* module, loom_func_like_t function) {
  if (context == NULL || context->profiles_by_function_name_id == NULL ||
      module == NULL || !loom_func_like_isa(function)) {
    return NULL;
  }
  const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
  if (!loom_symbol_ref_is_valid(function_ref) || function_ref.module_id != 0 ||
      function_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_string_id_t function_name_id =
      module->symbols.entries[function_ref.symbol_id].name_id;
  if (function_name_id >= context->profile_capacity) {
    return NULL;
  }
  return context->profiles_by_function_name_id[function_name_id];
}
