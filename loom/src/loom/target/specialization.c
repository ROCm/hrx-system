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
#include "loom/ops/target/ops.h"
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

  // Provider and exact target-family facts resolved for this specialization.
  loom_resolved_target_t resolved_target;

  // Profile facts projected once for all requests sharing |target_profile|.
  const loom_target_facts_t* projected_profile_facts;

  // First request sharing |target_profile| and its projected facts.
  iree_host_size_t projected_profile_owner_ordinal;

  // Context ordinal assigned to a targetless use of the projected profile.
  // Only the projected-profile owner stores this shared value.
  loom_target_context_ordinal_t targetless_context_ordinal;

  // Context ordinal retained by the produced function version.
  loom_target_context_ordinal_t target_context_ordinal;

  // Function facts projected at specialization construction.
  const loom_func_symbol_facts_t* function_facts;

  // Authored target witness name, or empty for a targetless function.
  iree_string_view_t authored_target_name;

  // Optional target requirement symbol facts projected from the witness.
  const loom_target_symbol_facts_t* target_requirement_symbol_facts;

  // Compiler-owned target-refined function version.
  loom_target_function_version_t* version;
} loom_target_resolved_specialization_t;

// One validated target declaration binding indexed by target symbol ID.
typedef struct loom_target_resolved_declaration_binding_t {
  // Structured profile borrowed from the binding.
  const loom_target_profile_t* target_profile;

  // Provider selected from the target environment for |target_profile|.
  const loom_target_provider_t* target_provider;
} loom_target_resolved_declaration_binding_t;

static iree_string_view_t loom_target_specialization_normalize_symbol_name(
    iree_string_view_t symbol_name) {
  symbol_name = iree_string_view_trim(symbol_name);
  (void)iree_string_view_consume_prefix_char(&symbol_name, '@');
  return iree_string_view_trim(symbol_name);
}

static iree_status_t loom_target_specialization_resolve_profile(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile, const char* request_kind,
    iree_host_size_t request_ordinal,
    const loom_target_provider_t** out_target_provider) {
  *out_target_provider = NULL;
  if (target_profile == NULL || target_profile->type == NULL ||
      target_profile->type->fact_type == NULL ||
      target_profile->target_bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s %" PRIhsz " has no complete target profile",
                            request_kind, request_ordinal);
  }
  const loom_target_provider_t* target_provider =
      loom_target_environment_lookup_fact_provider(
          environment, target_profile->type->fact_type);
  if (target_provider == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s %" PRIhsz
        " projects target fact family '%.*s' not linked into the target "
        "environment",
        request_kind, request_ordinal, (int)target_profile->type->name.size,
        target_profile->type->name.data);
  }
  *out_target_provider = target_provider;
  return iree_ok_status();
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

static iree_status_t loom_target_specialization_resolve_function_symbol(
    loom_module_t* module, loom_symbol_id_t function_symbol_id,
    iree_bitmap_t specialized_symbols, loom_symbol_fact_table_t* fact_table,
    loom_target_resolved_specialization_t* out_specialization) {
  loom_symbol_t* function_symbol = &module->symbols.entries[function_symbol_id];
  const loom_string_id_t function_name_id = function_symbol->name_id;
  const iree_string_view_t function_name =
      module->strings.entries[function_name_id];
  if (iree_bitmap_test(specialized_symbols, function_symbol_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target specialization function '@%.*s' was assigned more than once",
        (int)function_name.size, function_name.data);
  }

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
  const loom_target_symbol_facts_t* target_requirement_symbol_facts = NULL;
  if (loom_symbol_ref_is_valid(authored_target_ref)) {
    IREE_RETURN_IF_ERROR(loom_target_specialization_lookup_target(
        module, fact_table, authored_target_ref, &authored_target_name,
        &target_requirement_symbol_facts));
  }

  *out_specialization = (loom_target_resolved_specialization_t){
      .function = function,
      .function_name_id = function_name_id,
      .function_facts = function_facts,
      .authored_target_name = authored_target_name,
      .target_requirement_symbol_facts = target_requirement_symbol_facts,
      .targetless_context_ordinal = LOOM_TARGET_CONTEXT_ORDINAL_INVALID,
      .target_context_ordinal = LOOM_TARGET_CONTEXT_ORDINAL_INVALID,
  };
  iree_bitmap_set(specialized_symbols, function_symbol_id);
  return iree_ok_status();
}

static iree_status_t loom_target_specialization_resolve_function_name(
    loom_module_t* module, iree_string_view_t requested_name,
    iree_bitmap_t specialized_symbols, loom_symbol_fact_table_t* fact_table,
    loom_target_resolved_specialization_t* out_specialization) {
  const iree_string_view_t function_name =
      loom_target_specialization_normalize_symbol_name(requested_name);
  if (iree_string_view_is_empty(function_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target specialization function name is empty");
  }

  const loom_string_id_t function_name_id =
      loom_module_lookup_string(module, function_name);
  const loom_symbol_id_t function_symbol_id =
      function_name_id != LOOM_STRING_ID_INVALID
          ? loom_module_find_symbol(module, function_name_id)
          : LOOM_SYMBOL_ID_INVALID;
  if (function_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target specialization function '@%.*s' does not exist",
        (int)function_name.size, function_name.data);
  }
  return loom_target_specialization_resolve_function_symbol(
      module, function_symbol_id, specialized_symbols, fact_table,
      out_specialization);
}

static iree_status_t loom_target_specialization_resolve_declaration_binding(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_target_declaration_binding_t* binding,
    iree_host_size_t binding_ordinal,
    loom_target_resolved_declaration_binding_t** bindings_by_target_symbol,
    loom_target_resolved_declaration_binding_t* out_binding) {
  const iree_string_view_t target_name =
      loom_target_specialization_normalize_symbol_name(binding->target_name);
  if (iree_string_view_is_empty(target_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target declaration binding %" PRIhsz
                            " has an empty symbol name",
                            binding_ordinal);
  }

  const loom_string_id_t target_name_id =
      loom_module_lookup_string(module, target_name);
  const loom_symbol_id_t target_symbol_id =
      target_name_id != LOOM_STRING_ID_INVALID
          ? loom_module_find_symbol(module, target_name_id)
          : LOOM_SYMBOL_ID_INVALID;
  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target declaration binding symbol '@%.*s' does not exist",
        (int)target_name.size, target_name.data);
  }
  if (!loom_target_decl_isa(
          module->symbols.entries[target_symbol_id].defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target declaration binding symbol '@%.*s' is not a target.decl",
        (int)target_name.size, target_name.data);
  }
  if (bindings_by_target_symbol[target_symbol_id] != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target declaration '@%.*s' was bound more than once",
        (int)target_name.size, target_name.data);
  }

  const loom_target_provider_t* target_provider = NULL;
  IREE_RETURN_IF_ERROR(loom_target_specialization_resolve_profile(
      environment, binding->target_profile, "target declaration binding",
      binding_ordinal, &target_provider));
  *out_binding = (loom_target_resolved_declaration_binding_t){
      .target_profile = binding->target_profile,
      .target_provider = target_provider,
  };
  bindings_by_target_symbol[target_symbol_id] = out_binding;
  return iree_ok_status();
}

static const loom_target_resolved_declaration_binding_t*
loom_target_specialization_find_function_binding(
    const loom_module_t* module, loom_symbol_id_t function_symbol_id,
    loom_target_resolved_declaration_binding_t* const*
        bindings_by_target_symbol) {
  loom_func_like_t function = loom_func_like_cast(
      module, module->symbols.entries[function_symbol_id].defining_op);
  if (!loom_func_like_isa(function) ||
      function.vtable->target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return NULL;
  }
  const loom_symbol_ref_t target_ref = loom_func_like_target(function);
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return NULL;
  }
  return bindings_by_target_symbol[target_ref.symbol_id];
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
    specialization->projected_profile_owner_ordinal = i;
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_target_resolved_specialization_t* prior = &specializations[j];
      if (prior->target_profile == specialization->target_profile) {
        specialization->projected_profile_facts =
            prior->projected_profile_facts;
        specialization->projected_profile_owner_ordinal =
            prior->projected_profile_owner_ordinal;
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

  iree_host_size_t next_target_context_ordinal = 0;
  for (iree_host_size_t i = 0; i < specialization_count; ++i) {
    loom_target_resolved_specialization_t* specialization = &specializations[i];
    const loom_target_facts_t* projected_profile_facts =
        specialization->projected_profile_facts;
    if (specialization->target_requirement_symbol_facts != NULL &&
        !loom_target_facts_satisfy_specialization_requirement(
            projected_profile_facts,
            specialization->target_requirement_symbol_facts->projection)) {
      IREE_RETURN_IF_ERROR(loom_target_specialization_emit_conflict(
          diagnostic_emitter, module, specialization,
          loom_target_facts_identity_name(projected_profile_facts)));
      if (*out_error_count != UINT32_MAX) {
        ++*out_error_count;
      }
      continue;
    }

    const loom_target_facts_t* resolved_facts = projected_profile_facts;
    if (specialization->target_requirement_symbol_facts != NULL) {
      for (iree_host_size_t j = 0; j < i; ++j) {
        const loom_target_resolved_specialization_t* prior =
            &specializations[j];
        if (prior->projected_profile_facts == projected_profile_facts &&
            prior->target_requirement_symbol_facts ==
                specialization->target_requirement_symbol_facts &&
            prior->resolved_target.facts != NULL) {
          resolved_facts = prior->resolved_target.facts;
          specialization->target_context_ordinal =
              prior->target_context_ordinal;
          break;
        }
      }
      if (resolved_facts == projected_profile_facts) {
        loom_target_facts_t* refined_context_facts = NULL;
        IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
            projected_profile_facts, arena, &refined_context_facts));
        loom_target_facts_builder_apply_requirement(
            specialization->target_requirement_symbol_facts->projection,
            refined_context_facts);
        resolved_facts = refined_context_facts;
      }
    } else {
      loom_target_resolved_specialization_t* profile_owner =
          &specializations[specialization->projected_profile_owner_ordinal];
      specialization->target_context_ordinal =
          profile_owner->targetless_context_ordinal;
    }
    if (specialization->target_context_ordinal ==
        LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
      if (next_target_context_ordinal >= LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "target specialization exceeds %u invocation contexts",
            (unsigned)LOOM_TARGET_CONTEXT_ORDINAL_INVALID);
      }
      specialization->target_context_ordinal =
          (loom_target_context_ordinal_t)next_target_context_ordinal++;
      if (specialization->target_requirement_symbol_facts == NULL) {
        loom_target_resolved_specialization_t* profile_owner =
            &specializations[specialization->projected_profile_owner_ordinal];
        profile_owner->targetless_context_ordinal =
            specialization->target_context_ordinal;
      }
    }
    specialization->resolved_target = (loom_resolved_target_t){
        .provider = specialization->resolved_target.provider,
        .facts = resolved_facts,
    };

    const iree_string_view_t target_name =
        !iree_string_view_is_empty(specialization->authored_target_name)
            ? specialization->authored_target_name
            : loom_target_facts_identity_name(resolved_facts);
    bool contract_valid = false;
    const loom_target_facts_t* function_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_facts(
        module, specialization->function_facts, target_name, resolved_facts,
        diagnostic_emitter, arena, &contract_valid, &function_facts));
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
        .target_requirement_facts =
            specialization->target_requirement_symbol_facts != NULL
                ? specialization->target_requirement_symbol_facts->projection
                : NULL,
        .resolved_target = specialization->resolved_target,
        .target_context_ordinal = specialization->target_context_ordinal,
        .authored_target_is_exact =
            specialization->target_requirement_symbol_facts != NULL &&
            loom_target_facts_are_equivalent(
                specialization->target_requirement_symbol_facts->projection,
                specialization->resolved_target.facts),
        .function_target_facts = function_facts,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_target_specialize_functions(
    const loom_target_environment_t* environment, loom_module_t* module,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_target_specialization_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_target_specialization_result_t){0};
  loom_function_version_owner_initialize(arena, &out_result->function_versions);
  if (requests.count != 0 && requests.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target specialization request count is nonzero but values is NULL");
  }
  if (bindings.count != 0 && bindings.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target declaration binding count is nonzero but values is NULL");
  }
  if (requests.count == 0 && bindings.count == 0) {
    return iree_ok_status();
  }

  loom_target_resolved_declaration_binding_t* resolved_bindings = NULL;
  loom_target_resolved_declaration_binding_t** bindings_by_target_symbol = NULL;
  if (bindings.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, bindings.count,
                                                   sizeof(*resolved_bindings),
                                                   (void**)&resolved_bindings));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*bindings_by_target_symbol),
        (void**)&bindings_by_target_symbol));
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      bindings_by_target_symbol[i] = NULL;
    }
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_target_specialization_resolve_declaration_binding(
              environment, module, &bindings.values[i], i,
              bindings_by_target_symbol, &resolved_bindings[i]));
    }
  }

  iree_host_size_t bound_function_count = 0;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    if (bindings_by_target_symbol != NULL &&
        loom_target_specialization_find_function_binding(
            module, symbol_id, bindings_by_target_symbol) != NULL) {
      ++bound_function_count;
    }
  }
  iree_host_size_t specialization_count = 0;
  if (!iree_host_size_checked_add(requests.count, bound_function_count,
                                  &specialization_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target specialization count overflow");
  }
  if (specialization_count == 0) {
    return iree_ok_status();
  }

  loom_target_resolved_specialization_t* specializations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, specialization_count,
                                                 sizeof(*specializations),
                                                 (void**)&specializations));
  const iree_host_size_t specialized_symbol_word_count =
      iree_bitmap_calculate_words(module->symbols.count);
  uint64_t* specialized_symbol_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, specialized_symbol_word_count, sizeof(*specialized_symbol_words),
      (void**)&specialized_symbol_words));
  for (iree_host_size_t i = 0; i < specialized_symbol_word_count; ++i) {
    specialized_symbol_words[i] = 0;
  }
  const iree_bitmap_t specialized_symbols = {
      .bit_count = module->symbols.count,
      .words = specialized_symbol_words,
  };
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);

  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    const loom_target_specialization_request_t* request = &requests.values[i];
    const loom_target_provider_t* target_provider = NULL;
    IREE_RETURN_IF_ERROR(loom_target_specialization_resolve_profile(
        environment, request->target_profile, "target specialization request",
        i, &target_provider));
    IREE_RETURN_IF_ERROR(loom_target_specialization_resolve_function_name(
        module, request->function_name, specialized_symbols, &fact_table,
        &specializations[i]));
    specializations[i].target_profile = request->target_profile;
    specializations[i].resolved_target.provider = target_provider;
  }

  iree_host_size_t next_specialization_ordinal = requests.count;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    const loom_target_resolved_declaration_binding_t* binding =
        bindings_by_target_symbol != NULL
            ? loom_target_specialization_find_function_binding(
                  module, symbol_id, bindings_by_target_symbol)
            : NULL;
    if (binding == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_target_specialization_resolve_function_symbol(
        module, symbol_id, specialized_symbols, &fact_table,
        &specializations[next_specialization_ordinal]));
    specializations[next_specialization_ordinal].target_profile =
        binding->target_profile;
    specializations[next_specialization_ordinal].resolved_target.provider =
        binding->target_provider;
    ++next_specialization_ordinal;
  }
  IREE_ASSERT(next_specialization_ordinal == specialization_count);

  loom_target_function_version_t* target_versions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, specialization_count,
                                                 sizeof(*target_versions),
                                                 (void**)&target_versions));
  IREE_RETURN_IF_ERROR(loom_function_version_owner_reserve(
      &out_result->function_versions, specialization_count));
  for (iree_host_size_t i = 0; i < specialization_count; ++i) {
    specializations[i].version = &target_versions[i];
  }

  IREE_RETURN_IF_ERROR(loom_target_specialization_prepare_versions(
      module, specializations, specialization_count, diagnostic_emitter, arena,
      &out_result->error_count));
  if (out_result->error_count != 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < specialization_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_function_version_owner_append(
        &out_result->function_versions, &target_versions[i].base));
  }
  return iree_ok_status();
}
