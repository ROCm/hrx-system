// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_decision_model.h"

#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/template/ops.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_template_decision_build_family_t {
  // Destination model populated from this build record.
  loom_template_decision_model_t* model;

  // Cached family declaration facts.
  const loom_func_symbol_facts_t* family_facts;

  // Resolved family target identity facts, or NULL.
  const loom_target_facts_t* family_target_facts;

  // Total contextual features across the family and its providers.
  uint32_t feature_count;

  // Total scalar predicates across the family and its providers.
  uint32_t predicate_count;

  // Total literal predicate operands across the family and its providers.
  uint32_t constant_count;

  // True when providers are already in descending stable priority order.
  bool providers_are_ranked;
} loom_template_decision_build_family_t;

typedef enum loom_template_decision_signature_lookup_mode_e {
  LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_LINEAR = 0,
  LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_ORDERED = 1,
  LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_MAPPED = 2,
} loom_template_decision_signature_lookup_mode_t;

typedef struct loom_template_decision_operand_map_entry_t {
  // Contract-local SSA value ID.
  loom_value_id_t value_id;

  // Compiled argument or result reference.
  loom_decision_program_operand_ref_t operand_ref;
} loom_template_decision_operand_map_entry_t;

typedef struct loom_template_decision_signature_lookup_t {
  // Borrowed argument value IDs in signature order.
  const loom_value_id_t* argument_ids;

  // Borrowed result value IDs in signature order.
  const loom_value_id_t* result_ids;

  // Prepared ordered map when mode is MAPPED.
  const loom_template_decision_operand_map_entry_t* operand_map;

  // Number of signature arguments.
  uint16_t argument_count;

  // Number of signature results.
  uint16_t result_count;

  // Lookup strategy selected for this contract.
  loom_template_decision_signature_lookup_mode_t mode;
} loom_template_decision_signature_lookup_t;

static bool loom_template_decision_operand_map_entry_less(
    const loom_template_decision_operand_map_entry_t* lhs,
    const loom_template_decision_operand_map_entry_t* rhs) {
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_template_decision_sort_operand_map,
                          loom_template_decision_operand_map_entry_t,
                          loom_template_decision_operand_map_entry_less)

typedef struct loom_template_decision_ranked_choice_t {
  // Generic choice populated before priority ordering.
  loom_decision_program_choice_t choice;

  // Provider priority used only during construction.
  int64_t priority;
} loom_template_decision_ranked_choice_t;

static bool loom_template_decision_ranked_choice_less(
    const loom_template_decision_ranked_choice_t* lhs,
    const loom_template_decision_ranked_choice_t* rhs) {
  if (lhs->priority != rhs->priority) return lhs->priority > rhs->priority;
  return lhs->choice.action_ordinal < rhs->choice.action_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_template_decision_sort_ranked_choices,
                          loom_template_decision_ranked_choice_t,
                          loom_template_decision_ranked_choice_less)

// Above this count a one-level radix directory bounds lookup to one byte of
// the module-local symbol ID. Every demanded family contributes at least one
// application site, so the fixed directory cost is amortized by its users.
#define LOOM_TEMPLATE_DECISION_SYMBOL_PAGE_THRESHOLD 128
#define LOOM_TEMPLATE_DECISION_SYMBOL_PAGE_COUNT 256

// Building and sorting a signature-local map costs more than bounded linear
// lookup below this many worst-case comparisons. Repeated references at least
// as numerous as the signature still amortize the map independently.
#define LOOM_TEMPLATE_DECISION_LINEAR_SIGNATURE_COMPARISON_LIMIT \
  UINT64_C(1000000)

static bool loom_template_decision_contract_has_target_identity(
    loom_symbol_ref_t target_symbol, const loom_target_facts_t* target_facts) {
  return loom_symbol_ref_is_valid(target_symbol) || target_facts != NULL;
}

static iree_status_t loom_template_decision_add_count(iree_host_size_t amount,
                                                      uint32_t* inout_count) {
  if (amount > UINT32_MAX - *inout_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template decision model exceeds uint32_t range");
  }
  *inout_count += (uint32_t)amount;
  return iree_ok_status();
}

static iree_status_t loom_template_decision_count_contract(
    loom_symbol_ref_t target_symbol, const loom_target_facts_t* target_facts,
    const loom_predicate_t* predicates, uint16_t predicate_count,
    const loom_target_condition_t* target_conditions,
    uint16_t target_condition_count,
    loom_template_decision_build_family_t* build_family,
    uint32_t* out_value_reference_count) {
  *out_value_reference_count = 0;
  const uint32_t target_identity_count =
      loom_template_decision_contract_has_target_identity(target_symbol,
                                                          target_facts)
          ? 1
          : 0;
  if ((uint32_t)target_condition_count + target_identity_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "template applicability contract has too many target requirements");
  }
  IREE_RETURN_IF_ERROR(loom_template_decision_add_count(
      target_identity_count + target_condition_count,
      &build_family->feature_count));
  IREE_RETURN_IF_ERROR(loom_template_decision_add_count(
      predicate_count, &build_family->predicate_count));

  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    for (uint8_t j = 0; j < predicate->arg_count; ++j) {
      if (predicate->arg_tags[j] == LOOM_PRED_ARG_CONST) {
        IREE_RETURN_IF_ERROR(
            loom_template_decision_add_count(1, &build_family->constant_count));
      } else if (predicate->arg_tags[j] == LOOM_PRED_ARG_VALUE) {
        ++*out_value_reference_count;
      }
    }
  }

  if (predicate_count > 0) {
    build_family->model->flags |=
        LOOM_TEMPLATE_DECISION_MODEL_FLAG_HAS_SCALAR_PREDICATES;
  }
  for (uint16_t i = 0; i < target_condition_count; ++i) {
    if (target_conditions[i].descriptor->project_query_predicate != NULL) {
      build_family->model->flags |=
          LOOM_TEMPLATE_DECISION_MODEL_FLAG_HAS_PROJECTABLE_TARGET_CONDITIONS;
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_template_decision_lookup_function_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    loom_symbol_ref_t symbol, const loom_func_symbol_facts_t** out_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(symbol_facts, module,
                                                         symbol, &base_facts));
  *out_facts = loom_func_symbol_facts_cast(base_facts);
  if (*out_facts == NULL) {
    IREE_ASSERT_UNREACHABLE("verified template family has no function facts");
    IREE_BUILTIN_UNREACHABLE();
  }
  return iree_ok_status();
}

static iree_status_t loom_template_decision_lookup_target_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    loom_symbol_ref_t target_symbol,
    const loom_target_facts_t** out_target_facts) {
  *out_target_facts = NULL;
  if (!loom_symbol_ref_is_valid(target_symbol)) return iree_ok_status();
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      symbol_facts, module, target_symbol, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts != NULL) *out_target_facts = target_facts->projection;
  return iree_ok_status();
}

static iree_status_t loom_template_decision_allocate_array(
    iree_arena_allocator_t* arena, uint32_t count,
    iree_host_size_t element_size, void** out_values) {
  *out_values = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_values);
}

static loom_decision_program_operand_ref_t
loom_template_decision_find_signature_operand_linear(
    const loom_template_decision_signature_lookup_t* lookup,
    loom_value_id_t value_id) {
  for (uint16_t i = 0; i < lookup->argument_count; ++i) {
    if (lookup->argument_ids[i] == value_id) {
      return loom_decision_program_argument_ref(i);
    }
  }
  for (uint16_t i = 0; i < lookup->result_count; ++i) {
    if (lookup->result_ids[i] == value_id) {
      return loom_decision_program_result_ref(i);
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "verified template predicate value is outside its family signature");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_template_decision_value_ids_are_ordered(
    const loom_value_id_t* value_ids, uint16_t count) {
  for (uint16_t i = 1; i < count; ++i) {
    if (value_ids[i - 1] >= value_ids[i]) return false;
  }
  return true;
}

static bool loom_template_decision_may_need_operand_map(
    uint32_t signature_count, uint32_t value_reference_count) {
  if (signature_count <= 8 || value_reference_count <= 4) return false;
  return value_reference_count >= signature_count ||
         (uint64_t)value_reference_count * signature_count >
             LOOM_TEMPLATE_DECISION_LINEAR_SIGNATURE_COMPARISON_LIMIT;
}

static bool loom_template_decision_signature_needs_operand_map(
    const loom_value_id_t* argument_ids, uint16_t argument_count,
    const loom_value_id_t* result_ids, uint16_t result_count,
    uint32_t value_reference_count) {
  const uint32_t signature_count = (uint32_t)argument_count + result_count;
  if (!loom_template_decision_may_need_operand_map(signature_count,
                                                   value_reference_count)) {
    return false;
  }
  return !loom_template_decision_value_ids_are_ordered(argument_ids,
                                                       argument_count) ||
         !loom_template_decision_value_ids_are_ordered(result_ids,
                                                       result_count);
}

static void loom_template_decision_prepare_signature_lookup(
    const loom_value_id_t* argument_ids, uint16_t argument_count,
    const loom_value_id_t* result_ids, uint16_t result_count,
    const loom_predicate_t* predicates, uint16_t predicate_count,
    loom_template_decision_operand_map_entry_t* operand_map,
    loom_template_decision_signature_lookup_t* out_lookup) {
  *out_lookup = (loom_template_decision_signature_lookup_t){
      .argument_ids = argument_ids,
      .result_ids = result_ids,
      .argument_count = argument_count,
      .result_count = result_count,
      .mode = LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_LINEAR,
  };
  const uint32_t signature_count = (uint32_t)argument_count + result_count;
  if (signature_count <= 8) return;
  if (loom_template_decision_value_ids_are_ordered(argument_ids,
                                                   argument_count) &&
      loom_template_decision_value_ids_are_ordered(result_ids, result_count)) {
    out_lookup->mode = LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_ORDERED;
    return;
  }

  uint32_t value_reference_count = 0;
  for (uint16_t i = 0; i < predicate_count; ++i) {
    for (uint8_t j = 0; j < predicates[i].arg_count; ++j) {
      value_reference_count += predicates[i].arg_tags[j] == LOOM_PRED_ARG_VALUE;
    }
  }
  if (!loom_template_decision_may_need_operand_map(signature_count,
                                                   value_reference_count)) {
    return;
  }

  for (uint16_t i = 0; i < argument_count; ++i) {
    operand_map[i] = (loom_template_decision_operand_map_entry_t){
        .value_id = argument_ids[i],
        .operand_ref = loom_decision_program_argument_ref(i),
    };
  }
  for (uint16_t i = 0; i < result_count; ++i) {
    operand_map[argument_count + i] =
        (loom_template_decision_operand_map_entry_t){
            .value_id = result_ids[i],
            .operand_ref = loom_decision_program_result_ref(i),
        };
  }
  loom_template_decision_sort_operand_map(operand_map, signature_count);
  out_lookup->operand_map = operand_map;
  out_lookup->mode = LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_MAPPED;
}

static bool loom_template_decision_find_ordered_value_id(
    const loom_value_id_t* value_ids, uint16_t count, loom_value_id_t value_id,
    uint16_t* out_ordinal) {
  uint16_t begin = 0;
  uint16_t end = count;
  while (begin < end) {
    const uint16_t middle = begin + (uint16_t)((end - begin) / 2);
    if (value_ids[middle] < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == count || value_ids[begin] != value_id) return false;
  *out_ordinal = begin;
  return true;
}

static loom_decision_program_operand_ref_t
loom_template_decision_find_signature_operand_ordered(
    const loom_template_decision_signature_lookup_t* lookup,
    loom_value_id_t value_id) {
  uint16_t ordinal = 0;
  if (loom_template_decision_find_ordered_value_id(
          lookup->argument_ids, lookup->argument_count, value_id, &ordinal)) {
    return loom_decision_program_argument_ref(ordinal);
  }
  if (loom_template_decision_find_ordered_value_id(
          lookup->result_ids, lookup->result_count, value_id, &ordinal)) {
    return loom_decision_program_result_ref(ordinal);
  }
  IREE_ASSERT_UNREACHABLE(
      "verified template predicate value is outside its family signature");
  IREE_BUILTIN_UNREACHABLE();
}

static loom_decision_program_operand_ref_t
loom_template_decision_find_signature_operand_mapped(
    const loom_template_decision_signature_lookup_t* lookup,
    loom_value_id_t value_id) {
  const uint32_t signature_count =
      (uint32_t)lookup->argument_count + lookup->result_count;
  uint32_t begin = 0;
  uint32_t end = signature_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    if (lookup->operand_map[middle].value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  IREE_ASSERT(begin < signature_count);
  IREE_ASSERT(lookup->operand_map[begin].value_id == value_id);
  return lookup->operand_map[begin].operand_ref;
}

typedef struct loom_template_decision_build_cursor_t {
  // Next feature output row.
  loom_template_decision_feature_t* features;

  // Next predicate output row.
  loom_decision_program_predicate_t* predicates;

  // Next constant output row.
  int64_t* constants;

  // Absolute feature ordinal for the next output row.
  uint32_t feature_ordinal;

  // Absolute predicate ordinal for the next output row.
  uint32_t predicate_ordinal;

  // Absolute constant ordinal for the next output row.
  uint32_t constant_ordinal;

  // Absolute feature ordinal corresponding to model-local ordinal zero.
  uint32_t feature_ref_base;

  // Absolute predicate ordinal corresponding to model-local ordinal zero.
  uint32_t predicate_ref_base;

  // Absolute constant ordinal corresponding to model-local ordinal zero.
  uint32_t constant_ref_base;
} loom_template_decision_build_cursor_t;

static loom_decision_program_operand_ref_t
loom_template_decision_compile_value_operand(
    const loom_template_decision_signature_lookup_t* signature_lookup,
    loom_value_id_t value_id) {
  switch (signature_lookup->mode) {
    case LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_LINEAR:
      return loom_template_decision_find_signature_operand_linear(
          signature_lookup, value_id);
    case LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_ORDERED:
      return loom_template_decision_find_signature_operand_ordered(
          signature_lookup, value_id);
    case LOOM_TEMPLATE_DECISION_SIGNATURE_LOOKUP_MAPPED:
      return loom_template_decision_find_signature_operand_mapped(
          signature_lookup, value_id);
  }
  IREE_ASSERT_UNREACHABLE("invalid template signature lookup mode");
  IREE_BUILTIN_UNREACHABLE();
}

static void loom_template_decision_compile_contract(
    const loom_module_t* requirement_module, loom_symbol_ref_t target_symbol,
    const loom_target_facts_t* target_facts,
    const loom_value_id_t* argument_ids, uint16_t argument_count,
    const loom_value_id_t* result_ids, uint16_t result_count,
    const loom_predicate_t* source_predicates, uint16_t source_predicate_count,
    const loom_target_condition_t* target_conditions,
    uint16_t target_condition_count,
    loom_template_decision_operand_map_entry_t* operand_map,
    loom_template_decision_build_cursor_t* cursor,
    loom_decision_program_conjunction_t* out_conjunction) {
  loom_template_decision_signature_lookup_t signature_lookup;
  loom_template_decision_prepare_signature_lookup(
      argument_ids, argument_count, result_ids, result_count, source_predicates,
      source_predicate_count, operand_map, &signature_lookup);
  *out_conjunction = (loom_decision_program_conjunction_t){
      .first_predicate = cursor->predicate_ordinal - cursor->predicate_ref_base,
      .first_feature = cursor->feature_ordinal - cursor->feature_ref_base,
      .predicate_count = source_predicate_count,
      .feature_count =
          (uint16_t)(target_condition_count +
                     (loom_template_decision_contract_has_target_identity(
                          target_symbol, target_facts)
                          ? 1
                          : 0)),
  };

  if (loom_template_decision_contract_has_target_identity(target_symbol,
                                                          target_facts)) {
    *cursor->features++ = (loom_template_decision_feature_t){
        .kind = LOOM_TEMPLATE_DECISION_FEATURE_TARGET_IDENTITY,
        .value.target_identity =
            {
                .module = requirement_module,
                .target_symbol = target_symbol,
                .target_facts = target_facts,
            },
    };
    ++cursor->feature_ordinal;
  }
  for (uint16_t i = 0; i < target_condition_count; ++i) {
    *cursor->features++ = (loom_template_decision_feature_t){
        .kind = LOOM_TEMPLATE_DECISION_FEATURE_TARGET_CONDITION,
        .value.target_condition = &target_conditions[i],
    };
    ++cursor->feature_ordinal;
  }

  for (uint16_t i = 0; i < source_predicate_count; ++i) {
    const loom_predicate_t* source = &source_predicates[i];
    loom_decision_program_predicate_t* predicate = cursor->predicates++;
    *predicate = (loom_decision_program_predicate_t){
        .kind = source->kind,
        .operand_count = source->arg_count,
    };
    for (uint8_t j = 0; j < source->arg_count; ++j) {
      if (source->arg_tags[j] == LOOM_PRED_ARG_CONST) {
        cursor->constants[cursor->constant_ordinal] = source->args[j];
        predicate->operands[j] = loom_decision_program_constant_ref(
            cursor->constant_ordinal++ - cursor->constant_ref_base);
      } else {
        predicate->operands[j] = loom_template_decision_compile_value_operand(
            &signature_lookup, (loom_value_id_t)source->args[j]);
      }
    }
    ++cursor->predicate_ordinal;
  }
}

static void loom_template_decision_build_family(
    const loom_template_decision_build_family_t* build_family,
    loom_template_decision_operand_map_entry_t* operand_map,
    loom_template_decision_ranked_choice_t* ranked_choices,
    loom_decision_program_choice_t* choices,
    loom_decision_program_priority_group_t* priority_groups,
    uint32_t* out_priority_group_count,
    loom_template_decision_build_cursor_t* cursor) {
  loom_template_decision_model_t* model = build_family->model;

  const uint32_t first_feature = cursor->feature_ordinal;
  const uint32_t first_predicate = cursor->predicate_ordinal;
  const uint32_t first_constant = cursor->constant_ordinal;
  cursor->feature_ref_base = first_feature;
  cursor->predicate_ref_base = first_predicate;
  cursor->constant_ref_base = first_constant;
  loom_template_decision_compile_contract(
      model->module, build_family->family_facts->target_symbol,
      build_family->family_target_facts,
      build_family->family_facts->argument_ids,
      build_family->family_facts->argument_count,
      build_family->family_facts->result_ids,
      build_family->family_facts->result_count,
      build_family->family_facts->predicates,
      build_family->family_facts->predicate_count,
      build_family->family_facts->target_conditions,
      build_family->family_facts->target_condition_count, operand_map, cursor,
      &model->program.hard_requirements);

  for (uint32_t i = 0; i < model->providers.count; ++i) {
    const loom_template_provider_summary_t* provider =
        &model->providers.providers[i];
    loom_decision_program_conjunction_t conjunction = {0};
    loom_template_decision_compile_contract(
        provider->module, provider->target_symbol, provider->target_facts,
        provider->argument_ids, provider->argument_count, provider->result_ids,
        provider->result_count, provider->predicates, provider->predicate_count,
        provider->target_conditions, provider->target_condition_count,
        operand_map, cursor, &conjunction);
    const loom_decision_program_choice_t choice = {
        .conjunction = conjunction,
        .action_ordinal = i,
    };
    if (build_family->providers_are_ranked) {
      choices[i] = choice;
    } else {
      ranked_choices[i] = (loom_template_decision_ranked_choice_t){
          .choice = choice,
          .priority = provider->priority,
      };
    }
  }

  uint32_t priority_group_count = 0;
  if (build_family->providers_are_ranked) {
    for (uint32_t i = 0; i < model->providers.count; ++i) {
      const int64_t priority = model->providers.providers[i].priority;
      if (i == 0 || priority != model->providers.providers[i - 1].priority) {
        priority_groups[priority_group_count++] =
            (loom_decision_program_priority_group_t){.choice_count = 1};
      } else {
        ++priority_groups[priority_group_count - 1].choice_count;
      }
    }
    model->highest_provider_priority = model->providers.providers[0].priority;
  } else {
    loom_template_decision_sort_ranked_choices(ranked_choices,
                                               model->providers.count);
    for (uint32_t i = 0; i < model->providers.count; ++i) {
      choices[i] = ranked_choices[i].choice;
      if (i == 0 ||
          ranked_choices[i].priority != ranked_choices[i - 1].priority) {
        priority_groups[priority_group_count++] =
            (loom_decision_program_priority_group_t){.choice_count = 1};
      } else {
        ++priority_groups[priority_group_count - 1].choice_count;
      }
    }
    model->highest_provider_priority = ranked_choices[0].priority;
  }

  model->features = build_family->feature_count > 0
                        ? cursor->features - build_family->feature_count
                        : NULL;
  model->program.predicates =
      build_family->predicate_count > 0
          ? cursor->predicates - build_family->predicate_count
          : NULL;
  model->program.constants = build_family->constant_count > 0
                                 ? cursor->constants + first_constant
                                 : NULL;
  model->program.choices = choices;
  model->program.priority_groups = priority_groups;
  model->program.feature_count = build_family->feature_count;
  model->program.predicate_count = build_family->predicate_count;
  model->program.constant_count = build_family->constant_count;
  model->program.choice_count = (uint32_t)model->providers.count;
  model->program.priority_group_count = priority_group_count;
  *out_priority_group_count = priority_group_count;

  IREE_ASSERT(cursor->feature_ordinal ==
              first_feature + build_family->feature_count);
  IREE_ASSERT(cursor->predicate_ordinal ==
              first_predicate + build_family->predicate_count);
  IREE_ASSERT(cursor->constant_ordinal ==
              first_constant + build_family->constant_count);
}

iree_status_t loom_template_decision_model_catalog_build(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_symbol_reference_table_t* references,
    const loom_template_provider_catalog_t* providers,
    iree_arena_allocator_t* arena,
    loom_template_decision_model_catalog_t* out_catalog) {
  *out_catalog = (loom_template_decision_model_catalog_t){
      .module = module,
  };
  IREE_ASSERT(references->module == module);
  IREE_ASSERT(providers->module == module);

  for (iree_host_size_t i = 0; i < references->template_demands.family_count;
       ++i) {
    const loom_symbol_id_t symbol_id =
        references->template_demands.family_symbol_ids[i];
    const loom_template_provider_slice_t family_providers =
        loom_template_provider_catalog_lookup(
            providers,
            (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id});
    if (family_providers.count == 0) continue;
    ++out_catalog->model_count;
    out_catalog->maximum_choice_count = iree_max(
        out_catalog->maximum_choice_count, (uint32_t)family_providers.count);
  }

  loom_template_decision_model_t* models = NULL;
  loom_template_decision_build_family_t* build_families = NULL;
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, (uint32_t)out_catalog->model_count, sizeof(*models),
      (void**)&models));
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, (uint32_t)out_catalog->model_count, sizeof(*build_families),
      (void**)&build_families));
  out_catalog->models = models;
  loom_template_decision_model_symbol_page_t* symbol_pages = NULL;
  if (out_catalog->model_count > LOOM_TEMPLATE_DECISION_SYMBOL_PAGE_THRESHOLD) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, LOOM_TEMPLATE_DECISION_SYMBOL_PAGE_COUNT, sizeof(*symbol_pages),
        (void**)&symbol_pages));
    memset(symbol_pages, 0,
           LOOM_TEMPLATE_DECISION_SYMBOL_PAGE_COUNT * sizeof(*symbol_pages));
    out_catalog->symbol_pages = symbol_pages;
  }

  uint32_t total_feature_count = 0;
  uint32_t total_predicate_count = 0;
  uint32_t total_constant_count = 0;
  uint32_t total_choice_count = 0;
  uint32_t maximum_operand_map_count = 0;
  uint32_t maximum_unranked_choice_count = 0;
  uint32_t model_ordinal = 0;
  for (iree_host_size_t i = 0; i < references->template_demands.family_count;
       ++i) {
    const loom_symbol_id_t symbol_id =
        references->template_demands.family_symbol_ids[i];
    const loom_symbol_ref_t family = {
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    const loom_template_provider_slice_t family_providers =
        loom_template_provider_catalog_lookup(providers, family);
    if (family_providers.count == 0) continue;

    loom_template_decision_model_t* model = &models[model_ordinal];
    *model = (loom_template_decision_model_t){
        .module = module,
        .family = family,
        .providers = family_providers,
    };
    if (symbol_pages != NULL) {
      loom_template_decision_model_symbol_page_t* page =
          &symbol_pages[symbol_id >> 8];
      if (page->model_count == 0) {
        page->first_model_ordinal = (uint16_t)model_ordinal;
      }
      ++page->model_count;
    }
    loom_template_decision_build_family_t* build_family =
        &build_families[model_ordinal++];
    *build_family = (loom_template_decision_build_family_t){
        .model = model,
        .providers_are_ranked = true,
    };
    IREE_RETURN_IF_ERROR(loom_template_decision_lookup_function_facts(
        module, symbol_facts, family, &build_family->family_facts));
    IREE_RETURN_IF_ERROR(loom_template_decision_lookup_target_facts(
        module, symbol_facts, build_family->family_facts->target_symbol,
        &build_family->family_target_facts));
    uint32_t family_value_reference_count = 0;
    IREE_RETURN_IF_ERROR(loom_template_decision_count_contract(
        build_family->family_facts->target_symbol,
        build_family->family_target_facts,
        build_family->family_facts->predicates,
        build_family->family_facts->predicate_count,
        build_family->family_facts->target_conditions,
        build_family->family_facts->target_condition_count, build_family,
        &family_value_reference_count));
    if (loom_template_decision_signature_needs_operand_map(
            build_family->family_facts->argument_ids,
            build_family->family_facts->argument_count,
            build_family->family_facts->result_ids,
            build_family->family_facts->result_count,
            family_value_reference_count)) {
      const uint32_t signature_count =
          build_family->family_facts->argument_count +
          build_family->family_facts->result_count;
      maximum_operand_map_count =
          iree_max(maximum_operand_map_count, signature_count);
    }
    for (uint32_t i = 0; i < family_providers.count; ++i) {
      const loom_template_provider_summary_t* provider =
          &family_providers.providers[i];
      if (!loom_template_decision_contract_has_target_identity(
              provider->target_symbol, provider->target_facts)) {
        ++model->target_independent_provider_count;
      }
      if (i > 0 &&
          family_providers.providers[i - 1].priority < provider->priority) {
        build_family->providers_are_ranked = false;
      }
      uint32_t provider_value_reference_count = 0;
      IREE_RETURN_IF_ERROR(loom_template_decision_count_contract(
          provider->target_symbol, provider->target_facts, provider->predicates,
          provider->predicate_count, provider->target_conditions,
          provider->target_condition_count, build_family,
          &provider_value_reference_count));
      if (loom_template_decision_signature_needs_operand_map(
              provider->argument_ids, provider->argument_count,
              provider->result_ids, provider->result_count,
              provider_value_reference_count)) {
        const uint32_t provider_signature_count =
            provider->argument_count + provider->result_count;
        maximum_operand_map_count =
            iree_max(maximum_operand_map_count, provider_signature_count);
      }
    }
    if (!build_family->providers_are_ranked) {
      maximum_unranked_choice_count = iree_max(
          maximum_unranked_choice_count, (uint32_t)family_providers.count);
    }
    IREE_RETURN_IF_ERROR(loom_template_decision_add_count(
        build_family->feature_count, &total_feature_count));
    IREE_RETURN_IF_ERROR(loom_template_decision_add_count(
        build_family->predicate_count, &total_predicate_count));
    IREE_RETURN_IF_ERROR(loom_template_decision_add_count(
        build_family->constant_count, &total_constant_count));
    IREE_RETURN_IF_ERROR(loom_template_decision_add_count(
        family_providers.count, &total_choice_count));
    if (build_family->feature_count > 0 &&
        build_family->feature_count - 1 >
            LOOM_DECISION_PROGRAM_FEATURE_ORDINAL_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "template decision contextual feature table is too large");
    }
    if (build_family->predicate_count > 0 &&
        build_family->predicate_count - 1 >
            LOOM_DECISION_PROGRAM_PREDICATE_ORDINAL_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template decision predicate table is too large");
    }
    if (build_family->constant_count > 0 &&
        build_family->constant_count - 1 >
            LOOM_DECISION_PROGRAM_CONSTANT_ORDINAL_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template decision constant table is too large");
    }
  }
  IREE_ASSERT(model_ordinal == out_catalog->model_count);

  loom_template_decision_feature_t* features = NULL;
  loom_decision_program_predicate_t* predicates_storage = NULL;
  int64_t* constants = NULL;
  loom_decision_program_choice_t* choices = NULL;
  loom_decision_program_priority_group_t* priority_groups = NULL;
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, total_feature_count, sizeof(*features), (void**)&features));
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, total_predicate_count, sizeof(*predicates_storage),
      (void**)&predicates_storage));
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, total_constant_count, sizeof(*constants), (void**)&constants));
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, total_choice_count, sizeof(*choices), (void**)&choices));
  IREE_RETURN_IF_ERROR(loom_template_decision_allocate_array(
      arena, total_choice_count, sizeof(*priority_groups),
      (void**)&priority_groups));
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(arena);
  loom_template_decision_ranked_choice_t* ranked_choices = NULL;
  loom_template_decision_operand_map_entry_t* operand_map = NULL;
  iree_status_t status = loom_template_decision_allocate_array(
      arena, maximum_unranked_choice_count, sizeof(*ranked_choices),
      (void**)&ranked_choices);
  if (iree_status_is_ok(status)) {
    status = loom_template_decision_allocate_array(
        arena, maximum_operand_map_count, sizeof(*operand_map),
        (void**)&operand_map);
  }

  if (iree_status_is_ok(status)) {
    loom_template_decision_build_cursor_t cursor = {
        .features = features,
        .predicates = predicates_storage,
        .constants = constants,
    };
    uint32_t choice_ordinal = 0;
    uint32_t priority_group_ordinal = 0;
    for (uint32_t i = 0; i < out_catalog->model_count; ++i) {
      uint32_t family_priority_group_count = 0;
      loom_template_decision_build_family(
          &build_families[i], operand_map, ranked_choices,
          choices + choice_ordinal, priority_groups + priority_group_ordinal,
          &family_priority_group_count, &cursor);
      choice_ordinal += build_families[i].model->program.choice_count;
      priority_group_ordinal += family_priority_group_count;
    }
    IREE_ASSERT(cursor.feature_ordinal == total_feature_count);
    IREE_ASSERT(cursor.predicate_ordinal == total_predicate_count);
    IREE_ASSERT(cursor.constant_ordinal == total_constant_count);
    IREE_ASSERT(choice_ordinal == total_choice_count);
  }
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  return status;
}

const loom_template_decision_model_t* loom_template_decision_model_lookup(
    const loom_template_decision_model_catalog_t* catalog,
    loom_symbol_ref_t family) {
  if (!loom_symbol_ref_is_valid(family) || family.module_id != 0) {
    return NULL;
  }
  iree_host_size_t begin = 0;
  iree_host_size_t end = catalog->model_count;
  if (catalog->symbol_pages != NULL) {
    const loom_template_decision_model_symbol_page_t* page =
        &catalog->symbol_pages[family.symbol_id >> 8];
    if (page->model_count == 0) return NULL;
    begin = page->first_model_ordinal;
    end = begin + page->model_count;
  }
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_template_decision_model_t* model = &catalog->models[middle];
    if (model->family.symbol_id < family.symbol_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == catalog->model_count ||
      catalog->models[begin].family.symbol_id != family.symbol_id) {
    return NULL;
  }
  return &catalog->models[begin];
}

loom_template_decision_fact_requirements_t
loom_template_decision_model_application_fact_requirements(
    const loom_template_decision_model_t* model,
    const loom_template_demand_t* demand) {
  const bool has_scalar_predicates = iree_any_bit_set(
      model->flags, LOOM_TEMPLATE_DECISION_MODEL_FLAG_HAS_SCALAR_PREDICATES);
  loom_template_decision_fact_requirements_t requirements =
      has_scalar_predicates ? LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_VALUES
                            : 0;

  if (!demand->has_lexical_condition) {
    return requirements;
  }
  if (has_scalar_predicates ||
      iree_any_bit_set(
          model->flags,
          LOOM_TEMPLATE_DECISION_MODEL_FLAG_HAS_PROJECTABLE_TARGET_CONDITIONS)) {
    return requirements | LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_VALUES |
           LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_PATH;
  }
  return requirements;
}

static loom_decision_truth_t loom_template_decision_truth_from_feasibility(
    loom_template_provider_feasibility_t feasibility) {
  switch (feasibility) {
    case LOOM_TEMPLATE_PROVIDER_REJECT:
      return LOOM_DECISION_TRUTH_FALSE;
    case LOOM_TEMPLATE_PROVIDER_MATCH:
      return LOOM_DECISION_TRUTH_TRUE;
    case LOOM_TEMPLATE_PROVIDER_MAYBE:
      return LOOM_DECISION_TRUTH_UNKNOWN;
  }
  IREE_ASSERT_UNREACHABLE("invalid template provider feasibility");
  IREE_BUILTIN_UNREACHABLE();
}

typedef struct loom_template_decision_evaluation_context_t {
  // Model being evaluated.
  const loom_template_decision_model_t* model;

  // Application context bound to the model.
  const loom_template_decision_site_t* site;

  // Target identity evidence accumulated during the decision traversal.
  loom_template_decision_evidence_summary_t* summary;
} loom_template_decision_evaluation_context_t;

static loom_decision_truth_t loom_template_decision_evaluate_feature(
    void* user_data, uint32_t feature_ordinal) {
  loom_template_decision_evaluation_context_t* context =
      (loom_template_decision_evaluation_context_t*)user_data;
  const loom_template_decision_feature_t* feature =
      &context->model->features[feature_ordinal];
  switch ((loom_template_decision_feature_kind_t)feature->kind) {
    case LOOM_TEMPLATE_DECISION_FEATURE_TARGET_IDENTITY: {
      const loom_template_provider_feasibility_t feasibility =
          loom_template_applicability_evaluate_target_requirement(
              context->model->module, feature->value.target_identity.module,
              feature->value.target_identity.target_symbol,
              feature->value.target_identity.target_facts,
              context->site->application_target);
      const loom_decision_program_conjunction_t hard_requirements =
          context->model->program.hard_requirements;
      const bool is_hard_requirement =
          feature_ordinal >= hard_requirements.first_feature &&
          feature_ordinal <
              hard_requirements.first_feature + hard_requirements.feature_count;
      if (is_hard_requirement) {
        context->summary->family_target_identity = feasibility;
      } else {
        context->summary->target_identity_match_count +=
            feasibility == LOOM_TEMPLATE_PROVIDER_MATCH;
        context->summary->target_identity_unresolved_count +=
            feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE;
      }
      return loom_template_decision_truth_from_feasibility(feasibility);
    }
    case LOOM_TEMPLATE_DECISION_FEATURE_TARGET_CONDITION:
      return loom_template_decision_truth_from_feasibility(
          loom_template_applicability_evaluate_target_condition(
              context->model->module, feature->value.target_condition,
              context->site->application_target->facts,
              context->site->application_facts));
  }
  IREE_ASSERT_UNREACHABLE("invalid template decision feature kind");
  IREE_BUILTIN_UNREACHABLE();
}

static loom_decision_truth_t loom_template_decision_refine_predicate(
    void* user_data, uint8_t predicate_kind,
    const loom_decision_predicate_operand_t operands[3]) {
  const loom_template_decision_evaluation_context_t* context =
      (const loom_template_decision_evaluation_context_t*)user_data;
  return loom_template_applicability_refine_predicate(
      context->site->application_facts, predicate_kind, operands);
}

static loom_decision_program_binding_t loom_template_decision_site_binding(
    const loom_template_decision_site_t* site) {
  const loom_value_slice_t arguments =
      loom_template_apply_operands(site->application_op);
  const loom_value_slice_t results =
      loom_template_apply_results(site->application_op);
  return (loom_decision_program_binding_t){
      .facts = site->application_facts->values,
      .argument_values = arguments.values,
      .result_values = results.values,
  };
}

static loom_decision_program_predicate_refiner_t
loom_template_decision_site_predicate_refiner(
    const loom_template_decision_site_t* site,
    loom_template_decision_evaluation_context_t* context) {
  if (site->application_facts->path.integer_relation_count == 0) {
    return loom_decision_program_predicate_refiner_empty();
  }
  return (loom_decision_program_predicate_refiner_t){
      .fn = loom_template_decision_refine_predicate,
      .user_data = context,
  };
}

static void loom_template_decision_finalize_evidence_summary(
    const loom_template_decision_model_t* model,
    const loom_decision_program_result_t* result,
    loom_template_decision_evidence_summary_t* summary) {
  const bool family_unresolved =
      result->kind == LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED &&
      result->unresolved_action_ordinal == LOOM_DECISION_PROGRAM_ACTION_INVALID;
  if (result->kind == LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT ||
      family_unresolved) {
    summary->target_identity_match_count = 0;
    summary->target_identity_unresolved_count = 0;
    return;
  }
  summary->target_identity_match_count +=
      model->target_independent_provider_count;
}

void loom_template_decision_model_evaluate(
    const loom_template_decision_model_t* model,
    const loom_template_decision_site_t* site,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_template_decision_evidence_summary_t* out_summary,
    uint32_t* live_provider_ordinals, uint32_t* out_live_provider_count,
    loom_decision_program_result_t* out_result) {
  *out_summary = (loom_template_decision_evidence_summary_t){
      .family_target_identity = LOOM_TEMPLATE_PROVIDER_MATCH,
      .highest_unresolved_provider_ordinal =
          LOOM_DECISION_PROGRAM_ACTION_INVALID,
      .highest_unresolved_constraint = LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
  };
  loom_template_decision_evaluation_context_t context = {
      .model = model,
      .site = site,
      .summary = out_summary,
  };
  const loom_decision_program_binding_t binding =
      loom_template_decision_site_binding(site);
  loom_decision_program_evaluate(
      &model->program, &binding,
      (loom_decision_program_feature_evaluator_t){
          .fn = loom_template_decision_evaluate_feature,
          .user_data = &context,
      },
      loom_template_decision_site_predicate_refiner(site, &context),
      resolution_policy, live_provider_ordinals, out_live_provider_count,
      out_result);
  loom_template_decision_finalize_evidence_summary(model, out_result,
                                                   out_summary);
}

void loom_template_decision_model_evaluate_all(
    const loom_template_decision_model_t* model,
    const loom_template_decision_site_t* site,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_decision_program_choice_evidence_t* provider_evidence,
    loom_template_decision_evidence_summary_t* out_summary,
    uint32_t* live_provider_ordinals, uint32_t* out_live_provider_count,
    loom_decision_program_result_t* out_result) {
  *out_summary = (loom_template_decision_evidence_summary_t){
      .family_target_identity = LOOM_TEMPLATE_PROVIDER_MATCH,
      .highest_unresolved_provider_ordinal =
          LOOM_DECISION_PROGRAM_ACTION_INVALID,
      .highest_unresolved_constraint = LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
  };
  loom_template_decision_evaluation_context_t context = {
      .model = model,
      .site = site,
      .summary = out_summary,
  };
  const loom_decision_program_binding_t binding =
      loom_template_decision_site_binding(site);
  loom_decision_program_evaluate_all(
      &model->program, &binding,
      (loom_decision_program_feature_evaluator_t){
          .fn = loom_template_decision_evaluate_feature,
          .user_data = &context,
      },
      loom_template_decision_site_predicate_refiner(site, &context),
      resolution_policy, provider_evidence, live_provider_ordinals,
      out_live_provider_count, out_result);
  loom_template_decision_finalize_evidence_summary(model, out_result,
                                                   out_summary);
}

loom_template_decision_constraint_info_t
loom_template_decision_model_constraint_info(
    const loom_template_decision_model_t* model,
    loom_decision_program_constraint_ref_t constraint) {
  loom_template_decision_constraint_info_t info = {0};
  if (constraint == LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID) return info;
  if (!loom_decision_program_constraint_is_feature(constraint)) {
    info.reason = LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE;
    return info;
  }
  const loom_template_decision_feature_t* feature =
      &model->features[loom_decision_program_constraint_ordinal(constraint)];
  if (feature->kind == LOOM_TEMPLATE_DECISION_FEATURE_TARGET_IDENTITY) {
    info.reason = LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_IDENTITY;
  } else {
    info.reason = LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION;
    info.target_condition = feature->value.target_condition;
  }
  return info;
}

void loom_template_decision_model_summarize_choice_evidence(
    const loom_template_decision_model_t* model,
    const loom_decision_program_choice_evidence_t* provider_evidence,
    loom_template_decision_evidence_summary_t* inout_summary) {
  uint32_t first_choice = 0;
  bool found_best_match_group = false;
  for (uint32_t group_ordinal = 0;
       group_ordinal < model->program.priority_group_count; ++group_ordinal) {
    const uint32_t group_choice_count =
        model->program.priority_groups[group_ordinal].choice_count;
    uint32_t group_match_count = 0;
    for (uint32_t i = 0; i < group_choice_count; ++i) {
      const uint32_t choice_ordinal = first_choice + i;
      const loom_decision_program_choice_evidence_t* evidence =
          &provider_evidence[choice_ordinal];
      inout_summary->possible_count +=
          evidence->feasibility != LOOM_DECISION_TRUTH_FALSE;
      group_match_count += evidence->feasibility == LOOM_DECISION_TRUTH_TRUE;
      if (evidence->feasibility == LOOM_DECISION_TRUTH_UNKNOWN &&
          inout_summary->highest_unresolved_provider_ordinal ==
              LOOM_DECISION_PROGRAM_ACTION_INVALID) {
        inout_summary->highest_unresolved_provider_ordinal =
            model->program.choices[choice_ordinal].action_ordinal;
        inout_summary->highest_unresolved_constraint =
            evidence->unresolved_constraint;
      }
    }
    if (!found_best_match_group && group_match_count > 0) {
      inout_summary->best_match_count = group_match_count;
      found_best_match_group = true;
    }
    first_choice += group_choice_count;
  }
}
