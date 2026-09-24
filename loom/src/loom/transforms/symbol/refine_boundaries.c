// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/refine_boundaries.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/analysis/type_refinement.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refinement.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/transforms/cleanup/canonicalizer.h"
#include "loom/transforms/symbol/boundary_graph.h"
#include "loom/transforms/symbol/boundary_pruning.h"
#include "loom/transforms/symbol/boundary_specialization.h"
#include "loom/util/fact_table.h"
#include "loom/util/reference_transfer.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_REFINE_BOUNDARIES_STATISTICS(V, statistics_type)                \
  V(statistics_type, functions_canonicalized, "functions-canonicalized",     \
    "Number of function bodies canonicalized.")                              \
  V(statistics_type, functions_changed, "functions-changed",                 \
    "Number of function canonicalizer runs that changed IR.")                \
  V(statistics_type, boundary_facts_changed, "boundary-facts-changed",       \
    "Number of fixed-point rounds that changed boundary facts.")             \
  V(statistics_type, boundary_replacements_changed,                          \
    "boundary-replacements-changed",                                         \
    "Number of fixed-point rounds that changed boundary replacements.")      \
  V(statistics_type, boundary_replacements_applied,                          \
    "boundary-replacements-applied",                                         \
    "Number of direct boundary value replacements applied.")                 \
  V(statistics_type, boundary_constants_materialized,                        \
    "boundary-constants-materialized",                                       \
    "Number of exact boundary constants materialized.")                      \
  V(statistics_type, boundary_arguments_pruned, "boundary-arguments-pruned", \
    "Number of unused internal function arguments removed.")                 \
  V(statistics_type, boundary_results_pruned, "boundary-results-pruned",     \
    "Number of unused internal function results removed.")                   \
  V(statistics_type, boundary_signature_types_refined,                       \
    "boundary-signature-types-refined",                                      \
    "Number of internal boundary value types refined.")                      \
  V(statistics_type, boundary_specializations_created,                       \
    "boundary-specializations-created",                                      \
    "Number of private function specializations created.")

LOOM_PASS_STATISTICS_DEFINE(loom_refine_boundaries_statistics,
                            loom_refine_boundaries_statistics_t,
                            LOOM_REFINE_BOUNDARIES_STATISTICS)

static const loom_pass_option_def_t kRefineBoundariesOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of boundary fixed-point iterations.")},
};

static const loom_pass_info_t loom_refine_boundaries_pass_info_storage = {
    .name = IREE_SVL("refine-boundaries"),
    .description = IREE_SVL(
        "Propagate direct-call boundary facts and types and canonicalize."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kRefineBoundariesOptions,
    .option_count = IREE_ARRAYSIZE(kRefineBoundariesOptions),
    .statistic_layout = &loom_refine_boundaries_statistics_layout,
};

const loom_pass_info_t* loom_refine_boundaries_pass_info(void) {
  return &loom_refine_boundaries_pass_info_storage;
}

static iree_status_t loom_refine_boundaries_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_refine_boundaries_options_t* options =
      (loom_refine_boundaries_options_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    if (options->max_iterations != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-iterations' for pass 'refine-boundaries'");
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("refine-boundaries"), name, value, &options->max_iterations));
    if (options->max_iterations == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass 'refine-boundaries' option "
                              "'max-iterations' must be greater than 0");
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'refine-boundaries'",
                          (int)name.size, name.data);
}

iree_status_t loom_refine_boundaries_create(loom_pass_t* pass,
                                            iree_string_view_t options_string) {
  loom_refine_boundaries_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  memset(options, 0, sizeof(*options));
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        options->max_iterations = option->uint32_value;
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown decoded option '%.*s' for pass 'refine-boundaries'",
          (int)option->schema->name.size, option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options_string,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_refine_boundaries_parse_option,
                                    .user_data = options,
                                }));
  }
  pass->state = options;
  return iree_ok_status();
}

static const loom_op_t* loom_refine_boundaries_module_anchor(
    const loom_module_t* module) {
  if (!module->body || module->body->block_count == 0) {
    return NULL;
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(module->body);
  return entry_block->first_op;
}

static iree_status_t loom_refine_boundaries_emit_boundary_type_conflict(
    loom_pass_t* pass, const loom_module_t* module, const loom_op_t* op,
    loom_type_t actual_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(pass->info->name),
      loom_param_type(actual_type),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_015,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_refine_boundaries_emit_call_result_type_conflict(
    loom_pass_t* pass, const loom_module_t* module, const loom_op_t* op,
    loom_type_t actual_type, loom_type_t candidate_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(pass->info->name),
      loom_param_type(actual_type),
      loom_param_type(candidate_type),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_016,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_refine_boundaries_emit_nonconvergence(
    loom_pass_t* pass, const loom_module_t* module, uint32_t max_iterations) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(pass->info->name),
      loom_param_u32(max_iterations),
  };
  loom_diagnostic_emission_t emission = {
      .op = loom_refine_boundaries_module_anchor(module),
      .error = LOOM_ERR_LOWERING_043,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(pass->diagnostic_emitter, &emission);
}

//===----------------------------------------------------------------------===//
// Replacement summaries
//===----------------------------------------------------------------------===//

#define LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS 32

typedef enum loom_refine_boundaries_replacement_state_e {
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE = 0,
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE = 1,
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT = 2,
} loom_refine_boundaries_replacement_state_t;

typedef struct loom_refine_boundaries_replacement_entry_t {
  // Original value ID summarized by this entry.
  loom_value_id_t old_value;

  // Current summary state for this value.
  loom_refine_boundaries_replacement_state_t state;

  // Replacement value when state is VALUE.
  loom_value_id_t replacement;
} loom_refine_boundaries_replacement_entry_t;

typedef struct loom_refine_boundaries_replacement_table_t {
  // Arena that owns sparse replacement entries.
  iree_arena_allocator_t* arena;

  // Open-addressed replacement entries keyed by original value ID.
  loom_refine_boundaries_replacement_entry_t* entries;

  // Number of installed replacement entries.
  iree_host_size_t entry_count;

  // Allocated entry count.
  iree_host_size_t entry_capacity;
} loom_refine_boundaries_replacement_table_t;

static iree_host_size_t loom_refine_boundaries_value_hash(
    loom_value_id_t value_id) {
  uint32_t hash = value_id;
  hash ^= hash >> 16;
  hash *= 0x7feb352du;
  hash ^= hash >> 15;
  hash *= 0x846ca68bu;
  hash ^= hash >> 16;
  return (iree_host_size_t)hash;
}

static iree_status_t loom_refine_boundaries_replacement_capacity_for_count(
    iree_host_size_t entry_count, iree_host_size_t* out_entry_capacity) {
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_mul(entry_count, 2, &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replacement table capacity overflow");
  }
  minimum_capacity = iree_max(minimum_capacity, 16);
  iree_host_size_t entry_capacity =
      iree_host_size_next_power_of_two(minimum_capacity);
  if (entry_capacity < minimum_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replacement table capacity overflow");
  }
  *out_entry_capacity = entry_capacity;
  return iree_ok_status();
}

static void loom_refine_boundaries_replacement_entries_initialize(
    loom_refine_boundaries_replacement_entry_t* entries,
    iree_host_size_t entry_capacity) {
  for (iree_host_size_t i = 0; i < entry_capacity; ++i) {
    entries[i] = (loom_refine_boundaries_replacement_entry_t){
        .old_value = LOOM_VALUE_ID_INVALID,
        .state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE,
        .replacement = LOOM_VALUE_ID_INVALID,
    };
  }
}

static loom_refine_boundaries_replacement_entry_t*
loom_refine_boundaries_replacement_table_find_slot(
    loom_refine_boundaries_replacement_entry_t* entries,
    iree_host_size_t entry_capacity, loom_value_id_t old_value) {
  IREE_ASSERT(entry_capacity > 0);
  IREE_ASSERT(iree_host_size_is_power_of_two(entry_capacity));
  const iree_host_size_t mask = entry_capacity - 1;
  iree_host_size_t slot = loom_refine_boundaries_value_hash(old_value) & mask;
  while (true) {
    loom_refine_boundaries_replacement_entry_t* entry = &entries[slot];
    if (entry->old_value == LOOM_VALUE_ID_INVALID ||
        entry->old_value == old_value) {
      return entry;
    }
    slot = (slot + 1) & mask;
  }
}

static const loom_refine_boundaries_replacement_entry_t*
loom_refine_boundaries_replacement_table_find_const_slot(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value) {
  if (table->entry_capacity == 0) {
    return NULL;
  }
  const loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_slot(
          table->entries, table->entry_capacity, old_value);
  return entry->old_value == old_value ? entry : NULL;
}

static void loom_refine_boundaries_replacement_table_insert_entry(
    loom_refine_boundaries_replacement_entry_t* entries,
    iree_host_size_t entry_capacity,
    loom_refine_boundaries_replacement_entry_t source_entry) {
  loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_slot(
          entries, entry_capacity, source_entry.old_value);
  IREE_ASSERT(entry->old_value == LOOM_VALUE_ID_INVALID);
  *entry = source_entry;
}

static iree_status_t loom_refine_boundaries_replacement_table_ensure_capacity(
    loom_refine_boundaries_replacement_table_t* table,
    iree_host_size_t required_entry_count) {
  if (required_entry_count <= table->entry_capacity / 2) {
    return iree_ok_status();
  }
  iree_host_size_t new_entry_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_capacity_for_count(
      required_entry_count, &new_entry_capacity));
  loom_refine_boundaries_replacement_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(table->arena, new_entry_capacity,
                                sizeof(*new_entries), (void**)&new_entries));
  loom_refine_boundaries_replacement_entries_initialize(new_entries,
                                                        new_entry_capacity);
  for (iree_host_size_t i = 0; i < table->entry_capacity; ++i) {
    loom_refine_boundaries_replacement_entry_t entry = table->entries[i];
    if (entry.old_value == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_refine_boundaries_replacement_table_insert_entry(
        new_entries, new_entry_capacity, entry);
  }
  table->entries = new_entries;
  table->entry_capacity = new_entry_capacity;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_initialize(
    loom_refine_boundaries_replacement_table_t* table,
    iree_arena_allocator_t* arena) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  return iree_ok_status();
}

static bool loom_refine_boundaries_replacement_table_lookup(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t* out_replacement) {
  const loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_const_slot(table,
                                                               old_value);
  if (!entry || entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) {
    return false;
  }
  *out_replacement = entry->replacement;
  return true;
}

static bool loom_refine_boundaries_replacement_table_resolve(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t* out_replacement) {
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_refine_boundaries_replacement_table_lookup(table, old_value,
                                                       &replacement)) {
    return false;
  }

  for (iree_host_size_t depth = 0; depth < table->entry_count; ++depth) {
    loom_value_id_t next = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(table, replacement,
                                                         &next)) {
      *out_replacement = replacement;
      return true;
    }
    if (next == replacement || next == old_value) {
      return false;
    }
    replacement = next;
  }
  return false;
}

static iree_status_t loom_refine_boundaries_replacement_table_entry(
    loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value,
    loom_refine_boundaries_replacement_entry_t** out_entry) {
  *out_entry = NULL;
  if (old_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (table->entry_capacity > 0) {
    loom_refine_boundaries_replacement_entry_t* entry =
        loom_refine_boundaries_replacement_table_find_slot(
            table->entries, table->entry_capacity, old_value);
    if (entry->old_value == old_value) {
      *out_entry = entry;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_ensure_capacity(
      table, table->entry_count + 1));
  loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_slot(
          table->entries, table->entry_capacity, old_value);
  if (entry->old_value == LOOM_VALUE_ID_INVALID) {
    entry->old_value = old_value;
    entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE;
    entry->replacement = LOOM_VALUE_ID_INVALID;
    ++table->entry_count;
  }
  *out_entry = entry;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_define(
    loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t replacement) {
  if (replacement == LOOM_VALUE_ID_INVALID || old_value == replacement) {
    return iree_ok_status();
  }

  loom_refine_boundaries_replacement_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_refine_boundaries_replacement_table_entry(table, old_value, &entry));
  if (!entry) {
    return iree_ok_status();
  }
  if (entry->state == LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE) {
    entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE;
    entry->replacement = replacement;
    return iree_ok_status();
  }
  if (entry->state == LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT) {
    return iree_ok_status();
  }
  if (entry->replacement != replacement) {
    entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT;
    entry->replacement = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_block(
    loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value) {
  loom_refine_boundaries_replacement_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_refine_boundaries_replacement_table_entry(table, old_value, &entry));
  if (!entry) {
    return iree_ok_status();
  }
  entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT;
  entry->replacement = LOOM_VALUE_ID_INVALID;
  return iree_ok_status();
}

static bool loom_refine_boundaries_replacement_tables_equal(
    const loom_refine_boundaries_replacement_table_t* lhs,
    const loom_refine_boundaries_replacement_table_t* rhs) {
  for (iree_host_size_t i = 0; i < lhs->entry_capacity; ++i) {
    const loom_refine_boundaries_replacement_entry_t* lhs_entry =
        &lhs->entries[i];
    if (lhs_entry->old_value == LOOM_VALUE_ID_INVALID ||
        lhs_entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) {
      continue;
    }
    loom_value_id_t rhs_replacement = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(
            rhs, lhs_entry->old_value, &rhs_replacement)) {
      return false;
    }
    if (lhs_entry->replacement != rhs_replacement) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < rhs->entry_capacity; ++i) {
    const loom_refine_boundaries_replacement_entry_t* rhs_entry =
        &rhs->entries[i];
    if (rhs_entry->old_value == LOOM_VALUE_ID_INVALID ||
        rhs_entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) {
      continue;
    }
    loom_value_id_t lhs_replacement = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(
            lhs, rhs_entry->old_value, &lhs_replacement)) {
      return false;
    }
    if (rhs_entry->replacement != lhs_replacement) {
      return false;
    }
  }
  return true;
}

typedef struct loom_refine_boundaries_boundary_state_t {
  // Fixed-point facts known for function boundary values in one round.
  loom_value_fact_table_t facts;

  // Fixed-point value replacements known for boundary values in one round.
  loom_refine_boundaries_replacement_table_t replacements;
} loom_refine_boundaries_boundary_state_t;

static iree_status_t loom_refine_boundaries_boundary_state_initialize(
    loom_refine_boundaries_boundary_state_t* state,
    iree_arena_allocator_t* arena, iree_host_size_t fact_value_capacity) {
  memset(state, 0, sizeof(*state));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&state->facts, arena,
                                                        fact_value_capacity));
  loom_type_registry_configure_fact_context(&state->facts.context);
  return loom_refine_boundaries_replacement_table_initialize(
      &state->replacements, arena);
}

//===----------------------------------------------------------------------===//
// Boundary type refinement
//===----------------------------------------------------------------------===//

static iree_status_t loom_refine_boundaries_refine_value_type_with_facts(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* owner_op,
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    int64_t* changed_count) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t current_type = loom_module_value_type(module, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_specialize_with_value_facts(
      module, current_type, facts, &module->arena, &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return loom_refine_boundaries_emit_boundary_type_conflict(
        pass, module, owner_op, current_type);
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
      loom_type_equal(current_type, refined_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(module, value_id, refined_type));
  *changed_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_refine_function_signature(
    loom_pass_t* pass, loom_module_t* module,
    const loom_value_fact_table_t* facts,
    const loom_refine_boundaries_function_t* function_info,
    int64_t* changed_count) {
  if (!function_info->can_refine_boundary) {
    return iree_ok_status();
  }
  for (uint8_t projection_index = 0;
       !loom_pass_has_error_diagnostics(pass) &&
       projection_index < function_info->argument_projection_count;
       ++projection_index) {
    const loom_refine_boundaries_argument_projection_t* projection =
        &function_info->argument_projections[projection_index];
    for (uint16_t i = 0; !loom_pass_has_error_diagnostics(pass) &&
                         i < function_info->argument_count;
         ++i) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_value_type_with_facts(
          pass, module, function_info->function.op, facts,
          loom_block_arg_id(projection->entry_block, i), changed_count));
    }
  }
  const loom_value_id_t* results =
      loom_op_const_results(function_info->function.op);
  for (uint16_t i = 0; !loom_pass_has_error_diagnostics(pass) &&
                       i < function_info->result_count;
       ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_value_type_with_facts(
        pass, module, function_info->function.op, facts, results[i],
        changed_count));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fact summaries
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_refine_boundaries_scalar_fact(
    loom_value_facts_t facts) {
  facts.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  return facts;
}

static iree_status_t loom_refine_boundaries_join_facts(
    loom_value_fact_table_t* target_table,
    const loom_value_fact_table_t* existing_table,
    loom_value_facts_t existing_facts,
    const loom_value_fact_table_t* incoming_table,
    loom_value_facts_t incoming_facts, loom_value_facts_t* out_joined_facts) {
  if (loom_value_fact_table_facts_equal(existing_table, existing_facts,
                                        incoming_table, incoming_facts)) {
    if (existing_table == target_table) {
      *out_joined_facts = existing_facts;
      return iree_ok_status();
    }
    return loom_value_fact_table_clone_fact(target_table, existing_table,
                                            existing_facts, out_joined_facts);
  }

  loom_value_facts_t existing_scalar =
      loom_refine_boundaries_scalar_fact(existing_facts);
  loom_value_facts_t incoming_scalar =
      loom_refine_boundaries_scalar_fact(incoming_facts);
  loom_value_facts_meet(&existing_scalar, &incoming_scalar, out_joined_facts);
  if (loom_value_fact_table_extensions_equal(existing_table, existing_facts,
                                             incoming_table, incoming_facts)) {
    if (existing_table == target_table) {
      out_joined_facts->extension_id = existing_facts.extension_id;
    } else {
      loom_value_facts_t cloned_existing = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target_table, existing_table, existing_facts, &cloned_existing));
      out_joined_facts->extension_id = cloned_existing.extension_id;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_merge_fact(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    const loom_value_fact_table_t* source_table, loom_value_facts_t facts) {
  // An observed unknown input participates in the join. Keeping it distinct
  // from an unseen boundary prevents later callers from narrowing the result.
  if (!loom_value_fact_table_has_entry(table, value_id)) {
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
        table, source_table, facts, &cloned_facts));
    return loom_value_fact_table_define(table, value_id, cloned_facts);
  }
  loom_value_facts_t existing = loom_value_fact_table_lookup(table, value_id);
  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_join_facts(
      table, table, existing, source_table, facts, &joined));
  return loom_value_fact_table_define(table, value_id, joined);
}

static bool loom_refine_boundaries_fact_tables_equal(
    const loom_value_fact_table_t* lhs, const loom_value_fact_table_t* rhs) {
  iree_host_size_t count = lhs->count > rhs->count ? lhs->count : rhs->count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_value_facts_t lhs_facts =
        loom_value_fact_table_lookup(lhs, (loom_value_id_t)i);
    loom_value_facts_t rhs_facts =
        loom_value_fact_table_lookup(rhs, (loom_value_id_t)i);
    if (!loom_value_fact_table_facts_equal(lhs, lhs_facts, rhs, rhs_facts)) {
      return false;
    }
  }
  return true;
}

static void loom_refine_boundaries_merge_canonicalize_result(
    loom_canonicalizer_result_t* target,
    const loom_canonicalizer_result_t* source) {
  target->changed |= source->changed;
  target->facts_changed |= source->facts_changed;
  target->types_changed |= source->types_changed;
  target->boundary_maybe_changed |= source->boundary_maybe_changed;
  target->ops_modified += source->ops_modified;
  target->type_propagation_conflicts += source->type_propagation_conflicts;
  target->type_propagation_rejection_cache_hits +=
      source->type_propagation_rejection_cache_hits;
}

static int32_t loom_refine_boundaries_find_argument_index(
    const loom_refine_boundaries_function_t* function,
    loom_value_id_t value_id) {
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    if (function->argument_ids[i] == value_id) {
      return (int32_t)i;
    }
  }
  return LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
}

//===----------------------------------------------------------------------===//
// Replacement application
//===----------------------------------------------------------------------===//

static bool loom_refine_boundaries_value_has_uses(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return value->use_count > 0 ||
         loom_module_value_has_type_uses(module, value_id);
}

static iree_status_t loom_refine_boundaries_apply_direct_value_replacement(
    loom_module_t* module, loom_value_id_t old_value,
    loom_value_id_t replacement, int64_t* applied_count) {
  if (replacement == LOOM_VALUE_ID_INVALID ||
      replacement >= module->values.count ||
      old_value >= module->values.count || replacement == old_value) {
    return iree_ok_status();
  }
  if (!loom_refine_boundaries_value_has_uses(module, old_value)) {
    return iree_ok_status();
  }
  if (!loom_type_equal(loom_module_value_type(module, old_value),
                       loom_module_value_type(module, replacement))) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(module, old_value, replacement));
  *applied_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_apply_value_replacement(
    loom_module_t* module,
    const loom_refine_boundaries_replacement_table_t* replacements,
    loom_value_id_t old_value, int64_t* applied_count) {
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_refine_boundaries_replacement_table_resolve(replacements, old_value,
                                                        &replacement)) {
    return iree_ok_status();
  }
  return loom_refine_boundaries_apply_direct_value_replacement(
      module, old_value, replacement, applied_count);
}

typedef struct loom_refine_boundaries_apply_t {
  // Module being rewritten.
  loom_module_t* module;

  // Replacement summary for the current fixed-point round.
  const loom_refine_boundaries_replacement_table_t* replacements;

  // Boundary facts from the current fixed-point round.
  const loom_value_fact_table_t* boundary_facts;

  // Local boundary values supplied to function canonicalization.
  struct {
    // Walk storage kept alive until canonicalization consumes the seeds.
    iree_arena_allocator_t* arena;
    // Unique arguments and op results with boundary facts.
    loom_value_id_t* values;
    // Number of populated values.
    iree_host_size_t count;
    // Allocated capacity of values.
    iree_host_size_t capacity;
  } seeds;

  // Number of replacements applied while walking this function.
  int64_t* applied_count;

  // Number of constants materialized while walking this function.
  int64_t* materialized_count;
} loom_refine_boundaries_apply_t;

static iree_status_t loom_refine_boundaries_append_seed(
    loom_refine_boundaries_apply_t* apply, loom_value_id_t value_id) {
  if (!loom_value_fact_table_has_entry(apply->boundary_facts, value_id)) {
    return iree_ok_status();
  }
  if (apply->seeds.count == apply->seeds.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        apply->seeds.arena, apply->seeds.count, apply->seeds.count + 1,
        sizeof(*apply->seeds.values), &apply->seeds.capacity,
        (void**)&apply->seeds.values));
  }
  apply->seeds.values[apply->seeds.count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_materialize_exact_value(
    loom_module_t* module, const loom_value_fact_table_t* boundary_facts,
    loom_builder_t* builder, loom_value_id_t fact_value,
    loom_value_id_t old_value, loom_location_id_t location,
    int64_t* materialized_count) {
  if (!loom_refine_boundaries_value_has_uses(module, old_value)) {
    return iree_ok_status();
  }
  if (!loom_value_fact_table_has_entry(boundary_facts, fact_value)) {
    return iree_ok_status();
  }
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(boundary_facts, fact_value);
  loom_type_t type = loom_module_value_type(module, old_value);
  if (!loom_value_facts_can_materialize_constant(facts, type)) {
    return iree_ok_status();
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_constant_build(builder, facts, type, location, &replacement));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(module, old_value, replacement));
  *materialized_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_apply_op_boundary_values(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (op->result_count == 0) {
    return iree_ok_status();
  }

  loom_refine_boundaries_apply_t* apply =
      (loom_refine_boundaries_apply_t*)user_data;
  loom_builder_t builder;
  loom_builder_initialize(apply->module, &apply->module->arena,
                          op->parent_block, &builder);
  loom_builder_set_before(&builder, op);

  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_append_seed(apply, results[i]));
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_apply_value_replacement(
        apply->module, apply->replacements, results[i], apply->applied_count));
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_materialize_exact_value(
        apply->module, apply->boundary_facts, &builder, results[i], results[i],
        op->location, apply->materialized_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_apply_logical_argument_replacement(
    loom_module_t* module,
    const loom_refine_boundaries_replacement_table_t* replacements,
    const loom_refine_boundaries_function_t* function_info,
    const loom_refine_boundaries_argument_projection_t* projection,
    uint16_t argument_index, int64_t* applied_count) {
  loom_value_id_t canonical_argument =
      function_info->argument_ids[argument_index];
  loom_value_id_t canonical_replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_refine_boundaries_replacement_table_resolve(
          replacements, canonical_argument, &canonical_replacement)) {
    return iree_ok_status();
  }

  int32_t replacement_index = loom_refine_boundaries_find_argument_index(
      function_info, canonical_replacement);
  if (replacement_index < 0 ||
      (uint16_t)replacement_index >= projection->entry_block->arg_count) {
    return iree_ok_status();
  }

  loom_value_id_t old_value =
      loom_block_arg_id(projection->entry_block, argument_index);
  loom_value_id_t replacement =
      loom_block_arg_id(projection->entry_block, (uint16_t)replacement_index);
  return loom_refine_boundaries_apply_direct_value_replacement(
      module, old_value, replacement, applied_count);
}

static iree_status_t loom_refine_boundaries_apply_projected_argument_values(
    loom_module_t* module,
    const loom_refine_boundaries_replacement_table_t* replacements,
    const loom_value_fact_table_t* boundary_facts,
    loom_refine_boundaries_function_t* function_info, int64_t* applied_count,
    int64_t* materialized_count) {
  for (uint8_t projection_index = 0;
       projection_index < function_info->argument_projection_count;
       ++projection_index) {
    const loom_refine_boundaries_argument_projection_t* projection =
        &function_info->argument_projections[projection_index];
    loom_block_t* entry_block = projection->entry_block;
    loom_builder_t entry_builder;
    loom_builder_initialize(module, &module->arena, entry_block,
                            &entry_builder);
    if (entry_block->first_op) {
      loom_builder_set_before(&entry_builder, entry_block->first_op);
    } else {
      entry_builder.ip.parent_op = function_info->function.op;
    }
    for (uint16_t i = 0; i < function_info->argument_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_refine_boundaries_apply_logical_argument_replacement(
              module, replacements, function_info, projection, i,
              applied_count));
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_materialize_exact_value(
          module, boundary_facts, &entry_builder,
          function_info->argument_ids[i], loom_block_arg_id(entry_block, i),
          function_info->function.op->location, materialized_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_apply_function_boundary_values(
    loom_module_t* module,
    const loom_refine_boundaries_replacement_table_t* replacements,
    const loom_value_fact_table_t* boundary_facts,
    loom_refine_boundaries_function_t* function_info,
    iree_arena_allocator_t* walk_arena, int64_t* out_applied_count,
    int64_t* out_materialized_count,
    loom_value_fact_table_view_t* out_seed_facts) {
  *out_applied_count = 0;
  *out_materialized_count = 0;
  *out_seed_facts = (loom_value_fact_table_view_t){0};
  loom_region_t* body = function_info->body;
  if (!body) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_refine_boundaries_apply_projected_argument_values(
      module, replacements, boundary_facts, function_info, out_applied_count,
      out_materialized_count));

  loom_refine_boundaries_apply_t apply = {
      .module = module,
      .replacements = replacements,
      .boundary_facts = boundary_facts,
      .seeds.arena = walk_arena,
      .applied_count = out_applied_count,
      .materialized_count = out_materialized_count,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  for (uint16_t i = 0; i < function_info->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_append_seed(
        &apply, function_info->argument_ids[i]));
  }
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function_info->function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_apply_op_boundary_values,
                             &apply},
      walk_arena, &walk_result));
  *out_seed_facts = (loom_value_fact_table_view_t){
      .table = boundary_facts,
      .value_ids = apply.seeds.values,
      .value_count = apply.seeds.count,
  };
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Boundary collection
//===----------------------------------------------------------------------===//

typedef struct loom_refine_boundaries_collect_t {
  // Function graph for resolving callees.
  loom_refine_boundaries_graph_t* graph;

  // Current function facts after canonicalization.
  const loom_value_fact_table_t* function_facts;

  // Boundary facts being produced for the next fixed-point round.
  loom_value_fact_table_t* next_boundary_facts;

  // Boundary replacements being produced for the next fixed-point round.
  loom_refine_boundaries_replacement_table_t* next_boundary_replacements;

  // Function whose body is being walked.
  loom_refine_boundaries_function_t* current_function;

  // Projected invocation containing the calls currently being collected.
  loom_value_fact_reference_origin_t reference_origin;
} loom_refine_boundaries_collect_t;

static bool loom_refine_boundaries_values_have_equal_types(
    const loom_module_t* module, loom_value_id_t lhs, loom_value_id_t rhs) {
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID ||
      lhs >= module->values.count || rhs >= module->values.count) {
    return false;
  }
  return loom_type_equal(loom_module_value_type(module, lhs),
                         loom_module_value_type(module, rhs));
}

static void loom_refine_boundaries_join_forward_index(
    int32_t* forward_indices, iree_host_size_t result_index,
    int32_t forward_index) {
  int32_t* slot = &forward_indices[result_index];
  if (*slot == LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN) {
    *slot = forward_index;
    return;
  }
  if (*slot != forward_index) {
    *slot = LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
  }
}

static int32_t loom_refine_boundaries_find_prior_result_index(
    const loom_module_t* module,
    const loom_refine_boundaries_function_t* function,
    loom_value_slice_t operands, iree_host_size_t result_index) {
  if (result_index >= operands.count) {
    return LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
  }
  const loom_value_id_t* function_results =
      loom_op_const_results(function->function.op);
  for (iree_host_size_t i = 0; i < result_index; ++i) {
    if (operands.values[i] != operands.values[result_index]) {
      continue;
    }
    if (!loom_refine_boundaries_values_have_equal_types(
            module, function_results[i], function_results[result_index])) {
      continue;
    }
    return (int32_t)i;
  }
  return LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
}

static iree_status_t loom_refine_boundaries_collect_return(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* op) {
  loom_value_slice_t operands = {
      .values = loom_op_operands((loom_op_t*)op),
      .count = op->operand_count,
  };
  loom_refine_boundaries_function_t* function = collect->current_function;
  iree_host_size_t count = operands.count < function->result_count
                               ? operands.count
                               : function->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    int32_t argument_index = loom_refine_boundaries_find_argument_index(
        function, operands.values[i]);
    loom_refine_boundaries_join_forward_index(
        function->return_forward_argument_indices, i, argument_index);

    int32_t result_index = loom_refine_boundaries_find_prior_result_index(
        collect->graph->module, function, operands, i);
    loom_refine_boundaries_join_forward_index(
        function->return_forward_result_indices, i, result_index);

    loom_value_facts_t facts = loom_value_fact_table_lookup(
        collect->function_facts, operands.values[i]);
    if (!function->return_fact_defined[i]) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          collect->next_boundary_facts, collect->function_facts, facts,
          &function->return_facts[i]));
      function->return_fact_defined[i] = true;
    } else {
      loom_value_facts_t joined_facts = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_join_facts(
          collect->next_boundary_facts, collect->next_boundary_facts,
          function->return_facts[i], collect->function_facts, facts,
          &joined_facts));
      function->return_facts[i] = joined_facts;
    }
  }
  for (iree_host_size_t i = count; i < function->result_count; ++i) {
    loom_refine_boundaries_join_forward_index(
        function->return_forward_argument_indices, i,
        LOOM_REFINE_BOUNDARIES_FORWARD_NONE);
    loom_refine_boundaries_join_forward_index(
        function->return_forward_result_indices, i,
        LOOM_REFINE_BOUNDARIES_FORWARD_NONE);
  }
  function->has_return_facts = true;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_argument_equality(
    loom_refine_boundaries_collect_t* collect,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands) {
  if (!callee_info->can_refine_boundary) {
    return iree_ok_status();
  }
  for (uint16_t i = 1; i < callee_info->argument_count; ++i) {
    loom_value_id_t old_argument = callee_info->argument_ids[i];
    if (i >= operands.count) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_block(
          collect->next_boundary_replacements, old_argument));
      continue;
    }

    loom_value_id_t replacement_argument = LOOM_VALUE_ID_INVALID;
    for (uint16_t j = 0; j < i && j < operands.count; ++j) {
      if (operands.values[j] != operands.values[i]) {
        continue;
      }
      if (!loom_refine_boundaries_values_have_equal_types(
              collect->graph->module, old_argument,
              callee_info->argument_ids[j])) {
        continue;
      }
      replacement_argument = callee_info->argument_ids[j];
      break;
    }

    if (replacement_argument == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_block(
          collect->next_boundary_replacements, old_argument));
    } else {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_define(
          collect->next_boundary_replacements, old_argument,
          replacement_argument));
    }
  }
  return iree_ok_status();
}

static bool loom_refine_boundaries_call_result_is_tied(const loom_op_t* call_op,
                                                       iree_host_size_t index) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(call_op);
  for (uint16_t i = 0; i < call_op->tied_result_count; ++i) {
    if (tied_results[i].result_index == index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_refine_boundaries_collect_return_forwarding(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* call_op,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results) {
  if (!callee_info->return_forward_argument_indices) {
    return iree_ok_status();
  }
  iree_host_size_t count = results.count < callee_info->result_count
                               ? results.count
                               : callee_info->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (loom_refine_boundaries_call_result_is_tied(call_op, i)) {
      continue;
    }

    int32_t argument_index = callee_info->return_forward_argument_indices[i];
    if (argument_index < 0 ||
        (iree_host_size_t)argument_index >= operands.count) {
      continue;
    }
    loom_value_id_t result = results.values[i];
    loom_value_id_t replacement = operands.values[argument_index];
    if (!loom_refine_boundaries_values_have_equal_types(collect->graph->module,
                                                        result, replacement)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_define(
        collect->next_boundary_replacements, result, replacement));
  }
  if (!callee_info->return_forward_result_indices) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (callee_info->return_forward_argument_indices[i] >= 0) {
      continue;
    }
    if (loom_refine_boundaries_call_result_is_tied(call_op, i)) {
      continue;
    }

    int32_t result_index = callee_info->return_forward_result_indices[i];
    if (result_index < 0 || (iree_host_size_t)result_index >= results.count) {
      continue;
    }
    loom_value_id_t result = results.values[i];
    loom_value_id_t replacement = results.values[result_index];
    if (!loom_refine_boundaries_values_have_equal_types(collect->graph->module,
                                                        result, replacement)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_define(
        collect->next_boundary_replacements, result, replacement));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_call(
    loom_refine_boundaries_collect_t* collect, loom_op_t* op) {
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(collect->graph->module, op, NULL,
                                        &callee, &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(collect->graph, callee,
                                          &callee_node)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_function_t* callee_info =
      &collect->graph->functions[callee_node];
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_argument_equality(
      collect, callee_info, operands));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_return_forwarding(
      collect, op, callee_info, operands, results));

  if (callee_info->can_refine_argument_facts) {
    iree_host_size_t count = operands.count < callee_info->argument_count
                                 ? operands.count
                                 : callee_info->argument_count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_value_facts_t facts = loom_value_fact_table_lookup(
          collect->function_facts, operands.values[i]);
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_merge_fact(
          collect->next_boundary_facts, callee_info->argument_ids[i],
          collect->function_facts, facts));
    }
  }

  if (!callee_info->has_return_facts) {
    return iree_ok_status();
  }
  loom_reference_call_t reference_call;
  bool has_reference_call = false;
  iree_host_size_t count = results.count < callee_info->result_count
                               ? results.count
                               : callee_info->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!callee_info->return_fact_defined[i]) {
      continue;
    }
    loom_value_facts_t facts = callee_info->return_facts[i];
    const loom_value_fact_reference_origin_t origin =
        loom_value_facts_reference_origin(
            &collect->next_boundary_facts->context, facts);
    if (origin.kind != LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN) {
      if (!has_reference_call) {
        loom_reference_call_initialize(
            collect->graph->module, collect->function_facts, operands,
            collect->reference_origin, &reference_call);
        has_reference_call = true;
      }
      IREE_RETURN_IF_ERROR(loom_value_facts_rebind_reference_origin(
          &collect->next_boundary_facts->context,
          loom_reference_call_result_origin(&reference_call, origin), &facts));
    }
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_merge_fact(
        collect->next_boundary_facts, results.values[i],
        collect->next_boundary_facts, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_refine_boundaries_collect_t* collect =
      (loom_refine_boundaries_collect_t*)user_data;
  if (op->kind == collect->current_function->body_exit_kind &&
      op->parent_block->parent_region == collect->current_function->body) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_return(collect, op));
  }
  return loom_refine_boundaries_collect_call(collect, op);
}

static iree_status_t loom_refine_boundaries_collect_function(
    loom_refine_boundaries_graph_t* graph,
    const loom_value_fact_table_t* function_facts,
    loom_value_fact_table_t* next_boundary_facts,
    loom_refine_boundaries_replacement_table_t* next_boundary_replacements,
    loom_refine_boundaries_function_t* function_info) {
  loom_refine_boundaries_collect_t collect = {
      .graph = graph,
      .function_facts = function_facts,
      .next_boundary_facts = next_boundary_facts,
      .next_boundary_replacements = next_boundary_replacements,
      .current_function = function_info,
  };
  for (uint8_t i = 0; i < loom_func_like_region_count(function_info->function);
       ++i) {
    loom_region_t* region = loom_func_like_region(function_info->function, i);
    if (!region) {
      continue;
    }
    collect.reference_origin = (loom_value_fact_reference_origin_t){
        .function_symbol_id =
            loom_func_like_callee(function_info->function).symbol_id,
        .entry_value_id = LOOM_VALUE_ID_INVALID,
        .region_index = i,
        .kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
    };
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    iree_arena_reset(graph->walk_arena);
    IREE_RETURN_IF_ERROR(loom_walk_region(
        graph->module, region, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_refine_boundaries_collect_op, &collect},
        graph->walk_arena, &walk_result));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Function refinement
//===----------------------------------------------------------------------===//

static bool loom_refine_boundaries_can_refine_boundary(void* user_data,
                                                       loom_op_t* op) {
  const loom_refine_boundaries_graph_t* graph =
      (const loom_refine_boundaries_graph_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_func_like_t function = loom_func_like_cast(graph->module, op);
  if (loom_func_like_isa(function)) {
    callee = loom_func_like_callee(function);
  } else {
    loom_value_slice_t operands = {0};
    loom_value_slice_t results = {0};
    if (!loom_refine_boundaries_read_call(graph->module, op, NULL, &callee,
                                          &operands, &results)) {
      return false;
    }
  }
  iree_host_size_t node = 0;
  return loom_refine_boundaries_callee_node(graph, callee, &node) &&
         graph->functions[node].can_refine_boundary;
}

static iree_status_t loom_refine_boundaries_run_function(
    loom_pass_t* pass, loom_canonicalizer_t* canonicalizer,
    loom_refine_boundaries_graph_t* graph, loom_value_fact_table_t* seed_facts,
    const loom_refine_boundaries_replacement_table_t* seed_replacements,
    loom_value_fact_table_t* next_boundary_facts,
    loom_refine_boundaries_replacement_table_t* next_boundary_replacements,
    loom_refine_boundaries_function_t* function_info,
    int64_t* signature_type_changed_count) {
  loom_refine_boundaries_statistics_t* statistics =
      loom_refine_boundaries_statistics(pass);
  int64_t replacements_applied = 0;
  int64_t constants_materialized = 0;
  loom_canonicalizer_options_t options = {
      .refine_boundary = {loom_refine_boundaries_can_refine_boundary, graph},
  };
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_apply_function_boundary_values(
      graph->module, seed_replacements, seed_facts, function_info,
      graph->walk_arena, &replacements_applied, &constants_materialized,
      &options.seed_facts));
  loom_canonicalizer_result_t canonicalize_result = {0};
  loom_canonicalizer_result_t body_result = {0};
  IREE_RETURN_IF_ERROR(loom_canonicalizer_run_function(
      canonicalizer, function_info->function, &options, &body_result));
  loom_refine_boundaries_merge_canonicalize_result(&canonicalize_result,
                                                   &body_result);
  const loom_value_fact_table_t* function_facts =
      loom_canonicalizer_fact_table(canonicalizer);
  if (function_facts) {
    int64_t signature_type_changes_before = *signature_type_changed_count;
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_function_signature(
        pass, graph->module, function_facts, function_info,
        signature_type_changed_count));
    if (loom_pass_has_error_diagnostics(pass)) {
      return iree_ok_status();
    }
    if (signature_type_changes_before != *signature_type_changed_count) {
      loom_pass_mark_changed(pass);
    }

    IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_function(
        graph, function_facts, next_boundary_facts, next_boundary_replacements,
        function_info));
  }

  ++statistics->functions_canonicalized;
  if (replacements_applied > 0 || constants_materialized > 0 ||
      canonicalize_result.changed) {
    ++statistics->functions_changed;
  }
  if (replacements_applied > 0) {
    statistics->boundary_replacements_applied += replacements_applied;
  }
  if (constants_materialized > 0) {
    statistics->boundary_constants_materialized += constants_materialized;
  }
  if (replacements_applied > 0 || constants_materialized > 0 ||
      canonicalize_result.changed) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Boundary call result type refinement
//===----------------------------------------------------------------------===//

static iree_status_t loom_refine_boundaries_substitute_call_result_type(
    loom_module_t* module, const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results,
    loom_type_t callee_type, loom_type_t* out_type) {
  *out_type = callee_type;
  for (uint16_t i = 0; i < callee_info->argument_count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
        module, *out_type, callee_info->argument_ids[i], operands.values[i],
        out_type, &changed));
  }

  const loom_value_id_t* callee_results =
      loom_op_const_results(callee_info->function.op);
  for (uint16_t i = 0; i < callee_info->result_count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
        module, *out_type, callee_results[i], results.values[i], out_type,
        &changed));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_refine_call_result_type(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* call_op,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results,
    iree_host_size_t result_index, int64_t* changed_count) {
  if (result_index >= results.count ||
      result_index >= callee_info->result_count) {
    return iree_ok_status();
  }

  const loom_value_id_t* callee_results =
      loom_op_const_results(callee_info->function.op);
  loom_type_t candidate_type =
      loom_module_value_type(module, callee_results[result_index]);
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_substitute_call_result_type(
      module, callee_info, operands, results, candidate_type, &candidate_type));

  loom_value_id_t result_value = results.values[result_index];
  loom_type_t current_type = loom_module_value_type(module, result_value);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_with_candidate(
      current_type, candidate_type, &module->arena, &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return loom_refine_boundaries_emit_call_result_type_conflict(
        pass, module, call_op, current_type, candidate_type);
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
      loom_type_equal(current_type, refined_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(module, result_value, refined_type));
  *changed_count += 1;
  return iree_ok_status();
}

typedef struct loom_refine_boundaries_signature_call_walk_t {
  // Current pass instance used for structured diagnostics.
  loom_pass_t* pass;

  // Module being rewritten.
  loom_module_t* module;

  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Number of call result type changes applied while walking.
  int64_t* changed_count;
} loom_refine_boundaries_signature_call_walk_t;

static iree_status_t loom_refine_boundaries_refine_call_result_types(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_signature_call_walk_t* walk =
      (loom_refine_boundaries_signature_call_walk_t*)user_data;
  if (loom_pass_has_error_diagnostics(walk->pass)) {
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }

  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->module, op, NULL, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  const loom_refine_boundaries_function_t* callee_info =
      &walk->graph->functions[callee_node];
  if (!callee_info->can_refine_boundary) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0;
       !loom_pass_has_error_diagnostics(walk->pass) && i < results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_call_result_type(
        walk->pass, walk->module, op, callee_info, operands, results, i,
        walk->changed_count));
  }
  if (loom_pass_has_error_diagnostics(walk->pass)) {
    *out_result = LOOM_WALK_ABORT;
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_refine_internal_signature_types(
    loom_pass_t* pass, loom_module_t* module,
    const loom_refine_boundaries_graph_t* graph,
    const loom_value_fact_table_t* boundary_facts,
    iree_arena_allocator_t* walk_arena, int64_t* out_changed_count) {
  *out_changed_count = 0;
  for (iree_host_size_t node = 0;
       !loom_pass_has_error_diagnostics(pass) && node < graph->function_count;
       ++node) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_function_signature(
        pass, module, boundary_facts, &graph->functions[node],
        out_changed_count));
  }
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_signature_call_walk_t walk = {
      .pass = pass,
      .module = module,
      .graph = graph,
      .changed_count = out_changed_count,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_region(
      module, module->body, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_refine_call_result_types,
                             &walk},
      walk_arena, &walk_result);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_refine_boundaries_run_with_options(
    loom_pass_t* pass, loom_module_t* module,
    const loom_refine_boundaries_options_t* options) {
  loom_refine_boundaries_statistics_t* statistics =
      loom_refine_boundaries_statistics(pass);
  uint32_t max_iterations = options && options->max_iterations > 0
                                ? options->max_iterations
                                : LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS;
  const iree_host_size_t boundary_fact_value_capacity =
      loom_value_table_capacity(&module->values);

  iree_arena_allocator_t facts_arena_a;
  iree_arena_allocator_t facts_arena_b;
  iree_arena_allocator_t iteration_arena;
  iree_arena_allocator_t walk_arena;
  iree_arena_initialize(pass->arena->block_pool, &facts_arena_a);
  iree_arena_initialize(pass->arena->block_pool, &facts_arena_b);
  iree_arena_initialize(pass->arena->block_pool, &iteration_arena);
  iree_arena_initialize(pass->arena->block_pool, &walk_arena);

  iree_arena_allocator_t* current_facts_arena = &facts_arena_a;
  iree_arena_allocator_t* next_facts_arena = &facts_arena_b;
  loom_refine_boundaries_boundary_state_t boundary_state_a = {0};
  loom_refine_boundaries_boundary_state_t boundary_state_b = {0};
  loom_refine_boundaries_boundary_state_t* current_boundary = &boundary_state_a;
  loom_refine_boundaries_boundary_state_t* next_boundary = &boundary_state_b;

  iree_status_t status = loom_refine_boundaries_boundary_state_initialize(
      current_boundary, current_facts_arena, boundary_fact_value_capacity);

  loom_canonicalizer_t canonicalizer = {0};
  bool canonicalizer_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_canonicalizer_initialize(module, pass->arena,
                                           pass->value_facts, &canonicalizer);
    canonicalizer_initialized = iree_status_is_ok(status);
  }

  bool converged = false;
  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
       iteration < max_iterations;
       ++iteration) {
    iree_arena_reset(&iteration_arena);
    iree_arena_reset(next_facts_arena);

    loom_refine_boundaries_graph_t graph = {0};
    loom_scc_list_t sccs = {0};
    status = loom_refine_boundaries_build_graph(module, &iteration_arena,
                                                &walk_arena, &graph, &sccs);
    if (!iree_status_is_ok(status)) {
      break;
    }

    status = loom_refine_boundaries_boundary_state_initialize(
        next_boundary, next_facts_arena, boundary_fact_value_capacity);
    if (!iree_status_is_ok(status)) {
      break;
    }

    int64_t signature_type_changed_count = 0;
    for (iree_host_size_t scc_index = 0;
         iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
         scc_index < sccs.count;
         ++scc_index) {
      const loom_scc_t* scc = &sccs.values[scc_index];
      for (iree_host_size_t member = 0;
           iree_status_is_ok(status) &&
           !loom_pass_has_error_diagnostics(pass) && member < scc->node_count;
           ++member) {
        iree_host_size_t node = scc->nodes[member];
        status = loom_refine_boundaries_run_function(
            pass, &canonicalizer, &graph, &current_boundary->facts,
            &current_boundary->replacements, &next_boundary->facts,
            &next_boundary->replacements, &graph.functions[node],
            &signature_type_changed_count);
      }
    }
    if (!iree_status_is_ok(status) || loom_pass_has_error_diagnostics(pass)) {
      break;
    }

    bool boundary_facts_changed = !loom_refine_boundaries_fact_tables_equal(
        &current_boundary->facts, &next_boundary->facts);
    bool boundary_replacements_changed =
        !loom_refine_boundaries_replacement_tables_equal(
            &current_boundary->replacements, &next_boundary->replacements);
    if (!boundary_facts_changed && !boundary_replacements_changed) {
      int64_t call_result_type_changed_count = 0;
      status = loom_refine_boundaries_refine_internal_signature_types(
          pass, module, &graph, &next_boundary->facts, &walk_arena,
          &call_result_type_changed_count);
      if (!iree_status_is_ok(status) || loom_pass_has_error_diagnostics(pass)) {
        break;
      }
      signature_type_changed_count += call_result_type_changed_count;
      if (signature_type_changed_count > 0) {
        loom_pass_mark_changed(pass);
        statistics->boundary_signature_types_refined +=
            signature_type_changed_count;
        continue;
      }

      int64_t specialization_count = 0;
      status = loom_refine_boundaries_specialize_internal_boundaries(
          module, &graph, &iteration_arena, &walk_arena, &specialization_count);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (specialization_count > 0) {
        loom_pass_mark_changed(pass);
        statistics->boundary_specializations_created += specialization_count;
        continue;
      }

      int64_t pruned_argument_count = 0;
      int64_t pruned_result_count = 0;
      status = loom_refine_boundaries_prune_internal_boundaries(
          module, &graph, &iteration_arena, &walk_arena, &pruned_argument_count,
          &pruned_result_count);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (pruned_argument_count > 0 || pruned_result_count > 0) {
        loom_pass_mark_changed(pass);
        if (pruned_argument_count > 0) {
          statistics->boundary_arguments_pruned += pruned_argument_count;
        }
        if (pruned_result_count > 0) {
          statistics->boundary_results_pruned += pruned_result_count;
        }
        continue;
      }
      converged = true;
      break;
    }
    if (signature_type_changed_count > 0) {
      statistics->boundary_signature_types_refined +=
          signature_type_changed_count;
    }
    if (boundary_facts_changed) {
      ++statistics->boundary_facts_changed;
    }
    if (boundary_replacements_changed) {
      ++statistics->boundary_replacements_changed;
    }

    iree_arena_allocator_t* old_current_arena = current_facts_arena;
    current_facts_arena = next_facts_arena;
    next_facts_arena = old_current_arena;
    loom_refine_boundaries_boundary_state_t* old_current_boundary =
        current_boundary;
    current_boundary = next_boundary;
    next_boundary = old_current_boundary;
  }
  if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
      !converged) {
    status = loom_refine_boundaries_emit_nonconvergence(pass, module,
                                                        max_iterations);
  }

  if (canonicalizer_initialized) {
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  iree_arena_deinitialize(&walk_arena);
  iree_arena_deinitialize(&iteration_arena);
  iree_arena_deinitialize(&facts_arena_b);
  iree_arena_deinitialize(&facts_arena_a);
  return status;
}

iree_status_t loom_refine_boundaries_run(loom_pass_t* pass,
                                         loom_module_t* module) {
  return loom_refine_boundaries_run_with_options(
      pass, module, (const loom_refine_boundaries_options_t*)pass->state);
}
