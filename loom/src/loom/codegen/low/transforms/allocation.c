// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/allocation.h"

#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_live_range_splitting.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/rematerialization.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/target/function_version.h"

typedef struct loom_low_materialize_allocation_pass_state_t {
  // Fixed register budget overrides parsed from the pass options.
  loom_low_allocation_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // True when the pass should emit spill materialization diagnostics.
  bool emit_spill_diagnostics;
  // True once the diagnostics option has been parsed.
  bool has_diagnostics_option;
  // True when the pass received a complete target spill-storage capability and
  // should reject generated storage outside |spill_storage_spaces|.
  bool has_spill_storage_spaces;
  // Storage spaces allowed for newly generated spill storage.
  loom_low_storage_space_set_t spill_storage_spaces;
} loom_low_materialize_allocation_pass_state_t;

typedef struct loom_low_materialize_allocation_parse_context_t {
  // Pass instance that owns option storage.
  loom_pass_t* pass;
  // Mutable pass state being populated.
  loom_low_materialize_allocation_pass_state_t* state;
} loom_low_materialize_allocation_parse_context_t;

static iree_status_t loom_low_materialize_allocation_emit_rematerialization(
    loom_pass_t* pass,
    const loom_low_materialize_allocation_pass_state_t* state,
    const loom_low_allocation_table_t* table,
    loom_low_allocation_rematerialization_trigger_t trigger,
    const loom_low_allocation_rematerialization_result_t* result) {
  if (!state || !state->emit_spill_diagnostics) {
    return iree_ok_status();
  }
  return loom_low_allocation_rematerialization_emit_decision(
      table, trigger, result, pass->diagnostic_emitter);
}

static iree_status_t loom_low_materialize_allocation_emit_live_range_split(
    loom_pass_t* pass,
    const loom_low_materialize_allocation_pass_state_t* state,
    const loom_low_allocation_table_t* table,
    loom_low_allocation_live_range_split_trigger_t trigger,
    const loom_low_allocation_live_range_split_result_t* result) {
  if (!state || !state->emit_spill_diagnostics) {
    return iree_ok_status();
  }
  return loom_low_allocation_live_range_split_emit_decision(
      table, trigger, result, pass->diagnostic_emitter);
}

static const loom_pass_option_def_t kLowMaterializeAllocationOptions[] = {
    {IREE_SVL("budgets"),
     IREE_SVL("Semicolon-separated register class budgets, such as "
              "test.i32=2;x86.zmm=1.")},
    {IREE_SVL("diagnostics"),
     IREE_SVL("Diagnostic feedback to emit: none or spills.")},
    {IREE_SVL("spill-storage-spaces"),
     IREE_SVL("Semicolon-separated storage spaces supported by the target "
              "spill lowering path, or all/none.")},
};

#define LOOM_LOW_MATERIALIZE_ALLOCATION_STATISTICS(V, statistics_type)         \
  V(statistics_type, storage, "storage",                                       \
    "Number of low.storage.reserve ops inserted.")                             \
  V(statistics_type, spills, "spills", "Number of low.spill stores inserted.") \
  V(statistics_type, reloads, "reloads", "Number of low.reload ops inserted.") \
  V(statistics_type, rematerializations, "rematerializations",                 \
    "Number of allocation-pressure rematerialized packets inserted.")          \
  V(statistics_type, live_range_splits, "live_range_splits",                   \
    "Number of allocation-pressure low.copy split packets inserted.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_materialize_allocation_statistics,
                            loom_low_materialize_allocation_statistics_t,
                            LOOM_LOW_MATERIALIZE_ALLOCATION_STATISTICS)

static const loom_pass_info_t
    loom_low_materialize_allocation_pass_info_storage = {
        .name = IREE_SVL("low-materialize-allocation"),
        .description =
            IREE_SVL("Allocate target-low values and materialize spills."),
        .kind = LOOM_PASS_FUNCTION,
        .option_defs = kLowMaterializeAllocationOptions,
        .option_count = IREE_ARRAYSIZE(kLowMaterializeAllocationOptions),
        .statistic_layout = &loom_low_materialize_allocation_statistics_layout,
};

const loom_pass_info_t* loom_low_materialize_allocation_pass_info(void) {
  return &loom_low_materialize_allocation_pass_info_storage;
}

static iree_status_t loom_low_materialize_allocation_count_budgets(
    iree_string_view_t text, iree_host_size_t* out_count) {
  iree_host_size_t count = 0;
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ';', &token, &remaining);
    token = iree_string_view_trim(token);
    if (iree_string_view_is_empty(token)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass 'low-materialize-allocation' option 'budgets' contains an "
          "empty budget entry");
    }
    ++count;
    text = iree_string_view_trim(remaining);
  }
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_budget(
    iree_string_view_t token, loom_low_allocation_budget_t* out_budget) {
  iree_string_view_t register_class = iree_string_view_empty();
  iree_string_view_t budget_text = iree_string_view_empty();
  iree_string_view_split(token, '=', &register_class, &budget_text);
  register_class = iree_string_view_trim(register_class);
  budget_text = iree_string_view_trim(budget_text);
  if (iree_string_view_is_empty(register_class) ||
      iree_string_view_is_empty(budget_text)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'budgets' entries must have "
        "the form <register-class>=<units>");
  }
  uint32_t max_units = 0;
  if (!iree_string_view_atoi_uint32(budget_text, &max_units)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' budget expected a uint32 value, "
        "got '%.*s'",
        (int)budget_text.size, budget_text.data);
  }
  *out_budget = (loom_low_allocation_budget_t){
      .register_class = register_class,
      .max_units = max_units,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_budgets(
    iree_string_view_t text,
    loom_low_materialize_allocation_parse_context_t* context) {
  if (context->state->budgets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'budgets' for pass 'low-materialize-allocation'");
  }

  iree_host_size_t budget_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_materialize_allocation_count_budgets(text, &budget_count));
  if (budget_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'budgets' must not be empty");
  }

  loom_low_allocation_budget_t* budgets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->pass->instance_arena,
                                                 budget_count, sizeof(*budgets),
                                                 (void**)&budgets));
  memset(budgets, 0, budget_count * sizeof(*budgets));

  iree_host_size_t index = 0;
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ';', &token, &remaining);
    token = iree_string_view_trim(token);
    IREE_RETURN_IF_ERROR(
        loom_low_materialize_allocation_parse_budget(token, &budgets[index++]));
    text = iree_string_view_trim(remaining);
  }

  context->state->budgets = budgets;
  context->state->budget_count = budget_count;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_diagnostics(
    iree_string_view_t text,
    loom_low_materialize_allocation_parse_context_t* context) {
  if (context->state->has_diagnostics_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'diagnostics' for pass 'low-materialize-allocation'");
  }
  text = iree_string_view_trim(text);
  if (iree_string_view_equal(text, IREE_SV("none"))) {
    context->state->emit_spill_diagnostics = false;
  } else if (iree_string_view_equal(text, IREE_SV("spills"))) {
    context->state->emit_spill_diagnostics = true;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'diagnostics' expected "
        "'none' or 'spills', got '%.*s'",
        (int)text.size, text.data);
  }
  context->state->has_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_storage_space(
    iree_string_view_t token,
    loom_low_materialize_allocation_parse_context_t* context) {
  loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_COUNT_;
  if (!loom_low_storage_type_space_parse(token, &storage_space)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'spill-storage-spaces' "
        "expected storage space 'stack', 'scratch', 'private', or "
        "'workgroup', got '%.*s'",
        (int)token.size, token.data);
  }
  const loom_low_storage_space_set_t storage_space_set =
      loom_low_storage_space_set_for(storage_space);
  if (iree_any_bit_set(context->state->spill_storage_spaces,
                       storage_space_set)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'spill-storage-spaces' "
        "contains duplicate storage space '%.*s'",
        (int)token.size, token.data);
  }
  context->state->spill_storage_spaces |= storage_space_set;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_spill_storage_spaces(
    iree_string_view_t text,
    loom_low_materialize_allocation_parse_context_t* context) {
  if (context->state->has_spill_storage_spaces) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'spill-storage-spaces' for pass "
                            "'low-materialize-allocation'");
  }
  text = iree_string_view_trim(text);
  context->state->has_spill_storage_spaces = true;
  context->state->spill_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_NONE;
  if (iree_string_view_is_empty(text)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'spill-storage-spaces' "
        "must be all, none, or a semicolon-separated storage space list");
  }
  if (iree_string_view_equal(text, IREE_SV("all"))) {
    context->state->spill_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_ALL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(text, IREE_SV("none"))) {
    return iree_ok_status();
  }
  if (text.data[text.size - 1] == ';') {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'spill-storage-spaces' "
        "contains an empty storage space entry");
  }
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ';', &token, &remaining);
    token = iree_string_view_trim(token);
    if (iree_string_view_is_empty(token)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass 'low-materialize-allocation' option 'spill-storage-spaces' "
          "contains an empty storage space entry");
    }
    IREE_RETURN_IF_ERROR(
        loom_low_materialize_allocation_parse_storage_space(token, context));
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_materialize_allocation_parse_context_t* context =
      (loom_low_materialize_allocation_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("budgets"))) {
    return loom_low_materialize_allocation_parse_budgets(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return loom_low_materialize_allocation_parse_diagnostics(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("spill-storage-spaces"))) {
    return loom_low_materialize_allocation_parse_spill_storage_spaces(value,
                                                                      context);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown option '%.*s' for pass 'low-materialize-allocation'",
      (int)name.size, name.data);
}

iree_status_t loom_low_materialize_allocation_create(
    loom_pass_t* pass, iree_string_view_t options) {
  loom_low_materialize_allocation_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));

  loom_low_materialize_allocation_parse_context_t context = {
      .pass = pass,
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("budgets"))) {
        IREE_RETURN_IF_ERROR(loom_low_materialize_allocation_parse_budgets(
            option->string_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("diagnostics"))) {
        iree_string_view_t value =
            option->schema->enum_values[option->enum_value_index].value;
        IREE_RETURN_IF_ERROR(
            loom_low_materialize_allocation_parse_diagnostics(value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("spill-storage-spaces"))) {
        IREE_RETURN_IF_ERROR(
            loom_low_materialize_allocation_parse_spill_storage_spaces(
                option->string_value, &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'low-materialize-allocation'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_options_parse(
        pass->info->name, options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_low_materialize_allocation_parse_option,
            .user_data = &context,
        }));
  }
  pass->state = state;
  return iree_ok_status();
}

// Builds one allocation table while the corresponding immutable function
// model owns the module value-ordinal scratch map.
static iree_status_t loom_low_materialize_allocation_build_table(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table) {
  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      module, low_func_op, function_target_facts, descriptor_registry,
      options->emitter, LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE, arena,
      &model);
  if (iree_status_is_ok(status)) {
    status = loom_low_allocate_function(&model, options, arena, out_table);
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

iree_status_t loom_low_materialize_allocation_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op)) {
    return iree_ok_status();
  }

  loom_low_materialize_allocation_pass_state_t* state =
      (loom_low_materialize_allocation_pass_state_t*)pass->state;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_target_facts_t* function_target_facts =
      loom_target_function_version_target_facts(pass->function_version);
  loom_low_allocation_options_t allocation_options = {
      .budgets = state ? state->budgets : NULL,
      .budget_count = state ? state->budget_count : 0,
      .emitter = pass->diagnostic_emitter,
  };
  loom_low_allocation_options_t allocation_probe_options = allocation_options;
  allocation_probe_options.emitter = (iree_diagnostic_emitter_t){0};
  iree_host_size_t iteration_count = 0;
  iree_host_size_t iteration_limit = 0;
  iree_host_size_t rematerialization_iteration_count = 0;
  iree_host_size_t rematerialization_iteration_limit = 0;
  for (;;) {
    loom_low_allocation_table_t table = {0};
    IREE_RETURN_IF_ERROR(loom_low_materialize_allocation_build_table(
        module, function.op, function_target_facts, descriptor_registry,
        &allocation_probe_options, pass->arena, &table));
    if (iteration_limit == 0) {
      if (table.liveness.value_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low allocation materialization iteration limit overflows host "
            "size");
      }
      iteration_limit = table.liveness.value_count + 1;
    }
    if (table.error_count != 0) {
      if (rematerialization_iteration_limit == 0) {
        if (table.liveness.value_count == IREE_HOST_SIZE_MAX) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low allocation rematerialization iteration limit overflows "
              "host size");
        }
        rematerialization_iteration_limit = table.liveness.value_count + 1;
      }
      if (rematerialization_iteration_count >=
          rematerialization_iteration_limit) {
        // Rematerialization is an optimization; preserve normal diagnostics
        // when pressure repair does not converge.
        IREE_RETURN_IF_ERROR(loom_low_materialize_allocation_build_table(
            module, function.op, function_target_facts, descriptor_registry,
            &allocation_options, pass->arena, &table));
        return iree_ok_status();
      }
      loom_low_allocation_rematerialization_result_t result = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_failure(
          module, &table, pass->arena, &result));
      if (result.value.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_materialize_allocation_emit_rematerialization(
                pass, state, &table,
                LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_ALLOCATION_FAILURE,
                &result));
        loom_low_materialize_allocation_statistics_t* statistics =
            loom_low_materialize_allocation_statistics(pass);
        statistics->rematerializations +=
            (int64_t)result.value.cloned_packet_count;
        loom_pass_mark_changed(pass);
        ++rematerialization_iteration_count;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_materialize_allocation_build_table(
          module, function.op, function_target_facts, descriptor_registry,
          &allocation_options, pass->arena, &table));
      return iree_ok_status();
    }
    if (table.spill_plan_count == 0) {
      return iree_ok_status();
    }
    if (rematerialization_iteration_limit == 0) {
      if (table.liveness.value_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low allocation rematerialization iteration limit overflows "
            "host size");
      }
      rematerialization_iteration_limit = table.liveness.value_count + 1;
    }
    if (rematerialization_iteration_count < rematerialization_iteration_limit) {
      loom_low_allocation_rematerialization_result_t rematerialization_result =
          {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_spill_plan(
          module, &table, pass->arena, &rematerialization_result));
      if (rematerialization_result.value.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_materialize_allocation_emit_rematerialization(
                pass, state, &table,
                LOOM_LOW_ALLOCATION_REMATERIALIZATION_TRIGGER_SPILL_PLAN,
                &rematerialization_result));
        loom_low_materialize_allocation_statistics_t* statistics =
            loom_low_materialize_allocation_statistics(pass);
        statistics->rematerializations +=
            (int64_t)rematerialization_result.value.cloned_packet_count;
        loom_pass_mark_changed(pass);
        ++rematerialization_iteration_count;
        continue;
      }
      loom_low_allocation_live_range_split_result_t split_result = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_split_fixed_value_spill_plan(
          module, &table, pass->arena, &split_result));
      if (split_result.rewritten_operand_count != 0) {
        IREE_RETURN_IF_ERROR(
            loom_low_materialize_allocation_emit_live_range_split(
                pass, state, &table,
                LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN,
                &split_result));
        loom_low_materialize_allocation_statistics_t* statistics =
            loom_low_materialize_allocation_statistics(pass);
        statistics->live_range_splits +=
            (int64_t)split_result.transfer_packet_count;
        loom_pass_mark_changed(pass);
        ++rematerialization_iteration_count;
        continue;
      }
    }
    if (iteration_count >= iteration_limit) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation materialization did not reach a spill-free fixed "
          "point after %zu iteration(s)",
          iteration_count);
    }

    loom_low_allocation_materialization_result_t result = {0};
    loom_low_allocation_materialization_options_t materialization_options = {
        .has_supported_storage_spaces =
            state && state->has_spill_storage_spaces,
        .supported_storage_spaces = state ? state->spill_storage_spaces
                                          : LOOM_LOW_STORAGE_SPACE_SET_ALL,
        .emit_spill_diagnostics = state && state->emit_spill_diagnostics,
        .max_spill_plan_count = 1,
        .emitter = pass->diagnostic_emitter,
    };
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_spills(
        &table, &materialization_options, pass->arena, &result));
    if (result.error_count != 0) {
      return iree_ok_status();
    }
    if (result.storage_count == 0 && result.spill_count == 0 &&
        result.reload_count == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation materialization made no progress with %zu pending "
          "spill plan(s)",
          table.spill_plan_count);
    }

    loom_low_materialize_allocation_statistics_t* statistics =
        loom_low_materialize_allocation_statistics(pass);
    statistics->storage += (int64_t)result.storage_count;
    statistics->spills += (int64_t)result.spill_count;
    statistics->reloads += (int64_t)result.reload_count;
    loom_pass_mark_changed(pass);
    ++iteration_count;
  }
}
