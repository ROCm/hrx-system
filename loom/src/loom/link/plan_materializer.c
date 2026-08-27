// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/plan_materializer.h"

#include <string.h>

#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/link/kernel_config_materializer.h"
#include "loom/link/linker.h"
#include "loom/link/plan_projection.h"

static loom_diagnostic_sink_t loom_link_plan_materialization_diagnostic_sink(
    const loom_link_plan_materialization_environment_t* environment,
    const loom_link_module_index_provider_t* provider) {
  if (environment->diagnostic_sink == NULL) {
    return (loom_diagnostic_sink_t){0};
  }
  return environment->diagnostic_sink(environment->user_data, provider);
}

static void loom_link_plan_materialization_read_options_initialize(
    const loom_link_plan_materialization_environment_t* environment,
    const loom_link_module_index_provider_t* provider,
    loom_bytecode_read_options_t* out_options) {
  *out_options = (loom_bytecode_read_options_t){
      .diagnostic_sink =
          loom_link_plan_materialization_diagnostic_sink(environment, provider),
      .low_repr_environment = environment->low_repr_environment,
  };
}

static iree_status_t loom_link_plan_materialize_bytecode_module(
    const loom_link_module_index_provider_t* provider,
    const loom_link_module_index_module_t* module,
    loom_bytecode_symbol_ordinal_list_t selection,
    const loom_link_plan_materialization_environment_t* environment,
    bool complete_module, loom_module_t** out_module) {
  *out_module = NULL;
  loom_bytecode_read_options_t read_options;
  loom_link_plan_materialization_read_options_initialize(environment, provider,
                                                         &read_options);
  loom_bytecode_read_result_t read_result = {0};
  loom_module_t* materialized_module = NULL;
  iree_status_t status = iree_ok_status();
  if (complete_module) {
    status = loom_bytecode_read_module_ordinal(
        provider->bytecode.contents, provider->bytecode.filename,
        environment->context, environment->block_pool,
        (uint16_t)module->provider_module_ordinal, &read_options, &read_result,
        &materialized_module, environment->allocator);
  } else {
    status = loom_bytecode_materialize_module_symbols(
        provider->bytecode.contents, provider->bytecode.filename,
        environment->context, environment->block_pool,
        &provider->bytecode.metadata, (uint16_t)module->provider_module_ordinal,
        selection, &read_options, &read_result, &materialized_module,
        environment->allocator);
  }
  if (iree_status_is_ok(status) &&
      (read_result.error_count != 0 || materialized_module == NULL)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bytecode provider '%.*s' module %" PRIhsz
                              " did not materialize",
                              (int)provider->name.size, provider->name.data,
                              module->provider_module_ordinal);
  }
  if (!iree_status_is_ok(status)) {
    loom_module_free(materialized_module);
    return status;
  }
  *out_module = materialized_module;
  return iree_ok_status();
}

static void loom_link_plan_materialization_scatter_target_symbols(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_selection_t* selection,
    const loom_symbol_ref_t* module_target_symbols, bool projected_module,
    loom_symbol_ref_t* target_symbols,
    loom_symbol_ref_t* target_template_families,
    const loom_symbol_ref_t* source_configuration_functions,
    loom_symbol_ref_t* target_kernel_configurations) {
  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    const loom_link_module_index_symbol_t* source_symbol =
        selection->symbols.values[i].source_symbol;
    const loom_symbol_ref_t target_symbol =
        module_target_symbols[projected_module
                                  ? selection->symbols.values[i]
                                        .materialized_symbol_ordinal
                                  : i];
    target_symbols[source_symbol->ordinal] = target_symbol;
    if (source_configuration_functions &&
        loom_symbol_ref_is_valid(source_configuration_functions[i])) {
      IREE_ASSERT(target_kernel_configurations != NULL);
      loom_symbol_ref_t* target_configuration =
          &target_kernel_configurations[target_symbol.symbol_id];
      const loom_symbol_ref_t source_configuration =
          module_target_symbols[source_configuration_functions[i].symbol_id];
      IREE_ASSERT(
          !loom_symbol_ref_is_valid(*target_configuration) ||
          (target_configuration->module_id == source_configuration.module_id &&
           target_configuration->symbol_id == source_configuration.symbol_id));
      *target_configuration = source_configuration;
    }
    if (source_symbol->kind != LOOM_SYMBOL_TEMPLATE_DECL) {
      continue;
    }
    IREE_ASSERT(source_symbol->template_family_ordinal !=
                LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID);
    IREE_ASSERT(source_symbol->template_family_ordinal <
                loom_link_module_index_template_family_count(index));
    loom_symbol_ref_t* target_family =
        &target_template_families[source_symbol->template_family_ordinal];
    if (loom_symbol_ref_is_valid(*target_family)) {
      IREE_ASSERT(target_family->module_id == target_symbol.module_id);
      IREE_ASSERT(target_family->symbol_id == target_symbol.symbol_id);
    } else {
      *target_family = target_symbol;
    }
  }
}

static iree_status_t loom_link_plan_materialize_module(
    const loom_link_plan_t* plan, const loom_link_module_index_t* index,
    const loom_link_plan_module_selection_t* selection,
    const loom_link_plan_materialization_environment_t* environment,
    iree_arena_allocator_t* arena, iree_host_size_t* source_symbol_ordinals,
    loom_linker_symbol_output_t* source_symbol_outputs,
    loom_symbol_ref_t* module_target_symbols, loom_symbol_ref_t* target_symbols,
    loom_symbol_ref_t* target_template_families,
    loom_symbol_ref_t* target_kernel_configurations, loom_linker_t* linker) {
  const loom_link_module_index_module_t* module = selection->source_module;
  if (loom_link_plan_module_requires_symbol_projection(selection)) {
    loom_link_kernel_config_module_projection_t projected = {0};
    IREE_RETURN_IF_ERROR(loom_link_plan_project_kernel_config_module(
        plan, selection, environment, module->name, arena, &projected));
    const iree_host_size_t projected_symbol_count =
        projected.module->symbols.count;
    for (iree_host_size_t i = 0; i < projected_symbol_count; ++i) {
      module_target_symbols[i] = loom_symbol_ref_null();
      if (source_symbol_outputs) {
        source_symbol_outputs[i] = LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY;
      }
    }
    if (source_symbol_outputs) {
      for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
        source_symbol_outputs[selection->symbols.values[i]
                                  .materialized_symbol_ordinal] =
            selection->symbols.values[i].plan_symbol->reason ==
                    LOOM_LINK_PLAN_LIVE_ROOT
                ? LOOM_LINKER_SYMBOL_OUTPUT_ROOT
                : LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY;
      }
    }
    const loom_linker_source_symbol_output_list_t source_outputs = {
        .count = source_symbol_outputs ? projected_symbol_count : 0,
        .values = source_symbol_outputs,
    };
    const loom_linker_target_symbol_list_t out_target_symbols = {
        .count = projected_symbol_count,
        .values = module_target_symbols,
    };
    iree_status_t status = loom_linker_add_exact_module(
        linker, projected.module, source_outputs, out_target_symbols);
    if (iree_status_is_ok(status)) {
      loom_link_plan_materialization_scatter_target_symbols(
          index, selection, module_target_symbols,
          /*projected_module=*/true, target_symbols, target_template_families,
          projected.configuration_functions.values,
          target_kernel_configurations);
    }
    loom_module_free(projected.module);
    return status;
  }

  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    source_symbol_ordinals[i] =
        selection->symbols.values[i].source_symbol->module_symbol_ordinal;
    if (source_symbol_outputs) {
      source_symbol_outputs[i] =
          selection->symbols.values[i].plan_symbol->reason ==
                  LOOM_LINK_PLAN_LIVE_ROOT
              ? LOOM_LINKER_SYMBOL_OUTPUT_ROOT
              : LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY;
    }
    module_target_symbols[i] = loom_symbol_ref_null();
  }
  const loom_linker_source_symbol_list_t source_symbols = {
      .count = selection->symbols.count,
      .ordinals = source_symbol_ordinals,
  };
  loom_linker_target_symbol_list_t out_target_symbols = {
      .count = selection->symbols.count,
      .values = module_target_symbols,
  };
  loom_linker_source_symbol_output_list_t source_outputs = {
      .count = source_symbol_outputs ? selection->symbols.count : 0,
      .values = source_symbol_outputs,
  };

  const bool complete_module = selection->symbols.count == module->symbol_count;
  iree_status_t status = iree_ok_status();
  if (module->materialized_module != NULL) {
    if (complete_module) {
      status = loom_linker_add_exact_module(linker, module->materialized_module,
                                            source_outputs, out_target_symbols);
    } else {
      status = loom_linker_add_module_symbols(
          linker, module->materialized_module, source_symbols,
          loom_linker_source_symbol_binding_list_empty(), source_outputs,
          out_target_symbols);
    }
  } else {
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(index, module->provider_ordinal);
    IREE_ASSERT(provider->kind == LOOM_LINK_PROVIDER_BYTECODE);
    loom_module_t* materialized_module = NULL;
    status = loom_link_plan_materialize_bytecode_module(
        provider, module,
        (loom_bytecode_symbol_ordinal_list_t){
            .count = source_symbols.count,
            .ordinals = source_symbols.ordinals,
        },
        environment, complete_module, &materialized_module);
    if (iree_status_is_ok(status)) {
      status = loom_linker_add_exact_module(linker, materialized_module,
                                            source_outputs, out_target_symbols);
    }
    loom_module_free(materialized_module);
  }
  if (iree_status_is_ok(status)) {
    loom_link_plan_materialization_scatter_target_symbols(
        index, selection, module_target_symbols,
        /*projected_module=*/false, target_symbols, target_template_families,
        /*source_configuration_functions=*/NULL, target_kernel_configurations);
  }
  return status;
}

// Recovers provider-level context after the canonical linker rejects two
// selected concrete definitions. Collision recovery is deliberately confined
// to the failing path: valid plans rely only on the target module's symbol map
// and never maintain a second per-plan name-membership structure.
static iree_status_t loom_link_plan_annotate_global_collision(
    iree_status_t status, const loom_link_plan_t* plan,
    const loom_link_module_index_t* index,
    const loom_link_plan_module_selection_t* selection) {
  if (iree_status_code(status) != IREE_STATUS_ALREADY_EXISTS) {
    return status;
  }

  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        selection->symbols.values[i].source_symbol;
    if (symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_GLOBAL ||
        !iree_any_bit_set(symbol->flags,
                          LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION)) {
      continue;
    }

    const loom_link_module_index_symbol_t* previous =
        loom_link_module_index_lookup_name(index, symbol->name);
    while (previous) {
      const bool is_prior_selected_definition =
          previous->module_ordinal < symbol->module_ordinal &&
          previous->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL &&
          iree_any_bit_set(previous->flags,
                           LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION) &&
          loom_link_plan_contains_symbol(plan, previous->ordinal);
      if (is_prior_selected_definition) {
        return loom_link_module_index_annotate_global_collision(
            status, index, previous, symbol);
      }
      previous = loom_link_module_index_next_same_name(index, previous);
    }
  }
  return status;
}

static iree_status_t loom_link_plan_materialize_modules(
    const loom_link_plan_t* plan, const loom_link_module_index_t* index,
    const loom_link_plan_module_projection_t* projection,
    const loom_link_plan_materialization_environment_t* environment,
    bool sealed_output, iree_arena_allocator_t* arena,
    loom_symbol_ref_t* target_symbols,
    loom_symbol_ref_t* target_template_families,
    loom_symbol_ref_t* target_kernel_configurations, loom_linker_t* linker) {
  const iree_host_size_t max_module_symbol_count =
      projection->maximum_materialized_symbol_count;
  iree_host_size_t* source_symbol_ordinals = NULL;
  loom_linker_symbol_output_t* source_symbol_outputs = NULL;
  loom_symbol_ref_t* module_target_symbols = NULL;
  if (max_module_symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, max_module_symbol_count, sizeof(*source_symbol_ordinals),
        (void**)&source_symbol_ordinals));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, max_module_symbol_count, sizeof(*module_target_symbols),
        (void**)&module_target_symbols));
    if (sealed_output) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, max_module_symbol_count, sizeof(*source_symbol_outputs),
          (void**)&source_symbol_outputs));
    }
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < projection->modules.count && iree_status_is_ok(status); ++i) {
    status = loom_link_plan_materialize_module(
        plan, index, &projection->modules.values[i], environment, arena,
        source_symbol_ordinals, source_symbol_outputs, module_target_symbols,
        target_symbols, target_template_families, target_kernel_configurations,
        linker);
    if (!iree_status_is_ok(status)) {
      status = loom_link_plan_annotate_global_collision(
          status, plan, index, &projection->modules.values[i]);
    }
  }
  return status;
}

iree_status_t loom_link_plan_materialize(
    const loom_link_plan_t* plan,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_arena_allocator_t* arena,
    loom_link_plan_materialization_t* out_materialization) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_materialization);
  *out_materialization = (loom_link_plan_materialization_t){0};

  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const iree_host_size_t index_symbol_count =
      loom_link_module_index_symbol_count(index);
  loom_link_plan_module_projection_t module_projection = {0};
  IREE_RETURN_IF_ERROR(
      loom_link_plan_project_modules(plan, arena, &module_projection));

  loom_symbol_ref_t* target_symbols = NULL;
  loom_symbol_ref_t* target_kernel_configurations = NULL;
  iree_host_size_t planned_symbol_capacity = 0;
  if (loom_link_plan_mode(plan) == LOOM_LINK_PLAN_SELECTIVE &&
      !iree_host_size_checked_add(loom_link_plan_symbol_count(plan),
                                  module_projection.synthetic_symbol_count,
                                  &planned_symbol_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "planned symbol capacity overflow");
  }
  if (index_symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, index_symbol_count,
                                                   sizeof(*target_symbols),
                                                   (void**)&target_symbols));
    for (iree_host_size_t i = 0; i < index_symbol_count; ++i) {
      target_symbols[i] = loom_symbol_ref_null();
    }
    if (module_projection.synthetic_symbol_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, planned_symbol_capacity, sizeof(*target_kernel_configurations),
          (void**)&target_kernel_configurations));
      for (iree_host_size_t i = 0; i < planned_symbol_capacity; ++i) {
        target_kernel_configurations[i] = loom_symbol_ref_null();
      }
    }
  }
  const iree_host_size_t template_family_count =
      loom_link_module_index_template_family_count(index);
  loom_symbol_ref_t* target_template_families = NULL;
  if (template_family_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, template_family_count, sizeof(*target_template_families),
        (void**)&target_template_families));
    for (iree_host_size_t i = 0; i < template_family_count; ++i) {
      target_template_families[i] = loom_symbol_ref_null();
    }
  }

  const loom_linker_options_t linker_options = {
      .module_name = module_name,
      .planned_symbol_capacity = planned_symbol_capacity,
  };
  loom_linker_t* linker = NULL;
  IREE_RETURN_IF_ERROR(loom_linker_allocate(
      environment->context, &linker_options, environment->block_pool,
      environment->allocator, &linker));

  loom_module_t* output_module = NULL;
  iree_status_t status = loom_link_plan_materialize_modules(
      plan, index, &module_projection, environment,
      loom_link_plan_mode(plan) == LOOM_LINK_PLAN_SELECTIVE, arena,
      target_symbols, target_template_families, target_kernel_configurations,
      linker);
  if (iree_status_is_ok(status)) {
    status = loom_linker_finish(linker, &output_module);
  }
  if (iree_status_is_ok(status) && environment->prepare_module != NULL) {
    status = environment->prepare_module(environment->user_data, output_module);
  }
  loom_linker_free(linker);

  if (!iree_status_is_ok(status)) {
    loom_module_free(output_module);
    return status;
  }
  out_materialization->module = output_module;
  out_materialization->target_symbols.values = target_symbols;
  out_materialization->target_symbols.count = index_symbol_count;
  out_materialization->target_template_families.values =
      target_template_families;
  out_materialization->target_template_families.count = template_family_count;
  out_materialization->target_kernel_configurations.values =
      target_kernel_configurations;
  out_materialization->target_kernel_configurations.count =
      target_kernel_configurations ? output_module->symbols.count : 0;
  return iree_ok_status();
}
