// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/plan_materializer.h"

#include <string.h>

#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
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
    const loom_symbol_ref_t* module_target_symbols,
    loom_symbol_ref_t* target_symbols,
    loom_symbol_ref_t* target_template_families) {
  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    const loom_link_module_index_symbol_t* source_symbol =
        selection->symbols.values[i].source_symbol;
    target_symbols[source_symbol->ordinal] = module_target_symbols[i];
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
      IREE_ASSERT(target_family->module_id ==
                  module_target_symbols[i].module_id);
      IREE_ASSERT(target_family->symbol_id ==
                  module_target_symbols[i].symbol_id);
    } else {
      *target_family = module_target_symbols[i];
    }
  }
}

static iree_status_t loom_link_plan_materialize_module(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_selection_t* selection,
    loom_linker_source_provider_import_list_t provider_imports,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t* source_symbol_ordinals,
    loom_linker_symbol_output_t* source_symbol_outputs,
    loom_symbol_ref_t* module_target_symbols, loom_symbol_ref_t* target_symbols,
    loom_symbol_ref_t* target_template_families, loom_linker_t* linker) {
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
  const loom_linker_target_symbol_list_t out_target_symbols = {
      .count = selection->symbols.count,
      .values = module_target_symbols,
  };
  const loom_linker_source_symbol_output_list_t source_outputs = {
      .count = source_symbol_outputs ? selection->symbols.count : 0,
      .values = source_symbol_outputs,
  };

  const loom_link_module_index_module_t* module = selection->source_module;
  const bool complete_module = selection->symbols.count == module->symbol_count;
  iree_status_t status = iree_ok_status();
  if (module->materialized_module != NULL) {
    if (complete_module) {
      status = loom_linker_add_exact_module(linker, module->materialized_module,
                                            source_outputs, provider_imports,
                                            out_target_symbols);
    } else {
      status = loom_linker_add_module_symbols(
          linker, module->materialized_module, source_symbols, source_outputs,
          provider_imports, out_target_symbols);
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
                                            source_outputs, provider_imports,
                                            out_target_symbols);
    }
    loom_module_free(materialized_module);
  }
  if (iree_status_is_ok(status)) {
    loom_link_plan_materialization_scatter_target_symbols(
        index, selection, module_target_symbols, target_symbols,
        target_template_families);
  }
  return status;
}

static iree_status_t loom_link_plan_materialize_modules(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_projection_t* projection,
    const loom_link_plan_linker_import_projection_t* provider_imports,
    const loom_link_plan_materialization_environment_t* environment,
    bool sealed_output, iree_arena_allocator_t* arena,
    loom_symbol_ref_t* target_symbols,
    loom_symbol_ref_t* target_template_families, loom_linker_t* linker) {
  iree_host_size_t max_module_symbol_count = 0;
  for (iree_host_size_t i = 0; i < projection->modules.count; ++i) {
    max_module_symbol_count = iree_max(
        max_module_symbol_count, projection->modules.values[i].symbols.count);
  }
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
        index, &projection->modules.values[i],
        provider_imports->modules.values[i], environment,
        source_symbol_ordinals, source_symbol_outputs, module_target_symbols,
        target_symbols, target_template_families, linker);
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
  loom_symbol_ref_t* target_symbols = NULL;
  if (index_symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, index_symbol_count,
                                                   sizeof(*target_symbols),
                                                   (void**)&target_symbols));
    for (iree_host_size_t i = 0; i < index_symbol_count; ++i) {
      target_symbols[i] = loom_symbol_ref_null();
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

  loom_link_plan_module_projection_t module_projection = {0};
  loom_link_plan_linker_import_projection_t provider_import_projection = {0};
  IREE_RETURN_IF_ERROR(
      loom_link_plan_project_modules(plan, arena, &module_projection));
  IREE_RETURN_IF_ERROR(loom_link_plan_project_linker_imports(
      index, &module_projection, arena, &provider_import_projection));

  const loom_linker_options_t linker_options = {
      .module_name = module_name,
      .planned_symbol_capacity =
          loom_link_plan_mode(plan) == LOOM_LINK_PLAN_SELECTIVE
              ? loom_link_plan_symbol_count(plan)
              : 0,
      .provider_imports =
          {
              .count = provider_import_projection.provider_imports.count,
              .anchor_count =
                  provider_import_projection.provider_import_anchors.count,
          },
  };
  loom_linker_t* linker = NULL;
  IREE_RETURN_IF_ERROR(loom_linker_allocate(
      environment->context, &linker_options, environment->block_pool,
      environment->allocator, &linker));

  loom_module_t* output_module = NULL;
  iree_status_t status = loom_link_plan_materialize_modules(
      index, &module_projection, &provider_import_projection, environment,
      loom_link_plan_mode(plan) == LOOM_LINK_PLAN_SELECTIVE, arena,
      target_symbols, target_template_families, linker);
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
  return iree_ok_status();
}
