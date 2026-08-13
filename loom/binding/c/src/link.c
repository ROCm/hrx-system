// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/link.h"

#include <string.h>

#include "config.h"
#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "link_index.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/link/linker.h"
#include "loom/link/module_index.h"
#include "loom/link/plan_projection.h"
#include "loom/link/planner.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "source.h"
#include "target.h"
#include "workspace.h"

enum {
  LOOMC_LINK_KNOWN_FLAGS = LOOMC_LINK_FLAG_INCLUDE_EXPORTED_ROOTS |
                           LOOMC_LINK_FLAG_ALLOW_UNRESOLVED_SYMBOLS |
                           LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS,
};

struct loomc_linker_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for linker-owned storage.
  loomc_allocator_t allocator;
  // Context retained by the prepared linker.
  loomc_context_t* context;
  // Copied default output module name.
  loomc_string_view_t module_name;
};

typedef struct loomc_link_materialization_context_t {
  // Prepared linker driving this invocation.
  loomc_linker_t* linker;
  // Frozen public index being linked.
  loomc_link_index_t* link_index;
  // Internal frozen module index.
  const loom_link_module_index_t* module_index;
  // Result receiving materialization diagnostics.
  loomc_result_t* result;
  // Workspace block pool for transient materialized input modules.
  iree_arena_block_pool_t* block_pool;
  // Host allocator used for transient module objects.
  loomc_allocator_t allocator;
} loomc_link_materialization_context_t;

typedef struct loomc_link_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
  // Source associated with emitted diagnostics.
  const loomc_source_t* source;
} loomc_link_diagnostic_capture_t;

typedef struct loomc_link_bytecode_read_state_t {
  // Diagnostic bridge retaining the public source identity.
  loomc_link_diagnostic_capture_t capture;
  // Bytecode reader options referencing capture.
  loom_bytecode_read_options_t options;
} loomc_link_bytecode_read_state_t;

static iree_allocator_t loomc_link_iree_allocator(loomc_allocator_t allocator) {
  return iree_allocator_from_loomc(allocator);
}

static bool loomc_link_any_flag_set(loomc_link_flags_t flags,
                                    loomc_link_flags_t bits) {
  return (flags & bits) != 0;
}

static loomc_status_t loomc_link_validate_linker_options(
    const loomc_linker_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LINKER_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "linker options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "linker options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "linker option extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_link_validate_options(
    const loomc_linker_t* linker, loomc_workspace_t* workspace,
    const loomc_link_options_t* options) {
  if (linker == NULL || workspace == NULL || options == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "linker, workspace, and link options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LINK_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "link option extensions are not supported");
  }
  if (options->link_index == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link_index must not be NULL");
  }
  if (loomc_link_index_context(options->link_index) != linker->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link index was created with another context");
  }
  if (options->root_symbol_count != 0 && options->root_symbols == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "root_symbol_count is non-zero but root_symbols is NULL");
  }
  if ((options->flags & ~LOOMC_LINK_KNOWN_FLAGS) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link options contain unknown flag bits");
  }
  return loomc_config_validate_options(&options->config);
}

static loomc_status_t loomc_link_result_set_failed(loomc_result_t* result) {
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}

static loomc_status_t loomc_link_result_fail_status(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_string_view_t code, loomc_status_t status) {
  LOOMC_RETURN_IF_ERROR(loomc_result_add_status_diagnostic(
      result, source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status));
  return loomc_link_result_set_failed(result);
}

static loomc_status_t loomc_link_result_fail_iree_status(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_string_view_t code, iree_status_t status) {
  loomc_status_t public_status = loomc_status_from_iree(status);
  loomc_status_t add_status =
      loomc_link_result_fail_status(result, source, code, public_status);
  loomc_status_free(public_status);
  return add_status;
}

static iree_status_t loomc_link_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_link_diagnostic_capture_t* capture =
      (loomc_link_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      capture->result, capture->source, diagnostic));
}

static void loomc_link_bytecode_read_state_initialize(
    loomc_link_materialization_context_t* context, const loomc_source_t* source,
    loomc_link_bytecode_read_state_t* out_state) {
  *out_state = (loomc_link_bytecode_read_state_t){
      .capture =
          {
              .result = context->result,
              .source = source,
          },
  };
  out_state->options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = loomc_link_capture_diagnostic,
      .user_data = &out_state->capture,
  };
  loomc_target_pass_environment_initialize_low_repr_environment(
      loomc_context_target_pass_environment(context->linker->context),
      &out_state->options.low_repr_environment);
}

static iree_status_t loomc_link_read_complete_bytecode_module(
    loomc_link_materialization_context_t* context,
    const loom_link_module_index_provider_t* provider,
    const loom_link_module_index_module_t* module, const loomc_source_t* source,
    loom_module_t** out_module) {
  *out_module = NULL;
  IREE_ASSERT(provider->kind == LOOM_LINK_PROVIDER_BYTECODE);
  loomc_link_bytecode_read_state_t read_state;
  loomc_link_bytecode_read_state_initialize(context, source, &read_state);
  loom_bytecode_read_result_t read_result = {0};
  loom_module_t* materialized_module = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_read_module_ordinal(
      provider->bytecode.contents, provider->bytecode.filename,
      loomc_context_loom_context(context->linker->context), context->block_pool,
      (uint16_t)module->provider_module_ordinal, &read_state.options,
      &read_result, &materialized_module,
      loomc_link_iree_allocator(context->allocator)));
  if (read_result.error_count != 0 || materialized_module == NULL) {
    loom_module_free(materialized_module);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode provider '%.*s' module %" PRIhsz
                            " did not materialize",
                            (int)provider->name.size, provider->name.data,
                            module->provider_module_ordinal);
  }
  *out_module = materialized_module;
  return iree_ok_status();
}

static iree_status_t loomc_link_read_selected_bytecode_module(
    loomc_link_materialization_context_t* context,
    const loom_link_module_index_provider_t* provider,
    const loom_link_module_index_module_t* module, const loomc_source_t* source,
    loom_bytecode_symbol_ordinal_list_t selection, loom_module_t** out_module) {
  *out_module = NULL;
  IREE_ASSERT(provider->kind == LOOM_LINK_PROVIDER_BYTECODE);
  loomc_link_bytecode_read_state_t read_state;
  loomc_link_bytecode_read_state_initialize(context, source, &read_state);
  loom_bytecode_read_result_t read_result = {0};
  loom_module_t* materialized_module = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_materialize_module_symbols(
      provider->bytecode.contents, provider->bytecode.filename,
      loomc_context_loom_context(context->linker->context), context->block_pool,
      &provider->bytecode.metadata, (uint16_t)module->provider_module_ordinal,
      selection, &read_state.options, &read_result, &materialized_module,
      loomc_link_iree_allocator(context->allocator)));
  if (read_result.error_count != 0 || materialized_module == NULL) {
    loom_module_free(materialized_module);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode provider '%.*s' module %" PRIhsz
                            " did not materialize selected symbols",
                            (int)provider->name.size, provider->name.data,
                            module->provider_module_ordinal);
  }
  *out_module = materialized_module;
  return iree_ok_status();
}

static iree_status_t loomc_link_add_projected_module(
    loomc_link_materialization_context_t* context,
    const loom_link_plan_module_selection_t* selection,
    loom_linker_source_provider_import_list_t provider_imports,
    iree_host_size_t* source_symbol_ordinals, loom_linker_t* linker) {
  for (iree_host_size_t i = 0; i < selection->symbols.count; ++i) {
    source_symbol_ordinals[i] =
        selection->symbols.values[i].source_symbol->module_symbol_ordinal;
  }
  const loom_linker_source_symbol_list_t source_symbols = {
      .count = selection->symbols.count,
      .ordinals = source_symbol_ordinals,
  };

  const loom_link_module_index_module_t* module = selection->source_module;
  const bool selects_complete_module =
      selection->symbols.count == module->symbol_count;
  if (module->materialized_module != NULL) {
    if (selects_complete_module) {
      return loom_linker_add_exact_module(linker, module->materialized_module,
                                          provider_imports);
    }
    return loom_linker_add_module_symbols(linker, module->materialized_module,
                                          source_symbols, provider_imports);
  }

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(context->module_index,
                                         module->provider_ordinal);
  IREE_ASSERT(provider->kind == LOOM_LINK_PROVIDER_BYTECODE);
  const loomc_source_t* source = loomc_link_index_source_for_provider(
      context->link_index, module->provider_ordinal);
  loom_module_t* materialized_module = NULL;
  iree_status_t status = iree_ok_status();
  if (selects_complete_module) {
    status = loomc_link_read_complete_bytecode_module(
        context, provider, module, source, &materialized_module);
  } else {
    status = loomc_link_read_selected_bytecode_module(
        context, provider, module, source,
        (loom_bytecode_symbol_ordinal_list_t){
            .count = selection->symbols.count,
            .ordinals = source_symbol_ordinals,
        },
        &materialized_module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_add_exact_module(linker, materialized_module,
                                          provider_imports);
  }
  loom_module_free(materialized_module);
  return status;
}

static void loomc_link_materialization_context_initialize(
    loomc_linker_t* linker, loomc_workspace_t* workspace,
    loomc_link_index_t* link_index, loomc_result_t* result,
    loomc_link_materialization_context_t* out_context) {
  *out_context = (loomc_link_materialization_context_t){
      .linker = linker,
      .link_index = link_index,
      .module_index = loomc_link_index_module_index(link_index),
      .result = result,
      .block_pool = loomc_workspace_block_pool(workspace),
      .allocator = linker->allocator,
  };
}

static bool loomc_link_options_selective(const loomc_link_options_t* options) {
  return options->root_symbol_count != 0 ||
         loomc_link_any_flag_set(options->flags,
                                 LOOMC_LINK_FLAG_INCLUDE_EXPORTED_ROOTS);
}

static iree_status_t loomc_link_add_projected_modules(
    loomc_link_materialization_context_t* context,
    const loom_link_plan_module_projection_t* projection,
    const loom_link_plan_linker_import_projection_t* provider_imports,
    iree_arena_allocator_t* arena, loom_linker_t* linker) {
  iree_host_size_t max_module_symbol_count = 0;
  for (iree_host_size_t i = 0; i < projection->modules.count; ++i) {
    max_module_symbol_count = iree_max(
        max_module_symbol_count, projection->modules.values[i].symbols.count);
  }
  iree_host_size_t* source_symbol_ordinals = NULL;
  if (max_module_symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, max_module_symbol_count, sizeof(*source_symbol_ordinals),
        (void**)&source_symbol_ordinals));
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < projection->modules.count && iree_status_is_ok(status); ++i) {
    const loom_link_plan_module_selection_t* selection =
        &projection->modules.values[i];
    status = loomc_link_add_projected_module(
        context, selection, provider_imports->modules.values[i],
        source_symbol_ordinals, linker);
  }
  return status;
}

static iree_string_view_t loomc_link_module_name(
    const loomc_linker_t* linker, const loomc_link_options_t* options) {
  if (!loomc_string_view_is_empty(options->module_name)) {
    return iree_string_view_from_loomc(options->module_name);
  }
  if (!loomc_string_view_is_empty(linker->module_name)) {
    return iree_string_view_from_loomc(linker->module_name);
  }
  return IREE_SV("linked");
}

static loomc_status_t loomc_link_translate_operation_status(
    loomc_result_t* result, loomc_host_size_t before_diagnostics,
    loomc_string_view_t code, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return loomc_ok_status();
  }
  if (loomc_result_diagnostic_count(result) == before_diagnostics) {
    return loomc_link_result_fail_iree_status(result, /*source=*/NULL, code,
                                              status);
  }
  iree_status_free(status);
  return loomc_link_result_set_failed(result);
}

loomc_status_t loomc_linker_create(loomc_context_t* context,
                                   const loomc_linker_options_t* options,
                                   loomc_allocator_t allocator,
                                   loomc_linker_t** out_linker) {
  if (context == NULL || out_linker == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context and out_linker must not be NULL");
  }
  *out_linker = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_link_validate_linker_options(options));

  loomc_linker_t* linker = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*linker), (void**)&linker));
  memset(linker, 0, sizeof(*linker));
  iree_atomic_ref_count_init(&linker->ref_count);
  linker->allocator = allocator;
  linker->context = context;
  loomc_context_retain(context);

  loomc_status_t status = loomc_string_view_clone(
      options ? options->module_name : loomc_string_view_empty(), allocator,
      &linker->module_name);
  if (loomc_status_is_ok(status)) {
    *out_linker = linker;
  } else {
    loomc_linker_release(linker);
  }
  return status;
}

void loomc_linker_retain(loomc_linker_t* linker) {
  if (linker == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&linker->ref_count);
}

void loomc_linker_release(loomc_linker_t* linker) {
  if (linker == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&linker->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = linker->allocator;
  loomc_context_release(linker->context);
  loomc_allocator_free(allocator, (void*)linker->module_name.data);
  loomc_allocator_free(allocator, linker);
}

loomc_status_t loomc_link_module(loomc_linker_t* linker,
                                 loomc_workspace_t* workspace,
                                 const loomc_link_options_t* options,
                                 loomc_module_t** out_module,
                                 loomc_result_t** out_result) {
  if (out_module == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_module and out_result must not be NULL");
  }
  *out_module = NULL;
  *out_result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_link_validate_options(linker, workspace, options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                            linker->allocator, &result));

  iree_arena_allocator_t arena = {0};
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &arena);
  loomc_link_materialization_context_t materialization = {0};
  loom_link_plan_t* plan = NULL;
  loom_link_plan_module_projection_t module_projection = {0};
  loom_link_plan_linker_import_projection_t provider_import_projection = {0};
  loomc_module_t* module = NULL;
  loom_linker_t* internal_linker = NULL;
  loom_module_t* linked_module = NULL;
  loomc_link_materialization_context_initialize(
      linker, workspace, options->link_index, result, &materialization);
  loomc_status_t status = loomc_ok_status();

  iree_string_view_t* root_symbols = NULL;
  if (loomc_status_is_ok(status) && options->root_symbol_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &arena, options->root_symbol_count, sizeof(*root_symbols),
        (void**)&root_symbols));
    for (loomc_host_size_t i = 0;
         loomc_status_is_ok(status) && i < options->root_symbol_count; ++i) {
      root_symbols[i] = iree_string_view_from_loomc(options->root_symbols[i]);
    }
  }

  const bool selective = loomc_link_options_selective(options);
  loom_link_plan_options_t plan_options = {
      .mode = selective ? LOOM_LINK_PLAN_SELECTIVE : LOOM_LINK_PLAN_ARCHIVE,
      .root_symbols =
          {
              .count = options->root_symbol_count,
              .values = root_symbols,
          },
      .include_exported_roots = loomc_link_any_flag_set(
          options->flags, LOOMC_LINK_FLAG_INCLUDE_EXPORTED_ROOTS),
      .unresolved_policy =
          loomc_link_any_flag_set(options->flags,
                                  LOOMC_LINK_FLAG_ALLOW_UNRESOLVED_SYMBOLS)
              ? LOOM_LINK_PLAN_UNRESOLVED_ALLOW
              : LOOM_LINK_PLAN_UNRESOLVED_ERROR,
      .test_symbol_policy =
          loomc_link_any_flag_set(options->flags,
                                  LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS)
              ? LOOM_LINK_PLAN_TEST_SYMBOL_STRIP
              : LOOM_LINK_PLAN_TEST_SYMBOL_KEEP,
      .provider_resolver =
          loomc_link_index_provider_resolver(options->link_index),
  };

  iree_status_t operation_status = iree_ok_status();
  loomc_host_size_t before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status)) {
    operation_status = loom_link_plan_build(
        materialization.module_index, &plan_options,
        loomc_link_iree_allocator(linker->allocator), &plan);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/PLAN"),
        operation_status);
  }

  before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    operation_status =
        loom_link_plan_project_modules(plan, &arena, &module_projection);
    if (iree_status_is_ok(operation_status)) {
      operation_status = loom_link_plan_project_linker_imports(
          materialization.module_index, &module_projection, &arena,
          &provider_import_projection);
    }
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/PROJECT"),
        operation_status);
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_module_create_empty(linker->context, workspace,
                                       linker->allocator, &module);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loom_linker_options_t linker_options = {
        .module_name = loomc_link_module_name(linker, options),
        .provider_imports =
            {
                .count = provider_import_projection.provider_imports.count,
                .anchor_count =
                    provider_import_projection.provider_import_anchors.count,
            },
    };
    status = loomc_status_from_iree(loom_linker_create(
        loomc_context_loom_context(linker->context), &linker_options,
        loomc_module_block_pool(module),
        loomc_link_iree_allocator(linker->allocator), &internal_linker));
  }
  before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    operation_status = loomc_link_add_projected_modules(
        &materialization, &module_projection, &provider_import_projection,
        &arena, internal_linker);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/ADD"),
        operation_status);
  }
  before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      options->root_symbol_count != 0) {
    operation_status = loom_linker_finalize_roots(
        internal_linker, (iree_string_view_list_t){
                             .count = options->root_symbol_count,
                             .values = root_symbols,
                         });
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/ROOTS"),
        operation_status);
  }
  before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    operation_status = loom_linker_finish(internal_linker, &linked_module);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/FINISH"),
        operation_status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_config_apply_to_module_options_t config_apply_options = {
        .config = &options->config,
        .module = linked_module,
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = linker->allocator,
    };
    status = loomc_config_apply_to_module(&config_apply_options);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_module_set_loom_module(module, linked_module);
    if (loomc_status_is_ok(status)) {
      linked_module = NULL;
    }
  }

  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_module = module;
      module = NULL;
    }
    *out_result = result;
    result = NULL;
  }

  loom_linker_free(internal_linker);
  loom_module_free(linked_module);
  loomc_module_release(module);
  loom_link_plan_free(plan);
  iree_arena_deinitialize(&arena);
  loomc_result_release(result);
  return status;
}
