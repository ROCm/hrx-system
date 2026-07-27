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
#include "loom/link/linker.h"
#include "loom/link/module_index.h"
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
                           LOOMC_LINK_FLAG_STRIP_CHECK_SYMBOLS,
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

typedef struct loomc_link_materialization_cache_t {
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
  // Host allocator used for cache arrays and transient module objects.
  loomc_allocator_t allocator;
  // Owned bytecode-materialized modules indexed by module ordinal.
  loom_module_t** materialized_modules;
  // Number of entries in materialized_modules.
  loomc_host_size_t module_count;
} loomc_link_materialization_cache_t;

typedef struct loomc_link_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
  // Source associated with emitted diagnostics.
  const loomc_source_t* source;
} loomc_link_diagnostic_capture_t;

typedef struct loomc_link_config_known_key_query_t {
  // Frozen index that supplied the link plan.
  const loom_link_module_index_t* index;
  // Module ordinals selected by the link plan.
  const uint8_t* selected_modules;
  // Number of entries in |selected_modules|.
  iree_host_size_t selected_module_count;
} loomc_link_config_known_key_query_t;

static iree_allocator_t loomc_link_iree_allocator(loomc_allocator_t allocator) {
  return iree_allocator_from_loomc(allocator);
}

static bool loomc_link_any_flag_set(loomc_link_flags_t flags,
                                    loomc_link_flags_t bits) {
  return (flags & bits) != 0;
}

static bool loomc_link_config_options_have_bindings(
    const loomc_config_options_t* config) {
  return config && (config->binding_count != 0 ||
                    !loomc_string_view_is_empty(config->json_object));
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
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_target_selection_options_resolve(options->next, &target_selection));
  LOOMC_RETURN_IF_ERROR(loomc_target_selection_validate_environment(
      target_selection, loomc_context_target_environment(linker->context)));
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

static iree_status_t loomc_link_read_bytecode_module(
    loomc_link_materialization_cache_t* cache,
    const loom_link_module_index_provider_t* provider,
    const loom_link_module_index_module_t* module, const loomc_source_t* source,
    loom_module_t** out_module) {
  *out_module = NULL;
  if (module->provider_module_ordinal > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "provider module ordinal %" PRIhsz
                            " exceeds bytecode reader limit",
                            module->provider_module_ordinal);
  }

  loomc_byte_span_t contents = loomc_source_contents(source);
  loomc_string_view_t identifier = loomc_source_identifier(source);
  loomc_link_diagnostic_capture_t capture = {
      .result = cache->result,
      .source = source,
  };
  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink =
          {
              .fn = loomc_link_capture_diagnostic,
              .user_data = &capture,
          },
  };
  loom_bytecode_read_result_t read_result = {0};
  loom_module_t* materialized_module = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_read_module_ordinal(
      iree_make_const_byte_span(contents.data, contents.data_length),
      iree_string_view_from_loomc(identifier),
      loomc_context_loom_context(cache->linker->context), cache->block_pool,
      (uint16_t)module->provider_module_ordinal, &read_options, &read_result,
      &materialized_module, loomc_link_iree_allocator(cache->allocator)));
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

static iree_status_t loomc_link_materialize_module(
    loomc_link_materialization_cache_t* cache,
    const loom_link_module_index_module_t* module,
    const loom_module_t** out_module) {
  *out_module = NULL;
  if (module == NULL || module->ordinal >= cache->module_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link plan referenced a stale module");
  }
  if (module->materialized_module != NULL) {
    *out_module = module->materialized_module;
    return iree_ok_status();
  }
  if (cache->materialized_modules[module->ordinal] != NULL) {
    *out_module = cache->materialized_modules[module->ordinal];
    return iree_ok_status();
  }

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(cache->module_index,
                                         module->provider_ordinal);
  if (provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link plan referenced a stale provider");
  }
  if (provider->kind != LOOM_LINK_PROVIDER_BYTECODE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "provider '%.*s' module '%.*s' has no materialized IR",
        (int)provider->name.size, provider->name.data, (int)module->name.size,
        module->name.data);
  }

  const loomc_source_t* source = loomc_link_index_source_for_provider(
      cache->link_index, module->provider_ordinal);
  if (source == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "bytecode provider '%.*s' has no retained source",
                            (int)provider->name.size, provider->name.data);
  }

  loom_module_t* materialized_module = NULL;
  IREE_RETURN_IF_ERROR(loomc_link_read_bytecode_module(
      cache, provider, module, source, &materialized_module));
  cache->materialized_modules[module->ordinal] = materialized_module;
  *out_module = materialized_module;
  return iree_ok_status();
}

static iree_status_t loomc_link_materialize_module_callback(
    void* user_data, const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module,
    const loom_module_t** out_module) {
  loomc_link_materialization_cache_t* cache =
      (loomc_link_materialization_cache_t*)user_data;
  if (index != cache->module_index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link plan materializer index mismatch");
  }
  return loomc_link_materialize_module(cache, module, out_module);
}

static loomc_status_t loomc_link_materialization_cache_initialize(
    loomc_linker_t* linker, loomc_workspace_t* workspace,
    loomc_link_index_t* link_index, loomc_result_t* result,
    loomc_link_materialization_cache_t* out_cache) {
  memset(out_cache, 0, sizeof(*out_cache));
  out_cache->linker = linker;
  out_cache->link_index = link_index;
  out_cache->module_index = loomc_link_index_module_index(link_index);
  out_cache->result = result;
  out_cache->block_pool = loomc_workspace_block_pool(workspace);
  out_cache->allocator = linker->allocator;
  out_cache->module_count =
      loom_link_module_index_module_count(out_cache->module_index);
  if (out_cache->module_count == 0) {
    return loomc_ok_status();
  }
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      out_cache->allocator,
      out_cache->module_count * sizeof(*out_cache->materialized_modules),
      (void**)&out_cache->materialized_modules));
  memset(out_cache->materialized_modules, 0,
         out_cache->module_count * sizeof(*out_cache->materialized_modules));
  return loomc_ok_status();
}

static void loomc_link_materialization_cache_deinitialize(
    loomc_link_materialization_cache_t* cache) {
  if (cache->materialized_modules != NULL) {
    for (loomc_host_size_t i = 0; i < cache->module_count; ++i) {
      loom_module_free(cache->materialized_modules[i]);
    }
  }
  loomc_allocator_free(cache->allocator, cache->materialized_modules);
}

static bool loomc_link_options_selective(const loomc_link_options_t* options) {
  return options->root_symbol_count != 0 ||
         loomc_link_any_flag_set(options->flags,
                                 LOOMC_LINK_FLAG_INCLUDE_EXPORTED_ROOTS);
}

static iree_status_t loomc_link_plan_build_roots(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    iree_string_view_list_t** out_root_lists) {
  *out_root_lists = NULL;
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(index);
  if (module_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t* root_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_count, sizeof(*root_counts), (void**)&root_counts));
  memset(root_counts, 0, module_count * sizeof(*root_counts));

  const iree_host_size_t plan_symbol_count = loom_link_plan_symbol_count(plan);
  for (iree_host_size_t i = 0; i < plan_symbol_count; ++i) {
    const loom_link_plan_symbol_t* planned = loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, planned->symbol_ordinal);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    if (module == NULL || module->ordinal >= module_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "link plan referenced a stale module");
    }
    ++root_counts[module->ordinal];
  }

  iree_string_view_list_t* root_lists = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_count, sizeof(*root_lists), (void**)&root_lists));
  memset(root_lists, 0, module_count * sizeof(*root_lists));
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    root_lists[i].count = root_counts[i];
    if (root_counts[i] == 0) {
      continue;
    }
    iree_string_view_t* values = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, root_counts[i], sizeof(*values), (void**)&values));
    root_lists[i].values = values;
  }

  iree_host_size_t* root_positions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_count, sizeof(*root_positions), (void**)&root_positions));
  memset(root_positions, 0, module_count * sizeof(*root_positions));
  for (iree_host_size_t i = 0; i < plan_symbol_count; ++i) {
    const loom_link_plan_symbol_t* planned = loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, planned->symbol_ordinal);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    iree_string_view_t* values =
        (iree_string_view_t*)root_lists[module->ordinal].values;
    values[root_positions[module->ordinal]++] = symbol->name;
  }

  *out_root_lists = root_lists;
  return iree_ok_status();
}

static iree_status_t loomc_link_plan_build_module_bitmap(
    const loom_link_plan_t* plan, iree_arena_allocator_t* arena,
    const uint8_t** out_selected_modules,
    iree_host_size_t* out_selected_module_count) {
  *out_selected_modules = NULL;
  *out_selected_module_count = 0;
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(index);
  if (module_count == 0) {
    return iree_ok_status();
  }

  uint8_t* selected_modules = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module_count,
                                                 sizeof(*selected_modules),
                                                 (void**)&selected_modules));
  memset(selected_modules, 0, module_count * sizeof(*selected_modules));

  const iree_host_size_t plan_symbol_count = loom_link_plan_symbol_count(plan);
  for (iree_host_size_t i = 0; i < plan_symbol_count; ++i) {
    const loom_link_plan_symbol_t* planned = loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, planned->symbol_ordinal);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    if (module == NULL || module->ordinal >= module_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "link plan referenced a stale module");
    }
    selected_modules[module->ordinal] = 1;
  }

  *out_selected_modules = selected_modules;
  *out_selected_module_count = module_count;
  return iree_ok_status();
}

static bool loomc_link_config_key_known(void* user_data,
                                        iree_string_view_t key) {
  const loomc_link_config_known_key_query_t* query =
      (const loomc_link_config_known_key_query_t*)user_data;
  if (!query || !query->index || !query->selected_modules) {
    return false;
  }
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(query->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(query->index, i);
    if (!symbol || symbol->module_ordinal >= query->selected_module_count ||
        !query->selected_modules[symbol->module_ordinal] ||
        !iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
      continue;
    }
    if (iree_string_view_equal(symbol->name, key)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loomc_link_add_planned_modules(
    loomc_link_materialization_cache_t* cache, const loom_link_plan_t* plan,
    bool selective, iree_arena_allocator_t* arena, loom_linker_t* linker) {
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(index);
  uint8_t* added_modules = NULL;
  if (module_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module_count, sizeof(*added_modules), (void**)&added_modules));
    memset(added_modules, 0, module_count * sizeof(*added_modules));
  }

  iree_string_view_list_t* root_lists = NULL;
  if (selective) {
    IREE_RETURN_IF_ERROR(loomc_link_plan_build_roots(plan, arena, &root_lists));
  }

  const iree_host_size_t plan_symbol_count = loom_link_plan_symbol_count(plan);
  for (iree_host_size_t i = 0; i < plan_symbol_count; ++i) {
    const loom_link_plan_symbol_t* planned = loom_link_plan_symbol_at(plan, i);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, planned->symbol_ordinal);
    const loom_link_module_index_module_t* module =
        loom_link_module_index_symbol_module(index, symbol);
    if (module == NULL || module->ordinal >= module_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "link plan referenced a stale module");
    }
    if (added_modules[module->ordinal]) {
      continue;
    }

    const loom_module_t* materialized_module = NULL;
    IREE_RETURN_IF_ERROR(
        loomc_link_materialize_module(cache, module, &materialized_module));
    loom_linker_add_options_t add_options = {0};
    if (selective) {
      add_options.root_symbols = root_lists[module->ordinal];
    }
    IREE_RETURN_IF_ERROR(
        loom_linker_add_module(linker, materialized_module, &add_options));
    added_modules[module->ordinal] = 1;
  }
  return iree_ok_status();
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
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_target_selection_options_resolve(options->next, &target_selection));
  const loom_target_selection_t internal_target_selection =
      loomc_target_selection_loom_target_selection(target_selection);
  const loom_target_environment_t* internal_target_environment =
      loomc_target_environment_loom_target_environment(
          loomc_context_target_environment(linker->context));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                            linker->allocator, &result));

  iree_arena_allocator_t arena = {0};
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &arena);
  loomc_link_materialization_cache_t cache = {0};
  loom_link_plan_t* plan = NULL;
  loomc_module_t* module = NULL;
  loom_linker_t* internal_linker = NULL;
  loom_module_t* linked_module = NULL;
  const uint8_t* config_selected_modules = NULL;
  iree_host_size_t config_selected_module_count = 0;
  loomc_link_config_known_key_query_t config_known_key_query = {0};

  loomc_status_t status = loomc_link_materialization_cache_initialize(
      linker, workspace, options->link_index, result, &cache);

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
      .check_policy = loomc_link_any_flag_set(
                          options->flags, LOOMC_LINK_FLAG_STRIP_CHECK_SYMBOLS)
                          ? LOOM_LINK_PLAN_CHECK_STRIP
                          : LOOM_LINK_PLAN_CHECK_KEEP,
      .materialize_module = loomc_link_materialize_module_callback,
      .materialize_module_user_data = &cache,
  };

  iree_status_t operation_status = iree_ok_status();
  loomc_host_size_t before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status)) {
    operation_status = loom_link_plan_build(
        cache.module_index, &plan_options,
        loomc_workspace_block_pool(workspace),
        loomc_link_iree_allocator(linker->allocator), &plan);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/PLAN"),
        operation_status);
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_module_create_empty(linker->context, workspace,
                                       linker->allocator, &module);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loom_linker_options_t linker_options = {
        .module_name = loomc_link_module_name(linker, options),
    };
    status = loomc_status_from_iree(loom_linker_create(
        loomc_context_loom_context(linker->context), &linker_options,
        loomc_module_block_pool(module),
        loomc_link_iree_allocator(linker->allocator), &internal_linker));
  }
  before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    operation_status = loomc_link_add_planned_modules(&cache, plan, selective,
                                                      &arena, internal_linker);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/ADD"),
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
    if (iree_any_bit_set(options->config.flags,
                         LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN) &&
        loomc_link_config_options_have_bindings(&options->config)) {
      status = loomc_status_from_iree(loomc_link_plan_build_module_bitmap(
          plan, &arena, &config_selected_modules,
          &config_selected_module_count));
      config_known_key_query = (loomc_link_config_known_key_query_t){
          .index = cache.module_index,
          .selected_modules = config_selected_modules,
          .selected_module_count = config_selected_module_count,
      };
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_config_apply_to_module_options_t config_apply_options = {
        .config = &options->config,
        .module = linked_module,
        .result = result,
        .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
        .is_known_key =
            config_selected_modules ? loomc_link_config_key_known : NULL,
        .known_key_user_data = &config_known_key_query,
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = linker->allocator,
    };
    status = loomc_config_apply_to_module(&config_apply_options);
  }
  before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      internal_target_environment != NULL &&
      !loom_target_selection_is_empty(internal_target_selection)) {
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    operation_status = loom_target_environment_materialize_selection(
        internal_target_environment, linked_module, internal_target_selection,
        &target_ref);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/TARGET"),
        operation_status);
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
  loomc_link_materialization_cache_deinitialize(&cache);
  iree_arena_deinitialize(&arena);
  loomc_result_release(result);
  return status;
}
