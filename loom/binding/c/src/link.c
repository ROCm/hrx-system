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
#include "link_materialization.h"
#include "loom/link/index_materializer.h"
#include "loom/link/module_index.h"
#include "loomc/iree.h"
#include "module.h"
#include "module_bytecode.h"
#include "product.h"
#include "request_index_overlay.h"
#include "result.h"
#include "source.h"
#include "target.h"
#include "workspace.h"

enum {
  LOOMC_LINK_KNOWN_FLAGS = LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS |
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

// Invocation-local module index combining an optional immutable library prefix
// with borrowed materialized module providers.
typedef struct loomc_link_module_overlay_t {
  // Index consumed by planning and materialization.
  const loom_link_module_index_t* module_index;
  // Owned overlay when direct providers are present, or NULL.
  loom_link_module_index_t* owned_index;
  // Number of provider ordinals owned by the immutable prefix.
  iree_host_size_t base_provider_count;
} loomc_link_module_overlay_t;

static bool loomc_link_any_flag_set(loomc_link_flags_t flags,
                                    loomc_link_flags_t bits) {
  return (flags & bits) != 0;
}

static loom_link_provider_role_t loomc_link_provider_role_to_loom(
    loomc_link_provider_role_t role) {
  switch (role) {
    case LOOMC_LINK_PROVIDER_ROLE_INPUT:
      return LOOM_LINK_PROVIDER_ROLE_INPUT;
    case LOOMC_LINK_PROVIDER_ROLE_LIBRARY:
      return LOOM_LINK_PROVIDER_ROLE_LIBRARY;
  }
  IREE_ASSERT_UNREACHABLE("unknown public link provider role");
  return LOOM_LINK_PROVIDER_ROLE_INPUT;
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
    const loomc_link_options_t* options,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
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
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, out_target_specialization));
  if (options->link_index == NULL && options->module_provider_count == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "linking requires at least one provider");
  }
  if (options->link_index != NULL &&
      loomc_link_index_context(options->link_index) != linker->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link index was created with another context");
  }
  if (options->module_provider_count != 0 &&
      options->module_providers == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module_provider_count is non-zero but module_providers is NULL");
  }
  if (options->module_provider_count != 0 && options->link_index != NULL &&
      loom_link_module_index_input_provider_count(
          loomc_link_index_module_index(options->link_index)) != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "a link index combined with module providers must contain only "
        "libraries");
  }
  for (loomc_host_size_t i = 0; i < options->module_provider_count; ++i) {
    const loomc_link_module_provider_t* provider =
        &options->module_providers[i];
    if (provider->module == NULL) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "module provider must not be NULL");
    }
    if (loomc_module_context(provider->module) != linker->context) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module provider was created with another context");
    }
    if (provider->provider_name.data == NULL &&
        provider->provider_name.size != 0) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "module provider name has length but no data");
    }
    switch (provider->role) {
      case LOOMC_LINK_PROVIDER_ROLE_INPUT:
      case LOOMC_LINK_PROVIDER_ROLE_LIBRARY:
        break;
      default:
        return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                                 "module provider has an unknown role");
    }
  }
  if (options->root_symbol_count != 0 && options->root_symbols == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "root_symbol_count is non-zero but root_symbols is NULL");
  }
  if (options->root_provider_count != 0 &&
      options->root_provider_ordinals == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "root_provider_count is non-zero but root_provider_ordinals is NULL");
  }
  if ((options->flags & ~LOOMC_LINK_KNOWN_FLAGS) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link options contain unknown flag bits");
  }
  if (options->mode != LOOMC_LINK_MODE_MERGE &&
      options->mode != LOOMC_LINK_MODE_LINK) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link options contain an unknown mode");
  }
  const bool has_roots =
      options->root_symbol_count != 0 || options->root_provider_count != 0 ||
      loomc_link_any_flag_set(options->flags,
                              LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS);
  if (options->mode == LOOMC_LINK_MODE_MERGE && has_roots) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "merge mode does not accept roots or include input exports");
  }
  if (options->mode == LOOMC_LINK_MODE_LINK && !has_roots) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "link mode requires roots or include input exports");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_target_specialization_options_validate_environment(
          *out_target_specialization,
          loomc_context_target_environment(linker->context)));
  return loomc_config_validate_text_options(&options->config);
}

static loomc_status_t loomc_link_validate_request_options(
    const loomc_linker_t* linker, loomc_workspace_t* workspace,
    const loomc_request_t* request, const loomc_link_request_options_t* options,
    loomc_allocator_t allocator,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
  if (linker == NULL || workspace == NULL || request == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "linker, workspace, and input request must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  if (loomc_source_format(loomc_request_source(request)) !=
      LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "input request source is not Loom bytecode");
  }
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "request link options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "request link options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, out_target_specialization));
  if (options->library_index != NULL) {
    if (loomc_link_index_context(options->library_index) != linker->context) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "request library index was created with another context");
    }
    if (loom_link_module_index_input_provider_count(
            loomc_link_index_module_index(options->library_index)) != 0) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "request library index contains primary input providers");
    }
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_target_specialization_options_validate_environment(
          *out_target_specialization,
          loomc_context_target_environment(linker->context)));
  return loomc_config_validate_text_options(&options->config);
}

static loomc_status_t loomc_link_result_fail_iree_status(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_string_view_t code, iree_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code,
      loomc_status_from_iree(status));
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

static iree_string_view_t loomc_link_request_module_name(
    const loomc_linker_t* linker, const loomc_link_request_options_t* options) {
  if (options != NULL && !loomc_string_view_is_empty(options->module_name)) {
    return iree_string_view_from_loomc(options->module_name);
  }
  if (!loomc_string_view_is_empty(linker->module_name)) {
    return iree_string_view_from_loomc(linker->module_name);
  }
  return IREE_SV("linked-request");
}

static iree_status_t loomc_link_request_make_source_identifier(
    iree_string_view_t module_name, iree_arena_allocator_t* arena,
    loomc_string_view_t* out_identifier) {
  const iree_string_view_t extension = IREE_SV(".loombc");
  iree_host_size_t identifier_length = 0;
  if (!iree_host_size_checked_add(module_name.size, extension.size,
                                  &identifier_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "linked request identifier is too large");
  }
  char* identifier = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, identifier_length, (void**)&identifier));
  memcpy(identifier, module_name.data, module_name.size);
  memcpy(identifier + module_name.size, extension.data, extension.size);
  *out_identifier = loomc_make_string_view(identifier, identifier_length);
  return iree_ok_status();
}

static loomc_status_t loomc_link_request_map_target_roots(
    const loom_link_plan_materialization_t* materialization,
    const iree_host_size_t* source_symbol_ordinals, iree_host_size_t root_count,
    loom_symbol_id_t* out_module_symbol_ids) {
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    const iree_host_size_t source_symbol_ordinal = source_symbol_ordinals[i];
    if (source_symbol_ordinal >= materialization->target_symbols.count) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "linked request root is absent from the target projection");
    }
    const loom_symbol_ref_t target_symbol =
        materialization->target_symbols.values[source_symbol_ordinal];
    if (!loom_symbol_ref_is_valid(target_symbol) ||
        target_symbol.module_id != 0) {
      return loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "linked request root did not materialize as a local symbol");
    }
    out_module_symbol_ids[i] = target_symbol.symbol_id;
  }
  return loomc_ok_status();
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
  loomc_result_set_status_code(result,
                               (loomc_status_code_t)iree_status_code(status));
  iree_status_free(status);
  return loomc_ok_status();
}

static loomc_status_t loomc_link_module_overlay_initialize(
    const loomc_linker_t* linker, loomc_workspace_t* workspace,
    const loomc_link_options_t* options,
    loomc_link_module_overlay_t* out_overlay) {
  *out_overlay = (loomc_link_module_overlay_t){0};
  const loom_link_module_index_t* base_index =
      loomc_link_index_module_index(options->link_index);
  const iree_host_size_t base_provider_count =
      loom_link_module_index_provider_count(base_index);
  if (options->module_provider_count == 0) {
    out_overlay->module_index = base_index;
    out_overlay->base_provider_count = base_provider_count;
    return loomc_ok_status();
  }

  loom_link_module_index_t* module_index = NULL;
  const iree_allocator_t allocator =
      iree_allocator_from_loomc(linker->allocator);
  loomc_status_t status =
      base_index != NULL
          ? loomc_status_from_iree(loom_link_module_index_allocate_overlay(
                base_index, loomc_workspace_block_pool(workspace), allocator,
                &module_index))
          : loomc_status_from_iree(loom_link_module_index_allocate(
                loomc_context_loom_context(linker->context),
                loomc_workspace_block_pool(workspace), allocator,
                &module_index));
  for (loomc_host_size_t i = 0;
       loomc_status_is_ok(status) && i < options->module_provider_count; ++i) {
    const loomc_link_module_provider_t* provider =
        &options->module_providers[i];
    const loom_link_module_index_add_options_t add_options = {
        .provider_name = iree_string_view_from_loomc(provider->provider_name),
        .role = loomc_link_provider_role_to_loom(provider->role),
    };
    status = loomc_status_from_iree(loom_link_module_index_add_materialized(
        module_index, loomc_module_const_loom_module(provider->module),
        &add_options, /*out_provider_ordinal=*/NULL));
  }
  if (loomc_status_is_ok(status)) {
    *out_overlay = (loomc_link_module_overlay_t){
        .module_index = module_index,
        .owned_index = module_index,
        .base_provider_count = base_provider_count,
    };
    module_index = NULL;
  }
  loom_link_module_index_free(module_index);
  return status;
}

static void loomc_link_module_overlay_deinitialize(
    loomc_link_module_overlay_t* overlay) {
  loom_link_module_index_free(overlay->owned_index);
  *overlay = (loomc_link_module_overlay_t){0};
}

static loomc_status_t loomc_link_capture_source_table(
    const loom_source_table_resolver_t* source_table,
    const loom_link_source_projection_t* projection,
    loomc_module_t* target_module) {
  for (iree_host_size_t i = 0; i < source_table->count; ++i) {
    const loom_source_entry_t* entry = &source_table->entries[i];
    if (entry->source_id == LOOM_SOURCE_ID_INVALID) {
      continue;
    }
    IREE_ASSERT_LT(entry->source_id, projection->count);
    const loom_source_id_t target_source_id =
        projection->values[entry->source_id];
    if (target_source_id == LOOM_SOURCE_ID_INVALID) {
      continue;
    }
    LOOMC_RETURN_IF_ERROR(loomc_module_insert_source_snapshot(
        target_module, target_source_id, entry->filename, entry->source));
  }
  return loomc_ok_status();
}

// Copies exact source snapshots through the source-ID projection produced by
// linking. Bytecode providers have no source projection because their container
// does not carry authored source contents.
static loomc_status_t loomc_link_capture_source_snapshots(
    loomc_context_t* context, const loomc_link_index_t* link_index,
    const loomc_link_module_provider_t* module_providers,
    loomc_host_size_t module_provider_count,
    const loomc_link_module_overlay_t* overlay,
    const loom_link_plan_materialization_t* materialization,
    loomc_module_t* target_module) {
  if (loomc_context_source_retention(context) ==
      LOOMC_SOURCE_RETENTION_METADATA_ONLY) {
    return loomc_ok_status();
  }

  for (iree_host_size_t i = 0; i < materialization->target_sources.count; ++i) {
    const loom_link_source_projection_t* projection =
        &materialization->target_sources.values[i];
    if (projection->count == 0) {
      continue;
    }
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_module_at(overlay->module_index, i);
    if (indexed_module == NULL || indexed_module->materialized_module == NULL) {
      continue;
    }

    if (indexed_module->provider_ordinal >= overlay->base_provider_count) {
      const iree_host_size_t provider_index =
          indexed_module->provider_ordinal - overlay->base_provider_count;
      IREE_ASSERT_LT(provider_index, module_provider_count);
      LOOMC_RETURN_IF_ERROR(loomc_link_capture_source_table(
          loomc_module_source_table(module_providers[provider_index].module),
          projection, target_module));
      continue;
    }

    const loomc_source_t* source = loomc_link_index_source_for_provider(
        link_index, indexed_module->provider_ordinal);
    if (source == NULL) {
      continue;
    }

    if (indexed_module->primary_source_id == LOOM_SOURCE_ID_INVALID) {
      continue;
    }
    const loom_source_id_t target_source_id =
        projection->values[indexed_module->primary_source_id];
    if (target_source_id == LOOM_SOURCE_ID_INVALID) {
      continue;
    }
    const iree_string_view_t identifier =
        iree_string_view_from_loomc(loomc_source_identifier(source));
    const loomc_byte_span_t contents = loomc_source_contents(source);
    LOOMC_RETURN_IF_ERROR(loomc_module_insert_source_snapshot(
        target_module, target_source_id, identifier,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length)));
  }
  return loomc_ok_status();
}

loomc_status_t loomc_linker_create(loomc_context_t* context,
                                   const loomc_linker_options_t* options,
                                   loomc_allocator_t allocator,
                                   loomc_linker_t** out_linker) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (context == NULL || out_linker == NULL) {
    loomc_status_t return_status =
        loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                          "context and out_linker must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  *out_linker = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_link_validate_linker_options(options));

  loomc_linker_t* linker = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_allocator_malloc(allocator, sizeof(*linker), (void**)&linker));
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
  IREE_TRACE_ZONE_END(z0);
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
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_module == NULL || out_result == NULL) {
    loomc_status_t return_status =
        loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                          "out_module and out_result must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  *out_module = NULL;
  *out_result = NULL;
  const loomc_target_specialization_options_t* target_specialization = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_link_validate_options(linker, workspace, options,
                                      &target_specialization));

  loomc_result_t* result = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_result_create(LOOMC_STATUS_OK,
                              loomc_context_source_retention(linker->context),
                              linker->allocator, &result));

  iree_arena_allocator_t arena = {0};
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &arena);
  loomc_link_module_overlay_t overlay = {0};
  loomc_link_materialization_state_t materialization_state = {0};
  loom_link_index_materialization_t index_materialization = {0};
  loomc_module_t* module = NULL;
  loomc_link_materialization_state_initialize(
      linker->context, workspace, options->link_index, &options->config,
      target_specialization, result, &arena, linker->allocator,
      &materialization_state);
  loomc_status_t status = loomc_link_module_overlay_initialize(
      linker, workspace, options, &overlay);

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

  loom_link_plan_options_t plan_options = {
      .mode = options->mode == LOOMC_LINK_MODE_LINK ? LOOM_LINK_PLAN_LINK
                                                    : LOOM_LINK_PLAN_MERGE,
      .root_symbols =
          {
              .count = options->root_symbol_count,
              .values = root_symbols,
          },
      .include_input_exports = loomc_link_any_flag_set(
          options->flags, LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS),
      .root_providers =
          {
              .count = options->root_provider_count,
              .values = options->root_provider_ordinals,
          },
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
  };

  const loom_link_plan_materialization_environment_t environment =
      loomc_link_materialization_state_environment(&materialization_state);
  loomc_host_size_t before_diagnostics = loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    iree_status_t operation_status = loom_link_index_materialize(
        overlay.module_index, &plan_options, &environment,
        loomc_link_module_name(linker, options), &index_materialization);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/MATERIALIZE"),
        operation_status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_module_create_empty(linker->context, workspace,
                                       linker->allocator, &module);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_link_capture_source_snapshots(
        linker->context, options->link_index, options->module_providers,
        options->module_provider_count, &overlay,
        &index_materialization.product, module);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_module_set_loom_module(module, index_materialization.product.module,
                                 LOOMC_MODULE_INPUT_UNVERIFIED);
    index_materialization.product.module = NULL;
  }

  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_module = module;
      module = NULL;
    }
    *out_result = result;
    result = NULL;
  }

  loom_link_index_materialization_deinitialize(&index_materialization);
  loomc_link_module_overlay_deinitialize(&overlay);
  loomc_module_release(module);
  iree_arena_deinitialize(&arena);
  loomc_result_release(result);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

loomc_status_t loomc_link_request(loomc_linker_t* linker,
                                  loomc_workspace_t* workspace,
                                  const loomc_request_t* input_request,
                                  const loomc_link_request_options_t* options,
                                  loomc_allocator_t allocator,
                                  loomc_request_t** out_request,
                                  loomc_result_t** out_result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_request == NULL || out_result == NULL) {
    loomc_status_t return_status =
        loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                          "out_request and out_result must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  *out_request = NULL;
  *out_result = NULL;
  const loomc_target_specialization_options_t* target_specialization = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_link_validate_request_options(linker, workspace, input_request,
                                              options, allocator,
                                              &target_specialization));

  loomc_result_t* result = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_result_create(LOOMC_STATUS_OK,
                              loomc_context_source_retention(linker->context),
                              allocator, &result));

  const loomc_config_options_t empty_config = {0};
  const loomc_config_options_t* config =
      options != NULL ? &options->config : &empty_config;
  const loomc_link_index_t* library_index =
      options != NULL ? options->library_index : NULL;
  const loomc_source_t* input_source = loomc_request_source(input_request);
  const iree_host_size_t root_count = loomc_request_root_count(input_request);

  iree_arena_allocator_t scratch_arena = {0};
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loomc_request_index_overlay_t request_overlay = {0};
  loomc_link_materialization_state_t materialization_state = {0};
  loom_link_index_materialization_t materialization = {0};
  loomc_source_t* output_source = NULL;
  loomc_request_t* output_request = NULL;
  loomc_status_t status = loomc_ok_status();

  status = loomc_request_index_overlay_initialize(
      linker->context, loomc_workspace_block_pool(workspace), library_index,
      input_request, result, &scratch_arena, allocator, &request_overlay);

  loom_symbol_id_t* target_root_symbol_ids = NULL;
  loom_symbol_id_t* bytecode_root_symbol_ordinals = NULL;
  loomc_request_root_t* output_roots = NULL;
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      root_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*target_root_symbol_ids),
        (void**)&target_root_symbol_ids));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      root_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*bytecode_root_symbol_ordinals),
        (void**)&bytecode_root_symbol_ordinals));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      root_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*output_roots),
        (void**)&output_roots));
  }
  const loom_link_plan_options_t plan_options = {
      .mode = LOOM_LINK_PLAN_LINK,
      .unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ERROR,
      .root_symbol_ordinals =
          {
              .count = root_count,
              .values = request_overlay.root_symbol_ordinals,
          },
  };
  loomc_link_materialization_state_initialize_overlay(
      linker->context, workspace, library_index,
      request_overlay.request_provider_ordinal, input_source, config,
      target_specialization, result, &scratch_arena, allocator,
      &materialization_state);
  const loom_link_plan_materialization_environment_t environment =
      loomc_link_materialization_state_environment(&materialization_state);
  const loomc_host_size_t before_diagnostics =
      loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const iree_status_t operation_status = loom_link_index_materialize(
        request_overlay.module_index, &plan_options, &environment,
        loomc_link_request_module_name(linker, options), &materialization);
    status = loomc_link_translate_operation_status(
        result, before_diagnostics, loomc_make_cstring_view("LINK/REQUEST"),
        operation_status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_link_request_map_target_roots(
        &materialization.product, request_overlay.root_symbol_ordinals,
        root_count, target_root_symbol_ids);
  }

  loomc_string_view_t output_identifier = loomc_string_view_empty();
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_status_from_iree(loomc_link_request_make_source_identifier(
        loomc_link_request_module_name(linker, options), &scratch_arena,
        &output_identifier));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loomc_module_symbol_projection_t projection = {
        .module_symbol_ids = target_root_symbol_ids,
        .bytecode_symbol_ordinals = bytecode_root_symbol_ordinals,
        .count = root_count,
    };
    status = loomc_module_serialize_internal_bytecode_to_source(
        linker->context, materialization.product.module, output_identifier,
        &projection, allocator, &output_source);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      output_roots[i] = (loomc_request_root_t){
          .module_ordinal = 0,
          .symbol_ordinal = bytecode_root_symbol_ordinals[i],
      };
    }
    status = loomc_request_create_take_source(
        loomc_request_product_descriptor(input_request), &output_source,
        output_roots, root_count, loomc_request_bindings(input_request),
        loomc_request_binding_count(input_request), allocator, &output_request);
  }

  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_request = output_request;
      output_request = NULL;
    }
    *out_result = result;
    result = NULL;
  }

  loomc_request_release(output_request);
  loomc_source_release(output_source);
  loom_link_index_materialization_deinitialize(&materialization);
  loomc_request_index_overlay_deinitialize(&request_overlay);
  iree_arena_deinitialize(&scratch_arena);
  loomc_result_release(result);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
