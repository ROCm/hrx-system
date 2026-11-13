// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "config.h"

#include <string.h>

#include "diagnostic.h"
#include "loom/config/application.h"
#include "loomc/iree.h"

iree_status_t loomc_config_binding_list_append(
    loomc_config_binding_list_t* list, iree_string_view_t key,
    iree_string_view_t value, iree_arena_allocator_t* arena) {
  iree_host_size_t allocation_size = sizeof(loomc_config_binding_record_t);
  if (!iree_host_size_checked_add(allocation_size, key.size,
                                  &allocation_size) ||
      !iree_host_size_checked_add(allocation_size, value.size,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "applied config binding is too large");
  }
  loomc_config_binding_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, allocation_size, (void**)&record));
  char* storage = (char*)(record + 1);
  memcpy(storage, key.data, key.size);
  memcpy(storage + key.size, value.data, value.size);
  *record = (loomc_config_binding_record_t){
      .key = iree_make_string_view(storage, key.size),
      .value = iree_make_string_view(storage + key.size, value.size),
  };
  if (list->tail) {
    list->tail->next = record;
  } else {
    list->head = record;
  }
  list->tail = record;
  return iree_ok_status();
}

loomc_status_t loomc_config_require_resolved_module(
    const loom_module_t* module, loom_source_resolver_t source_resolver,
    loomc_result_t* result) {
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(result, /*source=*/NULL, module,
                                      source_resolver, LOOM_EMITTER_PASS,
                                      /*text_print_options=*/NULL, &capture);
  const iree_diagnostic_emitter_t emitter = {
      .fn = loomc_diagnostic_capture_emission,
      .user_data = &capture,
  };
  loom_config_resolution_result_t resolution = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_config_require_resolved_module(module, emitter, &resolution)));
  if (resolution.unresolved_count != 0) {
    loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
  }
  return loomc_ok_status();
}

loomc_status_t loomc_config_validate_policy_flags(
    loomc_config_policy_flags_t flags) {
  const loomc_config_policy_flags_t known_flags =
      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  if ((flags & ~known_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "config policy contains unknown flag bits");
  }
  return loomc_ok_status();
}

static bool loomc_config_resolve_source(void* user_data,
                                        const loom_module_t* module,
                                        loom_location_id_t location,
                                        loom_source_range_t* out_range) {
  const loomc_config_apply_module_options_t* options = user_data;
  const loom_source_resolver_t resolver = module == options->config_module
                                              ? options->config_source_resolver
                                              : options->target_source_resolver;
  return loom_source_resolve(resolver, module, location, out_range);
}

loomc_status_t loomc_config_apply_module(
    const loomc_config_apply_module_options_t* options,
    loomc_config_application_result_t* out_application_result) {
  *out_application_result = (loomc_config_application_result_t){0};

  loom_config_application_result_t materialize_result = {0};
  loomc_status_t status = loomc_ok_status();
  if (options->config_module != NULL) {
    status = loomc_result_verify_loom_module(options->context,
                                             options->config_module,
                                             options->config_source_resolver,
                                             /*max_errors=*/0, options->result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      options->config_module != NULL) {
    loomc_diagnostic_capture_t capture;
    const loom_source_resolver_t resolver = {
        .fn = loomc_config_resolve_source,
        .user_data = (void*)options,
    };
    loomc_diagnostic_capture_initialize(options->result, NULL,
                                        options->target_module, resolver,
                                        LOOM_EMITTER_PASS, NULL, &capture);
    const iree_diagnostic_emitter_t emitter = {
        .fn = loomc_diagnostic_capture_emission,
        .user_data = &capture,
    };
    status = loomc_status_from_iree(loom_config_overlay_module(
        options->target_module, options->config_module,
        options->applied_value_sink, emitter, options->block_pool,
        &materialize_result));
    if (materialize_result.error_count) {
      loomc_result_set_status_code(options->result,
                                   LOOMC_STATUS_FAILED_PRECONDITION);
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      materialize_result.materialized_count != 0) {
    status = loomc_result_verify_loom_module(options->context,
                                             options->target_module,
                                             options->target_source_resolver,
                                             /*max_errors=*/0, options->result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(options->result) &&
      iree_any_bit_set(options->policy_flags,
                       LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED)) {
    status = loomc_config_require_resolved_module(
        options->target_module, options->target_source_resolver,
        options->result);
  }
  if (loomc_status_is_ok(status)) {
    *out_application_result = (loomc_config_application_result_t){
        .materialized_count = materialize_result.materialized_count,
        .ignored_count = materialize_result.ignored_count,
    };
  }
  return status;
}
