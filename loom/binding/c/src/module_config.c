// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/module_config.h"

#include "config.h"
#include "context.h"
#include "loom/config/schema.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "module.h"
#include "option_chain.h"
#include "result.h"
#include "workspace.h"

loomc_status_t loomc_module_apply_config(
    loomc_module_t* module, loomc_workspace_t* workspace,
    const loomc_module_config_options_t* options, loomc_allocator_t allocator,
    loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (module == NULL || workspace == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and workspace must not be NULL");
  }
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module config options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module config options structure_size is too small");
    }
    loomc_option_chain_t option_chain = {0};
    LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
        options->next, /*allowed_options=*/0, &option_chain));
    LOOMC_RETURN_IF_ERROR(loomc_config_validate_text_options(&options->config));
  }
  loom_module_t* internal_module = loomc_module_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  loomc_context_t* context = loomc_module_context(module);
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(context), allocator,
      &result));
  loomc_status_t status = loomc_module_admit(
      module, loomc_context_target_environment(context), result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loomc_config_options_t* config = options ? &options->config : NULL;
    const bool has_bindings =
        config != NULL && (config->binding_count != 0 ||
                           !loomc_string_view_is_empty(config->json_object));
    const loomc_config_apply_text_to_module_options_t apply_options = {
        .context = context,
        .config = config,
        .module = internal_module,
        .source_resolver = loomc_module_source_resolver(module),
        .applied_value_sink = loomc_module_config_applied_value_sink(module),
        .result = result,
        .block_pool = loomc_workspace_block_pool(workspace),
        .allocator = allocator,
    };
    status = loomc_config_apply_text_to_module(&apply_options);
    if (has_bindings) {
      loomc_module_invalidate_target_verification(module);
      if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
        status = loomc_module_admit(
            module, loomc_context_target_environment(context), result);
      }
      if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
        loomc_module_invalidate_verification(module);
        loomc_module_invalidate_compilation(module);
      }
    }
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

static loomc_status_t loomc_module_add_config_schema(
    const loomc_module_t* module, loomc_result_t* result) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loomc_status_t status = loomc_status_from_iree(loom_config_format_schema_json(
      loomc_module_const_loom_module(module), &stream));
  if (loomc_status_is_ok(status)) {
    iree_host_size_t length = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_REPORT,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_JSON),
        loomc_make_cstring_view("config-schema.json"),
        loomc_make_byte_span(storage, length));
    if (!loomc_status_is_ok(status)) {
      loomc_allocator_free(allocator, storage);
    }
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

loomc_status_t loomc_module_query_config_schema(loomc_module_t* module,
                                                loomc_allocator_t allocator,
                                                loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }
  if (loomc_module_const_loom_module(module) == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  loomc_context_t* context = loomc_module_context(module);
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(context), allocator,
      &result));
  loomc_status_t status = loomc_module_admit(
      module, loomc_context_target_environment(context), result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_module_add_config_schema(module, result);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}
