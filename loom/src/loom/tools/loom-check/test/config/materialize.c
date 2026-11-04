// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "iree/base/api.h"
#include "loom/config/text.h"
#include "loom/config/text_binding.h"
#include "loom/format/text/printer.h"
#include "loom/tools/loom-check/test/config/providers.h"
#include "loom/util/stream.h"

static bool loom_check_test_config_materialize_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("config-materialize"));
}

static iree_status_t loom_check_test_config_materialize_record_value(
    void* user_data, const loom_config_applied_value_t* applied_value) {
  iree_string_builder_t* output = (iree_string_builder_t*)user_data;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "applied @%.*s = ", (int)applied_value->key.size,
      applied_value->key.data));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_text_print_attribute(
      &applied_value->value, applied_value->module, &stream));
  return iree_string_builder_append_cstring(output, "\n");
}

static iree_status_t loom_check_test_config_materialize_record_status(
    iree_status_t application_status, iree_string_builder_t* output) {
  const iree_status_code_t status_code = iree_status_code(application_status);
  iree_status_t output_status = iree_string_builder_append_format(
      output, "status: %s\n", iree_status_code_string(status_code));
  iree_status_free(application_status);
  return output_status;
}

static iree_status_t loom_check_test_config_materialize_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module) {
  (void)provider;
  iree_string_view_t key = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  if (iree_string_view_split(request->target_options, '=', &key, &value) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config-materialize requires one key=value binding");
  }

  loom_config_text_binding_set_t binding_set;
  loom_config_text_binding_set_initialize(request->host_allocator,
                                          &binding_set);
  iree_status_t status =
      loom_config_text_binding_set_append(&binding_set, key, value);
  loom_config_application_result_t application_result = {0};
  iree_status_t application_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    loom_config_text_materialize_options_t options;
    loom_config_text_materialize_options_initialize(&options);
    options.binding_set = &binding_set;
    options.applied_value_sink = (loom_config_applied_value_sink_t){
        .fn = loom_check_test_config_materialize_record_value,
        .user_data = &request->result->actual_output,
    };
    application_status = loom_config_text_materialize_module(
        native_module->module, &options, request->block_pool,
        &application_result);
  }
  if (iree_status_is_ok(status)) {
    const bool application_succeeded = iree_status_is_ok(application_status);
    status = loom_check_test_config_materialize_record_status(
        application_status, &request->result->actual_output);
    application_status = iree_ok_status();
    if (iree_status_is_ok(status) && application_succeeded) {
      status = iree_string_builder_append_format(
          &request->result->actual_output,
          "materialized: %" PRIhsz "\nignored: %" PRIhsz "\n",
          application_result.materialized_count,
          application_result.ignored_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_print_module_to_builder(native_module->module,
                                               &request->result->actual_output,
                                               LOOM_TEXT_PRINT_DEFAULT);
  }
  iree_status_free(application_status);
  loom_config_text_binding_set_deinitialize(&binding_set);
  return status;
}

static iree_status_t loom_check_test_config_materialize_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "config-materialize");
}

const loom_check_test_native_emit_provider_t
    loom_check_test_config_materialize_provider = {
        .base =
            {
                .name = IREE_SVL("config-materialize"),
                .match = loom_check_test_config_materialize_matches,
                .execute = loom_check_test_native_emit_provider_execute,
                .append_names = loom_check_test_config_materialize_append_names,
            },
        .execute_native = loom_check_test_config_materialize_execute,
};
