// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/config/schema.h"

#include "iree/base/api.h"
#include "loom/tools/loom-check/test/config/providers.h"
#include "loom/util/stream.h"

static bool loom_check_test_config_schema_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("config-schema"));
}

static iree_status_t loom_check_test_config_schema_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module) {
  (void)provider;
  if (!iree_string_view_is_empty(request->target_options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config-schema accepts no options");
  }
  loom_output_stream_t output_stream;
  loom_output_stream_for_builder(&request->result->actual_output,
                                 &output_stream);
  return loom_config_format_schema_json(native_module->module, &output_stream);
}

static iree_status_t loom_check_test_config_schema_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "config-schema");
}

const loom_check_emit_provider_t loom_check_test_config_schema_provider = {
    .name = IREE_SVL("config-schema"),
    .match = loom_check_test_config_schema_matches,
    .execute_native = loom_check_test_config_schema_execute,
    .append_names = loom_check_test_config_schema_append_names,
};
