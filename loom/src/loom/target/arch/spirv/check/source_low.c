// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/check/source_low.h"

#include "loom/target/arch/spirv/check/profile.h"
#include "loom/tools/loom-check/source_low.h"

typedef struct loom_spirv_check_source_low_request_t {
  // Control-flow representation selected for source lowering.
  loomc_target_control_flow_lowering_t control_flow_lowering;
  // Optional source function selected for target specialization.
  iree_string_view_t function_name;
  // Complete family:selector profile specification.
  iree_string_view_t target_specification;
  // Public feature enum names; a leading '-' records an unavailable feature.
  iree_string_view_t features;
  // Public numeric limit enum names paired with unsigned values.
  iree_string_view_t limits;
} loom_spirv_check_source_low_request_t;

static iree_status_t loom_spirv_check_source_low_parse(
    iree_string_view_t options,
    loom_spirv_check_source_low_request_t* out_request) {
  *out_request = (loom_spirv_check_source_low_request_t){
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };
  bool has_control_flow = false;
  options = iree_string_view_trim(options);
  while (!iree_string_view_is_empty(options)) {
    iree_string_view_t token;
    iree_string_view_split(options, ' ', &token, &options);
    options = iree_string_view_trim(options);
    if (iree_string_view_consume_prefix(&token, IREE_SV("@"))) {
      if (iree_string_view_is_empty(token) ||
          !iree_string_view_is_empty(out_request->function_name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "spirv-low expects at most one @function");
      }
      out_request->function_name = token;
      continue;
    }
    iree_string_view_t name;
    iree_string_view_t value;
    iree_string_view_split(token, '=', &name, &value);
    if (iree_string_view_equal(name, IREE_SV("features"))) {
      if (!iree_string_view_is_empty(out_request->features) ||
          iree_string_view_is_empty(value) ||
          iree_string_view_ends_with(value, IREE_SV(","))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "spirv-low requires one nonempty features list");
      }
      out_request->features = value;
    } else if (iree_string_view_equal(name, IREE_SV("limits"))) {
      if (!iree_string_view_is_empty(out_request->limits) ||
          iree_string_view_is_empty(value) ||
          iree_string_view_ends_with(value, IREE_SV(","))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "spirv-low requires one nonempty limits list");
      }
      out_request->limits = value;
    } else if (iree_string_view_equal(name, IREE_SV("target"))) {
      if (!iree_string_view_is_empty(out_request->target_specification) ||
          iree_string_view_is_empty(value)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "spirv-low requires one nonempty target");
      }
      out_request->target_specification = value;
    } else if (iree_string_view_equal(name, IREE_SV("control-flow"))) {
      if (has_control_flow) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate spirv-low option 'control-flow'");
      }
      has_control_flow = true;
      if (iree_string_view_equal(value, IREE_SV("cfg"))) {
        out_request->control_flow_lowering =
            LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
      } else if (iree_string_view_equal(value, IREE_SV("structured-low"))) {
        out_request->control_flow_lowering =
            LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "spirv-low control-flow expects cfg or "
                                "structured-low");
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown spirv-low option '%.*s'", (int)name.size,
                              name.data);
    }
  }
  if ((!iree_string_view_is_empty(out_request->function_name) ||
       !iree_string_view_is_empty(out_request->features) ||
       !iree_string_view_is_empty(out_request->limits)) &&
      iree_string_view_is_empty(out_request->target_specification)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "spirv-low @function, features, or limits requires "
                            "target=family:selector");
  }
  return iree_ok_status();
}

static bool loom_spirv_loom_check_lower_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("spirv-low"));
}

static iree_status_t loom_spirv_loom_check_lower_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  loom_spirv_check_source_low_request_t parsed;
  IREE_RETURN_IF_ERROR(
      loom_spirv_check_source_low_parse(request->target_options, &parsed));
  loomc_target_profile_t* target_profile = NULL;
  iree_status_t status = loom_spirv_check_profile_select(
      request, parsed.target_specification, parsed.features, parsed.limits,
      &target_profile);
  const loom_check_source_low_request_t source_request = {
      .output = LOOM_CHECK_EMIT_SOURCE_LOW_OUTPUT_LOW,
      .control_flow_lowering = parsed.control_flow_lowering,
      .function_name = parsed.function_name,
  };
  if (iree_status_is_ok(status) &&
      (iree_string_view_is_empty(parsed.target_specification) ||
       target_profile != NULL)) {
    status =
        loom_check_source_low_execute(request, &source_request, target_profile);
  }
  loomc_target_profile_release(target_profile);
  return status;
}

static iree_status_t loom_spirv_loom_check_lower_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "spirv-low");
}

const loom_check_emit_provider_t loom_spirv_loom_check_lower_provider = {
    .name = IREE_SVL("spirv-low"),
    .match = loom_spirv_loom_check_lower_provider_matches,
    .execute = loom_spirv_loom_check_lower_provider_execute,
    .append_names = loom_spirv_loom_check_lower_provider_append_names,
};
