// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/check/loom_check.h"

#include <stdint.h>

#include "loom/target/tool/spirv.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/requirements.h"

typedef struct loom_spirv_loom_check_emit_request_t {
  // Whether to lower source IR to Low before SPIR-V emission.
  bool lower_source_to_low;
  // Whether to validate the emitted binary before disassembly.
  bool validate;
  // Source-to-low control-flow shape when lowering source IR.
  loomc_target_control_flow_lowering_t control_flow_lowering;
  // Optional source function selected for target specialization.
  iree_string_view_t function_name;
  // Complete family:selector target spelling when the TARGET flag is present.
  iree_string_view_t target_specification;
} loom_spirv_loom_check_emit_request_t;

static bool loom_spirv_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("spirv-dis"));
}

static iree_status_t loom_spirv_loom_check_parse_emit_request(
    iree_string_view_t target_options,
    loom_spirv_loom_check_emit_request_t* out_request) {
  *out_request = (loom_spirv_loom_check_emit_request_t){
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };

  enum {
    LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_INPUT = 1u << 0,
    LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_CONTROL_FLOW = 1u << 1,
    LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_TARGET = 1u << 2,
    LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_VALIDATE = 1u << 3,
  };
  uint32_t parse_options = 0;
  target_options = iree_string_view_trim(target_options);
  while (!iree_string_view_is_empty(target_options)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(target_options, ' ', &token, &target_options);
    token = iree_string_view_trim(token);
    target_options = iree_string_view_trim(target_options);
    if (iree_string_view_equal(token, IREE_SV("validate"))) {
      if (iree_any_bit_set(parse_options,
                           LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_VALIDATE)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate spirv-dis option 'validate'");
      }
      parse_options |= LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_VALIDATE;
      out_request->validate = true;
      continue;
    }
    if (iree_string_view_starts_with(token, IREE_SV("@"))) {
      if (token.size == 1 ||
          !iree_string_view_is_empty(out_request->function_name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "spirv-dis expects at most one @function");
      }
      out_request->function_name =
          iree_string_view_substr(token, 1, IREE_HOST_SIZE_MAX);
      continue;
    }
    iree_string_view_t option_name = iree_string_view_empty();
    iree_string_view_t option_value = iree_string_view_empty();
    iree_string_view_split(token, '=', &option_name, &option_value);
    if (iree_string_view_equal(option_name, IREE_SV("target"))) {
      if (iree_any_bit_set(parse_options,
                           LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_TARGET)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate spirv-dis option 'target'");
      }
      out_request->target_specification = option_value;
      parse_options |= LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_TARGET;
      continue;
    }
    if (iree_string_view_equal(option_name, IREE_SV("input"))) {
      if (iree_any_bit_set(parse_options,
                           LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_INPUT)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate spirv-dis option 'input'");
      }
      if (iree_string_view_equal(option_value, IREE_SV("low"))) {
        out_request->lower_source_to_low = false;
      } else if (iree_string_view_equal(option_value, IREE_SV("source-low"))) {
        out_request->lower_source_to_low = true;
      } else {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "spirv-dis option 'input' expected 'low' or 'source-low', got "
            "'%.*s'",
            (int)option_value.size, option_value.data);
      }
      parse_options |= LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_INPUT;
      continue;
    }
    if (iree_string_view_equal(option_name, IREE_SV("control-flow"))) {
      if (iree_any_bit_set(parse_options,
                           LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_CONTROL_FLOW)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate spirv-dis option 'control-flow'");
      }
      if (iree_string_view_equal(option_value, IREE_SV("cfg"))) {
        out_request->control_flow_lowering =
            LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
      } else if (iree_string_view_equal(option_value,
                                        IREE_SV("structured-low"))) {
        out_request->control_flow_lowering =
            LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
      } else {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "spirv-dis option 'control-flow' expected 'cfg' or "
            "'structured-low', got '%.*s'",
            (int)option_value.size, option_value.data);
      }
      parse_options |= LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_CONTROL_FLOW;
      continue;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown spirv-dis option '%.*s'", (int)token.size,
                            token.data);
  }
  if (!iree_string_view_is_empty(out_request->function_name) &&
      !iree_any_bit_set(parse_options,
                        LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_TARGET)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "spirv-dis @function requires target=family:selector");
  }
  if (!out_request->lower_source_to_low &&
      iree_any_bit_set(parse_options,
                       LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_CONTROL_FLOW |
                           LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_TARGET)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "spirv-dis target and control-flow options require input=source-low");
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_emit_provider_check_requirements(
    const loom_check_emit_provider_t* provider,
    const loom_test_case_t* test_case, iree_string_view_t target_options,
    loom_check_result_t* result, bool* out_continue_execution) {
  IREE_RETURN_IF_ERROR(loom_check_require_declared_requirement(
      test_case, IREE_SV("spirv-dis"), result, out_continue_execution));
  if (!*out_continue_execution) {
    return iree_ok_status();
  }

  loom_spirv_loom_check_emit_request_t request = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_loom_check_parse_emit_request(target_options, &request));
  if (request.validate) {
    IREE_RETURN_IF_ERROR(loom_check_require_declared_requirement(
        test_case, IREE_SV("spirv-val"), result, out_continue_execution));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_strip_disassembly_comments(
    iree_string_view_t input, iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    if (iree_string_view_starts_with(trimmed, IREE_SV(";"))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  loom_spirv_loom_check_emit_request_t emit_request = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_loom_check_parse_emit_request(
      request->target_options, &emit_request));

  const loom_check_compile_artifact_options_t compile_options = {
      .artifact_format = IREE_SV("spirv"),
      .root = emit_request.function_name,
      .target = emit_request.target_specification,
      .lower_source_to_low = emit_request.lower_source_to_low,
      .control_flow_lowering = emit_request.control_flow_lowering,
  };
  loomc_source_t* artifact_source = NULL;
  iree_status_t status =
      loom_check_compile_artifact(request, &compile_options, &artifact_source);

  const loomc_byte_span_t source_contents =
      loomc_source_contents(artifact_source);
  const iree_const_byte_span_t contents = iree_make_const_byte_span(
      source_contents.data, source_contents.data_length);

  loom_spirv_toolchain_t toolchain;
  loom_spirv_toolchain_initialize_from_environment(&toolchain);
  if (iree_status_is_ok(status) && artifact_source != NULL &&
      emit_request.validate) {
    status = loom_spirv_tool_validate_binary(&toolchain, contents,
                                             request->host_allocator);
  }

  loom_tool_output_t disassembly = {0};
  if (iree_status_is_ok(status) && artifact_source != NULL) {
    status = loom_spirv_tool_disassemble_binary(
        &toolchain, contents, request->host_allocator, &disassembly);
  }
  if (iree_status_is_ok(status) && artifact_source != NULL) {
    status = loom_spirv_loom_check_strip_disassembly_comments(
        iree_make_string_view(disassembly.data, disassembly.length),
        &request->result->actual_output);
  }

  loom_tool_output_deinitialize(&disassembly, request->host_allocator);
  loomc_source_release(artifact_source);
  return status;
}

static iree_status_t loom_spirv_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "spirv-dis");
}

static bool loom_spirv_loom_check_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  return iree_string_view_equal(requirement, IREE_SV("spirv-as")) ||
         iree_string_view_equal(requirement, IREE_SV("spirv-dis")) ||
         iree_string_view_equal(requirement, IREE_SV("spirv-val"));
}

static iree_status_t loom_spirv_loom_check_query_spirv_tool(
    loom_spirv_tool_kind_t tool_kind, iree_allocator_t allocator) {
  loom_spirv_toolchain_t toolchain;
  loom_spirv_toolchain_initialize_from_environment(&toolchain);
  loom_tool_output_t version_text = {0};
  iree_status_t status = loom_spirv_tool_query_version(
      &toolchain, tool_kind, allocator, &version_text);
  loom_tool_output_deinitialize(&version_text, allocator);
  return status;
}

static iree_status_t loom_spirv_loom_check_requirement_provider_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  if (iree_string_view_equal(requirement, IREE_SV("spirv-as"))) {
    return loom_spirv_loom_check_query_spirv_tool(LOOM_SPIRV_TOOL_SPIRV_AS,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("spirv-dis"))) {
    return loom_spirv_loom_check_query_spirv_tool(LOOM_SPIRV_TOOL_SPIRV_DIS,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("spirv-val"))) {
    return loom_spirv_loom_check_query_spirv_tool(LOOM_SPIRV_TOOL_SPIRV_VAL,
                                                  allocator);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown SPIR-V loom-check requirement '%.*s'",
                          (int)requirement.size, requirement.data);
}

static iree_status_t loom_spirv_loom_check_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder,
                                            "spirv-as, spirv-dis, spirv-val");
}

const loom_check_emit_provider_t loom_spirv_loom_check_emit_provider = {
    .name = IREE_SVL("spirv"),
    .match = loom_spirv_loom_check_emit_provider_matches,
    .check_requirements =
        loom_spirv_loom_check_emit_provider_check_requirements,
    .execute = loom_spirv_loom_check_emit_provider_execute,
    .append_names = loom_spirv_loom_check_emit_provider_append_names,
};

const loom_check_requirement_provider_t
    loom_spirv_loom_check_requirement_provider = {
        .name = IREE_SVL("spirv"),
        .match = loom_spirv_loom_check_requirement_provider_matches,
        .query = loom_spirv_loom_check_requirement_provider_query,
        .append_names = loom_spirv_loom_check_requirement_provider_append_names,
};
