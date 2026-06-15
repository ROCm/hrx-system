// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/check/loom_check.h"

#include <stdint.h>

#include "loom/target/emit/spirv/module_emitter.h"
#include "loom/target/entry_selection.h"
#include "loom/target/tool/spirv.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tools/loom-check/diagnostics.h"

typedef enum loom_spirv_loom_check_input_e {
  LOOM_SPIRV_LOOM_CHECK_INPUT_LOW = 0,
  LOOM_SPIRV_LOOM_CHECK_INPUT_SOURCE_LOW = 1,
} loom_spirv_loom_check_input_t;

typedef enum loom_spirv_loom_check_emit_flag_bits_e {
  LOOM_SPIRV_LOOM_CHECK_EMIT_FLAG_NONE = 0u,
  LOOM_SPIRV_LOOM_CHECK_EMIT_FLAG_VALIDATE = 1u << 0,
} loom_spirv_loom_check_emit_flag_bits_t;
typedef uint32_t loom_spirv_loom_check_emit_flags_t;

typedef struct loom_spirv_loom_check_emit_request_t {
  // Input form consumed by SPIR-V emission.
  loom_spirv_loom_check_input_t input;
  // Source-to-low control-flow shape when |input| is source-low.
  loom_target_control_flow_lowering_t control_flow_lowering;
  // Additional emit behavior requested by the RUN line.
  loom_spirv_loom_check_emit_flags_t flags;
} loom_spirv_loom_check_emit_request_t;

static bool loom_spirv_loom_check_case_has_requirement(
    const loom_check_case_t* test_case, iree_string_view_t requirement) {
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    if (iree_string_view_equal(test_case->requirements[i], requirement)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_spirv_loom_check_fail_missing_requirement(
    iree_string_view_t emit_target, iree_string_view_t requirement,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  return iree_string_builder_append_format(
      &result->detail,
      "RUN: emit %.*s requires '// REQUIRES: %.*s'; external tool "
      "dependencies must be declared even when they are available\n",
      (int)emit_target.size, emit_target.data, (int)requirement.size,
      requirement.data);
}

static iree_status_t loom_spirv_loom_check_require_declared_requirement(
    const loom_check_case_t* test_case, iree_string_view_t requirement,
    loom_check_result_t* result, bool* out_continue_execution) {
  if (loom_spirv_loom_check_case_has_requirement(test_case, requirement)) {
    return iree_ok_status();
  }
  *out_continue_execution = false;
  return loom_spirv_loom_check_fail_missing_requirement(test_case->emit_target,
                                                        requirement, result);
}

static bool loom_spirv_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("spirv-dis"));
}

static iree_status_t loom_spirv_loom_check_consume_token(
    iree_string_view_t* remaining, iree_string_view_t* out_token) {
  iree_string_view_t text = iree_string_view_trim(*remaining);
  iree_string_view_t token = iree_string_view_empty();
  iree_string_view_t rest = iree_string_view_empty();
  iree_string_view_split(text, ' ', &token, &rest);
  token = iree_string_view_trim(token);
  *remaining = iree_string_view_trim(rest);
  *out_token = token;
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_parse_emit_request(
    iree_string_view_t target_options,
    loom_spirv_loom_check_emit_request_t* out_request) {
  *out_request = (loom_spirv_loom_check_emit_request_t){
      .input = LOOM_SPIRV_LOOM_CHECK_INPUT_LOW,
      .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .flags = LOOM_SPIRV_LOOM_CHECK_EMIT_FLAG_NONE,
  };

  enum {
    LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_INPUT = 1u << 0,
    LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_CONTROL_FLOW = 1u << 1,
  };
  uint32_t parse_options = 0;
  iree_string_view_t token = iree_string_view_empty();
  while (!iree_string_view_is_empty(target_options)) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_loom_check_consume_token(&target_options, &token));
    if (iree_string_view_is_empty(token)) {
      continue;
    }
    if (iree_string_view_equal(token, IREE_SV("validate"))) {
      out_request->flags |= LOOM_SPIRV_LOOM_CHECK_EMIT_FLAG_VALIDATE;
      continue;
    }
    if (iree_string_view_starts_with(token, IREE_SV("@"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "spirv-dis emits the whole split test-case module; RUN lines must "
          "not select a function symbol");
    }
    iree_string_view_t option_name = iree_string_view_empty();
    iree_string_view_t option_value = iree_string_view_empty();
    iree_string_view_split(token, '=', &option_name, &option_value);
    option_name = iree_string_view_trim(option_name);
    option_value = iree_string_view_trim(option_value);
    if (iree_string_view_equal(option_name, IREE_SV("input"))) {
      if (iree_any_bit_set(parse_options,
                           LOOM_SPIRV_LOOM_CHECK_PARSE_OPTION_INPUT)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate spirv-dis option 'input'");
      }
      if (iree_string_view_equal(option_value, IREE_SV("low"))) {
        out_request->input = LOOM_SPIRV_LOOM_CHECK_INPUT_LOW;
      } else if (iree_string_view_equal(option_value, IREE_SV("source-low"))) {
        out_request->input = LOOM_SPIRV_LOOM_CHECK_INPUT_SOURCE_LOW;
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
            LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
      } else if (iree_string_view_equal(option_value,
                                        IREE_SV("structured-low"))) {
        out_request->control_flow_lowering =
            LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
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
  if (out_request->input == LOOM_SPIRV_LOOM_CHECK_INPUT_LOW &&
      out_request->control_flow_lowering !=
          LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "spirv-dis option 'control-flow' requires input=source-low");
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_emit_provider_check_requirements(
    const loom_check_emit_provider_t* provider,
    const loom_check_case_t* test_case, loom_check_result_t* result,
    bool* out_continue_execution) {
  iree_string_view_t emit_target =
      iree_string_view_trim(test_case->emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t target_options = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, &target_options);
  target_name = iree_string_view_trim(target_name);
  target_options = iree_string_view_trim(target_options);

  if (!iree_string_view_equal(target_name, IREE_SV("spirv-dis"))) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_spirv_loom_check_require_declared_requirement(
      test_case, IREE_SV("spirv-dis"), result, out_continue_execution));
  if (!*out_continue_execution) {
    return iree_ok_status();
  }

  loom_spirv_loom_check_emit_request_t request = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_loom_check_parse_emit_request(target_options, &request));
  if (iree_any_bit_set(request.flags,
                       LOOM_SPIRV_LOOM_CHECK_EMIT_FLAG_VALIDATE)) {
    IREE_RETURN_IF_ERROR(loom_spirv_loom_check_require_declared_requirement(
        test_case, IREE_SV("spirv-val"), result, out_continue_execution));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_spirv_loom_check_consume_line(
    iree_string_view_t* remaining) {
  iree_host_size_t newline_position =
      iree_string_view_find(*remaining, IREE_SV("\n"), 0);
  if (newline_position == IREE_STRING_VIEW_NPOS) {
    iree_string_view_t line = *remaining;
    *remaining = iree_string_view_empty();
    return line;
  }
  iree_string_view_t line =
      iree_string_view_substr(*remaining, 0, newline_position);
  *remaining = iree_string_view_substr(*remaining, newline_position + 1,
                                       IREE_HOST_SIZE_MAX);
  return line;
}

static iree_status_t loom_spirv_loom_check_strip_disassembly_comments(
    iree_string_view_t input, iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_spirv_loom_check_consume_line(&remaining);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    if (iree_string_view_starts_with(trimmed, IREE_SV(";"))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_run_source_low_pipeline(
    const loom_check_emit_provider_request_t* request,
    const loom_spirv_loom_check_emit_request_t* emit_request) {
  if (request->environment->target_environment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spirv-dis input=source-low requires a target "
                            "environment");
  }

  loom_compile_pipeline_options_t compile_options = {0};
  loom_compile_pipeline_options_initialize(&compile_options);
  compile_options.pipeline = IREE_SV("default");
  compile_options.default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW;
  compile_options.target_pipeline_options.control_flow_lowering =
      emit_request->control_flow_lowering;
  compile_options.target_environment = request->environment->target_environment;
  compile_options.low_descriptor_registry = request->low_registry;
  compile_options.diagnostic_sink =
      (loom_diagnostic_sink_t){.fn = loom_check_diagnostic_collector_sink,
                               .user_data = request->diagnostic_collector};
  compile_options.source_resolver = request->source_resolver;
  compile_options.max_errors = 20;

  loom_pass_run_result_t run_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_compile_run_pipeline(request->module, &compile_options,
                                request->case_arena->block_pool, &run_result));
  if (run_result.error_count != 0 &&
      request->diagnostic_collector->count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "source-low pipeline reported errors without diagnostics");
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_verify_low_module(
    const loom_check_emit_provider_request_t* request) {
  const loom_target_entry_options_t entry_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = request->diagnostic_collector},
      .source_resolver = request->source_resolver,
      .max_errors = 20,
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(
      request->module, &entry_options, 20, &verify_result));
  if (verify_result.error_count != 0 &&
      request->diagnostic_collector->count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "lowered module verifier reported errors without diagnostics");
  }
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_target_entry_diagnostic_emitter_t verifier_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      request->module, &entry_options, LOOM_EMITTER_VERIFIER,
      &verifier_emitter);
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(request->module);
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      request->module, request->low_registry, &verifier_emitter,
      loom_target_selection_empty(), 20,
      request->environment->low_verify_provider_list, &low_verify_scratch,
      &low_verify_result));
  if (low_verify_result.error_count != 0 &&
      request->diagnostic_collector->count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "lowered low verifier reported errors without diagnostics");
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  loom_spirv_loom_check_emit_request_t emit_request = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_loom_check_parse_emit_request(
      request->target_options, &emit_request));

  loom_check_diagnostic_emitter_capture_t capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t diagnostic_emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &capture,
  };

  if (emit_request.input == LOOM_SPIRV_LOOM_CHECK_INPUT_SOURCE_LOW) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_loom_check_run_source_low_pipeline(request, &emit_request));
    if (request->diagnostic_collector->count != 0) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_loom_check_verify_low_module(request));
  if (request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }

  loom_spirv_module_binary_t module = {0};
  iree_status_t status = loom_spirv_emit_low_module(
      request->module, &request->low_registry->registry,
      loom_target_selection_empty(), diagnostic_emitter, request->case_arena,
      /*options=*/NULL, &module, request->host_allocator);

  loom_spirv_toolchain_t toolchain;
  loom_spirv_toolchain_initialize_from_environment(&toolchain);
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(emit_request.flags,
                       LOOM_SPIRV_LOOM_CHECK_EMIT_FLAG_VALIDATE)) {
    status = loom_spirv_tool_validate_binary(
        &toolchain, loom_spirv_module_binary_byte_span(&module),
        request->host_allocator);
  }

  loom_spirv_tool_output_t disassembly = {0};
  if (iree_status_is_ok(status)) {
    status = loom_spirv_tool_disassemble_binary(
        &toolchain, loom_spirv_module_binary_byte_span(&module),
        request->host_allocator, &disassembly);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_loom_check_strip_disassembly_comments(
        iree_make_string_view(disassembly.data, disassembly.length),
        &request->result->actual_output);
  }

  loom_spirv_tool_output_deinitialize(&disassembly, request->host_allocator);
  loom_spirv_module_binary_deinitialize(&module, request->host_allocator);
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
  loom_spirv_tool_output_t version_text = {0};
  iree_status_t status = loom_spirv_tool_query_version(
      &toolchain, tool_kind, allocator, &version_text);
  loom_spirv_tool_output_deinitialize(&version_text, allocator);
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
