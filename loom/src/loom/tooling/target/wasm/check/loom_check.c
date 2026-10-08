// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/wasm/check/loom_check.h"

#include "loom/target/tool/wasm.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/requirements.h"

static bool loom_wasm_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("wasm-dis"));
}

static iree_status_t loom_wasm_loom_check_emit_provider_check_requirements(
    const loom_check_emit_provider_t* provider,
    const loom_test_case_t* test_case, iree_string_view_t target_options,
    loom_check_result_t* result, bool* out_continue_execution) {
  IREE_RETURN_IF_ERROR(loom_check_require_declared_requirement(
      test_case, IREE_SV("wasm-objdump"), result, out_continue_execution));
  if (!*out_continue_execution) {
    return iree_ok_status();
  }
  if (!iree_string_view_is_empty(target_options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wasm-dis does not accept options");
  }
  return iree_ok_status();
}

static bool loom_wasm_loom_check_is_hex_digit(char value) {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

static iree_string_view_t loom_wasm_loom_check_trim_trailing_ascii_whitespace(
    iree_string_view_t value) {
  while (value.size > 0) {
    char character = value.data[value.size - 1];
    if (character != ' ' && character != '\t' && character != '\r' &&
        character != '\n') {
      break;
    }
    --value.size;
  }
  return value;
}

static iree_string_view_t loom_wasm_loom_check_strip_objdump_byte_marker(
    iree_string_view_t line, bool* out_instruction) {
  *out_instruction = false;
  iree_host_size_t position = 0;
  while (position < line.size && line.data[position] == ' ') {
    ++position;
  }

  const iree_host_size_t hex_start = position;
  while (position < line.size &&
         loom_wasm_loom_check_is_hex_digit(line.data[position])) {
    ++position;
  }
  if (position == hex_start) {
    return line;
  }

  if (position < line.size && line.data[position] == ':') {
    ++position;
    while (position < line.size &&
           (line.data[position] == ' ' || line.data[position] == '\t')) {
      ++position;
    }
    *out_instruction = true;
    return iree_string_view_substr(line, position, IREE_HOST_SIZE_MAX);
  }

  if (position < line.size && line.data[position] == ' ') {
    while (position < line.size && line.data[position] == ' ') {
      ++position;
    }
    if (position < line.size && line.data[position] == '<') {
      return iree_string_view_substr(line, position, IREE_HOST_SIZE_MAX);
    }
  }

  return line;
}

static iree_status_t loom_wasm_loom_check_append_objdump_line(
    iree_string_view_t line, iree_string_builder_t* output) {
  bool instruction = false;
  line = loom_wasm_loom_check_strip_objdump_byte_marker(line, &instruction);
  if (instruction) {
    const iree_host_size_t comment_position =
        iree_string_view_find(line, IREE_SV("#"), 0);
    if (comment_position != IREE_STRING_VIEW_NPOS) {
      line = iree_string_view_substr(line, 0, comment_position);
    }
  }
  line = loom_wasm_loom_check_trim_trailing_ascii_whitespace(line);
  if (instruction && !iree_string_view_is_empty(line)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "  "));
  }
  iree_host_size_t chunk_start = 0;
  for (iree_host_size_t i = 0; i < line.size; ++i) {
    if (line.data[i] != '\t') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output, iree_string_view_substr(line, chunk_start, i - chunk_start)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "  "));
    chunk_start = i + 1;
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      output, iree_string_view_substr(line, chunk_start, IREE_HOST_SIZE_MAX)));
  return iree_string_builder_append_cstring(output, "\n");
}

static iree_status_t loom_wasm_loom_check_strip_objdump_preamble(
    iree_string_view_t input, iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  bool found_disassembly = false;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    if (!found_disassembly) {
      found_disassembly = iree_string_view_starts_with(
          line, IREE_SV("Disassembly of section "));
      if (!found_disassembly) {
        continue;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_wasm_loom_check_append_objdump_line(line, output));
  }
  if (!found_disassembly) {
    return iree_string_builder_append_string(output, input);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  const loom_check_compile_artifact_options_t compile_options = {
      .artifact_format = IREE_SV("wasm-binary"),
  };
  loomc_source_t* artifact_source = NULL;
  iree_status_t status =
      loom_check_compile_artifact(request, &compile_options, &artifact_source);
  const loomc_byte_span_t source_contents =
      loomc_source_contents(artifact_source);
  const iree_const_byte_span_t contents = iree_make_const_byte_span(
      source_contents.data, source_contents.data_length);

  loom_wasm_toolchain_t toolchain;
  loom_wasm_toolchain_initialize_from_environment(&toolchain);
  loom_tool_output_t disassembly = {0};
  if (iree_status_is_ok(status) && artifact_source != NULL) {
    status = loom_wasm_tool_disassemble_binary(
        &toolchain, contents, request->host_allocator, &disassembly);
  }
  if (iree_status_is_ok(status) && artifact_source != NULL) {
    status = loom_wasm_loom_check_strip_objdump_preamble(
        iree_make_string_view(disassembly.data, disassembly.length),
        &request->result->actual_output);
  }

  loom_tool_output_deinitialize(&disassembly, request->host_allocator);
  loomc_source_release(artifact_source);
  return status;
}

static iree_status_t loom_wasm_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "wasm-dis");
}

static bool loom_wasm_loom_check_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  return iree_string_view_equal(requirement, IREE_SV("wasm-objdump"));
}

static iree_status_t loom_wasm_loom_check_requirement_provider_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  if (!iree_string_view_equal(requirement, IREE_SV("wasm-objdump"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Wasm loom-check requirement '%.*s'",
                            (int)requirement.size, requirement.data);
  }
  loom_wasm_toolchain_t toolchain;
  loom_wasm_toolchain_initialize_from_environment(&toolchain);
  loom_tool_output_t version_text = {0};
  iree_status_t status = loom_wasm_tool_query_version(
      &toolchain, LOOM_WASM_TOOL_LLVM_OBJDUMP, allocator, &version_text);
  loom_tool_output_deinitialize(&version_text, allocator);
  return status;
}

static iree_status_t loom_wasm_loom_check_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "wasm-objdump");
}

const loom_check_emit_provider_t loom_wasm_loom_check_emit_provider = {
    .name = IREE_SVL("wasm"),
    .match = loom_wasm_loom_check_emit_provider_matches,
    .check_requirements = loom_wasm_loom_check_emit_provider_check_requirements,
    .execute = loom_wasm_loom_check_emit_provider_execute,
    .append_names = loom_wasm_loom_check_emit_provider_append_names,
};

const loom_check_requirement_provider_t
    loom_wasm_loom_check_requirement_provider = {
        .name = IREE_SVL("wasm"),
        .match = loom_wasm_loom_check_requirement_provider_matches,
        .query = loom_wasm_loom_check_requirement_provider_query,
        .append_names = loom_wasm_loom_check_requirement_provider_append_names,
};
