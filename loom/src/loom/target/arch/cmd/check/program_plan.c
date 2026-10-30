// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/check/program_plan.h"

#include "loom/target/arch/cmd/program.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loomc/iree.h"
#include "loomc/target/cmd/program.h"

typedef struct loom_cmd_program_plan_check_options_t {
  // Command program symbol names selected in caller order.
  loomc_string_view_t* root_names;
  // Number of entries in |root_names|.
  iree_host_size_t root_count;
  // Exercise source-backed request publication instead of opaque entry binding.
  bool publish_kernels;
} loom_cmd_program_plan_check_options_t;

static bool loom_cmd_program_plan_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("command-program"));
}

static iree_status_t loom_cmd_program_plan_check_parse_roots(
    iree_string_view_t roots_text, iree_arena_allocator_t* arena,
    loom_cmd_program_plan_check_options_t* options) {
  if (iree_string_view_is_empty(roots_text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program emit requires a root symbol");
  }

  iree_host_size_t root_count = 1;
  for (iree_host_size_t i = 0; i < roots_text.size; ++i) {
    if (roots_text.data[i] == ',') {
      ++root_count;
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, root_count,
                                                 sizeof(*options->root_names),
                                                 (void**)&options->root_names));

  iree_string_view_t remaining = roots_text;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    iree_string_view_t root = iree_string_view_empty();
    iree_string_view_t tail = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &root, &tail);
    if (!iree_string_view_starts_with(root, IREE_SV("@")) || root.size == 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command program root expected a symbol name, got '%.*s'",
          (int)root.size, root.data);
    }
    root = iree_string_view_substr(root, 1, IREE_HOST_SIZE_MAX);
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(
              root, iree_string_view_from_loomc(options->root_names[j]))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate command program root '@%.*s'",
                                (int)root.size, root.data);
      }
    }
    options->root_names[i] = loomc_string_view_from_iree(root);
    remaining = tail;
  }
  options->root_count = root_count;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_check_parse_options(
    const loom_check_emit_provider_request_t* request,
    loom_cmd_program_plan_check_options_t* out_options) {
  *out_options = (loom_cmd_program_plan_check_options_t){0};
  IREE_ASSERT(
      iree_string_view_equal(request->target_name, IREE_SV("command-program")));

  iree_string_view_t roots_text = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &roots_text,
                         &option_text);
  roots_text = iree_string_view_trim(roots_text);
  option_text = iree_string_view_trim(option_text);
  IREE_RETURN_IF_ERROR(loom_cmd_program_plan_check_parse_roots(
      roots_text, request->case_arena, out_options));

  if (iree_string_view_equal(option_text, IREE_SV("publish-kernels"))) {
    out_options->publish_kernels = true;
  } else if (!iree_string_view_is_empty(option_text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected command program emit option '%.*s'",
                            (int)option_text.size, option_text.data);
  }
  return iree_ok_status();
}

// Validates every executable returned by the public transaction with the
// runtime's untrusted-byte parser, independently of its textual listing.
static iree_status_t loom_cmd_program_plan_check_roundtrip_artifacts(
    const loomc_product_t* product, iree_allocator_t host_allocator) {
  const loomc_host_size_t artifact_count =
      loomc_product_artifact_count(product);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < artifact_count && iree_status_is_ok(status);
       ++i) {
    const loomc_artifact_t* artifact = loomc_product_artifact_at(product, i);
    loomc_source_t* source = NULL;
    status = iree_status_from_loomc(loomc_artifact_create_source(
        artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
        loomc_allocator_from_iree(host_allocator), &source));
    if (iree_status_is_ok(status)) {
      const loomc_byte_span_t bytes = loomc_source_contents(source);
      loom_cmd_program_t program = {0};
      status = loom_cmd_program_parse(
          iree_make_const_byte_span(bytes.data, bytes.data_length), &program);
    }
    loomc_source_release(source);
  }
  return status;
}

static loomc_status_t loom_cmd_program_plan_check_append_listing_segment(
    void* user_data, loomc_byte_span_t segment) {
  return loomc_status_from_iree(iree_string_builder_append_string(
      (iree_string_builder_t*)user_data,
      iree_make_string_view((const char*)segment.data, segment.data_length)));
}

static loomc_status_t loom_cmd_program_plan_check_release_request(
    void* user_data, loomc_request_t* request) {
  (void)user_data;
  loomc_request_release(request);
  return loomc_ok_status();
}

static iree_status_t loom_cmd_program_plan_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_cmd_program_plan_check_options_t options = {0};
  IREE_RETURN_IF_ERROR(
      loom_cmd_program_plan_check_parse_options(request, &options));

  const loomc_cmd_program_module_options_t product_options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS,
      .structure_size = sizeof(product_options),
      .provider_name = loomc_make_cstring_view("command_program_check"),
      .root_symbols = options.root_names,
      .root_symbol_count = options.root_count,
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING,
      .request_sink = {.publish =
                           options.publish_kernels
                               ? loom_cmd_program_plan_check_release_request
                               : NULL},
  };
  loomc_product_t* product = NULL;
  loomc_result_t* result = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_cmd_program_product_build_module(
          request->environment->compile_session->workspace,
          request->public_module, &product_options,
          loomc_allocator_from_iree(request->host_allocator), &product,
          &result));
  bool succeeded = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              result, &succeeded);
  }
  if (iree_status_is_ok(status) && succeeded) {
    status = loom_cmd_program_plan_check_roundtrip_artifacts(
        product, request->host_allocator);
  }
  const loomc_artifact_t* listing = NULL;
  if (iree_status_is_ok(status) && succeeded) {
    status = loom_check_compile_select_artifact(
        result, LOOMC_ARTIFACT_KIND_TEXT,
        IREE_SV(LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM_LISTING), &listing);
  }
  if (iree_status_is_ok(status) && listing != NULL) {
    status = iree_status_from_loomc(loomc_byte_sequence_enumerate(
        listing->contents,
        (loomc_byte_sequence_callback_t){
            .fn = loom_cmd_program_plan_check_append_listing_segment,
            .user_data = &request->result->actual_output,
        }));
  }
  loomc_result_release(result);
  loomc_product_release(product);
  return status;
}

static iree_status_t loom_cmd_program_plan_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "command-program");
}

const loom_check_emit_provider_t loom_cmd_program_plan_check_emit_provider = {
    .name = IREE_SVL("command program plan"),
    .match = loom_cmd_program_plan_check_emit_provider_matches,
    .execute = loom_cmd_program_plan_check_emit_provider_execute,
    .append_names = loom_cmd_program_plan_check_emit_provider_append_names,
};
