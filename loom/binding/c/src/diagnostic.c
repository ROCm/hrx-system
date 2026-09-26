// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "diagnostic.h"

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/error/renderer.h"
#include "loom/error/source.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"
#include "loomc/iree.h"

static loomc_diagnostic_severity_t loomc_diagnostic_severity_from_loom(
    loom_diagnostic_severity_t severity) {
  switch (severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      return LOOMC_DIAGNOSTIC_SEVERITY_ERROR;
    case LOOM_DIAGNOSTIC_WARNING:
      return LOOMC_DIAGNOSTIC_SEVERITY_WARNING;
    case LOOM_DIAGNOSTIC_REMARK:
      return LOOMC_DIAGNOSTIC_SEVERITY_NOTE;
    case LOOM_DIAGNOSTIC_COUNT_:
      break;
  }
  return LOOMC_DIAGNOSTIC_SEVERITY_ERROR;
}

static loomc_status_t loomc_format_loom_diagnostic_code(
    const loom_diagnostic_t* diagnostic, iree_string_builder_t* builder) {
  const char* domain =
      loom_error_domain_name(loom_error_def_domain(diagnostic->error));
  return loomc_status_from_iree(iree_string_builder_append_format(
      builder, "%s/%03u", domain, loom_error_def_code(diagnostic->error)));
}

static loomc_status_t loomc_render_loom_diagnostic_message(
    const loom_diagnostic_t* diagnostic, iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_type_formatter_t formatter = {loom_type_format_minimal, NULL};
  return loomc_status_from_iree(loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count, formatter,
      &stream));
}

// Retains the matching input owner or copies a diagnostic's borrowed identity
// and optional spelling before frontend/module storage is released.
static loomc_status_t loomc_source_from_loom_range(
    const loomc_source_t* source, const loom_source_range_t* range,
    loomc_source_format_t format, loomc_allocator_t allocator,
    loomc_source_t** out_source) {
  *out_source = NULL;
  loomc_byte_span_t contents =
      loomc_make_byte_span(range->source.data, range->source.size);
  if (source != NULL) {
    const loomc_byte_span_t input_contents = loomc_source_contents(source);
    if (iree_string_view_equal(
            iree_string_view_from_loomc(loomc_source_identifier(source)),
            range->filename) &&
        contents.data == input_contents.data &&
        contents.data_length == input_contents.data_length) {
      *out_source = (loomc_source_t*)source;
      loomc_source_retain(*out_source);
      return loomc_ok_status();
    }
  }
  if (range->filename.size == 0 && contents.data_length == 0) {
    return loomc_ok_status();
  }
  const loomc_source_options_t options = {
      .format = format,
      .identifier = loomc_string_view_from_iree(range->filename),
      .contents = contents,
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  return loomc_source_create(&options, allocator, out_source);
}

// Ranges from one snapshot can share a retained source even when its bytes
// were copied during capture. Filenames alone do not identify a snapshot.
static bool loomc_diagnostic_ranges_share_source(
    const loom_source_range_t* lhs, const loom_source_range_t* rhs) {
  return iree_string_view_equal(lhs->filename, rhs->filename) &&
         lhs->source.data == rhs->source.data &&
         lhs->source.size == rhs->source.size;
}

static loomc_source_range_t loomc_source_range_from_loom(
    const loom_source_range_t* range, const loomc_source_t* source) {
  return (loomc_source_range_t){
      .source = source,
      .start = range->start,
      .end = range->end,
      .start_line = range->start_line,
      .start_column = range->start_column,
      .end_line = range->end_line,
      .end_column = range->end_column,
  };
}

loomc_status_t loomc_result_add_loom_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    const loom_diagnostic_t* diagnostic) {
  if (result == NULL || diagnostic == NULL || diagnostic->error == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and diagnostic must not be NULL");
  }
  iree_allocator_t allocator =
      iree_allocator_from_loomc(loomc_result_allocator(result));
  iree_string_builder_t code_builder;
  iree_string_builder_initialize(allocator, &code_builder);
  iree_string_builder_t message_builder;
  iree_string_builder_initialize(allocator, &message_builder);

  loom_source_range_t primary_range = diagnostic->source_location;
  loomc_source_format_t primary_format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  // Reader offsets identify the bytecode input itself. Later compiler
  // locations and related notes identify their own recorded source.
  if (source && diagnostic->emitter == LOOM_EMITTER_BYTECODE_READER) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    primary_range.source =
        iree_make_string_view((const char*)contents.data, contents.data_length);
    primary_format = LOOMC_SOURCE_FORMAT_BYTECODE;
  }
  loomc_source_t* diagnostic_source = NULL;
  loomc_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS] = {0};
  iree_host_size_t related_location_count = 0;
  loomc_status_t status =
      loomc_format_loom_diagnostic_code(diagnostic, &code_builder);
  if (loomc_status_is_ok(status)) {
    status = loomc_render_loom_diagnostic_message(diagnostic, &message_builder);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_source_from_loom_range(
        source, &primary_range, primary_format, loomc_result_allocator(result),
        &diagnostic_source);
  }
  for (iree_host_size_t i = 0;
       loomc_status_is_ok(status) && i < diagnostic->related_location_count;
       ++i) {
    const loom_diagnostic_related_location_t* related =
        &diagnostic->related_locations[i];
    loomc_source_t* related_source = NULL;
    if (loomc_diagnostic_ranges_share_source(&primary_range,
                                             &related->source_location)) {
      related_source = diagnostic_source;
    }
    for (iree_host_size_t j = 0; !related_source && j < i; ++j) {
      if (loomc_diagnostic_ranges_share_source(
              &diagnostic->related_locations[j].source_location,
              &related->source_location)) {
        related_source = (loomc_source_t*)related_locations[j].range.source;
      }
    }
    if (related_source) {
      loomc_source_retain(related_source);
    } else {
      status = loomc_source_from_loom_range(
          source, &related->source_location, LOOMC_SOURCE_FORMAT_UNKNOWN,
          loomc_result_allocator(result), &related_source);
    }
    if (loomc_status_is_ok(status)) {
      related_locations[related_location_count++] =
          (loomc_diagnostic_related_location_t){
              .label = loomc_string_view_from_iree(related->label),
              .range = loomc_source_range_from_loom(&related->source_location,
                                                    related_source),
          };
    }
  }
  if (loomc_status_is_ok(status)) {
    loomc_diagnostic_t public_diagnostic = {
        .severity = loomc_diagnostic_severity_from_loom(diagnostic->severity),
        .code = loomc_string_view_from_iree(
            iree_string_builder_view(&code_builder)),
        .message = loomc_string_view_from_iree(
            iree_string_builder_view(&message_builder)),
        .range =
            loomc_source_range_from_loom(&primary_range, diagnostic_source),
        .related_locations = related_locations,
        .related_location_count = related_location_count,
        .related_location_omitted_count =
            diagnostic->related_location_omitted_count,
    };
    status = loomc_result_add_diagnostic(result, &public_diagnostic);
  }

  for (iree_host_size_t i = 0; i < related_location_count; ++i) {
    loomc_source_release((loomc_source_t*)related_locations[i].range.source);
  }
  loomc_source_release(diagnostic_source);
  iree_string_builder_deinitialize(&message_builder);
  iree_string_builder_deinitialize(&code_builder);
  return status;
}

loomc_status_t loomc_result_add_loom_diagnostic_emission(
    loomc_result_t* result, const loom_module_t* module, loom_emitter_t emitter,
    const loom_diagnostic_emission_t* emission) {
  if (emission == NULL || emission->error == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "diagnostic emission must not be NULL");
  }
  loom_diagnostic_t diagnostic = {
      .severity = loom_error_def_severity(emission->error),
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = emitter,
  };
  if (emission->op) {
    const loom_module_t* primary_module =
        emission->module ? emission->module : module;
    loom_source_resolve((loom_source_resolver_t){0}, primary_module,
                        emission->op->location, &diagnostic.source_location);
    diagnostic.origin = diagnostic.source_location;
  }
  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  for (iree_host_size_t i = 0; i < emission->related_op_count; ++i) {
    const loom_diagnostic_related_op_t* related = &emission->related_ops[i];
    const loom_module_t* related_module =
        related->module ? related->module : module;
    loom_source_range_t range;
    if (!related->op ||
        !loom_source_resolve((loom_source_resolver_t){0}, related_module,
                             related->op->location, &range)) {
      continue;
    }
    if (diagnostic.related_location_count ==
        LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS) {
      ++diagnostic.related_location_omitted_count;
      continue;
    }
    related_locations[diagnostic.related_location_count++] =
        (loom_diagnostic_related_location_t){
            .label = related->label,
            .source_location = range,
        };
  }
  diagnostic.related_locations = related_locations;
  return loomc_result_add_loom_diagnostic(result, NULL, &diagnostic);
}

static iree_status_t loomc_result_verify_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      (loomc_result_t*)user_data, NULL, diagnostic));
}

loomc_status_t loomc_result_verify_loom_module(const loom_module_t* module,
                                               loomc_result_t* result) {
  if (module == NULL || result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and result must not be NULL");
  }
  loom_verify_options_t verify_options = {
      .sink =
          {
              .fn = loomc_result_verify_capture_diagnostic,
              .user_data = result,
          },
      .max_errors = 20,
  };
  loom_verify_result_t verify_result = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_verify_module(module, &verify_options, &verify_result)));
  if (verify_result.error_count != 0) {
    return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  return loomc_ok_status();
}

loomc_status_t loomc_result_add_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status) {
  if (result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result must not be NULL");
  }
  loomc_string_view_t message = loomc_status_message(status);
  if (loomc_string_view_is_empty(message)) {
    message = loomc_make_cstring_view(
        loomc_status_code_string(loomc_status_code(status)));
  }
  loomc_diagnostic_t diagnostic = {
      .severity = severity,
      .code = code,
      .message = message,
      .range =
          {
              .source = source,
          },
  };
  return loomc_result_add_diagnostic(result, &diagnostic);
}

bool loomc_status_is_result_diagnostic(loomc_status_t status) {
  switch (loomc_status_code(status)) {
    case LOOMC_STATUS_INVALID_ARGUMENT:
    case LOOMC_STATUS_NOT_FOUND:
    case LOOMC_STATUS_FAILED_PRECONDITION:
    case LOOMC_STATUS_OUT_OF_RANGE:
    case LOOMC_STATUS_UNIMPLEMENTED:
    case LOOMC_STATUS_INCOMPATIBLE:
      return true;
    default:
      return false;
  }
}

loomc_status_t loomc_result_fail_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status) {
  LOOMC_RETURN_IF_ERROR(loomc_result_add_status_diagnostic(
      result, source, severity, code, status));
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}

loomc_status_t loomc_result_fail_status_diagnostic_consume(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status) {
  loomc_status_t add_status = loomc_result_fail_status_diagnostic(
      result, source, severity, code, status);
  loomc_status_free(status);
  return add_status;
}
