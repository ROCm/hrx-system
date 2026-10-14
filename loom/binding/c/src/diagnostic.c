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
#include "loom/format/text/printer.h"
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

static iree_status_t loomc_diagnostic_format_type(
    loom_type_t type, void* user_data, loom_output_stream_t* stream) {
  const loomc_diagnostic_type_printer_t* printer =
      (const loomc_diagnostic_type_printer_t*)user_data;
  if (printer->text_print_options) {
    return loom_text_print_type_with_options(type, printer->module, stream,
                                             printer->text_print_options);
  }
  return loom_text_print_type(type, printer->module, stream);
}

static loomc_status_t loomc_render_loom_diagnostic_message(
    const loom_diagnostic_t* diagnostic,
    const loomc_diagnostic_type_printer_t* type_printer,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_type_formatter_t formatter =
      loomc_diagnostic_type_printer_formatter(type_printer);
  return loomc_status_from_iree(loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count, formatter,
      &stream));
}

static loomc_status_t loomc_format_loom_diagnostic(
    const loom_diagnostic_t* diagnostic,
    const loomc_diagnostic_type_printer_t* type_printer,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_diagnostic_format_options_t options = {
      .type_formatter = loomc_diagnostic_type_printer_formatter(type_printer),
  };
  return loomc_status_from_iree(
      loom_diagnostic_format_with_options(diagnostic, &options, &stream));
}

void loomc_diagnostic_type_printer_initialize(
    const loom_module_t* module,
    const loom_text_print_options_t* text_print_options,
    loomc_diagnostic_type_printer_t* out_printer) {
  *out_printer = (loomc_diagnostic_type_printer_t){
      .module = module,
      .text_print_options = text_print_options,
  };
}

loom_type_formatter_t loomc_diagnostic_type_printer_formatter(
    const loomc_diagnostic_type_printer_t* printer) {
  return (loom_type_formatter_t){
      .fn = printer ? loomc_diagnostic_format_type : loom_type_format_minimal,
      .user_data = (void*)printer,
  };
}

// Retains the matching input owner or copies a diagnostic's borrowed identity
// and optional spelling before frontend/module storage is released.
static loomc_status_t loomc_source_from_loom_range(
    const loomc_source_t* source, const loom_source_range_t* range,
    loomc_source_format_t format, loomc_source_retention_t source_retention,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  const loomc_byte_span_t contents =
      source_retention == LOOMC_SOURCE_RETENTION_EXACT
          ? loomc_make_byte_span(range->source.data, range->source.size)
          : loomc_byte_span_empty();
  if (source != NULL) {
    const loomc_byte_span_t input_contents = loomc_source_contents(source);
    const bool matching_identifier = iree_string_view_equal(
        iree_string_view_from_loomc(loomc_source_identifier(source)),
        range->filename);
    const bool matching_contents =
        input_contents.data == (const uint8_t*)range->source.data &&
        input_contents.data_length == range->source.size;
    if (matching_identifier &&
        (source_retention == LOOMC_SOURCE_RETENTION_METADATA_ONLY ||
         matching_contents)) {
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
    const loom_diagnostic_t* diagnostic,
    const loomc_diagnostic_type_printer_t* type_printer) {
  if (result == NULL || diagnostic == NULL || diagnostic->error == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and diagnostic must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_diagnostic_emit(
      loomc_result_loom_diagnostic_sink(result), diagnostic)));
  iree_allocator_t allocator =
      iree_allocator_from_loomc(loomc_result_allocator(result));
  iree_string_builder_t code_builder;
  iree_string_builder_initialize(allocator, &code_builder);
  iree_string_builder_t message_builder;
  iree_string_builder_initialize(allocator, &message_builder);
  iree_string_builder_t formatted_builder;
  iree_string_builder_initialize(allocator, &formatted_builder);
  iree_string_builder_t parameter_values_builder;
  iree_string_builder_initialize(allocator, &parameter_values_builder);

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
  loomc_diagnostic_parameter_t* parameters = NULL;
  iree_host_size_t parameter_count = 0;
  const loomc_source_retention_t source_retention =
      loomc_result_source_retention(result);
  loomc_status_t status =
      loomc_format_loom_diagnostic_code(diagnostic, &code_builder);
  if (loomc_status_is_ok(status)) {
    status = loomc_render_loom_diagnostic_message(diagnostic, type_printer,
                                                  &message_builder);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_format_loom_diagnostic(diagnostic, type_printer,
                                          &formatted_builder);
  }
  if (loomc_status_is_ok(status) && diagnostic->params &&
      diagnostic->param_count != 0 && diagnostic->error->param_count != 0) {
    parameter_count =
        iree_min(diagnostic->param_count, diagnostic->error->param_count);
    status = loomc_status_from_iree(iree_allocator_malloc_array(
        allocator, parameter_count, sizeof(*parameters), (void**)&parameters));
  }
  loom_output_stream_t parameter_values_stream;
  loom_output_stream_for_builder(&parameter_values_builder,
                                 &parameter_values_stream);
  const loom_type_formatter_t type_formatter =
      loomc_diagnostic_type_printer_formatter(type_printer);
  for (iree_host_size_t i = 0;
       loomc_status_is_ok(status) && i < parameter_count; ++i) {
    parameters[i].name = loomc_make_cstring_view(
        loom_error_def_param_name(diagnostic->error, i));
    const iree_host_size_t value_start = parameter_values_builder.size;
    status = loomc_status_from_iree(loom_diagnostic_render_param_value(
        &diagnostic->params[i], type_formatter, &parameter_values_stream));
    parameters[i].value = loomc_make_string_view(
        NULL, parameter_values_builder.size - value_start);
  }
  if (loomc_status_is_ok(status)) {
    const char* value_cursor =
        iree_string_builder_view(&parameter_values_builder).data;
    for (iree_host_size_t i = 0; i < parameter_count; ++i) {
      parameters[i].value.data = value_cursor;
      if (parameters[i].value.size != 0) {
        value_cursor += parameters[i].value.size;
      }
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_source_from_loom_range(
        source, &primary_range, primary_format, source_retention,
        loomc_result_allocator(result), &diagnostic_source);
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
          source_retention, loomc_result_allocator(result), &related_source);
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
        .formatted_text = loomc_string_view_from_iree(
            iree_string_builder_view(&formatted_builder)),
        .range =
            loomc_source_range_from_loom(&primary_range, diagnostic_source),
        .related_locations = related_locations,
        .related_location_count = related_location_count,
        .related_location_omitted_count =
            diagnostic->related_location_omitted_count,
        .parameters = parameters,
        .parameter_count = parameter_count,
    };
    status = loomc_result_add_diagnostic(result, &public_diagnostic);
  }

  for (iree_host_size_t i = 0; i < related_location_count; ++i) {
    loomc_source_release((loomc_source_t*)related_locations[i].range.source);
  }
  loomc_source_release(diagnostic_source);
  iree_allocator_free(allocator, parameters);
  iree_string_builder_deinitialize(&parameter_values_builder);
  iree_string_builder_deinitialize(&formatted_builder);
  iree_string_builder_deinitialize(&message_builder);
  iree_string_builder_deinitialize(&code_builder);
  return status;
}

loomc_status_t loomc_result_add_loom_diagnostic_emission(
    loomc_result_t* result, const loom_module_t* module,
    loom_source_resolver_t source_resolver, loom_emitter_t emitter,
    const loom_diagnostic_emission_t* emission,
    const loomc_diagnostic_type_printer_t* type_printer) {
  loom_diagnostic_t diagnostic;
  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  loom_source_resolve_diagnostic(module, source_resolver, emitter, emission,
                                 related_locations, &diagnostic);
  return loomc_result_add_loom_diagnostic(result, NULL, &diagnostic,
                                          type_printer);
}

void loomc_diagnostic_capture_initialize(
    loomc_result_t* result, const loomc_source_t* source,
    const loom_module_t* module, loom_source_resolver_t source_resolver,
    loom_emitter_t emitter, const loom_text_print_options_t* text_print_options,
    loomc_diagnostic_capture_t* out_capture) {
  *out_capture = (loomc_diagnostic_capture_t){
      .result = result,
      .source = source,
      .module = module,
      .source_resolver = source_resolver,
      .emitter = emitter,
  };
  loomc_diagnostic_type_printer_initialize(module, text_print_options,
                                           &out_capture->type_printer);
}

iree_status_t loomc_diagnostic_capture(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  const loomc_diagnostic_capture_t* capture = user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      capture->result, capture->source, diagnostic,
      capture->module ? &capture->type_printer : NULL));
}

iree_status_t loomc_diagnostic_capture_emission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  const loomc_diagnostic_capture_t* capture = user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, capture->module, capture->source_resolver,
      capture->emitter, emission, &capture->type_printer));
}

loomc_status_t loomc_result_verify_loom_module(
    const loom_module_t* module, loom_source_resolver_t source_resolver,
    loomc_result_t* result) {
  if (module == NULL || result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and result must not be NULL");
  }
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(result, /*source=*/NULL, module,
                                      source_resolver, LOOM_EMITTER_VERIFIER,
                                      /*text_print_options=*/NULL, &capture);
  const loom_verify_options_t verify_options = {
      .sink =
          {
              .fn = loomc_diagnostic_capture,
              .user_data = &capture,
          },
      .max_errors = 20,
      .source_resolver = source_resolver,
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
