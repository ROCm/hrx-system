// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/compilation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/util/json.h"
#include "loomc/iree.h"

static void compilation_copy(loomc_string_view_t value,
                             iree_host_size_t capacity, char* target) {
  const iree_host_size_t length = iree_min(value.size, capacity);
  if (length) {
    memcpy(target, value.data, length);
  }
}

iree_status_t loom_serve_compilation_result(
    loomc_status_t operation, loomc_result_t* result,
    loom_serve_compilation_report_t* report) {
  const loomc_diagnostic_t* first_error = NULL;
  for (iree_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    ++report->severities[diagnostic->severity];
    if (!first_error &&
        diagnostic->severity == LOOMC_DIAGNOSTIC_SEVERITY_ERROR) {
      first_error = diagnostic;
    }
    if (report->diagnostic_count == IREE_ARRAYSIZE(report->diagnostics)) {
      continue;
    }
    const loomc_string_view_t source =
        loomc_source_identifier(diagnostic->range.source);
    loom_serve_compilation_diagnostic_t* captured =
        &report->diagnostics[report->diagnostic_count++];
    *captured = (loom_serve_compilation_diagnostic_t){
        .severity = diagnostic->severity,
        .line = diagnostic->range.start_line,
        .column = diagnostic->range.start_column,
        .omitted_locations = diagnostic->related_location_count +
                             diagnostic->related_location_omitted_count,
        .omitted_parameters = diagnostic->parameter_count,
        .source_length = source.size,
        .code_length = diagnostic->code.size,
        .message_length = diagnostic->message.size};
    compilation_copy(source, sizeof(captured->source), captured->source);
    compilation_copy(diagnostic->code, sizeof(captured->code), captured->code);
    compilation_copy(diagnostic->message, sizeof(captured->message),
                     captured->message);
  }
  iree_status_t status = iree_status_from_loomc(operation);
  if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
    if (first_error) {
      const loomc_string_view_t source =
          loomc_source_identifier(first_error->range.source);
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "%.*s:%u:%u: %.*s: %.*s",
          (int)source.size, source.data, first_error->range.start_line,
          first_error->range.start_column, (int)first_error->code.size,
          first_error->code.data, (int)first_error->message.size,
          first_error->message.data);
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Loom compilation failed");
    }
  } else if (!iree_status_is_ok(status) && first_error) {
    const loomc_string_view_t source =
        loomc_source_identifier(first_error->range.source);
    status = iree_status_annotate_f(
        status, "%.*s:%u:%u: %.*s: %.*s", (int)source.size, source.data,
        first_error->range.start_line, first_error->range.start_column,
        (int)first_error->code.size, first_error->code.data,
        (int)first_error->message.size, first_error->message.data);
  }
  loomc_result_release(result);
  return status;
}

void loom_serve_compilation_merge(
    loom_serve_compilation_report_t* report,
    const loom_serve_compilation_report_t* source) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(report->severities); ++i) {
    report->severities[i] += source->severities[i];
  }
  const iree_host_size_t count =
      iree_min(source->diagnostic_count,
               IREE_ARRAYSIZE(report->diagnostics) - report->diagnostic_count);
  memcpy(report->diagnostics + report->diagnostic_count, source->diagnostics,
         count * sizeof(*source->diagnostics));
  report->diagnostic_count += count;
  report->commands += source->commands;
  report->kernels += source->kernels;
  report->reused_kernels += source->reused_kernels;
  report->entries += source->entries;
  report->workers = iree_max(report->workers, source->workers);
  report->duration_ns += source->duration_ns;
}

static iree_status_t compilation_write_text(loom_output_stream_t* output,
                                            const char* name, const char* value,
                                            uint64_t length,
                                            iree_host_size_t capacity) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output, ",\"%s_length\":%" PRIu64 ",\"%s\":", name, length, name));
  return loom_json_write_escaped_string(
      output, iree_make_string_view(value, iree_min(length, capacity)));
}

iree_status_t loom_serve_compilation_write(
    const loom_serve_compilation_report_t* report,
    loom_output_stream_t* output) {
  const uint64_t count =
      report->severities[0] + report->severities[1] + report->severities[2];
  if (!count && !report->commands && !report->kernels &&
      !report->reused_kernels) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      "{\"event\":\"compilation\",\"commands\":%" PRIu64 ",\"kernels\":%" PRIu64
      ",\"reused_kernels\":%" PRIu64 ",\"entries\":%" PRIu64
      ",\"workers\":%" PRIu64 ",\"duration_ns\":%" PRIu64 ",\"notes\":%" PRIu64
      ",\"warnings\":%" PRIu64 ",\"errors\":%" PRIu64
      ",\"omitted_diagnostics\":%" PRIu64 ",\"diagnostics\":[",
      report->commands, report->kernels, report->reused_kernels,
      report->entries, report->workers, report->duration_ns,
      report->severities[0], report->severities[1], report->severities[2],
      count - report->diagnostic_count));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < report->diagnostic_count && iree_status_is_ok(status); ++i) {
    const loom_serve_compilation_diagnostic_t* diagnostic =
        &report->diagnostics[i];
    status = loom_output_stream_write_format(
        output,
        "%s{\"severity\":%u,\"line\":%u,\"column\":%u,"
        "\"omitted_locations\":%" PRIu64 ",\"omitted_parameters\":%" PRIu64,
        i ? "," : "", diagnostic->severity, diagnostic->line,
        diagnostic->column, diagnostic->omitted_locations,
        diagnostic->omitted_parameters);
    if (iree_status_is_ok(status)) {
      status = compilation_write_text(output, "source", diagnostic->source,
                                      diagnostic->source_length,
                                      sizeof(diagnostic->source));
    }
    if (iree_status_is_ok(status)) {
      status = compilation_write_text(output, "code", diagnostic->code,
                                      diagnostic->code_length,
                                      sizeof(diagnostic->code));
    }
    if (iree_status_is_ok(status)) {
      status = compilation_write_text(output, "message", diagnostic->message,
                                      diagnostic->message_length,
                                      sizeof(diagnostic->message));
    }
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write_cstring(output, "}");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "]}\n");
  }
  return status;
}
