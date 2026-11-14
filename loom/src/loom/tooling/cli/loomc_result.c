// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/cli/loomc_result.h"

#include "loomc/iree.h"
#include "loomc/source.h"

static const char* loom_tooling_cli_diagnostic_severity_name(
    loomc_diagnostic_severity_t severity) {
  switch (severity) {
    case LOOMC_DIAGNOSTIC_SEVERITY_NOTE:
      return "note";
    case LOOMC_DIAGNOSTIC_SEVERITY_WARNING:
      return "warning";
    case LOOMC_DIAGNOSTIC_SEVERITY_ERROR:
      return "error";
    default:
      return "diagnostic";
  }
}

static void loom_tooling_cli_print_loomc_source_range(
    FILE* file, const loomc_source_range_t* range) {
  if (range->source == NULL) {
    return;
  }
  const iree_string_view_t identifier =
      iree_string_view_from_loomc(loomc_source_identifier(range->source));
  if (iree_string_view_is_empty(identifier)) {
    return;
  }
  fprintf(file, "%.*s", (int)identifier.size, identifier.data);
  if (range->start_line != 0) {
    fprintf(file, ":%u", range->start_line);
    if (range->start_column != 0) {
      fprintf(file, ":%u", range->start_column);
    }
  }
  fputs(": ", file);
}

static void loom_tooling_cli_print_loomc_diagnostic(
    FILE* file, const loomc_diagnostic_t* diagnostic) {
  if (!loomc_string_view_is_empty(diagnostic->formatted_text)) {
    fwrite(diagnostic->formatted_text.data, 1, diagnostic->formatted_text.size,
           file);
    return;
  }
  loom_tooling_cli_print_loomc_source_range(file, &diagnostic->range);
  fprintf(file, "%s",
          loom_tooling_cli_diagnostic_severity_name(diagnostic->severity));
  if (!loomc_string_view_is_empty(diagnostic->code)) {
    fprintf(file, " [%.*s]", (int)diagnostic->code.size, diagnostic->code.data);
  }
  if (!loomc_string_view_is_empty(diagnostic->export_name)) {
    fprintf(file, " in export '%.*s'", (int)diagnostic->export_name.size,
            diagnostic->export_name.data);
  }
  fprintf(file, ": %.*s\n", (int)diagnostic->message.size,
          diagnostic->message.data);
  for (loomc_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
    const loomc_diagnostic_related_location_t* related =
        &diagnostic->related_locations[i];
    loom_tooling_cli_print_loomc_source_range(file, &related->range);
    fprintf(file, "note: %.*s\n", (int)related->label.size,
            related->label.data);
  }
  if (diagnostic->related_location_omitted_count != 0) {
    fprintf(
        file, "note: %zu additional related location%s omitted\n",
        (size_t)diagnostic->related_location_omitted_count,
        diagnostic->related_location_omitted_count == 1 ? " was" : "s were");
  }
}

iree_status_t loom_tooling_cli_print_loomc_result(FILE* file,
                                                  const loomc_result_t* result,
                                                  bool* out_succeeded) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    loom_tooling_cli_print_loomc_diagnostic(file, diagnostic);
  }
  if (ferror(file)) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "failed to write compiler diagnostics");
  }
  *out_succeeded = loomc_result_succeeded(result);
  return iree_ok_status();
}
