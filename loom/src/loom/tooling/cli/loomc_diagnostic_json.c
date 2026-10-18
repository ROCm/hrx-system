// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/cli/loomc_diagnostic_json.h"

#include "loom/error/error_defs.h"
#include "loom/util/json.h"
#include "loomc/iree.h"
#include "loomc/source.h"

static iree_string_view_t loom_tooling_cli_loomc_source_identifier(
    const loomc_source_range_t* range) {
  return range->source != NULL ? iree_string_view_from_loomc(
                                     loomc_source_identifier(range->source))
                               : iree_string_view_empty();
}

static const loom_error_def_t* loom_tooling_cli_loomc_diagnostic_error(
    loomc_string_view_t diagnostic_code, loom_error_domain_t* out_domain,
    uint16_t* out_code) {
  *out_domain = LOOM_ERROR_DOMAIN_COUNT_;
  *out_code = 0;
  loom_error_ref_t ref = LOOM_ERROR_REF_NONE;
  if (!loom_error_ref_parse(iree_string_view_from_loomc(diagnostic_code),
                            &ref)) {
    return NULL;
  }
  *out_domain = loom_error_ref_domain(ref);
  *out_code = loom_error_ref_code(ref);
  return loom_error_def_lookup_ref(ref);
}

static iree_status_t loom_tooling_cli_write_loomc_source_range_json(
    loom_json_object_writer_t* object, iree_string_view_t field_name,
    const loomc_source_range_t* range) {
  if (range->source == NULL && range->start_line == 0 && range->start == 0 &&
      range->end == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, field_name));
  loom_json_object_writer_t range_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &range_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &range_object, IREE_SV("filename"),
      loom_tooling_cli_loomc_source_identifier(range)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("start_line"), range->start_line));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("start_column"), range->start_column));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("end_line"), range->end_line));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("end_column"), range->end_column));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &range_object, IREE_SV("start_byte"), range->start));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &range_object, IREE_SV("end_byte"), range->end));
  return loom_json_object_end(&range_object);
}

static iree_string_view_t loom_tooling_cli_loomc_json_severity(
    loomc_diagnostic_severity_t severity) {
  switch (severity) {
    case LOOMC_DIAGNOSTIC_SEVERITY_NOTE:
      return IREE_SV("remark");
    case LOOMC_DIAGNOSTIC_SEVERITY_WARNING:
      return IREE_SV("warning");
    case LOOMC_DIAGNOSTIC_SEVERITY_ERROR:
      return IREE_SV("error");
  }
  return IREE_SV("error");
}

iree_status_t loom_tooling_cli_write_loomc_diagnostic_json(
    loom_output_stream_t* stream, const loomc_diagnostic_t* diagnostic) {
  if (!loomc_string_view_is_empty(diagnostic->formatted_json)) {
    return loom_output_stream_write(
        stream, iree_string_view_from_loomc(diagnostic->formatted_json));
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("severity"),
      loom_tooling_cli_loomc_json_severity(diagnostic->severity)));

  loom_error_domain_t domain = LOOM_ERROR_DOMAIN_COUNT_;
  uint16_t code = 0;
  const loom_error_def_t* error =
      loom_tooling_cli_loomc_diagnostic_error(diagnostic->code, &domain, &code);
  if (error != NULL) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("error_id"),
        iree_make_cstring_view(loom_error_def_id(error))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("domain"),
        iree_make_cstring_view(loom_error_domain_name(domain))));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint32_field(&object, IREE_SV("code"), code));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("summary"),
        iree_make_cstring_view(loom_error_def_summary(error))));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("diagnostic_code"),
        iree_string_view_from_loomc(diagnostic->code)));
  }
  IREE_RETURN_IF_ERROR(loom_tooling_cli_write_loomc_source_range_json(
      &object, IREE_SV("origin"), &diagnostic->range));
  IREE_RETURN_IF_ERROR(loom_tooling_cli_write_loomc_source_range_json(
      &object, IREE_SV("source_location"), &diagnostic->range));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("message"),
      iree_string_view_from_loomc(diagnostic->message)));
  if (!loomc_string_view_is_empty(diagnostic->formatted_text)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("formatted_text"),
        iree_string_view_from_loomc(diagnostic->formatted_text)));
  }
  if (diagnostic->parameter_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("params")));
    loom_json_object_writer_t parameters;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &parameters));
    for (loomc_host_size_t i = 0; i < diagnostic->parameter_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &parameters,
          iree_string_view_from_loomc(diagnostic->parameters[i].name),
          iree_string_view_from_loomc(diagnostic->parameters[i].value)));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&parameters));
  }
  if (diagnostic->related_location_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("related_locations")));
    loom_json_array_writer_t related_locations;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &related_locations));
    for (loomc_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
      const loomc_diagnostic_related_location_t* related =
          &diagnostic->related_locations[i];
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&related_locations));
      loom_json_object_writer_t related_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &related_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &related_object, IREE_SV("label"),
          iree_string_view_from_loomc(related->label)));
      IREE_RETURN_IF_ERROR(loom_tooling_cli_write_loomc_source_range_json(
          &related_object, IREE_SV("source_location"), &related->range));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&related_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&related_locations));
  }
  if (diagnostic->related_location_omitted_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("related_location_omitted_count"),
        diagnostic->related_location_omitted_count));
  }
  return loom_json_object_end(&object);
}
