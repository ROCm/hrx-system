// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic_loomc.h"

#include <string.h>

#include "loomc/iree.h"

static iree_status_t loom_test_diagnostic_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_copy) {
  *out_copy = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return iree_ok_status();
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&storage));
  memcpy(storage, source.data, source.size);
  *out_copy = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static loom_diagnostic_severity_t loom_test_diagnostic_severity_from_loomc(
    loomc_diagnostic_severity_t severity) {
  switch (severity) {
    case LOOMC_DIAGNOSTIC_SEVERITY_NOTE:
      return LOOM_DIAGNOSTIC_REMARK;
    case LOOMC_DIAGNOSTIC_SEVERITY_WARNING:
      return LOOM_DIAGNOSTIC_WARNING;
    case LOOMC_DIAGNOSTIC_SEVERITY_ERROR:
      return LOOM_DIAGNOSTIC_ERROR;
  }
  return LOOM_DIAGNOSTIC_ERROR;
}

static void loom_test_diagnostic_error_from_loomc(
    loomc_string_view_t code, loom_error_domain_t* out_domain,
    uint16_t* out_code) {
  *out_domain = LOOM_ERROR_DOMAIN_COUNT_;
  *out_code = 0;
  loom_error_ref_t ref = LOOM_ERROR_REF_NONE;
  if (!loom_error_ref_parse(iree_string_view_from_loomc(code), &ref)) {
    return;
  }
  *out_domain = loom_error_ref_domain(ref);
  *out_code = loom_error_ref_code(ref);
}

static iree_status_t loom_test_diagnostic_source_filename(
    const loomc_source_range_t* range, iree_arena_allocator_t* arena,
    iree_string_view_t* out_filename) {
  *out_filename = iree_string_view_empty();
  if (range->source == NULL) {
    return iree_ok_status();
  }
  const iree_string_view_t identifier =
      iree_string_view_from_loomc(loomc_source_identifier(range->source));
  return loom_test_diagnostic_copy_string(arena, identifier, out_filename);
}

iree_status_t loom_test_diagnostic_materialize_loomc(
    const loomc_diagnostic_t* source, iree_arena_allocator_t* arena,
    loom_test_diagnostic_t* out_diagnostic) {
  *out_diagnostic = (loom_test_diagnostic_t){0};
  out_diagnostic->severity =
      loom_test_diagnostic_severity_from_loomc(source->severity);
  loom_test_diagnostic_error_from_loomc(source->code, &out_diagnostic->domain,
                                        &out_diagnostic->code);
  IREE_RETURN_IF_ERROR(loom_test_diagnostic_source_filename(
      &source->range, arena, &out_diagnostic->origin.filename));
  out_diagnostic->origin.line = source->range.start_line;
  IREE_RETURN_IF_ERROR(loom_test_diagnostic_copy_string(
      arena, iree_string_view_from_loomc(source->message),
      &out_diagnostic->message));
  if (source->parameter_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, source->parameter_count, sizeof(*out_diagnostic->params),
        (void**)&out_diagnostic->params));
    for (loomc_host_size_t j = 0; j < source->parameter_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_test_diagnostic_copy_string(
          arena, iree_string_view_from_loomc(source->parameters[j].name),
          &out_diagnostic->params[j].name));
      IREE_RETURN_IF_ERROR(loom_test_diagnostic_copy_string(
          arena, iree_string_view_from_loomc(source->parameters[j].value),
          &out_diagnostic->params[j].value));
    }
    out_diagnostic->param_count = source->parameter_count;
  }
  const iree_string_view_t formatted =
      !loomc_string_view_is_empty(source->formatted_text)
          ? iree_string_view_from_loomc(source->formatted_text)
          : iree_string_view_from_loomc(source->message);
  IREE_RETURN_IF_ERROR(loom_test_diagnostic_copy_string(
      arena, formatted, &out_diagnostic->formatted_diagnostic));
  return iree_ok_status();
}
