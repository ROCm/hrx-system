// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/diagnostics.h"

static iree_status_t id4_pipeline_diagnostics_ignore_emit(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(event);
  return iree_ok_status();
}

void id4_pipeline_diagnostics_sink_initialize_ignore(
    id4_pipeline_diagnostics_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  out_sink->emit = id4_pipeline_diagnostics_ignore_emit;
  out_sink->user_data = NULL;
}

iree_status_t id4_pipeline_diagnostics_validate_sink(
    const id4_pipeline_diagnostics_sink_t* sink,
    iree_string_view_t usage_name) {
  if (!sink) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s diagnostics sink is required",
                            (int)usage_name.size, usage_name.data);
  }
  if (!sink->emit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s diagnostics sink emit callback is required",
                            (int)usage_name.size, usage_name.data);
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_diagnostics_emit(
    id4_pipeline_diagnostics_sink_t* sink,
    const id4_pipeline_diagnostic_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_RETURN_IF_ERROR(
      id4_pipeline_diagnostics_validate_sink(sink, IREE_SV("emit")));
  return sink->emit(sink->user_data, event);
}
