// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/diagnostics.h"

iree_status_t id4_pipeline_diagnostics_emit(
    id4_pipeline_diagnostics_sink_t* sink,
    const id4_pipeline_diagnostic_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  if (!sink || !sink->emit) return iree_ok_status();
  return sink->emit(sink->user_data, event);
}
