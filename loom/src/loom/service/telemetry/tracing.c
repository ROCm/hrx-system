// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/telemetry/tracing.h"

void loom_service_trace_initialize(void) {
  IREE_TRACE_SECTION_SETUP(LOOM_SERVICE_TRACE_SECTION_PREPARE, "model-prepare");
  IREE_TRACE_SECTION_SETUP(LOOM_SERVICE_TRACE_SECTION_WARMUP, "model-warmup");
  IREE_TRACE_SECTION_SETUP(LOOM_SERVICE_TRACE_SECTION_DEACTIVATE,
                           "model-deactivate");
  IREE_TRACE_SECTION_SETUP(LOOM_SERVICE_TRACE_SECTION_RETIRE, "model-retire");
  IREE_TRACE_SECTION_SETUP(LOOM_SERVICE_TRACE_SECTION_EXECUTE, "model-execute");
}

iree_section_id_t loom_service_trace_section_enter(
    loom_service_trace_section_t section, iree_string_view_t name) {
  switch (section) {
    case LOOM_SERVICE_TRACE_SECTION_PREPARE:
      return IREE_TRACE_SECTION_ENTER_TAGGED(section, "model-prepare",
                                             name.data, name.size);
    case LOOM_SERVICE_TRACE_SECTION_WARMUP:
      return IREE_TRACE_SECTION_ENTER_TAGGED(section, "model-warmup", name.data,
                                             name.size);
    case LOOM_SERVICE_TRACE_SECTION_DEACTIVATE:
      return IREE_TRACE_SECTION_ENTER_TAGGED(section, "model-deactivate",
                                             name.data, name.size);
    case LOOM_SERVICE_TRACE_SECTION_RETIRE:
      return IREE_TRACE_SECTION_ENTER_TAGGED(section, "model-retire", name.data,
                                             name.size);
    case LOOM_SERVICE_TRACE_SECTION_EXECUTE:
      return IREE_TRACE_SECTION_ENTER_TAGGED(section, "model-execute",
                                             name.data, name.size);
  }
  IREE_ASSERT_UNREACHABLE();
  return 0;
}
