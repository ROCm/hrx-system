// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/artifact_analyzer.h"

iree_status_t loom_run_artifact_analyzer_analyze(
    const loom_run_artifact_analyzer_t* analyzer,
    const loom_run_artifact_analysis_request_t* request) {
  if (analyzer == NULL || analyzer->fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact analyzer requires a callback");
  }
  if (request == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact analyzer requires a request");
  }
  return analyzer->fn(analyzer->user_data, request);
}
