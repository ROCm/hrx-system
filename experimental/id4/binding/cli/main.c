// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, model, "",
          "Path to an Ideogram 4 parameter file or parameter directory.");
IREE_FLAG(string, prompt_json, "",
          "Full JSON prompt/configuration payload for one generation.");
IREE_FLAG(string, output, "", "Output image path.");
IREE_FLAG(string, dump_plan, "",
          "Path to write the structured pipeline plan JSON.");
IREE_FLAG(string, dump_diagnostics, "",
          "Directory for loomc, HAL, tensor, and stage diagnostics.");
IREE_FLAG(string, profile_output, "",
          "Path to write queue and dispatch-level profiling data.");

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "id4",
      "Experimental Ideogram 4 HAL pipeline runner.\n"
      "\n"
      "The CLI target is wired for the pipeline skeleton. Passing --model,\n"
      "--prompt_json, or --output requires a registered model "
      "implementation.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  int exit_code = EXIT_SUCCESS;
  if (argc > 1) {
    fprintf(stderr, "Error: no positional arguments expected.\n");
    exit_code = EXIT_FAILURE;
  } else if (strlen(FLAG_model) != 0 || strlen(FLAG_prompt_json) != 0 ||
             strlen(FLAG_output) != 0) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "no Ideogram 4 model implementation is registered yet");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  } else {
    fprintf(stdout,
            "ID4 pipeline CLI is available. Build stage implementations before "
            "running a model.\n");
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
