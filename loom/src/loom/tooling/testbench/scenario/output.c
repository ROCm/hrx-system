// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/output.h"

// Encode the authored symbol injectively as one portable path component. The
// fixed prefix avoids platform-reserved device names and empty components;
// encoding uppercase bytes preserves identity on case-insensitive filesystems.
static iree_status_t loom_testbench_append_scenario_output_path(
    const loom_testbench_scenario_trial_values_t* values,
    iree_string_builder_t* path) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(path, "scenario-"));
  const iree_string_view_t name = values->scenario_plan->name;
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    const uint8_t c = (uint8_t)name.data[i];
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
        c == '-') {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          path, iree_make_string_view(name.data + i, 1)));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(path, "%%%02X", c));
    }
  }
  return iree_string_builder_append_format(
      path, "/configuration-%" PRIhsz "/trial-%" PRIhsz "-%" PRIhsz "/",
      values->identity.configuration_ordinal, values->identity.trial_index,
      values->identity.trial_ordinal);
}

iree_status_t loom_testbench_scenario_write_files(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_scenario_trial_values_t* values, bool trial_failed) {
  const loom_testbench_scenario_action_plan_t* action =
      &values->trial_plan->action;
  iree_string_builder_t path;
  iree_string_builder_initialize(options->host_allocator, &path);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < action->file_write_count; ++i) {
    const loom_testbench_file_write_plan_t* output = &action->file_writes[i];
    if (output->mode == LOOM_CHECK_FILE_WRITE_NPY_MODE_ON_FAILURE &&
        !trial_failed) {
      continue;
    }
    for (unsigned side = 0; iree_status_is_ok(status) && side < 2; ++side) {
      if (!(output->realizations & (1u << side))) {
        continue;
      }
      iree_string_builder_reset(&path);
      status = loom_testbench_append_scenario_output_path(values, &path);
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(
            &path, side == 0 ? "actual/" : "expected/");
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_string(&path, output->path);
      }
      if (iree_status_is_ok(status)) {
        status = loom_testbench_write_file(
            options, output, side == 0 ? &values->target : &values->oracle,
            iree_string_builder_view(&path));
      }
    }
  }
  iree_string_builder_deinitialize(&path);
  return status;
}
