// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/artifact_manifest.h"

#include "loom/target/artifact_manifest.h"
#include "loomc/iree.h"

static loomc_artifact_manifest_mode_t loomc_artifact_manifest_mode_from_iree(
    loom_target_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY:
      return LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY;
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS:
      return LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS;
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return LOOMC_ARTIFACT_MANIFEST_MODE_ANALYSIS;
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE:
    default:
      return LOOMC_ARTIFACT_MANIFEST_MODE_NONE;
  }
}

static loom_target_artifact_manifest_mode_t
loomc_artifact_manifest_mode_to_iree(loomc_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
    case LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;
    case LOOMC_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
    case LOOMC_ARTIFACT_MANIFEST_MODE_NONE:
    default:
      return LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
  }
}

loomc_status_t loomc_artifact_manifest_mode_parse(
    loomc_string_view_t value, loomc_artifact_manifest_mode_t* out_mode) {
  if (out_mode == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_mode must not be NULL");
  }
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact manifest mode has length but no data");
  }
  loom_target_artifact_manifest_mode_t mode =
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_target_artifact_manifest_mode_parse(
          iree_string_view_from_loomc(value), &mode)));
  *out_mode = loomc_artifact_manifest_mode_from_iree(mode);
  return loomc_ok_status();
}

loomc_string_view_t loomc_artifact_manifest_mode_name(
    loomc_artifact_manifest_mode_t mode) {
  return loomc_string_view_from_iree(loom_target_artifact_manifest_mode_name(
      loomc_artifact_manifest_mode_to_iree(mode)));
}
