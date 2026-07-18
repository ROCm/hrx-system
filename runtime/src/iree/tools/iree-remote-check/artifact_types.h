// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_IREE_REMOTE_CHECK_ARTIFACT_TYPES_H_
#define IREE_TOOLS_IREE_REMOTE_CHECK_ARTIFACT_TYPES_H_

#include "iree/hal/api.h"

// Backend-native executable artifact selected for one remote device.
typedef struct iree_remote_check_artifact_t {
  // Borrowed target row from the connected remote device spec.
  const iree_hal_executable_target_t* executable_target;
  // Human-readable embedded artifact file name.
  iree_string_view_t file_name;
  // Family-owned key describing the artifact's compilation target.
  iree_string_view_t artifact_target_key;
  // Embedded native executable bytes.
  iree_const_byte_span_t executable_data;
  // Exported function used by the diagnostic dispatch.
  iree_string_view_t entry_point;
  // Provider-owned inline constants passed to the diagnostic dispatch.
  iree_const_byte_span_t dispatch_constants;
  // Workgroup grid used by the diagnostic dispatch.
  iree_hal_dispatch_config_t dispatch_config;
} iree_remote_check_artifact_t;

#endif  // IREE_TOOLS_IREE_REMOTE_CHECK_ARTIFACT_TYPES_H_
