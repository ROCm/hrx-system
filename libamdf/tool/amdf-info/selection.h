// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_SELECTION_H_
#define AMDF_TOOL_INFO_SELECTION_H_

#include "libamdf/tool/amdf-info/inventory.h"
#include "libamdf/tool/amdf-info/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum amdf_info_exit_code_e {
  AMDF_INFO_EXIT_SUCCESS = 0,
  AMDF_INFO_EXIT_FAILURE = 1,
  AMDF_INFO_EXIT_USAGE = 2,
  AMDF_INFO_EXIT_NOT_FOUND = 3,
  AMDF_INFO_EXIT_AMBIGUOUS = 4,
} amdf_info_exit_code_t;

typedef enum amdf_info_selector_kind_e {
  AMDF_INFO_SELECTOR_ALL,
  AMDF_INFO_SELECTOR_ORDINAL,
  AMDF_INFO_SELECTOR_ENDPOINT_ID,
  AMDF_INFO_SELECTOR_GPU_ID,
  AMDF_INFO_SELECTOR_NATIVE_PATH,
} amdf_info_selector_kind_t;

typedef struct amdf_info_selector_t {
  // Match mechanism; a bare family selects ordinal zero.
  amdf_info_selector_kind_t kind;
  // Required broad engine, or UNKNOWN for the unfiltered inventory/path.
  amdf_engine_kind_t engine;
  // Required GPU architecture, or UNKNOWN for any GPU architecture.
  amdf_gpu_architecture_t architecture;
  // Family-view ordinal for an ordinal selector.
  uint32_t ordinal;
  // Exact provider endpoint identity for an endpoint-ID selector.
  amdf_endpoint_id_t endpoint_id;
  // Exact nonzero native GPU/partition identity for a GPU-ID selector.
  uint64_t gpu_id;
  // Borrowed native pathname, resolved only after argument parsing succeeds.
  const char* path;
  // Native path identity supplied by the platform adapter before matching.
  amdf_endpoint_native_identity_t native_identity;
} amdf_info_selector_t;

// Returns a static diagnostic for invalid syntax. NULL input selects inventory.
// This is grammar only: no filesystem or provider operations are performed.
const char* amdf_info_selector_parse(const char* text,
                                     amdf_info_selector_t* out_selector);

// Marks the selected set in native enumeration order. Failed prerequisite
// queries cannot become negative matches or silently change family ordinals.
amdf_info_exit_code_t amdf_info_selection_apply(
    const amdf_info_options_t* options, const amdf_info_selector_t* selector,
    uint32_t count, amdf_info_endpoint_t* endpoints);

// Formats an exact endpoint URI accepted by the native AMD HAL. The output
// needs 48 bytes, including NUL; it never uses a view-relative ordinal. An
// unclassified engine has no executable selector and produces an empty string.
void amdf_info_selection_uri(const amdf_info_endpoint_t* endpoint,
                             char out_uri[48]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_SELECTION_H_
