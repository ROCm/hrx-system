// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_INVENTORY_H_
#define AMDF_TOOL_INFO_INVENTORY_H_

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "amdf/xdna.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public tables negotiated once by this invocation. An absent family stays
// absent; it does not remove the platform's endpoint identities.
typedef struct amdf_info_library_t {
  // Core table, borrowed from the loaded library.
  const amdf_api_t* api;
  // GPU table when the extension negotiation succeeded.
  const amdf_gpu_api_t* gpu;
  // XDNA table when the extension negotiation succeeded.
  const amdf_xdna_api_t* xdna;
  // Exact GPU extension negotiation result.
  amdf_status_t gpu_status;
  // Exact XDNA extension negotiation result.
  amdf_status_t xdna_status;
} amdf_info_library_t;

// Last complete public record retained during identity acquisition.
typedef enum amdf_info_endpoint_stage_e {
  AMDF_INFO_ENDPOINT_SUMMARY,
  AMDF_INFO_ENDPOINT_OPEN,
  AMDF_INFO_ENDPOINT_INFO,
  AMDF_INFO_ENDPOINT_TARGET,
} amdf_info_endpoint_stage_t;

// Small selection index retaining public identity records, not queues, memory
// profiles, or a second hardware graph. Later reporting uses these same facts.
typedef struct amdf_info_endpoint_t {
  // Complete native enumeration record, retained even if qualification fails.
  amdf_endpoint_summary_t summary;
  // Query-only handle owned until checked endpoint_close.
  amdf_endpoint_t* handle;
  // Core identity, valid at INFO or TARGET stage.
  amdf_endpoint_info_t info;
  // Engine identity, valid only at TARGET stage for the matching engine kind.
  union {
    // Passive GPU identity and compute geometry.
    amdf_gpu_endpoint_info_t gpu;
    // Passive NPU compiler target identity.
    amdf_xdna_endpoint_info_t xdna;
  } target;
  // Last fully acquired record; no failed output is interpreted as data.
  amdf_info_endpoint_stage_t stage;
  // Failure of the next acquisition stage, or OK at TARGET stage.
  amdf_status_t status;
  // Native snapshot position, independent of sorted report order.
  uint32_t enumeration_ordinal;
  // Primary family ordinal, or UINT32_MAX when earlier qualification failed.
  uint32_t family_ordinal;
  // Whether this endpoint belongs to the selected output set.
  bool selected;
  // A failed native child teardown forbids closing this endpoint or instance.
  bool retain_on_exit;
} amdf_info_endpoint_t;

// Opens and qualifies a summary in place. Failure retains already acquired
// identity and the query handle for checked teardown; it is not a constructor.
void amdf_info_endpoint_inspect(const amdf_info_library_t* library,
                                amdf_instance_t* instance,
                                amdf_info_endpoint_t* endpoint);

// Acquires one bounded complete snapshot and opens its query-only endpoints.
// Individual qualification failures remain in their entries. A failed native
// enumeration or allocation publishes no array and retains no handles.
// The caller closes each opened handle and frees the array with free().
amdf_status_t amdf_info_inventory_read(const amdf_info_library_t* library,
                                       amdf_instance_t* instance,
                                       uint32_t* out_count,
                                       amdf_info_endpoint_t** out_endpoints);

// Returns the primary user-facing scheme without guessing from display names.
const char* amdf_info_endpoint_scheme(const amdf_info_endpoint_t* endpoint);

// Writes the exact compiler target name, or an empty string if unavailable.
void amdf_info_endpoint_target(const amdf_info_endpoint_t* endpoint,
                               char out_target[AMDF_XDNA_TARGET_ID_CAPACITY]);

// Formats the AMD GPU compiler target spelling from the public numeric IP.
void amdf_info_gpu_target(const amdf_gpu_endpoint_info_t* gpu,
                          char out_target[AMDF_XDNA_TARGET_ID_CAPACITY]);

// Sorts completed index entries by engine and opaque identity. Family ordinals
// retain their original native-view meaning through this presentation sort.
int amdf_info_endpoint_compare(const void* lhs, const void* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_INVENTORY_H_
