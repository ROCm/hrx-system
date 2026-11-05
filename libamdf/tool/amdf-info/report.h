// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_REPORT_H_
#define AMDF_TOOL_INFO_REPORT_H_

#include "libamdf/tool/amdf-info/inventory.h"
#include "libamdf/tool/amdf-info/output.h"
#include "libamdf/tool/amdf-info/selection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_info_result_t {
  // Whether all requested facts and lifecycle operations completed.
  bool complete;
  // Process result, including scalar selection failures.
  amdf_info_exit_code_t exit_code;
} amdf_info_result_t;

typedef enum amdf_info_query_requirement_e {
  AMDF_INFO_QUERY_OPTIONAL,
  AMDF_INFO_QUERY_REQUIRED,
} amdf_info_query_requirement_t;

// Typed serializers consume complete public query results. The caller owns
// query order and native lifetimes; renderers never probe or activate hardware.
void amdf_info_report_begin(amdf_info_output_t* output, const char* platform);
void amdf_info_report_library(amdf_info_output_t* output,
                              const amdf_info_library_t* library);
void amdf_info_report_end(amdf_info_output_t* output, bool complete,
                          amdf_info_exit_code_t exit_code, bool activated);

// Denied/unsupported optional facts can produce a useful partial inventory.
// Exact selection, activation, output, and cleanup remain required operations.
bool amdf_info_status_is_optional(amdf_status_t status);
// Records and renders a failed query without erasing its native error domain.
// Returns true only on success, permitting direct query/serialize callsites.
bool amdf_info_report_query(amdf_info_output_t* output,
                            amdf_info_result_t* result, const char* field,
                            const char* operation, amdf_status_t status,
                            amdf_info_query_requirement_t requirement);
void amdf_info_report_error(amdf_info_output_t* output, const char* field,
                            const char* operation, amdf_status_t status);
void amdf_info_report_selection_error(amdf_info_output_t* output,
                                      amdf_info_exit_code_t exit_code);

void amdf_info_report_pools_begin(amdf_info_output_t* output);
void amdf_info_report_pool(amdf_info_output_t* output, uint32_t ordinal,
                           const amdf_memory_scope_info_t* info);
void amdf_info_report_pools_end(amdf_info_output_t* output);
void amdf_info_report_endpoints_begin(amdf_info_output_t* output);
void amdf_info_report_endpoint_begin(amdf_info_output_t* output,
                                     const amdf_info_endpoint_t* endpoint);
void amdf_info_report_endpoint_identity(amdf_info_output_t* output,
                                        const amdf_endpoint_info_t* info);
void amdf_info_report_gpu_target(amdf_info_output_t* output,
                                 const amdf_gpu_endpoint_info_t* info);
void amdf_info_report_xdna_target(amdf_info_output_t* output,
                                  const amdf_xdna_endpoint_info_t* info);
void amdf_info_report_pci_address(amdf_info_output_t* output,
                                  const amdf_endpoint_pci_address_t* info);
void amdf_info_report_host_locality(amdf_info_output_t* output,
                                    const amdf_endpoint_host_locality_t* info);
void amdf_info_report_endpoint_end(amdf_info_output_t* output);
void amdf_info_report_endpoints_end(amdf_info_output_t* output);

void amdf_info_report_queues_begin(amdf_info_output_t* output);
void amdf_info_report_queue(amdf_info_output_t* output,
                            const amdf_queue_family_info_t* info);
void amdf_info_report_queues_end(amdf_info_output_t* output);
void amdf_info_report_placement(amdf_info_output_t* output,
                                const amdf_queue_family_placement_info_t* info);
void amdf_info_report_gpu_device(amdf_info_output_t* output,
                                 const amdf_gpu_device_info_t* info);
void amdf_info_report_xdna_device(amdf_info_output_t* output,
                                  const amdf_xdna_device_info_t* info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_REPORT_H_
