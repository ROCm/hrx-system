// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_ACTIVATION_H_
#define AMDF_TOOL_INFO_ACTIVATION_H_

#include "libamdf/tool/amdf-info/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checks one device's creation, achieved facts, placement, and destruction.
// No queue, context, allocation, or workload is created. Native teardown
// failure marks the endpoint retained; its borrowed parents survive until
// process exit.
void amdf_info_activate(const amdf_info_library_t* library,
                        amdf_info_endpoint_t* endpoint,
                        amdf_info_output_t* output, amdf_info_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_ACTIVATION_H_
