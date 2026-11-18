// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_MEMORY_H_
#define AMDF_TOOL_INFO_MEMORY_H_

#include "libamdf/tool/amdf-info/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries and streams the scopes of |device|, or |instance| when device is
// NULL. The unused owner may be NULL. The owner stays live throughout the
// query; scope handles are borrowed and no memory is allocated from the scopes.
void amdf_info_query_memory_pools(const amdf_api_t* api,
                                  amdf_instance_t* instance,
                                  amdf_device_t* device,
                                  amdf_info_output_t* output,
                                  amdf_info_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_MEMORY_H_
