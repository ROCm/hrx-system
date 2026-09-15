// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_ENDPOINT_H_
#define AMDF_SRC_PLATFORM_ENDPOINT_H_

#include "amdf/amdf.h"
#include "libamdf/src/platform/instance.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_platform_endpoint_t amdf_platform_endpoint_t;

// Opens one endpoint directly and returns all immutable base properties.
amdf_status_t amdf_platform_endpoint_open(
    amdf_platform_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_platform_endpoint_t** out_endpoint, amdf_endpoint_info_t* out_info);

// Queries publication paths implemented for one native command type.
amdf_queue_publication_modes_t
amdf_platform_endpoint_query_queue_publication_modes(
    const amdf_platform_endpoint_t* endpoint,
    amdf_queue_command_type_t command_type);

// Closes an endpoint query handle without waiting for device work.
amdf_status_t amdf_platform_endpoint_close(amdf_platform_endpoint_t* endpoint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_ENDPOINT_H_
