// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_EXECUTION_CONTEXT_TEST_UTIL_H_
#define LIBHRX_SRC_BINDING_HIP_EXECUTION_CONTEXT_TEST_UTIL_H_

#include "binding/hip/execution_context.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*iree_hip_execution_context_primary_resource_factory_t)(
    iree_hal_streaming_device_t* device,
    const iree_hal_queue_family_t* queue_family, hipDevResource* out_resource);

// Atomically exchanges the primary-context SM resource factory used by tests
// and returns the previous factory. Passing NULL restores production
// construction. This only substitutes the resource-construction dependency;
// primary-context validation, ownership, publication, and teardown are
// unchanged.
iree_hip_execution_context_primary_resource_factory_t
iree_hip_execution_context_exchange_primary_resource_factory_for_testing(
    iree_hip_execution_context_primary_resource_factory_t factory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_EXECUTION_CONTEXT_TEST_UTIL_H_
