// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_MODULE_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_MODULE_H_

#include "iree/module/hal/types.h"
#include "iree/vm/module.h"
#include "loom/serve/runtime/parameter/sources.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the source-facing "parameters" read capability:
//   gather(i32 source, buffer key, hal.fence wait, hal.buffer descriptors,
//          i64 descriptor_offset, i64 count, hal.buffer target) -> hal.fence
//
// Source ordinals identify the cold/request catalog, not resident bindings.
// Each 24-byte descriptor is three little-endian u64 byte values: parameter
// offset, target offset, and length. Target spans must not overlap. The wait
// fence covers descriptor production, destination availability, and previous
// consumers; NULL means no prerequisite. Returned readiness orders consumers
// and descriptor reuse. No execution frontier is captured or changed here.
//
// Retains the exact queue, its device, and the source catalog. Accepted reads
// independently retain their provider and buffers through actual retirement,
// even after rebinding or module removal. A failed or discarded fence does not
// release those obligations. The hosting device group and registered reference
// types outlive every accepted operation and escaped reference.
iree_status_t loom_serve_parameters_module_create(
    iree_vm_environment_t* environment, const iree_hal_module_types_t* types,
    iree_hal_queue_t* queue, loom_serve_parameter_sources_t* sources,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

// Publishes a retained immutable source catalog at a serialized invocation
// boundary. NULL clears access to all sources. Does not wait for or redirect
// already accepted reads; each captured its exact provider at submission.
void loom_serve_parameters_module_bind_sources(
    iree_vm_module_t* module, loom_serve_parameter_sources_t* sources);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_MODULE_H_
