// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// HIP binding internal header. Includes the streaming types that are
// now compiled as part of this binding (no longer a separate library).

#ifndef HRX_BINDING_HIP_BINDING_INTERNAL_H_
#define HRX_BINDING_HIP_BINDING_INTERNAL_H_

#include "binding/hip/api.h"
#include "common/internal.h"

// Returns a dlopen handle scoped to THIS shared object (the HIP shim), or NULL
// if it could not be established. The handle is resolved once and cached for
// the process lifetime; callers must NOT dlclose() it.
//
// Dynamic entry-point lookup resolves symbols through this handle. A runtime
// opened with RTLD_LOCAL is intentionally absent from the process-global
// scope, so dlsym(dlopen(NULL), ...) cannot find its symbols. Defined in api.c
// (which enables the GNU extensions it needs).
void* iree_hip_self_dl_handle(void);

// Converts and consumes an IREE status returned by the common binding layer.
hipError_t iree_hip_status_to_result(iree_status_t status);

#endif  // HRX_BINDING_HIP_BINDING_INTERNAL_H_
