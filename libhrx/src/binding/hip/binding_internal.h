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
// This is the single source of truth for the "resolve HIP symbols against our
// own library" behavior shared by hipGetProcAddress() and the _spt lookup
// (hipGetProcAddress_spt/hipGetDriverEntryPoint_spt). It exists because a
// consumer may dlopen the HIP runtime with RTLD_LOCAL (Triton's AMD backend
// does exactly this), which keeps our symbols out of the process-global scope;
// a dlsym(dlopen(NULL), ...) lookup would then spuriously fail. Callers should
// dlsym() against this handle and fall back to the global scope only if it is
// NULL. Defined in api.c (which enables the GNU extensions it needs).
void* iree_hip_self_dl_handle(void);

// Resolves a public HIP symbol while honoring the property-structure ABI
// selected by |hip_version|. This helper does not publish last-error state.
hipError_t iree_hip_proc_address_lookup(const char* symbol, void** function,
                                        int hip_version, uint64_t flags,
                                        void* symbol_status);

// Resolves a driver entry point without source-ABI name rewriting. This helper
// does not publish last-error state and returns hipErrorNotFound for a missing
// symbol so the public driver boundary can apply its documented translation.
hipError_t iree_hip_driver_entry_point_lookup(const char* symbol,
                                              void** function, uint64_t flags,
                                              void* symbol_status);

// Applies the _spt lookup default: only a DEFAULT request is changed to the
// per-thread stream variant. Explicit legacy and per-thread requests are
// preserved. Invalid flags are rejected without modifying |out_flags|.
hipError_t iree_hip_normalize_spt_lookup_flags(uint64_t flags,
                                               uint64_t* out_flags);

#endif  // HRX_BINDING_HIP_BINDING_INTERNAL_H_
