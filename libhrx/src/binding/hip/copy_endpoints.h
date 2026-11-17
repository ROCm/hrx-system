// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_COPY_ENDPOINTS_H_
#define HRX_BINDING_HIP_COPY_ENDPOINTS_H_

#include "binding/hip/api.h"
#include "common/memory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One resolved pointer. The application keeps the allocation alive; resolving
// an address does not retain its buffer or pool.
typedef struct iree_hip_copy_endpoint_t {
  // Borrowed allocation and byte offset, or empty for pageable host memory.
  iree_hal_streaming_buffer_ref_t reference;
  // Allocation context, borrowed when local and retained when foreign.
  // NULL for pageable host memory.
  iree_hal_streaming_context_t* context;
} iree_hip_copy_endpoint_t;

// Temporary preparation state shared by copy recording and parameter updates.
typedef struct iree_hip_copy_endpoints_t {
  // Effective direction, with Default classified and NoCU normalized.
  hipMemcpyKind kind;
  // Source mapping and its context lifetime during preparation.
  iree_hip_copy_endpoint_t source;
  // Destination mapping and its context lifetime during preparation.
  iree_hip_copy_endpoint_t target;
} iree_hip_copy_endpoints_t;

// Resolves each endpoint once, preferring |context|, and validates the supplied
// spans. Device operands respect the allocation's logical HIP extent. A local
// registered host operand must contain its entire span; an unknown host operand
// remains pageable. Explicit host-to-host copies need no registry lookup.
// On success, deinitialize with the same |context| after consuming the result.
// On failure, no references remain owned by |out_endpoints|.
hipError_t iree_hip_copy_endpoints_initialize(
    iree_hal_streaming_context_t* context, hipMemcpyKind kind,
    const void* target, iree_host_size_t target_span, const void* source,
    iree_host_size_t source_span, iree_hip_copy_endpoints_t* out_endpoints);

// Releases only foreign-context references; local preparation does no reference
// counting because its caller already owns the selected context.
void iree_hip_copy_endpoints_deinitialize(iree_hal_streaming_context_t* context,
                                          iree_hip_copy_endpoints_t* endpoints);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_COPY_ENDPOINTS_H_
