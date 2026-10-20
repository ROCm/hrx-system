// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_DISPATCH_H_

#include "iree/hal/drivers/amd/gpu/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One placed native function. Firmware consumes the descriptor directly;
// there is no host register program or synthetic shader-visible packet.
typedef struct iree_hal_amd_gpu_aql_function_t {
  // Executable-owned reflection and shared argument representation.
  iree_hal_amd_gpu_function_t base;
  // Device address of the image's placed AMDHSA descriptor.
  uint64_t kernel_object;
  // Descriptor properties admitted by the image frontend.
  uint16_t code_properties;
} iree_hal_amd_gpu_aql_function_t;
static_assert(sizeof(void*) != 8 ||
                  sizeof(iree_hal_amd_gpu_aql_function_t) == 32,
              "two AQL functions fit in one cache line");

// Admits external arguments and geometry for a queue with native scratch
// servicing. Firmware supplies dispatch/queue pointers and fixed private
// storage. Dynamic stacks and hidden host services require separate policies
// and are rejected. Success initializes out_dispatch; failure leaves it intact.
iree_status_t iree_hal_amd_gpu_aql_dispatch_prepare(
    const iree_hal_amd_gpu_aql_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_dispatch_geometry_t* out_dispatch);

// Encodes admitted geometry and the placed function as a native packet.
// header contains resolved ordering/fence bits, without type/setup. Indirect
// geometry starts at zero with rank three; the device materializer owns the
// final counts and header publication. Invocation addresses are supplied by
// the storage owner and remain live through execution, not just consumption.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_dispatch_packet(
    const iree_hal_amd_gpu_aql_function_t* function,
    const iree_hal_amd_gpu_dispatch_geometry_t* dispatch, uint32_t header,
    uint64_t kernarg_address, uint64_t completion_signal,
    iree_hal_dispatch_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_DISPATCH_H_
