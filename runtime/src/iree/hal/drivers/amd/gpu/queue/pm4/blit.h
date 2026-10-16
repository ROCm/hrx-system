// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BLIT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BLIT_H_

#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resident-transfer arguments. Byte ranges may be unaligned. Native address
// admission bounds length below the signed offset limit; source is ignored for
// fills. Every invocation overwrites all consumed fields before dispatch.
typedef struct iree_hal_amd_gpu_pm4_blit_arguments_t {
  // Native source byte zero, unused for fills.
  uint64_t source;
  // Native destination byte zero.
  uint64_t target;
  // Complete nonoverlapping extent, in bytes; zero is a no-op.
  uint64_t length;
  // Four repeating bytes, beginning at target byte zero.
  uint32_t pattern;
  // Zero copies; one fills with pattern.
  uint32_t fill;
} iree_hal_amd_gpu_pm4_blit_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_blit_arguments_t) == 32,
              "blit arguments match the compiled ABI");

// Selects relocation-free native instructions and their descriptor. The cold
// owner places this once; no compiler or HSACO parser is in the runtime path.
iree_status_t iree_hal_amd_gpu_pm4_blit_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

// Emits a fixed 256-group launch covering an arbitrary admitted byte extent.
// Address fixups target the kernarg SGPR pair at DWORD 24. The owner supplies
// visibility and lifetime, just as for an ordinary recorded shader dispatch.
uint32_t* iree_hal_amd_gpu_pm4_blit_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t argument_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BLIT_H_
