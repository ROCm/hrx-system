// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_MATERIALIZE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_MATERIALIZE_H_

#include "iree/hal/drivers/amd/gpu/image/image.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One nonempty native packet block, including complete indirect pairs. The
// caller reserves every destination slot before launching the materializer.
// Its first template header is valid; later INVALID templates are indirect
// targets preceded by their count-patch dispatch. Private arguments follow
// the templates within the same resident instance's 32-bit byte extent.
typedef struct iree_hal_amd_gpu_aql_materialize_arguments_t {
  // GPU buffer root of the native ring.
  uint64_t ring;
  // GPU buffer root of this block's resident templates.
  uint64_t packets;
  // Same ring address as integer data for indirect target pointers. Loom
  // buffer roots preserve storage identity and are not integer operands.
  uint64_t ring_address;
  // Same template address as integer data for private-argument offsets.
  uint64_t packet_address;
  // Power-of-two native packet capacity minus one.
  uint32_t ring_mask;
  // First reserved native slot, already reduced modulo ring capacity.
  uint32_t first;
  // Complete packet extent; indirect producer/target pairs are indivisible.
  uint32_t packet_count;
  // Initialized ABI padding, ignored by the shader.
  uint32_t reserved;
} iree_hal_amd_gpu_aql_materialize_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t) == 48,
              "materialization arguments occupy one cache line");

// Returns relocation-free instructions and descriptor facts. One wave copies
// a complete block, then publishes its first header with SYSTEM release. The
// ring is write-only; indirect pointer preparation reads resident templates.
iree_status_t iree_hal_amd_gpu_aql_materialize_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

// Encodes one wave. The caller resolves ordering/cache scopes in header and
// owns both the argument upload and resident instance through native use.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_materialize_packet(
    uint64_t kernel_object, uint64_t arguments, uint32_t header);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_MATERIALIZE_H_
