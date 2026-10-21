// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_packet_dispatch(
    const iree_hal_amd_gpu_aql_dispatch_t* dispatch) {
  return (iree_hal_amd_gpu_aql_packet_t){
      .words = {
          dispatch->header,
          (uint32_t)dispatch->workgroup_size[0] |
              ((uint32_t)dispatch->workgroup_size[1] << 16),
          dispatch->workgroup_size[2],
          dispatch->grid_size[0],
          dispatch->grid_size[1],
          dispatch->grid_size[2],
          dispatch->private_segment_size,
          dispatch->group_segment_size,
          (uint32_t)dispatch->kernel_object,
          (uint32_t)(dispatch->kernel_object >> 32),
          (uint32_t)dispatch->kernarg_address,
          (uint32_t)(dispatch->kernarg_address >> 32),
          0,
          0,
          (uint32_t)dispatch->completion_signal,
          (uint32_t)(dispatch->completion_signal >> 32),
      }};
}

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_packet_barrier(
    uint32_t header, uint32_t dependency_count, const uint64_t* dependencies,
    uint64_t completion_signal) {
  iree_hal_amd_gpu_aql_packet_t packet = {0};
  packet.words[0] = header;
  packet.words[14] = (uint32_t)completion_signal;
  packet.words[15] = (uint32_t)(completion_signal >> 32);
  for (uint32_t i = 0; i < dependency_count; ++i) {
    packet.words[2 + 2 * i] = (uint32_t)dependencies[i];
    packet.words[3 + 2 * i] = (uint32_t)(dependencies[i] >> 32);
  }
  return packet;
}

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_packet_indirect_buffer(
    uint32_t header, uint64_t address, uint32_t word_count,
    uint64_t completion_signal) {
  return (iree_hal_amd_gpu_aql_packet_t){
      .words = {(1u << 16) | header, UINT32_C(0xc0023f00), (uint32_t)address,
                (uint32_t)(address >> 32), (1u << 23) | word_count, 10, 0, 0, 0,
                0, 0, 0, 0, 0, (uint32_t)completion_signal,
                (uint32_t)(completion_signal >> 32)}};
}

uint32_t* iree_hal_amd_gpu_aql_emit_single_executor(uint32_t* words,
                                                    uint32_t xcc_count,
                                                    uint32_t body_word_count) {
  if (xcc_count > 1) {
    words[0] = UINT32_C(0xc0002300);
    words[1] = UINT32_C(0x01000000) | body_word_count;
    words += 2;
  }
  return words;
}
