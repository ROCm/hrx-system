// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/control_packet.h"

#include "iree/base/internal/math.h"

uint32_t loom_aie2p_control_packet_parity(uint32_t word) {
  return word | ((iree_math_count_ones_u32(word) & 1u) ^ 1u) << 31;
}

uint32_t loom_aie2p_control_packet_write(uint32_t address,
                                         iree_host_size_t word_count) {
  return loom_aie2p_control_packet_parity(address | (uint32_t)(word_count - 1)
                                                        << 20);
}

uint32_t loom_aie2p_control_packet_read(uint32_t address,
                                        iree_host_size_t word_count,
                                        uint8_t reply_packet) {
  return loom_aie2p_control_packet_parity(
      address | (uint32_t)(word_count - 1) << 20 | UINT32_C(1) << 22 |
      (uint32_t)reply_packet << 24);
}
