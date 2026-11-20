// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/control_packet.h"

#include <string.h>

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

iree_host_size_t loom_aie2p_control_packet_write_word_count(
    uint32_t address, iree_host_size_t byte_length) {
  if (!byte_length) {
    return 0;
  }
  const iree_host_size_t packets = ((address & 15u) + byte_length + 15) / 16;
  return 2 * packets + (byte_length + 3) / 4;
}

uint32_t* loom_aie2p_control_packet_store_writes(
    uint32_t packet_header, uint32_t address, iree_const_byte_span_t contents,
    uint32_t* output) {
  while (contents.data_length) {
    const iree_host_size_t size =
        iree_min(contents.data_length, 16u - (address & 15u));
    *output++ = packet_header;
    *output++ = loom_aie2p_control_packet_write(address, (size + 3) / 4);
    const iree_host_size_t whole_words = size / 4;
    for (iree_host_size_t i = 0; i < whole_words; ++i) {
      *output++ = iree_unaligned_load_le_u32(contents.data + i * 4);
    }
    if (size & 3u) {
      uint8_t tail[4] = {0};
      memcpy(tail, contents.data + whole_words * 4, size % 4);
      *output++ = iree_unaligned_load_le_u32(tail);
    }
    address += (uint32_t)size;
    contents.data += size;
    contents.data_length -= size;
  }
  return output;
}
