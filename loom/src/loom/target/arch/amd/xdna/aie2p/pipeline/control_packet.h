// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONTROL_PACKET_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONTROL_PACKET_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adds odd parity to an admitted 31-bit packet/header value.
uint32_t loom_aie2p_control_packet_parity(uint32_t word);

// Commands use a tile-relative 20-bit register address and 1..4 words. The
// transport supplies TLAST on the final write payload or read command word.
// A write's admitted register range stays within one 16-byte block.
uint32_t loom_aie2p_control_packet_write(uint32_t address,
                                         iree_host_size_t word_count);
uint32_t loom_aie2p_control_packet_read(uint32_t address,
                                        iree_host_size_t word_count,
                                        uint8_t reply_packet);

// Number of stream words needed to write a contiguous byte range starting at
// a word-aligned tile-control address. Each packet stays within its 16-byte
// aperture and the final data word is zero-padded.
iree_host_size_t loom_aie2p_control_packet_write_word_count(
    uint32_t address, iree_host_size_t byte_length);

// Frames the same range in |output| and returns the next free word. The
// caller supplies the parity-protected stream header for its physical route.
uint32_t* loom_aie2p_control_packet_store_writes(
    uint32_t packet_header, uint32_t address, iree_const_byte_span_t contents,
    uint32_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONTROL_PACKET_H_
