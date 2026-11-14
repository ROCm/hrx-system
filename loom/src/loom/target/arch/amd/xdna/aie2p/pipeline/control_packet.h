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
uint32_t loom_aie2p_control_packet_write(uint32_t address,
                                         iree_host_size_t word_count);
uint32_t loom_aie2p_control_packet_read(uint32_t address,
                                        iree_host_size_t word_count,
                                        uint8_t reply_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONTROL_PACKET_H_
