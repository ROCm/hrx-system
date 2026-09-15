// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_DRM_ELF_PACKET_H_
#define AMDF_SRC_XDNA_UMD_DRM_ELF_PACKET_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ERT_START_NPU_PREEMPT_ELF with one CU mask and no save/restore or properties.
typedef struct amdf_linux_xdna_elf_packet_t {
  // Native little-endian transport words, including firmware-written state.
  uint32_t words[12];
} amdf_linux_xdna_elf_packet_t;

// Frames a validated instruction range without accessing its contents. The
// native queue exclusively owns packet storage through checked retirement.
void amdf_linux_xdna_elf_packet_build(uint64_t instruction_address,
                                      uint32_t instruction_byte_length,
                                      amdf_linux_xdna_elf_packet_t* out_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_XDNA_UMD_DRM_ELF_PACKET_H_
