// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_SOURCE_TYPES_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_SOURCE_TYPES_H_

#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_aie2p_source_type_kind_t;
enum loom_aie2p_source_type_kind_e {
  // Not an AIE2P source-only type.
  LOOM_AIE2P_SOURCE_TYPE_NONE = 0,
  // One architectural load-FIFO state chain.
  LOOM_AIE2P_SOURCE_TYPE_LOAD_FIFO = 1,
  // One architectural store-FIFO state chain.
  LOOM_AIE2P_SOURCE_TYPE_STORE_FIFO = 2,
  // Store-FIFO output-pointer carrier.
  LOOM_AIE2P_SOURCE_TYPE_STORE_FIFO_POINTER = 3,
  // Store-FIFO available-position carrier.
  LOOM_AIE2P_SOURCE_TYPE_STORE_FIFO_POSITION = 4,
  // One native BFP16EBS8 matrix operand.
  LOOM_AIE2P_SOURCE_TYPE_BFP16EBS8_OPERAND = 5,
};

// Classifies the source-only AIE2P type, or returns
// LOOM_AIE2P_SOURCE_TYPE_NONE for every other type.
loom_aie2p_source_type_kind_t loom_aie2p_source_type_kind(
    const loom_module_t* module, loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_SOURCE_TYPES_H_
