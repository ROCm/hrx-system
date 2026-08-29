// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P compute-tile target artifact emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARTIFACT_EMITTER_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARTIFACT_EMITTER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Production target emitter for one specialized AIE2P tile ELF unit.
extern const loom_target_emitter_t loom_aie2p_tile_elf_emitter;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_ARTIFACT_EMITTER_H_
