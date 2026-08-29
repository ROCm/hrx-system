// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final AIE2P compute-tile ELF image emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_IMAGE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_IMAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/target/emit/native/object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes one fully resolved leaf contribution as an executable AIE2P tile ELF.
//
// The image starts its executable load segment and entry address at AIE program
// address zero, carries EM_AIE and the AIE2P generation flag, and retains an
// ordinary ELF32 symbol table outside of the load segment. The contribution
// must contain one executable section and no unresolved fixups. Temporary
// section, symbol, and ELF layout storage is allocated from |scratch_arena|.
iree_status_t loom_aie2p_tile_image_write(
    const loom_native_object_contribution_t* object, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_IMAGE_H_
