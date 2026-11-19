// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE_FULL_ELF_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE_FULL_ELF_H_

#include "iree/base/byte_sequence.h"
#include "iree/hal/driver/amd/xdna/image/aie2p/target.h"
#include "iree/hal/driver/amd/xdna/image/tables.h"

// AIEBU OS ABI for AIE2P transaction programs.
enum { IREE_HAL_AMD_AIE_OS_ABI = 0x45 };

// Admits grouped AIE2P controller programs, immutable PDIs, and external DMA
// buffer arguments from a standalone AIEBU ELF. The caller owns the tables and
// retains the source for their load ranges.
iree_status_t iree_hal_amd_xdna_image_read_aie_full_elf(
    iree_byte_sequence_t* source,
    const iree_hal_amd_xdna_aie2p_target_t* target, iree_allocator_t allocator,
    iree_hal_amd_xdna_image_tables_t* out_tables);

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_AIE_FULL_ELF_H_
