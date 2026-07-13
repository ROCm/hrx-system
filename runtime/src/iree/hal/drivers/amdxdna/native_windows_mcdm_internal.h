// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_NATIVE_WINDOWS_MCDM_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_NATIVE_WINDOWS_MCDM_INTERNAL_H_

#include <cstddef>
#include <cstdint>

// Internal transaction parser seam shared with focused unit tests.
bool iree_hal_amdxdna_native_windows_find_partial_elf_bd_ops(
    const uint8_t* bytes, size_t total, uint32_t op_count,
    size_t queue_offset, uint32_t key, const uint8_t** out_dma,
    const uint8_t** out_ddr);

#endif  // IREE_HAL_DRIVERS_AMDXDNA_NATIVE_WINDOWS_MCDM_INTERNAL_H_
