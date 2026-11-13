// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_EXECUTABLE_MEASUREMENT_H_
#define IREE_HAL_DRIVER_AMD_XDNA_EXECUTABLE_MEASUREMENT_H_

#include "iree/hal/driver/amd/xdna/image/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  IREE_HAL_AMD_XDNA_MEASUREMENT_EXTRA_BYTES = 24,
  IREE_HAL_AMD_XDNA_MEASUREMENT_BEGIN_ID = 1,
  IREE_HAL_AMD_XDNA_MEASUREMENT_END_ID = 2,
};

// Materializes an optional measured command from an admitted invocation with a
// declared measurement interval. The caller supplies exclusive storage of
// byte_length + EXTRA_BYTES, retaining the ordinary command and its allocation
// closure. Checks the external native header before copying, then inserts two
// fixed-ID timer instructions at compiler-owned boundaries without decoding the
// command body. No output bytes are modified on failure.
iree_status_t iree_hal_amd_xdna_executable_measurement_initialize(
    const iree_xdna_elf_invocation_record_t* invocation, const uint8_t* command,
    uint8_t* target);

// Copies changed dynamic relocation word pairs from the ordinary command into
// its measured copy. Offsets and nonintersecting boundaries were admitted by
// the image reader. Returns the smallest byte span requiring cache publication;
// an empty span means this command contains no dynamic relocations.
iree_byte_span_t iree_hal_amd_xdna_executable_measurement_patch(
    const iree_hal_amd_xdna_image_tables_t* tables,
    const iree_xdna_elf_entry_record_t* entry,
    const iree_xdna_elf_invocation_record_t* invocation, const uint8_t* command,
    uint8_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_EXECUTABLE_MEASUREMENT_H_
