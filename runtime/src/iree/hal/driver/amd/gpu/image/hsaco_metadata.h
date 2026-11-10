// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_METADATA_H_
#define IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_METADATA_H_

#include "iree/hal/driver/amd/gpu/image/metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decodes one or more AMDHSA MessagePack metadata notes into native kernel ABI
// and reflection records. Payloads contain the descriptor bytes only, excluding
// ELF note headers/names/padding. Their immutable storage is borrowed for the
// lifetime of the metadata. Multiple notes must agree on version and on any
// supplied target. Linked code objects may omit the redundant metadata target;
// the HSACO reader derives the authoritative target from the ELF header.
//
// Allocation and external-input validation occur here. Successful consumers use
// indexed records without parsing strings or revisiting source containers.
// On failure, |out_metadata| is empty and has no cleanup obligation.
iree_status_t iree_hal_amd_gpu_hsaco_metadata_initialize(
    iree_host_size_t payload_count, const iree_const_byte_span_t* payloads,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_metadata_t* out_metadata);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_IMAGE_HSACO_METADATA_H_
