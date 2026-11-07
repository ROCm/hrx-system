// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_COMMANDS_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_COMMANDS_H_

#include "amdf/gpu.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native SDMA v1 writers for admitted command spans. The caller establishes
// storage capacity, operand access/alignment, visibility, and lifetime. Writers
// return the next DWORD and perform no validation, allocation, or publication.
// Format features come from the native family, independently of the shader ISA.

enum {
  // Prepared command extent admitted by the bounded native publisher. Its
  // private completion and event tail has separate storage.
  IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS = 224u,
  // Common byte-count bound, including the narrowest admitted COPY encoding.
  // Logical transfers use as many packets as needed; this bounds one packet.
  IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH = 0x3fff00u,
};

// Emits seven DWORDs copying a nonempty byte-aligned range. byte_length is at
// most MAX_PACKET_LENGTH. NPD remains clear; dependent operations still require
// their native execution boundary and separately qualified cache operations.
uint32_t* iree_hal_amd_gpu_sdma_emit_copy(uint32_t* words,
                                          amdf_queue_format_features_t features,
                                          uint64_t source_address,
                                          uint64_t target_address,
                                          uint32_t byte_length);

// Emits four DWORDs writing a confirmed, aligned coherent completion word.
// Private binary dependency use does not advertise general HAL atomic support.
uint32_t* iree_hal_amd_gpu_sdma_emit_fence(
    uint32_t* words, amdf_queue_format_features_t features, uint64_t address,
    uint32_t value);

// Emits six DWORDs waiting for an exact binary producer value. The address and
// memory scope belong to the device's qualified shared native signal domain.
uint32_t* iree_hal_amd_gpu_sdma_emit_wait(uint32_t* words,
                                          amdf_queue_format_features_t features,
                                          uint64_t address, uint32_t value);

// Emits five DWORDs of USER_GCR with admitted native control bits. This
// requires the family feature; KERNEL SUBMISSION_GCR is a retirement boundary
// instead of an in-stream encoding. Dependency waits precede an acquire.
uint32_t* iree_hal_amd_gpu_sdma_emit_cache(uint32_t* words, uint32_t control);

// Emits five DWORDs filling a nonempty DWORD-aligned range. byte_length is a
// multiple of four and at most MAX_PACKET_LENGTH; pattern is a four-byte
// period.
uint32_t* iree_hal_amd_gpu_sdma_emit_fill(uint32_t* words,
                                          amdf_queue_format_features_t features,
                                          uint64_t address,
                                          uint32_t byte_length,
                                          uint32_t pattern);

// Emits three DWORDs sampling the raw global clock into a private, 32-byte-
// aligned target. Sampling joins earlier commands; a subsequent NOP completes
// this posted write before a transfer reads the sample.
uint32_t* iree_hal_amd_gpu_sdma_emit_timestamp(
    uint32_t* words, amdf_queue_format_features_t features, uint64_t address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_COMMANDS_H_
