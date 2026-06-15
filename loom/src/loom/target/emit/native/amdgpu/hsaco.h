// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code-object emission.
//
// This layer owns the AMDGPU-specific ELF policy on top of the tiny generic
// ELF writer: metadata notes, dynamic symbols, descriptor bytes, executable
// text placement, and the virtual-address relationships required by AMDHSA
// kernel descriptors.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Descriptor-only ABI controls for one kernel entry.
typedef struct loom_amdgpu_hsaco_kernel_descriptor_options_t {
  // Descriptor-only ABI flags requested by low/kernel lowering.
  loom_amdgpu_kernel_descriptor_flags_t flags;
  // Minimum user SGPR count implied by descriptor-only ABI flags.
  uint32_t user_sgpr_count;
} loom_amdgpu_hsaco_kernel_descriptor_options_t;

// One kernel entry emitted into an AMDGPU HSA code object.
typedef struct loom_amdgpu_hsaco_kernel_t {
  // Kernel metadata row shared by the AMDGPU note and kernel descriptor.
  loom_amdgpu_metadata_kernel_t metadata;
  // Descriptor-only ABI controls that are not present in metadata.
  loom_amdgpu_hsaco_kernel_descriptor_options_t descriptor_options;
  // Encoded native instructions for the kernel entry symbol.
  iree_const_byte_span_t text;
} loom_amdgpu_hsaco_kernel_t;

// Complete AMDGPU HSA code object description.
typedef struct loom_amdgpu_hsaco_file_t {
  // Full AMDHSA target id such as `amdgcn-amd-amdhsa--gfx1100`.
  iree_string_view_t target;
  // Processor such as `gfx1100`, used for ELF flags and descriptor packing.
  iree_string_view_t processor;
  // Kernel entries emitted into this code object.
  const loom_amdgpu_hsaco_kernel_t* kernels;
  // Number of entries in |kernels|.
  iree_host_size_t kernel_count;
} loom_amdgpu_hsaco_file_t;

// Writes |file| as an AMDGPU HSA code-object ELF to |stream|.
//
// The writer uses |scratch_arena| for all transient payloads assembled before
// streaming. The arena must remain live until this call returns and can be
// reset immediately after. The emitted object is self-contained and does not
// depend on LLVM, LLD, or HAL reader code.
iree_status_t loom_amdgpu_hsaco_write_file(
    const loom_amdgpu_hsaco_file_t* file, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_H_
