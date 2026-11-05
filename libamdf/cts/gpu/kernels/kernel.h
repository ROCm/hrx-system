// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_KERNEL_H_
#define AMDF_CTS_GPU_KERNELS_KERNEL_H_

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "amdf/gpu.h"
#include "libamdf/cts/gpu/kernels/image.h"

namespace kernels {

// Immutable compiler product and its launch contract. The generated definition
// owns all borrowed storage for the lifetime of the test executable.
struct Kernel {
  // Exact physical compiler selector.
  const char* target;
  // SHA-256 of the complete compiler code object before extraction.
  const char* hsaco_sha256;
  // Read-only data and text with their linked placement preserved.
  Image executable;
  // Entry point relative to the beginning of executable.
  uint32_t entry_byte_offset;
  // Function symbol extent, excluding the compiler's trailing fetch padding.
  uint32_t entry_byte_length;
  // Complete emitted text section, including trailing fetch padding.
  uint32_t text_byte_length;
  // Native kernarg layout emitted by the compiler.
  struct {
    // Complete segment extent, including fetch padding.
    uint32_t byte_length;
    // Required alignment of the segment base.
    uint32_t alignment;
    // Per-argument byte positions in declaration order.
    std::span<const uint32_t> byte_offsets;
    // Per-argument semantic byte extents.
    std::span<const uint32_t> byte_lengths;
    // Per-argument alignment requirements.
    std::span<const uint32_t> alignments;
    // Source argument names in declaration order.
    std::span<const std::string_view> names;
    // AMDHSA value kinds in declaration order.
    std::span<const std::string_view> value_kinds;
  } arguments;
  // Required workgroup XYZ; all zero when launch geometry is caller-defined.
  std::array<uint32_t, 3> required_workgroup_size;
  // Maximum flat workgroup size accepted by the compiled program.
  uint32_t maximum_workgroup_size;
  // Number of workitems in each compiled wave (32 or 64).
  uint32_t wavefront_size;
  // Fixed LDS bytes required by each workgroup.
  uint32_t group_segment_byte_length;
  // Fixed private bytes required by each workitem.
  uint32_t private_segment_byte_length;
  // Compiler-owned register settings and descriptor properties.
  struct {
    // COMPUTE_PGM_RSRC1, including register allocation and float mode.
    uint32_t resource1;
    // COMPUTE_PGM_RSRC2, including enabled user/system inputs.
    uint32_t resource2;
    // COMPUTE_PGM_RSRC3, interpreted by the target's command recipe.
    uint32_t resource3;
    // Allocated scalar register count.
    uint32_t scalar_register_count;
    // Allocated vector register count.
    uint32_t vector_register_count;
    // AMDHSA kernel_code_properties descriptor bits.
    uint32_t code_properties;
    // AMDHSA kernarg_preload descriptor bits.
    uint32_t argument_preload;
  } program;

  // Zero denotes caller-defined geometry, matching required_workgroup_size.
  uint32_t workgroup_size() const {
    return required_workgroup_size[0] * required_workgroup_size[1] *
           required_workgroup_size[2];
  }
};

// Target variants of one authored behavior. Selection is cold test setup;
// neither selection nor this object owns a native resource.
struct KernelSet {
  // Compiler products generated from the same source and entry point.
  std::span<const Kernel> variants;

  // Returns the exact physical product, or null when absent. Revision-specific
  // instruction sets never fall back to an incompatible base product.
  const Kernel* Find(const amdf_gpu_endpoint_info_t& endpoint) const;
};

}  // namespace kernels

#endif  // AMDF_CTS_GPU_KERNELS_KERNEL_H_
