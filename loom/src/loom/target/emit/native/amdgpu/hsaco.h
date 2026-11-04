// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code-object serialization.
//
// This layer consumes a trusted, fully resolved code-object plan. All target
// policy, payload construction, symbol resolution, and virtual-address layout
// happen before this boundary.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/target/emit/native/elf.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Maximum number of final ELF sections in an AMDGPU HSACO plan.
  LOOM_AMDGPU_HSACO_PLAN_SECTION_CAPACITY = 10,
  // Number of final ELF program segments in an AMDGPU HSACO plan.
  LOOM_AMDGPU_HSACO_PLAN_SEGMENT_COUNT = 7,
};

// Trusted final AMDGPU code-object plan consumed by the serializer.
//
// Section payloads borrow storage owned by the plan arena. All section
// and segment indices, addresses, links, and target flags are final.
typedef struct loom_amdgpu_hsaco_plan_t {
  // Processor-specific ELF e_flags.
  uint32_t elf_flags;
  // Exact executable `.text` extent including inter-symbol alignment.
  uint64_t executable_code_byte_length;
  // Compact array of final code-object sections.
  loom_native_elf_section_t sections[LOOM_AMDGPU_HSACO_PLAN_SECTION_CAPACITY];
  // Number of initialized entries in |sections|.
  iree_host_size_t section_count;
  // Final code-object program segments.
  loom_native_elf_segment_t segments[LOOM_AMDGPU_HSACO_PLAN_SEGMENT_COUNT];
} loom_amdgpu_hsaco_plan_t;

// Serializes trusted |plan| into |stream|.
//
// The writer performs no AMDGPU semantic discovery or validation. Generic ELF
// layout scratch storage uses |scratch_arena| and can be reset after return.
iree_status_t loom_amdgpu_hsaco_write_plan(
    const loom_amdgpu_hsaco_plan_t* plan, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

// Serializes trusted |plan| into one exact-size contiguous byte sequence.
//
// Generic artifact APIs may preserve segmented producer storage, but native
// executable loading requires one contiguous span. Producing that span here
// lets native executable loaders that accept borrowed spans avoid a second
// complete code-object allocation and copy at load time.
iree_status_t loom_amdgpu_hsaco_serialize_plan(
    const loom_amdgpu_hsaco_plan_t* plan, iree_allocator_t allocator,
    iree_arena_allocator_t* scratch_arena, iree_byte_sequence_t** out_sequence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HSACO_H_
