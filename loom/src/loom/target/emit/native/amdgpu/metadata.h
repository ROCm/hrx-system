// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code object metadata emission.
//
// The HSA loader consumes an AMDGPU ELF note whose descriptor is MessagePack
// metadata. LLVM's assembler accepts the same information as a YAML-like
// `.amdgpu_metadata` assembly block and lowers it to the note payload. Loom
// owns both forms here so the temporary assembly path and future direct ELF
// writer share one metadata contract.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_METADATA_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_METADATA_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_metadata_argument_kind_e {
  // Argument bytes are copied directly into the kernarg segment.
  LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE = 0,
  // Argument is a pointer to global memory.
  LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER = 1,
  // Argument reserves hidden ABI space but does not need a value.
  LOOM_AMDGPU_METADATA_ARGUMENT_HIDDEN_NONE = 2,
} loom_amdgpu_metadata_argument_kind_t;

typedef struct loom_amdgpu_metadata_argument_t {
  // Source-level or ABI-level argument name, if one is available.
  iree_string_view_t name;
  // Byte offset in the kernel kernarg segment.
  uint32_t offset;
  // Byte length of the argument storage.
  uint32_t size;
  // Byte alignment when known, or 0 to omit the field.
  uint32_t alignment;
  // AMDGPU metadata `.value_kind` classification.
  loom_amdgpu_metadata_argument_kind_t kind;
  // AMDGPU address-space string such as `global`, or empty to omit.
  iree_string_view_t address_space;
  // Source access qualifier string such as `read_only`, or empty to omit.
  iree_string_view_t access;
  // Effective access qualifier string such as `read_only`, or empty to omit.
  iree_string_view_t actual_access;
} loom_amdgpu_metadata_argument_t;

typedef struct loom_amdgpu_metadata_kernel_t {
  // Kernel reflection name, usually the exported entry-point symbol.
  iree_string_view_t name;
  // Kernel descriptor symbol name, usually `<name>.kd`.
  iree_string_view_t descriptor_symbol;
  // Kernel kernarg segment size in bytes.
  uint32_t kernarg_segment_size;
  // Kernel kernarg segment alignment in bytes.
  uint32_t kernarg_segment_alignment;
  // HSA wavefront size used by the kernel.
  uint32_t wavefront_size;
  // Fixed LDS/group segment size in bytes.
  uint32_t group_segment_fixed_size;
  // Fixed scratch/private segment size in bytes.
  uint32_t private_segment_fixed_size;
  // Physical SGPR high-water count for the kernel body.
  uint32_t sgpr_count;
  // Physical VGPR high-water count for the kernel body.
  uint32_t vgpr_count;
  // Maximum flat workgroup size advertised for dispatch validation/tuning.
  uint32_t max_flat_workgroup_size;
  // Required workgroup size, when |has_required_workgroup_size| is true.
  loom_target_workgroup_size_t required_workgroup_size;
  // True when |required_workgroup_size| should be emitted.
  bool has_required_workgroup_size;
  // Argument records in kernarg offset order.
  const loom_amdgpu_metadata_argument_t* arguments;
  // Number of records in |arguments|.
  iree_host_size_t argument_count;
} loom_amdgpu_metadata_kernel_t;

typedef struct loom_amdgpu_code_object_metadata_t {
  // Full target ID string, such as `amdgcn-amd-amdhsa--gfx1100`.
  iree_string_view_t target;
  // Kernel records in this code object.
  const loom_amdgpu_metadata_kernel_t* kernels;
  // Number of records in |kernels|.
  iree_host_size_t kernel_count;
} loom_amdgpu_code_object_metadata_t;

// Appends an LLVM AMDGPU assembler `.amdgpu_metadata` block.
iree_status_t loom_amdgpu_metadata_append_assembly(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder);

// Appends the MessagePack payload for the AMDGPU ELF metadata note.
//
// The returned bytes are not a complete ELF note. Direct object emitters must
// wrap the payload in an `AMDGPU` note with type NT_AMDGPU_METADATA.
iree_status_t loom_amdgpu_metadata_append_msgpack(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder);

// Appends one complete AMDGPU ELF note record containing MessagePack metadata.
//
// The returned bytes are not a complete ELF file, section, or program segment.
// Direct object emitters can place the record in a SHT_NOTE section and expose
// the same bytes through a PT_NOTE segment. The note name is `AMDGPU` and the
// note type is NT_AMDGPU_METADATA.
iree_status_t loom_amdgpu_metadata_append_elf_note(
    const loom_amdgpu_code_object_metadata_t* metadata,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_METADATA_H_
