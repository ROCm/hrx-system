// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hsaco.h"

#include "iree/io/memory_stream.h"

// Returns the ELF envelope borrowing the trusted plan storage.
static loom_native_elf64le_file_t loom_amdgpu_hsaco_plan_file(
    const loom_amdgpu_hsaco_plan_t* plan) {
  return (loom_native_elf64le_file_t){
      .type = LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      .machine = LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      .os_abi = LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA,
      .abi_version = LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6,
      .flags = plan->elf_flags,
      .entry = 0,
      .sections = plan->sections,
      .section_count = plan->section_count,
      .segments = plan->segments,
      .segment_count = LOOM_AMDGPU_HSACO_PLAN_SEGMENT_COUNT,
  };
}

iree_status_t loom_amdgpu_hsaco_write_plan(
    const loom_amdgpu_hsaco_plan_t* plan, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  const loom_native_elf64le_file_t elf_file = loom_amdgpu_hsaco_plan_file(plan);
  loom_native_elf_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(
      loom_native_elf64le_build_layout(&elf_file, scratch_arena, &layout));
  return loom_native_elf64le_write_file(&elf_file, &layout, stream);
}

iree_status_t loom_amdgpu_hsaco_serialize_plan(
    const loom_amdgpu_hsaco_plan_t* plan, iree_allocator_t allocator,
    iree_arena_allocator_t* scratch_arena,
    iree_byte_sequence_t** out_sequence) {
  *out_sequence = NULL;
  const loom_native_elf64le_file_t elf_file = loom_amdgpu_hsaco_plan_file(plan);
  loom_native_elf_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(
      loom_native_elf64le_build_layout(&elf_file, scratch_arena, &layout));
  if (layout.file_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSACO file size exceeds host limits");
  }

  iree_byte_span_t storage = {
      .data_length = (iree_host_size_t)layout.file_size,
  };
  iree_status_t status = iree_allocator_malloc_uninitialized(
      allocator, storage.data_length, (void**)&storage.data);
  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_memory_stream_wrap(IREE_IO_STREAM_MODE_WRITABLE, storage,
                                        iree_io_stream_release_callback_null(),
                                        allocator, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_native_elf64le_write_file(&elf_file, &layout, stream);
  }
  iree_io_stream_release(stream);
  if (iree_status_is_ok(status)) {
    status = iree_byte_sequence_create_from_span_move(&storage, allocator,
                                                      out_sequence);
  }
  iree_allocator_free(allocator, storage.data);
  return status;
}
