// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/program.h"

#include "iree/hal/drivers/amd/status.h"

// Cold placement input. Each descriptor names one aligned instruction range
// within the same allocation; no parser or runtime compiler is involved.
typedef struct iree_hal_amd_gpu_aql_builtin_t {
  // Relocation-free immutable instruction bytes.
  iree_const_byte_span_t code;
  // Native descriptor register and ABI facts.
  iree_hal_amd_gpu_image_kernel_t kernel;
  // Complete argument extent in bytes.
  uint32_t argument_length;
  // Pool field receiving the placed descriptor address.
  uint64_t* kernel_object;
} iree_hal_amd_gpu_aql_builtin_t;

iree_status_t iree_hal_amd_gpu_aql_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_aql_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_aql_program_pool_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_pool_initialize(
      context, native_family_ordinal, queue_count, host_allocator,
      &out_pool->storage));
  const iree_hal_amd_gpu_gfxip_version_t gfxip = {
      context->target_info.gfx_ip.major, context->target_info.gfx_ip.minor,
      context->target_info.gfx_ip.stepping};
  iree_hal_amd_gpu_aql_builtin_t builtins[] = {
      {.argument_length = sizeof(iree_hal_amd_gpu_blit_arguments_t),
       .kernel_object = &out_pool->blit},
      {.argument_length = sizeof(iree_hal_amd_gpu_fixup_arguments_t),
       .kernel_object = &out_pool->fixup},
      {.argument_length = sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t),
       .kernel_object = &out_pool->materialize},
      {.argument_length = sizeof(iree_hal_amd_gpu_aql_indirect_arguments_t),
       .kernel_object = &out_pool->indirect},
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_blit_program_query(
      gfxip, &builtins[0].code, &builtins[0].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_fixup_program_query(
      gfxip, &builtins[1].code, &builtins[1].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_materialize_program_query(
      gfxip, &builtins[2].code, &builtins[2].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_indirect_program_query(
      gfxip, &builtins[3].code, &builtins[3].kernel));
  const uint64_t descriptor_length = 64;
  const uint64_t code_begin =
      iree_align_uint64(IREE_ARRAYSIZE(builtins) * descriptor_length, 256);
  uint64_t byte_length = code_begin;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(builtins); ++i) {
    byte_length += iree_align_uint64(builtins[i].code.data_length, 256);
  }
  byte_length += 256;  // Instruction fetch padding after the final kernel.
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, &out_pool->storage.upload, byte_length, 256,
      context->event_sink, &out_pool->code, NULL));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_pool->code.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "query AQL builtin address"));
  uint8_t* contents = out_pool->code.contents.data;
  memset(contents, 0, byte_length);
  uint64_t code_offset = code_begin;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(builtins); ++i) {
    const iree_hal_amd_gpu_aql_builtin_t* builtin = &builtins[i];
    const uint64_t descriptor_offset = i * descriptor_length;
    uint8_t* descriptor = contents + descriptor_offset;
    // Fixed LDS/private sizes and all reserved descriptor bytes remain zero.
    iree_unaligned_store_le_u32(descriptor + 8, builtin->argument_length);
    iree_unaligned_store_le_u64(descriptor + 16,
                                code_offset - descriptor_offset);
    iree_unaligned_store_le_u32(descriptor + 44,
                                builtin->kernel.compute_pgm_rsrc3);
    iree_unaligned_store_le_u32(descriptor + 48,
                                builtin->kernel.compute_pgm_rsrc1);
    iree_unaligned_store_le_u32(descriptor + 52,
                                builtin->kernel.compute_pgm_rsrc2);
    iree_unaligned_store_le_u16(descriptor + 56,
                                builtin->kernel.code_properties);
    iree_unaligned_store_le_u16(descriptor + 58,
                                builtin->kernel.kernarg_preload);
    memcpy(contents + code_offset, builtin->code.data,
           builtin->code.data_length);
    *builtin->kernel_object = address + descriptor_offset;
    code_offset += iree_align_uint64(builtin->code.data_length, 256);
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(out_pool->code.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, byte_length),
      "publish AQL builtins");
}

void iree_hal_amd_gpu_aql_program_pool_deinitialize(
    iree_hal_amd_gpu_aql_program_pool_t* pool) {
  if (!pool->storage.context) {
    return;
  }
  iree_hal_amd_fabric_memory_deinitialize(
      pool->storage.context->api, pool->storage.context->event_sink, &pool->code);
  iree_hal_amd_gpu_program_pool_deinitialize(&pool->storage);
}
