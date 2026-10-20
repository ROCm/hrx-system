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
      {.argument_length =
           sizeof(iree_hal_amd_gpu_aql_materialize_profile_arguments_t),
       .kernel_object = &out_pool->materialize_profile},
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_blit_program_query(
      gfxip, &builtins[0].code, &builtins[0].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_fixup_program_query(
      gfxip, &builtins[1].code, &builtins[1].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_materialize_program_query(
      gfxip, false, &builtins[2].code, &builtins[2].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_indirect_program_query(
      gfxip, &builtins[3].code, &builtins[3].kernel));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_materialize_program_query(
      gfxip, true, &builtins[4].code, &builtins[4].kernel));
  const uint64_t descriptor_length = 64;
  const uint64_t code_begin =
      iree_align_uint64(IREE_ARRAYSIZE(builtins) * descriptor_length, 256);
  uint64_t byte_length = code_begin;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(builtins); ++i) {
    byte_length += iree_align_uint64(builtins[i].code.data_length, 256);
  }
  byte_length += 256;  // Instruction fetch padding after the final kernel.
  const iree_hal_amd_fabric_memory_options_t* code_memory =
      out_pool->storage.resident.host_access ? &out_pool->storage.resident
                                             : &out_pool->storage.upload;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, code_memory, byte_length, 256, context->event_sink,
      &out_pool->code, NULL));
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

uint32_t iree_hal_amd_gpu_aql_program_prefix_count(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance, bool first) {
  return 1 + (first && !instance->initialized) +
         (first && program->fixup_count);
}

static iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_program_dispatch(
    uint64_t kernel, uint64_t arguments, uint32_t groups) {
  const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
      .header = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16) |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT,
      .workgroup_size = {64, 1, 1},
      .grid_size = {groups * 64, 1, 1},
      .kernel_object = kernel,
      .kernarg_address = arguments,
  };
  return iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
}

void iree_hal_amd_gpu_aql_program_emit(
    const iree_hal_amd_gpu_aql_program_pool_t* pool,
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance, uint32_t block_offset,
    uint32_t packet_count, iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    uint64_t argument_address, uint64_t signal_address, void* upload,
    iree_hal_amd_gpu_aql_program_flags_t flags) {
  const bool profiling = (flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_PROFILE) != 0;
  iree_hal_amd_gpu_aql_packet_t prefix[3];
  uint32_t count = 0;
  iree_hal_amd_gpu_aql_materialize_arguments_t* materialize = upload;
  uint64_t materialize_address = argument_address;
  if (flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_FIRST) {
    iree_hal_amd_gpu_aql_program_arguments_t* arguments = upload;
    const iree_hal_amd_gpu_program_source_t source =
        iree_hal_amd_gpu_program_source(program);
    if (flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_INITIALIZE) {
      arguments->copy = (iree_hal_amd_gpu_blit_arguments_t){
          .source = source.address,
          .target = instance->address,
          .length = instance->ordinal
                        ? program->byte_length
                        : program->fixup_offset +
                              (uint64_t)program->fixup_count *
                                  sizeof(iree_hal_amd_gpu_fixup_t),
      };
      prefix[count++] = iree_hal_amd_gpu_aql_program_dispatch(
          pool->blit, argument_address, 256);
    }
    if (program->fixup_count) {
      arguments->fixup = (iree_hal_amd_gpu_fixup_arguments_t){
          .entries = source.fixups,
          .values =
              argument_address + sizeof(*arguments) +
              (profiling
                   ? sizeof(
                         iree_hal_amd_gpu_aql_materialize_profile_arguments_t) -
                         sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t)
                   : 0),
          .target = instance->address,
          .entry_count = program->fixup_count,
      };
      prefix[count++] = iree_hal_amd_gpu_aql_program_dispatch(
          pool->fixup,
          argument_address +
              offsetof(iree_hal_amd_gpu_aql_program_arguments_t, fixup),
          (program->fixup_count + 63) / 64);
    }
    materialize = &arguments->materialize;
    materialize_address +=
        offsetof(iree_hal_amd_gpu_aql_program_arguments_t, materialize);
  }
  const uint64_t source = instance->address + block_offset;
  *materialize = (iree_hal_amd_gpu_aql_materialize_arguments_t){
      .ring = ring->device_address,
      .packets = source,
      .ring_address = ring->device_address,
      .packet_address = source,
      .ring_mask = ring->capacity_mask,
      .first = (uint32_t)(position + count + 1) & ring->capacity_mask,
      .packet_count = packet_count,
  };
  if (profiling) {
    iree_hal_amd_gpu_aql_materialize_profile_arguments_t* profile =
        (void*)materialize;
    profile->signals = signal_address;
    profile->signal_address = signal_address;
  }
  prefix[count++] = iree_hal_amd_gpu_aql_materialize_packet(
      profiling ? pool->materialize_profile : pool->materialize,
      materialize_address,
      IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
          IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
  // Every argument is complete before a valid header can run ahead of the host.
  for (uint32_t i = 0; i < count; ++i) {
    iree_hal_amd_gpu_aql_ring_write_body(ring, position + i, &prefix[i]);
    iree_hal_amd_gpu_aql_ring_publish_header(ring, position + i,
                                             prefix[i].words[0]);
  }
}
