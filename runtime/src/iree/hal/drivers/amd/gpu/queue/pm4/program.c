// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/status.h"

iree_status_t iree_hal_amd_gpu_pm4_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, bool chain, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_pm4_program_pool_t){.chain = chain};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_pool_initialize(
      context, native_family_ordinal, queue_count, host_allocator,
      &out_pool->storage));
  iree_hal_amd_gpu_gfxip_version_t gfxip = {
      context->target_info.gfx_ip.major, context->target_info.gfx_ip.minor,
      context->target_info.gfx_ip.stepping};
  iree_const_byte_span_t fixup_code, blit_code;
  iree_hal_amd_gpu_image_kernel_t fixup_kernel, blit_kernel;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_fixup_program_query(gfxip, &fixup_code, &fixup_kernel));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_blit_program_query(gfxip, &blit_code, &blit_kernel));
  const uint64_t blit_offset = iree_align_uint64(fixup_code.data_length, 256);
  const uint64_t byte_length =
      blit_offset + iree_align_uint64(blit_code.data_length, 256) + 256;
  iree_byte_span_t contents;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_code_allocate(
      context, &out_pool->storage.resident, &out_pool->storage.upload, byte_length, 256,
      host_allocator, &out_pool->code, &out_pool->code_upload, &contents));
  memset(contents.data, 0, byte_length);
  memcpy(contents.data, fixup_code.data, fixup_code.data_length);
  memcpy(contents.data + blit_offset, blit_code.data, blit_code.data_length);
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_pool->code.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "query program builtin address"));
  const amdf_queue_format_features_t format_features =
      context->families[native_family_ordinal].format_features;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_launch_initialize(
      gfxip, format_features, &fixup_kernel, address, &out_pool->fixup));
  blit_kernel.entry_offset = blit_offset;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_launch_initialize(
      gfxip, format_features, &blit_kernel, address, &out_pool->blit));
  return out_pool->code_upload ? iree_ok_status()
                               : iree_hal_amd_gpu_code_publish(
                                     context, NULL, 0, native_family_ordinal,
                                     &out_pool->code, &out_pool->code_upload);
}

void iree_hal_amd_gpu_pm4_program_pool_deinitialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool) {
  if (!pool->storage.context) {
    return;
  }
  iree_hal_amd_gpu_code_upload_free(pool->storage.context, pool->code_upload);
  iree_hal_amd_fabric_memory_deinitialize(
      pool->storage.context->api, pool->storage.context->event_sink, &pool->code);
  iree_hal_amd_gpu_program_pool_deinitialize(&pool->storage);
}

uint32_t* iree_hal_amd_gpu_pm4_fixup_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t arguments_address, uint32_t entry_count) {
  const uint32_t size[3] = {64, 1, 1};
  const uint32_t partial[3] = {0, 0, 0};
  const uint32_t groups[3] = {(entry_count - 1) / 64 + 1, 1, 1};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {.kernarg_pointer =
                                                           arguments_address};
  words = iree_hal_amd_gpu_pm4_launch_emit(words, launch, size, partial, 0,
                                           &inputs);
  return iree_hal_amd_gpu_pm4_emit_dispatch_direct(words, groups,
                                                   launch->dispatch_initiator);
}

uint32_t* iree_hal_amd_gpu_pm4_blit_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t argument_address) {
  const uint32_t size[3] = {64, 1, 1};
  const uint32_t partial[3] = {0, 0, 0};
  const uint32_t groups[3] = {256, 1, 1};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .kernarg_pointer = argument_address,
  };
  words = iree_hal_amd_gpu_pm4_launch_emit(words, launch, size, partial, 0,
                                           &inputs);
  return iree_hal_amd_gpu_pm4_emit_dispatch_direct(words, groups,
                                                   launch->dispatch_initiator);
}

static uint64_t iree_hal_amd_gpu_pm4_program_copy_length(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance) {
  return instance->ordinal
             ? program->byte_length
             : program->fixup_offset + (uint64_t)program->fixup_count *
                                           sizeof(iree_hal_amd_gpu_fixup_t);
}

uint32_t iree_hal_amd_gpu_pm4_program_word_count(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile) {
  uint32_t count = 8 + 4;  // Host input publication and resident IB entry.
  if (!instance->initialized) {
    if (iree_hal_amd_gpu_pm4_program_copy_length(program, instance) >
        UINT64_C(0x03ffffff)) {
      count += 31 + 2 + 8;  // Shader copy, execution join, and visibility.
    } else {
      count += 7 + 7;  // CP-DMA copy and execution join.
      if (!program->fixup_count && profile->command_processor_bypasses_l2) {
        count += 8;  // Publish copied commands to bypassing command fetch.
      }
    }
  }
  if (program->fixup_count) {
    count += (instance->join_previous ? 2 : 0) + 31 + 2 + 8;
  }
  return count;
}

uint32_t* iree_hal_amd_gpu_pm4_program_emit(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    iree_hal_amd_gpu_pm4_program_block_t entry,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    uint64_t argument_address,
    iree_hal_amd_gpu_pm4_program_arguments_t* arguments, uint32_t padding,
    uint32_t* words) {
  const iree_hal_amd_gpu_pm4_program_pool_t* pool = iree_containerof(
      program->pool, iree_hal_amd_gpu_pm4_program_pool_t, storage);
  const bool fixup = program->fixup_count != 0;
  const uint32_t command_publication =
      profile->command_processor_bypasses_l2 ? profile->release : 0;
  const iree_hal_amd_gpu_program_source_t source =
      iree_hal_amd_gpu_program_source(program);
  arguments->fixup = (iree_hal_amd_gpu_fixup_arguments_t){
      .entries = source.fixups,
      .values = argument_address + sizeof(*arguments),
      .target = instance->address,
      .entry_count = program->fixup_count,
  };
  if (instance->join_previous && fixup) {
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  }
  words = iree_hal_amd_gpu_pm4_emit_acquire(words, profile->acquire | 1u);
  if (!instance->initialized) {
    const uint64_t length =
        iree_hal_amd_gpu_pm4_program_copy_length(program, instance);
    if (length > UINT64_C(0x03ffffff)) {
      arguments->copy = (iree_hal_amd_gpu_blit_arguments_t){
          .source = source.address,
          .target = instance->address,
          .length = length,
      };
      words = iree_hal_amd_gpu_pm4_blit_emit(
          words, &pool->blit,
          argument_address +
              offsetof(iree_hal_amd_gpu_pm4_program_arguments_t, copy));
      words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
      words = iree_hal_amd_gpu_pm4_emit_acquire(
          words, profile->shader | (fixup ? 0 : command_publication));
    } else {
      words = iree_hal_amd_gpu_pm4_emit_dma_copy(
          words, source.address, instance->address, (uint32_t)length);
      words = iree_hal_amd_gpu_pm4_emit_dma_flush(words);
      if (!fixup && command_publication) {
        // CP-DMA writes through L2 even where command fetch bypasses it. An
        // immutable IB still needs publication after its cold native copy.
        words = iree_hal_amd_gpu_pm4_emit_acquire(words, command_publication);
      }
    }
  }
  if (fixup) {
    words = iree_hal_amd_gpu_pm4_fixup_emit(
        words, &pool->fixup, argument_address, program->fixup_count);
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
    words = iree_hal_amd_gpu_pm4_emit_acquire(
        words, profile->shader | command_publication);
  }
  if (padding) {
    words = iree_hal_amd_gpu_pm4_emit_nop(words, padding);
  }
  return iree_hal_amd_gpu_pm4_emit_indirect_buffer(
      words, instance->address + entry.offset, entry.word_count, pool->chain);
}
