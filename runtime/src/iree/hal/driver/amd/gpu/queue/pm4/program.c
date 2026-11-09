// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/program.h"

#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/status.h"

iree_status_t iree_hal_amd_gpu_pm4_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, bool chain, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_pm4_program_pool_t){.chain = chain};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_pool_initialize(
      context, native_family_ordinal, queue_count,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
          AMDF_MEMORY_ACCESS_EXECUTE,
      host_allocator, &out_pool->storage));
  iree_hal_amd_gpu_gfxip_version_t gfxip = {
      context->target_info.gfx_ip.major, context->target_info.gfx_ip.minor,
      context->target_info.gfx_ip.stepping};
  iree_const_byte_span_t code[2 + IREE_HAL_AMD_GPU_CONTROL_PROGRAM_COUNT];
  iree_hal_amd_gpu_image_kernel_t kernels[IREE_ARRAYSIZE(code)];
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_fixup_program_query(gfxip, &code[0], &kernels[0]));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_blit_program_query(gfxip, &code[1], &kernels[1]));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < IREE_HAL_AMD_GPU_CONTROL_PROGRAM_COUNT && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_control_program_query(gfxip, i, &code[2 + i],
                                                    &kernels[2 + i]);
  }
  IREE_RETURN_IF_ERROR(status);
  uint64_t byte_length =
      256;  // Instruction fetch padding after the final entry.
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(code); ++i) {
    byte_length += iree_align_uint64(code[i].data_length, 256);
  }
  iree_byte_span_t contents;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_code_allocate(
      context, &out_pool->storage.resident, byte_length, 256, host_allocator,
      &out_pool->code, &out_pool->code_upload, &contents));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_pool->code.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "query program builtin address"));
  const amdf_queue_format_features_t format_features =
      context->families[native_family_ordinal].format_features;
  memset(contents.data, 0, byte_length);
  uint64_t offset = 0;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(code) && iree_status_is_ok(status);
       ++i) {
    memcpy(contents.data + offset, code[i].data, code[i].data_length);
    kernels[i].entry_offset = offset;
    iree_hal_amd_gpu_pm4_launch_t* launch = i == 0 ? &out_pool->fixup
                                            : i == 1
                                                ? &out_pool->blit
                                                : &out_pool->control[i - 2];
    status = iree_hal_amd_gpu_pm4_launch_initialize(
        gfxip, format_features, &kernels[i], address, launch);
    offset += iree_align_uint64(code[i].data_length, 256);
  }
  IREE_RETURN_IF_ERROR(status);
  return out_pool->code_upload
             ? iree_ok_status()
             : iree_hal_amd_gpu_code_publish(context, NULL, &out_pool->code,
                                             &out_pool->code_upload);
}

void iree_hal_amd_gpu_pm4_program_pool_deinitialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool) {
  if (!pool->storage.context) {
    return;
  }
  iree_hal_amd_gpu_code_upload_free(pool->code_upload);
  iree_hal_amd_fabric_memory_deinitialize(pool->storage.context->api,
                                          pool->storage.context->event_sink,
                                          &pool->code);
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

uint32_t* iree_hal_amd_gpu_pm4_control_program_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t argument_address) {
  const uint32_t one[3] = {1, 1, 1};
  const uint32_t partial[3] = {0, 0, 0};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .kernarg_pointer = argument_address,
  };
  words =
      iree_hal_amd_gpu_pm4_launch_emit(words, launch, one, partial, 0, &inputs);
  return iree_hal_amd_gpu_pm4_emit_dispatch_direct(words, one,
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

uint32_t iree_hal_amd_gpu_pm4_fixup_packet_word_count(
    const iree_hal_amd_gpu_fixup_t* entries, uint32_t entry_count) {
  uint32_t word_count = entry_count;
  for (uint32_t i = 0; i < entry_count; ++i) {
    if (entries[i].target_offset & 3u) {
      return 0;
    }
    if (!i || entries[i].target_offset != entries[i - 1].target_offset + 4) {
      word_count += 4;
    }
  }
  return word_count;
}

uint32_t* iree_hal_amd_gpu_pm4_fixup_packets_emit(
    const iree_hal_amd_gpu_fixup_t* entries, uint32_t entry_count,
    const uint64_t* values, uint64_t target_address, uint32_t* words) {
  uint32_t payload[IREE_HAL_AMD_GPU_PM4_PACKET_FIXUP_CAPACITY];
  for (uint32_t first = 0; first < entry_count;) {
    uint32_t end = first + 1;
    while (end < entry_count &&
           entries[end].target_offset == entries[end - 1].target_offset + 4) {
      ++end;
    }
    for (uint32_t i = first; i < end; ++i) {
      const iree_hal_amd_gpu_fixup_t entry = entries[i];
      payload[i - first] =
          (uint32_t)((values[entry.source & 0xFFFFFFu] + entry.addend) >>
                     ((entry.source >> 24) & 63));
    }
    words = iree_hal_amd_gpu_pm4_emit_write(
        words, target_address + entries[first].target_offset, end - first,
        payload);
    first = end;
  }
  return words;
}

uint32_t iree_hal_amd_gpu_pm4_program_word_count(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    uint32_t fixup_word_count) {
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
    count += (instance->join_previous ? 2 : 0) +
             (fixup_word_count ? fixup_word_count : 31 + 2) + 8;
  }
  return count;
}

uint32_t* iree_hal_amd_gpu_pm4_program_emit(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    iree_hal_amd_gpu_pm4_program_block_t entry,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    uint64_t argument_address,
    iree_hal_amd_gpu_pm4_program_arguments_t* arguments,
    uint32_t fixup_word_count, const uint32_t* fixup_words, uint32_t padding,
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
      // A shader-produced clone must reach bypassing CP clients before either
      // packet fixups overwrite it or command fetch reads the unpatched body.
      words = iree_hal_amd_gpu_pm4_emit_acquire(
          words, profile->shader |
                     (!fixup || fixup_word_count ? command_publication : 0));
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
  if (fixup_word_count) {
    memcpy(words, fixup_words, fixup_word_count * sizeof(*words));
    words += fixup_word_count;
    words = iree_hal_amd_gpu_pm4_emit_acquire(
        words, profile->command_processor_bypasses_l2 ? profile->acquire
                                                      : profile->shader);
  } else if (fixup) {
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
