// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/status.h"

static iree_status_t iree_hal_amd_gpu_pm4_program_select_memory(
    iree_hal_amd_gpu_pm4_program_pool_t* pool, amdf_memory_scope_t* scope,
    bool upload, bool* out_found,
    iree_hal_amd_fabric_memory_options_t* out_options) {
  *out_found = false;
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      pool->context->api->memory_scope_query_info(scope, &info),
      "query resident program scope"));
  if (!upload && info.kind != AMDF_MEMORY_SCOPE_KIND_LOCAL) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < info.memory_profile_count; ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(access),
    };
    amdf_status_t status =
        pool->context->api->memory_scope_query_device_profile(
            scope, i, 1, &pool->access, &profile, &access);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        status, "qualify resident program profile"));
    const bool mapped =
        iree_all_bits_set(profile.roles, AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        iree_all_bits_set(profile.supported_flags,
                          AMDF_MEMORY_FLAG_HOST_VISIBLE);
    if (!iree_all_bits_set(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE) ||
        (upload && !mapped) ||
        (!upload && !iree_all_bits_set(profile.supported_flags,
                                       AMDF_MEMORY_FLAG_DEVICE_LOCAL))) {
      continue;
    }
    *out_options = (iree_hal_amd_fabric_memory_options_t){
        .scope = scope,
        .profile = profile,
        .access_count = 1,
        .accesses = &pool->access,
        .required_flags = (mapped ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0) |
                          (upload ? 0 : AMDF_MEMORY_FLAG_DEVICE_LOCAL),
        .host_access = mapped ? AMDF_MEMORY_MAP_FLAG_WRITE : 0,
    };
    *out_found = true;
    break;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_program_pool_initialize(
    const iree_hal_amd_gpu_context_t* context,
    iree_hal_memory_maintenance_t* service, uint32_t queue_count, bool chain,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_pm4_program_pool_t){
      .context = context,
      .service = service,
      .host_allocator = host_allocator,
      .access = {.device = context->device,
                 .requirements =
                     {
                         .access = AMDF_MEMORY_ACCESS_READ |
                                   AMDF_MEMORY_ACCESS_WRITE |
                                   AMDF_MEMORY_ACCESS_EXECUTE,
                         .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                         .address_kinds = UINT64_C(1)
                                          << AMDF_MEMORY_ADDRESS_GPU,
                     }},
      .queue_count = queue_count,
      .chain = chain,
  };
  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_select_memory(
      out_pool, context->fabric->system_scope, true, &found,
      &out_pool->upload));
  if (!found) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "GPU has no host-visible cached program profile");
  }
  out_pool->resident = out_pool->upload;
  found = false;
  for (uint32_t i = 0; i < context->memory_scope_count && !found; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_select_memory(
        out_pool, context->memory_scopes[i], false, &found,
        &out_pool->resident));
  }
  iree_hal_amd_gpu_gfxip_version_t gfxip = {
      context->target_info.gfx_ip.major, context->target_info.gfx_ip.minor,
      context->target_info.gfx_ip.stepping};
  iree_const_byte_span_t fixup_code, blit_code;
  iree_hal_amd_gpu_image_kernel_t fixup_kernel, blit_kernel;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_fixup_program_query(
      gfxip, &fixup_code, &fixup_kernel));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_pm4_blit_program_query(gfxip, &blit_code, &blit_kernel));
  const uint64_t blit_offset = iree_align_uint64(fixup_code.data_length, 256);
  const uint64_t byte_length =
      blit_offset + iree_align_uint64(blit_code.data_length, 256) + 256;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, &out_pool->upload, byte_length, 256, context->event_sink,
      &out_pool->code));
  memset(out_pool->code.contents.data, 0, byte_length);
  memcpy(out_pool->code.contents.data, fixup_code.data, fixup_code.data_length);
  memcpy(out_pool->code.contents.data + blit_offset, blit_code.data,
         blit_code.data_length);
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(out_pool->code.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, byte_length),
      "publish resident program builtins"));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_pool->code.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "query program builtin address"));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_launch_initialize(
      gfxip, &fixup_kernel, address, &out_pool->fixup));
  blit_kernel.entry_offset = blit_offset;
  return iree_hal_amd_gpu_pm4_launch_initialize(gfxip, &blit_kernel, address,
                                                &out_pool->blit);
}

void iree_hal_amd_gpu_pm4_program_pool_trim(
    iree_hal_amd_gpu_pm4_program_pool_t* pool) {
  for (iree_hal_amd_gpu_pm4_program_t* program = pool->head; program;
       program = program->next) {
    iree_hal_amd_gpu_pm4_program_trim(program);
  }
}

void iree_hal_amd_gpu_pm4_program_pool_deinitialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool) {
  if (!pool->context) {
    return;
  }
  IREE_ASSERT(!pool->head);
  iree_hal_amd_fabric_memory_deinitialize(
      pool->context->api, pool->context->event_sink, &pool->code);
}

static void iree_hal_amd_gpu_pm4_program_instance_free(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  if (!instance) {
    return;
  }
  const iree_hal_amd_gpu_context_t* context = program->pool->context;
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &instance->memory);
  iree_allocator_free(program->pool->host_allocator, instance);
}

static iree_status_t iree_hal_amd_gpu_pm4_program_instance_allocate(
    iree_hal_amd_gpu_pm4_program_t* program, uint32_t ordinal,
    uint64_t byte_length, bool mapped,
    iree_hal_amd_gpu_pm4_program_instance_t** out_instance) {
  *out_instance = NULL;
  iree_hal_amd_gpu_pm4_program_pool_t* pool = program->pool;
  iree_hal_amd_gpu_pm4_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator, sizeof(*instance), (void**)&instance));
  instance->ordinal = ordinal;
  iree_hal_amd_fabric_memory_options_t options = pool->resident;
  if (!mapped) {
    options.host_access = 0;
    options.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  }
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      pool->context->api, &options, byte_length, program->alignment,
      pool->context->event_sink, &instance->memory);
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->memory_query_address(instance->memory.handle, 0,
                                                 AMDF_MEMORY_ADDRESS_GPU,
                                                 &instance->address),
        "query resident program address");
  }
  if (iree_status_is_ok(status)) {
    *out_instance = instance;
  } else {
    iree_hal_amd_gpu_pm4_program_instance_free(program, instance);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_pm4_program_initialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool, uint32_t alignment,
    uint32_t byte_length, uint32_t fixup_count,
    iree_hal_amd_gpu_pm4_program_t* out_program) {
  *out_program = (iree_hal_amd_gpu_pm4_program_t){
      .pool = pool,
      .capacity = pool->queue_count,
      .byte_length = byte_length,
      .alignment = alignment,
      .fixup_count = fixup_count,
      .fixup_offset = (uint32_t)iree_align_uint64(byte_length, 16),
  };
  iree_slim_mutex_initialize(&out_program->mutex);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator,
      2 * pool->queue_count * sizeof(*out_program->instances),
      (void**)&out_program->instances));
  out_program->preferred = out_program->instances + pool->queue_count;
  const uint64_t allocation_length =
      out_program->fixup_offset +
      (uint64_t)fixup_count * sizeof(iree_hal_amd_gpu_pm4_fixup_t);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_instance_allocate(
      out_program, 0, allocation_length, true, &out_program->instances[0]));
  iree_hal_amd_gpu_pm4_program_instance_t* initial = out_program->instances[0];
  if (initial->memory.contents.data) {
    out_program->contents = initial->memory.contents;
    out_program->upload_address = initial->address;
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
        pool->context->api, &pool->upload, allocation_length, alignment,
        pool->context->event_sink, &out_program->upload));
    out_program->contents = out_program->upload.contents;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->memory_query_address(out_program->upload.handle, 0,
                                                 AMDF_MEMORY_ADDRESS_GPU,
                                                 &out_program->upload_address),
        "query resident program upload address"));
  }
  for (uint32_t i = 0; i < pool->queue_count; ++i) {
    out_program->preferred[i] = initial;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_program_seal(
    iree_hal_amd_gpu_pm4_program_t* program) {
  iree_hal_amd_gpu_pm4_program_instance_t* initial = program->instances[0];
  amdf_host_mapping_t* mapping = program->upload.mapping
                                     ? program->upload.mapping
                                     : initial->memory.mapping;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      program->pool->context->api->host_mapping_cache_control(
          mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
          program->fixup_offset + (uint64_t)program->fixup_count *
                                      sizeof(iree_hal_amd_gpu_pm4_fixup_t)),
      "publish sealed resident program"));
  if (!program->upload.handle) {
    initial->initialized = true;
    iree_atomic_store(&program->source_ready, 1, iree_memory_order_release);
  }
  program->next = program->pool->head;
  if (program->next) {
    program->next->previous = program;
  }
  program->pool->head = program;
  return iree_ok_status();
}

iree_hal_amd_gpu_pm4_program_source_t iree_hal_amd_gpu_pm4_program_source(
    const iree_hal_amd_gpu_pm4_program_t* program) {
  const uint64_t address =
      iree_atomic_load(&program->source_ready, iree_memory_order_acquire)
          ? program->instances[0]->address
          : program->upload_address;
  return (iree_hal_amd_gpu_pm4_program_source_t){
      .address = address,
      .fixups = address + program->fixup_offset,
  };
}

static uint64_t iree_hal_amd_gpu_pm4_program_copy_length(
    const iree_hal_amd_gpu_pm4_program_t* program,
    const iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  return instance->ordinal
             ? program->byte_length
             : program->fixup_offset + (uint64_t)program->fixup_count *
                                           sizeof(iree_hal_amd_gpu_pm4_fixup_t);
}

uint32_t iree_hal_amd_gpu_pm4_program_word_count(
    const iree_hal_amd_gpu_pm4_program_t* program,
    const iree_hal_amd_gpu_pm4_program_instance_t* instance,
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
    const iree_hal_amd_gpu_pm4_program_t* program,
    const iree_hal_amd_gpu_pm4_program_instance_t* instance,
    iree_hal_amd_gpu_pm4_program_block_t entry,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    uint64_t argument_address,
    iree_hal_amd_gpu_pm4_program_arguments_t* arguments, uint32_t padding,
    uint32_t* words) {
  const bool fixup = program->fixup_count != 0;
  const uint32_t command_publication =
      profile->command_processor_bypasses_l2 ? profile->release : 0;
  const iree_hal_amd_gpu_pm4_program_source_t source =
      iree_hal_amd_gpu_pm4_program_source(program);
  arguments->fixup = (iree_hal_amd_gpu_pm4_fixup_arguments_t){
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
      arguments->copy = (iree_hal_amd_gpu_pm4_blit_arguments_t){
          .source = source.address,
          .target = instance->address,
          .length = length,
      };
      words = iree_hal_amd_gpu_pm4_blit_emit(
          words, &program->pool->blit,
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
        words, &program->pool->fixup, argument_address, program->fixup_count);
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
    words = iree_hal_amd_gpu_pm4_emit_acquire(
        words, profile->shader | command_publication);
  }
  if (padding) {
    words = iree_hal_amd_gpu_pm4_emit_nop(words, padding);
  }
  return iree_hal_amd_gpu_pm4_emit_indirect_buffer(
      words, instance->address + entry.offset, entry.word_count,
      program->pool->chain);
}

static bool iree_hal_amd_gpu_pm4_program_instance_available(
    const iree_hal_amd_gpu_pm4_program_instance_t* instance,
    const uint64_t* frontier) {
  if (instance->reserved) {
    return false;
  }
  if (!instance->use_count) {
    return true;
  }
  // Only the last writer is needed because every reuse is ordered after that
  // writer. Clone reads of the initial instance do not constrain its mutable
  // words and cannot release its permanent ownership.
  return frontier[instance->last_queue_ordinal] >= instance->last_epoch;
}

iree_hal_amd_gpu_pm4_program_instance_t*
iree_hal_amd_gpu_pm4_program_try_acquire(
    iree_hal_amd_gpu_pm4_program_t* program, uint32_t queue_ordinal,
    const uint64_t* frontier) {
  iree_slim_mutex_lock(&program->mutex);
  iree_hal_amd_gpu_pm4_program_instance_t* instance =
      program->preferred[queue_ordinal];
  if (!instance ||
      !iree_hal_amd_gpu_pm4_program_instance_available(instance, frontier)) {
    instance = NULL;
    for (uint32_t i = 0; i < program->capacity && !instance; ++i) {
      iree_hal_amd_gpu_pm4_program_instance_t* candidate =
          program->instances[i];
      if (candidate && iree_hal_amd_gpu_pm4_program_instance_available(
                           candidate, frontier)) {
        instance = candidate;
      }
    }
  }
  if (instance) {
    instance->join_previous =
        instance->use_count && instance->last_queue_ordinal == queue_ordinal;
    ++instance->use_count;
    instance->reserved = true;
    program->preferred[queue_ordinal] = instance;
  }
  iree_slim_mutex_unlock(&program->mutex);
  return instance;
}

iree_status_t iree_hal_amd_gpu_pm4_program_grow(
    iree_hal_amd_gpu_pm4_program_t* program) {
  uint32_t ordinal = program->capacity;
  bool idle = false;
  iree_slim_mutex_lock(&program->mutex);
  for (uint32_t i = 0; i < program->capacity; ++i) {
    if (program->instances[i]) {
      idle |= !program->instances[i]->use_count;
    } else if (ordinal == program->capacity) {
      ordinal = i;
    }
  }
  iree_slim_mutex_unlock(&program->mutex);
  if (idle) {
    return iree_ok_status();
  }
  // One unpublished use per queue, with every published FIFO prefix in its
  // frontier, bounds independent instances by the provisioned queue count.
  IREE_ASSERT_LT(ordinal, program->capacity);
  iree_hal_amd_gpu_pm4_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_instance_allocate(
      program, ordinal, program->byte_length, false, &instance));
  iree_slim_mutex_lock(&program->mutex);
  program->instances[ordinal] = instance;
  iree_slim_mutex_unlock(&program->mutex);
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_program_publish(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance, uint32_t queue_ordinal,
    uint64_t epoch) {
  iree_slim_mutex_lock(&program->mutex);
  instance->initialized = true;
  instance->last_queue_ordinal = queue_ordinal;
  instance->last_epoch = epoch;
  instance->reserved = false;
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_cancel(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  instance->reserved = false;
  --instance->use_count;
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_release(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  --instance->use_count;
  if (!instance->ordinal) {
    iree_atomic_store(&program->source_ready, 1, iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_trim(
    iree_hal_amd_gpu_pm4_program_t* program) {
  for (uint32_t i = 1; i < program->capacity; ++i) {
    iree_slim_mutex_lock(&program->mutex);
    iree_hal_amd_gpu_pm4_program_instance_t* instance = program->instances[i];
    if (instance && !instance->use_count) {
      program->instances[i] = NULL;
      for (uint32_t q = 0; q < program->capacity; ++q) {
        if (program->preferred[q] == instance) {
          program->preferred[q] = program->instances[0];
        }
      }
    } else {
      instance = NULL;
    }
    iree_slim_mutex_unlock(&program->mutex);
    iree_hal_amd_gpu_pm4_program_instance_free(program, instance);
  }
  bool idle = true;
  iree_slim_mutex_lock(&program->mutex);
  for (uint32_t i = 0; i < program->capacity; ++i) {
    idle &= !program->instances[i] || !program->instances[i]->use_count;
  }
  iree_slim_mutex_unlock(&program->mutex);
  if (idle &&
      iree_atomic_load(&program->source_ready, iree_memory_order_acquire)) {
    iree_hal_amd_fabric_memory_deinitialize(program->pool->context->api,
                                            program->pool->context->event_sink,
                                            &program->upload);
  }
}

void iree_hal_amd_gpu_pm4_program_deinitialize(
    iree_hal_amd_gpu_pm4_program_t* program) {
  if (!program->pool) {
    return;
  }
  if (program->previous) {
    program->previous->next = program->next;
  }
  if (program->next) {
    program->next->previous = program->previous;
  }
  if (program->pool->head == program) {
    program->pool->head = program->next;
  }
  iree_hal_amd_fabric_memory_deinitialize(program->pool->context->api,
                                          program->pool->context->event_sink,
                                          &program->upload);
  if (program->instances) {
    for (uint32_t i = 0; i < program->capacity; ++i) {
      iree_hal_amd_gpu_pm4_program_instance_free(program,
                                                 program->instances[i]);
    }
    iree_allocator_free(program->pool->host_allocator, program->instances);
  }
  iree_slim_mutex_deinitialize(&program->mutex);
}
