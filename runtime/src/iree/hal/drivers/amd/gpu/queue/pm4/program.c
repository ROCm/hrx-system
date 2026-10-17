// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/program.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/status.h"

static_assert(sizeof(iree_hal_amd_gpu_pm4_program_queue_t) <= 32,
              "two publisher candidates must fit in one cache line");

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
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, bool chain, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_pm4_program_pool_t){
      .context = context,
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
  iree_slim_mutex_initialize(&out_pool->registry_mutex);
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
  const iree_hal_amd_fabric_memory_request_t request = {
      .options = &out_pool->upload,
      .byte_length = byte_length,
      .alignment = 256};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_acquire_memory_sync(
      context, 1, &request, &out_pool->code));
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
  const amdf_queue_format_features_t format_features =
      context->families[native_family_ordinal].format_features;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_launch_initialize(
      gfxip, format_features, &fixup_kernel, address, &out_pool->fixup));
  blit_kernel.entry_offset = blit_offset;
  return iree_hal_amd_gpu_pm4_launch_initialize(
      gfxip, format_features, &blit_kernel, address, &out_pool->blit);
}

void iree_hal_amd_gpu_pm4_program_pool_trim(
    iree_hal_amd_gpu_pm4_program_pool_t* pool) {
  iree_slim_mutex_lock(&pool->registry_mutex);
  for (iree_hal_amd_gpu_pm4_program_t* program = pool->head; program;
       program = program->next) {
    iree_hal_amd_gpu_pm4_program_trim(program);
  }
  iree_slim_mutex_unlock(&pool->registry_mutex);
}

void iree_hal_amd_gpu_pm4_program_pool_deinitialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool) {
  if (!pool->context) {
    return;
  }
  IREE_ASSERT(!pool->head);
  iree_hal_amd_gpu_context_release_memory_sync(pool->context, 1, &pool->code);
  iree_slim_mutex_deinitialize(&pool->registry_mutex);
  pool->context = NULL;
}

static void iree_hal_amd_gpu_pm4_program_instance_released(void* user_data) {
  iree_hal_amd_gpu_pm4_program_instance_t* instance = user_data;
  iree_allocator_free(instance->host_allocator, instance);
}

static void iree_hal_amd_gpu_pm4_program_instance_free(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  if (!instance) {
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      program->pool->context, 1, &instance->memory,
      (iree_hal_slab_release_callback_t){
          .fn = iree_hal_amd_gpu_pm4_program_instance_released,
          .user_data = instance},
      &instance->release);
}

struct iree_hal_amd_gpu_pm4_program_upload_t {
  // Native staging view retained through all readers of initial program bytes.
  iree_hal_amd_fabric_memory_t memory;
  // Release storage remains owned independently of command-buffer metadata.
  iree_hal_amd_gpu_memory_release_t release;
  // Device allocator owning this upload through native release completion.
  iree_allocator_t host_allocator;
};

static void iree_hal_amd_gpu_pm4_program_upload_released(void* user_data) {
  iree_hal_amd_gpu_pm4_program_upload_t* upload = user_data;
  iree_allocator_free(upload->host_allocator, upload);
}

static void iree_hal_amd_gpu_pm4_program_upload_free(
    iree_hal_amd_gpu_pm4_program_pool_t* pool,
    iree_hal_amd_gpu_pm4_program_upload_t* upload) {
  if (!upload) {
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      pool->context, 1, &upload->memory,
      (iree_hal_slab_release_callback_t){
          .fn = iree_hal_amd_gpu_pm4_program_upload_released,
          .user_data = upload},
      &upload->release);
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
  instance->host_allocator = pool->host_allocator;
  iree_hal_amd_fabric_memory_options_t options = pool->resident;
  if (!mapped) {
    options.host_access = 0;
    options.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  }
  const iree_hal_amd_fabric_memory_request_t request = {
      .options = &options,
      .byte_length = byte_length,
      .alignment = program->alignment,
  };
  iree_status_t status = iree_hal_amd_gpu_context_acquire_memory_sync(
      pool->context, 1, &request, &instance->memory);
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
      pool->queue_count *
          (sizeof(*out_program->queues) + sizeof(*out_program->instances)),
      (void**)&out_program->queues));
  out_program->instances = (void*)(out_program->queues + pool->queue_count);
  const uint64_t allocation_length =
      out_program->fixup_offset +
      (uint64_t)fixup_count * sizeof(iree_hal_amd_gpu_pm4_fixup_t);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_program_instance_allocate(
      out_program, 0, allocation_length, true, &out_program->instances[0]));
  iree_hal_amd_gpu_pm4_program_instance_t* initial = out_program->instances[0];
  out_program->initial = initial;
  if (initial->memory.contents.data) {
    out_program->contents = initial->memory.contents;
    out_program->upload_address = initial->address;
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                               sizeof(*out_program->upload),
                                               (void**)&out_program->upload));
    out_program->upload->host_allocator = pool->host_allocator;
    const iree_hal_amd_fabric_memory_request_t request = {
        .options = &pool->upload,
        .byte_length = allocation_length,
        .alignment = alignment,
    };
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_acquire_memory_sync(
        pool->context, 1, &request, &out_program->upload->memory));
    out_program->contents = out_program->upload->memory.contents;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->memory_query_address(
            out_program->upload->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU,
            &out_program->upload_address),
        "query resident program upload address"));
  }
  for (uint32_t i = 0; i < pool->queue_count; ++i) {
    out_program->queues[i].preferred = initial;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_program_seal(
    iree_hal_amd_gpu_pm4_program_t* program) {
  iree_hal_amd_gpu_pm4_program_instance_t* initial = program->initial;
  amdf_host_mapping_t* mapping = program->upload
                                     ? program->upload->memory.mapping
                                     : initial->memory.mapping;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      program->pool->context->api->host_mapping_cache_control(
          mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
          program->fixup_offset + (uint64_t)program->fixup_count *
                                      sizeof(iree_hal_amd_gpu_pm4_fixup_t)),
      "publish sealed resident program"));
  if (!program->upload) {
    initial->initialized = true;
    iree_atomic_store(&program->source_ready, 1, iree_memory_order_release);
  }
  iree_slim_mutex_lock(&program->pool->registry_mutex);
  program->next = program->pool->head;
  if (program->next) {
    program->next->previous = program;
  }
  program->pool->head = program;
  iree_slim_mutex_unlock(&program->pool->registry_mutex);
  return iree_ok_status();
}

iree_hal_amd_gpu_pm4_program_source_t iree_hal_amd_gpu_pm4_program_source(
    const iree_hal_amd_gpu_pm4_program_t* program) {
  const uint64_t address =
      iree_atomic_load(&program->source_ready, iree_memory_order_acquire)
          ? program->initial->address
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
      program->queues[queue_ordinal].preferred;
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
    program->queues[queue_ordinal].preferred = instance;
  } else {
    program->queues[queue_ordinal].observed_epoch = program->availability_epoch;
  }
  iree_slim_mutex_unlock(&program->mutex);
  return instance;
}

// Pending reservations and cancellation can leave several live instances with
// the same last writer. The queue count is a useful initial size, not a bound
// on physical overlap. Per-queue metadata never moves when this table grows.
static iree_status_t iree_hal_amd_gpu_pm4_program_reserve_instance(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance, bool* out_reserved) {
  *out_reserved = false;
  const iree_allocator_t allocator = program->pool->host_allocator;
  iree_hal_amd_gpu_pm4_program_instance_t** proposal = NULL;
  uint32_t proposal_capacity = 0;
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete && iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_program_instance_t** replaced = NULL;
    uint32_t required_capacity = 0;
    iree_slim_mutex_lock(&program->mutex);
    uint32_t ordinal = program->capacity;
    bool idle = false;
    for (uint32_t i = 0; i < program->capacity; ++i) {
      if (program->instances[i]) {
        idle |= !program->instances[i]->use_count &&
                !program->instances[i]->reserved;
      } else if (ordinal == program->capacity) {
        ordinal = i;
      }
    }
    if (idle) {
      complete = true;
    } else {
      if (ordinal == program->capacity &&
          proposal_capacity > program->capacity) {
        memcpy(proposal, program->instances,
               program->capacity * sizeof(*proposal));
        replaced = program->instances;
        program->instances = proposal;
        program->capacity = proposal_capacity;
        proposal = NULL;
        proposal_capacity = 0;
      }
      if (ordinal < program->capacity) {
        instance->ordinal = ordinal;
        program->instances[ordinal] = instance;
        *out_reserved = true;
        complete = true;
      } else {
        required_capacity = program->capacity;
      }
    }
    iree_slim_mutex_unlock(&program->mutex);
    if (replaced &&
        replaced != (void*)(program->queues + program->pool->queue_count)) {
      iree_allocator_free(allocator, replaced);
    }
    if (!complete) {
      iree_allocator_free(allocator, proposal);
      proposal = NULL;
      if (required_capacity > UINT32_MAX / 2) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "resident instance table exceeds its index space");
      } else {
        proposal_capacity = required_capacity * 2;
        status = iree_allocator_malloc_array(
            allocator, proposal_capacity, sizeof(*proposal), (void**)&proposal);
      }
    }
  }
  iree_allocator_free(allocator, proposal);
  return status;
}

iree_status_t iree_hal_amd_gpu_pm4_program_grow_begin(
    iree_hal_amd_gpu_pm4_program_t* program, uint32_t queue_ordinal,
    iree_hal_slab_acquire_callback_t callback,
    iree_hal_amd_gpu_pm4_program_growth_t* operation) {
  iree_hal_amd_gpu_pm4_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      program->pool->host_allocator, sizeof(*instance), (void**)&instance));
  instance->host_allocator = program->pool->host_allocator;
  instance->reserved = true;
  bool reserved = false;
  iree_status_t status = iree_hal_amd_gpu_pm4_program_reserve_instance(
      program, instance, &reserved);
  if (!reserved) {
    iree_allocator_free(instance->host_allocator, instance);
    instance = NULL;
  }
  IREE_RETURN_IF_ERROR(status);
  *operation = (iree_hal_amd_gpu_pm4_program_growth_t){
      .program = program,
      .instance = instance,
      .queue_ordinal = queue_ordinal,
      .options = program->pool->resident,
  };
  operation->options.host_access = 0;
  operation->options.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  program->queues[queue_ordinal].growth = operation;
  if (!instance) {
    callback.fn(callback.user_data, iree_ok_status());
    return iree_ok_status();
  }
  operation->request = (iree_hal_amd_fabric_memory_request_t){
      .options = &operation->options,
      .byte_length = program->byte_length,
      .alignment = program->alignment,
  };
  iree_hal_amd_gpu_context_acquire_memory(
      program->pool->context, 1, &operation->request, &instance->memory,
      callback, &operation->acquire);
  return iree_ok_status();
}

// All list access and callbacks occur under the program metadata mutex. A
// callback only posts a typed hint; it cannot reenter this program. Unlinking
// therefore joins callback access without another borrower/refcount protocol.
static void iree_hal_amd_gpu_pm4_program_unlink_wait(
    iree_hal_amd_gpu_pm4_program_growth_t* operation) {
  if (operation->previous_wait) {
    *operation->previous_wait = operation->next_wait;
    if (operation->next_wait) {
      operation->next_wait->previous_wait = operation->previous_wait;
    }
    operation->previous_wait = NULL;
    operation->next_wait = NULL;
  }
}

static void iree_hal_amd_gpu_pm4_program_notify_reuse(
    iree_hal_amd_gpu_pm4_program_t* program) {
  ++program->availability_epoch;
  // Only enrolled requests are visited, once per change that can satisfy them.
  // Ordinary publication and retirement have no list work or notification.
  while (program->waiters) {
    iree_hal_amd_gpu_pm4_program_growth_t* operation = program->waiters;
    iree_hal_amd_gpu_pm4_program_unlink_wait(operation);
    operation->notify_reuse(operation->reuse_user_data);
  }
}

void iree_hal_amd_gpu_pm4_program_grow_await_reuse(
    iree_hal_amd_gpu_pm4_program_growth_t* operation,
    void (*notify)(void* user_data), void* user_data) {
  iree_hal_amd_gpu_pm4_program_t* program = operation->program;
  iree_slim_mutex_lock(&program->mutex);
  if (!operation->previous_wait) {
    operation->notify_reuse = notify;
    operation->reuse_user_data = user_data;
    if (program->queues[operation->queue_ordinal].observed_epoch !=
        program->availability_epoch) {
      notify(user_data);
    } else {
      operation->previous_wait = &program->waiters;
      operation->next_wait = program->waiters;
      if (operation->next_wait) {
        operation->next_wait->previous_wait = &operation->next_wait;
      }
      program->waiters = operation;
    }
  }
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_grow_cancel_reuse(
    iree_hal_amd_gpu_pm4_program_growth_t* operation) {
  iree_slim_mutex_lock(&operation->program->mutex);
  iree_hal_amd_gpu_pm4_program_unlink_wait(operation);
  iree_slim_mutex_unlock(&operation->program->mutex);
}

iree_status_t iree_hal_amd_gpu_pm4_program_grow_end(
    iree_hal_amd_gpu_pm4_program_growth_t* operation, iree_status_t status) {
  iree_hal_amd_gpu_pm4_program_t* program = operation->program;
  iree_hal_amd_gpu_pm4_program_instance_t* instance = operation->instance;
  program->queues[operation->queue_ordinal].growth = NULL;
  if (instance && iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        program->pool->context->api->memory_query_address(
            instance->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU,
            &instance->address),
        "query resident program address");
  }
  iree_slim_mutex_lock(&program->mutex);
  iree_hal_amd_gpu_pm4_program_unlink_wait(operation);
  if (instance && iree_status_is_ok(status)) {
    instance->reserved = false;
    iree_hal_amd_gpu_pm4_program_notify_reuse(program);
  } else if (instance) {
    program->instances[instance->ordinal] = NULL;
  }
  iree_slim_mutex_unlock(&program->mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_program_instance_free(program, instance);
  }
  return status;
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
  // Only this publisher can already dominate its newly accepted coordinate.
  // Other queues cannot observe that point until signal publication follows.
  // Waking their clone demand here would retry work that is still ineligible.
  iree_hal_amd_gpu_pm4_program_growth_t* waiting =
      program->queues[queue_ordinal].growth;
  if (waiting && waiting->previous_wait) {
    iree_hal_amd_gpu_pm4_program_unlink_wait(waiting);
    waiting->notify_reuse(waiting->reuse_user_data);
  }
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_cancel(
    iree_hal_amd_gpu_pm4_program_t* program,
    iree_hal_amd_gpu_pm4_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  instance->reserved = false;
  --instance->use_count;
  iree_hal_amd_gpu_pm4_program_notify_reuse(program);
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
  if (!instance->use_count) {
    iree_hal_amd_gpu_pm4_program_notify_reuse(program);
  }
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_pm4_program_trim(
    iree_hal_amd_gpu_pm4_program_t* program) {
  iree_slim_mutex_lock(&program->mutex);
  const uint32_t capacity = program->capacity;
  iree_slim_mutex_unlock(&program->mutex);
  for (uint32_t i = 1; i < capacity; ++i) {
    iree_slim_mutex_lock(&program->mutex);
    iree_hal_amd_gpu_pm4_program_instance_t* instance = program->instances[i];
    if (instance && !instance->use_count && !instance->reserved) {
      program->instances[i] = NULL;
      for (uint32_t q = 0; q < program->pool->queue_count; ++q) {
        if (program->queues[q].preferred == instance) {
          program->queues[q].preferred = program->initial;
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
    idle &= !program->instances[i] || (!program->instances[i]->use_count &&
                                       !program->instances[i]->reserved);
  }
  iree_hal_amd_gpu_pm4_program_upload_t* upload = NULL;
  if (idle &&
      iree_atomic_load(&program->source_ready, iree_memory_order_acquire)) {
    upload = program->upload;
    program->upload = NULL;
  }
  iree_slim_mutex_unlock(&program->mutex);
  iree_hal_amd_gpu_pm4_program_upload_free(program->pool, upload);
}

void iree_hal_amd_gpu_pm4_program_deinitialize(
    iree_hal_amd_gpu_pm4_program_t* program) {
  if (!program->pool) {
    return;
  }
  iree_slim_mutex_lock(&program->pool->registry_mutex);
  if (program->previous) {
    program->previous->next = program->next;
  }
  if (program->next) {
    program->next->previous = program->previous;
  }
  if (program->pool->head == program) {
    program->pool->head = program->next;
  }
  iree_slim_mutex_unlock(&program->pool->registry_mutex);
  iree_hal_amd_gpu_pm4_program_upload_free(program->pool, program->upload);
  program->upload = NULL;
  if (program->instances) {
    for (uint32_t i = 0; i < program->capacity; ++i) {
      iree_hal_amd_gpu_pm4_program_instance_free(program,
                                                 program->instances[i]);
    }
    if (program->instances !=
        (void*)(program->queues + program->pool->queue_count)) {
      iree_allocator_free(program->pool->host_allocator, program->instances);
    }
    iree_allocator_free(program->pool->host_allocator, program->queues);
  }
  iree_slim_mutex_deinitialize(&program->mutex);
}
