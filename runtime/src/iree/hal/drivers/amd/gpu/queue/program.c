// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/program.h"

#include "iree/base/bitmap.h"
#include "iree/hal/drivers/amd/status.h"

static_assert(sizeof(iree_hal_amd_gpu_program_queue_t) <= 32,
              "two publisher candidates must fit in one cache line");

// Native allocation granularity determines page capacity. The bitmap describes
// small immutable byte ranges, not GPU pages; all bookkeeping stays on the CPU.
enum { IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT = 16 };

struct iree_hal_amd_gpu_program_constant_page_t {
  // Borrowed pool dominated by every range owner and cold acquisition caller.
  iree_hal_amd_gpu_program_pool_t* pool;
  // Previous available-list link, or NULL when the page is full/detached.
  iree_hal_amd_gpu_program_constant_page_t** previous;
  // Next page with unallocated bytes, protected by pool->constants.mutex.
  iree_hal_amd_gpu_program_constant_page_t* next;
  // Native address of byte zero, queried once during page construction.
  uint64_t address;
  // Complete logical page extent, in bytes.
  uint64_t byte_length;
  // Reserved bitmap units in bytes, protected by pool->constants.mutex.
  uint64_t used_bytes;
  // Host-only range bitmap stored after this descriptor.
  iree_bitmap_t bitmap;
  // Native backing and its persistent write-only host view.
  iree_hal_amd_fabric_memory_t memory;
  // Native release metadata secured before any range is exposed.
  iree_hal_amd_gpu_memory_release_t release;
  // Owns this descriptor after the pool has detached its final native release.
  iree_allocator_t host_allocator;
};

static void iree_hal_amd_gpu_program_constant_page_link(
    iree_hal_amd_gpu_program_constant_page_t* page) {
  iree_hal_amd_gpu_program_pool_t* pool = page->pool;
  page->previous = &pool->constants.pages;
  page->next = pool->constants.pages;
  if (page->next) {
    page->next->previous = &page->next;
  }
  pool->constants.pages = page;
}

static void iree_hal_amd_gpu_program_constant_page_unlink(
    iree_hal_amd_gpu_program_constant_page_t* page) {
  *page->previous = page->next;
  if (page->next) {
    page->next->previous = page->previous;
  }
  page->previous = NULL;
  page->next = NULL;
}

static void iree_hal_amd_gpu_program_constant_page_released(void* user_data) {
  iree_hal_amd_gpu_program_constant_page_t* page = user_data;
  iree_allocator_free(page->host_allocator, page);
}

static void iree_hal_amd_gpu_program_constant_page_free(
    iree_hal_amd_gpu_program_constant_page_t* page) {
  iree_hal_amd_gpu_context_release_memory(
      page->pool->context, 1, &page->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_program_constant_page_released,
          .user_data = page},
      &page->release);
}

static void iree_hal_amd_gpu_program_constants_trim(
    iree_hal_amd_gpu_program_pool_t* pool) {
  iree_slim_mutex_lock(&pool->constants.mutex);
  iree_hal_amd_gpu_program_constant_page_t* page = pool->constants.spare;
  if (page) {
    pool->constants.spare = NULL;
    iree_hal_amd_gpu_program_constant_page_unlink(page);
  }
  iree_slim_mutex_unlock(&pool->constants.mutex);
  if (page) {
    iree_hal_amd_gpu_program_constant_page_free(page);
  }
}

// Cold selection can search fragmented pages. Allocation and return never
// inspect device bytes; a retained range makes its page stable after unlock.
static bool iree_hal_amd_gpu_program_constant_try_allocate(
    iree_hal_amd_gpu_program_pool_t* pool, uint64_t byte_length,
    uint32_t alignment, iree_hal_amd_gpu_program_constant_t* out_constant) {
  const uint64_t unit = IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT;
  for (iree_hal_amd_gpu_program_constant_page_t* page = pool->constants.pages;
       page; page = page->next) {
    if (page->byte_length - page->used_bytes < byte_length) {
      continue;
    }
    uint64_t offset =
        iree_align_uint64(page->address, alignment) - page->address;
    while (offset <= page->byte_length - byte_length) {
      const iree_host_size_t first = iree_bitmap_find_first_unset_span(
          page->bitmap, offset / unit, byte_length / unit);
      if (first == page->bitmap.bit_count) {
        break;
      }
      const uint64_t candidate = first * unit;
      offset = iree_align_uint64(page->address + candidate, alignment) -
               page->address;
      if (offset != candidate) {
        continue;
      }
      iree_bitmap_set_span(page->bitmap, first, byte_length / unit);
      page->used_bytes += byte_length;
      if (page == pool->constants.spare) {
        pool->constants.spare = NULL;
      }
      if (page->used_bytes == page->byte_length) {
        iree_hal_amd_gpu_program_constant_page_unlink(page);
      }
      ++pool->constants.count;
      *out_constant = (iree_hal_amd_gpu_program_constant_t){
          .page = page, .offset = offset, .byte_length = byte_length};
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_amd_gpu_program_constant_page_allocate(
    iree_hal_amd_gpu_program_pool_t* pool, uint64_t minimum_length,
    uint32_t alignment, iree_hal_amd_gpu_program_constant_page_t** out_page) {
  *out_page = NULL;
  const uint64_t unit = IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT;
  const uint64_t granularity = iree_max(
      pool->resident.profile.allocation.native_byte_length_granularity, unit);
  const uint64_t byte_length =
      iree_host_size_ceil_div(minimum_length, granularity) * granularity;
  const uint64_t bit_count = byte_length / unit;
  iree_host_size_t total_size = 0, bitmap_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_program_constant_page_t), &total_size,
      IREE_STRUCT_FIELD(iree_bitmap_calculate_words(bit_count), uint64_t,
                        &bitmap_offset)));
  iree_hal_amd_gpu_program_constant_page_t* page = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(pool->host_allocator, total_size, (void**)&page));
  *page = (iree_hal_amd_gpu_program_constant_page_t){
      .pool = pool,
      .byte_length = byte_length,
      .bitmap = {.bit_count = bit_count,
                 .words = (void*)((uint8_t*)page + bitmap_offset)},
      .host_allocator = pool->host_allocator,
  };
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      pool->context->api, &pool->resident, byte_length, alignment,
      pool->context->event_sink, &page->memory, NULL);
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->memory_query_address(
            page->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU, &page->address),
        "query immutable program data address");
  }
  if (iree_status_is_ok(status)) {
    *out_page = page;
  } else {
    iree_hal_amd_gpu_program_constant_page_free(page);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_program_constant_allocate(
    iree_hal_amd_gpu_program_pool_t* pool, uint32_t byte_length,
    uint32_t alignment, iree_hal_amd_gpu_program_constant_t* out_constant,
    iree_byte_span_t* out_contents, uint64_t* out_address) {
  *out_constant = (iree_hal_amd_gpu_program_constant_t){0};
  *out_contents = iree_byte_span_empty();
  *out_address = 0;
  const uint64_t length =
      iree_align_uint64(byte_length, IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT);
  alignment = iree_max(alignment, IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT);
  iree_slim_mutex_lock(&pool->constants.mutex);
  bool acquired = iree_hal_amd_gpu_program_constant_try_allocate(
      pool, length, alignment, out_constant);
  iree_slim_mutex_unlock(&pool->constants.mutex);
  iree_status_t status = iree_ok_status();
  if (!acquired) {
    // Serialize only callers that need growth. Returns and acquisitions from
    // existing pages continue while a native allocation blocks in the driver.
    iree_slim_mutex_lock(&pool->constants.growth_mutex);
    iree_slim_mutex_lock(&pool->constants.mutex);
    acquired = iree_hal_amd_gpu_program_constant_try_allocate(
        pool, length, alignment, out_constant);
    iree_slim_mutex_unlock(&pool->constants.mutex);
    if (!acquired) {
      iree_hal_amd_gpu_program_constant_page_t* page = NULL;
      status = iree_hal_amd_gpu_program_constant_page_allocate(
          pool, length, alignment, &page);
      if (iree_status_is_ok(status)) {
        iree_bitmap_set_span(page->bitmap, 0,
                             length / IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT);
        page->used_bytes = length;
        iree_slim_mutex_lock(&pool->constants.mutex);
        if (length < page->byte_length) {
          iree_hal_amd_gpu_program_constant_page_link(page);
        }
        ++pool->constants.count;
        iree_slim_mutex_unlock(&pool->constants.mutex);
        *out_constant = (iree_hal_amd_gpu_program_constant_t){
            .page = page, .byte_length = length};
      }
    }
    iree_slim_mutex_unlock(&pool->constants.growth_mutex);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_program_constant_t constant = *out_constant;
    *out_contents = iree_make_byte_span(
        constant.page->memory.contents.data + constant.offset, byte_length);
    *out_address = constant.page->address + constant.offset;
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_program_constant_publish(
    const iree_hal_amd_gpu_program_constant_t* constant) {
  const iree_hal_amd_gpu_program_constant_page_t* page = constant->page;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      page->pool->context->api->host_mapping_cache_control(
          page->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
          constant->offset, constant->byte_length),
      "publish immutable program data");
}

void iree_hal_amd_gpu_program_constant_release(
    iree_hal_amd_gpu_program_constant_t* constant) {
  iree_hal_amd_gpu_program_constant_page_t* page = constant->page;
  if (!page) {
    return;
  }
  iree_hal_amd_gpu_program_pool_t* pool = page->pool;
  iree_slim_mutex_lock(&pool->constants.mutex);
  if (page->used_bytes == page->byte_length) {
    iree_hal_amd_gpu_program_constant_page_link(page);
  }
  iree_bitmap_reset_span(
      page->bitmap, constant->offset / IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT,
      constant->byte_length / IREE_HAL_AMD_GPU_PROGRAM_CONSTANT_UNIT);
  page->used_bytes -= constant->byte_length;
  --pool->constants.count;
  const bool release = !page->used_bytes && pool->constants.spare;
  if (release) {
    iree_hal_amd_gpu_program_constant_page_unlink(page);
  } else if (!page->used_bytes) {
    pool->constants.spare = page;
  }
  iree_slim_mutex_unlock(&pool->constants.mutex);
  *constant = (iree_hal_amd_gpu_program_constant_t){0};
  if (release) {
    iree_hal_amd_gpu_program_constant_page_free(page);
  }
}

iree_status_t iree_hal_amd_gpu_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_program_pool_t){
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
  };
  iree_slim_mutex_initialize(&out_pool->registry_mutex);
  iree_slim_mutex_initialize(&out_pool->constants.growth_mutex);
  iree_slim_mutex_initialize(&out_pool->constants.mutex);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_query_program_memory(
      context, native_family_ordinal, &out_pool->access, &out_pool->upload,
      &out_pool->resident));
  return iree_ok_status();
}

// The registry owner serializes cold queue growth. Instance-table growth can
// still race the host allocation, so prepare a replacement outside the hot
// metadata mutex and install only while it covers the latest instance extent.
static iree_status_t iree_hal_amd_gpu_program_reserve_queues(
    iree_hal_amd_gpu_program_t* program, uint32_t queue_count) {
  const iree_allocator_t allocator = program->pool->host_allocator;
  iree_status_t status = iree_ok_status();
  bool complete = false;
  while (!complete && iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&program->mutex);
    complete = queue_count <= program->queue_count;
    const uint32_t capacity = program->capacity;
    iree_slim_mutex_unlock(&program->mutex);
    if (complete) {
      break;
    }
    iree_host_size_t total_size = 0, instances_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &total_size,
        IREE_STRUCT_FIELD(queue_count, iree_hal_amd_gpu_program_queue_t, NULL),
        IREE_STRUCT_FIELD(capacity, iree_hal_amd_gpu_program_instance_t*,
                          &instances_offset));
    iree_hal_amd_gpu_program_queue_t* proposal = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(allocator, total_size, (void**)&proposal);
    }
    iree_hal_amd_gpu_program_queue_t* replaced_queues = NULL;
    iree_hal_amd_gpu_program_instance_t** replaced_instances = NULL;
    if (iree_status_is_ok(status)) {
      iree_hal_amd_gpu_program_instance_t** instances =
          (void*)((uint8_t*)proposal + instances_offset);
      iree_slim_mutex_lock(&program->mutex);
      if (capacity == program->capacity) {
        memcpy(proposal, program->queues,
               program->queue_count * sizeof(*proposal));
        for (uint32_t i = program->queue_count; i < queue_count; ++i) {
          proposal[i].preferred = program->initial;
        }
        memcpy(instances, program->instances, capacity * sizeof(*instances));
        replaced_queues = program->queues;
        if (program->instances !=
            (void*)(program->queues + program->queue_count)) {
          replaced_instances = program->instances;
        }
        program->queue_count = queue_count;
        program->queues = proposal;
        program->instances = instances;
        proposal = NULL;
        complete = true;
      }
      iree_slim_mutex_unlock(&program->mutex);
    }
    iree_allocator_free(allocator, proposal);
    iree_allocator_free(allocator, replaced_instances);
    iree_allocator_free(allocator, replaced_queues);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_program_pool_reserve_queues(
    iree_hal_amd_gpu_program_pool_t* pool, uint32_t queue_count) {
  iree_slim_mutex_lock(&pool->registry_mutex);
  pool->queue_count = iree_max(pool->queue_count, queue_count);
  iree_status_t status = iree_ok_status();
  for (iree_hal_amd_gpu_program_t* program = pool->head;
       program && iree_status_is_ok(status); program = program->next) {
    status =
        iree_hal_amd_gpu_program_reserve_queues(program, pool->queue_count);
  }
  iree_slim_mutex_unlock(&pool->registry_mutex);
  return status;
}

void iree_hal_amd_gpu_program_pool_trim(iree_hal_amd_gpu_program_pool_t* pool) {
  iree_slim_mutex_lock(&pool->registry_mutex);
  for (iree_hal_amd_gpu_program_t* program = pool->head; program;
       program = program->next) {
    iree_hal_amd_gpu_program_trim(program);
  }
  iree_slim_mutex_unlock(&pool->registry_mutex);
  iree_hal_amd_gpu_program_constants_trim(pool);
}

void iree_hal_amd_gpu_program_pool_deinitialize(
    iree_hal_amd_gpu_program_pool_t* pool) {
  if (!pool->context) {
    return;
  }
  IREE_ASSERT(!pool->head);
  IREE_ASSERT(!pool->constants.count);
  iree_hal_amd_gpu_program_constants_trim(pool);
  iree_slim_mutex_deinitialize(&pool->constants.growth_mutex);
  iree_slim_mutex_deinitialize(&pool->constants.mutex);
  iree_slim_mutex_deinitialize(&pool->registry_mutex);
  pool->context = NULL;
}

static void iree_hal_amd_gpu_program_instance_released(void* user_data) {
  iree_hal_amd_gpu_program_instance_t* instance = user_data;
  iree_allocator_free(instance->host_allocator, instance);
}

static void iree_hal_amd_gpu_program_instance_free(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance) {
  if (!instance) {
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      program->pool->context, 1, &instance->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_program_instance_released,
          .user_data = instance},
      &instance->release);
}

struct iree_hal_amd_gpu_program_upload_t {
  // Native staging view retained through all readers of initial program bytes.
  iree_hal_amd_fabric_memory_t memory;
  // Release storage remains owned independently of command-buffer metadata.
  iree_hal_amd_gpu_memory_release_t release;
  // Device allocator owning this upload through native release completion.
  iree_allocator_t host_allocator;
};

static void iree_hal_amd_gpu_program_upload_released(void* user_data) {
  iree_hal_amd_gpu_program_upload_t* upload = user_data;
  iree_allocator_free(upload->host_allocator, upload);
}

static void iree_hal_amd_gpu_program_upload_free(
    iree_hal_amd_gpu_program_pool_t* pool,
    iree_hal_amd_gpu_program_upload_t* upload) {
  if (!upload) {
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      pool->context, 1, &upload->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_program_upload_released, .user_data = upload},
      &upload->release);
}

static iree_status_t iree_hal_amd_gpu_program_instance_allocate(
    iree_hal_amd_gpu_program_t* program, uint32_t ordinal, uint64_t byte_length,
    bool mapped, iree_hal_amd_gpu_program_instance_t** out_instance) {
  *out_instance = NULL;
  iree_hal_amd_gpu_program_pool_t* pool = program->pool;
  iree_hal_amd_gpu_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator, sizeof(*instance), (void**)&instance));
  instance->ordinal = ordinal;
  instance->host_allocator = pool->host_allocator;
  iree_hal_amd_fabric_memory_options_t options = pool->resident;
  if (!mapped) {
    options.host_access = 0;
    options.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  }
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      pool->context->api, &options, byte_length, program->alignment,
      pool->context->event_sink, &instance->memory, NULL);
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
    iree_hal_amd_gpu_program_instance_free(program, instance);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_program_initialize(
    iree_hal_amd_gpu_program_pool_t* pool, uint32_t alignment,
    uint32_t byte_length, uint32_t fixup_count,
    iree_hal_amd_gpu_program_t* out_program) {
  iree_slim_mutex_lock(&pool->registry_mutex);
  const uint32_t queue_count = pool->queue_count;
  iree_slim_mutex_unlock(&pool->registry_mutex);
  *out_program = (iree_hal_amd_gpu_program_t){
      .pool = pool,
      .capacity = queue_count,
      .queue_count = queue_count,
      .byte_length = byte_length,
      .alignment = alignment,
      .fixup_count = fixup_count,
      .fixup_offset = (uint32_t)iree_align_uint64(byte_length, 16),
  };
  iree_slim_mutex_initialize(&out_program->mutex);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator,
      queue_count *
          (sizeof(*out_program->queues) + sizeof(*out_program->instances)),
      (void**)&out_program->queues));
  out_program->instances = (void*)(out_program->queues + queue_count);
  const uint64_t allocation_length =
      out_program->fixup_offset +
      (uint64_t)fixup_count * sizeof(iree_hal_amd_gpu_fixup_t);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_instance_allocate(
      out_program, 0, allocation_length, true, &out_program->instances[0]));
  iree_hal_amd_gpu_program_instance_t* initial = out_program->instances[0];
  out_program->initial = initial;
  if (initial->memory.contents.data) {
    out_program->contents = initial->memory.contents;
    out_program->upload_address = initial->address;
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                               sizeof(*out_program->upload),
                                               (void**)&out_program->upload));
    out_program->upload->host_allocator = pool->host_allocator;
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
        pool->context->api, &pool->upload, allocation_length, alignment,
        pool->context->event_sink, &out_program->upload->memory, NULL));
    out_program->contents = out_program->upload->memory.contents;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->memory_query_address(
            out_program->upload->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU,
            &out_program->upload_address),
        "query resident program upload address"));
  }
  for (uint32_t i = 0; i < queue_count; ++i) {
    out_program->queues[i].preferred = initial;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_program_seal(
    iree_hal_amd_gpu_program_t* program) {
  iree_hal_amd_gpu_program_instance_t* initial = program->initial;
  amdf_host_mapping_t* mapping = program->upload
                                     ? program->upload->memory.mapping
                                     : initial->memory.mapping;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      program->pool->context->api->host_mapping_cache_control(
          mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
          program->fixup_offset + (uint64_t)program->fixup_count *
                                      sizeof(iree_hal_amd_gpu_fixup_t)),
      "publish sealed resident program"));
  if (!program->upload) {
    initial->initialized = true;
    iree_atomic_store(&program->source_ready, 1, iree_memory_order_release);
  }
  iree_slim_mutex_lock(&program->pool->registry_mutex);
  iree_status_t status = iree_hal_amd_gpu_program_reserve_queues(
      program, program->pool->queue_count);
  if (iree_status_is_ok(status)) {
    program->next = program->pool->head;
    if (program->next) {
      program->next->previous = program;
    }
    program->pool->head = program;
  }
  iree_slim_mutex_unlock(&program->pool->registry_mutex);
  return status;
}

iree_hal_amd_gpu_program_source_t iree_hal_amd_gpu_program_source(
    const iree_hal_amd_gpu_program_t* program) {
  const uint64_t address =
      iree_atomic_load(&program->source_ready, iree_memory_order_acquire)
          ? program->initial->address
          : program->upload_address;
  return (iree_hal_amd_gpu_program_source_t){
      .address = address,
      .fixups = address + program->fixup_offset,
  };
}

static bool iree_hal_amd_gpu_program_instance_available(
    const iree_hal_amd_gpu_program_instance_t* instance,
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

iree_hal_amd_gpu_program_instance_t* iree_hal_amd_gpu_program_try_acquire(
    iree_hal_amd_gpu_program_t* program, uint32_t queue_ordinal,
    const uint64_t* frontier) {
  iree_slim_mutex_lock(&program->mutex);
  iree_hal_amd_gpu_program_instance_t* instance =
      program->queues[queue_ordinal].preferred;
  if (!instance ||
      !iree_hal_amd_gpu_program_instance_available(instance, frontier)) {
    instance = NULL;
    for (uint32_t i = 0; i < program->capacity && !instance; ++i) {
      iree_hal_amd_gpu_program_instance_t* candidate = program->instances[i];
      if (candidate &&
          iree_hal_amd_gpu_program_instance_available(candidate, frontier)) {
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
// on physical overlap. Per-queue metadata never moves for instance-only growth.
static iree_status_t iree_hal_amd_gpu_program_reserve_instance(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance, bool* out_reserved) {
  *out_reserved = false;
  const iree_allocator_t allocator = program->pool->host_allocator;
  iree_hal_amd_gpu_program_instance_t** proposal = NULL;
  uint32_t proposal_capacity = 0;
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete && iree_status_is_ok(status)) {
    iree_hal_amd_gpu_program_instance_t** replaced = NULL;
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
        if (program->instances !=
            (void*)(program->queues + program->queue_count)) {
          replaced = program->instances;
        }
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
    iree_allocator_free(allocator, replaced);
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

iree_hal_amd_gpu_program_growth_t* iree_hal_amd_gpu_program_pending_growth(
    iree_hal_amd_gpu_program_t* program, uint32_t queue_ordinal) {
  iree_slim_mutex_lock(&program->mutex);
  iree_hal_amd_gpu_program_growth_t* pending =
      program->queues[queue_ordinal].growth;
  iree_slim_mutex_unlock(&program->mutex);
  return pending;
}

iree_status_t iree_hal_amd_gpu_program_grow_begin(
    iree_hal_amd_gpu_program_t* program, uint32_t queue_ordinal,
    iree_hal_amd_fabric_memory_callback_t callback,
    iree_hal_amd_gpu_program_growth_t* operation) {
  iree_hal_amd_gpu_program_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      program->pool->host_allocator, sizeof(*instance), (void**)&instance));
  instance->host_allocator = program->pool->host_allocator;
  instance->reserved = true;
  bool reserved = false;
  iree_status_t status =
      iree_hal_amd_gpu_program_reserve_instance(program, instance, &reserved);
  if (!reserved) {
    iree_allocator_free(instance->host_allocator, instance);
    instance = NULL;
  }
  IREE_RETURN_IF_ERROR(status);
  *operation = (iree_hal_amd_gpu_program_growth_t){
      .program = program,
      .instance = instance,
      .queue_ordinal = queue_ordinal,
      .options = program->pool->resident,
  };
  operation->options.host_access = 0;
  operation->options.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  iree_slim_mutex_lock(&program->mutex);
  program->queues[queue_ordinal].growth = operation;
  iree_slim_mutex_unlock(&program->mutex);
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
static void iree_hal_amd_gpu_program_unlink_wait(
    iree_hal_amd_gpu_program_growth_t* operation) {
  if (operation->previous_wait) {
    *operation->previous_wait = operation->next_wait;
    if (operation->next_wait) {
      operation->next_wait->previous_wait = operation->previous_wait;
    }
    operation->previous_wait = NULL;
    operation->next_wait = NULL;
  }
}

static void iree_hal_amd_gpu_program_notify_reuse(
    iree_hal_amd_gpu_program_t* program) {
  ++program->availability_epoch;
  // Only enrolled requests are visited, once per change that can satisfy them.
  // Ordinary publication and retirement have no list work or notification.
  while (program->waiters) {
    iree_hal_amd_gpu_program_growth_t* operation = program->waiters;
    iree_hal_amd_gpu_program_unlink_wait(operation);
    operation->notify_reuse(operation->reuse_user_data);
  }
}

void iree_hal_amd_gpu_program_grow_await_reuse(
    iree_hal_amd_gpu_program_growth_t* operation,
    void (*notify)(void* user_data), void* user_data) {
  iree_hal_amd_gpu_program_t* program = operation->program;
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

void iree_hal_amd_gpu_program_grow_cancel_reuse(
    iree_hal_amd_gpu_program_growth_t* operation) {
  iree_slim_mutex_lock(&operation->program->mutex);
  iree_hal_amd_gpu_program_unlink_wait(operation);
  iree_slim_mutex_unlock(&operation->program->mutex);
}

iree_status_t iree_hal_amd_gpu_program_grow_end(
    iree_hal_amd_gpu_program_growth_t* operation, iree_status_t status) {
  iree_hal_amd_gpu_program_t* program = operation->program;
  iree_hal_amd_gpu_program_instance_t* instance = operation->instance;
  if (instance && iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        program->pool->context->api->memory_query_address(
            instance->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU,
            &instance->address),
        "query resident program address");
  }
  iree_slim_mutex_lock(&program->mutex);
  program->queues[operation->queue_ordinal].growth = NULL;
  iree_hal_amd_gpu_program_unlink_wait(operation);
  if (instance && iree_status_is_ok(status)) {
    instance->reserved = false;
    iree_hal_amd_gpu_program_notify_reuse(program);
  } else if (instance) {
    program->instances[instance->ordinal] = NULL;
  }
  iree_slim_mutex_unlock(&program->mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_program_instance_free(program, instance);
  }
  return status;
}

void iree_hal_amd_gpu_program_publish(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance, uint32_t queue_ordinal,
    uint64_t epoch) {
  iree_slim_mutex_lock(&program->mutex);
  instance->initialized = true;
  instance->last_queue_ordinal = queue_ordinal;
  instance->last_epoch = epoch;
  instance->reserved = false;
  // Only this publisher can already dominate its newly accepted coordinate.
  // Other queues cannot observe that point until signal publication follows.
  // Waking their clone demand here would retry work that is still ineligible.
  iree_hal_amd_gpu_program_growth_t* waiting =
      program->queues[queue_ordinal].growth;
  if (waiting && waiting->previous_wait) {
    iree_hal_amd_gpu_program_unlink_wait(waiting);
    waiting->notify_reuse(waiting->reuse_user_data);
  }
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_program_cancel(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  instance->reserved = false;
  --instance->use_count;
  iree_hal_amd_gpu_program_notify_reuse(program);
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_program_release(
    iree_hal_amd_gpu_program_t* program,
    iree_hal_amd_gpu_program_instance_t* instance) {
  iree_slim_mutex_lock(&program->mutex);
  --instance->use_count;
  if (!instance->ordinal) {
    iree_atomic_store(&program->source_ready, 1, iree_memory_order_release);
  }
  if (!instance->use_count) {
    iree_hal_amd_gpu_program_notify_reuse(program);
  }
  iree_slim_mutex_unlock(&program->mutex);
}

void iree_hal_amd_gpu_program_trim(iree_hal_amd_gpu_program_t* program) {
  iree_slim_mutex_lock(&program->mutex);
  const uint32_t capacity = program->capacity;
  iree_slim_mutex_unlock(&program->mutex);
  for (uint32_t i = 1; i < capacity; ++i) {
    iree_slim_mutex_lock(&program->mutex);
    iree_hal_amd_gpu_program_instance_t* instance = program->instances[i];
    if (instance && !instance->use_count && !instance->reserved) {
      program->instances[i] = NULL;
      for (uint32_t q = 0; q < program->queue_count; ++q) {
        if (program->queues[q].preferred == instance) {
          program->queues[q].preferred = program->initial;
        }
      }
    } else {
      instance = NULL;
    }
    iree_slim_mutex_unlock(&program->mutex);
    iree_hal_amd_gpu_program_instance_free(program, instance);
  }
  bool idle = true;
  iree_slim_mutex_lock(&program->mutex);
  for (uint32_t i = 0; i < program->capacity; ++i) {
    idle &= !program->instances[i] || (!program->instances[i]->use_count &&
                                       !program->instances[i]->reserved);
  }
  iree_hal_amd_gpu_program_upload_t* upload = NULL;
  if (idle &&
      iree_atomic_load(&program->source_ready, iree_memory_order_acquire)) {
    upload = program->upload;
    program->upload = NULL;
  }
  iree_slim_mutex_unlock(&program->mutex);
  iree_hal_amd_gpu_program_upload_free(program->pool, upload);
}

void iree_hal_amd_gpu_program_deinitialize(
    iree_hal_amd_gpu_program_t* program) {
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
  iree_hal_amd_gpu_program_upload_free(program->pool, program->upload);
  program->upload = NULL;
  if (program->instances) {
    for (uint32_t i = 0; i < program->capacity; ++i) {
      iree_hal_amd_gpu_program_instance_free(program, program->instances[i]);
    }
    if (program->instances != (void*)(program->queues + program->queue_count)) {
      iree_allocator_free(program->pool->host_allocator, program->instances);
    }
    iree_allocator_free(program->pool->host_allocator, program->queues);
  }
  iree_slim_mutex_deinitialize(&program->mutex);
}
