// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/program.h"

#include "iree/base/bitmap.h"
#include "iree/hal/driver/amd/status.h"

static_assert(sizeof(iree_hal_amd_gpu_program_queue_t) <= 32,
              "two publisher candidates must fit in one cache line");

// Native allocation granularity determines page capacity. The bitmap describes
// independently owned byte ranges, not GPU pages; all bookkeeping stays on the
// CPU.
enum { IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT = 16 };

struct iree_hal_amd_gpu_program_range_page_t {
  // Borrowed pool dominated by every range owner and cold acquisition caller.
  iree_hal_amd_gpu_program_range_pool_t* pool;
  // Previous available-list link, or NULL when the page is full/detached.
  iree_hal_amd_gpu_program_range_page_t** previous;
  // Next page with unallocated bytes, protected by pool->mutex.
  iree_hal_amd_gpu_program_range_page_t* next;
  // Native address of byte zero, queried once during page construction.
  uint64_t address;
  // Complete logical page extent, in bytes.
  uint64_t byte_length;
  // Reserved bitmap units in bytes, protected by pool->mutex.
  uint64_t used_bytes;
  // Independent host publication granularity, or one for unmapped storage.
  uint64_t publication_granularity;
  // Host-only range bitmap stored after this descriptor.
  iree_bitmap_t bitmap;
  // Native backing and its persistent write-only host view.
  iree_hal_amd_fabric_memory_t memory;
  // Native release metadata secured before any range is exposed.
  iree_hal_amd_gpu_memory_release_t release;
  // Owns this descriptor after the pool has detached its final native release.
  iree_allocator_t host_allocator;
};

static void iree_hal_amd_gpu_program_range_page_link(
    iree_hal_amd_gpu_program_range_page_t* page) {
  iree_hal_amd_gpu_program_range_pool_t* pool = page->pool;
  page->previous = &pool->pages;
  page->next = pool->pages;
  if (page->next) {
    page->next->previous = &page->next;
  }
  pool->pages = page;
}

static void iree_hal_amd_gpu_program_range_page_unlink(
    iree_hal_amd_gpu_program_range_page_t* page) {
  *page->previous = page->next;
  if (page->next) {
    page->next->previous = page->previous;
  }
  page->previous = NULL;
  page->next = NULL;
}

static void iree_hal_amd_gpu_program_range_page_released(void* user_data) {
  iree_hal_amd_gpu_program_range_page_t* page = user_data;
  iree_allocator_free(page->host_allocator, page);
}

static void iree_hal_amd_gpu_program_range_page_free(
    iree_hal_amd_gpu_program_range_page_t* page) {
  iree_hal_amd_gpu_context_release_memory(
      page->pool->context, 1, &page->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_program_range_page_released,
          .user_data = page},
      &page->release);
}

static void iree_hal_amd_gpu_program_ranges_trim(
    iree_hal_amd_gpu_program_range_pool_t* pool) {
  iree_hal_amd_gpu_program_range_page_t* retired = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_amd_gpu_program_range_page_t* page = pool->pages;
  while (page) {
    iree_hal_amd_gpu_program_range_page_t* next = page->next;
    if (!page->used_bytes) {
      iree_hal_amd_gpu_program_range_page_unlink(page);
      page->next = retired;
      retired = page;
    }
    page = next;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  while (retired) {
    page = retired;
    retired = page->next;
    iree_hal_amd_gpu_program_range_page_free(page);
  }
}

// Selection can search fragmented pages. Allocation and return never
// inspect device bytes; a retained range makes its page stable after unlock.
static bool iree_hal_amd_gpu_program_range_try_allocate(
    iree_hal_amd_gpu_program_range_pool_t* pool, uint64_t byte_length,
    uint32_t alignment, iree_hal_amd_gpu_program_range_t* out_range,
    uint64_t* out_payload_offset) {
  const uint64_t unit = IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT;
  for (iree_hal_amd_gpu_program_range_page_t* page = pool->pages; page;
       page = page->next) {
    if (page->byte_length - page->used_bytes < byte_length) {
      continue;
    }
    const uint64_t host_address = (uintptr_t)page->memory.contents.data;
    iree_host_size_t position = 0;
    while (position < page->bitmap.bit_count) {
      const iree_host_size_t first = iree_bitmap_find_first_unset_span(
          page->bitmap, position, byte_length / unit);
      if (first == page->bitmap.bit_count) {
        break;
      }
      const uint64_t candidate = first * unit;
      // A native allocation owns its outer boundary lines. Interior ranges
      // begin on a host publication boundary before satisfying GPU alignment.
      const uint64_t minimum_offset =
          candidate ? iree_align_uint64(host_address + candidate,
                                        page->publication_granularity) -
                          host_address
                    : 0;
      const uint64_t offset =
          iree_align_uint64(page->address + minimum_offset, alignment) -
          page->address;
      if (offset + byte_length > page->byte_length) {
        break;
      }
      const uint64_t prefix =
          (host_address + offset) & (page->publication_granularity - 1);
      const uint64_t allocation_offset = offset - iree_min(offset, prefix);
      const uint64_t end =
          iree_min(page->byte_length,
                   iree_align_uint64(host_address + offset + byte_length,
                                     page->publication_granularity) -
                       host_address);
      const uint64_t length = end - allocation_offset;
      const iree_host_size_t allocation_first = allocation_offset / unit;
      position = iree_bitmap_find_first_unset_span(
          page->bitmap, allocation_first, length / unit);
      if (position != allocation_first) {
        continue;
      }
      iree_bitmap_set_span(page->bitmap, allocation_first, length / unit);
      page->used_bytes += length;
      if (page->used_bytes == page->byte_length) {
        iree_hal_amd_gpu_program_range_page_unlink(page);
      }
      ++pool->count;
      *out_range = (iree_hal_amd_gpu_program_range_t){
          .page = page, .offset = allocation_offset, .byte_length = length};
      *out_payload_offset = offset;
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_amd_gpu_program_range_page_allocate(
    iree_hal_amd_gpu_program_range_pool_t* pool, uint64_t minimum_length,
    uint32_t alignment, iree_hal_amd_gpu_program_range_page_t** out_page) {
  *out_page = NULL;
  const uint64_t unit = IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT;
  const uint64_t granularity = iree_max(
      pool->options->profile.allocation.native_byte_length_granularity, unit);
  const uint64_t byte_length =
      iree_host_size_ceil_div(minimum_length, granularity) * granularity;
  const uint64_t bit_count = byte_length / unit;
  iree_host_size_t total_size = 0, bitmap_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_program_range_page_t), &total_size,
      IREE_STRUCT_FIELD(iree_bitmap_calculate_words(bit_count), uint64_t,
                        &bitmap_offset)));
  iree_hal_amd_gpu_program_range_page_t* page = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(pool->host_allocator, total_size, (void**)&page));
  *page = (iree_hal_amd_gpu_program_range_page_t){
      .pool = pool,
      .byte_length = byte_length,
      .publication_granularity = 1,
      .bitmap = {.bit_count = bit_count,
                 .words = (void*)((uint8_t*)page + bitmap_offset)},
      .host_allocator = pool->host_allocator,
  };
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      pool->context->api, pool->options, byte_length, alignment,
      pool->context->event_sink, &page->memory, NULL);
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->memory_query_address(
            page->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU, &page->address),
        "query program storage address");
  }
  if (iree_status_is_ok(status) && page->memory.mapping) {
    amdf_host_mapping_info_t mapping = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(mapping),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        pool->context->api->host_mapping_query_info(page->memory.mapping,
                                                    &mapping),
        "query program publication boundaries");
    page->publication_granularity =
        iree_max(mapping.flush.range_granularity, unit);
  }
  if (iree_status_is_ok(status)) {
    *out_page = page;
  } else {
    iree_hal_amd_gpu_program_range_page_free(page);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_program_range_allocate(
    iree_hal_amd_gpu_program_range_pool_t* pool, uint64_t byte_length,
    uint32_t alignment, iree_hal_amd_gpu_program_range_t* out_range,
    iree_byte_span_t* out_contents, uint64_t* out_address) {
  *out_range = (iree_hal_amd_gpu_program_range_t){0};
  *out_contents = iree_byte_span_empty();
  *out_address = 0;
  const uint64_t length =
      iree_align_uint64(byte_length, IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT);
  alignment = iree_max(alignment, IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT);
  uint64_t payload_offset = 0;
  iree_slim_mutex_lock(&pool->mutex);
  bool acquired = iree_hal_amd_gpu_program_range_try_allocate(
      pool, length, alignment, out_range, &payload_offset);
  iree_slim_mutex_unlock(&pool->mutex);
  iree_status_t status = iree_ok_status();
  if (!acquired) {
    // Serialize only callers that need growth. Returns and acquisitions from
    // existing pages continue while a native allocation blocks in the driver.
    iree_slim_mutex_lock(&pool->growth_mutex);
    iree_slim_mutex_lock(&pool->mutex);
    acquired = iree_hal_amd_gpu_program_range_try_allocate(
        pool, length, alignment, out_range, &payload_offset);
    iree_slim_mutex_unlock(&pool->mutex);
    if (!acquired) {
      iree_hal_amd_gpu_program_range_page_t* page = NULL;
      status = iree_hal_amd_gpu_program_range_page_allocate(pool, length,
                                                            alignment, &page);
      if (iree_status_is_ok(status)) {
        const uint64_t host_address = (uintptr_t)page->memory.contents.data;
        const uint64_t reserved_length =
            iree_min(page->byte_length,
                     iree_align_uint64(host_address + length,
                                       page->publication_granularity) -
                         host_address);
        iree_bitmap_set_span(
            page->bitmap, 0,
            reserved_length / IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT);
        page->used_bytes = reserved_length;
        iree_slim_mutex_lock(&pool->mutex);
        if (reserved_length < page->byte_length) {
          iree_hal_amd_gpu_program_range_page_link(page);
        }
        ++pool->count;
        iree_slim_mutex_unlock(&pool->mutex);
        *out_range = (iree_hal_amd_gpu_program_range_t){
            .page = page, .byte_length = reserved_length};
      }
    }
    iree_slim_mutex_unlock(&pool->growth_mutex);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_amd_gpu_program_range_t range = *out_range;
    if (range.page->memory.contents.data) {
      *out_contents = iree_make_byte_span(
          range.page->memory.contents.data + payload_offset, byte_length);
    }
    *out_address = range.page->address + payload_offset;
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_program_range_publish(
    const iree_hal_amd_gpu_program_range_t* range) {
  const iree_hal_amd_gpu_program_range_page_t* page = range->page;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      page->pool->context->api->host_mapping_cache_control(
          page->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, range->offset,
          range->byte_length),
      "publish program storage");
}

void iree_hal_amd_gpu_program_range_release(
    iree_hal_amd_gpu_program_range_t* range) {
  iree_hal_amd_gpu_program_range_page_t* page = range->page;
  if (!page) {
    return;
  }
  iree_hal_amd_gpu_program_range_pool_t* pool = page->pool;
  iree_slim_mutex_lock(&pool->mutex);
  if (page->used_bytes == page->byte_length) {
    iree_hal_amd_gpu_program_range_page_link(page);
  }
  iree_bitmap_reset_span(
      page->bitmap, range->offset / IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT,
      range->byte_length / IREE_HAL_AMD_GPU_PROGRAM_RANGE_UNIT);
  page->used_bytes -= range->byte_length;
  --pool->count;
  iree_slim_mutex_unlock(&pool->mutex);
  *range = (iree_hal_amd_gpu_program_range_t){0};
}

static void iree_hal_amd_gpu_program_ranges_initialize(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_program_range_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_program_range_pool_t){
      .context = context,
      .host_allocator = host_allocator,
      .options = options,
  };
  iree_slim_mutex_initialize(&out_pool->growth_mutex);
  iree_slim_mutex_initialize(&out_pool->mutex);
}

static void iree_hal_amd_gpu_program_ranges_deinitialize(
    iree_hal_amd_gpu_program_range_pool_t* pool) {
  IREE_ASSERT(!pool->count);
  iree_hal_amd_gpu_program_ranges_trim(pool);
  iree_slim_mutex_deinitialize(&pool->growth_mutex);
  iree_slim_mutex_deinitialize(&pool->mutex);
}

// A registry traversal borrows only its current program. Removal redirects
// pending cursors under the registry mutex, so their next nodes need no borrow.
// The stack record outlives every visit and is unlinked before its caller
// exits.
struct iree_hal_amd_gpu_program_visit_t {
  // Borrowed pool, live through the control caller's final registry access.
  iree_hal_amd_gpu_program_pool_t* pool;
  // Next active traversal, protected by the registry mutex.
  iree_hal_amd_gpu_program_visit_t* next;
  // Program currently borrowed outside the registry mutex, or NULL.
  iree_hal_amd_gpu_program_t* current;
  // Next registered program to visit, redirected by concurrent removal.
  iree_hal_amd_gpu_program_t* pending;
  // A destructor is joining the current borrow and requires a return wake.
  bool joining;
};

// Called under registry exclusion after publishing any wider queue extent.
static void iree_hal_amd_gpu_program_visit_initialize(
    iree_hal_amd_gpu_program_pool_t* pool,
    iree_hal_amd_gpu_program_visit_t* out_visit) {
  *out_visit = (iree_hal_amd_gpu_program_visit_t){
      .pool = pool,
      .next = pool->registry.visits,
      .pending = pool->registry.head,
  };
  pool->registry.visits = out_visit;
}

static iree_hal_amd_gpu_program_t* iree_hal_amd_gpu_program_visit_next(
    iree_hal_amd_gpu_program_visit_t* visit) {
  iree_hal_amd_gpu_program_pool_t* pool = visit->pool;
  iree_slim_mutex_lock(&pool->registry.mutex);
  const bool joining = visit->joining;
  visit->joining = false;
  iree_hal_amd_gpu_program_t* program = visit->pending;
  visit->current = program;
  visit->pending = program ? program->next : NULL;
  iree_slim_mutex_unlock(&pool->registry.mutex);
  if (joining) {
    iree_notification_post(&pool->registry.notification, IREE_ALL_WAITERS);
  }
  return program;
}

static void iree_hal_amd_gpu_program_visit_deinitialize(
    iree_hal_amd_gpu_program_visit_t* visit) {
  iree_hal_amd_gpu_program_pool_t* pool = visit->pool;
  iree_slim_mutex_lock(&pool->registry.mutex);
  iree_hal_amd_gpu_program_visit_t** link = &pool->registry.visits;
  while (*link != visit) {
    link = &(*link)->next;
  }
  *link = visit->next;
  const bool joining = visit->joining;
  iree_slim_mutex_unlock(&pool->registry.mutex);
  if (joining) {
    iree_notification_post(&pool->registry.notification, IREE_ALL_WAITERS);
  }
}

iree_status_t iree_hal_amd_gpu_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, amdf_memory_access_t access,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_program_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_program_pool_t){
      .context = context,
      .host_allocator = host_allocator,
      .access = {.device = context->device,
                 .requirements =
                     {
                         .access = access,
                         .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                         .address_kinds = UINT64_C(1)
                                          << AMDF_MEMORY_ADDRESS_GPU,
                     }},
      .registry.queue_count = queue_count,
  };
  iree_slim_mutex_initialize(&out_pool->registry.mutex);
  iree_notification_initialize(&out_pool->registry.notification);
  iree_hal_amd_gpu_program_ranges_initialize(
      context, &out_pool->resident, host_allocator, &out_pool->constants);
  iree_hal_amd_gpu_program_ranges_initialize(
      context, &out_pool->resident, host_allocator, &out_pool->recordings);
  iree_hal_amd_gpu_program_ranges_initialize(
      context, &out_pool->upload, host_allocator, &out_pool->uploads);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_query_program_memory(
      context, native_family_ordinal, &out_pool->access, &out_pool->upload,
      &out_pool->resident));
  return iree_ok_status();
}

// Prepare outside metadata exclusion and reconcile both dimensions before
// installation. Another traversal can widen the queue table, or an invocation
// can grow the instance table, while this caller allocates its proposal.
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
      complete = queue_count <= program->queue_count;
      if (!complete && capacity == program->capacity) {
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
  iree_hal_amd_gpu_program_visit_t visit;
  iree_slim_mutex_lock(&pool->registry.mutex);
  pool->registry.queue_count =
      iree_max(pool->registry.queue_count, queue_count);
  queue_count = pool->registry.queue_count;
  iree_hal_amd_gpu_program_visit_initialize(pool, &visit);
  iree_slim_mutex_unlock(&pool->registry.mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amd_gpu_program_t* program = NULL;
  while (iree_status_is_ok(status) &&
         (program = iree_hal_amd_gpu_program_visit_next(&visit))) {
    status = iree_hal_amd_gpu_program_reserve_queues(program, queue_count);
  }
  iree_hal_amd_gpu_program_visit_deinitialize(&visit);
  return status;
}

void iree_hal_amd_gpu_program_pool_trim(iree_hal_amd_gpu_program_pool_t* pool) {
  iree_hal_amd_gpu_program_visit_t visit;
  iree_slim_mutex_lock(&pool->registry.mutex);
  iree_hal_amd_gpu_program_visit_initialize(pool, &visit);
  iree_slim_mutex_unlock(&pool->registry.mutex);
  iree_hal_amd_gpu_program_t* program = NULL;
  while ((program = iree_hal_amd_gpu_program_visit_next(&visit))) {
    iree_hal_amd_gpu_program_trim(program);
  }
  iree_hal_amd_gpu_program_visit_deinitialize(&visit);
  iree_hal_amd_gpu_program_ranges_trim(&pool->constants);
  iree_hal_amd_gpu_program_ranges_trim(&pool->recordings);
  iree_hal_amd_gpu_program_ranges_trim(&pool->uploads);
}

void iree_hal_amd_gpu_program_pool_deinitialize(
    iree_hal_amd_gpu_program_pool_t* pool) {
  if (!pool->context) {
    return;
  }
  IREE_ASSERT(!pool->registry.head);
  IREE_ASSERT(!pool->registry.visits);
  iree_hal_amd_gpu_program_ranges_deinitialize(&pool->constants);
  iree_hal_amd_gpu_program_ranges_deinitialize(&pool->recordings);
  iree_hal_amd_gpu_program_ranges_deinitialize(&pool->uploads);
  iree_notification_deinitialize(&pool->registry.notification);
  iree_slim_mutex_deinitialize(&pool->registry.mutex);
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
  if (instance == program->initial) {
    iree_hal_amd_gpu_program_range_release(&instance->storage.range);
    iree_allocator_free(instance->host_allocator, instance);
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      program->pool->context, 1, &instance->storage.allocation.memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_program_instance_released,
          .user_data = instance},
      &instance->storage.allocation.release);
}

struct iree_hal_amd_gpu_program_upload_t {
  // Cached initialization bytes retained through every initial copy reader.
  iree_hal_amd_gpu_program_range_t range;
};

static void iree_hal_amd_gpu_program_upload_free(
    iree_hal_amd_gpu_program_pool_t* pool,
    iree_hal_amd_gpu_program_upload_t* upload) {
  if (!upload) {
    return;
  }
  iree_hal_amd_gpu_program_range_release(&upload->range);
  iree_allocator_free(pool->host_allocator, upload);
}

iree_status_t iree_hal_amd_gpu_program_initialize(
    iree_hal_amd_gpu_program_pool_t* pool, uint32_t alignment,
    uint32_t byte_length, uint32_t fixup_count,
    iree_hal_amd_gpu_program_t* out_program) {
  iree_slim_mutex_lock(&pool->registry.mutex);
  const uint32_t queue_count = pool->registry.queue_count;
  iree_slim_mutex_unlock(&pool->registry.mutex);
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
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*out_program->initial),
                                             (void**)&out_program->initial));
  iree_hal_amd_gpu_program_instance_t* initial = out_program->initial;
  initial->host_allocator = pool->host_allocator;
  out_program->instances[0] = initial;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_range_allocate(
      &pool->recordings, allocation_length, alignment, &initial->storage.range,
      &out_program->contents, &initial->address));
  if (out_program->contents.data) {
    out_program->upload_address = initial->address;
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                               sizeof(*out_program->upload),
                                               (void**)&out_program->upload));
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_range_allocate(
        &pool->uploads, allocation_length, alignment,
        &out_program->upload->range, &out_program->contents,
        &out_program->upload_address));
  }
  for (uint32_t i = 0; i < queue_count; ++i) {
    out_program->queues[i].preferred = initial;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_program_seal(
    iree_hal_amd_gpu_program_t* program) {
  iree_hal_amd_gpu_program_instance_t* initial = program->initial;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_program_range_publish(
      program->upload ? &program->upload->range : &initial->storage.range));
  if (!program->upload) {
    initial->initialized = true;
    iree_atomic_store(&program->source_ready, 1, iree_memory_order_release);
  }
  iree_hal_amd_gpu_program_pool_t* pool = program->pool;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&pool->registry.mutex);
  while (program->queue_count < pool->registry.queue_count &&
         iree_status_is_ok(status)) {
    const uint32_t queue_count = pool->registry.queue_count;
    iree_slim_mutex_unlock(&pool->registry.mutex);
    status = iree_hal_amd_gpu_program_reserve_queues(program, queue_count);
    iree_slim_mutex_lock(&pool->registry.mutex);
  }
  if (iree_status_is_ok(status)) {
    program->next = pool->registry.head;
    if (program->next) {
      program->next->previous = program;
    }
    pool->registry.head = program;
  }
  iree_slim_mutex_unlock(&pool->registry.mutex);
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
      program->pool->context, 1, &operation->request,
      &instance->storage.allocation.memory, callback, &operation->acquire);
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
            instance->storage.allocation.memory.handle, 0,
            AMDF_MEMORY_ADDRESS_GPU, &instance->address),
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

static bool iree_hal_amd_gpu_program_visits_joined(void* user_data) {
  iree_hal_amd_gpu_program_t* program = user_data;
  iree_hal_amd_gpu_program_pool_t* pool = program->pool;
  bool joined = true;
  iree_slim_mutex_lock(&pool->registry.mutex);
  for (iree_hal_amd_gpu_program_visit_t* visit = pool->registry.visits; visit;
       visit = visit->next) {
    joined &= visit->current != program;
  }
  iree_slim_mutex_unlock(&pool->registry.mutex);
  return joined;
}

void iree_hal_amd_gpu_program_deinitialize(
    iree_hal_amd_gpu_program_t* program) {
  if (!program->pool) {
    return;
  }
  iree_hal_amd_gpu_program_pool_t* pool = program->pool;
  iree_slim_mutex_lock(&pool->registry.mutex);
  if (program->previous) {
    program->previous->next = program->next;
  }
  if (program->next) {
    program->next->previous = program->previous;
  }
  if (pool->registry.head == program) {
    pool->registry.head = program->next;
  }
  bool joining = false;
  for (iree_hal_amd_gpu_program_visit_t* visit = pool->registry.visits; visit;
       visit = visit->next) {
    if (visit->pending == program) {
      visit->pending = program->next;
    }
    if (visit->current == program) {
      visit->joining = true;
      joining = true;
    }
  }
  iree_slim_mutex_unlock(&pool->registry.mutex);
  if (joining) {
    iree_notification_await(&pool->registry.notification,
                            iree_hal_amd_gpu_program_visits_joined, program,
                            iree_infinite_timeout());
  }
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
