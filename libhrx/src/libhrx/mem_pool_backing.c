// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "mem_pool_backing.h"

#include <stdlib.h>
#include <string.h>

#include "hrx_internal.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"

// Default slab length for growable GPU allocation pools.
static const iree_device_size_t HRX_MEM_POOL_GPU_SLAB_LENGTH_DEFAULT =
    (iree_device_size_t)256 * 1024 * 1024;

// Default range length for CPU/local allocation pools.
static const iree_device_size_t HRX_MEM_POOL_CPU_RANGE_LENGTH_DEFAULT =
    (iree_device_size_t)64 * 1024 * 1024;

// Minimum byte alignment for HRX memory-pool reservations.
static const iree_device_size_t HRX_MEM_POOL_ALIGNMENT = 256;

static iree_status_t hrx_mem_pool_parse_range_length_env(
    const char* name, bool* out_found, iree_device_size_t* out_length) {
  *out_found = false;
  *out_length = 0;

  const char* value = getenv(name);
  if (!value || !value[0]) {
    return iree_ok_status();
  }

  char* end = NULL;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (value[0] < '0' || value[0] > '9' || !end || *end != '\0' || parsed == 0 ||
      parsed > (unsigned long long)IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s must be a positive byte count, got '%s'", name,
                            value);
  }

  *out_found = true;
  *out_length = (iree_device_size_t)parsed;
  return iree_ok_status();
}

static iree_status_t hrx_mem_pool_query_range_length(
    hrx_device_t device, iree_device_size_t* out_range_length) {
  bool has_env_range_length = false;
  iree_device_size_t range_length = 0;
  IREE_RETURN_IF_ERROR(hrx_mem_pool_parse_range_length_env(
      "HRX_MEM_POOL_BYTES", &has_env_range_length, &range_length));
  if (!has_env_range_length) {
    IREE_RETURN_IF_ERROR(hrx_mem_pool_parse_range_length_env(
        "HRX_HIP_POOL_BYTES", &has_env_range_length, &range_length));
  }

  if (!has_env_range_length) {
    if (device->type == HRX_ACCELERATOR_GPU) {
      // TLSF grows by this range length one slab at a time. A slab is fully
      // committed on first use, so its size sets the growth and idle-retention
      // granularity rather than expressing a fraction of total device memory.
      range_length = HRX_MEM_POOL_GPU_SLAB_LENGTH_DEFAULT;
    } else {
      range_length = HRX_MEM_POOL_CPU_RANGE_LENGTH_DEFAULT;
    }
  }

  if (!iree_device_size_checked_align(range_length, HRX_MEM_POOL_ALIGNMENT,
                                      &range_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory pool range length overflows alignment");
  }
  *out_range_length = range_length;
  return iree_ok_status();
}

typedef enum hrx_mem_pool_source_kind_e {
  HRX_MEM_POOL_SOURCE_DEVICE = 0,
  HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE,
  HRX_MEM_POOL_SOURCE_HOST,
  HRX_MEM_POOL_SOURCE_UNCACHED_HOST,
  HRX_MEM_POOL_SOURCE_SIGNAL,
  HRX_MEM_POOL_SOURCE_HOST_IMPORT,
  HRX_MEM_POOL_SOURCE_DEVICE_IMPORT,
} hrx_mem_pool_source_kind_t;

static iree_hal_pool_t** hrx_mem_pool_backing_source_slot(
    hrx_mem_pool_backing_t* backing, hrx_mem_pool_source_kind_t kind,
    hrx_buffer_usage_t usage) {
  const bool mapping_source =
      iree_any_bit_set(usage, HRX_BUFFER_USAGE_MAPPING_SOURCE);
  switch (kind) {
    case HRX_MEM_POOL_SOURCE_DEVICE:
      return mapping_source ? &backing->device.mapping
                            : &backing->device.ordinary;
    case HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE:
      return mapping_source ? &backing->uncached_device.mapping
                            : &backing->uncached_device.ordinary;
    case HRX_MEM_POOL_SOURCE_HOST:
      return mapping_source ? &backing->host.mapping : &backing->host.ordinary;
    case HRX_MEM_POOL_SOURCE_UNCACHED_HOST:
      return mapping_source ? &backing->uncached_host.mapping
                            : &backing->uncached_host.ordinary;
    case HRX_MEM_POOL_SOURCE_SIGNAL:
      return &backing->signals;
    case HRX_MEM_POOL_SOURCE_HOST_IMPORT:
      return &backing->host_import_source;
    case HRX_MEM_POOL_SOURCE_DEVICE_IMPORT:
      return &backing->device_import_source;
  }
  return NULL;
}

// Ordinary HRX sources span every family of their device. Remappable backing
// spans the sealed group so later HIP access grants can name any participating
// device. Construction qualifies that complete access set before allocation.
// No source allocates payload until used.
static iree_status_t hrx_mem_pool_backing_create_source(
    hrx_device_t device, hrx_mem_pool_source_kind_t kind,
    hrx_buffer_usage_t usage, iree_hal_pool_t** out_pool) {
  const bool mapping_source =
      iree_any_bit_set(usage, HRX_BUFFER_USAGE_MAPPING_SOURCE);
  const bool uncached = kind == HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE ||
                        kind == HRX_MEM_POOL_SOURCE_UNCACHED_HOST;
  const bool host_coherent = kind == HRX_MEM_POOL_SOURCE_HOST ||
                             kind == HRX_MEM_POOL_SOURCE_UNCACHED_HOST ||
                             kind == HRX_MEM_POOL_SOURCE_SIGNAL ||
                             kind == HRX_MEM_POOL_SOURCE_HOST_IMPORT;
  const iree_host_size_t device_count =
      mapping_source
          ? iree_hal_device_group_device_count(device->hal_device_group)
          : 1;
  iree_host_size_t family_count = 0;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_hal_device_t* participant =
        mapping_source
            ? iree_hal_device_group_device_at(device->hal_device_group, i)
            : device->hal_device;
    family_count +=
        iree_hal_device_spec_queues(iree_hal_device_spec(participant))
            ->family_count;
  }
  iree_hal_pool_family_access_t* families = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(iree_allocator_system(), family_count,
                                  sizeof(*families), (void**)&families));
  iree_host_size_t family_index = 0;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_hal_device_t* participant =
        mapping_source
            ? iree_hal_device_group_device_at(device->hal_device_group, i)
            : device->hal_device;
    const iree_hal_device_queue_spec_t* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(participant));
    for (iree_host_size_t j = 0; j < queues->family_count; ++j) {
      iree_hal_pool_family_access_t* family = &families[family_index++];
      family->family = iree_hal_device_queue_family(participant, j);
      family->usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
      if (mapping_source) {
        family->interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
      } else {
        family->interfaces =
            UINT64_C(1) << (device->type == HRX_ACCELERATOR_CPU
                                ? IREE_HAL_BUFFER_INTERFACE_HOST
                                : IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS);
      }
      family->requirements =
          (uncached ? IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED : 0) |
          (host_coherent ? IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST : 0);
      if (kind == HRX_MEM_POOL_SOURCE_SIGNAL &&
          family->family == iree_hal_queue_family(device->dispatch_queue)) {
        const iree_hal_atomic_operation_flags_t operations =
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
        family->atomics.queue_zero_compute =
            (iree_hal_atomic_operation_capabilities_t){
                .device_scope_32 = operations,
                .device_scope_64 = operations,
                .system_scope_32 = operations,
                .system_scope_64 = operations,
            };
      }
    }
  }
  const iree_hal_pool_host_access_t host = {
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT,
      .properties = IREE_HAL_HOST_MEMORY_PROPERTY_NONE,
  };
  iree_hal_pool_scope_t scope = {
      .family_count = family_count,
      .families = families,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  const bool imports_host = kind == HRX_MEM_POOL_SOURCE_HOST_IMPORT;
  const bool imports_device = kind == HRX_MEM_POOL_SOURCE_DEVICE_IMPORT;
  const bool import_only = imports_host || imports_device;
  iree_hal_pool_import_class_t import_class = {0};
  if (import_only) {
    import_class.storage_class = iree_hal_make_external_storage_class(
        imports_host ? IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER
                     : IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS);
    options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
  }
  const bool requires_host =
      host_coherent || (kind == HRX_MEM_POOL_SOURCE_DEVICE &&
                        device->type == HRX_ACCELERATOR_CPU);
  if (requires_host) {
    scope.host = host;
  } else if (!import_only && kind != HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE) {
    options.preferences.host = host;
  }
  if ((kind == HRX_MEM_POOL_SOURCE_DEVICE ||
       kind == HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE) &&
      device->type != HRX_ACCELERATOR_CPU) {
    const iree_hal_device_topology_info_t* topology_info =
        iree_hal_device_topology_info(device->hal_device);
    const iree_hal_queue_family_spec_t* dispatch_family =
        iree_hal_queue_family_spec(
            iree_hal_queue_family(device->dispatch_queue));
    const iree_hal_topology_t* topology =
        iree_hal_device_group_topology(device->hal_device_group);
    for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
         ++i) {
      const iree_hal_topology_node_t* node =
          iree_hal_topology_node_at(topology, i);
      if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
          node->device_ordinal == topology_info->topology_index &&
          iree_any_bit_set(node->physical_device_affinity,
                           dispatch_family->physical_device_affinity)) {
        options.placement.mode = IREE_HAL_POOL_PLACEMENT_PREFERRED;
        options.placement.node = node->ordinal;
        break;
      }
    }
  }
  switch (kind) {
    case HRX_MEM_POOL_SOURCE_DEVICE:
      options.trace_name = iree_make_cstring_view("hrx-device-source");
      break;
    case HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE:
      options.trace_name = iree_make_cstring_view("hrx-uncached-source");
      break;
    case HRX_MEM_POOL_SOURCE_HOST:
      options.trace_name = iree_make_cstring_view("hrx-host-source");
      break;
    case HRX_MEM_POOL_SOURCE_UNCACHED_HOST:
      options.trace_name = iree_make_cstring_view("hrx-uncached-host-source");
      break;
    case HRX_MEM_POOL_SOURCE_SIGNAL:
      options.trace_name = iree_make_cstring_view("hrx-signal-source");
      break;
    case HRX_MEM_POOL_SOURCE_HOST_IMPORT:
      options.trace_name = iree_make_cstring_view("hrx-host-import-source");
      break;
    case HRX_MEM_POOL_SOURCE_DEVICE_IMPORT:
      options.trace_name = iree_make_cstring_view("hrx-device-import-source");
      break;
  }
  iree_status_t status =
      iree_hal_slab_pool_create(device->hal_device_group, scope, &options,
                                iree_allocator_system(), out_pool);
  iree_allocator_free(iree_allocator_system(), families);
  return status;
}

static iree_status_t hrx_mem_pool_backing_create_signal_pool(
    hrx_device_t device, iree_hal_pool_t** out_pool) {
  // A 4 KiB logical slab holds 512 signals. Native allocation granularity can
  // round backing upward; per-signal reservations remain eight bytes. ASAN
  // guards are accounted for by the fixed-block pool's backing request.
  iree_hal_fixed_block_pool_options_t options = {
      .block_size = sizeof(uint64_t),
      .blocks_per_slab = 512,
      .alignment = sizeof(uint64_t),
      .asan = iree_hal_device_spec_sanitizer(
                  iree_hal_device_spec(device->hal_device))
                  ->asan.pool_options,
      .trace_name = IREE_SV("hrx-signals"),
  };
  iree_hal_pool_t* source = NULL;
  iree_status_t status = hrx_mem_pool_backing_create_source(
      device, HRX_MEM_POOL_SOURCE_SIGNAL, HRX_BUFFER_USAGE_DEFAULT, &source);
  iree_hal_slab_cache_options_t cache_options = {.max_count = 1};
  if (iree_status_is_ok(status)) {
    status = iree_hal_fixed_block_pool_query_backing_request(
        source, &options, &cache_options.slab);
  }
  iree_hal_pool_t* cache = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_slab_cache_create(source, &cache_options,
                                        iree_allocator_system(), &cache);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_fixed_block_pool_create(
        cache, &options, iree_allocator_system(), out_pool);
  }
  iree_hal_pool_release(cache);
  iree_hal_pool_release(source);
  return status;
}

static iree_status_t hrx_mem_pool_backing_ensure_cache_locked(
    hrx_device_t device) {
  hrx_mem_pool_backing_t* backing = &device->mem_pool_backing;
  if (backing->cache) {
    return iree_ok_status();
  }

  iree_device_size_t range_length = 0;
  IREE_RETURN_IF_ERROR(hrx_mem_pool_query_range_length(device, &range_length));

  iree_hal_tlsf_pool_options_t options = {0};
  options.tlsf_options.range_length = range_length;
  options.tlsf_options.alignment = HRX_MEM_POOL_ALIGNMENT;
  options.tlsf_options.frontier_capacity =
      IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY;
  options.asan =
      iree_hal_device_spec_sanitizer(iree_hal_device_spec(device->hal_device))
          ->asan.pool_options;
  options.budget_limit = 0;
  options.trace_name = iree_make_cstring_view("hrx-mem-pool");

  iree_hal_pool_t* backing_pool = NULL;
  iree_status_t status = iree_ok_status();
  if (!backing->device.ordinary) {
    status = hrx_mem_pool_backing_create_source(
        device, HRX_MEM_POOL_SOURCE_DEVICE, HRX_BUFFER_USAGE_DEFAULT,
        &backing->device.ordinary);
  }
  if (iree_status_is_ok(status)) {
    backing_pool = backing->device.ordinary;
    iree_hal_pool_retain(backing_pool);
  }
  iree_hal_pool_t* backing_cache = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_slab_cache_options_t cache_options = {.max_count = UINT32_MAX};
    status = iree_hal_tlsf_pool_query_backing_request(backing_pool, &options,
                                                      &cache_options.slab);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_slab_cache_create(backing_pool, &cache_options,
                                     iree_allocator_system(), &backing_cache);
    }
  }
  iree_hal_pool_release(backing_pool);
  if (iree_status_is_ok(status)) {
    backing->cache = backing_cache;
    backing->pool_options = options;
  } else {
    iree_hal_pool_release(backing_cache);
  }
  return status;
}

void hrx_mem_pool_backing_initialize(hrx_mem_pool_backing_t* backing) {
  memset(backing, 0, sizeof(*backing));
  iree_slim_mutex_initialize(&backing->mutex);
}

void hrx_mem_pool_backing_deinitialize(hrx_mem_pool_backing_t* backing) {
  // Every attached pool retains the device containing this owner.
  IREE_ASSERT(!backing->retention_head);
  iree_hal_pool_release(backing->cache);
  iree_hal_pool_release(backing->device.ordinary);
  iree_hal_pool_release(backing->device.mapping);
  iree_hal_pool_release(backing->uncached_device.ordinary);
  iree_hal_pool_release(backing->uncached_device.mapping);
  iree_hal_pool_release(backing->host.ordinary);
  iree_hal_pool_release(backing->host.mapping);
  iree_hal_pool_release(backing->uncached_host.ordinary);
  iree_hal_pool_release(backing->uncached_host.mapping);
  iree_hal_pool_release(backing->signals);
  iree_hal_pool_release(backing->host_import_source);
  iree_hal_pool_release(backing->device_import_source);
  iree_slim_mutex_deinitialize(&backing->mutex);
}

static iree_status_t hrx_mem_pool_backing_retain_source(
    hrx_device_t device, hrx_mem_pool_source_kind_t kind,
    hrx_buffer_usage_t usage, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  hrx_mem_pool_backing_t* backing = &device->mem_pool_backing;
  iree_slim_mutex_lock(&backing->mutex);
  iree_hal_pool_t** source_slot =
      hrx_mem_pool_backing_source_slot(backing, kind, usage);
  iree_status_t status = iree_ok_status();
  if (!*source_slot) {
    status = kind == HRX_MEM_POOL_SOURCE_SIGNAL
                 ? hrx_mem_pool_backing_create_signal_pool(device, source_slot)
                 : hrx_mem_pool_backing_create_source(device, kind, usage,
                                                      source_slot);
  }
  if (iree_status_is_ok(status)) {
    *out_pool = *source_slot;
    iree_hal_pool_retain(*out_pool);
  }
  iree_slim_mutex_unlock(&backing->mutex);
  return status;
}

iree_status_t hrx_mem_pool_backing_retain_allocation_source(
    hrx_device_t device, hrx_memory_type_t memory_type,
    hrx_buffer_usage_t usage, iree_hal_pool_t** out_pool) {
  hrx_mem_pool_source_kind_t kind = HRX_MEM_POOL_SOURCE_DEVICE;
  const bool host_local =
      iree_all_bits_set(memory_type, HRX_MEMORY_TYPE_HOST_LOCAL);
  const bool uncached =
      iree_any_bit_set(memory_type, HRX_MEMORY_TYPE_DEVICE_UNCACHED);
  if (host_local && uncached) {
    kind = HRX_MEM_POOL_SOURCE_UNCACHED_HOST;
  } else if (host_local) {
    kind = HRX_MEM_POOL_SOURCE_HOST;
  } else if (uncached) {
    kind = HRX_MEM_POOL_SOURCE_UNCACHED_DEVICE;
  }
  return hrx_mem_pool_backing_retain_source(device, kind, usage, out_pool);
}

iree_status_t hrx_mem_pool_backing_retain_signal_pool(
    hrx_device_t device, iree_hal_pool_t** out_pool) {
  return hrx_mem_pool_backing_retain_source(device, HRX_MEM_POOL_SOURCE_SIGNAL,
                                            HRX_BUFFER_USAGE_DEFAULT, out_pool);
}

iree_status_t hrx_mem_pool_backing_retain_import_source(
    hrx_device_t device, iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_t** out_pool) {
  hrx_mem_pool_source_kind_t kind;
  switch (storage_class.type) {
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER:
      kind = HRX_MEM_POOL_SOURCE_HOST_IMPORT;
      break;
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS:
      kind = HRX_MEM_POOL_SOURCE_DEVICE_IMPORT;
      break;
    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "HRX has no source for external storage type %u",
                              (uint32_t)storage_class.type);
  }
  return hrx_mem_pool_backing_retain_source(device, kind, HRX_BUFFER_USAGE_NONE,
                                            out_pool);
}

iree_status_t hrx_mem_pool_backing_create_pool(hrx_device_t device,
                                               iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  hrx_mem_pool_backing_t* backing = &device->mem_pool_backing;
  iree_slim_mutex_lock(&backing->mutex);
  iree_status_t status = hrx_mem_pool_backing_ensure_cache_locked(device);
  iree_slim_mutex_unlock(&backing->mutex);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  // The device retains the cache and immutable options throughout this call.
  return iree_hal_tlsf_pool_create(backing->cache, &backing->pool_options,
                                   iree_allocator_system(), out_pool);
}

// Retention is changed only by cold policy operations. Allocation and steady
// free completion use the cached maximum without traversing other pools.
static void hrx_mem_pool_backing_recompute_retention(
    hrx_mem_pool_backing_t* backing) {
  uint64_t min_bytes_to_keep = 0;
  for (hrx_mem_pool_retention_t* entry = backing->retention_head; entry;
       entry = entry->next) {
    min_bytes_to_keep = iree_max(min_bytes_to_keep, entry->min_bytes_to_keep);
  }
  backing->min_bytes_to_keep = min_bytes_to_keep;
}

static void hrx_mem_pool_backing_set_retention_locked(
    hrx_mem_pool_backing_t* backing, hrx_mem_pool_retention_t* retention,
    uint64_t min_bytes_to_keep) {
  const uint64_t previous = retention->min_bytes_to_keep;
  retention->min_bytes_to_keep = min_bytes_to_keep;
  if (min_bytes_to_keep > backing->min_bytes_to_keep) {
    backing->min_bytes_to_keep = min_bytes_to_keep;
  } else if (min_bytes_to_keep < previous &&
             previous == backing->min_bytes_to_keep) {
    hrx_mem_pool_backing_recompute_retention(backing);
  }
}

void hrx_mem_pool_backing_attach(hrx_mem_pool_backing_t* backing,
                                 hrx_mem_pool_retention_t* retention) {
  iree_slim_mutex_lock(&backing->mutex);
  *retention = (hrx_mem_pool_retention_t){.next = backing->retention_head};
  if (retention->next) {
    retention->next->previous = retention;
  }
  backing->retention_head = retention;
  iree_slim_mutex_unlock(&backing->mutex);
}

void hrx_mem_pool_backing_detach(hrx_mem_pool_backing_t* backing,
                                 hrx_mem_pool_retention_t* retention) {
  iree_slim_mutex_lock(&backing->mutex);
  if (retention->previous) {
    retention->previous->next = retention->next;
  } else {
    backing->retention_head = retention->next;
  }
  if (retention->next) {
    retention->next->previous = retention->previous;
  }
  if (retention->min_bytes_to_keep == backing->min_bytes_to_keep) {
    hrx_mem_pool_backing_recompute_retention(backing);
  }
  iree_slim_mutex_unlock(&backing->mutex);
}

void hrx_mem_pool_backing_set_retention(hrx_mem_pool_backing_t* backing,
                                        hrx_mem_pool_retention_t* retention,
                                        uint64_t min_bytes_to_keep) {
  iree_slim_mutex_lock(&backing->mutex);
  hrx_mem_pool_backing_set_retention_locked(backing, retention,
                                            min_bytes_to_keep);
  iree_slim_mutex_unlock(&backing->mutex);
}

void hrx_mem_pool_backing_trim(hrx_mem_pool_backing_t* backing,
                               hrx_mem_pool_retention_t* retention,
                               uint64_t min_bytes_to_keep) {
  iree_slim_mutex_lock(&backing->mutex);
  hrx_mem_pool_backing_set_retention_locked(backing, retention,
                                            min_bytes_to_keep);
  if (backing->cache) {
    iree_hal_pool_trim(backing->cache, IREE_HAL_POOL_TRIM_FLAG_EXCESS,
                       (iree_device_size_t)iree_min(backing->min_bytes_to_keep,
                                                    IREE_DEVICE_SIZE_MAX));
  }
  iree_slim_mutex_unlock(&backing->mutex);
}
