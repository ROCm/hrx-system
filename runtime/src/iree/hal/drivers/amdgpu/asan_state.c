// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/asan_state.h"

#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

#define IREE_HAL_AMDGPU_ASAN_SHADOW_ADDRESSABLE 0x00u
#define IREE_HAL_AMDGPU_ASAN_SHADOW_HEAP_REDZONE 0xFAu

static bool iree_hal_amdgpu_asan_align_address(uint64_t address,
                                               uint64_t alignment,
                                               uint64_t* out_address) {
  *out_address = address;
  if (alignment <= 1) return true;
  if (address > UINT64_MAX - (alignment - 1)) return false;
  *out_address = (address + (alignment - 1)) & ~(alignment - 1);
  return true;
}

static uint32_t iree_hal_amdgpu_asan_shadow_fill_pattern(uint8_t value) {
  return ((uint32_t)value << 24) | ((uint32_t)value << 16) |
         ((uint32_t)value << 8) | (uint32_t)value;
}

static iree_status_t iree_hal_amdgpu_asan_write_shadow_bytes(
    iree_hal_amdgpu_asan_state_t* state, uint64_t shadow_address,
    iree_device_size_t length, uint8_t value) {
  if (length == 0) return iree_ok_status();
  if (IREE_UNLIKELY(length > SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ASAN shadow write length %" PRIu64
                            " exceeds HSA capacity",
                            (uint64_t)length);
  }

  const uint8_t fill_bytes[4] = {value, value, value, value};
  iree_device_size_t prefix_length = 0;
  const uint64_t aligned_shadow_address = iree_device_align(shadow_address, 4);
  if (aligned_shadow_address != shadow_address) {
    prefix_length = iree_min(length, aligned_shadow_address - shadow_address);
    IREE_RETURN_IF_ERROR(iree_hsa_memory_copy(
        IREE_LIBHSA(state->libhsa), (void*)(uintptr_t)shadow_address,
        fill_bytes, (size_t)prefix_length));
    shadow_address += prefix_length;
    length -= prefix_length;
  }

  const iree_device_size_t fill_length = length & ~(iree_device_size_t)3;
  if (fill_length > 0) {
    IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_fill(
        IREE_LIBHSA(state->libhsa), (void*)(uintptr_t)shadow_address,
        iree_hal_amdgpu_asan_shadow_fill_pattern(value),
        (size_t)(fill_length / sizeof(uint32_t))));
    shadow_address += fill_length;
    length -= fill_length;
  }

  if (length > 0) {
    IREE_RETURN_IF_ERROR(iree_hsa_memory_copy(IREE_LIBHSA(state->libhsa),
                                              (void*)(uintptr_t)shadow_address,
                                              fill_bytes, (size_t)length));
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_asan_write_shadow_bytes_raw(
    iree_hal_amdgpu_asan_state_t* state, uint64_t shadow_address,
    iree_device_size_t length, uint8_t value) {
  if (length == 0) return;
  IREE_ASSERT(length <= SIZE_MAX);

  const uint8_t fill_bytes[4] = {value, value, value, value};
  iree_device_size_t prefix_length = 0;
  const uint64_t aligned_shadow_address = iree_device_align(shadow_address, 4);
  if (aligned_shadow_address != shadow_address) {
    prefix_length = iree_min(length, aligned_shadow_address - shadow_address);
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_memory_copy_raw(
        state->libhsa, (void*)(uintptr_t)shadow_address, fill_bytes,
        (size_t)prefix_length));
    shadow_address += prefix_length;
    length -= prefix_length;
  }

  const iree_device_size_t fill_length = length & ~(iree_device_size_t)3;
  if (fill_length > 0) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_memory_fill_raw(
        state->libhsa, (void*)(uintptr_t)shadow_address,
        iree_hal_amdgpu_asan_shadow_fill_pattern(value),
        (size_t)(fill_length / sizeof(uint32_t))));
    shadow_address += fill_length;
    length -= fill_length;
  }

  if (length > 0) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_memory_copy_raw(
        state->libhsa, (void*)(uintptr_t)shadow_address, fill_bytes,
        (size_t)length));
  }
}

static iree_status_t iree_hal_amdgpu_asan_read_shadow_byte(
    iree_hal_amdgpu_asan_state_t* state, uint64_t shadow_address,
    uint8_t* out_value) {
  *out_value = 0;
  return iree_hsa_memory_copy(IREE_LIBHSA(state->libhsa), out_value,
                              (void*)(uintptr_t)shadow_address,
                              sizeof(*out_value));
}

static bool iree_hal_amdgpu_asan_shadow_value_is_magic(
    iree_hal_amdgpu_asan_state_t* state, uint8_t value) {
  const uint16_t shadow_granule = (uint16_t)1u
                                  << state->shadow_map.shadow_scale_shift;
  return value >= shadow_granule;
}

static uint8_t iree_hal_amdgpu_asan_max_shadow_value(
    iree_hal_amdgpu_asan_state_t* state, uint8_t lhs, uint8_t rhs) {
  return iree_hal_amdgpu_asan_shadow_value_is_magic(state, lhs) || lhs < rhs
             ? rhs
             : lhs;
}

static iree_status_t iree_hal_amdgpu_asan_application_range_allocate(
    iree_hal_amdgpu_asan_state_t* state, uint64_t address,
    iree_device_size_t length,
    iree_hal_amdgpu_asan_application_range_t** out_range) {
  IREE_ASSERT_ARGUMENT(out_range);
  *out_range = NULL;
  iree_hal_amdgpu_asan_application_range_t* range = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(state->host_allocator,
                                             sizeof(*range), (void**)&range));
  range->address = address;
  range->length = length;
  range->next = NULL;
  *out_range = range;
  return iree_ok_status();
}

static void iree_hal_amdgpu_asan_application_range_free(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* range) {
  iree_allocator_free(state->host_allocator, range);
}

static void iree_hal_amdgpu_asan_application_range_free_list(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* range) {
  while (range) {
    iree_hal_amdgpu_asan_application_range_t* next_range = range->next;
    iree_hal_amdgpu_asan_application_range_free(state, range);
    range = next_range;
  }
}

static void iree_hal_amdgpu_asan_quarantine_release_list(
    iree_hal_amdgpu_asan_quarantine_entry_t* entry) {
  while (entry) {
    iree_hal_amdgpu_asan_quarantine_entry_t* next_entry = entry->next;
    entry->next = NULL;
    entry->release_fn(entry->user_data);
    entry = next_entry;
  }
}

static void iree_hal_amdgpu_asan_state_flush_quarantine(
    iree_hal_amdgpu_asan_state_t* state) {
  iree_slim_mutex_lock(&state->quarantine_mutex);
  iree_hal_amdgpu_asan_quarantine_entry_t* entry = state->quarantine_head;
  state->quarantine_head = NULL;
  state->quarantine_tail = NULL;
  state->quarantine_size = 0;
  iree_slim_mutex_unlock(&state->quarantine_mutex);
  iree_hal_amdgpu_asan_quarantine_release_list(entry);
}

typedef struct iree_hal_amdgpu_asan_shadow_backing_selection_t {
  // HSA memory pool used for physical shadow slabs.
  hsa_amd_memory_pool_t memory_pool;

  // HSA VMM memory type used for physical shadow slabs.
  iree_hal_amdgpu_vmem_memory_type_t memory_type;

  // Human-readable policy name used in diagnostics.
  const char* name;
} iree_hal_amdgpu_asan_shadow_backing_selection_t;

static iree_status_t iree_hal_amdgpu_asan_state_select_shadow_backing(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_asan_shadow_backing_selection_t* out_selection) {
  memset(out_selection, 0, sizeof(*out_selection));
  switch (options->asan.shadow_backing) {
    case IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL:
      out_selection->memory_pool =
          physical_device->coarse_block_pools.large.memory_pool;
      out_selection->memory_type = IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_DEFAULT;
      out_selection->name = "device-local";
      if (IREE_UNLIKELY(!out_selection->memory_pool.handle)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU ASAN device-local shadow backing requires a "
            "coarse-grained device-local memory pool");
      }
      return iree_ok_status();
    case IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_HOST_LOCAL:
      out_selection->memory_pool =
          physical_device->coarse_block_pools.large.memory_pool;
      out_selection->memory_type = IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST;
      out_selection->name = "host-local";
      if (IREE_UNLIKELY(!out_selection->memory_pool.handle)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU ASAN host-local shadow backing requires a "
            "coarse-grained device-local memory pool");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid AMDGPU ASAN shadow backing value %u",
                              (uint32_t)options->asan.shadow_backing);
  }
}

static iree_status_t iree_hal_amdgpu_asan_state_initialize_shadow_map(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    uint64_t application_coverage_base, iree_allocator_t host_allocator,
    iree_hal_amdgpu_shadow_map_t* out_shadow_map) {
  if (IREE_UNLIKELY(physical_device_count == 0 || !physical_devices ||
                    !physical_devices[0])) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN requires at least one initialized physical device");
  }

  iree_hal_amdgpu_physical_device_t* physical_device = physical_devices[0];
  iree_hal_amdgpu_asan_shadow_backing_selection_t shadow_backing = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_state_select_shadow_backing(
      options, physical_device, &shadow_backing));

  hsa_amd_memory_access_desc_t access_descs[IREE_HAL_AMDGPU_MAX_CPU_AGENT +
                                            IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  iree_host_size_t access_desc_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_build_access_descs_for_topology(
      &system->topology, physical_device->device_agent,
      IREE_HAL_AMDGPU_ACCESS_MODE_SHARED, IREE_ARRAYSIZE(access_descs),
      access_descs, &access_desc_count));

  iree_hal_amdgpu_shadow_map_hsa_params_t shadow_map_params = {
      .host_allocator = host_allocator,
      .libhsa = &system->libhsa,
      .memory_pool = shadow_backing.memory_pool,
      .memory_type = shadow_backing.memory_type,
      .shadow_scale_shift = options->asan.shadow_scale_shift,
      .application_window_base = application_coverage_base,
      .shadow_size = options->asan.shadow_size,
      .requested_slab_size = options->asan.shadow_slab_size,
      .mapping_mode = options->asan.shadow_mode ==
                              IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_PREMAPPED
                          ? IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_PREMAPPED
                          : IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_SPARSE,
      .initial_slab_value = IREE_HAL_AMDGPU_ASAN_SHADOW_HEAP_REDZONE,
      .access_desc_count = access_desc_count,
      .access_descs = access_descs,
  };
  iree_status_t status = iree_hal_amdgpu_shadow_map_initialize_hsa(
      &shadow_map_params, out_shadow_map);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status,
                                  "initializing AMDGPU ASAN %s shadow backing",
                                  shadow_backing.name);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_asan_state_calculate_application_coverage(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_device_size_t* out_application_coverage_size) {
  *out_application_coverage_size = 0;
  if (IREE_UNLIKELY(
          options->asan.shadow_size == 0 ||
          !iree_device_size_is_power_of_two(options->asan.shadow_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN shadow size %" PRIu64
                            " must be a non-zero power of two",
                            (uint64_t)options->asan.shadow_size);
  }
  if (IREE_UNLIKELY(
          options->asan.shadow_slab_size == 0 ||
          !iree_device_size_is_power_of_two(options->asan.shadow_slab_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN shadow slab size %" PRIu64
                            " must be a non-zero power of two",
                            (uint64_t)options->asan.shadow_slab_size);
  }
  if (IREE_UNLIKELY(options->asan.shadow_scale_shift >
                    IREE_HAL_AMDGPU_ASAN_MAX_SHADOW_SCALE_SHIFT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU ASAN shadow scale shift %u exceeds max representable shift %u",
        options->asan.shadow_scale_shift,
        IREE_HAL_AMDGPU_ASAN_MAX_SHADOW_SCALE_SHIFT);
  }
  if (IREE_UNLIKELY(
          options->asan.shadow_size >
          (IREE_DEVICE_SIZE_MAX >> options->asan.shadow_scale_shift))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU ASAN application coverage size overflows: shadow_size=%" PRIu64
        ", scale_shift=%u",
        (uint64_t)options->asan.shadow_size, options->asan.shadow_scale_shift);
  }
  *out_application_coverage_size = options->asan.shadow_size
                                   << options->asan.shadow_scale_shift;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_asan_state_reserve_owned_application_window(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_device_size_t window_size,
    iree_device_size_t window_alignment,
    IREE_AMDGPU_DEVICE_PTR void** out_base_ptr) {
  *out_base_ptr = NULL;
  return iree_hsa_amd_vmem_address_reserve_align(
      IREE_LIBHSA(libhsa), out_base_ptr, window_size,
      IREE_HAL_AMDGPU_ASAN_PREFERRED_APPLICATION_WINDOW_BASE, window_alignment,
      /*flags=*/0);
}

iree_status_t iree_hal_amdgpu_asan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_asan_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (!options->asan.enabled) return iree_ok_status();

  out_state->host_allocator = host_allocator;
  iree_device_size_t application_coverage_size = 0;
  iree_status_t status =
      iree_hal_amdgpu_asan_state_calculate_application_coverage(
          options, &application_coverage_size);
  if (iree_status_is_ok(status) &&
      (options->asan.owned_application_size == 0 ||
       !iree_device_size_is_power_of_two(
           options->asan.owned_application_size))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU ASAN owned application size %" PRIu64
                              " must be a non-zero power of two",
                              (uint64_t)options->asan.owned_application_size);
  }
  if (iree_status_is_ok(status) &&
      options->asan.owned_application_size > application_coverage_size) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU ASAN owned application size %" PRIu64
                              " exceeds application coverage size %" PRIu64,
                              (uint64_t)options->asan.owned_application_size,
                              (uint64_t)application_coverage_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_state_reserve_owned_application_window(
        &system->libhsa, options->asan.owned_application_size,
        options->asan.shadow_slab_size, &out_state->owned_application_base_ptr);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t owned_application_base =
        (uint64_t)(uintptr_t)out_state->owned_application_base_ptr;
    if (owned_application_base >
            UINT64_MAX - options->asan.owned_application_size ||
        owned_application_base + options->asan.owned_application_size >
            application_coverage_size) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU ASAN owned application window [0x%016" PRIx64 ", +%" PRIu64
          ") is outside application coverage [0x%016" PRIx64 ", +%" PRIu64 ")",
          owned_application_base,
          (uint64_t)options->asan.owned_application_size, (uint64_t)0,
          (uint64_t)application_coverage_size);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_state_initialize_shadow_map(
        options, system, physical_device_count, physical_devices,
        /*application_coverage_base=*/0, host_allocator,
        &out_state->shadow_map);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_application_range_allocate(
        out_state, (uint64_t)(uintptr_t)out_state->owned_application_base_ptr,
        options->asan.owned_application_size,
        &out_state->application_free_ranges);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_asan_application_range_free_list(
        out_state, out_state->application_free_ranges);
    if (out_state->owned_application_base_ptr) {
      status = iree_status_join(
          status,
          iree_hsa_amd_vmem_address_free(IREE_LIBHSA(&system->libhsa),
                                         out_state->owned_application_base_ptr,
                                         options->asan.owned_application_size));
    }
    iree_hal_amdgpu_shadow_map_deinitialize(&out_state->shadow_map);
    memset(out_state, 0, sizeof(*out_state));
    return status;
  }

  out_state->libhsa = &system->libhsa;
  out_state->owned_application_size = options->asan.owned_application_size;
  out_state->quarantine_limit = options->asan.quarantine_size;
  iree_slim_mutex_initialize(&out_state->application_mutex);
  iree_slim_mutex_initialize(&out_state->quarantine_mutex);
  out_state->is_enabled = true;
  return iree_ok_status();
}

void iree_hal_amdgpu_asan_state_deinitialize(
    iree_hal_amdgpu_asan_state_t* state) {
  if (!state || !state->is_enabled) return;
  iree_hal_amdgpu_asan_state_flush_quarantine(state);
  if (state->owned_application_base_ptr) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_address_free_raw(state->libhsa,
                                           state->owned_application_base_ptr,
                                           state->owned_application_size));
  }
  iree_hal_amdgpu_shadow_map_deinitialize(&state->shadow_map);
  iree_hal_amdgpu_asan_application_range_free_list(
      state, state->application_free_ranges);
  iree_slim_mutex_deinitialize(&state->quarantine_mutex);
  iree_slim_mutex_deinitialize(&state->application_mutex);
  memset(state, 0, sizeof(*state));
}

bool iree_hal_amdgpu_asan_state_is_enabled(
    const iree_hal_amdgpu_asan_state_t* state) {
  return state && state->is_enabled;
}

void iree_hal_amdgpu_asan_state_query_statistics(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_state_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  if (!iree_hal_amdgpu_asan_state_is_enabled(state)) return;

  iree_slim_mutex_lock(&state->quarantine_mutex);
  out_statistics->quarantine_size = state->quarantine_size;
  out_statistics->quarantine_eviction_count = state->quarantine_eviction_count;
  iree_slim_mutex_unlock(&state->quarantine_mutex);

  iree_hal_amdgpu_shadow_map_statistics_t shadow_statistics;
  iree_hal_amdgpu_shadow_map_query_statistics(&state->shadow_map,
                                              &shadow_statistics);
  out_statistics->shadow_mapped_slab_count =
      shadow_statistics.mapped_slab_count;
  out_statistics->shadow_committed_size = shadow_statistics.committed_size;
}

iree_hal_amdgpu_shadow_map_t* iree_hal_amdgpu_asan_state_shadow_map(
    iree_hal_amdgpu_asan_state_t* state) {
  return iree_hal_amdgpu_asan_state_is_enabled(state) ? &state->shadow_map
                                                      : NULL;
}

iree_status_t iree_hal_amdgpu_asan_state_map_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t application_length) {
  if (!iree_hal_amdgpu_asan_state_is_enabled(state)) return iree_ok_status();
  return iree_hal_amdgpu_shadow_map_map_range(
      &state->shadow_map, application_address, application_length,
      /*out_range=*/NULL);
}

iree_status_t iree_hal_amdgpu_asan_state_publish_imported_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t application_length) {
  if (!iree_hal_amdgpu_asan_state_is_enabled(state)) return iree_ok_status();
  if (application_length == 0) return iree_ok_status();
  if (IREE_UNLIKELY(application_length > UINT64_MAX - application_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ASAN imported range overflows: "
                            "base=0x%016" PRIx64 ", length=%" PRIu64,
                            application_address, (uint64_t)application_length);
  }

  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_shadow_map_map_range(
      &state->shadow_map, application_address, application_length, &range));
  if (range.shadow_length == 0) return iree_ok_status();

  const uint64_t application_end = application_address + application_length;
  const uint64_t shadow_granule_mask =
      ((uint64_t)1ull << state->shadow_map.shadow_scale_shift) - 1;
  const uint8_t end_offset = (uint8_t)(application_end & shadow_granule_mask);
  const uint64_t last_shadow_address =
      range.shadow_address + range.shadow_length - 1;
  const uint64_t end_shadow_address =
      end_offset == 0 ? last_shadow_address + 1 : last_shadow_address;

  if (range.shadow_address == end_shadow_address) {
    uint8_t shadow_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_read_shadow_byte(
        state, range.shadow_address, &shadow_value));
    if (shadow_value != IREE_HAL_AMDGPU_ASAN_SHADOW_ADDRESSABLE) {
      const uint8_t new_shadow_value = iree_hal_amdgpu_asan_max_shadow_value(
          state, shadow_value, end_offset);
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_write_shadow_bytes(
          state, range.shadow_address, /*length=*/1, new_shadow_value));
    }
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_write_shadow_bytes(
      state, range.shadow_address, end_shadow_address - range.shadow_address,
      IREE_HAL_AMDGPU_ASAN_SHADOW_ADDRESSABLE));
  if (end_offset != 0) {
    uint8_t shadow_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_read_shadow_byte(
        state, end_shadow_address, &shadow_value));
    if (shadow_value != IREE_HAL_AMDGPU_ASAN_SHADOW_ADDRESSABLE) {
      const uint8_t new_shadow_value = iree_hal_amdgpu_asan_max_shadow_value(
          state, shadow_value, end_offset);
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_write_shadow_bytes(
          state, end_shadow_address, /*length=*/1, new_shadow_value));
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_asan_state_publish_allocated_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t mapped_address,
    iree_device_size_t mapped_length, uint64_t accessible_address,
    iree_device_size_t accessible_length) {
  if (!iree_hal_amdgpu_asan_state_is_enabled(state)) return iree_ok_status();
  if (IREE_UNLIKELY(accessible_address < mapped_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN accessible address 0x%016" PRIx64
                            " precedes mapped range base 0x%016" PRIx64,
                            accessible_address, mapped_address);
  }
  const iree_device_size_t accessible_offset =
      accessible_address - mapped_address;
  if (IREE_UNLIKELY(accessible_offset > mapped_length ||
                    accessible_length > mapped_length - accessible_offset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN accessible range [0x%016" PRIx64
                            ", +%" PRIu64
                            ") exceeds mapped range "
                            "[0x%016" PRIx64 ", +%" PRIu64 ")",
                            accessible_address, (uint64_t)accessible_length,
                            mapped_address, (uint64_t)mapped_length);
  }
  if (mapped_length == 0) return iree_ok_status();

  const iree_device_size_t shadow_granule =
      (iree_device_size_t)1ull << state->shadow_map.shadow_scale_shift;
  if (IREE_UNLIKELY((mapped_address & (shadow_granule - 1)) != 0 ||
                    (mapped_length & (shadow_granule - 1)) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN mapped allocation range [0x%016" PRIx64
                            ", +%" PRIu64 ") must be shadow-granule aligned",
                            mapped_address, (uint64_t)mapped_length);
  }
  if (IREE_UNLIKELY((accessible_address & (shadow_granule - 1)) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN accessible address 0x%016" PRIx64
                            " must be shadow-granule aligned",
                            accessible_address);
  }

  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_shadow_map_map_range(
      &state->shadow_map, mapped_address, mapped_length, &range));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_write_shadow_bytes(
      state, range.shadow_address, range.shadow_length,
      IREE_HAL_AMDGPU_ASAN_SHADOW_HEAP_REDZONE));

  const iree_device_size_t accessible_shadow_offset =
      accessible_offset >> state->shadow_map.shadow_scale_shift;
  const uint64_t accessible_shadow_address =
      range.shadow_address + accessible_shadow_offset;
  const iree_device_size_t full_accessible_shadow_length =
      accessible_length >> state->shadow_map.shadow_scale_shift;
  const uint8_t partial_accessible_length =
      (uint8_t)(accessible_length & (shadow_granule - 1));
  if (full_accessible_shadow_length > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_write_shadow_bytes(
        state, accessible_shadow_address, full_accessible_shadow_length,
        IREE_HAL_AMDGPU_ASAN_SHADOW_ADDRESSABLE));
  }
  if (partial_accessible_length != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_write_shadow_bytes(
        state, accessible_shadow_address + full_accessible_shadow_length,
        /*length=*/1, partial_accessible_length));
  }
  return iree_ok_status();
}

void iree_hal_amdgpu_asan_state_publish_allocated_range_raw(
    iree_hal_amdgpu_asan_state_t* state, uint64_t mapped_address,
    iree_device_size_t mapped_length, uint64_t accessible_address,
    iree_device_size_t accessible_length) {
  if (!mapped_length) return;
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(state));
  IREE_ASSERT(accessible_address >= mapped_address);
  const iree_device_size_t accessible_offset =
      accessible_address - mapped_address;
  IREE_ASSERT(accessible_offset <= mapped_length);
  IREE_ASSERT(accessible_length <= mapped_length - accessible_offset);

  const iree_device_size_t shadow_granule =
      (iree_device_size_t)1ull << state->shadow_map.shadow_scale_shift;
  IREE_ASSERT((mapped_address & (shadow_granule - 1)) == 0);
  IREE_ASSERT((mapped_length & (shadow_granule - 1)) == 0);
  IREE_ASSERT((accessible_address & (shadow_granule - 1)) == 0);
  IREE_ASSERT(mapped_address >= state->shadow_map.application_window_base);

  const iree_device_size_t application_offset =
      mapped_address - state->shadow_map.application_window_base;
  IREE_ASSERT(application_offset <= state->shadow_map.application_window_size);
  IREE_ASSERT(mapped_length <=
              state->shadow_map.application_window_size - application_offset);

  const iree_device_size_t shadow_offset =
      application_offset >> state->shadow_map.shadow_scale_shift;
  const iree_device_size_t shadow_length =
      mapped_length >> state->shadow_map.shadow_scale_shift;
  const uint64_t shadow_address =
      (uint64_t)(uintptr_t)state->shadow_map.reservation_base_ptr +
      shadow_offset;
  iree_hal_amdgpu_asan_write_shadow_bytes_raw(
      state, shadow_address, shadow_length,
      IREE_HAL_AMDGPU_ASAN_SHADOW_HEAP_REDZONE);

  const iree_device_size_t accessible_shadow_offset =
      accessible_offset >> state->shadow_map.shadow_scale_shift;
  const uint64_t accessible_shadow_address =
      shadow_address + accessible_shadow_offset;
  const iree_device_size_t full_accessible_shadow_length =
      accessible_length >> state->shadow_map.shadow_scale_shift;
  const uint8_t partial_accessible_length =
      (uint8_t)(accessible_length & (shadow_granule - 1));
  if (full_accessible_shadow_length > 0) {
    iree_hal_amdgpu_asan_write_shadow_bytes_raw(
        state, accessible_shadow_address, full_accessible_shadow_length,
        IREE_HAL_AMDGPU_ASAN_SHADOW_ADDRESSABLE);
  }
  if (partial_accessible_length != 0) {
    iree_hal_amdgpu_asan_write_shadow_bytes_raw(
        state, accessible_shadow_address + full_accessible_shadow_length,
        /*length=*/1, partial_accessible_length);
  }
}

void iree_hal_amdgpu_asan_state_publish_released_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t mapped_length) {
  if (!mapped_length) return;
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(state));
  const iree_device_size_t shadow_granule =
      (iree_device_size_t)1ull << state->shadow_map.shadow_scale_shift;
  IREE_ASSERT((application_address & (shadow_granule - 1)) == 0);
  IREE_ASSERT((mapped_length & (shadow_granule - 1)) == 0);
  IREE_ASSERT(application_address >= state->shadow_map.application_window_base);

  const iree_device_size_t application_offset =
      application_address - state->shadow_map.application_window_base;
  IREE_ASSERT(application_offset <= state->shadow_map.application_window_size);
  IREE_ASSERT(mapped_length <=
              state->shadow_map.application_window_size - application_offset);
  const iree_device_size_t shadow_offset =
      application_offset >> state->shadow_map.shadow_scale_shift;
  const iree_device_size_t shadow_length =
      mapped_length >> state->shadow_map.shadow_scale_shift;
  const uint64_t shadow_address =
      (uint64_t)(uintptr_t)state->shadow_map.reservation_base_ptr +
      shadow_offset;
  iree_hal_amdgpu_asan_write_shadow_bytes_raw(
      state, shadow_address, shadow_length,
      IREE_HAL_AMDGPU_ASAN_SHADOW_HEAP_REDZONE);
}

void iree_hal_amdgpu_asan_state_quarantine_entry(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_quarantine_entry_t* entry,
    iree_device_size_t mapped_size,
    iree_hal_amdgpu_asan_quarantine_release_fn_t release_fn, void* user_data) {
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT_ARGUMENT(release_fn);
  if (!iree_hal_amdgpu_asan_state_is_enabled(state)) {
    release_fn(user_data);
    return;
  }
  if (state->quarantine_limit == 0) {
    iree_slim_mutex_lock(&state->quarantine_mutex);
    if (state->quarantine_eviction_count < UINT64_MAX) {
      ++state->quarantine_eviction_count;
    }
    iree_slim_mutex_unlock(&state->quarantine_mutex);
    release_fn(user_data);
    return;
  }

  entry->next = NULL;
  entry->mapped_size = mapped_size;
  entry->release_fn = release_fn;
  entry->user_data = user_data;

  iree_hal_amdgpu_asan_quarantine_entry_t* release_head = NULL;
  iree_hal_amdgpu_asan_quarantine_entry_t* release_tail = NULL;
  iree_slim_mutex_lock(&state->quarantine_mutex);
  if (state->quarantine_tail) {
    state->quarantine_tail->next = entry;
  } else {
    state->quarantine_head = entry;
  }
  state->quarantine_tail = entry;
  if (mapped_size > IREE_DEVICE_SIZE_MAX - state->quarantine_size) {
    state->quarantine_size = IREE_DEVICE_SIZE_MAX;
  } else {
    state->quarantine_size += mapped_size;
  }

  while (state->quarantine_size > state->quarantine_limit &&
         state->quarantine_head) {
    iree_hal_amdgpu_asan_quarantine_entry_t* oldest_entry =
        state->quarantine_head;
    state->quarantine_head = oldest_entry->next;
    if (!state->quarantine_head) state->quarantine_tail = NULL;
    oldest_entry->next = NULL;
    state->quarantine_size -=
        iree_min(state->quarantine_size, oldest_entry->mapped_size);
    if (state->quarantine_eviction_count < UINT64_MAX) {
      ++state->quarantine_eviction_count;
    }
    if (release_tail) {
      release_tail->next = oldest_entry;
    } else {
      release_head = oldest_entry;
    }
    release_tail = oldest_entry;
  }
  iree_slim_mutex_unlock(&state->quarantine_mutex);

  iree_hal_amdgpu_asan_quarantine_release_list(release_head);
}

iree_status_t iree_hal_amdgpu_asan_state_reserve_application_range(
    iree_hal_amdgpu_asan_state_t* state, iree_device_size_t length,
    iree_device_size_t alignment, uint64_t* out_application_address,
    iree_hal_amdgpu_asan_application_range_t** out_application_range) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_application_address);
  IREE_ASSERT_ARGUMENT(out_application_range);
  *out_application_address = 0;
  *out_application_range = NULL;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_asan_state_is_enabled(state))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN application address reservation requires ASAN state");
  }
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN application address reservation requires non-zero length");
  }
  if (IREE_UNLIKELY(!iree_device_size_is_valid_alignment(alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN application address reservation alignment %" PRIu64
        " is not a power of two",
        (uint64_t)alignment);
  }
  if (alignment == 0) alignment = 1;

  const uint64_t window_base =
      (uint64_t)(uintptr_t)state->owned_application_base_ptr;
  const uint64_t window_end = window_base + state->owned_application_size;
  uint64_t application_address = 0;
  iree_hal_amdgpu_asan_application_range_t* application_range = NULL;
  iree_hal_amdgpu_asan_application_range_t* suffix_range = NULL;

  iree_slim_mutex_lock(&state->application_mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_asan_application_range_t** current_range_ptr =
      &state->application_free_ranges;
  iree_hal_amdgpu_asan_application_range_t* current_range =
      state->application_free_ranges;
  while (current_range) {
    const uint64_t range_end = current_range->address + current_range->length;
    if (iree_hal_amdgpu_asan_align_address(current_range->address, alignment,
                                           &application_address) &&
        application_address <= range_end &&
        length <= range_end - application_address) {
      break;
    }
    current_range_ptr = &current_range->next;
    current_range = current_range->next;
  }

  if (!current_range) {
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU ASAN application window [0x%016" PRIx64 ", 0x%016" PRIx64
        ") has no space for length %" PRIu64 " with alignment %" PRIu64,
        window_base, window_end, (uint64_t)length, (uint64_t)alignment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_application_range_allocate(
        state, application_address, length, &application_range);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t allocation_end = application_address + length;
    const iree_device_size_t prefix_length =
        application_address - current_range->address;
    const uint64_t current_range_end =
        current_range->address + current_range->length;
    const iree_device_size_t suffix_length = current_range_end - allocation_end;
    if (prefix_length > 0 && suffix_length > 0) {
      status = iree_hal_amdgpu_asan_application_range_allocate(
          state, allocation_end, suffix_length, &suffix_range);
    }
    if (iree_status_is_ok(status)) {
      if (prefix_length > 0 && suffix_length > 0) {
        current_range->length = prefix_length;
        suffix_range->next = current_range->next;
        current_range->next = suffix_range;
        suffix_range = NULL;
      } else if (prefix_length > 0) {
        current_range->length = prefix_length;
      } else if (suffix_length > 0) {
        current_range->address = allocation_end;
        current_range->length = suffix_length;
      } else {
        *current_range_ptr = current_range->next;
        iree_hal_amdgpu_asan_application_range_free(state, current_range);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_application_address = application_address;
    *out_application_range = application_range;
    application_range = NULL;
  }
  iree_slim_mutex_unlock(&state->application_mutex);
  iree_hal_amdgpu_asan_application_range_free(state, application_range);
  iree_hal_amdgpu_asan_application_range_free(state, suffix_range);
  return status;
}

void iree_hal_amdgpu_asan_state_release_application_range(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* application_range) {
  if (!application_range) return;
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(state));

  iree_slim_mutex_lock(&state->application_mutex);
  iree_hal_amdgpu_asan_application_range_t** next_range_ptr =
      &state->application_free_ranges;
  iree_hal_amdgpu_asan_application_range_t* next_range =
      state->application_free_ranges;
  iree_hal_amdgpu_asan_application_range_t* previous_range = NULL;
  while (next_range && next_range->address < application_range->address) {
    previous_range = next_range;
    next_range_ptr = &next_range->next;
    next_range = next_range->next;
  }

  const uint64_t application_range_end =
      application_range->address + application_range->length;
  if (previous_range) {
    const uint64_t previous_range_end =
        previous_range->address + previous_range->length;
    IREE_ASSERT(previous_range_end <= application_range->address);
    if (previous_range_end == application_range->address) {
      previous_range->length += application_range->length;
      iree_hal_amdgpu_asan_application_range_free(state, application_range);
      application_range = previous_range;
    }
  }
  if (next_range) {
    IREE_ASSERT(application_range_end <= next_range->address);
    if (application_range_end == next_range->address) {
      if (application_range == previous_range) {
        application_range->length += next_range->length;
        application_range->next = next_range->next;
        iree_hal_amdgpu_asan_application_range_free(state, next_range);
      } else {
        next_range->address = application_range->address;
        next_range->length += application_range->length;
        iree_hal_amdgpu_asan_application_range_free(state, application_range);
        application_range = NULL;
      }
    }
  }
  if (application_range && application_range != previous_range) {
    application_range->next = next_range;
    *next_range_ptr = application_range;
  }
  iree_slim_mutex_unlock(&state->application_mutex);
}

void iree_hal_amdgpu_asan_state_populate_config(
    const iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_config);
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(state));

  const iree_hal_amdgpu_shadow_map_t* shadow_map = &state->shadow_map;
  *out_config = (iree_hal_amdgpu_asan_config_t){
      .record_length = sizeof(*out_config),
      .abi_version = IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0,
      .flags = IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED,
      .shadow_scale_shift = shadow_map->shadow_scale_shift,
      .shadow_base = shadow_map->shadow_base,
      .application_window_base = shadow_map->application_window_base,
      .application_window_size = shadow_map->application_window_size,
      .shadow_size = shadow_map->reservation_size,
      .shadow_slab_size = shadow_map->slab_size,
      .reserved = {0},
  };
}
