// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/shadow_map.h"

#include "iree/base/internal/math.h"

static iree_status_t iree_hal_amdgpu_shadow_map_validate_params(
    const iree_hal_amdgpu_shadow_map_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);

  if (IREE_UNLIKELY(iree_allocator_is_null(params->host_allocator))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map requires a host allocator");
  }
  if (IREE_UNLIKELY(params->shadow_scale_shift >= 63)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU shadow map scale shift %u is too large",
                            params->shadow_scale_shift);
  }
  if (IREE_UNLIKELY(params->shadow_size == 0 ||
                    !iree_device_size_is_power_of_two(params->shadow_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map size %" PRIu64
                            " must be a non-zero power of two",
                            (uint64_t)params->shadow_size);
  }
  if (IREE_UNLIKELY(params->slab_size == 0 ||
                    !iree_device_size_is_power_of_two(params->slab_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map slab size %" PRIu64
                            " must be a non-zero power of two",
                            (uint64_t)params->slab_size);
  }
  if (IREE_UNLIKELY(params->slab_size > params->shadow_size ||
                    params->shadow_size % params->slab_size != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map slab size %" PRIu64
                            " must divide shadow size %" PRIu64,
                            (uint64_t)params->slab_size,
                            (uint64_t)params->shadow_size);
  }
  switch (params->mapping_mode) {
    case IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_SPARSE:
    case IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_PREMAPPED:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU shadow map mapping mode %u is invalid",
                              (uint32_t)params->mapping_mode);
  }
  if (IREE_UNLIKELY(params->shadow_size >
                    (IREE_DEVICE_SIZE_MAX >> params->shadow_scale_shift))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU shadow map application window size overflows: "
        "shadow_size=%" PRIu64 ", scale_shift=%u",
        (uint64_t)params->shadow_size, params->shadow_scale_shift);
  }

  const iree_device_size_t application_window_size =
      params->shadow_size << params->shadow_scale_shift;
  if (IREE_UNLIKELY(application_window_size >
                    UINT64_MAX - params->application_window_base)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU shadow map application window overflows: base=0x%016" PRIx64
        ", size=%" PRIu64,
        params->application_window_base, (uint64_t)application_window_size);
  }
  const uint64_t shadow_granule = (uint64_t)1ull << params->shadow_scale_shift;
  if (IREE_UNLIKELY(params->application_window_base & (shadow_granule - 1))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU shadow map application window base 0x%016" PRIx64
        " must be aligned to shadow granule %" PRIu64,
        params->application_window_base, shadow_granule);
  }

  if (IREE_UNLIKELY(params->access_desc_count > 0 && !params->access_descs)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU shadow map access descriptor count is non-zero but descriptors "
        "are NULL");
  }
  if (IREE_UNLIKELY(!params->mapper.reserve ||
                    !params->mapper.release_reservation ||
                    !params->mapper.map_slab || !params->mapper.unmap_slab)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map mapper is incomplete");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_shadow_map_validate_hsa_params(
    const iree_hal_amdgpu_shadow_map_hsa_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);

  if (IREE_UNLIKELY(!params->libhsa)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map requires libhsa");
  }
  if (IREE_UNLIKELY(!params->memory_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU shadow map requires a memory pool");
  }
  return iree_ok_status();
}

static iree_hal_amdgpu_shadow_map_slab_t*
iree_hal_amdgpu_shadow_map_find_slab_locked(iree_hal_amdgpu_shadow_map_t* map,
                                            uint64_t slab_index) {
  for (iree_host_size_t i = 0; i < map->slab_count; ++i) {
    if (map->slabs[i].index == slab_index) return &map->slabs[i];
  }
  return NULL;
}

static iree_status_t iree_hal_amdgpu_shadow_map_grow_slabs_locked(
    iree_hal_amdgpu_shadow_map_t* map, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= map->slab_capacity) return iree_ok_status();
  return iree_allocator_grow_array(map->host_allocator, minimum_capacity,
                                   sizeof(map->slabs[0]), &map->slab_capacity,
                                   (void**)&map->slabs);
}

static iree_status_t iree_hal_amdgpu_shadow_map_hsa_reserve(
    iree_hal_amdgpu_shadow_map_t* map, iree_device_size_t reservation_size,
    iree_device_size_t alignment, IREE_AMDGPU_DEVICE_PTR void** out_base_ptr) {
  *out_base_ptr = NULL;
  return iree_hsa_amd_vmem_address_reserve_align(
      IREE_LIBHSA(map->hsa.libhsa), out_base_ptr, reservation_size,
      /*address=*/0, alignment, HSA_AMD_VMEM_ADDRESS_NO_REGISTER);
}

static void iree_hal_amdgpu_shadow_map_hsa_release_reservation(
    iree_hal_amdgpu_shadow_map_t* map, IREE_AMDGPU_DEVICE_PTR void* base_ptr,
    iree_device_size_t reservation_size) {
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_vmem_address_free_raw(
      map->hsa.libhsa, base_ptr, reservation_size));
}

static uint32_t iree_hal_amdgpu_shadow_map_fill_pattern(uint8_t value) {
  return ((uint32_t)value << 24) | ((uint32_t)value << 16) |
         ((uint32_t)value << 8) | (uint32_t)value;
}

static bool iree_hal_amdgpu_shadow_map_hsa_has_alias_slabs(
    const iree_hal_amdgpu_shadow_map_t* map) {
  return map->hsa.alias_allocation_handle.handle != 0;
}

static iree_status_t iree_hal_amdgpu_shadow_map_hsa_map_alias_slab(
    iree_hal_amdgpu_shadow_map_t* map, IREE_AMDGPU_DEVICE_PTR void* target_ptr,
    iree_device_size_t slab_size) {
  IREE_RETURN_IF_ERROR(iree_hsa_amd_vmem_map(
      IREE_LIBHSA(map->hsa.libhsa), target_ptr, slab_size, /*offset=*/0,
      map->hsa.alias_allocation_handle, /*flags=*/0));
  return iree_hsa_amd_vmem_set_access(IREE_LIBHSA(map->hsa.libhsa), target_ptr,
                                      slab_size, map->access_descs,
                                      map->access_desc_count);
}

static void iree_hal_amdgpu_shadow_map_hsa_unmap_alias_slab(
    iree_hal_amdgpu_shadow_map_t* map, IREE_AMDGPU_DEVICE_PTR void* target_ptr,
    iree_device_size_t slab_size) {
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_amd_vmem_unmap_raw(map->hsa.libhsa, target_ptr, slab_size));
}

static iree_status_t iree_hal_amdgpu_shadow_map_hsa_premap_alias_slabs(
    iree_hal_amdgpu_shadow_map_t* map) {
  hsa_amd_vmem_alloc_handle_t allocation_handle = {0};
  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(map->hsa.libhsa), map->hsa.memory_pool, map->slab_size,
      map->hsa.hsa_memory_type, /*flags=*/0, &allocation_handle);
  if (!iree_status_is_ok(status)) return status;

  map->hsa.alias_allocation_handle = allocation_handle;
  const uint64_t slab_count = map->reservation_size / map->slab_size;
  uint64_t mapped_count = 0;
  for (; iree_status_is_ok(status) && mapped_count < slab_count;
       ++mapped_count) {
    IREE_AMDGPU_DEVICE_PTR void* target_ptr =
        (uint8_t*)map->reservation_base_ptr + mapped_count * map->slab_size;
    status = iree_hsa_amd_vmem_map(
        IREE_LIBHSA(map->hsa.libhsa), target_ptr, map->slab_size,
        /*offset=*/0, allocation_handle, /*flags=*/0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_set_access(
        IREE_LIBHSA(map->hsa.libhsa), map->reservation_base_ptr,
        map->reservation_size, map->access_descs, map->access_desc_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_fill(
        IREE_LIBHSA(map->hsa.libhsa), map->reservation_base_ptr,
        iree_hal_amdgpu_shadow_map_fill_pattern(map->initial_slab_value),
        map->slab_size / sizeof(uint32_t));
  }

  if (!iree_status_is_ok(status)) {
    while (mapped_count > 0) {
      --mapped_count;
      IREE_AMDGPU_DEVICE_PTR void* target_ptr =
          (uint8_t*)map->reservation_base_ptr + mapped_count * map->slab_size;
      status = iree_status_join(
          status, iree_hsa_amd_vmem_unmap(IREE_LIBHSA(map->hsa.libhsa),
                                          target_ptr, map->slab_size));
    }
    status = iree_status_join(
        status, iree_hsa_amd_vmem_handle_release(IREE_LIBHSA(map->hsa.libhsa),
                                                 allocation_handle));
    memset(&map->hsa.alias_allocation_handle, 0,
           sizeof(map->hsa.alias_allocation_handle));
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_shadow_map_hsa_map_slab(
    iree_hal_amdgpu_shadow_map_t* map, IREE_AMDGPU_DEVICE_PTR void* target_ptr,
    iree_device_size_t slab_size, iree_host_size_t access_desc_count,
    const hsa_amd_memory_access_desc_t* access_descs,
    hsa_amd_vmem_alloc_handle_t* out_allocation_handle) {
  memset(out_allocation_handle, 0, sizeof(*out_allocation_handle));

  hsa_amd_vmem_alloc_handle_t allocation_handle = {0};
  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(map->hsa.libhsa), map->hsa.memory_pool, slab_size,
      map->hsa.hsa_memory_type, /*flags=*/0, &allocation_handle);

  bool alias_unmapped = false;
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_shadow_map_hsa_has_alias_slabs(map)) {
    status = iree_hsa_amd_vmem_unmap(IREE_LIBHSA(map->hsa.libhsa), target_ptr,
                                     slab_size);
    alias_unmapped = iree_status_is_ok(status);
  }
  bool mapped = false;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_map(IREE_LIBHSA(map->hsa.libhsa), target_ptr,
                                   slab_size, /*offset=*/0, allocation_handle,
                                   /*flags=*/0);
    mapped = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_set_access(IREE_LIBHSA(map->hsa.libhsa),
                                          target_ptr, slab_size, access_descs,
                                          access_desc_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_fill(
        IREE_LIBHSA(map->hsa.libhsa), target_ptr,
        iree_hal_amdgpu_shadow_map_fill_pattern(map->initial_slab_value),
        slab_size / sizeof(uint32_t));
  }

  if (iree_status_is_ok(status)) {
    *out_allocation_handle = allocation_handle;
  } else {
    if (mapped) {
      status = iree_status_join(
          status, iree_hsa_amd_vmem_unmap(IREE_LIBHSA(map->hsa.libhsa),
                                          target_ptr, slab_size));
    }
    if (allocation_handle.handle) {
      status = iree_status_join(
          status, iree_hsa_amd_vmem_handle_release(IREE_LIBHSA(map->hsa.libhsa),
                                                   allocation_handle));
    }
    if (alias_unmapped) {
      status = iree_status_join(
          status, iree_hal_amdgpu_shadow_map_hsa_map_alias_slab(map, target_ptr,
                                                                slab_size));
    }
  }
  return status;
}

static void iree_hal_amdgpu_shadow_map_hsa_unmap_slab(
    iree_hal_amdgpu_shadow_map_t* map, IREE_AMDGPU_DEVICE_PTR void* target_ptr,
    iree_device_size_t slab_size,
    hsa_amd_vmem_alloc_handle_t allocation_handle) {
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_amd_vmem_unmap_raw(map->hsa.libhsa, target_ptr, slab_size));
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_amd_vmem_handle_release_raw(map->hsa.libhsa, allocation_handle));
}

static void iree_hal_amdgpu_shadow_map_unmap_slab(
    iree_hal_amdgpu_shadow_map_t* map,
    iree_hal_amdgpu_shadow_map_slab_t* slab) {
  if (!slab->base_ptr) return;
  map->mapper.unmap_slab(map, slab->base_ptr, map->slab_size,
                         slab->allocation_handle);
  memset(slab, 0, sizeof(*slab));
}

static bool iree_hal_amdgpu_shadow_map_has_precise_slab(
    const iree_hal_amdgpu_shadow_map_t* map, uint64_t slab_index) {
  for (iree_host_size_t i = 0; i < map->slab_count; ++i) {
    if (map->slabs[i].index == slab_index) return true;
  }
  return false;
}

static void iree_hal_amdgpu_shadow_map_hsa_unmap_alias_slabs(
    iree_hal_amdgpu_shadow_map_t* map) {
  if (!iree_hal_amdgpu_shadow_map_hsa_has_alias_slabs(map)) return;
  const uint64_t slab_count = map->reservation_size / map->slab_size;
  for (uint64_t i = 0; i < slab_count; ++i) {
    if (iree_hal_amdgpu_shadow_map_has_precise_slab(map, i)) continue;
    IREE_AMDGPU_DEVICE_PTR void* target_ptr =
        (uint8_t*)map->reservation_base_ptr + i * map->slab_size;
    iree_hal_amdgpu_shadow_map_hsa_unmap_alias_slab(map, target_ptr,
                                                    map->slab_size);
  }
  iree_hal_amdgpu_hsa_cleanup_assert_success(
      iree_hsa_amd_vmem_handle_release_raw(map->hsa.libhsa,
                                           map->hsa.alias_allocation_handle));
  memset(&map->hsa.alias_allocation_handle, 0,
         sizeof(map->hsa.alias_allocation_handle));
}

static iree_status_t iree_hal_amdgpu_shadow_map_map_slab_locked(
    iree_hal_amdgpu_shadow_map_t* map, uint64_t slab_index,
    iree_hal_amdgpu_shadow_map_slab_t** out_slab) {
  if (out_slab) *out_slab = NULL;
  iree_hal_amdgpu_shadow_map_slab_t* existing_slab =
      iree_hal_amdgpu_shadow_map_find_slab_locked(map, slab_index);
  if (existing_slab) {
    if (out_slab) *out_slab = existing_slab;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_shadow_map_grow_slabs_locked(map, map->slab_count + 1));

  iree_hal_amdgpu_shadow_map_slab_t slab = {
      .index = slab_index,
      .base_ptr =
          (uint8_t*)map->reservation_base_ptr + slab_index * map->slab_size,
      .allocation_handle = {0},
  };
  IREE_RETURN_IF_ERROR(map->mapper.map_slab(
      map, slab.base_ptr, map->slab_size, map->access_desc_count,
      map->access_descs, &slab.allocation_handle));

  map->slabs[map->slab_count] = slab;
  if (out_slab) *out_slab = &map->slabs[map->slab_count];
  ++map->slab_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_shadow_map_initialize_internal(
    const iree_hal_amdgpu_shadow_map_params_t* params,
    const iree_hal_amdgpu_shadow_map_hsa_params_t* hsa_params,
    iree_hal_amdgpu_shadow_map_t* out_map) {
  IREE_ASSERT_ARGUMENT(out_map);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_shadow_map_validate_params(params));
  if (params->mapping_mode ==
          IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_PREMAPPED &&
      !hsa_params) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU premapped shadow maps require the HSA VMM mapper");
  }

  memset(out_map, 0, sizeof(*out_map));
  out_map->host_allocator = params->host_allocator;
  out_map->shadow_scale_shift = params->shadow_scale_shift;
  out_map->application_window_base = params->application_window_base;
  out_map->application_window_size = params->shadow_size
                                     << params->shadow_scale_shift;
  out_map->reservation_size = params->shadow_size;
  out_map->slab_size = params->slab_size;
  out_map->mapping_mode = params->mapping_mode;
  out_map->initial_slab_value = params->initial_slab_value;
  out_map->mapper = params->mapper;
  if (hsa_params) {
    out_map->hsa.libhsa = hsa_params->libhsa;
    out_map->hsa.memory_pool = hsa_params->memory_pool;
    out_map->hsa.memory_type = hsa_params->memory_type;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_translate_memory_type(
        hsa_params->memory_type, &out_map->hsa.hsa_memory_type));
  }
  iree_slim_mutex_initialize(&out_map->mutex);

  iree_status_t status = iree_ok_status();
  if (params->access_desc_count > 0) {
    status = iree_allocator_malloc_array(
        params->host_allocator, params->access_desc_count,
        sizeof(params->access_descs[0]), (void**)&out_map->access_descs);
  }
  if (iree_status_is_ok(status)) {
    out_map->access_desc_count = params->access_desc_count;
    if (out_map->access_desc_count > 0) {
      memcpy(out_map->access_descs, params->access_descs,
             out_map->access_desc_count * sizeof(out_map->access_descs[0]));
    }
  }
  if (iree_status_is_ok(status)) {
    status = out_map->mapper.reserve(out_map, out_map->reservation_size,
                                     out_map->slab_size,
                                     &out_map->reservation_base_ptr);
  }
  if (iree_status_is_ok(status) &&
      out_map->mapping_mode ==
          IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_PREMAPPED) {
    status = iree_hal_amdgpu_shadow_map_hsa_premap_alias_slabs(out_map);
  }
  if (iree_status_is_ok(status)) {
    out_map->shadow_base =
        (uint64_t)(uintptr_t)out_map->reservation_base_ptr -
        (out_map->application_window_base >> out_map->shadow_scale_shift);
    out_map->initialized = true;
  } else {
    if (out_map->reservation_base_ptr) {
      out_map->mapper.release_reservation(
          out_map, out_map->reservation_base_ptr, out_map->reservation_size);
    }
    iree_allocator_free(out_map->host_allocator, out_map->access_descs);
    iree_slim_mutex_deinitialize(&out_map->mutex);
    memset(out_map, 0, sizeof(*out_map));
  }
  return status;
}

iree_status_t iree_hal_amdgpu_shadow_map_initialize(
    const iree_hal_amdgpu_shadow_map_params_t* params,
    iree_hal_amdgpu_shadow_map_t* out_map) {
  return iree_hal_amdgpu_shadow_map_initialize_internal(params,
                                                        /*hsa_params=*/NULL,
                                                        out_map);
}

iree_status_t iree_hal_amdgpu_shadow_map_initialize_hsa(
    const iree_hal_amdgpu_shadow_map_hsa_params_t* params,
    iree_hal_amdgpu_shadow_map_t* out_map) {
  IREE_ASSERT_ARGUMENT(out_map);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_shadow_map_validate_hsa_params(params));

  iree_device_size_t allocation_granule = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_query_alloc_granule(
      params->libhsa, params->memory_pool, &allocation_granule));

  iree_device_size_t slab_size = params->requested_slab_size;
  if (slab_size < allocation_granule) slab_size = allocation_granule;
  iree_device_size_t aligned_slab_size = 0;
  if (!iree_device_size_checked_align(slab_size, allocation_granule,
                                      &aligned_slab_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU shadow map slab size %" PRIu64
                            " overflows HSA VMM allocation granule %" PRIu64,
                            (uint64_t)params->requested_slab_size,
                            (uint64_t)allocation_granule);
  }
  slab_size = iree_device_size_next_power_of_two(aligned_slab_size);
  if (IREE_UNLIKELY(!iree_device_size_is_power_of_two(slab_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU shadow map slab size %" PRIu64
                            " cannot be rounded to a power of two",
                            (uint64_t)params->requested_slab_size);
  }

  iree_hal_amdgpu_shadow_map_params_t map_params = {
      .host_allocator = params->host_allocator,
      .shadow_scale_shift = params->shadow_scale_shift,
      .application_window_base = params->application_window_base,
      .shadow_size = params->shadow_size,
      .slab_size = slab_size,
      .mapping_mode = params->mapping_mode,
      .initial_slab_value = params->initial_slab_value,
      .access_desc_count = params->access_desc_count,
      .access_descs = params->access_descs,
      .mapper =
          {
              .user_data = NULL,
              .reserve = iree_hal_amdgpu_shadow_map_hsa_reserve,
              .release_reservation =
                  iree_hal_amdgpu_shadow_map_hsa_release_reservation,
              .map_slab = iree_hal_amdgpu_shadow_map_hsa_map_slab,
              .unmap_slab = iree_hal_amdgpu_shadow_map_hsa_unmap_slab,
          },
  };
  return iree_hal_amdgpu_shadow_map_initialize_internal(&map_params, params,
                                                        out_map);
}

void iree_hal_amdgpu_shadow_map_deinitialize(
    iree_hal_amdgpu_shadow_map_t* map) {
  if (!map || !map->initialized) return;

  iree_slim_mutex_lock(&map->mutex);
  iree_hal_amdgpu_shadow_map_hsa_unmap_alias_slabs(map);
  for (iree_host_size_t i = 0; i < map->slab_count; ++i) {
    iree_hal_amdgpu_shadow_map_unmap_slab(map, &map->slabs[i]);
  }
  iree_slim_mutex_unlock(&map->mutex);

  map->mapper.release_reservation(map, map->reservation_base_ptr,
                                  map->reservation_size);
  iree_allocator_free(map->host_allocator, map->slabs);
  iree_allocator_free(map->host_allocator, map->access_descs);
  iree_slim_mutex_deinitialize(&map->mutex);
  memset(map, 0, sizeof(*map));
}

void iree_hal_amdgpu_shadow_map_query_statistics(
    iree_hal_amdgpu_shadow_map_t* map,
    iree_hal_amdgpu_shadow_map_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  if (!map || !map->initialized) return;

  iree_slim_mutex_lock(&map->mutex);
  out_statistics->mapped_slab_count = map->slab_count;
  if (!iree_device_size_checked_mul((iree_device_size_t)map->slab_count,
                                    map->slab_size,
                                    &out_statistics->committed_size)) {
    out_statistics->committed_size = IREE_DEVICE_SIZE_MAX;
  }
  if (iree_hal_amdgpu_shadow_map_hsa_has_alias_slabs(map) &&
      out_statistics->committed_size != IREE_DEVICE_SIZE_MAX &&
      !iree_device_size_checked_add(out_statistics->committed_size,
                                    map->slab_size,
                                    &out_statistics->committed_size)) {
    out_statistics->committed_size = IREE_DEVICE_SIZE_MAX;
  }
  iree_slim_mutex_unlock(&map->mutex);
}

iree_status_t iree_hal_amdgpu_shadow_map_calculate_range(
    const iree_hal_amdgpu_shadow_map_t* map, uint64_t application_address,
    iree_device_size_t application_length,
    iree_hal_amdgpu_shadow_map_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(out_range);
  memset(out_range, 0, sizeof(*out_range));
  out_range->application_address = application_address;
  out_range->application_length = application_length;
  if (IREE_UNLIKELY(!map->initialized)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU shadow map is not initialized");
  }
  if (application_length == 0) return iree_ok_status();

  if (IREE_UNLIKELY(application_length > UINT64_MAX - application_address)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU shadow map application range overflows: base=0x%016" PRIx64
        ", length=%" PRIu64,
        application_address, (uint64_t)application_length);
  }
  const uint64_t application_end = application_address + application_length;
  const uint64_t window_end =
      map->application_window_base + map->application_window_size;
  if (IREE_UNLIKELY(application_address < map->application_window_base ||
                    application_end > window_end)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU shadow map application range [0x%016" PRIx64 ", 0x%016" PRIx64
        ") is outside window [0x%016" PRIx64 ", 0x%016" PRIx64 ")",
        application_address, application_end, map->application_window_base,
        window_end);
  }

  const uint64_t application_offset =
      application_address - map->application_window_base;
  const uint64_t last_application_offset =
      application_offset + application_length - 1;
  const iree_device_size_t shadow_offset =
      application_offset >> map->shadow_scale_shift;
  const iree_device_size_t shadow_end_offset =
      (last_application_offset >> map->shadow_scale_shift) + 1;
  out_range->shadow_offset = shadow_offset;
  out_range->shadow_length = shadow_end_offset - shadow_offset;
  out_range->shadow_address =
      (uint64_t)(uintptr_t)map->reservation_base_ptr + shadow_offset;
  out_range->first_slab_index = shadow_offset / map->slab_size;
  out_range->slab_count =
      (iree_host_size_t)((iree_device_align(shadow_end_offset, map->slab_size) /
                          map->slab_size) -
                         out_range->first_slab_index);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_shadow_map_map_slab(
    iree_hal_amdgpu_shadow_map_t* map, uint64_t slab_index) {
  IREE_ASSERT_ARGUMENT(map);
  if (IREE_UNLIKELY(!map->initialized)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU shadow map is not initialized");
  }
  if (IREE_UNLIKELY(slab_index >= map->reservation_size / map->slab_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU shadow map slab index %" PRIu64
                            " exceeds reservation slab count %" PRIu64,
                            slab_index,
                            (uint64_t)(map->reservation_size / map->slab_size));
  }

  iree_slim_mutex_lock(&map->mutex);
  iree_status_t status =
      iree_hal_amdgpu_shadow_map_map_slab_locked(map, slab_index, NULL);
  iree_slim_mutex_unlock(&map->mutex);
  return status;
}

iree_status_t iree_hal_amdgpu_shadow_map_map_range(
    iree_hal_amdgpu_shadow_map_t* map, uint64_t application_address,
    iree_device_size_t application_length,
    iree_hal_amdgpu_shadow_map_range_t* out_range) {
  IREE_ASSERT_ARGUMENT(map);
  iree_hal_amdgpu_shadow_map_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_shadow_map_calculate_range(
      map, application_address, application_length, &range));
  if (range.slab_count == 0) {
    if (out_range) *out_range = range;
    return iree_ok_status();
  }

  iree_slim_mutex_lock(&map->mutex);
  const iree_host_size_t initial_slab_count = map->slab_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range.slab_count; ++i) {
    status = iree_hal_amdgpu_shadow_map_map_slab_locked(
        map, range.first_slab_index + i, NULL);
    if (!iree_status_is_ok(status)) break;
  }
  if (!iree_status_is_ok(status)) {
    while (map->slab_count > initial_slab_count) {
      --map->slab_count;
      iree_hal_amdgpu_shadow_map_unmap_slab(map, &map->slabs[map->slab_count]);
    }
  }
  iree_slim_mutex_unlock(&map->mutex);
  if (iree_status_is_ok(status) && out_range) *out_range = range;
  return status;
}
