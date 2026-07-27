// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_provider.h"

void iree_hal_slab_provider_initialize(
    const iree_hal_slab_provider_vtable_t* vtable,
    iree_hal_slab_provider_t* provider) {
  iree_atomic_ref_count_init(&provider->ref_count);
  provider->vtable = vtable;
}

void iree_hal_slab_provider_retain(iree_hal_slab_provider_t* provider) {
  if (IREE_LIKELY(provider)) {
    iree_atomic_ref_count_inc(&provider->ref_count);
  }
}

void iree_hal_slab_provider_release(iree_hal_slab_provider_t* provider) {
  if (IREE_LIKELY(provider) &&
      iree_atomic_ref_count_dec(&provider->ref_count) == 1) {
    provider->vtable->destroy(provider);
  }
}

iree_status_t iree_hal_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  return provider->vtable->acquire_slab(provider, min_length, out_slab);
}

void iree_hal_slab_provider_release_slab(iree_hal_slab_provider_t* provider,
                                         const iree_hal_slab_t* slab) {
  provider->vtable->release_slab(provider, slab);
}

iree_status_t iree_hal_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (slab_offset > slab->length ||
      allocation_size > slab->length - slab_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab buffer offset %" PRIdsz
                            " with length %" PRIdsz
                            " is outside slab length %" PRIdsz,
                            slab_offset, allocation_size, slab->length);
  }
  iree_hal_buffer_params_canonicalize(&params);
  return provider->vtable->wrap_buffer(provider, slab, slab_offset,
                                       allocation_size, params,
                                       release_callback, out_buffer);
}

static bool iree_hal_slab_provider_asan_range_is_valid(
    const iree_hal_slab_t* slab, iree_device_size_t backing_offset,
    const iree_hal_asan_allocation_layout_t* layout) {
  if (layout->backing_offset_alignment == 0 ||
      !iree_device_size_is_power_of_two(layout->backing_offset_alignment)) {
    return false;
  }
  if (layout->backing_length_alignment == 0 ||
      !iree_device_size_is_power_of_two(layout->backing_length_alignment)) {
    return false;
  }
  if (!iree_device_size_has_alignment(backing_offset,
                                      layout->backing_offset_alignment)) {
    return false;
  }
  if (layout->backing_length == 0 ||
      !iree_device_size_has_alignment(layout->backing_length,
                                      layout->backing_length_alignment)) {
    return false;
  }
  if (layout->user_offset > layout->backing_length ||
      layout->user_length > layout->backing_length - layout->user_offset) {
    return false;
  }
  if (backing_offset > slab->length ||
      layout->backing_length > slab->length - backing_offset) {
    return false;
  }
  return true;
}

static bool iree_hal_slab_provider_asan_advice_flags_are_valid(
    iree_hal_asan_range_advice_flags_t advice_flags) {
  switch (advice_flags) {
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED:
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED:
      return true;
    default:
      return false;
  }
}

iree_status_t iree_hal_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* provider,
    const iree_hal_asan_pool_options_t* options) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(options);
  IREE_RETURN_IF_ERROR(iree_hal_asan_pool_options_validate(options));
  if (!iree_hal_asan_pool_options_is_enabled(options)) {
    return iree_ok_status();
  }
  return provider->vtable->validate_asan_options(provider, options);
}

void iree_hal_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT(iree_hal_slab_provider_asan_advice_flags_are_valid(advice_flags),
              "unsupported ASAN range advice flags 0x%x", advice_flags);
  IREE_ASSERT(
      iree_hal_slab_provider_asan_range_is_valid(slab, backing_offset, layout),
      "invalid ASAN backing range advice");
  provider->vtable->advise_asan_range(provider, slab, backing_offset,
                                      advice_flags, layout);
}

void iree_hal_slab_provider_prefault(iree_hal_slab_provider_t* provider,
                                     iree_hal_slab_t* slab) {
  provider->vtable->prefault(provider, slab);
}

void iree_hal_slab_provider_trim(iree_hal_slab_provider_t* provider,
                                 iree_hal_slab_provider_trim_flags_t flags) {
  provider->vtable->trim(provider, flags);
}

void iree_hal_slab_provider_query_stats(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  provider->vtable->query_stats(provider, visited, out_stats);
}

void iree_hal_slab_provider_query_properties(
    const iree_hal_slab_provider_t* provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  provider->vtable->query_properties(provider, out_memory_type,
                                     out_supported_usage);
}

bool iree_hal_slab_provider_visited(
    iree_hal_slab_provider_visited_set_t* visited,
    const iree_hal_slab_provider_t* provider) {
  for (iree_host_size_t i = 0; i < visited->count; ++i) {
    if (visited->providers[i] == provider) {
      return true;
    }
  }
  if (visited->count >= IREE_HAL_SLAB_PROVIDER_MAX_VISITED) {
    return true;
  }
  visited->providers[visited->count++] = provider;
  return false;
}
