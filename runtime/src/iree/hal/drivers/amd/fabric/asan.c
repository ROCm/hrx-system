// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/asan.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/status.h"

#define IREE_HAL_AMD_ASAN_POISON 0xFAu

struct iree_hal_amd_fabric_asan_shadow_slab_t {
  // Logical slab index within the complete shadow reservation.
  uint64_t index;
  // Precise physical backing and its persistent CPU write view.
  iree_hal_amd_fabric_memory_t memory;
};

void iree_hal_amd_fabric_asan_options_initialize(
    iree_hal_amd_fabric_asan_options_t* out_options) {
  *out_options = (iree_hal_amd_fabric_asan_options_t){
      .shadow_mode = IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_SPARSE,
      .shadow_scale_shift = 3,
      .shadow_size = UINT64_C(32) << 40,
      .shadow_slab_size = UINT64_C(128) << 20,
      .owned_application_size = UINT64_C(8) << 40,
      .redzone_size = 256,
      .quarantine_size = UINT64_C(256) << 20,
  };
}

static iree_status_t iree_hal_amd_fabric_asan_validate_options(
    const iree_hal_amd_fabric_asan_options_t* options) {
  if (options->shadow_mode != IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_SPARSE &&
      options->shadow_mode != IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_PREMAPPED) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ASAN shadow mapping mode");
  }
  if (options->shadow_scale_shift > 7 ||
      !iree_device_size_is_power_of_two(options->shadow_size) ||
      !iree_device_size_is_power_of_two(options->shadow_slab_size) ||
      !iree_device_size_is_power_of_two(options->owned_application_size) ||
      options->shadow_slab_size > options->shadow_size ||
      options->shadow_size > (UINT64_MAX >> options->shadow_scale_shift) ||
      options->owned_application_size >
          (options->shadow_size << options->shadow_scale_shift)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid ASAN shadow/application geometry");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_fabric_asan_prepare_memory(
    iree_hal_amd_fabric_asan_t* state, amdf_memory_scope_t* scope,
    const amdf_memory_reservation_capabilities_t* capabilities) {
  state->access = (amdf_memory_device_access_t){
      .device = state->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        capabilities->required_access,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                       AMDF_MEMORY_FLAG_MAPPING_SOURCE,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      state->api->memory_scope_query_info(scope, &info), "ASAN shadow scope"));
  iree_status_t status = iree_ok_status();
  bool found = false;
  for (uint32_t i = 0;
       i < info.memory_profile_count && !found && iree_status_is_ok(status);
       ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(access),
    };
    const amdf_status_t native_status =
        state->api->memory_scope_query_device_profile(
            scope, i, 1, &state->access, &profile, &access);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(native_status, "ASAN shadow profile");
    const amdf_memory_flags_t flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | capabilities->required_source_flags;
    if (iree_status_is_ok(status) &&
        iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        iree_all_bits_set(profile.supported_flags, flags) &&
        iree_all_bits_set(
            profile.host_mapping.supported_access,
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE) &&
        state->config.shadow_slab_size %
                profile.allocation.byte_length_granularity ==
            0) {
      state->memory = (iree_hal_amd_fabric_memory_options_t){
          .scope = scope,
          .profile = profile,
          .access_count = 1,
          .accesses = &state->access,
          .required_flags = flags,
          .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      found = true;
    }
  }
  if (iree_status_is_ok(status) && !found) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected ASAN scope has no host-mapped shadow backing profile");
  }
  return status;
}

static iree_status_t iree_hal_amd_fabric_asan_allocate_shadow(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_memory_t* out_memory) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      state->api, &state->memory, state->config.shadow_slab_size,
      state->config.shadow_slab_size, state->event_sink, out_memory));
  memset(out_memory->contents.data, IREE_HAL_AMD_ASAN_POISON,
         state->config.shadow_slab_size);
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      state->api->host_mapping_cache_control(out_memory->mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             state->config.shadow_slab_size),
      "ASAN initial shadow publication");
}

static iree_status_t iree_hal_amd_fabric_asan_premap_shadow(
    iree_hal_amd_fabric_asan_t* state) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_asan_allocate_shadow(state, &state->alias_memory));
  // Bounded host descriptors, with one native call for each batch of aliases.
  amdf_memory_mapping_update_t updates[128];
  const uint64_t slab_size = state->config.shadow_slab_size;
  const uint64_t slab_count = state->config.shadow_size / slab_size;
  iree_status_t status = iree_ok_status();
  for (uint64_t first = 0; first < slab_count && iree_status_is_ok(status);
       first += IREE_ARRAYSIZE(updates)) {
    const uint32_t count =
        (uint32_t)iree_min(slab_count - first, IREE_ARRAYSIZE(updates));
    for (uint32_t i = 0; i < count; ++i) {
      updates[i] = (amdf_memory_mapping_update_t){
          .target_byte_offset = (first + i) * slab_size,
          .byte_length = slab_size,
          .memory = state->alias_memory.handle,
          .access = state->access.requirements.access,
      };
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        state->api->memory_reservation_update(state->shadow_reservation, 0,
                                              count, updates),
        "ASAN shared poison aliases");
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_asan_initialize(
    const amdf_api_t* api, amdf_instance_t* instance, amdf_device_t* device,
    uint32_t address_domain_ordinal, amdf_memory_scope_t* shadow_scope,
    const iree_hal_amd_fabric_asan_options_t* options,
    iree_hal_device_event_sink_t event_sink, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_asan_t* out_state) {
  *out_state = (iree_hal_amd_fabric_asan_t){
      .api = api,
      .device = device,
      .host_allocator = host_allocator,
      .event_sink = event_sink,
      .quarantine_limit = options->quarantine_size,
  };
  iree_slim_mutex_initialize(&out_state->shadow_mutex);
  iree_slim_mutex_initialize(&out_state->application_mutex);
  iree_slim_mutex_initialize(&out_state->quarantine_mutex);
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_asan_validate_options(options));
  out_state->config = (iree_hal_amd_fabric_asan_config_t){
      .record_length = sizeof(out_state->config),
      .flags = 1,
      .shadow_scale_shift = options->shadow_scale_shift,
      .application_window_size = options->shadow_size
                                 << options->shadow_scale_shift,
      .shadow_size = options->shadow_size,
      .shadow_slab_size = options->shadow_slab_size,
  };
  out_state->pool_options = (iree_hal_asan_pool_options_t){
      .mode = IREE_HAL_ASAN_POOL_MODE_SHADOW,
      .shadow_granule_size = UINT64_C(1) << options->shadow_scale_shift,
      .redzone_size = options->redzone_size,
      .quarantine_size = options->quarantine_size,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_asan_pool_options_validate(&out_state->pool_options));
  amdf_memory_reservation_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->device_query_memory_reservation_capabilities(
          device, address_domain_ordinal, &capabilities),
      "ASAN reservation capabilities"));
  if (options->shadow_slab_size % capabilities.mapping_granularity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ASAN shadow slab must satisfy native mapping "
                            "granularity %" PRIu64,
                            capabilities.mapping_granularity);
  }
  out_state->application_source_flags = capabilities.required_source_flags;
  const amdf_memory_reservation_access_t access = {
      .device = device,
      .address_domain_ordinal = address_domain_ordinal,
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                capabilities.required_access,
  };
  amdf_memory_reservation_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .byte_length = options->owned_application_size,
      .minimum_alignment = options->shadow_slab_size,
      .access_count = 1,
      .accesses = &access,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_reservation_create(instance, &create_info,
                                     &out_state->application_reservation),
      "ASAN application reservation"));
  out_state->application_info = (amdf_memory_reservation_access_info_t){
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      .structure_size = sizeof(out_state->application_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_reservation_query_access_info(
          out_state->application_reservation, 0, &out_state->application_info),
      "ASAN application address"));
  if (out_state->application_info.address >
      out_state->config.application_window_size -
          options->owned_application_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ASAN application reservation exceeds shadow "
                            "address coverage");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*out_state->application_free_ranges),
      (void**)&out_state->application_free_ranges));
  *out_state->application_free_ranges =
      (iree_hal_amd_fabric_asan_application_range_t){
          .address = out_state->application_info.address,
          .length = options->owned_application_size,
      };
  create_info.byte_length = options->shadow_size;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_reservation_create(instance, &create_info,
                                     &out_state->shadow_reservation),
      "ASAN shadow reservation"));
  amdf_memory_reservation_access_info_t shadow_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      .structure_size = sizeof(shadow_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_reservation_query_access_info(out_state->shadow_reservation,
                                                0, &shadow_info),
      "ASAN shadow address"));
  out_state->config.shadow_base = shadow_info.address;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_asan_prepare_memory(
      out_state, shadow_scope, &capabilities));
  if (options->shadow_mode == IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_PREMAPPED) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_asan_premap_shadow(out_state));
  }
  return iree_ok_status();
}

static bool iree_hal_amd_fabric_asan_align_address(uint64_t address,
                                                   uint64_t alignment,
                                                   uint64_t* out_address) {
  *out_address = address;
  if (alignment <= 1) {
    return true;
  }
  if (address > UINT64_MAX - (alignment - 1)) {
    return false;
  }
  *out_address = (address + (alignment - 1)) & ~(alignment - 1);
  return true;
}

static iree_status_t iree_hal_amd_fabric_asan_application_range_allocate(
    iree_hal_amd_fabric_asan_t* state, uint64_t address,
    iree_device_size_t length,
    iree_hal_amd_fabric_asan_application_range_t** out_range) {
  *out_range = NULL;
  iree_hal_amd_fabric_asan_application_range_t* range = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(state->host_allocator,
                                             sizeof(*range), (void**)&range));
  range->address = address;
  range->length = length;
  range->next = NULL;
  *out_range = range;
  return iree_ok_status();
}

static void iree_hal_amd_fabric_asan_application_range_free(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_application_range_t* range) {
  iree_allocator_free(state->host_allocator, range);
}

static void iree_hal_amd_fabric_asan_application_range_free_list(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_application_range_t* range) {
  while (range) {
    iree_hal_amd_fabric_asan_application_range_t* next_range = range->next;
    iree_hal_amd_fabric_asan_application_range_free(state, range);
    range = next_range;
  }
}

static void iree_hal_amd_fabric_asan_quarantine_release_list(
    iree_hal_amd_fabric_asan_quarantine_entry_t* entry) {
  while (entry) {
    iree_hal_amd_fabric_asan_quarantine_entry_t* next_entry = entry->next;
    entry->next = NULL;
    entry->release_fn(entry->user_data);
    entry = next_entry;
  }
}

static void iree_hal_amd_fabric_asan_flush_quarantine(
    iree_hal_amd_fabric_asan_t* state) {
  iree_slim_mutex_lock(&state->quarantine_mutex);
  iree_hal_amd_fabric_asan_quarantine_entry_t* entry = state->quarantine_head;
  state->quarantine_head = NULL;
  state->quarantine_tail = NULL;
  state->quarantine_size = 0;
  iree_slim_mutex_unlock(&state->quarantine_mutex);
  iree_hal_amd_fabric_asan_quarantine_release_list(entry);
}

void iree_hal_amd_fabric_asan_quarantine(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_quarantine_entry_t* entry,
    iree_device_size_t mapped_size,
    void(IREE_API_PTR* release_fn)(void* user_data), void* user_data) {
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

  iree_hal_amd_fabric_asan_quarantine_entry_t* release_head = NULL;
  iree_hal_amd_fabric_asan_quarantine_entry_t* release_tail = NULL;
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
    iree_hal_amd_fabric_asan_quarantine_entry_t* oldest_entry =
        state->quarantine_head;
    state->quarantine_head = oldest_entry->next;
    if (!state->quarantine_head) {
      state->quarantine_tail = NULL;
    }
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

  iree_hal_amd_fabric_asan_quarantine_release_list(release_head);
}

static iree_status_t iree_hal_amd_fabric_asan_reserve_application_range(
    iree_hal_amd_fabric_asan_t* state, iree_device_size_t length,
    iree_device_size_t alignment, uint64_t* out_application_address,
    iree_hal_amd_fabric_asan_application_range_t** out_application_range) {
  *out_application_address = 0;
  *out_application_range = NULL;
  const uint64_t window_base = state->application_info.address;
  const uint64_t window_end = window_base + state->application_info.byte_length;
  uint64_t application_address = 0;
  iree_hal_amd_fabric_asan_application_range_t* application_range = NULL;
  iree_hal_amd_fabric_asan_application_range_t* suffix_range = NULL;

  iree_slim_mutex_lock(&state->application_mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amd_fabric_asan_application_range_t** current_range_ptr =
      &state->application_free_ranges;
  iree_hal_amd_fabric_asan_application_range_t* current_range =
      state->application_free_ranges;
  while (current_range) {
    const uint64_t range_end = current_range->address + current_range->length;
    if (iree_hal_amd_fabric_asan_align_address(
            current_range->address, alignment, &application_address) &&
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
        "AMD ASAN application window [0x%016" PRIx64 ", 0x%016" PRIx64
        ") has no space for length %" PRIu64 " with alignment %" PRIu64,
        window_base, window_end, (uint64_t)length, (uint64_t)alignment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_asan_application_range_allocate(
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
      status = iree_hal_amd_fabric_asan_application_range_allocate(
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
        iree_hal_amd_fabric_asan_application_range_free(state, current_range);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_application_address = application_address;
    *out_application_range = application_range;
    application_range = NULL;
  }
  iree_slim_mutex_unlock(&state->application_mutex);
  iree_hal_amd_fabric_asan_application_range_free(state, application_range);
  iree_hal_amd_fabric_asan_application_range_free(state, suffix_range);
  return status;
}

static void iree_hal_amd_fabric_asan_release_application_range(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_application_range_t* application_range) {
  if (!application_range) {
    return;
  }

  iree_slim_mutex_lock(&state->application_mutex);
  iree_hal_amd_fabric_asan_application_range_t** next_range_ptr =
      &state->application_free_ranges;
  iree_hal_amd_fabric_asan_application_range_t* next_range =
      state->application_free_ranges;
  iree_hal_amd_fabric_asan_application_range_t* previous_range = NULL;
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
      iree_hal_amd_fabric_asan_application_range_free(state, application_range);
      application_range = previous_range;
    }
  }
  if (next_range) {
    IREE_ASSERT(application_range_end <= next_range->address);
    if (application_range_end == next_range->address) {
      if (application_range == previous_range) {
        application_range->length += next_range->length;
        application_range->next = next_range->next;
        iree_hal_amd_fabric_asan_application_range_free(state, next_range);
      } else {
        next_range->address = application_range->address;
        next_range->length += application_range->length;
        iree_hal_amd_fabric_asan_application_range_free(state,
                                                        application_range);
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

static void iree_hal_amd_fabric_asan_fail(iree_hal_amd_fabric_asan_t* state,
                                          amdf_status_t status,
                                          const char* operation) {
  int64_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &state->failure, &expected, (int64_t)status,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amd_report_failure(state->event_sink, IREE_SV("amd"), status,
                                operation);
    if (state->failure_callback) {
      iree_status_t failure = IREE_HAL_AMD_STATUS_FROM_AMDF(status, operation);
      state->failure_callback(state->failure_user_data,
                              iree_status_code(failure));
      iree_status_free(failure);
    }
  }
}

static iree_hal_amd_fabric_asan_shadow_slab_t*
iree_hal_amd_fabric_asan_find_slab(iree_hal_amd_fabric_asan_t* state,
                                   uint64_t index) {
  for (iree_host_size_t i = 0; i < state->slab_count; ++i) {
    if (state->slabs[i].index == index) {
      return &state->slabs[i];
    }
  }
  return NULL;
}

static iree_status_t iree_hal_amd_fabric_asan_commit_slab(
    iree_hal_amd_fabric_asan_t* state, uint64_t index) {
  if (iree_hal_amd_fabric_asan_find_slab(state, index)) {
    return iree_ok_status();
  }
  if (state->slab_count == state->slab_capacity) {
    const iree_host_size_t capacity =
        state->slab_capacity ? state->slab_capacity * 2 : 8;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        state->host_allocator, capacity * sizeof(*state->slabs),
        (void**)&state->slabs));
    state->slab_capacity = capacity;
  }
  iree_hal_amd_fabric_asan_shadow_slab_t* slab =
      &state->slabs[state->slab_count];
  *slab = (iree_hal_amd_fabric_asan_shadow_slab_t){.index = index};
  iree_status_t status =
      iree_hal_amd_fabric_asan_allocate_shadow(state, &slab->memory);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_fabric_memory_deinitialize(state->api, state->event_sink,
                                            &slab->memory);
    return status;
  }
  // The update may partially install backing on a native failure. Retain this
  // owner until complete reservation removal proves that it is safe to free.
  ++state->slab_count;
  const amdf_memory_mapping_update_t update = {
      .target_byte_offset = index * state->config.shadow_slab_size,
      .byte_length = state->config.shadow_slab_size,
      .memory = slab->memory.handle,
      .access = state->access.requirements.access,
  };
  const amdf_status_t native_status = state->api->memory_reservation_update(
      state->shadow_reservation, 0, 1, &update);
  if (!amdf_status_is_ok(native_status)) {
    iree_hal_amd_fabric_asan_fail(state, native_status,
                                  "ASAN precise shadow mapping");
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                       "ASAN precise shadow mapping");
}

iree_status_t iree_hal_amd_fabric_asan_map_range(
    iree_hal_amd_fabric_asan_t* state, uint64_t address, uint64_t length) {
  if (!length) {
    return iree_ok_status();
  }
  if (address > state->config.application_window_size ||
      length > state->config.application_window_size - address) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "application range is outside ASAN shadow coverage");
  }
  const uint32_t shift = state->config.shadow_scale_shift;
  const uint64_t first = (address >> shift) / state->config.shadow_slab_size;
  const uint64_t last =
      ((address + length - 1) >> shift) / state->config.shadow_slab_size;
  iree_slim_mutex_lock(&state->shadow_mutex);
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      (amdf_status_t)iree_atomic_load(&state->failure,
                                      iree_memory_order_acquire),
      "ASAN shadow publication");
  for (uint64_t i = first; i <= last && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_fabric_asan_commit_slab(state, i);
  }
  iree_slim_mutex_unlock(&state->shadow_mutex);
  return status;
}

// The caller holds shadow_mutex and has committed every intersected slab.
static amdf_status_t iree_hal_amd_fabric_asan_write_shadow(
    iree_hal_amd_fabric_asan_t* state, uint64_t offset, uint64_t length,
    uint8_t value) {
  amdf_status_t status = 0;
  while (length && amdf_status_is_ok(status)) {
    const uint64_t index = offset / state->config.shadow_slab_size;
    const uint64_t slab_offset = offset % state->config.shadow_slab_size;
    const uint64_t count =
        iree_min(length, state->config.shadow_slab_size - slab_offset);
    iree_hal_amd_fabric_asan_shadow_slab_t* slab =
        iree_hal_amd_fabric_asan_find_slab(state, index);
    // Pool advice may not allocate or discover missing shadow after handing
    // the backing to its caller. Slab acquisition establishes this invariant.
    IREE_ASSERT(slab);
    memset(slab->memory.contents.data + slab_offset, value, count);
    status = state->api->host_mapping_cache_control(
        slab->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, slab_offset,
        count);
    offset += count;
    length -= count;
  }
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_fabric_asan_fail(state, status,
                                  "ASAN shadow write publication");
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_asan_publish_imported_range(
    iree_hal_amd_fabric_asan_t* state, uint64_t address, uint64_t length) {
  if (!length) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_asan_map_range(state, address, length));
  const uint32_t shift = state->config.shadow_scale_shift;
  const uint64_t end = address + length;
  const uint64_t first_shadow = address >> shift;
  const uint64_t end_shadow = end >> shift;
  const uint8_t tail = (uint8_t)(end & ((UINT64_C(1) << shift) - 1));
  iree_slim_mutex_lock(&state->shadow_mutex);
  amdf_status_t status = iree_hal_amd_fabric_asan_write_shadow(
      state, first_shadow, end_shadow - first_shadow, 0);
  if (tail && amdf_status_is_ok(status)) {
    iree_hal_amd_fabric_asan_shadow_slab_t* slab =
        iree_hal_amd_fabric_asan_find_slab(
            state, end_shadow / state->config.shadow_slab_size);
    const uint8_t value =
        slab->memory.contents.data[end_shadow % state->config.shadow_slab_size];
    // An addressable granule remains fully addressable. A partial granule may
    // grow; poison becomes the imported range's accessible prefix.
    if (value && (value >= (1u << shift) || value < tail)) {
      status =
          iree_hal_amd_fabric_asan_write_shadow(state, end_shadow, 1, tail);
    }
  }
  iree_slim_mutex_unlock(&state->shadow_mutex);
  return IREE_HAL_AMD_STATUS_FROM_AMDF(status,
                                       "ASAN imported shadow publication");
}

void iree_hal_amd_fabric_asan_advise_range(iree_hal_amd_fabric_asan_t* state,
                                           uint64_t backing_address,
                                           uint64_t backing_length,
                                           uint64_t accessible_offset,
                                           uint64_t accessible_length) {
  if (!backing_length) {
    return;
  }
  const uint32_t shift = state->config.shadow_scale_shift;
  const uint64_t shadow = backing_address >> shift;
  const uint64_t accessible = shadow + (accessible_offset >> shift);
  const uint8_t tail =
      (uint8_t)(accessible_length & ((UINT64_C(1) << shift) - 1));
  iree_slim_mutex_lock(&state->shadow_mutex);
  amdf_status_t status = iree_hal_amd_fabric_asan_write_shadow(
      state, shadow, backing_length >> shift, IREE_HAL_AMD_ASAN_POISON);
  if (accessible_length && amdf_status_is_ok(status)) {
    status = iree_hal_amd_fabric_asan_write_shadow(
        state, accessible, accessible_length >> shift, 0);
    if (tail && amdf_status_is_ok(status)) {
      iree_hal_amd_fabric_asan_write_shadow(
          state, accessible + (accessible_length >> shift), 1, tail);
    }
  }
  iree_slim_mutex_unlock(&state->shadow_mutex);
}

iree_status_t iree_hal_amd_fabric_asan_map_application(
    iree_hal_amd_fabric_asan_t* state, amdf_memory_t* memory,
    uint32_t memory_access_ordinal, uint64_t length, uint64_t alignment,
    iree_hal_amd_fabric_asan_application_range_t** out_range) {
  *out_range = NULL;
  const uint64_t granularity = state->application_info.mapping_granularity;
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_asan_reserve_application_range(
      state, length, iree_max(alignment, granularity), &address, out_range));
  const amdf_memory_mapping_update_t update = {
      .target_byte_offset = address - state->application_info.address,
      .byte_length = length,
      .memory = memory,
      .memory_access_ordinal = memory_access_ordinal,
      .access = state->application_info.access,
  };
  iree_slim_mutex_lock(&state->application_mutex);
  const amdf_status_t status = state->api->memory_reservation_update(
      state->application_reservation, 0, 1, &update);
  if (!amdf_status_is_ok(status)) {
    // An application mapping failure invalidates the complete native group.
    // Keep proposed backing and native parents alive; no individual-range
    // detach can establish safety for a partially applied failed group.
    state->cleanup_blocked = true;
    iree_hal_amd_fabric_asan_fail(state, status, "ASAN application mapping");
  }
  iree_slim_mutex_unlock(&state->application_mutex);
  return iree_status_annotate_f(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "ASAN application mapping"),
      "offset=%" PRIu64 " length=%" PRIu64 " granularity=%" PRIu64,
      update.target_byte_offset, length, granularity);
}

bool iree_hal_amd_fabric_asan_unmap_application(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_application_range_t* range) {
  if (!range) {
    return true;
  }
  const amdf_memory_mapping_update_t update = {
      .target_byte_offset = range->address - state->application_info.address,
      .byte_length = range->length,
  };
  iree_slim_mutex_lock(&state->application_mutex);
  amdf_status_t status = 0;
  if (state->cleanup_blocked) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  } else {
    status = state->api->memory_reservation_update(
        state->application_reservation, 0, 1, &update);
    if (!amdf_status_is_ok(status)) {
      state->cleanup_blocked = true;
      iree_hal_amd_fabric_asan_fail(state, status, "ASAN application removal");
    }
  }
  iree_slim_mutex_unlock(&state->application_mutex);
  if (!amdf_status_is_ok(status)) {
    return false;
  }
  iree_hal_amd_fabric_asan_release_application_range(state, range);
  return true;
}

void iree_hal_amd_fabric_asan_query_statistics(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_device_asan_observation_t* out_statistics) {
  *out_statistics = (iree_hal_device_asan_observation_t){
      .flags = IREE_HAL_DEVICE_ASAN_OBSERVATION_FLAG_ALL};
  iree_slim_mutex_lock(&state->quarantine_mutex);
  out_statistics->quarantine_size = state->quarantine_size;
  out_statistics->quarantine_eviction_count = state->quarantine_eviction_count;
  iree_slim_mutex_unlock(&state->quarantine_mutex);
  iree_slim_mutex_lock(&state->shadow_mutex);
  out_statistics->shadow_mapped_slab_count = state->slab_count;
  out_statistics->shadow_committed_size =
      state->slab_count * state->config.shadow_slab_size +
      state->alias_memory.info.byte_length;
  iree_slim_mutex_unlock(&state->shadow_mutex);
}

bool iree_hal_amd_fabric_asan_deinitialize(iree_hal_amd_fabric_asan_t* state) {
  if (!state->api) {
    return true;
  }
  if (state->cleanup_blocked) {
    return false;
  }
  iree_hal_amd_fabric_asan_flush_quarantine(state);
  if (state->cleanup_blocked) {
    return false;
  }
  amdf_memory_reservation_t** reservations[] = {&state->application_reservation,
                                                &state->shadow_reservation};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(reservations); ++i) {
    if (!*reservations[i]) {
      continue;
    }
    const amdf_status_t status =
        state->api->memory_reservation_destroy(*reservations[i]);
    *reservations[i] = NULL;
    if (!amdf_status_is_ok(status)) {
      state->cleanup_blocked = true;
      iree_hal_amd_fabric_asan_fail(state, status, "ASAN reservation removal");
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < state->slab_count; ++i) {
    iree_hal_amd_fabric_memory_deinitialize(state->api, state->event_sink,
                                            &state->slabs[i].memory);
  }
  iree_hal_amd_fabric_memory_deinitialize(state->api, state->event_sink,
                                          &state->alias_memory);
  iree_allocator_free(state->host_allocator, state->slabs);
  iree_hal_amd_fabric_asan_application_range_free_list(
      state, state->application_free_ranges);
  iree_slim_mutex_deinitialize(&state->quarantine_mutex);
  iree_slim_mutex_deinitialize(&state->application_mutex);
  iree_slim_mutex_deinitialize(&state->shadow_mutex);
  *state = (iree_hal_amd_fabric_asan_t){0};
  return true;
}
