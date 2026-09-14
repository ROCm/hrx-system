// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "common/internal.h"

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_device_count(iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  *out_count = device_registry->device_count;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_select_primary_queue(
    iree_hal_streaming_device_t* device, iree_hal_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_queue);

  const iree_hal_queue_family_role_flags_t required_roles =
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
      IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device->hal_device));
  for (iree_host_size_t family_ordinal = 0;
       queue_spec && family_ordinal < queue_spec->family_count;
       ++family_ordinal) {
    const iree_hal_queue_family_spec_t* family_spec =
        &queue_spec->families[family_ordinal];
    if (family_spec->provisioned_queue_count == 0 ||
        !iree_all_bits_set(family_spec->role_flags, required_roles)) {
      continue;
    }
    iree_hal_queue_t* queue = iree_hal_device_queue(
        device->hal_device, (iree_hal_queue_family_ordinal_t)family_ordinal,
        /*queue_ordinal=*/0);
    if (IREE_UNLIKELY(!queue)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "device advertises provisioned queue family %" PRIhsz
          " but the queue is unavailable",
          family_ordinal);
    }
    *out_queue = queue;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "device has no provisioned transfer-and-dispatch queue");
}

static iree_status_t iree_hal_streaming_device_by_ordinal(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_hal_streaming_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  if (ordinal >= device_registry->device_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinal %zu out of range [0, %zu)", ordinal,
                            device_registry->device_count);
  }

  iree_hal_streaming_device_t* device = &device_registry->devices[ordinal];

  // Device is always created during initialization.
  // Primary context is created lazily on first access.
  IREE_ASSERT(device->hal_device);

  *out_device = device;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_name(
    iree_hal_streaming_device_ordinal_t ordinal, char* name,
    iree_host_size_t name_size) {
  IREE_ASSERT_ARGUMENT(name);
  if (name_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "name_size must be > 0");
  }

  iree_hal_streaming_device_t* device = NULL;
  iree_status_t status = iree_hal_streaming_device_by_ordinal(ordinal, &device);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Calculate safe copy length: min(source_length, dest_size - 1)
  const iree_host_size_t source_len = device->info.name.size;
  const iree_host_size_t copy_len =
      source_len < (name_size - 1) ? source_len : (name_size - 1);

  // Copy the name data safely
  if (copy_len > 0) {
    memcpy(name, device->info.name.data, copy_len);
  }

  // Always null-terminate
  name[copy_len] = '\0';

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_get_string_property(
    iree_hal_streaming_device_ordinal_t ordinal, const char* category,
    const char* key, char* value, iree_host_size_t value_size) {
  IREE_ASSERT_ARGUMENT(category);
  IREE_ASSERT_ARGUMENT(key);
  IREE_ASSERT_ARGUMENT(value);
  if (value_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_size must be > 0");
  }

  iree_hal_streaming_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_device_by_ordinal(ordinal, &device));

  // The streaming layer owns the set of string-valued device properties and
  // resolves them from values cached on iree_hal_streaming_device_t during
  // device initialization. The underlying IREE HAL intentionally does not
  // expose a string-property query; callers must go through this API.
  iree_string_view_t source = iree_string_view_empty();
  if (iree_string_view_equal(iree_make_cstring_view(category),
                             IREE_SV("hal.device"))) {
    const iree_string_view_t key_sv = iree_make_cstring_view(key);
    if (iree_string_view_equal(key_sv, IREE_SV("name"))) {
      source = device->info.name;
    } else if (iree_string_view_equal(key_sv, IREE_SV("path"))) {
      source = device->info.path;
    } else if (iree_string_view_equal(key_sv, IREE_SV("architecture")) ||
               iree_string_view_equal(key_sv, IREE_SV("gcn_arch_name"))) {
      source = iree_make_cstring_view(device->gcn_arch_name);
    }
  }

  if (iree_string_view_is_empty(source)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "unknown string device property '%s' in category '%s'", key, category);
  }

  if (source.size + 1 > value_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer of %" PRIhsz
                            " bytes is too small for property '%s:%s' "
                            "(requires %" PRIhsz " bytes including NUL)",
                            value_size, category, key, source.size + 1);
  }
  memcpy(value, source.data, source.size);
  value[source.size] = '\0';
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_memory_info(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_device_size_t* out_free_memory, iree_device_size_t* out_total_memory) {
  IREE_ASSERT_ARGUMENT(out_free_memory);
  IREE_ASSERT_ARGUMENT(out_total_memory);
  *out_free_memory = 0;
  *out_total_memory = 0;

  iree_hal_streaming_device_t* device = NULL;
  iree_status_t status = iree_hal_streaming_device_by_ordinal(ordinal, &device);
  if (iree_status_is_ok(status)) {
    *out_free_memory = (iree_device_size_t)iree_atomic_load(
        &device->free_memory, iree_memory_order_relaxed);
    *out_total_memory = device->total_memory;
  }
  return status;
}

iree_status_t iree_hal_streaming_device_set_primary_context_flags(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    const iree_hal_streaming_context_flags_t* flags) {
  IREE_ASSERT_ARGUMENT(flags);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_ordinal);
  if (!device) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "invalid device ordinal %" PRIhsz, device_ordinal));
  }

  iree_slim_mutex_lock(&device->primary_context_mutex);
  device->primary_context_flags = *flags;
  if (device->primary_context) {
    device->primary_context->flags = *flags;
  }
  iree_slim_mutex_unlock(&device->primary_context_mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_primary_context_state(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_context_flags_t* out_flags, bool* out_active) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_t* device =
      iree_hal_streaming_device_entry(device_ordinal);
  if (!device) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "invalid device ordinal %" PRIhsz, device_ordinal));
  }

  iree_slim_mutex_lock(&device->primary_context_mutex);
  if (out_flags) {
    *out_flags = device->primary_context ? device->primary_context->flags
                                         : device->primary_context_flags;
  }
  if (out_active) {
    *out_active = device->primary_context != NULL;
  }
  iree_slim_mutex_unlock(&device->primary_context_mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_device_ensure_default_mem_pool_locked(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  if (device->default_mem_pool && device->current_mem_pool) {
    return iree_ok_status();
  }

  if (!iree_hal_streaming_device_registry()) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device registry not initialized");
  }

  hrx_mem_pool_t default_mem_pool = NULL;
  if (!device->default_mem_pool) {
    hrx_mem_pool_props_t props = {
        .alloc_handle_type = 0,
        .location_type = 1,  // device
        .location_id = (int)device->ordinal,
    };
    IREE_RETURN_IF_ERROR(HRX_CALL(
        hrx_mem_pool_create(device->hrx_device, &props, &default_mem_pool)));
    device->default_mem_pool = default_mem_pool;
  }

  if (!device->current_mem_pool) {
    device->current_mem_pool = device->default_mem_pool;
    hrx_mem_pool_retain(device->current_mem_pool);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_device_ensure_default_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_status_t status =
      iree_hal_streaming_device_ensure_default_mem_pool_locked(device);
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return status;
}

// Requires |device->primary_context_mutex| to be held. The context and its
// default allocation pool become visible together so callers never observe a
// context that cannot service runtime allocations.
static iree_status_t iree_hal_streaming_device_create_primary_context_locked(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  if (device->primary_context) {
    return iree_hal_streaming_device_ensure_default_mem_pool_locked(device);
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device registry not initialized");
  }

  const bool had_default_mem_pool = device->default_mem_pool != NULL;
  const bool had_current_mem_pool = device->current_mem_pool != NULL;
  iree_hal_streaming_context_t* context = NULL;
  iree_status_t status = iree_hal_streaming_context_create(
      device, device->primary_context_flags, device_registry->host_allocator,
      &context);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_device_ensure_default_mem_pool_locked(device);
  }
  if (iree_status_is_ok(status)) {
    device->primary_context = context;
    return iree_ok_status();
  }

  iree_hal_streaming_context_release(context);
  if (!had_current_mem_pool) {
    hrx_mem_pool_release(device->current_mem_pool);
    device->current_mem_pool = NULL;
  }
  if (!had_default_mem_pool) {
    hrx_mem_pool_release(device->default_mem_pool);
    device->default_mem_pool = NULL;
  }
  return status;
}

iree_status_t iree_hal_streaming_device_get_or_create_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_context = NULL;

  iree_slim_mutex_lock(&device->primary_context_mutex);
  iree_status_t status =
      iree_hal_streaming_device_create_primary_context_locked(device);
  if (iree_status_is_ok(status)) {
    *out_context = device->primary_context;
  }
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_device_retain_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device->primary_context_mutex);

  iree_hal_streaming_context_t* retained_context = NULL;
  iree_status_t status = iree_ok_status();
  if (device->primary_context_ref_count == INT32_MAX) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "primary context reference count overflow");
  } else {
    ++device->primary_context_ref_count;
    status = iree_hal_streaming_device_create_primary_context_locked(device);
    if (iree_status_is_ok(status)) {
      retained_context = device->primary_context;
      iree_hal_streaming_context_retain(retained_context);
    } else {
      --device->primary_context_ref_count;
    }
  }

  iree_slim_mutex_unlock(&device->primary_context_mutex);
  if (iree_status_is_ok(status)) {
    *out_context = retained_context;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_device_release_primary_context(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&device->primary_context_mutex);

  iree_status_t status = iree_ok_status();
  if (device->primary_context_ref_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "primary context not retained");
  } else {
    iree_hal_streaming_context_t* retained_context = device->primary_context;
    --device->primary_context_ref_count;

    if (device->primary_context_ref_count == 0) {
      status = iree_hal_streaming_context_wait_idle(retained_context,
                                                    iree_infinite_timeout());

      if (iree_hal_streaming_context_current() == retained_context) {
        iree_hal_streaming_context_set_current(NULL);
      }

      // Release the device's primary-context ownership.
      iree_hal_streaming_context_release(retained_context);
      device->primary_context = NULL;

      hrx_mem_pool_release(device->current_mem_pool);
      device->current_mem_pool = NULL;
      hrx_mem_pool_release(device->default_mem_pool);
      device->default_mem_pool = NULL;
    }

    // Release the owning reference returned by the matching retain call.
    iree_hal_streaming_context_release(retained_context);
  }

  iree_slim_mutex_unlock(&device->primary_context_mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
