// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_spec.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/device_spec_serialization.h"

//===----------------------------------------------------------------------===//
// iree_hal_device_spec_t
//===----------------------------------------------------------------------===//

struct iree_hal_device_spec_t {
  // Retain/release counter.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for all owned storage.
  iree_allocator_t host_allocator;
  // Stable non-cryptographic fingerprint of the canonical byte image.
  uint64_t fingerprint;
  // Owned string table backing public string views.
  char* string_table;
  // Owned string table byte length.
  iree_host_size_t string_table_length;
  // Owned physical device records.
  iree_hal_physical_device_spec_t* physical_devices;
  // Owned memory heap records.
  iree_hal_memory_heap_spec_t* memory_heaps;
  // Owned memory type records.
  iree_hal_memory_type_spec_t* memory_types;
  // Owned external buffer handle records.
  iree_hal_external_buffer_handle_spec_t* external_buffer_handles;
  // Owned virtual memory class records.
  iree_hal_virtual_memory_class_spec_t* virtual_memory_classes;
  // Owned queue family records.
  iree_hal_queue_family_spec_t* queue_families;
  // Owned external timepoint handle records.
  iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles;
  // Owned executable target records.
  iree_hal_executable_target_t* executable_targets;
  // Owned driver-local extension facet records.
  iree_hal_device_spec_facet_t* facets;
  // Number of owned driver-local extension facet records.
  iree_host_size_t facet_count;
  // Owned driver-local extension facet payload storage.
  uint8_t* facet_payload_storage;
  // Owned driver-local extension facet payload storage byte length.
  iree_host_size_t facet_payload_storage_length;
  // Logical device identity facet.
  iree_hal_device_identity_spec_t identity;
  // Memory capability facet.
  iree_hal_device_memory_spec_t memory;
  // Virtual memory capability facet.
  iree_hal_device_virtual_memory_spec_t virtual_memory;
  // Queue capability facet.
  iree_hal_device_queue_spec_t queues;
  // Dispatch capability facet.
  iree_hal_device_dispatch_spec_t dispatch;
  // Timing and profiling capability facet.
  iree_hal_device_timing_spec_t timing;
  // Executable capability facet.
  iree_hal_device_executable_spec_t executables;
  // Sanitizer configuration facet.
  iree_hal_device_sanitizer_spec_t sanitizer;
};

static iree_status_t iree_hal_device_spec_add_size(
    iree_host_size_t value, iree_host_size_t* inout_total) {
  if (IREE_UNLIKELY(value > IREE_HOST_SIZE_MAX - *inout_total)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec storage size overflow");
  }
  *inout_total += value;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_accumulate_string(
    iree_string_view_t value, iree_host_size_t* inout_total) {
  if (IREE_UNLIKELY(value.size && !value.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty device spec string has NULL storage");
  }
  return iree_hal_device_spec_add_size(value.size, inout_total);
}

static iree_status_t iree_hal_device_spec_accumulate_bytes(
    iree_const_byte_span_t value, iree_host_size_t* inout_total) {
  if (IREE_UNLIKELY(value.data_length && !value.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty device spec payload has NULL storage");
  }
  return iree_hal_device_spec_add_size(value.data_length, inout_total);
}

static iree_string_view_t iree_hal_device_spec_copy_string(
    iree_string_view_t value, char* storage,
    iree_host_size_t* inout_storage_offset) {
  if (iree_string_view_is_empty(value)) return iree_string_view_empty();
  iree_string_view_t result =
      iree_make_string_view(storage + *inout_storage_offset, value.size);
  memcpy((void*)result.data, value.data, value.size);
  *inout_storage_offset += value.size;
  return result;
}

static iree_const_byte_span_t iree_hal_device_spec_copy_bytes(
    iree_const_byte_span_t value, uint8_t* storage,
    iree_host_size_t* inout_storage_offset) {
  if (iree_const_byte_span_is_empty(value)) {
    return iree_const_byte_span_empty();
  }
  iree_const_byte_span_t result = iree_make_const_byte_span(
      storage + *inout_storage_offset, value.data_length);
  memcpy((void*)result.data, value.data, value.data_length);
  *inout_storage_offset += value.data_length;
  return result;
}

static iree_status_t iree_hal_device_spec_validate_count_pointer(
    iree_host_size_t count, const void* ptr, const char* field_name) {
  if (IREE_UNLIKELY(count && !ptr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s has count %" PRIhsz
                            " but NULL storage",
                            field_name, count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_validate_concrete_affinity(
    const char* field_name, iree_host_size_t index,
    iree_hal_physical_device_affinity_t affinity,
    iree_hal_physical_device_affinity_t available_affinity) {
  if (IREE_UNLIKELY(affinity == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s %" PRIhsz
                            " has empty physical-device affinity",
                            field_name, index);
  }
  if (IREE_UNLIKELY(available_affinity != 0 &&
                    (affinity & ~available_affinity) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s %" PRIhsz
                            " has physical-device affinity 0x%016" PRIx64
                            " outside the advertised affinity 0x%016" PRIx64,
                            field_name, index, affinity, available_affinity);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_validate_params(
    const iree_hal_device_spec_params_t* params) {
  if (!params) return iree_ok_status();
  iree_hal_physical_device_affinity_t available_affinity = 0;
  const iree_hal_device_identity_spec_t* identity = params->identity;
  if (identity) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        identity->physical_device_count, identity->physical_devices,
        "identity.physical_devices"));
    if (IREE_UNLIKELY(identity->physical_device_count >
                      IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "device spec physical device count %" PRIhsz
                              " exceeds physical-device affinity capacity %d",
                              identity->physical_device_count,
                              IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT);
    }
    for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
      const iree_hal_physical_device_affinity_t affinity =
          identity->physical_devices[i].physical_device_affinity;
      if (IREE_UNLIKELY(!iree_is_power_of_two_uint64(affinity))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device spec physical device %" PRIhsz
            " must have one unique affinity bit; got 0x%016" PRIx64,
            i, affinity);
      }
      if (IREE_UNLIKELY((available_affinity & affinity) != 0)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device spec physical device %" PRIhsz
                                " reuses affinity bit 0x%016" PRIx64,
                                i, affinity);
      }
      available_affinity |= affinity;
    }
  }
  const iree_hal_device_memory_spec_t* memory = params->memory;
  if (memory) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        memory->heap_count, memory->heaps, "memory.heaps"));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        memory->memory_type_count, memory->memory_types,
        "memory.memory_types"));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        memory->external_buffer_handle_count, memory->external_buffer_handles,
        "memory.external_buffer_handles"));
    for (iree_host_size_t i = 0; i < memory->heap_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_concrete_affinity(
          "memory heap", i, memory->heaps[i].physical_device_affinity,
          available_affinity));
    }
    for (iree_host_size_t i = 0; i < memory->memory_type_count; ++i) {
      if (memory->memory_types[i].heap_index >= memory->heap_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device spec memory type %" PRIhsz
            " references heap %u outside heap count %" PRIhsz,
            i, memory->memory_types[i].heap_index, memory->heap_count);
      }
    }
  }
  const iree_hal_device_virtual_memory_spec_t* virtual_memory =
      params->virtual_memory;
  if (virtual_memory) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        virtual_memory->class_count, virtual_memory->classes,
        "virtual_memory.classes"));
  }
  const iree_hal_device_queue_spec_t* queues = params->queues;
  if (queues) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        queues->family_count, queues->families, "queues.families"));
    if (IREE_UNLIKELY(queues->family_count > IREE_HAL_MAX_QUEUE_FAMILIES)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "device spec queue family count %" PRIhsz
          " exceeds queue-family affinity capacity %" PRIhsz,
          queues->family_count, (iree_host_size_t)IREE_HAL_MAX_QUEUE_FAMILIES);
    }
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        queues->external_timepoint_handle_count,
        queues->external_timepoint_handles,
        "queues.external_timepoint_handles"));
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_concrete_affinity(
          "queue family", i, queues->families[i].physical_device_affinity,
          available_affinity));
    }
    for (iree_host_size_t i = 0; i < queues->external_timepoint_handle_count;
         ++i) {
      const iree_hal_external_timepoint_type_t handle_type =
          queues->external_timepoint_handles[i].handle_type;
      if (!iree_hal_external_timepoint_type_is_valid(handle_type)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device spec external timepoint handle %" PRIhsz
                                " has invalid type %" PRIu32,
                                i, (uint32_t)handle_type);
      }
    }
  }
  const iree_hal_device_executable_spec_t* executables = params->executables;
  if (executables) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        executables->target_count, executables->targets,
        "executables.targets"));
    for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
      const iree_hal_executable_target_t* target = &executables->targets[i];
      if (iree_string_view_is_empty(target->family)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device spec executable target %" PRIhsz " has an empty family", i);
      }
      if (iree_string_view_is_empty(target->target_key)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device spec executable target %" PRIhsz
                                " has an empty target key",
                                i);
      }
      if (target->kind > IREE_HAL_EXECUTABLE_TARGET_KIND_COMPOSITE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device spec executable target %" PRIhsz
                                " has invalid kind %" PRIu32,
                                i, target->kind);
      }
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_concrete_affinity(
          "executable target", i, target->physical_device_affinity,
          available_affinity));
    }
  }
  const iree_hal_device_sanitizer_spec_t* sanitizer = params->sanitizer;
  if (sanitizer) {
    const iree_hal_device_sanitizer_flags_t known_flags =
        IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN;
    if (sanitizer->flags & ~known_flags) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device spec sanitizer flags 0x%" PRIx32
                              " contain unsupported bits 0x%" PRIx32,
                              sanitizer->flags,
                              sanitizer->flags & ~known_flags);
    }
    const bool asan_flag_set =
        iree_any_bit_set(sanitizer->flags, IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN);
    const bool asan_options_enabled =
        iree_hal_asan_pool_options_is_enabled(&sanitizer->asan.pool_options);
    if (asan_flag_set) {
      IREE_RETURN_IF_ERROR(
          iree_hal_asan_pool_options_validate(&sanitizer->asan.pool_options));
      if (!asan_options_enabled) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device spec enables ASAN without enabled ASAN pool options");
      }
    } else if (asan_options_enabled) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device spec has ASAN pool options without ASAN enabled");
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
      params->facet_count, params->facets, "facets"));
  for (iree_host_size_t i = 0; i < params->facet_count; ++i) {
    const iree_string_view_t schema_id = params->facets[i].schema_id;
    if (schema_id.size && !schema_id.data) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device spec facet %" PRIhsz " schema ID has NULL storage", i);
    }
    if (iree_string_view_is_empty(schema_id)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device spec facet %" PRIhsz " has an empty schema ID", i);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(schema_id, params->facets[j].schema_id)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device spec facet %" PRIhsz
                                " duplicates schema ID from facet %" PRIhsz,
                                i, j);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_count_strings_and_payloads(
    const iree_hal_device_spec_params_t* params,
    iree_host_size_t* out_string_table_length,
    iree_host_size_t* out_payload_table_length) {
  iree_host_size_t string_table_length = 0;
  iree_host_size_t payload_table_length = 0;
  if (!params) {
    *out_string_table_length = 0;
    *out_payload_table_length = 0;
    return iree_ok_status();
  }
  const iree_hal_device_identity_spec_t* identity = params->identity;
  if (identity) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->logical_device_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->display_name, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->driver_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->driver_version, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->backend_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->device_path, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->vendor_name, &string_table_length));
    for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          identity->physical_devices[i].identity.display_name,
          &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          identity->physical_devices[i].identity.backend_path,
          &string_table_length));
    }
  }
  const iree_hal_device_memory_spec_t* memory = params->memory;
  if (memory) {
    for (iree_host_size_t i = 0; i < memory->heap_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          memory->heaps[i].name, &string_table_length));
    }
  }
  const iree_hal_device_queue_spec_t* queues = params->queues;
  if (queues) {
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          queues->families[i].name, &string_table_length));
    }
  }
  const iree_hal_device_executable_spec_t* executables = params->executables;
  if (executables) {
    for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
      const iree_hal_executable_target_t* target = &executables->targets[i];
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->family, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->target_key, &string_table_length));
    }
  }
  for (iree_host_size_t i = 0; i < params->facet_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        params->facets[i].schema_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_bytes(
        params->facets[i].payload, &payload_table_length));
  }
  *out_string_table_length = string_table_length;
  *out_payload_table_length = payload_table_length;
  return iree_ok_status();
}

static void iree_hal_device_spec_destroy(iree_hal_device_spec_t* spec) {
  iree_allocator_t host_allocator = spec->host_allocator;
  iree_allocator_free(host_allocator, spec->string_table);
  iree_allocator_free(host_allocator, spec->physical_devices);
  iree_allocator_free(host_allocator, spec->memory_heaps);
  iree_allocator_free(host_allocator, spec->memory_types);
  iree_allocator_free(host_allocator, spec->external_buffer_handles);
  iree_allocator_free(host_allocator, spec->virtual_memory_classes);
  iree_allocator_free(host_allocator, spec->queue_families);
  iree_allocator_free(host_allocator, spec->external_timepoint_handles);
  iree_allocator_free(host_allocator, spec->executable_targets);
  iree_allocator_free(host_allocator, spec->facets);
  iree_allocator_free(host_allocator, spec->facet_payload_storage);
  iree_allocator_free(host_allocator, spec);
}

static iree_status_t iree_hal_device_spec_clone_array(
    iree_allocator_t host_allocator, iree_host_size_t count,
    iree_host_size_t element_size, const void* source, void** out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  if (!count) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, count,
                                                   element_size, out_target));
  memcpy(*out_target, source, count * element_size);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_spec_create(
    const iree_hal_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_params(params));

  iree_host_size_t string_table_length = 0;
  iree_host_size_t facet_payload_storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_count_strings_and_payloads(
      params, &string_table_length, &facet_payload_storage_length));

  iree_hal_device_spec_t* spec = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*spec), (void**)&spec));
  memset(spec, 0, sizeof(*spec));
  iree_atomic_ref_count_init(&spec->ref_count);
  spec->host_allocator = host_allocator;
  spec->string_table_length = string_table_length;
  spec->facet_payload_storage_length = facet_payload_storage_length;

  iree_status_t status = iree_ok_status();
  if (string_table_length) {
    status = iree_allocator_malloc_uninitialized(
        host_allocator, string_table_length, (void**)&spec->string_table);
  }
  if (iree_status_is_ok(status) && facet_payload_storage_length) {
    status = iree_allocator_malloc_uninitialized(
        host_allocator, facet_payload_storage_length,
        (void**)&spec->facet_payload_storage);
  }

  char* string_storage = spec->string_table;
  iree_host_size_t string_offset = 0;
  uint8_t* payload_storage = spec->facet_payload_storage;
  iree_host_size_t payload_offset = 0;

  if (iree_status_is_ok(status) && params && params->identity) {
    spec->identity = *params->identity;
    spec->identity.logical_device_id = iree_hal_device_spec_copy_string(
        params->identity->logical_device_id, string_storage, &string_offset);
    spec->identity.display_name = iree_hal_device_spec_copy_string(
        params->identity->display_name, string_storage, &string_offset);
    spec->identity.driver_id = iree_hal_device_spec_copy_string(
        params->identity->driver_id, string_storage, &string_offset);
    spec->identity.driver_version = iree_hal_device_spec_copy_string(
        params->identity->driver_version, string_storage, &string_offset);
    spec->identity.backend_id = iree_hal_device_spec_copy_string(
        params->identity->backend_id, string_storage, &string_offset);
    spec->identity.device_path = iree_hal_device_spec_copy_string(
        params->identity->device_path, string_storage, &string_offset);
    spec->identity.vendor_name = iree_hal_device_spec_copy_string(
        params->identity->vendor_name, string_storage, &string_offset);
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->identity->physical_device_count,
        sizeof(*spec->physical_devices), params->identity->physical_devices,
        (void**)&spec->physical_devices);
    spec->identity.physical_devices = spec->physical_devices;
    for (iree_host_size_t i = 0;
         i < spec->identity.physical_device_count && iree_status_is_ok(status);
         ++i) {
      spec->physical_devices[i].identity.display_name =
          iree_hal_device_spec_copy_string(
              params->identity->physical_devices[i].identity.display_name,
              string_storage, &string_offset);
      spec->physical_devices[i].identity.backend_path =
          iree_hal_device_spec_copy_string(
              params->identity->physical_devices[i].identity.backend_path,
              string_storage, &string_offset);
    }
  }
  if (iree_status_is_ok(status) && params && params->memory) {
    spec->memory = *params->memory;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->memory->heap_count, sizeof(*spec->memory_heaps),
        params->memory->heaps, (void**)&spec->memory_heaps);
    spec->memory.heaps = spec->memory_heaps;
    for (iree_host_size_t i = 0;
         i < spec->memory.heap_count && iree_status_is_ok(status); ++i) {
      spec->memory_heaps[i].name = iree_hal_device_spec_copy_string(
          params->memory->heaps[i].name, string_storage, &string_offset);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->memory->memory_type_count,
          sizeof(*spec->memory_types), params->memory->memory_types,
          (void**)&spec->memory_types);
    }
    spec->memory.memory_types = spec->memory_types;
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->memory->external_buffer_handle_count,
          sizeof(*spec->external_buffer_handles),
          params->memory->external_buffer_handles,
          (void**)&spec->external_buffer_handles);
    }
    spec->memory.external_buffer_handles = spec->external_buffer_handles;
  }
  if (iree_status_is_ok(status) && params && params->virtual_memory) {
    spec->virtual_memory = *params->virtual_memory;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->virtual_memory->class_count,
        sizeof(*spec->virtual_memory_classes), params->virtual_memory->classes,
        (void**)&spec->virtual_memory_classes);
    spec->virtual_memory.classes = spec->virtual_memory_classes;
  }
  if (iree_status_is_ok(status) && params && params->queues) {
    spec->queues = *params->queues;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->queues->family_count,
        sizeof(*spec->queue_families), params->queues->families,
        (void**)&spec->queue_families);
    spec->queues.families = spec->queue_families;
    for (iree_host_size_t i = 0;
         i < spec->queues.family_count && iree_status_is_ok(status); ++i) {
      spec->queue_families[i].name = iree_hal_device_spec_copy_string(
          params->queues->families[i].name, string_storage, &string_offset);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->queues->external_timepoint_handle_count,
          sizeof(*spec->external_timepoint_handles),
          params->queues->external_timepoint_handles,
          (void**)&spec->external_timepoint_handles);
    }
    spec->queues.external_timepoint_handles = spec->external_timepoint_handles;
  }
  if (iree_status_is_ok(status) && params && params->dispatch) {
    spec->dispatch = *params->dispatch;
  }
  if (iree_status_is_ok(status) && params && params->timing) {
    spec->timing = *params->timing;
  }
  if (iree_status_is_ok(status) && params && params->executables) {
    spec->executables = *params->executables;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->executables->target_count,
        sizeof(*spec->executable_targets), params->executables->targets,
        (void**)&spec->executable_targets);
    spec->executables.targets = spec->executable_targets;
    for (iree_host_size_t i = 0;
         i < spec->executables.target_count && iree_status_is_ok(status); ++i) {
      const iree_hal_executable_target_t* source =
          &params->executables->targets[i];
      iree_hal_executable_target_t* target = &spec->executable_targets[i];
      target->family = iree_hal_device_spec_copy_string(
          source->family, string_storage, &string_offset);
      target->target_key = iree_hal_device_spec_copy_string(
          source->target_key, string_storage, &string_offset);
    }
  }
  if (iree_status_is_ok(status) && params && params->sanitizer) {
    spec->sanitizer = *params->sanitizer;
  }
  if (iree_status_is_ok(status) && params && params->facet_count) {
    spec->facet_count = params->facet_count;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->facet_count, sizeof(*spec->facets),
        params->facets, (void**)&spec->facets);
    for (iree_host_size_t i = 0;
         i < params->facet_count && iree_status_is_ok(status); ++i) {
      spec->facets[i].schema_id = iree_hal_device_spec_copy_string(
          params->facets[i].schema_id, string_storage, &string_offset);
      spec->facets[i].payload = iree_hal_device_spec_copy_bytes(
          params->facets[i].payload, payload_storage, &payload_offset);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_compute_fingerprint(spec, &spec->fingerprint);
  }
  if (iree_status_is_ok(status)) {
    *out_spec = spec;
  } else {
    iree_hal_device_spec_destroy(spec);
  }

  return status;
}

IREE_API_EXPORT void iree_hal_device_spec_retain(iree_hal_device_spec_t* spec) {
  if (IREE_LIKELY(spec)) {
    iree_atomic_ref_count_inc(&spec->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_device_spec_release(
    iree_hal_device_spec_t* spec) {
  if (IREE_LIKELY(spec) && iree_atomic_ref_count_dec(&spec->ref_count) == 1) {
    iree_hal_device_spec_destroy(spec);
  }
}

IREE_API_EXPORT uint64_t
iree_hal_device_spec_fingerprint(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return spec->fingerprint;
}

static bool iree_hal_device_spec_mask_overlaps(uint64_t available_mask,
                                               uint64_t requested_mask) {
  return requested_mask == 0 || (available_mask & requested_mask) != 0;
}

static bool iree_hal_device_spec_all_bits_available(uint64_t available_bits,
                                                    uint64_t requested_bits) {
  return requested_bits == 0 ||
         (available_bits & requested_bits) == requested_bits;
}

IREE_API_EXPORT const iree_hal_device_identity_spec_t*
iree_hal_device_spec_identity(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->identity;
}

IREE_API_EXPORT const iree_hal_device_memory_spec_t*
iree_hal_device_spec_memory(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->memory;
}

IREE_API_EXPORT const iree_hal_external_buffer_handle_spec_t*
iree_hal_device_spec_find_external_buffer_handle(
    const iree_hal_device_spec_t* spec,
    const iree_hal_external_buffer_handle_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  for (iree_host_size_t i = 0; i < spec->memory.external_buffer_handle_count;
       ++i) {
    const iree_hal_external_buffer_handle_spec_t* handle =
        &spec->memory.external_buffer_handles[i];
    if (!iree_hal_device_spec_mask_overlaps(handle->handle_type_mask,
                                            selection->handle_type_mask)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->direction_flags,
                                                 selection->direction_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->allowed_buffer_usage,
                                                 selection->buffer_usage)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->allowed_memory_access,
                                                 selection->memory_access)) {
      continue;
    }
    if (!iree_hal_device_spec_mask_overlaps(
            handle->compatible_memory_type_mask,
            selection->compatible_memory_type_mask)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->flags,
                                                 selection->capability_flags)) {
      continue;
    }
    return handle;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_device_virtual_memory_spec_t*
iree_hal_device_spec_virtual_memory(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->virtual_memory;
}

IREE_API_EXPORT const iree_hal_virtual_memory_class_spec_t*
iree_hal_device_spec_find_virtual_memory_class(
    const iree_hal_device_spec_t* spec,
    const iree_hal_virtual_memory_class_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  for (iree_host_size_t i = 0; i < spec->virtual_memory.class_count; ++i) {
    const iree_hal_virtual_memory_class_spec_t* memory_class =
        &spec->virtual_memory.classes[i];
    if (!iree_hal_device_spec_mask_overlaps(
            memory_class->compatible_memory_type_mask,
            selection->compatible_memory_type_mask)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(
            memory_class->allowed_buffer_usage, selection->buffer_usage)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(
            memory_class->allowed_memory_access, selection->memory_access)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(memory_class->operation_flags,
                                                 selection->operation_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(memory_class->protection_flags,
                                                 selection->protection_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(memory_class->advice_flags,
                                                 selection->advice_flags)) {
      continue;
    }
    return memory_class;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_device_queue_spec_t* iree_hal_device_spec_queues(
    const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->queues;
}

IREE_API_EXPORT const iree_hal_external_timepoint_handle_spec_t*
iree_hal_device_spec_find_external_timepoint_handle(
    const iree_hal_device_spec_t* spec,
    const iree_hal_external_timepoint_handle_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  for (iree_host_size_t i = 0; i < spec->queues.external_timepoint_handle_count;
       ++i) {
    const iree_hal_external_timepoint_handle_spec_t* handle =
        &spec->queues.external_timepoint_handles[i];
    if (selection->handle_type != IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_NONE &&
        handle->handle_type != selection->handle_type) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->direction_flags,
                                                 selection->direction_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->compatibility,
                                                 selection->compatibility)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->flags,
                                                 selection->capability_flags)) {
      continue;
    }
    return handle;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_device_dispatch_spec_t*
iree_hal_device_spec_dispatch(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->dispatch;
}

IREE_API_EXPORT const iree_hal_device_timing_spec_t*
iree_hal_device_spec_timing(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->timing;
}

IREE_API_EXPORT const iree_hal_device_executable_spec_t*
iree_hal_device_spec_executables(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->executables;
}

IREE_API_EXPORT const iree_hal_device_sanitizer_spec_t*
iree_hal_device_spec_sanitizer(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->sanitizer;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_device_spec_facet_count(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return spec->facet_count;
}

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_device_spec_facet_at(const iree_hal_device_spec_t* spec,
                              iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(spec);
  return index < spec->facet_count ? &spec->facets[index] : NULL;
}

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_device_spec_find_facet(const iree_hal_device_spec_t* spec,
                                iree_string_view_t schema_id) {
  IREE_ASSERT_ARGUMENT(spec);
  for (iree_host_size_t i = 0; i < spec->facet_count; ++i) {
    if (iree_string_view_equal(spec->facets[i].schema_id, schema_id)) {
      return &spec->facets[i];
    }
  }
  return NULL;
}

static bool iree_hal_device_spec_selection_string_matches(
    iree_string_view_t filter, iree_string_view_t value) {
  return iree_string_view_is_empty(filter) ||
         iree_string_view_equal(filter, value);
}

static bool iree_hal_device_spec_target_matches_selection(
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_target_selection_t* selection) {
  if (!iree_hal_device_spec_selection_string_matches(selection->family,
                                                     target->family)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->target_key,
                                                     target->target_key)) {
    return false;
  }
  const iree_hal_executable_target_kind_flags_t target_kind_flag =
      1u << target->kind;
  if (selection->kind_flags != 0 &&
      !iree_any_bit_set(selection->kind_flags, target_kind_flag)) {
    return false;
  }
  if (selection->physical_device_affinity &&
      !iree_all_bits_set(target->physical_device_affinity,
                         selection->physical_device_affinity)) {
    return false;
  }
  return true;
}

IREE_API_EXPORT iree_host_size_t iree_hal_device_spec_executable_target_ordinal(
    const iree_hal_device_spec_t* spec,
    const iree_hal_executable_target_t* target) {
  IREE_ASSERT_ARGUMENT(spec);
  for (iree_host_size_t i = 0; target && i < spec->executables.target_count;
       ++i) {
    if (target == &spec->executables.targets[i]) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

IREE_API_EXPORT iree_hal_executable_target_selection_result_t
iree_hal_device_spec_select_executable_target(
    const iree_hal_device_spec_t* spec,
    const iree_hal_executable_target_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);

  const iree_hal_executable_target_t* selected_target = NULL;
  iree_host_size_t selected_ordinal = IREE_HOST_SIZE_MAX;
  iree_hal_executable_target_selection_result_t result = {
      .outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH,
      .target = NULL,
      .target_ordinal = IREE_HOST_SIZE_MAX};
  for (iree_host_size_t i = 0; i < spec->executables.target_count; ++i) {
    const iree_hal_executable_target_t* candidate =
        &spec->executables.targets[i];
    if (!iree_hal_device_spec_target_matches_selection(candidate, selection)) {
      continue;
    }
    if (!selected_target || candidate->priority > selected_target->priority) {
      result.outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED;
      selected_target = candidate;
      selected_ordinal = i;
    } else if (candidate->priority == selected_target->priority) {
      result.outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS;
    }
  }
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    result.target = selected_target;
    result.target_ordinal = selected_ordinal;
  }
  return result;
}
