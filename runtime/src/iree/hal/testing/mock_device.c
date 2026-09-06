// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/testing/mock_device.h"

#include <string.h>

#include "iree/hal/utils/device_spec_builder.h"

//===----------------------------------------------------------------------===//
// Mock executable support
//===----------------------------------------------------------------------===//

typedef struct iree_hal_mock_executable_function_record_t {
  // Number of 32-bit constant words reflected for the function.
  uint8_t constant_count;
  // Number of buffer bindings reflected for the function.
  uint8_t binding_count;
  // Executable function flags byte.
  uint8_t flags;
  // Static workgroup size reflected for the function.
  uint8_t workgroup_size[3];
  // Byte length of the function name in the trailing name storage.
  uint8_t name_length;
  // Native ABI byte offset for one optional reflected buffer binding.
  uint8_t native_abi_offset;
  // Native ABI byte size for the optional reflected buffer binding.
  uint16_t parameter_size;
} iree_hal_mock_executable_function_record_t;

static_assert(sizeof(iree_hal_mock_executable_function_record_t) == 10,
              "mock executable metadata must have a stable byte layout");

typedef struct iree_hal_mock_executable_parameter_metadata_t {
  // Native ABI byte offset for the reflected buffer binding.
  uint8_t native_abi_offset;
  // Native ABI byte size for the reflected buffer binding.
  uint16_t size;
} iree_hal_mock_executable_parameter_metadata_t;

typedef struct iree_hal_mock_executable_t {
  // Common HAL executable state.
  iree_hal_executable_t base;
  // Host allocator used for executable-owned storage.
  iree_allocator_t host_allocator;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Function metadata records indexed by executable function ordinal.
  iree_hal_executable_function_info_t functions[];
} iree_hal_mock_executable_t;

static iree_hal_mock_executable_parameter_metadata_t*
iree_hal_mock_executable_parameter_metadata(
    iree_hal_mock_executable_t* executable) {
  return (
      iree_hal_mock_executable_parameter_metadata_t*)(executable->functions +
                                                      executable
                                                          ->function_count);
}

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable;

static iree_hal_mock_executable_t* iree_hal_mock_executable_cast(
    iree_hal_executable_t* base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable, &iree_hal_mock_executable_vtable);
  return (iree_hal_mock_executable_t*)base_executable;
}

static iree_status_t iree_hal_mock_executable_create(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  if (IREE_UNLIKELY(load_params->executable_data.data_length <
                    sizeof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mock executable data is too short");
  }
  uint32_t function_count = 0;
  memcpy(&function_count, load_params->executable_data.data,
         sizeof(function_count));
  iree_const_byte_span_t function_data = iree_make_const_byte_span(
      load_params->executable_data.data + sizeof(function_count),
      load_params->executable_data.data_length - sizeof(function_count));
  iree_host_size_t function_record_data_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        function_count,
                        sizeof(iree_hal_mock_executable_function_record_t),
                        &function_record_data_length) ||
                    function_record_data_length > function_data.data_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mock executable function metadata length mismatch");
  }
  iree_const_byte_span_t name_storage = iree_make_const_byte_span(
      function_data.data + function_record_data_length,
      function_data.data_length - function_record_data_length);

  const iree_hal_mock_executable_function_record_t* function_records =
      (const iree_hal_mock_executable_function_record_t*)function_data.data;
  iree_host_size_t expected_name_storage_length = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_mock_executable_function_record_t* record =
        &function_records[i];
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            expected_name_storage_length, record->name_length,
            &expected_name_storage_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock executable function name storage overflow");
    }
  }
  if (IREE_UNLIKELY(expected_name_storage_length != name_storage.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mock executable function name length mismatch");
  }

  iree_host_size_t function_info_size = 0;
  iree_host_size_t parameter_metadata_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              function_count, sizeof(iree_hal_executable_function_info_t),
              &function_info_size) ||
          !iree_host_size_checked_mul(
              function_count,
              sizeof(iree_hal_mock_executable_parameter_metadata_t),
              &parameter_metadata_size) ||
          !iree_host_size_checked_add(sizeof(iree_hal_mock_executable_t),
                                      function_info_size, &total_size) ||
          !iree_host_size_checked_add(total_size, parameter_metadata_size,
                                      &total_size) ||
          !iree_host_size_checked_add(total_size, name_storage.data_length,
                                      &total_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mock executable metadata is too large");
  }
  iree_hal_mock_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_executable_initialize(queue_family, &iree_hal_mock_executable_vtable,
                                 &executable->base);
  executable->host_allocator = host_allocator;
  executable->function_count = function_count;

  char* executable_name_storage = (char*)executable + sizeof(*executable) +
                                  function_info_size + parameter_metadata_size;
  if (name_storage.data_length != 0) {
    memcpy(executable_name_storage, name_storage.data,
           name_storage.data_length);
  }
  iree_host_size_t name_offset = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_mock_executable_function_record_t* record =
        &function_records[i];
    executable->functions[i].name = iree_make_string_view(
        executable_name_storage + name_offset, record->name_length);
    name_offset += record->name_length;
    executable->functions[i].flags = record->flags;
    executable->functions[i].constant_byte_length =
        record->constant_count * sizeof(uint32_t);
    executable->functions[i].binding_count = record->binding_count;
    executable->functions[i].parameter_count = record->parameter_size ? 1 : 0;
    executable->functions[i].workgroup_size[0] = record->workgroup_size[0];
    executable->functions[i].workgroup_size[1] = record->workgroup_size[1];
    executable->functions[i].workgroup_size[2] = record->workgroup_size[2];
    iree_hal_mock_executable_parameter_metadata(executable)[i] =
        (iree_hal_mock_executable_parameter_metadata_t){
            .native_abi_offset = record->native_abi_offset,
            .size = record->parameter_size,
        };
  }

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static void iree_hal_mock_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_mock_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t iree_hal_mock_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  const uint32_t function_ordinal =
      iree_hal_executable_function_index(function);
  *out_info = executable->functions[function_ordinal];
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  const uint32_t function_ordinal =
      iree_hal_executable_function_index(function);
  const iree_hal_mock_executable_parameter_metadata_t* parameter_metadata =
      &iree_hal_mock_executable_parameter_metadata(
          executable)[function_ordinal];
  if (!parameter_metadata->size) return iree_ok_status();
  if (capacity < 1 || !out_parameters) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mock executable parameter output too small");
  }
  out_parameters[0] = (iree_hal_executable_function_parameter_t){
      .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
      .flags = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET,
      .size = parameter_metadata->size,
      .offset = 0,
      .native_abi_offset = parameter_metadata->native_abi_offset,
      .name = iree_string_view_empty(),
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->functions[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

static iree_status_t iree_hal_mock_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
}

static iree_status_t iree_hal_mock_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)out_buffer;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
}

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable = {
    .destroy = iree_hal_mock_executable_destroy,
    .function_count = iree_hal_mock_executable_function_count,
    .function_info = iree_hal_mock_executable_function_info,
    .function_parameters = iree_hal_mock_executable_function_parameters,
    .lookup_function_by_name = iree_hal_mock_executable_lookup_function_by_name,
    .try_lookup_global_by_name =
        iree_hal_mock_executable_try_lookup_global_by_name,
    .global_info = iree_hal_mock_executable_global_info,
    .global_buffer = iree_hal_mock_executable_global_buffer,
};

//===----------------------------------------------------------------------===//
// iree_hal_mock_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_mock_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Identifier string (backed by trailing storage).
  iree_string_view_t identifier;

  // Status returned by assign_topology_info.
  iree_status_code_t assign_topology_info_status_code;

  // True when metadata-only mock executable loading is enabled.
  bool executable_loading_enabled;

  // Immutable device facts captured at creation time.
  iree_hal_device_spec_t* device_spec;

  // Number of queue family identities exposed by the device spec.
  iree_host_size_t queue_family_count;

  // Queue family identities indexed by canonical family ordinal.
  iree_hal_queue_family_t* queue_families;

  // Topology information assigned during group creation.
  iree_hal_device_topology_info_t topology_info;
} iree_hal_mock_device_t;

static const iree_hal_device_vtable_t iree_hal_mock_device_vtable;

static iree_hal_mock_device_t* iree_hal_mock_device_cast(
    iree_hal_device_t* base_device) {
  IREE_HAL_ASSERT_TYPE(base_device, &iree_hal_mock_device_vtable);
  return (iree_hal_mock_device_t*)base_device;
}

void iree_hal_mock_device_options_initialize(
    iree_hal_mock_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

static iree_status_t iree_hal_mock_device_default_spec_create(
    const iree_hal_mock_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  const iree_hal_physical_device_spec_t physical_device = {
      .identity =
          {
              .display_name = options->identifier,
              .backend_path = options->identifier,
          },
      .partition_count = 1,
      .physical_device_affinity = 1ull,
  };
  const iree_hal_device_identity_spec_t identity = {
      .logical_device_id = options->identifier,
      .display_name = options->identifier,
      .driver_id = IREE_SV("mock"),
      .backend_id = IREE_SV("mock"),
      .physical_device_count = 1,
      .physical_devices = &physical_device,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  const iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("dispatch"),
      .provisioned_queue_count = 0,
      .priority_count = 1,
      .physical_device_affinity = 1ull,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
  };
  const iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &queue_family,
  };
  const iree_hal_executable_target_t executable_target = {
      .family = IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_FAMILY),
      .target_key = IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_KEY),
      .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_VIRTUAL,
      .priority = 100,
      .physical_device_affinity = 1ull,
      .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(&builder, &identity);
  if (iree_status_is_ok(status) && options->executable_loading_enabled) {
    status = iree_hal_device_spec_builder_set_queues(&builder, &queues);
  }
  if (iree_status_is_ok(status) && options->executable_loading_enabled) {
    status = iree_hal_device_spec_builder_add_executable_target(
        &builder, &executable_target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

iree_status_t iree_hal_mock_device_create(
    const iree_hal_mock_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  iree_hal_mock_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + options->identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, sizeof(*device));
  iree_hal_resource_initialize(&iree_hal_mock_device_vtable, &device->resource);
  device->host_allocator = host_allocator;
  device->assign_topology_info_status_code =
      options->assign_topology_info_status_code;
  device->executable_loading_enabled = options->executable_loading_enabled;

  // Copy identifier into trailing storage.
  iree_string_view_append_to_buffer(
      options->identifier, &device->identifier,
      (char*)device + total_size - options->identifier.size);

  iree_status_t status = iree_ok_status();
  if (options->device_spec) {
    device->device_spec = options->device_spec;
    iree_hal_device_spec_retain(device->device_spec);
  } else {
    status = iree_hal_mock_device_default_spec_create(options, host_allocator,
                                                      &device->device_spec);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_device_queue_spec_t* queues =
        iree_hal_device_spec_queues(device->device_spec);
    for (iree_host_size_t i = 0;
         queues && i < queues->family_count && iree_status_is_ok(status); ++i) {
      if (IREE_UNLIKELY(queues->families[i].provisioned_queue_count != 0)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "mock device queue family %" PRIhsz
                                  " cannot expose provisioned queues",
                                  i);
      }
    }
    if (queues && queues->family_count != 0 && iree_status_is_ok(status)) {
      status = iree_allocator_malloc(
          host_allocator,
          queues->family_count * sizeof(*device->queue_families),
          (void**)&device->queue_families);
      if (iree_status_is_ok(status)) {
        device->queue_family_count = queues->family_count;
        for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
          iree_hal_queue_family_initialize((iree_hal_queue_family_ordinal_t)i,
                                           &device->queue_families[i]);
        }
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_device_release((iree_hal_device_t*)device);
    return status;
  }

  *out_device = (iree_hal_device_t*)device;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Implemented vtable methods
//===----------------------------------------------------------------------===//

static void iree_hal_mock_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  iree_hal_device_spec_release(device->device_spec);
  iree_allocator_free(host_allocator, device->queue_families);
  iree_allocator_free(host_allocator, device);
}

static iree_string_view_t iree_hal_mock_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_mock_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return device->host_allocator;
}

static const iree_hal_device_spec_t* iree_hal_mock_device_spec(
    iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return device->device_spec;
}

static const iree_hal_queue_family_t* iree_hal_mock_device_queue_family(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return family_ordinal < device->queue_family_count
             ? &device->queue_families[family_ordinal]
             : NULL;
}

static iree_hal_queue_t* iree_hal_mock_device_queue(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal) {
  return NULL;
}

static iree_status_t iree_hal_mock_device_sample_observation(
    iree_hal_device_t* base_device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  if (iree_any_bit_set(requested_flags,
                       IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_observation_populate_memory_total_from_spec(
            device->device_spec, out_observation));
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_mock_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_mock_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  // No refinement; tests observe the common spec-derived projection as-is.
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  if (!topology_info) {
    memset(&device->topology_info, 0, sizeof(device->topology_info));
    return iree_ok_status();
  } else if (device->assign_topology_info_status_code != IREE_STATUS_OK) {
    return iree_make_status(device->assign_topology_info_status_code);
  }
  device->topology_info = *topology_info;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stub vtable methods (UNIMPLEMENTED)
//===----------------------------------------------------------------------===//
//
// These exist only to fill the vtable. Any call through them is a test bug.

static iree_hal_allocator_t* iree_hal_mock_device_allocator(
    iree_hal_device_t* base_device) {
  return NULL;
}

static void iree_hal_mock_device_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
}

static iree_status_t iree_hal_mock_device_trim(iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_channel(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_command_buffer(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_load_executable(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_mock_device_t* device = iree_hal_mock_device_cast(base_device);
  if (!device->executable_loading_enabled) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "mock executable loading is not enabled");
  }
  if (!iree_string_view_equal(
          target->family, IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_FAMILY)) ||
      !iree_string_view_equal(target->target_key,
                              IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_KEY)) ||
      target->kind != IREE_HAL_EXECUTABLE_TARGET_KIND_VIRTUAL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported mock executable target `%.*s:%.*s`",
                            (int)target->family.size, target->family.data,
                            (int)target->target_key.size,
                            target->target_key.data);
  }
  return iree_hal_mock_executable_create(
      queue_family, load_params, device->host_allocator, out_executable);
}

static iree_status_t iree_hal_mock_device_import_file(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_create_semaphore(
    iree_hal_device_t* base_device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_hal_semaphore_compatibility_t
iree_hal_mock_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_mock_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_pool_backend_t* out_backend) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_mock_device_profiling_end(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_mock_device_vtable = {
    .destroy = iree_hal_mock_device_destroy,
    .id = iree_hal_mock_device_id,
    .host_allocator = iree_hal_mock_device_host_allocator,
    .device_allocator = iree_hal_mock_device_allocator,
    .replace_channel_provider = iree_hal_mock_device_replace_channel_provider,
    .trim = iree_hal_mock_device_trim,
    .device_spec = iree_hal_mock_device_spec,
    .queue_family = iree_hal_mock_device_queue_family,
    .queue = iree_hal_mock_device_queue,
    .sample_observation = iree_hal_mock_device_sample_observation,
    .topology_info = iree_hal_mock_device_topology_info,
    .refine_topology_edge = iree_hal_mock_device_refine_topology_edge,
    .assign_topology_info = iree_hal_mock_device_assign_topology_info,
    .create_channel = iree_hal_mock_device_create_channel,
    .create_command_buffer = iree_hal_mock_device_create_command_buffer,
    .load_executable = iree_hal_mock_device_load_executable,
    .import_file = iree_hal_mock_device_import_file,
    .create_semaphore = iree_hal_mock_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_mock_device_query_semaphore_compatibility,
    .query_queue_pool_backend = iree_hal_mock_device_query_queue_pool_backend,
    .profiling_begin = iree_hal_mock_device_profiling_begin,
    .profiling_flush = iree_hal_mock_device_profiling_flush,
    .profiling_end = iree_hal_mock_device_profiling_end,
};
