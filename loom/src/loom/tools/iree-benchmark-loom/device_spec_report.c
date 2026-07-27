// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/device_spec_report.h"

#include <inttypes.h>

#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/util/json.h"

static iree_status_t iree_benchmark_loom_write_hex_bytes_json(
    const uint8_t* bytes, iree_host_size_t byte_count,
    loom_output_stream_t* stream) {
  static const char kHexDigits[] = "0123456789abcdef";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\""));
  for (iree_host_size_t i = 0; i < byte_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_char(stream, kHexDigits[bytes[i] >> 4]));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_char(stream, kHexDigits[bytes[i] & 0x0F]));
  }
  return loom_output_stream_write_cstring(stream, "\"");
}

static iree_status_t iree_benchmark_loom_write_u32_array3_json(
    const uint32_t values[3], loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < 3; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_uint32_element(&array, values[i]));
  }
  return loom_json_array_end(&array);
}

static iree_status_t iree_benchmark_loom_write_device_spec_identity_json(
    const iree_hal_device_identity_spec_t* identity,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("logical_device_id"), identity->logical_device_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("display_name"), identity->display_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("driver_id"), identity->driver_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("driver_version"), identity->driver_version));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("backend_id"), identity->backend_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("device_path"), identity->device_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("vendor_name"), identity->vendor_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vendor_id"), identity->vendor_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("device_id"), identity->device_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("revision_id"), identity->revision_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("logical_ordinal"), identity->logical_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("physical_device_count"),
      identity->physical_device_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), identity->flags));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_physical_device_spec_json(
    const iree_hal_physical_device_spec_t* physical_device,
    loom_output_stream_t* stream) {
  const iree_hal_physical_device_identity_t* identity =
      &physical_device->identity;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("physical_ordinal"), physical_device->physical_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("partition_ordinal"),
                                          physical_device->partition_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("partition_count"), physical_device->partition_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("physical_device_affinity"),
      physical_device->physical_device_affinity));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("display_name"), identity->display_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("backend_path"), identity->backend_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vendor_id"), identity->vendor_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("device_id"), identity->device_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("revision_id"), identity->revision_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), identity->flags));
  if (iree_all_bits_set(identity->flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("uuid")));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hex_bytes_json(
        identity->uuid.bytes, IREE_ARRAYSIZE(identity->uuid.bytes), stream));
  }
  if (iree_all_bits_set(identity->flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("pci")));
    loom_json_object_writer_t pci_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &pci_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &pci_object, IREE_SV("domain"), identity->pci.domain));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &pci_object, IREE_SV("bus"), identity->pci.bus));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &pci_object, IREE_SV("device"), identity->pci.device));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &pci_object, IREE_SV("function"), identity->pci.function));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&pci_object));
  }
  if (iree_all_bits_set(identity->flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("numa")));
    loom_json_object_writer_t numa_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &numa_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &numa_object, IREE_SV("node_id"), identity->numa.node_id));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&numa_object));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_physical_device_specs_json(
    const iree_hal_device_identity_spec_t* identity,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_physical_device_spec_json(
        &identity->physical_devices[i], stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t iree_benchmark_loom_write_device_spec_memory_json(
    const iree_hal_device_memory_spec_t* memory, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("heap_count"), memory->heap_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("memory_type_count"), memory->memory_type_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("external_buffer_handle_count"),
      memory->external_buffer_handle_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), memory->flags));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_device_spec_virtual_memory_json(
    const iree_hal_device_virtual_memory_spec_t* virtual_memory,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("class_count"), virtual_memory->class_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), virtual_memory->flags));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_device_spec_queues_json(
    const iree_hal_device_queue_spec_t* queues, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("family_count"), queues->family_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("external_timepoint_handle_count"),
      queues->external_timepoint_handle_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), queues->flags));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_device_spec_dispatch_json(
    const iree_hal_device_dispatch_spec_t* dispatch,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("launch")));
  loom_json_object_writer_t launch_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &launch_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &launch_object, IREE_SV("maximum_workgroup_invocations"),
      dispatch->launch.maximum_workgroup_invocations));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
      &launch_object, IREE_SV("maximum_workgroup_size")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_u32_array3_json(
      dispatch->launch.maximum_workgroup_size, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
      &launch_object, IREE_SV("maximum_workgroup_count")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_u32_array3_json(
      dispatch->launch.maximum_workgroup_count, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&launch_object));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("subgroup")));
  loom_json_object_writer_t subgroup_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &subgroup_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &subgroup_object, IREE_SV("default_size"),
      dispatch->subgroup.default_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &subgroup_object, IREE_SV("minimum_size"),
      dispatch->subgroup.minimum_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &subgroup_object, IREE_SV("maximum_size"),
      dispatch->subgroup.maximum_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &subgroup_object, IREE_SV("supported_size_mask"),
      dispatch->subgroup.supported_size_mask));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&subgroup_object));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("execution")));
  loom_json_object_writer_t execution_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &execution_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("unit_count"),
      dispatch->execution.unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("group_count"),
      dispatch->execution.group_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("maximum_resident_workgroup_count"),
      dispatch->execution.maximum_resident_workgroup_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("maximum_resident_invocation_count"),
      dispatch->execution.maximum_resident_invocation_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("maximum_resident_subgroup_count"),
      dispatch->execution.maximum_resident_subgroup_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("maximum_register_count"),
      dispatch->execution.maximum_register_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &execution_object, IREE_SV("maximum_workgroup_register_count"),
      dispatch->execution.maximum_workgroup_register_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &execution_object, IREE_SV("maximum_local_memory_size"),
      dispatch->execution.maximum_local_memory_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &execution_object, IREE_SV("maximum_workgroup_local_memory_size"),
      dispatch->execution.maximum_workgroup_local_memory_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &execution_object, IREE_SV("maximum_workgroup_local_memory_size_optin"),
      dispatch->execution.maximum_workgroup_local_memory_size_optin));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&execution_object));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("addressing")));
  loom_json_object_writer_t addressing_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &addressing_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &addressing_object, IREE_SV("pointer_size_bits"),
      dispatch->addressing.pointer_size_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &addressing_object, IREE_SV("address_space_bits"),
      dispatch->addressing.address_space_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &addressing_object, IREE_SV("minimum_buffer_device_address_alignment"),
      dispatch->addressing.minimum_buffer_device_address_alignment));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&addressing_object));

  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), dispatch->flags));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_device_spec_timing_json(
    const iree_hal_device_timing_spec_t* timing, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("timestamp_valid_bits"), timing->timestamp_valid_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("timestamp_frequency_hz"),
      timing->timestamp_frequency_hz));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), timing->flags));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_executable_targets_json(
    const iree_hal_device_executable_spec_t* executables,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
    const iree_hal_executable_target_t* target = &executables->targets[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("family"), target->family));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("target_key"), target->target_key));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("kind"), target->kind));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("priority"), target->priority));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("physical_device_affinity"),
        target->physical_device_affinity));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("flags"), target->flags));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t iree_benchmark_loom_write_device_spec_executables_json(
    const iree_hal_device_executable_spec_t* executables,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("target_count"), executables->target_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), executables->flags));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("targets")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_executable_targets_json(executables, stream));
  return loom_json_object_end(&object);
}

iree_status_t iree_benchmark_loom_write_device_spec_json(
    const iree_hal_device_spec_t* device_spec, loom_output_stream_t* stream) {
  if (device_spec == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL device does not expose an immutable device spec");
  }
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  const iree_hal_device_virtual_memory_spec_t* virtual_memory =
      iree_hal_device_spec_virtual_memory(device_spec);
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("fingerprint")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"0x%016" PRIx64 "\"",
      iree_hal_device_spec_fingerprint(device_spec)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("identity")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_identity_json(identity, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("physical_devices")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_physical_device_specs_json(identity, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("memory")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_memory_json(memory, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("virtual_memory")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_virtual_memory_json(virtual_memory,
                                                                stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("queues")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_queues_json(queues, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("dispatch")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_dispatch_json(dispatch, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("timing")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_timing_json(timing, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("executables")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_device_spec_executables_json(
      executables, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("facet_count"),
      iree_hal_device_spec_facet_count(device_spec)));
  return loom_json_object_end(&object);
}
