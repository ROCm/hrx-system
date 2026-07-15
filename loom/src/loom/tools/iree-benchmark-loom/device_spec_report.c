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
  return loom_output_stream_write_format(
      stream, "[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "]", values[0], values[1],
      values[2]);
}

static iree_status_t iree_benchmark_loom_write_device_spec_identity_json(
    const iree_hal_device_identity_spec_t* identity,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "logical_device_id", identity->logical_device_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "display_name", identity->display_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "driver_id", identity->driver_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "driver_version", identity->driver_version));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "backend_id", identity->backend_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "device_path", identity->device_path));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "vendor_name", identity->vendor_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "vendor_id", identity->vendor_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "device_id", identity->device_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "revision_id", identity->revision_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "logical_ordinal", identity->logical_ordinal));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "physical_device_count",
      identity->physical_device_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", identity->flags));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_physical_device_spec_json(
    const iree_hal_physical_device_spec_t* physical_device,
    loom_output_stream_t* stream) {
  const iree_hal_physical_device_identity_t* identity =
      &physical_device->identity;
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "physical_ordinal",
      physical_device->physical_ordinal));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "partition_ordinal",
      physical_device->partition_ordinal));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "partition_count",
      physical_device->partition_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_field, "physical_device_affinity",
      physical_device->physical_device_affinity));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "display_name", identity->display_name));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_field, "backend_path", identity->backend_path));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "vendor_id", identity->vendor_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "device_id", identity->device_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "revision_id", identity->revision_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", identity->flags));
  if (iree_all_bits_set(identity->flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "uuid"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hex_bytes_json(
        identity->uuid.bytes, IREE_ARRAYSIZE(identity->uuid.bytes), stream));
  }
  if (iree_all_bits_set(identity->flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "pci"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        "{\"domain\":%" PRIu32 ",\"bus\":%" PRIu32 ",\"device\":%" PRIu32
        ",\"function\":%" PRIu32 "}",
        identity->pci.domain, identity->pci.bus, identity->pci.device,
        identity->pci.function));
  }
  if (iree_all_bits_set(identity->flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "numa"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"node_id\":%" PRIu32 "}", identity->numa.node_id));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_physical_device_specs_json(
    const iree_hal_device_identity_spec_t* identity,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_physical_device_spec_json(
        &identity->physical_devices[i], stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t iree_benchmark_loom_write_device_spec_memory_json(
    const iree_hal_device_memory_spec_t* memory, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "heap_count", memory->heap_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "memory_type_count", memory->memory_type_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "external_buffer_handle_count",
      memory->external_buffer_handle_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", memory->flags));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_device_spec_virtual_memory_json(
    const iree_hal_device_virtual_memory_spec_t* virtual_memory,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "class_count", virtual_memory->class_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", virtual_memory->flags));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_device_spec_queues_json(
    const iree_hal_device_queue_spec_t* queues, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "family_count", queues->family_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "external_timepoint_handle_count",
      queues->external_timepoint_handle_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", queues->flags));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_device_spec_dispatch_json(
    const iree_hal_device_dispatch_spec_t* dispatch,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "launch"));
  bool first_launch_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_launch_field, "maximum_workgroup_invocations",
      dispatch->launch.maximum_workgroup_invocations));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_launch_field, "maximum_workgroup_size"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_u32_array3_json(
      dispatch->launch.maximum_workgroup_size, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_launch_field, "maximum_workgroup_count"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_u32_array3_json(
      dispatch->launch.maximum_workgroup_count, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "subgroup"));
  bool first_subgroup_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_subgroup_field, "default_size",
      dispatch->subgroup.default_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_subgroup_field, "minimum_size",
      dispatch->subgroup.minimum_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_subgroup_field, "maximum_size",
      dispatch->subgroup.maximum_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_subgroup_field, "supported_size_mask",
      dispatch->subgroup.supported_size_mask));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "execution"));
  bool first_execution_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "unit_count",
      dispatch->execution.unit_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "group_count",
      dispatch->execution.group_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "maximum_resident_workgroup_count",
      dispatch->execution.maximum_resident_workgroup_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "maximum_resident_invocation_count",
      dispatch->execution.maximum_resident_invocation_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "maximum_resident_subgroup_count",
      dispatch->execution.maximum_resident_subgroup_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "maximum_register_count",
      dispatch->execution.maximum_register_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_execution_field, "maximum_workgroup_register_count",
      dispatch->execution.maximum_workgroup_register_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_execution_field, "maximum_local_memory_size",
      dispatch->execution.maximum_local_memory_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_execution_field, "maximum_workgroup_local_memory_size",
      dispatch->execution.maximum_workgroup_local_memory_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_execution_field,
      "maximum_workgroup_local_memory_size_optin",
      dispatch->execution.maximum_workgroup_local_memory_size_optin));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "addressing"));
  bool first_addressing_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_addressing_field, "pointer_size_bits",
      dispatch->addressing.pointer_size_bits));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_addressing_field, "address_space_bits",
      dispatch->addressing.address_space_bits));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_addressing_field,
      "minimum_buffer_device_address_alignment",
      dispatch->addressing.minimum_buffer_device_address_alignment));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", dispatch->flags));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_device_spec_timing_json(
    const iree_hal_device_timing_spec_t* timing, loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "timestamp_valid_bits",
      timing->timestamp_valid_bits));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_field, "timestamp_frequency_hz",
      timing->timestamp_frequency_hz));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", timing->flags));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_executable_formats_json(
    const iree_hal_device_executable_spec_t* executables,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < executables->format_count; ++i) {
    const iree_hal_executable_format_spec_t* format = &executables->formats[i];
    bool first_field = true;
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_field, "format", format->format));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
        stream, &first_field, "caching_modes", format->caching_modes));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
        stream, &first_field, "flags", format->flags));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t iree_benchmark_loom_write_executable_targets_json(
    const iree_hal_device_executable_spec_t* executables,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
    const iree_hal_executable_target_t* target = &executables->targets[i];
    bool first_field = true;
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_field, "family", target->family));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_field, "target_key", target->target_key));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
        stream, &first_field, "kind", target->kind));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
        stream, &first_field, "priority", target->priority));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "physical_device_affinity",
        target->physical_device_affinity));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
        stream, &first_field, "flags", target->flags));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t iree_benchmark_loom_write_device_spec_executables_json(
    const iree_hal_device_executable_spec_t* executables,
    loom_output_stream_t* stream) {
  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "format_count", executables->format_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "target_count", executables->target_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "flags", executables->flags));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "formats"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_executable_formats_json(executables, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "targets"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_executable_targets_json(executables, stream));
  return loom_output_stream_write_cstring(stream, "}");
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

  bool first_field = true;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "fingerprint"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"0x%016" PRIx64 "\"",
      iree_hal_device_spec_fingerprint(device_spec)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "identity"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_identity_json(identity, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "physical_devices"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_physical_device_specs_json(identity, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "memory"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_memory_json(memory, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "virtual_memory"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_virtual_memory_json(virtual_memory,
                                                                stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "queues"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_queues_json(queues, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "dispatch"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_dispatch_json(dispatch, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "timing"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_device_spec_timing_json(timing, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "executables"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_device_spec_executables_json(
      executables, stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "facet_count",
      iree_hal_device_spec_facet_count(device_spec)));
  return loom_output_stream_write_cstring(stream, "}");
}
