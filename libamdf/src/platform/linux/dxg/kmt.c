// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dxg/kmt.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>

#include "libamdf/src/allocator.h"

_Static_assert(sizeof(wchar_t) == 4,
               "Linux DXG conversion requires 32-bit wchar_t");

// Windows wire strings use UTF-16 regardless of the host wchar_t layout.
typedef uint16_t amdf_dxg_utf16_t;

typedef struct amdf_dxg_umd_filename_info_wire_t {
  // Requested user-mode driver interface version.
  KMTUMDVERSION version;
  // Returned Windows UTF-16 driver path.
  amdf_dxg_utf16_t file_name[MAX_PATH];
} amdf_dxg_umd_filename_info_wire_t;

typedef struct amdf_dxg_opengl_info_wire_t {
  // Returned Windows UTF-16 OpenGL driver path.
  amdf_dxg_utf16_t file_name[MAX_PATH];
  // OpenGL driver interface version.
  ULONG version;
  // OpenGL driver property bits.
  ULONG flags;
} amdf_dxg_opengl_info_wire_t;

typedef struct amdf_dxg_adapter_registry_info_wire_t {
  // Display adapter description.
  amdf_dxg_utf16_t adapter_string[MAX_PATH];
  // Display adapter BIOS description.
  amdf_dxg_utf16_t bios_string[MAX_PATH];
  // Display adapter DAC description.
  amdf_dxg_utf16_t dac_type[MAX_PATH];
  // Display adapter chip description.
  amdf_dxg_utf16_t chip_type[MAX_PATH];
} amdf_dxg_adapter_registry_info_wire_t;

typedef struct amdf_dxg_driver_description_wire_t {
  // Returned Windows UTF-16 driver description.
  amdf_dxg_utf16_t description[4096];
} amdf_dxg_driver_description_wire_t;

typedef struct amdf_dxg_adapter_guid_wire_t {
  // Returned Windows UTF-16 adapter GUID.
  amdf_dxg_utf16_t guid[40];
} amdf_dxg_adapter_guid_wire_t;

#pragma pack(push, 1)
typedef struct amdf_dxg_node_metadata_wire_t {
  // Engine class exposed by the node.
  DXGK_ENGINE_TYPE engine_type;
  // Returned Windows UTF-16 node name.
  amdf_dxg_utf16_t friendly_name[DXGK_MAX_METADATA_NAME_LENGTH];
  // Node scheduling and queue property bits.
  DXGK_NODEMETADATA_FLAGS flags;
  // Nonzero when the node supports GPU page tables.
  BOOLEAN gpu_mmu_supported;
  // Nonzero when the node supports an IO memory management unit.
  BOOLEAN io_mmu_supported;
} amdf_dxg_node_metadata_wire_t;

typedef struct amdf_dxg_query_node_metadata_wire_t {
  // Packed physical-adapter and node ordinal supplied by the caller.
  UINT node_ordinal_and_adapter_index;
  // Returned node metadata.
  amdf_dxg_node_metadata_wire_t node_data;
} amdf_dxg_query_node_metadata_wire_t;
#pragma pack(pop)

typedef struct amdf_dxg_query_registry_info_wire_t {
  // Registry namespace selected by the query.
  D3DDDI_QUERYREGISTRY_TYPE query_type;
  // Registry query behavior bits.
  D3DDDI_QUERYREGISTRY_FLAGS query_flags;
  // Windows UTF-16 value name supplied by the caller.
  amdf_dxg_utf16_t value_name[MAX_PATH];
  // Windows registry value type.
  ULONG value_type;
  // Physical adapter selected within a linked adapter.
  ULONG physical_adapter_index;
  // Returned or required value length in wire bytes.
  ULONG output_value_size;
  // Result of the registry lookup itself.
  D3DDDI_QUERYREGISTRY_STATUS status;
  // Returned registry value storage.
  union {
    // 32-bit integer result.
    DWORD output_dword;
    // 64-bit integer result.
    uint64_t output_qword;
    // Windows UTF-16 string result.
    amdf_dxg_utf16_t output_string[1];
    // Uninterpreted binary result.
    BYTE output_binary[1];
  } value;
} amdf_dxg_query_registry_info_wire_t;

static uint32_t amdf_dxg_utf16_ascii_lower(uint32_t value) {
  return value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
}

static size_t amdf_dxg_find_driver_store_marker(const amdf_dxg_utf16_t* source,
                                                size_t source_length) {
  static const char marker[] = "\\filerepository\\";
  const size_t marker_length = sizeof(marker) - 1;
  if (source_length < marker_length) {
    return SIZE_MAX;
  }
  for (size_t offset = 0; offset <= source_length - marker_length; ++offset) {
    bool matches = true;
    for (size_t i = 0; i < marker_length; ++i) {
      if (amdf_dxg_utf16_ascii_lower(source[offset + i]) !=
          (uint32_t)marker[i]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return offset + marker_length;
    }
  }
  return SIZE_MAX;
}

static bool amdf_dxg_append_wide(wchar_t value, void* target,
                                 size_t target_capacity,
                                 size_t* inout_target_index) {
  if (*inout_target_index >= target_capacity) {
    return false;
  }
  memcpy((uint8_t*)target + *inout_target_index * sizeof(value), &value,
         sizeof(value));
  ++*inout_target_index;
  return true;
}

static bool amdf_dxg_convert_utf16_segment(const amdf_dxg_utf16_t* source,
                                           size_t source_length,
                                           bool rewrite_driver_store,
                                           void* target, size_t target_capacity,
                                           size_t* inout_target_index) {
  size_t source_index = 0;
  bool driver_store_path = false;
  if (rewrite_driver_store) {
    const size_t driver_path_index =
        amdf_dxg_find_driver_store_marker(source, source_length);
    if (driver_path_index != SIZE_MAX) {
      static const wchar_t prefix[] = L"/usr/lib/wsl/drivers/";
      for (size_t i = 0; i < sizeof(prefix) / sizeof(prefix[0]) - 1; ++i) {
        if (!amdf_dxg_append_wide(prefix[i], target, target_capacity,
                                  inout_target_index)) {
          return false;
        }
      }
      source_index = driver_path_index;
      driver_store_path = true;
    }
  }

  while (source_index < source_length) {
    uint32_t code_point = source[source_index++];
    if (code_point >= 0xD800u && code_point <= 0xDBFFu) {
      if (source_index >= source_length) {
        return false;
      }
      const uint32_t low_surrogate = source[source_index++];
      if (low_surrogate < 0xDC00u || low_surrogate > 0xDFFFu) {
        return false;
      }
      code_point =
          0x10000u + ((code_point - 0xD800u) << 10) + (low_surrogate - 0xDC00u);
    } else if (code_point >= 0xDC00u && code_point <= 0xDFFFu) {
      return false;
    }
    wchar_t wide_character = (wchar_t)code_point;
    if (driver_store_path && wide_character == L'\\') {
      wide_character = L'/';
    }
    if (!amdf_dxg_append_wide(wide_character, target, target_capacity,
                              inout_target_index)) {
      return false;
    }
  }
  return true;
}

static bool amdf_dxg_convert_utf16_strings(const amdf_dxg_utf16_t* source,
                                           size_t source_capacity,
                                           bool rewrite_driver_store,
                                           bool multiple_strings, void* target,
                                           size_t target_capacity,
                                           size_t* out_target_length) {
  size_t source_index = 0;
  size_t target_index = 0;
  while (source_index < source_capacity) {
    size_t source_length = 0;
    while (source_index + source_length < source_capacity &&
           source[source_index + source_length] != 0) {
      ++source_length;
    }
    if (!amdf_dxg_convert_utf16_segment(source + source_index, source_length,
                                        rewrite_driver_store, target,
                                        target_capacity, &target_index)) {
      return false;
    }
    if (source_index + source_length == source_capacity) {
      if (out_target_length != NULL) {
        *out_target_length = target_index;
      }
      return true;
    }
    if (!amdf_dxg_append_wide(0, target, target_capacity, &target_index)) {
      return false;
    }
    source_index += source_length + 1;
    if (!multiple_strings || source_length == 0) {
      if (out_target_length != NULL) {
        *out_target_length = target_index;
      }
      return true;
    }
  }
  if (out_target_length != NULL) {
    *out_target_length = target_index;
  }
  return true;
}

typedef void (*amdf_dxg_query_copy_input_fn_t)(const void* source,
                                               void* target);
typedef bool (*amdf_dxg_query_copy_output_fn_t)(const void* source,
                                                void* target);

typedef struct amdf_dxg_query_conversion_t {
  // Query type selecting this fixed layout.
  KMTQUERYADAPTERINFOTYPE query_type;
  // Linux wchar_t-bearing structure size.
  size_t native_size;
  // Windows UTF-16 wire structure size.
  size_t wire_size;
  // Native-to-wire input conversion.
  amdf_dxg_query_copy_input_fn_t copy_input;
  // Wire-to-native output conversion.
  amdf_dxg_query_copy_output_fn_t copy_output;
} amdf_dxg_query_conversion_t;

static void amdf_dxg_copy_no_input(const void* source, void* target) {
  (void)source;
  (void)target;
}

static void amdf_dxg_copy_umd_filename_input(const void* source, void* target) {
  const D3DKMT_UMDFILENAMEINFO* native = source;
  amdf_dxg_umd_filename_info_wire_t* wire = target;
  wire->version = native->Version;
}

static bool amdf_dxg_copy_umd_filename_output(const void* source,
                                              void* target) {
  const amdf_dxg_umd_filename_info_wire_t* wire = source;
  D3DKMT_UMDFILENAMEINFO* native = target;
  return amdf_dxg_convert_utf16_strings(wire->file_name, MAX_PATH, true, false,
                                        native->UmdFileName, MAX_PATH, NULL);
}

static void amdf_dxg_copy_opengl_input(const void* source, void* target) {
  const D3DKMT_OPENGLINFO* native = source;
  amdf_dxg_opengl_info_wire_t* wire = target;
  wire->version = native->Version;
  wire->flags = native->Flags;
}

static bool amdf_dxg_copy_opengl_output(const void* source, void* target) {
  const amdf_dxg_opengl_info_wire_t* wire = source;
  D3DKMT_OPENGLINFO* native = target;
  return amdf_dxg_convert_utf16_strings(wire->file_name, MAX_PATH, true, false,
                                        native->UmdOpenGlIcdFileName, MAX_PATH,
                                        NULL);
}

static bool amdf_dxg_copy_adapter_registry_output(const void* source,
                                                  void* target) {
  const amdf_dxg_adapter_registry_info_wire_t* wire = source;
  D3DKMT_ADAPTERREGISTRYINFO* native = target;
  return amdf_dxg_convert_utf16_strings(wire->adapter_string, MAX_PATH, true,
                                        false, native->AdapterString, MAX_PATH,
                                        NULL) &&
         amdf_dxg_convert_utf16_strings(wire->bios_string, MAX_PATH, true,
                                        false, native->BiosString, MAX_PATH,
                                        NULL) &&
         amdf_dxg_convert_utf16_strings(wire->dac_type, MAX_PATH, true, false,
                                        native->DacType, MAX_PATH, NULL) &&
         amdf_dxg_convert_utf16_strings(wire->chip_type, MAX_PATH, true, false,
                                        native->ChipType, MAX_PATH, NULL);
}

static bool amdf_dxg_copy_driver_description_output(const void* source,
                                                    void* target) {
  const amdf_dxg_driver_description_wire_t* wire = source;
  D3DKMT_DRIVER_DESCRIPTION* native = target;
  return amdf_dxg_convert_utf16_strings(wire->description, 4096, true, false,
                                        native->DriverDescription, 4096, NULL);
}

static bool amdf_dxg_copy_adapter_guid_output(const void* source,
                                              void* target) {
  const amdf_dxg_adapter_guid_wire_t* wire = source;
  D3DKMT_QUERY_ADAPTER_UNIQUE_GUID* native = target;
  return amdf_dxg_convert_utf16_strings(wire->guid, 40, false, false,
                                        native->AdapterUniqueGUID, 40, NULL);
}

static void amdf_dxg_copy_node_metadata_input(const void* source,
                                              void* target) {
  const D3DKMT_NODEMETADATA* native = source;
  amdf_dxg_query_node_metadata_wire_t* wire = target;
  wire->node_ordinal_and_adapter_index = native->NodeOrdinalAndAdapterIndex;
}

static bool amdf_dxg_copy_node_metadata_output(const void* source,
                                               void* target) {
  const amdf_dxg_query_node_metadata_wire_t* wire = source;
  D3DKMT_NODEMETADATA* native = target;
  native->NodeData.EngineType = wire->node_data.engine_type;
  native->NodeData.Flags = wire->node_data.flags;
  native->NodeData.GpuMmuSupported = wire->node_data.gpu_mmu_supported;
  native->NodeData.IoMmuSupported = wire->node_data.io_mmu_supported;
  return amdf_dxg_convert_utf16_strings(wire->node_data.friendly_name,
                                        DXGK_MAX_METADATA_NAME_LENGTH, false,
                                        false, native->NodeData.FriendlyName,
                                        DXGK_MAX_METADATA_NAME_LENGTH, NULL);
}

static const amdf_dxg_query_conversion_t amdf_dxg_query_conversions[] = {
    {KMTQAITYPE_UMDRIVERNAME, sizeof(D3DKMT_UMDFILENAMEINFO),
     sizeof(amdf_dxg_umd_filename_info_wire_t),
     amdf_dxg_copy_umd_filename_input, amdf_dxg_copy_umd_filename_output},
    {KMTQAITYPE_UMOPENGLINFO, sizeof(D3DKMT_OPENGLINFO),
     sizeof(amdf_dxg_opengl_info_wire_t), amdf_dxg_copy_opengl_input,
     amdf_dxg_copy_opengl_output},
    {KMTQAITYPE_ADAPTERREGISTRYINFO, sizeof(D3DKMT_ADAPTERREGISTRYINFO),
     sizeof(amdf_dxg_adapter_registry_info_wire_t), amdf_dxg_copy_no_input,
     amdf_dxg_copy_adapter_registry_output},
    {KMTQAITYPE_ADAPTERREGISTRYINFO_RENDER, sizeof(D3DKMT_ADAPTERREGISTRYINFO),
     sizeof(amdf_dxg_adapter_registry_info_wire_t), amdf_dxg_copy_no_input,
     amdf_dxg_copy_adapter_registry_output},
    {KMTQAITYPE_DRIVER_DESCRIPTION, sizeof(D3DKMT_DRIVER_DESCRIPTION),
     sizeof(amdf_dxg_driver_description_wire_t), amdf_dxg_copy_no_input,
     amdf_dxg_copy_driver_description_output},
    {KMTQAITYPE_DRIVER_DESCRIPTION_RENDER, sizeof(D3DKMT_DRIVER_DESCRIPTION),
     sizeof(amdf_dxg_driver_description_wire_t), amdf_dxg_copy_no_input,
     amdf_dxg_copy_driver_description_output},
    {KMTQAITYPE_QUERY_ADAPTER_UNIQUE_GUID,
     sizeof(D3DKMT_QUERY_ADAPTER_UNIQUE_GUID),
     sizeof(amdf_dxg_adapter_guid_wire_t), amdf_dxg_copy_no_input,
     amdf_dxg_copy_adapter_guid_output},
    {KMTQAITYPE_NODEMETADATA, sizeof(D3DKMT_NODEMETADATA),
     sizeof(amdf_dxg_query_node_metadata_wire_t),
     amdf_dxg_copy_node_metadata_input, amdf_dxg_copy_node_metadata_output},
};

static const amdf_dxg_query_conversion_t* amdf_dxg_find_query_conversion(
    KMTQUERYADAPTERINFOTYPE query_type) {
  for (size_t i = 0; i < sizeof(amdf_dxg_query_conversions) /
                             sizeof(amdf_dxg_query_conversions[0]);
       ++i) {
    if (amdf_dxg_query_conversions[i].query_type == query_type) {
      return &amdf_dxg_query_conversions[i];
    }
  }
  return NULL;
}

static NTSTATUS amdf_dxg_allocate_conversion(const amdf_linux_dxg_kmt_t* kmt,
                                             size_t byte_length,
                                             void** out_allocation) {
  const amdf_status_t status =
      amdf_calloc(*kmt->host_allocator, byte_length, amdf_alignof(max_align_t),
                  out_allocation);
  return amdf_status_is_ok(status) ? 0 : STATUS_NO_MEMORY;
}

static bool amdf_dxg_query_registry_has_string_value(
    const D3DDDI_QUERYREGISTRY_INFO* query) {
  enum {
    AMDF_WINDOWS_REG_SZ = 1,
    AMDF_WINDOWS_REG_EXPAND_SZ = 2,
    AMDF_WINDOWS_REG_MULTI_SZ = 7,
  };
  return query->QueryType == D3DDDI_QUERYREGISTRY_DRIVERSTOREPATH ||
         query->QueryType == D3DDDI_QUERYREGISTRY_DRIVERIMAGEPATH ||
         query->ValueType == AMDF_WINDOWS_REG_SZ ||
         query->ValueType == AMDF_WINDOWS_REG_EXPAND_SZ ||
         query->ValueType == AMDF_WINDOWS_REG_MULTI_SZ;
}

static NTSTATUS amdf_dxg_query_registry(const amdf_linux_dxg_kmt_t* kmt,
                                        D3DKMT_QUERYADAPTERINFO* arguments) {
  if (arguments->pPrivateDriverData == NULL ||
      arguments->PrivateDriverDataSize < sizeof(D3DDDI_QUERYREGISTRY_INFO)) {
    return STATUS_INVALID_PARAMETER;
  }
  D3DDDI_QUERYREGISTRY_INFO* native = arguments->pPrivateDriverData;
  const bool string_value = amdf_dxg_query_registry_has_string_value(native);
  const size_t extra_native_bytes =
      arguments->PrivateDriverDataSize - sizeof(*native);
  if (string_value && extra_native_bytes % sizeof(wchar_t) != 0) {
    return STATUS_INVALID_PARAMETER;
  }
  const size_t extra_wire_bytes =
      string_value
          ? extra_native_bytes / sizeof(wchar_t) * sizeof(amdf_dxg_utf16_t)
          : extra_native_bytes;
  const size_t wire_size =
      sizeof(amdf_dxg_query_registry_info_wire_t) + extra_wire_bytes;

  amdf_dxg_query_registry_info_wire_t* wire = NULL;
  NTSTATUS status = amdf_dxg_allocate_conversion(kmt, wire_size, (void**)&wire);
  if (!NT_SUCCESS(status)) {
    return status;
  }
  wire->query_type = native->QueryType;
  wire->query_flags = native->QueryFlags;
  wire->value_type = native->ValueType;
  wire->physical_adapter_index = native->PhysicalAdapterIndex;
  wire->status = D3DDDI_QUERYREGISTRY_STATUS_FAIL;
  if (native->QueryType == D3DDDI_QUERYREGISTRY_SERVICEKEY ||
      native->QueryType == D3DDDI_QUERYREGISTRY_ADAPTERKEY) {
    for (size_t i = 0; i < MAX_PATH; ++i) {
      wchar_t wide_character = 0;
      memcpy(&wide_character,
             (const uint8_t*)native +
                 offsetof(D3DDDI_QUERYREGISTRY_INFO, ValueName) +
                 i * sizeof(wide_character),
             sizeof(wide_character));
      const uint32_t character = (uint32_t)wide_character;
      if (character > UINT16_MAX) {
        amdf_free(*kmt->host_allocator, wire);
        return STATUS_INVALID_PARAMETER;
      }
      wire->value_name[i] = (amdf_dxg_utf16_t)character;
    }
  }

  D3DKMT_QUERYADAPTERINFO local_arguments = *arguments;
  local_arguments.pPrivateDriverData = wire;
  local_arguments.PrivateDriverDataSize = (UINT)wire_size;
  status = (NTSTATUS)amdf_linux_dxg_connection_invoke(
      (void*)kmt->connection, _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO),
      &local_arguments);
  if (NT_SUCCESS(status)) {
    native->Status = wire->status;
    if (string_value) {
      if (wire->output_value_size % sizeof(amdf_dxg_utf16_t) != 0 ||
          wire->output_value_size / sizeof(amdf_dxg_utf16_t) >
              UINT32_MAX / sizeof(wchar_t)) {
        status = STATUS_INVALID_PARAMETER;
      } else {
        native->OutputValueSize = wire->output_value_size /
                                  sizeof(amdf_dxg_utf16_t) * sizeof(wchar_t);
      }
    } else {
      native->OutputValueSize = wire->output_value_size;
    }
    if (NT_SUCCESS(status) &&
        wire->status == D3DDDI_QUERYREGISTRY_STATUS_SUCCESS) {
      const size_t wire_value_capacity =
          wire_size - offsetof(amdf_dxg_query_registry_info_wire_t, value);
      const size_t native_value_capacity =
          arguments->PrivateDriverDataSize -
          offsetof(D3DDDI_QUERYREGISTRY_INFO, OutputDword);
      if (wire->output_value_size > wire_value_capacity) {
        status = STATUS_INVALID_PARAMETER;
      } else if (string_value) {
        const bool multiple_strings = native->ValueType == 7;
        size_t native_value_length = 0;
        if (!amdf_dxg_convert_utf16_strings(
                wire->value.output_string,
                wire->output_value_size / sizeof(amdf_dxg_utf16_t), true,
                multiple_strings,
                (uint8_t*)native +
                    offsetof(D3DDDI_QUERYREGISTRY_INFO, OutputDword),
                native_value_capacity / sizeof(wchar_t),
                &native_value_length)) {
          status = STATUS_INVALID_PARAMETER;
        } else {
          native->OutputValueSize =
              (ULONG)(native_value_length * sizeof(wchar_t));
        }
      } else {
        if (native->OutputValueSize > native_value_capacity) {
          status = STATUS_INVALID_PARAMETER;
        } else {
          memcpy((uint8_t*)native +
                     offsetof(D3DDDI_QUERYREGISTRY_INFO, OutputDword),
                 &wire->value.output_dword, wire->output_value_size);
        }
      }
    }
  }
  amdf_free(*kmt->host_allocator, wire);
  return status;
}

NTSTATUS amdf_linux_dxg_kmt_query_adapter_info(
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_QUERYADAPTERINFO* arguments) {
  if (arguments->Type == KMTQAITYPE_QUERYREGISTRY) {
    return amdf_dxg_query_registry(kmt, arguments);
  }
  const amdf_dxg_query_conversion_t* conversion =
      amdf_dxg_find_query_conversion(arguments->Type);
  if (conversion == NULL) {
    return (NTSTATUS)amdf_linux_dxg_connection_invoke(
        (void*)kmt->connection, _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO),
        arguments);
  }
  if (arguments->pPrivateDriverData == NULL ||
      arguments->PrivateDriverDataSize != conversion->native_size) {
    return STATUS_INVALID_PARAMETER;
  }

  void* wire_data = NULL;
  NTSTATUS status =
      amdf_dxg_allocate_conversion(kmt, conversion->wire_size, &wire_data);
  if (!NT_SUCCESS(status)) {
    return status;
  }
  conversion->copy_input(arguments->pPrivateDriverData, wire_data);
  D3DKMT_QUERYADAPTERINFO local_arguments = *arguments;
  local_arguments.pPrivateDriverData = wire_data;
  local_arguments.PrivateDriverDataSize = (UINT)conversion->wire_size;
  status = (NTSTATUS)amdf_linux_dxg_connection_invoke(
      (void*)kmt->connection, _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO),
      &local_arguments);
  if (NT_SUCCESS(status) &&
      !conversion->copy_output(wire_data, arguments->pPrivateDriverData)) {
    status = STATUS_INVALID_PARAMETER;
  }
  amdf_free(*kmt->host_allocator, wire_data);
  return status;
}

uint32_t amdf_linux_dxg_kmt_invoke(void* context, uint64_t request,
                                   void* arguments) {
  const amdf_linux_dxg_kmt_t* kmt = context;
  if (request == (uint64_t)_IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO)) {
    return (uint32_t)amdf_linux_dxg_kmt_query_adapter_info(
        kmt, (D3DKMT_QUERYADAPTERINFO*)arguments);
  }
  return amdf_linux_dxg_connection_invoke((void*)kmt->connection, request,
                                          arguments);
}
