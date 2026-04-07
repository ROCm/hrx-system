// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hsa_driver/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "hsa_driver/native_executable_hsaf.h"
#include "hsa_driver/status_util.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"
#include "iree/schemas/hip_executable_def_reader.h"
#include "iree/schemas/hip_executable_def_verifier.h"

typedef struct iree_hal_hsa_native_executable_per_device_data_t {
  // HSA agent handle for this device (needed for symbol lookups).
  hsa_agent_t agent;

  // Loaded HSA executables.
  iree_host_size_t executable_count;
  hsa_executable_t* executables;

  // Exported kernels referencing the loaded executables.
  iree_host_size_t export_count;
  iree_hal_hsa_kernel_params_t exports[];
} iree_hal_hsa_native_executable_per_device_data_t;

// Information about a single exported kernel.
typedef struct iree_hal_hsa_export_info_t {
  // Name of the kernel (null-terminated).
  // Note: TensileLib kernels can have very long names (600+ chars).
  char name[1024];
  // Block dimensions.
  uint32_t block_dims[3];
  // Shared memory size in bytes.
  uint32_t shared_memory_size;
  // Number of constants.
  uint32_t constant_count;
  // Number of bindings.
  uint32_t binding_count;
} iree_hal_hsa_export_info_t;

typedef struct iree_hal_hsa_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;


  // Number of exported kernels.
  iree_host_size_t export_count;
  // Export information for each kernel (allocated after per_device_data).
  iree_hal_hsa_export_info_t* export_infos;

  iree_host_size_t num_devices;
  iree_hal_hsa_native_executable_per_device_data_t* per_device_data[];
} iree_hal_hsa_native_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hsa_native_executable_vtable;

static iree_hal_hsa_native_executable_t* iree_hal_hsa_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_native_executable_vtable);
  return (iree_hal_hsa_native_executable_t*)base_value;
}

static iree_status_t iree_hal_hsa_read_native_flatbuffer_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    iree_const_byte_span_t* out_flatbuffer_data) {
  IREE_ASSERT_ARGUMENT(out_flatbuffer_data);
  *out_flatbuffer_data = iree_const_byte_span_empty();
  return iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_hip_ExecutableDef_file_identifier, out_flatbuffer_data);
}

static void iree_hal_hsa_apply_parsed_kernel_metadata(
    const iree_hal_hip_fat_binary_info_t* module_info,
    flatbuffers_string_t kernel_name, iree_allocator_t host_allocator,
    iree_hal_hsa_kernel_params_t* kernel_info) {
  if (!module_info || !kernel_name || !kernel_info) return;
  iree_string_view_t export_name = iree_make_string_view(
      kernel_name, flatbuffers_string_len(kernel_name));
  for (iree_host_size_t i = 0; i < module_info->kernel_count; ++i) {
    const iree_hal_hip_kernel_info_t* parsed_kernel = &module_info->kernels[i];
    if (!iree_string_view_equal(parsed_kernel->name, export_name)) continue;

    kernel_info->hidden_args = parsed_kernel->hidden_args;
    kernel_info->parameter_count = parsed_kernel->parameter_count;
    if (parsed_kernel->parameter_count == 0 || !parsed_kernel->parameters) {
      return;
    }

    if (iree_status_is_ok(iree_allocator_malloc(
            host_allocator,
            parsed_kernel->parameter_count * sizeof(*kernel_info->parameters),
            (void**)&kernel_info->parameters))) {
      kernel_info->explicit_kernarg_size = 0;
      for (uint32_t j = 0; j < parsed_kernel->parameter_count; ++j) {
        const iree_hal_hip_kernel_param_t* src = &parsed_kernel->parameters[j];
        kernel_info->parameters[j].offset = src->offset;
        kernel_info->parameters[j].size = src->size;
        kernel_info->parameters[j].type =
            src->type == 1
                ? IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING
                : IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT;
        uint32_t end = src->offset + src->size;
        if (end > kernel_info->explicit_kernarg_size) {
          kernel_info->explicit_kernarg_size = end;
        }
      }
    }
    return;
  }
}

iree_status_t iree_hal_hsa_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  // Read the size prefix (with unsafe inference if size is unknown).
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t contained_data = iree_const_byte_span_empty();

  iree_status_t native_hsa = iree_hal_hsa_read_native_header(
      executable_data, unsafe_infer_size, &contained_data);
  if (iree_status_is_ok(native_hsa)) {
    // Successfully read as native HSA executable (fat binary format).
    iree_string_view_t format = IREE_SV("FPIH");  // Use same format as HIP
    if (format.size >= executable_format_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable format buffer too small");
    }
    memcpy(executable_format, format.data, format.size + /*NUL*/ 1);
    *out_inferred_size = contained_data.data_length;
    return iree_ok_status();
  }
  iree_status_ignore(native_hsa);

  IREE_RETURN_IF_ERROR(iree_hal_hsa_read_native_flatbuffer_header(
      executable_data, unsafe_infer_size, &contained_data));

  // Verify the flatbuffer structure.
  int verify_ret = iree_hal_hip_ExecutableDef_verify_as_root(
      contained_data.data, contained_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer verification failed: %s",
        flatcc_verify_error_string(verify_ret));
  }

  // Write the format string.
  iree_string_view_t format = IREE_SV("HSACO");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);

  // Return the total size (header + flatbuffer).
  *out_inferred_size =
      sizeof(iree_flatbuffer_file_header_t) + contained_data.data_length;
  return iree_ok_status();
}

// Forward declaration for FPIH format loader.
static iree_status_t iree_hal_hsa_native_executable_create_fpih(
    iree_hal_hsa_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Forward declaration for flatbuffer format loader.
static iree_status_t iree_hal_hsa_native_executable_create_flatbuffer(
    iree_hal_hsa_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

iree_status_t iree_hal_hsa_native_executable_create(
    iree_hal_hsa_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  if (topology.count < 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "at least one device is required but none were provided");
  }

  *out_executable = NULL;

  // Dispatch based on executable format.
  iree_string_view_t executable_format = executable_params->executable_format;

  if (iree_string_view_equal(executable_format, IREE_SV("FPIH"))) {
    // FPIH (Fat Binary / native HSACO) format.
    return iree_hal_hsa_native_executable_create_fpih(
        topology, executable_params, host_allocator, out_executable);
  }

  // HSACO (flatbuffer) format.
  return iree_hal_hsa_native_executable_create_flatbuffer(
      topology, executable_params, host_allocator, out_executable);
}

static iree_status_t iree_hal_hsa_native_executable_create_flatbuffer(
    iree_hal_hsa_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Read and strip the flatbuffer header prefix.
  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_read_native_flatbuffer_header(
              executable_params->executable_data, /*unsafe_infer_size=*/false,
              &executable_flatbuffer));

  // Verify the flatbuffer structure.
  int verify_ret = iree_hal_hip_ExecutableDef_verify_as_root(
      executable_flatbuffer.data, executable_flatbuffer.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer verification failed: %s",
        flatcc_verify_error_string(verify_ret));
  }

  iree_hal_hip_ExecutableDef_table_t hip_executable_def = NULL;
  iree_hal_hip_ModuleDef_vec_t hip_modules_vec = NULL;
  iree_hal_hip_ExportDef_vec_t hip_exports_vec = NULL;
  hip_executable_def =
      iree_hal_hip_ExecutableDef_as_root(executable_flatbuffer.data);
  hip_modules_vec = iree_hal_hip_ExecutableDef_modules_get(hip_executable_def);
  iree_host_size_t module_count = iree_hal_hip_ModuleDef_vec_len(hip_modules_vec);
  hip_exports_vec = iree_hal_hip_ExecutableDef_exports_get(hip_executable_def);
  iree_host_size_t export_count = iree_hal_hip_ExportDef_vec_len(hip_exports_vec);

  // Calculate the total number of characters across all entry point names.
  iree_host_size_t total_export_info_length = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_hal_hip_ExportDef_table_t export_def =
          iree_hal_hip_ExportDef_vec_at(hip_exports_vec, i);
      iree_hal_debug_ExportDef_table_t export_debug_info =
          iree_hal_hip_ExportDef_debug_info_get(export_def);
      total_export_info_length +=
          iree_hal_debug_calculate_export_info_size(export_debug_info);
    }
  });

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hsa_native_executable_t* executable = NULL;
  iree_host_size_t native_executable_device_info_size =
      sizeof(*executable->per_device_data[0]) +
      module_count * sizeof(executable->per_device_data[0]->executables[0]) +
      export_count * sizeof(executable->per_device_data[0]->exports[0]) +
      total_export_info_length;
  native_executable_device_info_size =
      iree_host_align(native_executable_device_info_size, iree_max_align_t);
  const iree_host_size_t total_size =
      sizeof(*executable) +
      topology.count * sizeof(executable->per_device_data[0]) +
      topology.count * native_executable_device_info_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_hsa_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->num_devices = topology.count;

  const iree_host_size_t per_device_data_size =
      topology.count * sizeof(executable->per_device_data[0]);
  const uint8_t* per_device_data_location =
      (uint8_t*)executable + sizeof(*executable);

  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    const iree_host_size_t native_executable_device_info_size_offset =
        (i * native_executable_device_info_size);
    executable->per_device_data[i] =
        (iree_hal_hsa_native_executable_per_device_data_t*)(per_device_data_location +
                                                            per_device_data_size +
                                                            native_executable_device_info_size_offset);
  }

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_hip_ExecutableDef_source_files_get(hip_executable_def));

  iree_status_t status = iree_ok_status();
  iree_hal_hip_fat_binary_info_t* module_infos = NULL;
  if (module_count > 0) {
    status = iree_allocator_malloc(
        host_allocator, module_count * sizeof(*module_infos),
        (void**)&module_infos);
    if (iree_status_is_ok(status)) {
      memset(module_infos, 0, module_count * sizeof(*module_infos));
      for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status);
           ++i) {
        iree_hal_hip_ModuleDef_table_t module_def =
            iree_hal_hip_ModuleDef_vec_at(hip_modules_vec, i);
        flatbuffers_string_t hsaco_image =
            iree_hal_hip_ModuleDef_hsaco_image_get(module_def);
        status = iree_hal_hsa_parse_fat_binary_kernels(
            iree_make_const_byte_span(
                (const uint8_t*)hsaco_image,
                flatbuffers_string_len(hsaco_image)),
            iree_string_view_empty(), host_allocator, &module_infos[i]);
      }
    }
  }
  for (iree_host_size_t j = 0; j < topology.count && iree_status_is_ok(status);
       ++j) {
    iree_hal_hsa_native_executable_per_device_data_t* per_device_data =
        executable->per_device_data[j];
    hsa_agent_t agent = topology.devices[j].agent;

    per_device_data->agent = agent;
    per_device_data->executable_count = module_count;
    per_device_data->executables =
        (hsa_executable_t*)((uint8_t*)per_device_data + sizeof(*per_device_data) +
                            (export_count * sizeof(per_device_data->exports[0])));
    per_device_data->export_count = export_count;
    IREE_TRACE(uint8_t* export_info_ptr =
                   ((uint8_t*)per_device_data->executables +
                    module_count * sizeof(per_device_data->executables[0])));

    // Load each module first so that exports can reference them.
    for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status); ++i) {
      iree_hal_hip_ModuleDef_table_t module_def =
          iree_hal_hip_ModuleDef_vec_at(hip_modules_vec, i);
      flatbuffers_string_t hsaco_image =
          iree_hal_hip_ModuleDef_hsaco_image_get(module_def);
      size_t hsaco_size = flatbuffers_string_len(hsaco_image);

      // Create code object reader.
      hsa_code_object_reader_t code_reader;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_code_object_reader_create_from_memory(hsaco_image, hsaco_size,
                                                    &code_reader),
          "hsa_code_object_reader_create_from_memory");
      if (!iree_status_is_ok(status)) break;

      // Create executable.
      hsa_executable_t hsa_executable;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    NULL, &hsa_executable),
          "hsa_executable_create_alt");
      if (!iree_status_is_ok(status)) {
        hsa_code_object_reader_destroy(code_reader);
        break;
      }

      // Load code object.
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_load_agent_code_object(hsa_executable, agent,
                                                code_reader, NULL, NULL),
          "hsa_executable_load_agent_code_object");
      hsa_code_object_reader_destroy(code_reader);
      if (!iree_status_is_ok(status)) {
        hsa_executable_destroy(hsa_executable);
        break;
      }

      // Freeze executable.
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_freeze(hsa_executable, NULL),
          "hsa_executable_freeze");
      if (!iree_status_is_ok(status)) {
        hsa_executable_destroy(hsa_executable);
        break;
      }

      per_device_data->executables[i] = hsa_executable;
    }

    // Look up kernel symbols and populate export info.
    for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status); ++i) {
      iree_hal_hip_ExportDef_table_t hip_export_def =
          iree_hal_hip_ExportDef_vec_at(hip_exports_vec, i);
      uint32_t module_ordinal =
          iree_hal_hip_ExportDef_module_ordinal_get(hip_export_def);
      flatbuffers_string_t kernel_name =
          iree_hal_hip_ExportDef_kernel_name_get(hip_export_def);

      // Look up kernel symbol.
      hsa_executable_symbol_t kernel_symbol;
      hsa_executable_t hsa_executable =
          per_device_data->executables[module_ordinal];
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_get_symbol_by_name(hsa_executable, kernel_name,
                                            &agent, &kernel_symbol),
          "hsa_executable_get_symbol_by_name");
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        char kernel_symbol_name[1030];
        snprintf(kernel_symbol_name, sizeof(kernel_symbol_name), "%s.kd",
                 kernel_name);
        status = IREE_HSA_CALL_TO_STATUS(
            hsa_executable_get_symbol_by_name(hsa_executable,
                                              kernel_symbol_name, &agent,
                                              &kernel_symbol),
            "hsa_executable_get_symbol_by_name");
      }
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status, "failed to find kernel '%s' in executable", kernel_name);
        break;
      }

      // Get kernel object (entry point address).
      uint64_t kernel_object = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
              &kernel_object),
          "hsa_executable_symbol_get_info(KERNEL_OBJECT)");
      if (!iree_status_is_ok(status)) break;

      // Get kernarg segment size.
      uint32_t kernarg_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              kernel_symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
              &kernarg_segment_size),
          "hsa_executable_symbol_get_info(KERNARG_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;

      // Get group segment size.
      uint32_t group_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              kernel_symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
              &group_segment_size),
          "hsa_executable_symbol_get_info(GROUP_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;

      // Get private segment size.
      uint32_t private_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              kernel_symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
              &private_segment_size),
          "hsa_executable_symbol_get_info(PRIVATE_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches.
      iree_hal_hsa_kernel_params_t* kernel_info = &per_device_data->exports[i];
      kernel_info->kernel_object = kernel_object;
      kernel_info->kernarg_segment_size = kernarg_segment_size;
      // Explicit kernarg size will be calculated from flatbuffer parameters.
      kernel_info->explicit_kernarg_size = 0;  // Will be set below
      kernel_info->group_segment_size = group_segment_size;
      kernel_info->private_segment_size = private_segment_size;

      const iree_hal_hip_BlockDims_t* block_dims =
          iree_hal_hip_ExportDef_block_dims_get(hip_export_def);
      if (block_dims) {
        kernel_info->block_dims[0] = block_dims->x;
        kernel_info->block_dims[1] = block_dims->y;
        kernel_info->block_dims[2] = block_dims->z;
      } else {
        kernel_info->block_dims[0] = 1;
        kernel_info->block_dims[1] = 1;
        kernel_info->block_dims[2] = 1;
      }
      kernel_info->constant_count =
          iree_hal_hip_ExportDef_constant_count_get(hip_export_def);
      iree_hal_hip_BindingBits_vec_t binding_flags_vec =
          iree_hal_hip_ExportDef_binding_flags_get(hip_export_def);
      kernel_info->binding_count =
          iree_hal_hip_BindingBits_vec_len(binding_flags_vec);

      kernel_info->hidden_args = (iree_hal_hip_hidden_args_t){
          .block_count_x = UINT32_MAX,
          .block_count_y = UINT32_MAX,
          .block_count_z = UINT32_MAX,
          .group_size_x = UINT32_MAX,
          .group_size_y = UINT32_MAX,
          .group_size_z = UINT32_MAX,
          .remainder_x = UINT32_MAX,
          .remainder_y = UINT32_MAX,
          .remainder_z = UINT32_MAX,
          .grid_dims = UINT32_MAX,
          .global_offset_x = UINT32_MAX,
          .global_offset_y = UINT32_MAX,
          .global_offset_z = UINT32_MAX,
      };

      // Estimate explicit kernarg size: pointers for bindings + 4 bytes per constant.
      // This is a rough estimate; precise size comes from flatbuffer parameters.
      kernel_info->explicit_kernarg_size =
          kernel_info->binding_count * sizeof(void*) +
          kernel_info->constant_count * sizeof(uint32_t);
      if (module_infos) {
        if (module_ordinal < module_count) {
          iree_hal_hsa_apply_parsed_kernel_metadata(
              &module_infos[module_ordinal], kernel_name, host_allocator,
              kernel_info);
        }
      }

      IREE_TRACE({
        iree_hal_debug_export_info_t* export_info =
            (iree_hal_debug_export_info_t*)export_info_ptr;
        iree_hal_debug_ExportDef_table_t export_debug_info =
            iree_hal_hip_ExportDef_debug_info_get(hip_export_def);
        export_info_ptr +=
            iree_hal_debug_copy_export_info(export_debug_info, export_info);
        kernel_info->debug_info.function_name = export_info->function_name;
        kernel_info->debug_info.source_filename = export_info->source_filename;
        kernel_info->debug_info.source_line = export_info->source_line;
      });
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  if (module_infos) {
    for (iree_host_size_t i = 0; i < module_count; ++i) {
      iree_hal_hsa_free_fat_binary_kernel_info(
          host_allocator, module_infos[i].kernel_count,
          module_infos[i].kernels);
    }
    iree_allocator_free(host_allocator, module_infos);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Helper to extract the matching ELF data from an offload bundle for a given target.
// Returns the ELF data span within the bundle.
static iree_status_t iree_hal_hsa_extract_elf_from_bundle(
    iree_const_byte_span_t bundle_data, iree_string_view_t target_triple,
    iree_const_byte_span_t* out_elf_data) {
  // Skip magic (should be "__CLANG_OFFLOAD_BUNDLE__").
  const size_t magic_size = 24;  // strlen("__CLANG_OFFLOAD_BUNDLE__")
  if (bundle_data.data_length < magic_size + sizeof(uint64_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bundle too small for header");
  }

  const uint8_t* ptr = bundle_data.data + magic_size;
  uint64_t num_bundles;
  memcpy(&num_bundles, ptr, sizeof(num_bundles));
  ptr += sizeof(num_bundles);

  // Each entry: uint64_t offset, uint64_t size, uint64_t triple_size, char triple[]
  for (uint64_t i = 0; i < num_bundles; ++i) {
    typedef struct {
      uint64_t offset;
      uint64_t size;
      uint64_t triple_size;
    } bundle_entry_t;

    if ((size_t)(ptr - bundle_data.data) + sizeof(bundle_entry_t) >
        bundle_data.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bundle entry %" PRIu64 " header out of bounds", i);
    }

    bundle_entry_t entry;
    memcpy(&entry, ptr, sizeof(entry));
    ptr += sizeof(entry);

    if ((size_t)(ptr - bundle_data.data) + entry.triple_size >
        bundle_data.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bundle entry %" PRIu64 " triple out of bounds", i);
    }

    iree_string_view_t entry_triple =
        iree_make_string_view((const char*)ptr, entry.triple_size);
    ptr += entry.triple_size;

    // Check if this triple matches.
    if (iree_string_view_equal(entry_triple, target_triple)) {
      if (entry.offset + entry.size > bundle_data.data_length) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "bundle entry %" PRIu64 " data out of bounds", i);
      }
      *out_elf_data =
          iree_make_const_byte_span(bundle_data.data + entry.offset, entry.size);
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no ELF found for target triple '%.*s'",
                          (int)target_triple.size, target_triple.data);
}

static iree_status_t iree_hal_hsa_native_executable_create_fpih(
    iree_hal_hsa_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get device architecture name for target matching.
  char device_name[64] = {0};
  iree_status_t status = IREE_HSA_CALL_TO_STATUS(
      hsa_agent_get_info(topology.devices[0].agent, HSA_AGENT_INFO_NAME,
                         device_name),
      "hsa_agent_get_info(HSA_AGENT_INFO_NAME)");
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Construct target triple (e.g., "hipv4-amdgcn-amd-amdhsa--gfx942")
  char target_triple_str[256];
  snprintf(target_triple_str, sizeof(target_triple_str),
           "hipv4-amdgcn-amd-amdhsa--%s", device_name);
  iree_string_view_t target_triple = iree_make_cstring_view(target_triple_str);

  // Parse the fat binary to extract kernel information for this device.
  iree_hal_hsa_fat_binary_info_t fat_binary_info;
  status = iree_hal_hsa_parse_fat_binary_kernels(
      executable_params->executable_data, target_triple, host_allocator,
      &fat_binary_info);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Extract the matching ELF from the offload bundle for HSA.
  // HSA's code object reader expects raw ELF, not the bundle format.
  // Use fat_binary_info.bundle_data which contains the decompressed bundle
  // (or original data if not compressed).
  iree_const_byte_span_t bundle_span = iree_make_const_byte_span(
      fat_binary_info.bundle_data, fat_binary_info.bundle_size);

  iree_const_byte_span_t elf_data = iree_const_byte_span_empty();

  // Check if bundle_data is already a raw ELF (starts with \x7fELF).
  // This happens when the executable is a single ELF without offload bundle wrapper.
  if (fat_binary_info.bundle_size >= 4 &&
      fat_binary_info.bundle_data[0] == 0x7f &&
      fat_binary_info.bundle_data[1] == 'E' &&
      fat_binary_info.bundle_data[2] == 'L' &&
      fat_binary_info.bundle_data[3] == 'F') {
    // It's already a raw ELF - use it directly.
    elf_data = bundle_span;
  } else if (fat_binary_info.bundle_size >= 24 &&
             memcmp(fat_binary_info.bundle_data, "__CLANG_OFFLOAD_BUNDLE__", 24) == 0) {
    // It's an offload bundle - extract the matching ELF.
    status = iree_hal_hsa_extract_elf_from_bundle(
        bundle_span, target_triple, &elf_data);
    if (!iree_status_is_ok(status)) {
      iree_hal_hsa_free_kernel_info(host_allocator, fat_binary_info.kernel_count,
                                     fat_binary_info.kernels);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  } else {
    // Unknown format - try to extract anyway but this will likely fail.
    iree_hal_hsa_free_kernel_info(host_allocator, fat_binary_info.kernel_count,
                                   fat_binary_info.kernels);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized bundle format (magic: 0x%02x%02x%02x%02x)",
                            fat_binary_info.bundle_data[0],
                            fat_binary_info.bundle_data[1],
                            fat_binary_info.bundle_data[2],
                            fat_binary_info.bundle_data[3]);
  }

  iree_host_size_t export_count = fat_binary_info.kernel_count;

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hsa_native_executable_t* executable = NULL;
  iree_host_size_t native_executable_device_info_size =
      sizeof(*executable->per_device_data[0]) +
      1 * sizeof(executable->per_device_data[0]->executables[0]) +
      export_count * sizeof(executable->per_device_data[0]->exports[0]);
  native_executable_device_info_size =
      iree_host_align(native_executable_device_info_size, iree_max_align_t);
  const iree_host_size_t export_infos_size =
      export_count * sizeof(iree_hal_hsa_export_info_t);
  const iree_host_size_t total_size =
      sizeof(*executable) +
      topology.count * sizeof(executable->per_device_data[0]) +
      topology.count * native_executable_device_info_size +
      export_infos_size;

  status = iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (!iree_status_is_ok(status)) {
    iree_hal_hsa_free_kernel_info(host_allocator, fat_binary_info.kernel_count,
                                   fat_binary_info.kernels);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(executable, 0, total_size);

  iree_hal_resource_initialize(&iree_hal_hsa_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->export_count = export_count;
  executable->num_devices = topology.count;
  const iree_host_size_t per_device_data_size =
      topology.count * sizeof(executable->per_device_data[0]);
  const uint8_t* per_device_data_location =
      (uint8_t*)executable + sizeof(*executable);

  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    const iree_host_size_t offset = i * native_executable_device_info_size;
    executable->per_device_data[i] =
        (iree_hal_hsa_native_executable_per_device_data_t*)(
            per_device_data_location + per_device_data_size + offset);
  }

  // Set up export_infos pointer (after all per-device data).
  executable->export_infos = (iree_hal_hsa_export_info_t*)(
      per_device_data_location + per_device_data_size +
      topology.count * native_executable_device_info_size);

  // Populate export info from fat binary info.
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_hsa_export_info_t* info = &executable->export_infos[i];
    iree_hal_hip_kernel_info_t* kernel = &fat_binary_info.kernels[i];

    // Copy kernel name (truncate if necessary).
    iree_host_size_t name_len = kernel->name.size < sizeof(info->name) - 1
                                    ? kernel->name.size
                                    : sizeof(info->name) - 1;
    memcpy(info->name, kernel->name.data, name_len);
    info->name[name_len] = '\0';

    info->block_dims[0] = kernel->block_dims[0];
    info->block_dims[1] = kernel->block_dims[1];
    info->block_dims[2] = kernel->block_dims[2];
    info->shared_memory_size = 0;  // Will be queried from HSA later if needed.
    info->constant_count = kernel->constant_count;
    info->binding_count = kernel->binding_count;
  }

  // Load the code object for each device.
  for (iree_host_size_t device_idx = 0;
       device_idx < topology.count && iree_status_is_ok(status);
       ++device_idx) {
    iree_hal_hsa_native_executable_per_device_data_t* data =
        executable->per_device_data[device_idx];

    // Create HSA code object reader from the extracted ELF data.
    // HSA requires raw ELF, not the offload bundle format.
    hsa_code_object_reader_t code_reader;
    status = IREE_HSA_CALL_TO_STATUS(
        hsa_code_object_reader_create_from_memory(
            elf_data.data, elf_data.data_length, &code_reader),
        "hsa_code_object_reader_create_from_memory");
    if (!iree_status_is_ok(status)) break;

    // Create HSA executable.
    hsa_executable_t hsa_exec;
    status = IREE_HSA_CALL_TO_STATUS(
        hsa_executable_create_alt(HSA_PROFILE_FULL,
                                   HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                   NULL, &hsa_exec),
        "hsa_executable_create_alt");
    if (!iree_status_is_ok(status)) {
      hsa_code_object_reader_destroy(code_reader);
      break;
    }

    // Load code object into executable.
    status = IREE_HSA_CALL_TO_STATUS(
        hsa_executable_load_agent_code_object(
            hsa_exec, topology.devices[device_idx].agent, code_reader, NULL,
            NULL),
        "hsa_executable_load_agent_code_object");
    // Destroy the code reader now that we're done with it.
    hsa_code_object_reader_destroy(code_reader);
    if (!iree_status_is_ok(status)) {
      hsa_executable_destroy(hsa_exec);
      break;
    }

    // Freeze the executable.
    status = IREE_HSA_CALL_TO_STATUS(
                                      hsa_executable_freeze(hsa_exec, NULL),
                                      "hsa_executable_freeze");
    if (!iree_status_is_ok(status)) {
      hsa_executable_destroy(hsa_exec);
      hsa_code_object_reader_destroy(code_reader);
      break;
    }

    // Set up per-device data structure.
    // Layout in memory: [per_device_data_struct][exports[export_count]][executables[1]]
    data->agent = topology.devices[device_idx].agent;
    data->export_count = export_count;
    data->executable_count = 1;
    data->executables = (hsa_executable_t*)(
        (uint8_t*)data + sizeof(*data) +
        export_count * sizeof(data->exports[0]));
    data->executables[0] = hsa_exec;

    // Look up kernel symbols and set up export info.
    for (iree_host_size_t export_idx = 0;
         export_idx < export_count && iree_status_is_ok(status);
         ++export_idx) {
      iree_hal_hsa_kernel_params_t* export_info = &data->exports[export_idx];
      iree_string_view_t kernel_name = fat_binary_info.kernels[export_idx].name;

      // Copy kernel name to a null-terminated buffer.
      // Note: TensileLib kernels can have very long names (600+ chars).
      char kernel_name_cstr[1024];
      iree_host_size_t name_len =
          kernel_name.size < sizeof(kernel_name_cstr) - 1
              ? kernel_name.size
              : sizeof(kernel_name_cstr) - 1;
      memcpy(kernel_name_cstr, kernel_name.data, name_len);
      kernel_name_cstr[name_len] = '\0';

      // HSA expects the kernel descriptor name (with .kd suffix).
      char kernel_symbol_name[1030];
      snprintf(kernel_symbol_name, sizeof(kernel_symbol_name), "%s.kd",
               kernel_name_cstr);

      // Get kernel symbol from executable.
      hsa_executable_symbol_t symbol;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_get_symbol_by_name(
              hsa_exec, kernel_symbol_name,
              &topology.devices[device_idx].agent, &symbol),
          "hsa_executable_get_symbol_by_name");
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status, "failed to find kernel '%s' in executable", kernel_symbol_name);
        break;
      }

      // Get kernel object (GPU address of kernel descriptor).
      uint64_t kernel_object = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
          "hsa_executable_symbol_get_info(KERNEL_OBJECT)");
      if (!iree_status_is_ok(status)) break;

      export_info->kernel_object = kernel_object;
      export_info->block_dims[0] = fat_binary_info.kernels[export_idx].block_dims[0];
      export_info->block_dims[1] = fat_binary_info.kernels[export_idx].block_dims[1];
      export_info->block_dims[2] = fat_binary_info.kernels[export_idx].block_dims[2];
      export_info->binding_count = fat_binary_info.kernels[export_idx].binding_count;
      export_info->constant_count = fat_binary_info.kernels[export_idx].constant_count;
      export_info->hidden_args = fat_binary_info.kernels[export_idx].hidden_args;

      // Copy parameter info from fat binary.
      iree_hal_hip_kernel_info_t* kernel_info = &fat_binary_info.kernels[export_idx];
      export_info->parameter_count = kernel_info->parameter_count;
      export_info->parameters = NULL;

      // Calculate explicit kernarg size from parameters.
      uint32_t explicit_size = 0;
      if (export_info->parameter_count > 0 && kernel_info->parameters) {
        iree_status_t param_status = iree_allocator_malloc(
            host_allocator,
            export_info->parameter_count * sizeof(iree_hal_executable_export_parameter_t),
            (void**)&export_info->parameters);
        if (iree_status_is_ok(param_status)) {
          for (uint32_t p = 0; p < export_info->parameter_count; ++p) {
            export_info->parameters[p].offset = kernel_info->parameters[p].offset;
            export_info->parameters[p].size = kernel_info->parameters[p].size;
            // Map the type: 0=value (constant), 1=pointer (binding)
            export_info->parameters[p].type =
                (kernel_info->parameters[p].type == 1)
                    ? IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING
                    : IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT;
            // Track maximum extent of explicit arguments.
            uint32_t end = kernel_info->parameters[p].offset + kernel_info->parameters[p].size;
            if (end > explicit_size) explicit_size = end;
          }
        }
      }
      export_info->explicit_kernarg_size = explicit_size;

      // Get kernarg segment size.
      uint32_t kernarg_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
              &kernarg_segment_size),
          "hsa_executable_symbol_get_info(KERNARG_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;
      
      // If HSA returns 0 for kernarg_segment_size but we have a valid
      // explicit_kernarg_size from our fat binary parsing, use that instead.
      // This happens with some code objects (e.g., hipBLASLt's Tensile kernels)
      // where the HSA runtime doesn't correctly report the size.
      if (kernarg_segment_size == 0 && export_info->explicit_kernarg_size > 0) {
        kernarg_segment_size = export_info->explicit_kernarg_size;
      }
      export_info->kernarg_segment_size = kernarg_segment_size;

      // Get group segment size.
      uint32_t group_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
              &group_segment_size),
          "hsa_executable_symbol_get_info(GROUP_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;
      export_info->group_segment_size = group_segment_size;

      // Get private segment size.
      uint32_t private_segment_size = 0;
      status = IREE_HSA_CALL_TO_STATUS(
          hsa_executable_symbol_get_info(
              symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
              &private_segment_size),
          "hsa_executable_symbol_get_info(PRIVATE_SEGMENT_SIZE)");
      if (!iree_status_is_ok(status)) break;
      export_info->private_segment_size = private_segment_size;
    }
  }

  // Free the kernel info from fat binary parsing.
  iree_hal_hsa_free_kernel_info(host_allocator, fat_binary_info.kernel_count,
                                 fat_binary_info.kernels);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hsa_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->num_devices; ++i) {
    iree_hal_hsa_native_executable_per_device_data_t* data =
        executable->per_device_data[i];
    for (iree_host_size_t j = 0; j < data->export_count; ++j) {
      if (data->exports[j].parameters) {
        iree_allocator_free(host_allocator, data->exports[j].parameters);
      }
    }
    for (iree_host_size_t j = 0; j < data->executable_count; ++j) {
      if (data->executables[j].handle) {
        IREE_HSA_IGNORE_ERROR(
                              hsa_executable_destroy(data->executables[j]));
      }
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hsa_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t ordinal,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hsa_kernel_params_t** out_params) {
  *out_params = NULL;
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  int device_ordinal = 0;
  if (queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_ordinal > (int)executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity for non-existent queue was provided.");
  }

  const iree_hal_hsa_native_executable_per_device_data_t* data =
      executable->per_device_data[device_ordinal];
  if (ordinal >= data->export_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "export ordinal %d out of range; executable contains %" PRIhsz
        " exports",
        ordinal, data->export_count);
  }
  *out_params = &data->exports[ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_hsa_native_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  return executable->export_count;
}

static iree_status_t iree_hal_hsa_native_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32 " out of range",
                            export_ordinal);
  }
  iree_hal_hsa_export_info_t* export = &executable->export_infos[export_ordinal];
  // Get parameter info from per-device data (same for all devices).
  iree_hal_hsa_kernel_params_t* kernel_params =
      &executable->per_device_data[0]->exports[export_ordinal];

  memset(out_info, 0, sizeof(*out_info));
  out_info->name = iree_make_cstring_view(export->name);
  out_info->workgroup_size[0] = export->block_dims[0];
  out_info->workgroup_size[1] = export->block_dims[1];
  out_info->workgroup_size[2] = export->block_dims[2];
  out_info->constant_count = export->constant_count;
  out_info->binding_count = export->binding_count;
  out_info->parameter_count = kernel_params->parameter_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_native_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %" PRIu32 " out of range",
                            export_ordinal);
  }

  // Get parameter info from per-device data (same for all devices).
  iree_hal_hsa_kernel_params_t* kernel_params =
      &executable->per_device_data[0]->exports[export_ordinal];

  if (!kernel_params->parameters) {
    // No parameter information available.
    return iree_ok_status();
  }

  // Copy parameters up to capacity.
  iree_host_size_t copy_count = kernel_params->parameter_count < capacity
                                    ? kernel_params->parameter_count
                                    : capacity;
  for (iree_host_size_t i = 0; i < copy_count; ++i) {
    out_parameters[i] = kernel_params->parameters[i];
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_native_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static iree_status_t iree_hal_hsa_native_executable_lookup_global(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, uint64_t* out_device_address,
    iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(base_executable);
  IREE_ASSERT_ARGUMENT(out_device_address);
  *out_device_address = 0;
  if (out_size) *out_size = 0;

  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);

  int device_ordinal = 0;
  if (queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_ordinal >= (int)executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity for non-existent device was provided.");
  }

  const iree_hal_hsa_native_executable_per_device_data_t* data =
      executable->per_device_data[device_ordinal];

  // Create a null-terminated copy of the name.
  char name_cstr[1024];
  if (name.size >= sizeof(name_cstr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "global name too long: %.*s", (int)name.size,
                            name.data);
  }
  memcpy(name_cstr, name.data, name.size);
  name_cstr[name.size] = '\0';

  // Try to find the symbol in each HSA executable.
  for (iree_host_size_t i = 0; i < data->executable_count; ++i) {
    hsa_executable_t hsa_exec = data->executables[i];
    hsa_executable_symbol_t symbol;

    // Try to get the symbol by name using the device agent.
    hsa_status_t hsa_status = hsa_executable_get_symbol_by_name(
        hsa_exec, name_cstr, &data->agent, &symbol);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      // Symbol not found in this executable, try the next one.
      continue;
    }

    // Get the variable address.
    uint64_t address = 0;
    hsa_status = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &address);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "failed to get variable address for '%.*s'",
                              (int)name.size, name.data);
    }

    // Get the variable size.
    uint32_t size = 0;
    hsa_status = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &size);
    if (hsa_status != HSA_STATUS_SUCCESS) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "failed to get variable size for '%.*s'",
                              (int)name.size, name.data);
    }

    *out_device_address = address;
    if (out_size) *out_size = size;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "global variable '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static const iree_hal_executable_vtable_t
    iree_hal_hsa_native_executable_vtable = {
        .destroy = iree_hal_hsa_native_executable_destroy,
        .export_count = iree_hal_hsa_native_executable_export_count,
        .export_info = iree_hal_hsa_native_executable_export_info,
        .export_parameters = iree_hal_hsa_native_executable_export_parameters,
        .lookup_export_by_name =
            iree_hal_hsa_native_executable_lookup_export_by_name,
        .lookup_global = iree_hal_hsa_native_executable_lookup_global,
};
