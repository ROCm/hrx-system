// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/cuda_executable_def_reader.h"
#include "iree/schemas/cuda_executable_def_verifier.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"

typedef struct iree_hal_cuda_native_executable_global_t {
  // Next executable-owned global entry.
  struct iree_hal_cuda_native_executable_global_t* next;

  // Persistent executable-owned global name.
  iree_string_view_t name;

  // Byte length verified against the loaded CUDA modules.
  iree_device_size_t byte_length;

  // Executable-owned buffer alias for this global.
  iree_hal_buffer_t* buffer;
} iree_hal_cuda_native_executable_global_t;

typedef struct iree_hal_cuda_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  // Host allocator used for executable lifetime.
  iree_allocator_t host_allocator;

  // Borrowed HAL device used for buffer placement metadata.
  iree_hal_device_t* device;
  // Borrowed CUDA dynamic symbols used for module and global lookup.
  const iree_hal_cuda_dynamic_symbols_t* symbols;

  // CUDA context owning the executable modules.
  CUcontext cu_context;

  // Number of loaded CUDA modules.
  iree_host_size_t module_count;
  // Loaded CUDA modules.
  CUmodule* modules;

  // Guards executable-owned global entry and buffer publication.
  iree_slim_mutex_t global_mutex;

  // Executable-owned globals interned by name.
  iree_hal_cuda_native_executable_global_t* global_list;

  // Number of exported kernels referencing the loaded modules.
  iree_host_size_t export_count;
  // Exported kernels referencing the loaded modules.
  iree_hal_cuda_kernel_params_t exports[];
} iree_hal_cuda_native_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_cuda_native_executable_vtable;

static iree_hal_cuda_native_executable_t* iree_hal_cuda_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_native_executable_vtable);
  return (iree_hal_cuda_native_executable_t*)base_value;
}

typedef struct iree_hal_cuda_limits_t {
  uint32_t max_block_dims[3];
  uint32_t max_block_shared_memory_size;
} iree_hal_cuda_limits_t;
static iree_status_t iree_hal_cuda_query_limits(
    const iree_hal_cuda_dynamic_symbols_t* symbols, CUdevice device,
    iree_hal_cuda_limits_t* out_limits) {
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_CUDA_RETURN_IF_ERROR(
      symbols,
      cuDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[0],
                           CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device),
      "cuDeviceGetAttribute");
  IREE_CUDA_RETURN_IF_ERROR(
      symbols,
      cuDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[1],
                           CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device),
      "cuDeviceGetAttribute");
  IREE_CUDA_RETURN_IF_ERROR(
      symbols,
      cuDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[2],
                           CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device),
      "cuDeviceGetAttribute");

  IREE_CUDA_RETURN_IF_ERROR(
      symbols,
      cuDeviceGetAttribute(
          (int32_t*)&out_limits->max_block_shared_memory_size,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device),
      "cuDeviceGetAttribute");

  return iree_ok_status();
}

iree_status_t iree_hal_cuda_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  // Read the header prefix (with unsafe inference if size is unknown).
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_cuda_ExecutableDef_file_identifier, &flatbuffer_data));

  // Verify the flatbuffer structure.
  if (!iree_hal_cuda_ExecutableDef_verify_as_root(
          flatbuffer_data.data, flatbuffer_data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to verify executable flatbuffer structure");
  }

  // Write the format string.
  iree_string_view_t format = IREE_SV("PTXE");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);

  // Return the total size (header + flatbuffer).
  *out_inferred_size =
      sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
  return iree_ok_status();
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_cuda_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data,
    const iree_hal_cuda_limits_t* limits) {
  IREE_ASSERT(flatbuffer_data.data && flatbuffer_data.data_length >= 16);

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_hal_cuda_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_cuda_ExecutableDef_table_t executable_def =
      iree_hal_cuda_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_cuda_ModuleDef_vec_t modules_vec =
      iree_hal_cuda_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_cuda_ModuleDef_vec_len(modules_vec);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_cuda_ModuleDef_table_t module_def =
        iree_hal_cuda_ModuleDef_vec_at(modules_vec, i);
    if (!module_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] is NULL", i);
    }
    if (flatbuffers_string_len(
            iree_hal_cuda_ModuleDef_ptx_image_get(module_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] contents are empty", i);
    }
  }

  iree_hal_cuda_ExportDef_vec_t exports_vec =
      iree_hal_cuda_ExecutableDef_exports_get(executable_def);
  for (iree_host_size_t i = 0; i < iree_hal_cuda_ExportDef_vec_len(exports_vec);
       ++i) {
    iree_hal_cuda_ExportDef_table_t export_def =
        iree_hal_cuda_ExportDef_vec_at(exports_vec, i);
    if (!export_def) continue;

    uint32_t module_ordinal =
        iree_hal_cuda_ExportDef_module_ordinal_get(export_def);
    if (module_ordinal >= module_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz
                              "] module_ordinal %u is out of bounds %" PRIhsz,
                              i, module_ordinal, module_count);
    }

    if (flatbuffers_string_len(
            iree_hal_cuda_ExportDef_kernel_name_get(export_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] name is empty", i);
    }

    if (iree_hal_cuda_ExportDef_block_dims_is_present(export_def)) {
      const iree_hal_cuda_BlockDims_t* block_dims =
          iree_hal_cuda_ExportDef_block_dims_get(export_def);
      if (block_dims->x > limits->max_block_dims[0] ||
          block_dims->y > limits->max_block_dims[1] ||
          block_dims->z > limits->max_block_dims[2]) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "exports[%" PRIhsz
            "] block dims %ux%ux%u exceeds device maximum %ux%ux%u",
            i, block_dims->x, block_dims->y, block_dims->z,
            limits->max_block_dims[0], limits->max_block_dims[1],
            limits->max_block_dims[2]);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] blocks dims are missing",
                              i);
    }

    uint32_t block_shared_memory_size =
        iree_hal_cuda_ExportDef_block_shared_memory_size_get(export_def);
    if (block_shared_memory_size > limits->max_block_shared_memory_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz
                              "] requires %uB of shared memory and "
                              "exceeds the device maximum of %uB per block",
                              i, block_shared_memory_size,
                              limits->max_block_shared_memory_size);
    }

    uint32_t constant_count =
        iree_hal_cuda_ExportDef_constant_count_get(export_def);
    if (constant_count > IREE_HAL_CUDA_MAX_DISPATCH_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] constant_count %u exceeds maximum of %u", i,
          constant_count, IREE_HAL_CUDA_MAX_DISPATCH_CONSTANT_COUNT);
    }

    iree_hal_cuda_BindingBits_vec_t binding_flags_vec =
        iree_hal_cuda_ExportDef_binding_flags_get(export_def);
    if (iree_hal_cuda_BindingBits_vec_len(binding_flags_vec) >
        IREE_HAL_CUDA_MAX_DISPATCH_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] binding_flags count %zu exceeds maximum of %u",
          i, iree_hal_cuda_BindingBits_vec_len(binding_flags_vec),
          IREE_HAL_CUDA_MAX_DISPATCH_BINDING_COUNT);
    }

    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_export_def(
        iree_hal_cuda_ExportDef_debug_info_get(export_def)));
  }

  return iree_ok_status();
}

iree_status_t iree_hal_cuda_native_executable_create(
    iree_hal_device_t* device, const iree_hal_cuda_dynamic_symbols_t* symbols,
    CUdevice cu_device, CUcontext cu_context,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;

  // TODO: move to the executable cache to avoid repeated queries.
  iree_hal_cuda_limits_t limits = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_query_limits(symbols, cu_device, &limits));

  // Read and strip the flatbuffer header prefix.
  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_read_executable_flatbuffer_header(
          executable_params->executable_data, /*unsafe_infer_size=*/false,
          iree_hal_cuda_ExecutableDef_file_identifier, &executable_flatbuffer));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_native_executable_flatbuffer_verify(
              executable_flatbuffer, &limits));

  iree_hal_cuda_ExecutableDef_table_t executable_def =
      iree_hal_cuda_ExecutableDef_as_root(executable_flatbuffer.data);

  iree_hal_cuda_ModuleDef_vec_t modules_vec =
      iree_hal_cuda_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count = iree_hal_cuda_ModuleDef_vec_len(modules_vec);
  iree_hal_cuda_ExportDef_vec_t exports_vec =
      iree_hal_cuda_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count = iree_hal_cuda_ExportDef_vec_len(exports_vec);

  // Calculate the total number of characters across all entry point names so
  // that we can store copies of the names as the flatbuffer storing the strings
  // may be released while the executable is still live.
  iree_host_size_t total_export_name_length = 0;
  iree_host_size_t total_export_info_length = 0;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_cuda_ExportDef_table_t export_def =
        iree_hal_cuda_ExportDef_vec_at(exports_vec, i);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            total_export_name_length,
            flatbuffers_string_len(
                iree_hal_cuda_ExportDef_kernel_name_get(export_def)),
            &total_export_name_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "export name storage size overflow");
    }
    IREE_TRACE({
      total_export_info_length += iree_hal_debug_calculate_export_info_size(
          iree_hal_cuda_ExportDef_debug_info_get(export_def));
    });
  }

  // Allocate storage for the executable and its associated data structures.
  iree_hal_cuda_native_executable_t* executable = NULL;
  iree_host_size_t module_table_size = 0;
  iree_host_size_t export_table_size = 0;
  iree_host_size_t total_size = sizeof(*executable);
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(module_count,
                                                sizeof(executable->modules[0]),
                                                &module_table_size) ||
                    !iree_host_size_checked_mul(export_count,
                                                sizeof(executable->exports[0]),
                                                &export_table_size) ||
                    !iree_host_size_checked_add(total_size, export_table_size,
                                                &total_size) ||
                    !iree_host_size_checked_add(total_size, module_table_size,
                                                &total_size) ||
                    !iree_host_size_checked_add(
                        total_size, total_export_name_length, &total_size) ||
                    !iree_host_size_checked_add(
                        total_size, total_export_info_length, &total_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable storage size overflow");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);

  iree_hal_resource_initialize(&iree_hal_cuda_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->device = device;
  executable->symbols = symbols;
  executable->cu_context = cu_context;
  executable->module_count = module_count;
  executable->modules = (CUmodule*)((uint8_t*)executable + sizeof(*executable) +
                                    export_table_size);
  iree_slim_mutex_initialize(&executable->global_mutex);
  executable->export_count = export_count;
  char* export_name_ptr = (char*)executable->modules + module_table_size;
  IREE_TRACE(uint8_t* export_info_ptr =
                 (uint8_t*)export_name_ptr + total_export_name_length);

  // Publish any embedded source files to the tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_cuda_ExecutableDef_source_files_get(executable_def));

  // Load each module first so that exports can reference them.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_cuda_ModuleDef_table_t module_def =
        iree_hal_cuda_ModuleDef_vec_at(modules_vec, i);

    // WARNING: CUDA doesn't take an expected length here so we can't bound it.
    // It's likely that users could craft inputs that read beyond the extents of
    // the embedded binary.
    flatbuffers_string_t ptx_image =
        iree_hal_cuda_ModuleDef_ptx_image_get(module_def);

    // TODO: pass cuJitOption values to get log info and other info back.
    // We pass the error buffer today but could use the info log to diagnose
    // performance warnings.
    char error_log[8192] = {0};
    CUjit_option jit_options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void* jit_option_values[] = {
        (void*)error_log,
        (void*)(uint32_t)sizeof(error_log),
    };
    CUmodule module = NULL;
    status = IREE_CURESULT_TO_STATUS(
        symbols,
        cuModuleLoadDataEx(&module, ptx_image, IREE_ARRAYSIZE(jit_options),
                           jit_options, jit_option_values),
        "cuModuleLoadDataEx");
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate(
          status,
          IREE_SV("mismatched target chip? missing/wrong bitcode directory?"));
      if (strlen(error_log) > 0) {
        status =
            iree_status_annotate(status, iree_make_cstring_view(error_log));
      }
      break;
    }

    executable->modules[i] = module;
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_hal_cuda_ExportDef_table_t export_def =
          iree_hal_cuda_ExportDef_vec_at(exports_vec, i);

      // Lookup the function in the module; this should always succeed but
      // we cannot trust that the input was generated by our compiler.
      uint32_t module_ordinal =
          iree_hal_cuda_ExportDef_module_ordinal_get(export_def);
      CUmodule module = executable->modules[module_ordinal];
      flatbuffers_string_t kernel_name =
          iree_hal_cuda_ExportDef_kernel_name_get(export_def);
      CUfunction function = NULL;
      status = IREE_CURESULT_TO_STATUS(
          symbols, cuModuleGetFunction(&function, module, kernel_name),
          "cuModuleGetFunction");
      if (!iree_status_is_ok(status)) break;
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "exports[%" PRIhsz
                                  "] kernel `%s` not found in modules[%u]",
                                  i, kernel_name, module_ordinal);
        break;
      }

      uint32_t block_shared_memory_size =
          iree_hal_cuda_ExportDef_block_shared_memory_size_get(export_def);
      status = IREE_CURESULT_TO_STATUS(
          symbols,
          cuFuncSetAttribute(function,
                             CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             block_shared_memory_size),
          "cuFuncSetAttribute");
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_cuda_kernel_params_t* kernel_info = &executable->exports[i];
      const iree_host_size_t kernel_name_length =
          flatbuffers_string_len(kernel_name);
      kernel_info->name =
          iree_make_string_view(export_name_ptr, kernel_name_length);
      memcpy(export_name_ptr, kernel_name, kernel_name_length);
      export_name_ptr += kernel_name_length;
      kernel_info->function = function;
      const iree_hal_cuda_BlockDims_t* block_dims =
          iree_hal_cuda_ExportDef_block_dims_get(export_def);
      kernel_info->block_dims[0] = block_dims->x;
      kernel_info->block_dims[1] = block_dims->y;
      kernel_info->block_dims[2] = block_dims->z;
      kernel_info->block_shared_memory_size =
          iree_hal_cuda_ExportDef_block_shared_memory_size_get(export_def);
      kernel_info->constant_count =
          iree_hal_cuda_ExportDef_constant_count_get(export_def);
      iree_hal_cuda_BindingBits_vec_t binding_flags_vec =
          iree_hal_cuda_ExportDef_binding_flags_get(export_def);
      kernel_info->binding_count =
          iree_hal_cuda_BindingBits_vec_len(binding_flags_vec);

      IREE_TRACE({
        iree_hal_debug_export_info_t* export_info =
            (iree_hal_debug_export_info_t*)export_info_ptr;
        export_info_ptr += iree_hal_debug_copy_export_info(
            iree_hal_cuda_ExportDef_debug_info_get(export_def), export_info);
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

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_hal_cuda_native_executable_global_t*
iree_hal_cuda_native_executable_find_global_locked(
    iree_hal_cuda_native_executable_t* executable, iree_string_view_t name) {
  for (iree_hal_cuda_native_executable_global_t* global =
           executable->global_list;
       global; global = global->next) {
    if (iree_string_view_equal(global->name, name)) return global;
  }
  return NULL;
}

static iree_hal_cuda_native_executable_global_t*
iree_hal_cuda_native_executable_global_from_handle_locked(
    iree_hal_cuda_native_executable_t* executable,
    iree_hal_executable_global_t global) {
  if (!iree_hal_executable_global_is_valid(global)) return NULL;
  iree_hal_cuda_native_executable_global_t* expected_global =
      (iree_hal_cuda_native_executable_global_t*)(uintptr_t)global.value;
  for (iree_hal_cuda_native_executable_global_t* current_global =
           executable->global_list;
       current_global; current_global = current_global->next) {
    if (current_global == expected_global) return current_global;
  }
  return NULL;
}

static iree_status_t iree_hal_cuda_native_executable_global_allocate(
    iree_hal_cuda_native_executable_t* executable, iree_string_view_t name,
    iree_device_size_t byte_length,
    iree_hal_cuda_native_executable_global_t** out_global) {
  *out_global = NULL;

  iree_host_size_t name_storage_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(name.size, 1, &name_storage_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "CUDA executable global name storage overflow");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t name_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_cuda_native_executable_global_t), &total_size,
      IREE_STRUCT_FIELD(name_storage_size, char, &name_offset)));

  iree_hal_cuda_native_executable_global_t* global = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executable->host_allocator,
                                             total_size, (void**)&global));
  memset(global, 0, total_size);

  global->name = iree_make_string_view((char*)global + name_offset, name.size);
  memcpy((void*)global->name.data, name.data, name.size);
  ((char*)global->name.data)[name.size] = 0;
  global->byte_length = byte_length;

  *out_global = global;
  return iree_ok_status();
}

static void iree_hal_cuda_native_executable_global_free(
    iree_hal_cuda_native_executable_t* executable,
    iree_hal_cuda_native_executable_global_t* global) {
  if (!global) return;
  iree_hal_buffer_release(global->buffer);
  iree_allocator_free(executable->host_allocator, global);
}

static void iree_hal_cuda_native_executable_global_list_free(
    iree_hal_cuda_native_executable_t* executable) {
  iree_hal_cuda_native_executable_global_t* global = executable->global_list;
  while (global) {
    iree_hal_cuda_native_executable_global_t* next_global = global->next;
    iree_hal_cuda_native_executable_global_free(executable, global);
    global = next_global;
  }
  executable->global_list = NULL;
}

static void iree_hal_cuda_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_native_executable_global_list_free(executable);
  iree_slim_mutex_deinitialize(&executable->global_mutex);

  for (iree_host_size_t i = 0; i < executable->module_count; ++i) {
    if (executable->modules[i]) {
      IREE_CUDA_IGNORE_ERROR(executable->symbols,
                             cuModuleUnload(executable->modules[i]));
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    const iree_hal_cuda_kernel_params_t** out_params) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->export_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable contains %" PRIhsz
                            " exports",
                            function.value, executable->export_count);
  }
  const uint32_t ordinal = iree_hal_executable_function_index(function);
  *out_params = &executable->exports[ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_cuda_native_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  return executable->export_count;
}

static iree_status_t iree_hal_cuda_native_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  const iree_hal_cuda_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_native_executable_lookup_kernel_params(
      base_executable, function, &kernel_params));
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = kernel_params->name;
  out_info->flags = IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE;
  out_info->constant_byte_length =
      kernel_params->constant_count * sizeof(uint32_t);
  out_info->binding_count = (uint16_t)kernel_params->binding_count;
  memcpy(out_info->workgroup_size, kernel_params->block_dims,
         sizeof(out_info->workgroup_size));
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_native_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  (void)executable;
  // TODO(cuda): return export parameter information from kernel metadata.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "parameter reflection not implemented");
}

static iree_status_t iree_hal_cuda_native_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    if (iree_string_view_equal(executable->exports[i].name, name)) {
      *out_export_ordinal =
          iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "function '%.*s' not found in executable",
                          (int)name.size, name.data);
}

#define IREE_HAL_CUDA_MAX_STACK_GLOBAL_NAME_LENGTH \
  ((iree_host_size_t)(4 * 1024))

static iree_status_t iree_hal_cuda_native_executable_try_query_global(
    iree_hal_cuda_native_executable_t* executable, iree_string_view_t name,
    bool* out_found, CUdeviceptr* out_global_device_ptr,
    iree_device_size_t* out_byte_length) {
  *out_found = false;
  if (out_global_device_ptr) *out_global_device_ptr = 0;
  *out_byte_length = 0;

  IREE_RETURN_IF_ERROR(
      IREE_CURESULT_TO_STATUS(executable->symbols,
                              cuCtxSetCurrent(executable->cu_context)),
      "setting CUDA context for executable global lookup");

  char* global_name = (char*)iree_alloca(name.size + 1);
  memcpy(global_name, name.data, name.size);
  global_name[name.size] = 0;

  CUresult terminal_result = CUDA_ERROR_NOT_FOUND;
  CUdeviceptr global_device_ptr = 0;
  size_t global_size = 0;
  for (iree_host_size_t module_ordinal = 0;
       module_ordinal < executable->module_count; ++module_ordinal) {
    terminal_result = executable->symbols->cuModuleGetGlobal(
        &global_device_ptr, &global_size, executable->modules[module_ordinal],
        global_name);
    if (terminal_result == CUDA_SUCCESS) break;
    if (terminal_result != CUDA_ERROR_NOT_FOUND) {
      return iree_hal_cuda_result_to_status(
          executable->symbols, terminal_result, __FILE__, __LINE__);
    }
  }
  if (terminal_result != CUDA_SUCCESS) return iree_ok_status();

  *out_found = true;
  if (out_global_device_ptr) *out_global_device_ptr = global_device_ptr;
  *out_byte_length = (iree_device_size_t)global_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_native_executable_validate_global_name(
    iree_string_view_t name) {
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable global name is empty");
  }
  if (name.size > IREE_HAL_CUDA_MAX_STACK_GLOBAL_NAME_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable global name `%.*s` exceeds maximum length %" PRIhsz,
        (int)name.size, name.data, IREE_HAL_CUDA_MAX_STACK_GLOBAL_NAME_LENGTH);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_native_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();

  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_native_executable_validate_global_name(name));

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_cuda_native_executable_global_t* global =
      iree_hal_cuda_native_executable_find_global_locked(executable, name);
  if (global) {
    *out_found = true;
    *out_global =
        iree_hal_executable_global_from_value((uint64_t)(uintptr_t)global);
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_ok_status();
  }
  iree_slim_mutex_unlock(&executable->global_mutex);

  iree_device_size_t byte_length = 0;
  bool query_found = false;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_native_executable_try_query_global(
      executable, name, &query_found, /*out_global_device_ptr=*/NULL,
      &byte_length));
  if (!query_found) return iree_ok_status();

  iree_hal_cuda_native_executable_global_t* new_global = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_native_executable_global_allocate(
      executable, name, byte_length, &new_global));

  iree_slim_mutex_lock(&executable->global_mutex);
  global = iree_hal_cuda_native_executable_find_global_locked(executable, name);
  if (global) {
    *out_found = true;
    *out_global =
        iree_hal_executable_global_from_value((uint64_t)(uintptr_t)global);
  } else {
    new_global->next = executable->global_list;
    executable->global_list = new_global;
    *out_found = true;
    *out_global =
        iree_hal_executable_global_from_value((uint64_t)(uintptr_t)new_global);
    new_global = NULL;
  }
  iree_slim_mutex_unlock(&executable->global_mutex);

  iree_hal_cuda_native_executable_global_free(executable, new_global);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_native_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  memset(out_info, 0, sizeof(*out_info));

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_cuda_native_executable_global_t* global_entry =
      iree_hal_cuda_native_executable_global_from_handle_locked(executable,
                                                                global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid CUDA executable global handle");
  }
  out_info->name = global_entry->name;
  out_info->byte_length = global_entry->byte_length;
  iree_slim_mutex_unlock(&executable->global_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_native_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_cuda_native_executable_t* executable =
      iree_hal_cuda_native_executable_cast(base_executable);
  *out_buffer = NULL;

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_cuda_native_executable_global_t* global_entry =
      iree_hal_cuda_native_executable_global_from_handle_locked(executable,
                                                                global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid CUDA executable global handle");
  }
  if (global_entry->buffer) {
    *out_buffer = global_entry->buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_ok_status();
  }
  iree_string_view_t name = global_entry->name;
  iree_device_size_t expected_byte_length = global_entry->byte_length;
  iree_slim_mutex_unlock(&executable->global_mutex);

  CUdeviceptr global_device_ptr = 0;
  iree_device_size_t byte_length = 0;
  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_native_executable_try_query_global(
      executable, name, &found, &global_device_ptr, &byte_length));
  if (IREE_UNLIKELY(!found)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "verified executable global `%.*s` disappeared",
                            (int)name.size, name.data);
  }
  if (IREE_UNLIKELY(byte_length != expected_byte_length)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "verified executable global `%.*s` changed size from %" PRIu64
        " to %" PRIu64,
        (int)name.size, name.data, (uint64_t)expected_byte_length,
        (uint64_t)byte_length);
  }

  iree_hal_buffer_placement_t placement = {
      .device = executable->device,
      .queue_affinity = iree_hal_queue_affinity_is_empty(queue_affinity)
                            ? IREE_HAL_QUEUE_AFFINITY_ANY
                            : queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_t* new_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_buffer_wrap(
      placement, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_DEFAULT, byte_length, /*byte_offset=*/0,
      byte_length, IREE_HAL_CUDA_BUFFER_TYPE_EXTERNAL, global_device_ptr,
      /*host_ptr=*/NULL, iree_hal_buffer_release_callback_null(),
      executable->host_allocator, &new_buffer));

  iree_slim_mutex_lock(&executable->global_mutex);
  global_entry = iree_hal_cuda_native_executable_global_from_handle_locked(
      executable, global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    iree_hal_buffer_release(new_buffer);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid CUDA executable global handle");
  }
  if (global_entry->buffer) {
    *out_buffer = global_entry->buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
    iree_hal_buffer_release(new_buffer);
  } else {
    global_entry->buffer = new_buffer;
    *out_buffer = new_buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
  }
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_cuda_native_executable_vtable = {
        .destroy = iree_hal_cuda_native_executable_destroy,
        .function_count = iree_hal_cuda_native_executable_export_count,
        .function_info = iree_hal_cuda_native_executable_export_info,
        .function_parameters =
            iree_hal_cuda_native_executable_export_parameters,
        .lookup_function_by_name =
            iree_hal_cuda_native_executable_lookup_export_by_name,
        .try_lookup_global_by_name =
            iree_hal_cuda_native_executable_try_lookup_global_by_name,
        .global_info = iree_hal_cuda_native_executable_global_info,
        .global_buffer = iree_hal_cuda_native_executable_global_buffer,
};
