// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/hip/context_util.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/executable/format.h"
#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/status_util.h"

typedef struct iree_hal_hip_native_executable_per_device_data_t {
  // HIP context owning all modules in this per-device table.
  hipCtx_t hip_context;

  // Number of loaded HIP modules.
  iree_host_size_t module_count;
  // Loaded HIP modules.
  hipModule_t* modules;

  // Number of exported kernels referencing the loaded modules.
  iree_host_size_t export_count;
  // Exported kernels referencing the loaded modules.
  iree_hal_hip_kernel_params_t exports[];
} iree_hal_hip_native_executable_per_device_data_t;

typedef struct iree_hal_hip_native_executable_global_t {
  // Next executable-owned global entry.
  struct iree_hal_hip_native_executable_global_t* next;

  // Persistent executable-owned global name.
  iree_string_view_t name;

  // Byte length verified against the first loaded HIP device.
  iree_device_size_t byte_length;

  // One executable-owned buffer alias per loaded HIP device.
  iree_hal_buffer_t** device_buffers;
} iree_hal_hip_native_executable_global_t;

typedef struct iree_hal_hip_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  // Host allocator used for executable lifetime.
  iree_allocator_t host_allocator;

  // Borrowed HAL device used for buffer placement metadata.
  iree_hal_device_t* device;
  // Borrowed HIP dynamic symbols used for module and global lookup.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  // Number of HIP devices this executable was loaded onto.
  iree_host_size_t num_devices;

  // Guards executable-owned global entry and buffer publication.
  iree_slim_mutex_t global_mutex;

  // Executable-owned globals interned by name.
  iree_hal_hip_native_executable_global_t* global_list;

  // Per-device module and export tables.
  iree_hal_hip_native_executable_per_device_data_t* per_device_data[];
} iree_hal_hip_native_executable_t;

static const iree_hal_executable_vtable_t iree_hal_hip_native_executable_vtable;

static iree_hal_hip_native_executable_t* iree_hal_hip_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_native_executable_vtable);
  return (iree_hal_hip_native_executable_t*)base_value;
}

typedef struct iree_hal_hip_limits_t {
  // Maximum block dimensions along each axis.
  uint32_t max_block_dims[3];
  // Maximum number of threads in one block.
  uint32_t max_block_thread_count;
  // Maximum static shared memory size in bytes per block.
  uint32_t max_block_shared_memory_size;
} iree_hal_hip_limits_t;
static iree_status_t iree_hal_hip_query_limits(
    const iree_hal_hip_dynamic_symbols_t* symbols, hipDevice_t device,
    iree_hal_hip_limits_t* out_limits) {
  memset(out_limits, 0, sizeof(*out_limits));

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[0],
                            hipDeviceAttributeMaxBlockDimX, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[1],
                            hipDeviceAttributeMaxBlockDimY, device),
      "hipDeviceGetAttribute");
  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_dims[2],
                            hipDeviceAttributeMaxBlockDimZ, device),
      "hipDeviceGetAttribute");

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_thread_count,
                            hipDeviceAttributeMaxThreadsPerBlock, device),
      "hipDeviceGetAttribute");

  IREE_HIP_RETURN_IF_ERROR(
      symbols,
      hipDeviceGetAttribute((int32_t*)&out_limits->max_block_shared_memory_size,
                            hipDeviceAttributeMaxSharedMemoryPerBlock, device),
      "hipDeviceGetAttribute");

  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_verify(
    const iree_hal_hip_executable_format_t* format,
    const iree_hal_hip_limits_t* limits,
    iree_host_size_t* out_export_name_storage_size) {
  *out_export_name_storage_size = 0;
  for (iree_host_size_t i = 0; i < format->export_count; ++i) {
    iree_hal_hip_executable_export_t export_def;
    IREE_RETURN_IF_ERROR(
        iree_hal_hip_executable_format_read_export(format, i, &export_def));

    if (export_def.block_size[0] > limits->max_block_dims[0] ||
        export_def.block_size[1] > limits->max_block_dims[1] ||
        export_def.block_size[2] > limits->max_block_dims[2]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP executable export[%" PRIhsz
          "] block size %ux%ux%u exceeds device maximum %ux%ux%u",
          i, export_def.block_size[0], export_def.block_size[1],
          export_def.block_size[2], limits->max_block_dims[0],
          limits->max_block_dims[1], limits->max_block_dims[2]);
    }
    const uint64_t block_thread_count = (uint64_t)export_def.block_size[0] *
                                        export_def.block_size[1] *
                                        export_def.block_size[2];
    if (block_thread_count > limits->max_block_thread_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP executable export[%" PRIhsz "] block contains %" PRIu64
          " threads, exceeding the device maximum of %u",
          i, block_thread_count, limits->max_block_thread_count);
    }
    if (export_def.constant_count > IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIP executable export[%" PRIhsz
                              "] constant count %u exceeds maximum of %u",
                              i, export_def.constant_count,
                              IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT);
    }
    if (export_def.binding_count > IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIP executable export[%" PRIhsz
                              "] binding count %u exceeds maximum of %u",
                              i, export_def.binding_count,
                              IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT);
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            *out_export_name_storage_size, export_def.kernel_name.size,
            out_export_name_storage_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "HIP executable export name storage exceeds "
                              "host size limits");
    }
  }
  return iree_ok_status();
}

// Verifies a function against the device limits so that we can avoid doing so
// during runtime.
static iree_status_t iree_hal_hip_function_attributes_verify(
    iree_host_size_t id, const iree_hal_hip_dynamic_symbols_t* symbols,
    hipFunction_t function, uint64_t block_thread_count,
    const iree_hal_hip_limits_t* limits) {
  int max_block_thread_count = 0;
  IREE_RETURN_IF_ERROR(IREE_HIP_CALL_TO_STATUS(
      symbols,
      hipFuncGetAttribute(
          &max_block_thread_count,
          (hipFuncAttribute)HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
      "hipFuncGetAttribute"));
  if (IREE_UNLIKELY(max_block_thread_count <= 0)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "HIP executable export[%" PRIhsz
        "] kernel reported invalid maximum block thread count %d",
        id, max_block_thread_count);
  }
  if (block_thread_count > (uint32_t)max_block_thread_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP executable export[%" PRIhsz "] block contains %" PRIu64
        " threads, exceeding the kernel maximum of %u",
        id, block_thread_count, (uint32_t)max_block_thread_count);
  }

  int block_shared_memory_size;
  IREE_RETURN_IF_ERROR(IREE_HIP_CALL_TO_STATUS(
      symbols,
      hipFuncGetAttribute(
          &block_shared_memory_size,
          (hipFuncAttribute)HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
      "hipFuncGetAttribute"));
  if (IREE_UNLIKELY(block_shared_memory_size < 0)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "HIP executable export[%" PRIhsz
                            "] kernel reported invalid shared memory size %d",
                            id, block_shared_memory_size);
  }
  if ((uint32_t)block_shared_memory_size >
      limits->max_block_shared_memory_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP executable export[%" PRIhsz
        "] requires %uB of shared memory, exceeding the device maximum of "
        "%uB per block",
        id, (uint32_t)block_shared_memory_size,
        limits->max_block_shared_memory_size);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hip_native_executable_create(
    iree_hal_device_t* device, const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_device_topology_t topology,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  if (topology.count < 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "at least one device is required but none were provided");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;

  if (IREE_UNLIKELY(load_params->constant_count != 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HIP executable specialization constants are not supported");
  }

  iree_hal_hip_executable_format_t executable_format;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_executable_format_parse(load_params->executable_data,
                                               &executable_format));

  // TODO: cache immutable HIP limits on the device to avoid repeated queries.
  iree_hal_hip_limits_t limits = {0};
  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    iree_hal_hip_limits_t device_limits = {0};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_query_limits(symbols, topology.devices[i].hip_device,
                                      &device_limits));
    if (i == 0) {
      limits = device_limits;
    } else {
      for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(limits.max_block_dims);
           ++j) {
        limits.max_block_dims[j] =
            iree_min(limits.max_block_dims[j], device_limits.max_block_dims[j]);
      }
      limits.max_block_thread_count = iree_min(
          limits.max_block_thread_count, device_limits.max_block_thread_count);
      limits.max_block_shared_memory_size =
          iree_min(limits.max_block_shared_memory_size,
                   device_limits.max_block_shared_memory_size);
    }
  }

  iree_host_size_t total_export_name_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_native_executable_verify(&executable_format, &limits,
                                                &total_export_name_length));
  const iree_host_size_t module_count = executable_format.module_count;
  const iree_host_size_t export_count = executable_format.export_count;

  // Allocate storage for the executable and its associated data structures.
  iree_hal_hip_native_executable_t* executable = NULL;
  iree_host_size_t module_table_size = 0;
  iree_host_size_t export_table_size = 0;
  iree_host_size_t native_executable_device_info_size =
      sizeof(*executable->per_device_data[0]);
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              module_count, sizeof(executable->per_device_data[0]->modules[0]),
              &module_table_size) ||
          !iree_host_size_checked_mul(
              export_count, sizeof(executable->per_device_data[0]->exports[0]),
              &export_table_size) ||
          !iree_host_size_checked_add(native_executable_device_info_size,
                                      export_table_size,
                                      &native_executable_device_info_size) ||
          !iree_host_size_checked_add(native_executable_device_info_size,
                                      module_table_size,
                                      &native_executable_device_info_size) ||
          !iree_host_size_checked_add(native_executable_device_info_size,
                                      total_export_name_length,
                                      &native_executable_device_info_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "HIP executable metadata storage exceeds host "
                             "size limits"));
  }
  if (IREE_UNLIKELY(native_executable_device_info_size >
                    IREE_HOST_SIZE_MAX - iree_max_align_t)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "HIP executable metadata storage exceeds host "
                             "size limits"));
  }
  native_executable_device_info_size =
      iree_host_align(native_executable_device_info_size, iree_max_align_t);
  iree_host_size_t per_device_data_size = 0;
  iree_host_size_t native_executable_device_infos_size = 0;
  iree_host_size_t total_size = sizeof(*executable);
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(topology.count,
                                      sizeof(executable->per_device_data[0]),
                                      &per_device_data_size) ||
          !iree_host_size_checked_mul(topology.count,
                                      native_executable_device_info_size,
                                      &native_executable_device_infos_size) ||
          !iree_host_size_checked_add(total_size, per_device_data_size,
                                      &total_size) ||
          !iree_host_size_checked_add(
              total_size, native_executable_device_infos_size, &total_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "HIP executable metadata storage exceeds host "
                             "size limits"));
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_hip_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->device = device;
  executable->symbols = symbols;
  executable->num_devices = topology.count;
  iree_slim_mutex_initialize(&executable->global_mutex);
  const uint8_t* per_device_data_location =
      (uint8_t*)executable + sizeof(*executable);

  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    const iree_host_size_t native_executable_device_info_size_offset =
        (i * native_executable_device_info_size);

    executable->per_device_data[i] =
        (iree_hal_hip_native_executable_per_device_data_t*)(per_device_data_location +
                                                            per_device_data_size +
                                                            native_executable_device_info_size_offset);
    iree_hal_hip_native_executable_per_device_data_t* per_device_data =
        executable->per_device_data[i];
    per_device_data->hip_context = topology.devices[i].hip_context;
    per_device_data->module_count = module_count;
    per_device_data->modules =
        (hipModule_t*)((uint8_t*)per_device_data + sizeof(*per_device_data) +
                       export_table_size);
    per_device_data->export_count = export_count;
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t j = 0; j < topology.count && iree_status_is_ok(status);
       ++j) {
    status = IREE_HIP_CALL_TO_STATUS(
        symbols, hipCtxPushCurrent(topology.devices[j].hip_context));
    if (!iree_status_is_ok(status)) {
      break;
    }

    iree_hal_hip_native_executable_per_device_data_t* per_device_data =
        executable->per_device_data[j];
    char* export_name_ptr = (char*)per_device_data->modules + module_table_size;

    // Load each module first so that exports can reference them.
    for (iree_host_size_t i = 0; i < module_count && iree_status_is_ok(status);
         ++i) {
      iree_hal_hip_executable_module_t module_def;
      status = iree_hal_hip_executable_format_read_module(&executable_format, i,
                                                          &module_def);
      if (!iree_status_is_ok(status)) break;

      // HIP takes no image length and casts the input to aligned ELF structs
      // while deriving one. The parser proved all inspected ELF extents fit;
      // clone the exact span so the vendor call also receives aligned storage.
      void* aligned_hsaco_image = NULL;
      status = iree_allocator_clone(host_allocator, module_def.hsaco_image,
                                    &aligned_hsaco_image);
      if (!iree_status_is_ok(status)) break;

      // TODO: pass hipJitOption values to get log info and other info back.
      // We pass the error buffer today but could use the info log to diagnose
      // performance warnings.
      char error_log[8192] = {0};
      hipJitOption jit_options[] = {
          hipJitOptionErrorLogBuffer,
          hipJitOptionErrorLogBufferSizeBytes,
      };
      void* jit_option_values[] = {
          (void*)error_log,
          (void*)(uint32_t)sizeof(error_log),
      };
      hipModule_t module = NULL;
      status = IREE_HIP_CALL_TO_STATUS(
          symbols,
          hipModuleLoadDataEx(&module, aligned_hsaco_image,
                              IREE_ARRAYSIZE(jit_options), jit_options,
                              jit_option_values),
          "hipModuleLoadDataEx");
      iree_allocator_free(host_allocator, aligned_hsaco_image);
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate(
            status,
            IREE_SV(
                "mismatched target chip? missing/wrong bitcode directory?"));
        if (strlen(error_log) > 0) {
          status =
              iree_status_annotate(status, iree_make_cstring_view(error_log));
        }
        break;
      }

      per_device_data->modules[i] = module;
    }

    for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
         ++i) {
      iree_hal_hip_executable_export_t export_def;
      status = iree_hal_hip_executable_format_read_export(&executable_format, i,
                                                          &export_def);
      if (!iree_status_is_ok(status)) break;

      // Lookup the function in the module; this should always succeed but
      // we cannot trust that the input was generated by our compiler.
      hipModule_t module = per_device_data->modules[export_def.module_ordinal];
      hipFunction_t function = NULL;
      status = IREE_HIP_CALL_TO_STATUS(
          symbols,
          hipModuleGetFunction(&function, module, export_def.kernel_name.data),
          "hipModuleGetFunction");
      if (!iree_status_is_ok(status)) break;
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "HIP executable export[%" PRIhsz
                                  "] kernel `%s` not found in module[%u]",
                                  i, export_def.kernel_name.data,
                                  export_def.module_ordinal);
        break;
      }

      const uint64_t block_thread_count = (uint64_t)export_def.block_size[0] *
                                          export_def.block_size[1] *
                                          export_def.block_size[2];
      status = iree_hal_hip_function_attributes_verify(
          i, symbols, function, block_thread_count, &limits);
      if (!iree_status_is_ok(status)) break;

      // Package required parameters for kernel launches for each entry point.
      iree_hal_hip_kernel_params_t* kernel_info = &per_device_data->exports[i];
      kernel_info->name =
          iree_make_string_view(export_name_ptr, export_def.kernel_name.size);
      memcpy(export_name_ptr, export_def.kernel_name.data,
             export_def.kernel_name.size);
      export_name_ptr += export_def.kernel_name.size;
      kernel_info->function = function;
      memcpy(kernel_info->block_dims, export_def.block_size,
             sizeof(kernel_info->block_dims));
      kernel_info->constant_count = export_def.constant_count;
      kernel_info->binding_count = export_def.binding_count;
    }
    status = iree_status_join(
        status, IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_hal_hip_native_executable_global_t*
iree_hal_hip_native_executable_find_global_locked(
    iree_hal_hip_native_executable_t* executable, iree_string_view_t name) {
  for (iree_hal_hip_native_executable_global_t* global =
           executable->global_list;
       global; global = global->next) {
    if (iree_string_view_equal(global->name, name)) return global;
  }
  return NULL;
}

static iree_hal_hip_native_executable_global_t*
iree_hal_hip_native_executable_global_from_handle_locked(
    iree_hal_hip_native_executable_t* executable,
    iree_hal_executable_global_t global) {
  if (!iree_hal_executable_global_is_valid(global)) return NULL;
  iree_hal_hip_native_executable_global_t* expected_global =
      (iree_hal_hip_native_executable_global_t*)(uintptr_t)global.value;
  for (iree_hal_hip_native_executable_global_t* current_global =
           executable->global_list;
       current_global; current_global = current_global->next) {
    if (current_global == expected_global) return current_global;
  }
  return NULL;
}

static iree_status_t iree_hal_hip_native_executable_global_allocate(
    iree_hal_hip_native_executable_t* executable, iree_string_view_t name,
    iree_device_size_t byte_length,
    iree_hal_hip_native_executable_global_t** out_global) {
  *out_global = NULL;

  iree_host_size_t name_storage_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(name.size, 1, &name_storage_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HIP executable global name storage overflow");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t device_buffers_offset = 0;
  iree_host_size_t name_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_hip_native_executable_global_t), &total_size,
      IREE_STRUCT_FIELD(executable->num_devices, iree_hal_buffer_t*,
                        &device_buffers_offset),
      IREE_STRUCT_FIELD(name_storage_size, char, &name_offset)));

  iree_hal_hip_native_executable_global_t* global = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executable->host_allocator,
                                             total_size, (void**)&global));
  memset(global, 0, total_size);

  global->name = iree_make_string_view((char*)global + name_offset, name.size);
  memcpy((void*)global->name.data, name.data, name.size);
  ((char*)global->name.data)[name.size] = 0;
  global->byte_length = byte_length;
  global->device_buffers =
      (iree_hal_buffer_t**)((uint8_t*)global + device_buffers_offset);

  *out_global = global;
  return iree_ok_status();
}

static void iree_hal_hip_native_executable_global_free(
    iree_hal_hip_native_executable_t* executable,
    iree_hal_hip_native_executable_global_t* global) {
  if (!global) return;
  for (iree_host_size_t i = 0; i < executable->num_devices; ++i) {
    iree_hal_buffer_release(global->device_buffers[i]);
  }
  iree_allocator_free(executable->host_allocator, global);
}

static void iree_hal_hip_native_executable_global_list_free(
    iree_hal_hip_native_executable_t* executable) {
  iree_hal_hip_native_executable_global_t* global = executable->global_list;
  while (global) {
    iree_hal_hip_native_executable_global_t* next_global = global->next;
    iree_hal_hip_native_executable_global_free(executable, global);
    global = next_global;
  }
  executable->global_list = NULL;
}

static void iree_hal_hip_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_native_executable_global_list_free(executable);
  iree_slim_mutex_deinitialize(&executable->global_mutex);

  for (iree_host_size_t i = 0; i < executable->num_devices; ++i) {
    const iree_hal_hip_native_executable_per_device_data_t* data =
        executable->per_device_data[i];
    for (iree_host_size_t j = 0; j < data->module_count; ++j) {
      if (data->modules[j]) {
        IREE_HIP_IGNORE_ERROR(executable->symbols,
                              hipModuleUnload(data->modules[j]));
      }
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hip_kernel_params_t** out_params) {
  *out_params = NULL;
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  int device_ordinal = 0;
  if (queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  }
  if (device_ordinal >= executable->num_devices) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "affinity for non-existent queue was provided.");
  }

  const iree_hal_hip_native_executable_per_device_data_t* data =
      executable->per_device_data[device_ordinal];
  if (!iree_hal_executable_function_is_index_in_range(function,
                                                      data->export_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable contains %" PRIhsz
                            " exports",
                            function.value, data->export_count);
  }
  const uint32_t ordinal = iree_hal_executable_function_index(function);
  *out_params = &data->exports[ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_hip_native_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  return executable->per_device_data[0]->export_count;
}

static iree_status_t iree_hal_hip_native_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  const iree_hal_hip_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_hip_native_executable_lookup_kernel_params(
      base_executable, function, /*queue_affinity=*/0, &kernel_params));
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

static iree_status_t iree_hal_hip_native_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  (void)executable;
  // TODO(hip): return export parameter information from kernel metadata.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "parameter reflection not implemented");
}

static iree_status_t iree_hal_hip_native_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  const iree_hal_hip_native_executable_per_device_data_t* data =
      executable->per_device_data[0];
  for (iree_host_size_t i = 0; i < data->export_count; ++i) {
    if (iree_string_view_equal(data->exports[i].name, name)) {
      *out_export_ordinal =
          iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "function '%.*s' not found in executable",
                          (int)name.size, name.data);
}

#define IREE_HAL_HIP_MAX_STACK_GLOBAL_NAME_LENGTH ((iree_host_size_t)(4 * 1024))

static bool iree_hal_hip_native_executable_is_global_not_found(
    hipError_t result) {
  return result == hipErrorNotFound ||
         result == hipErrorSharedObjectSymbolNotFound;
}

static iree_status_t iree_hal_hip_native_executable_select_global_device(
    const iree_hal_hip_native_executable_t* executable,
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t* out_device_ordinal,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  *out_device_ordinal = 0;
  *out_queue_affinity = 0;

  iree_hal_queue_affinity_t available_affinity =
      executable->num_devices == IREE_HAL_MAX_QUEUES
          ? IREE_HAL_QUEUE_AFFINITY_ANY
          : (((iree_hal_queue_affinity_t)1) << executable->num_devices) - 1;
  iree_hal_queue_affinity_t selected_affinity = queue_affinity;
  if (iree_hal_queue_affinity_is_empty(selected_affinity) ||
      iree_hal_queue_affinity_is_any(selected_affinity)) {
    selected_affinity = available_affinity;
  } else {
    iree_hal_queue_affinity_and_into(selected_affinity, available_affinity);
  }
  if (iree_hal_queue_affinity_is_empty(selected_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "global lookup queue affinity 0x%" PRIx64
                            " does not select a loaded HIP device",
                            queue_affinity);
  }

  iree_host_size_t device_ordinal =
      iree_hal_queue_affinity_find_first_set(selected_affinity);
  *out_device_ordinal = device_ordinal;
  *out_queue_affinity = ((iree_hal_queue_affinity_t)1) << device_ordinal;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_try_query_global(
    iree_hal_hip_native_executable_t* executable, iree_string_view_t name,
    iree_host_size_t device_ordinal, bool* out_found,
    hipDeviceptr_t* out_global_device_ptr,
    iree_device_size_t* out_byte_length) {
  *out_found = false;
  if (out_global_device_ptr) *out_global_device_ptr = 0;
  *out_byte_length = 0;

  const iree_hal_hip_native_executable_per_device_data_t* per_device_data =
      executable->per_device_data[device_ordinal];
  IREE_RETURN_IF_ERROR(iree_hal_hip_set_context(executable->symbols,
                                                per_device_data->hip_context),
                       "setting HIP context for executable global lookup");

  char* global_name = (char*)iree_alloca(name.size + 1);
  memcpy(global_name, name.data, name.size);
  global_name[name.size] = 0;

  hipError_t terminal_result = hipErrorNotFound;
  hipDeviceptr_t global_device_ptr = 0;
  size_t global_size = 0;
  for (iree_host_size_t module_ordinal = 0;
       module_ordinal < per_device_data->module_count; ++module_ordinal) {
    terminal_result = executable->symbols->hipModuleGetGlobal(
        &global_device_ptr, &global_size,
        per_device_data->modules[module_ordinal], global_name);
    if (terminal_result == hipSuccess) break;
    if (!iree_hal_hip_native_executable_is_global_not_found(terminal_result)) {
      return IREE_HIP_RESULT_TO_STATUS(executable->symbols, terminal_result);
    }
  }
  if (terminal_result != hipSuccess) return iree_ok_status();

  *out_found = true;
  if (out_global_device_ptr) *out_global_device_ptr = global_device_ptr;
  *out_byte_length = (iree_device_size_t)global_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_validate_global_name(
    iree_string_view_t name) {
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable global name is empty");
  }
  if (name.size > IREE_HAL_HIP_MAX_STACK_GLOBAL_NAME_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable global name `%.*s` exceeds maximum length %" PRIhsz,
        (int)name.size, name.data, IREE_HAL_HIP_MAX_STACK_GLOBAL_NAME_LENGTH);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();

  IREE_RETURN_IF_ERROR(
      iree_hal_hip_native_executable_validate_global_name(name));

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_hip_native_executable_global_t* global =
      iree_hal_hip_native_executable_find_global_locked(executable, name);
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
  IREE_RETURN_IF_ERROR(iree_hal_hip_native_executable_try_query_global(
      executable, name, /*device_ordinal=*/0, &query_found,
      /*out_global_device_ptr=*/NULL, &byte_length));
  if (!query_found) return iree_ok_status();

  iree_hal_hip_native_executable_global_t* new_global = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_hip_native_executable_global_allocate(
      executable, name, byte_length, &new_global));

  iree_slim_mutex_lock(&executable->global_mutex);
  global = iree_hal_hip_native_executable_find_global_locked(executable, name);
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

  iree_hal_hip_native_executable_global_free(executable, new_global);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  memset(out_info, 0, sizeof(*out_info));

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_hip_native_executable_global_t* global_entry =
      iree_hal_hip_native_executable_global_from_handle_locked(executable,
                                                               global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP executable global handle");
  }
  out_info->name = global_entry->name;
  out_info->byte_length = global_entry->byte_length;
  iree_slim_mutex_unlock(&executable->global_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_native_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_hip_native_executable_t* executable =
      iree_hal_hip_native_executable_cast(base_executable);
  *out_buffer = NULL;

  iree_host_size_t device_ordinal = 0;
  iree_hal_queue_affinity_t selected_queue_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hip_native_executable_select_global_device(
      executable, queue_affinity, &device_ordinal, &selected_queue_affinity));

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_hip_native_executable_global_t* global_entry =
      iree_hal_hip_native_executable_global_from_handle_locked(executable,
                                                               global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP executable global handle");
  }
  iree_hal_buffer_t* cached_buffer =
      global_entry->device_buffers[device_ordinal];
  if (cached_buffer) {
    *out_buffer = cached_buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_ok_status();
  }
  iree_string_view_t name = global_entry->name;
  iree_device_size_t expected_byte_length = global_entry->byte_length;
  iree_slim_mutex_unlock(&executable->global_mutex);

  hipDeviceptr_t global_device_ptr = 0;
  iree_device_size_t byte_length = 0;
  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_hip_native_executable_try_query_global(
      executable, name, device_ordinal, &found, &global_device_ptr,
      &byte_length));
  if (IREE_UNLIKELY(!found)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "verified executable global `%.*s` disappeared on HIP device %" PRIhsz,
        (int)name.size, name.data, device_ordinal);
  }
  if (IREE_UNLIKELY(byte_length != expected_byte_length)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "verified executable global `%.*s` changed size on HIP device %" PRIhsz
        " from %" PRIu64 " to %" PRIu64,
        (int)name.size, name.data, device_ordinal,
        (uint64_t)expected_byte_length, (uint64_t)byte_length);
  }

  iree_hal_buffer_placement_t placement = {
      .device = executable->device,
      .queue_affinity = selected_queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_t* new_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_hip_buffer_wrap(
      placement, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_DEFAULT, byte_length, /*byte_offset=*/0,
      byte_length, IREE_HAL_HIP_BUFFER_TYPE_EXTERNAL, global_device_ptr,
      /*host_ptr=*/NULL, iree_hal_buffer_release_callback_null(),
      executable->host_allocator, &new_buffer));

  iree_slim_mutex_lock(&executable->global_mutex);
  global_entry = iree_hal_hip_native_executable_global_from_handle_locked(
      executable, global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    iree_hal_buffer_release(new_buffer);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid HIP executable global handle");
  }
  cached_buffer = global_entry->device_buffers[device_ordinal];
  if (cached_buffer) {
    *out_buffer = cached_buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
    iree_hal_buffer_release(new_buffer);
  } else {
    global_entry->device_buffers[device_ordinal] = new_buffer;
    *out_buffer = new_buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
  }
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_hip_native_executable_vtable = {
        .destroy = iree_hal_hip_native_executable_destroy,
        .function_count = iree_hal_hip_native_executable_export_count,
        .function_info = iree_hal_hip_native_executable_export_info,
        .function_parameters = iree_hal_hip_native_executable_export_parameters,
        .lookup_function_by_name =
            iree_hal_hip_native_executable_lookup_export_by_name,
        .try_lookup_global_by_name =
            iree_hal_hip_native_executable_try_lookup_global_by_name,
        .global_info = iree_hal_hip_native_executable_global_info,
        .global_buffer = iree_hal_hip_native_executable_global_buffer,
};
