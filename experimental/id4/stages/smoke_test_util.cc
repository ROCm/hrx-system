// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/smoke_test_util.h"

#include <cstring>

#include "experimental/id4/stages/smoke.h"
#include "iree/hal/local/local_executable.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"

namespace id4::test {

const char kSmokeKernelSource[] = R"(
config.decl @id4.smoke.workgroups_x : %value: index where [range(%value, 1, 8)]
config.decl @id4.smoke.workgroup_size_x : %value: index where [range(%value, 1, 256)]

kernel.def export("id4_smoke_configured") @id4_configured_smoke() {
  %unit = index.constant 1 : index
  %workgroups_x = config.get @id4.smoke.workgroups_x : index
  %workgroup_size_x = config.get @id4.smoke.workgroup_size_x : index
  kernel.launch.config workgroups(%workgroups_x, %unit, %unit) workgroup_size(%workgroup_size_x, %unit, %unit) : index
} launch(%output: buffer) {
  %zero_offset = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %value = scalar.constant 7 : i32
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32, #dense>
  view.store %value, %view[%zero_index] : i32, view<1xi32, #dense>
  kernel.return
}
)";

typedef struct SmokeExecutable {
  // Local HAL executable base.
  iree_hal_local_executable_t base;
} SmokeExecutable;

static iree_status_t SmokeExecutableIssueCall(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  (void)executable;
  (void)workgroup_state;
  (void)worker_id;
  if (ordinal != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "smoke executable ordinal is out of range");
  }
  if (dispatch_state->binding_count != 1 ||
      dispatch_state->binding_lengths[0] < sizeof(uint32_t) ||
      !dispatch_state->binding_ptrs[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke executable requires one output binding");
  }
  uint32_t* output = static_cast<uint32_t*>(dispatch_state->binding_ptrs[0]);
  output[0] = 7;
  return iree_ok_status();
}

static const iree_hal_executable_dispatch_attrs_v0_t kSmokeDispatchAttrs[] = {{
    /*.flags=*/IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
    /*.local_memory_pages=*/0,
    /*.binding_count=*/1,
    /*.reserved_0=*/0,
    /*.workgroup_size_x=*/64,
    /*.workgroup_size_y=*/1,
    /*.workgroup_size_z=*/1,
    /*.parameter_count=*/1,
    /*.constant_byte_length=*/0,
}};

static const iree_hal_executable_dispatch_v0_t kSmokeDispatchPtrs[] = {
    nullptr,
};

static const char* const kSmokeExportNames[] = {
    "id4_smoke_configured",
};

static SmokeExecutable* SmokeExecutableCast(
    iree_hal_executable_t* base_executable) {
  return reinterpret_cast<SmokeExecutable*>(base_executable);
}

static void SmokeExecutableDestroy(iree_hal_executable_t* base_executable) {
  SmokeExecutable* executable = SmokeExecutableCast(base_executable);
  iree_allocator_t host_allocator = executable->base.host_allocator;
  iree_hal_local_executable_deinitialize(&executable->base);
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t SmokeExecutableFunctionCount(
    iree_hal_executable_t* base_executable) {
  return 1;
}

static iree_status_t SmokeExecutableFunctionInfo(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  (void)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(function, 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "smoke executable function is out of range");
  }
  std::memset(out_info, 0, sizeof(*out_info));
  out_info->name = IREE_SV("id4_smoke_configured");
  out_info->binding_count = 1;
  out_info->parameter_count = 1;
  out_info->workgroup_size[0] = 64;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t SmokeExecutableFunctionParameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(function, 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "smoke executable function is out of range");
  }
  if (capacity < 1 || !out_parameters) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "smoke executable parameter storage is required");
  }
  out_parameters[0] = (iree_hal_executable_function_parameter_t){
      // Output is passed as binding ordinal zero.
      /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
      // No parameter flags are required.
      /*.flags=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NONE,
      // Binding parameters have no scalar byte size.
      /*.size=*/0,
      // Binding ordinal for the output buffer.
      /*.offset=*/0,
      // Human-readable parameter name.
      /*.name=*/IREE_SV("output"),
  };
  return iree_ok_status();
}

static iree_status_t SmokeExecutableLookupFunctionByName(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  (void)base_executable;
  if (iree_string_view_equal(name, IREE_SV("id4_smoke_configured"))) {
    *out_function = iree_hal_executable_function_from_index(0);
    return iree_ok_status();
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "smoke executable function was not found");
}

static iree_status_t SmokeExecutableTryLookupGlobalByName(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t SmokeExecutableGlobalInfo(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static iree_status_t SmokeExecutableGlobalBuffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  *out_buffer = nullptr;
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static const iree_hal_local_executable_vtable_t kSmokeExecutableVTable = {
    // Base HAL executable vtable.
    /*.base=*/
    {
        // Releases executable storage.
        /*.destroy=*/SmokeExecutableDestroy,
        // Returns the single smoke function.
        /*.function_count=*/SmokeExecutableFunctionCount,
        // Returns smoke function metadata.
        /*.function_info=*/SmokeExecutableFunctionInfo,
        // Returns smoke function parameters.
        /*.function_parameters=*/SmokeExecutableFunctionParameters,
        // Resolves the smoke function by name.
        /*.lookup_function_by_name=*/SmokeExecutableLookupFunctionByName,
        // Reports no globals.
        /*.try_lookup_global_by_name=*/
        SmokeExecutableTryLookupGlobalByName,
        // Reports no global metadata.
        /*.global_info=*/SmokeExecutableGlobalInfo,
        // Reports no global buffers.
        /*.global_buffer=*/SmokeExecutableGlobalBuffer,
    },
    // Issues the local smoke dispatch.
    /*.issue_call=*/SmokeExecutableIssueCall,
};

static iree_status_t SmokeExecutableCreate(
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = nullptr;
  SmokeExecutable* executable = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*executable),
                            reinterpret_cast<void**>(&executable)));
  std::memset(executable, 0, sizeof(*executable));
  iree_hal_local_executable_initialize(&kSmokeExecutableVTable, host_allocator,
                                       &executable->base);
  executable->base.dispatch_attrs = kSmokeDispatchAttrs;
  executable->base.dispatch_ptrs = kSmokeDispatchPtrs;
  executable->base.export_count = 1;
  executable->base.export_names = kSmokeExportNames;
  *out_executable = reinterpret_cast<iree_hal_executable_t*>(executable);
  return iree_ok_status();
}

static SmokeExecutableCache* SmokeExecutableCacheCast(
    iree_hal_executable_cache_t* base_executable_cache) {
  return reinterpret_cast<SmokeExecutableCache*>(base_executable_cache);
}

static void SmokeExecutableCacheDestroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  SmokeExecutableCache* cache = SmokeExecutableCacheCast(base_executable_cache);
  iree_allocator_free(cache->host_allocator, cache);
}

static iree_status_t SmokeExecutableCacheInferFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  SmokeExecutableCache* cache = SmokeExecutableCacheCast(base_executable_cache);
  ++cache->infer_count;
  if (executable_data.data_length < 4 || executable_data.data[0] != 0x7f ||
      executable_data.data[1] != 'E' || executable_data.data[2] != 'L' ||
      executable_data.data[3] != 'F') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected an ELF executable");
  }
  const iree_string_view_t format = IREE_SV("gfx1100");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  std::memcpy(executable_format, format.data, format.size);
  executable_format[format.size] = 0;
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static bool SmokeExecutableCacheCanPrepareFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  SmokeExecutableCache* cache = SmokeExecutableCacheCast(base_executable_cache);
  ++cache->can_prepare_count;
  return iree_string_view_equal(executable_format, IREE_SV("gfx1100"));
}

static iree_status_t SmokeExecutableCachePrepareExecutable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  SmokeExecutableCache* cache = SmokeExecutableCacheCast(base_executable_cache);
  ++cache->prepare_count;
  cache->last_caching_mode = executable_params->caching_mode;
  if (executable_params->executable_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data is empty");
  }
  return SmokeExecutableCreate(cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t kSmokeExecutableCacheVTable = {
    // Releases the fake executable cache.
    /*.destroy=*/SmokeExecutableCacheDestroy,
    // Infers the fake gfx1100 executable format from HSACO bytes.
    /*.infer_format=*/SmokeExecutableCacheInferFormat,
    // Accepts the fake gfx1100 executable format.
    /*.can_prepare_format=*/SmokeExecutableCacheCanPrepareFormat,
    // Prepares a local executable for the smoke dispatch.
    /*.prepare_executable=*/SmokeExecutableCachePrepareExecutable,
};

iree_status_t CreateExecutableCache(iree_allocator_t host_allocator,
                                    SmokeExecutableCache** out_cache) {
  *out_cache = nullptr;
  SmokeExecutableCache* cache = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*cache),
                                             reinterpret_cast<void**>(&cache)));
  std::memset(cache, 0, sizeof(*cache));
  iree_hal_resource_initialize(&kSmokeExecutableCacheVTable, &cache->resource);
  cache->host_allocator = host_allocator;
  *out_cache = cache;
  return iree_ok_status();
}

iree_status_t CreateSmokeStage(iree_hal_device_group_t* device_group,
                               SmokeExecutableCache* executable_cache,
                               id4_pipeline_kernel_cache_t* kernel_cache,
                               iree_allocator_t host_allocator,
                               id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache);
  services.host_allocator = host_allocator;

  id4_smoke_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = kernel_cache;
  options.source_identifier = IREE_SV("smoke_configured.loom");
  options.source_contents = iree_make_const_byte_span(
      kSmokeKernelSource, std::strlen(kSmokeKernelSource));
  options.module_name = IREE_SV("id4_smoke_configured");
  options.executable_identifier = IREE_SV("id4_smoke_configured.hsaco");
  options.function_name = IREE_SV("id4_smoke_configured");
  options.workgroups_x = 1;
  options.workgroup_size_x = 64;
  return id4_smoke_stage_create(&options, host_allocator, out_stage);
}

iree_io_parameter_provider_t* CreateSmokeParameterProvider() {
  iree_io_parameter_index_t* index = nullptr;
  IREE_CHECK_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = IREE_SV("smoke.weight");
  entry.length = 16;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length = 4;
  entry.storage.splat.pattern[0] = 0x11;
  entry.storage.splat.pattern[1] = 0x22;
  entry.storage.splat.pattern[2] = 0x33;
  entry.storage.splat.pattern[3] = 0x44;
  IREE_CHECK_OK(iree_io_parameter_index_add(index, &entry));

  iree_io_parameter_provider_t* provider = nullptr;
  IREE_CHECK_OK(iree_io_parameter_index_provider_create(
      IREE_SV("smoke"), index,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), &provider));
  iree_io_parameter_index_release(index);
  return provider;
}

}  // namespace id4::test
